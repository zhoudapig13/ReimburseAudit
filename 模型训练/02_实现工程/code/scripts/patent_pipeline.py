"""Unified input validation, real-backbone encoding, staged training and inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import yaml

from audit_engine.contracts import AuditCase, ConditionAnnotation
from audit_engine.inference.runner import AuditRunner, CaseFeatures
from audit_engine.models.heads import AuditHeads
from audit_engine.training.patent import (
    feature_path,
    load_dataset,
    load_features,
    read_jsonl,
    reader_adapter_transfer_allowed,
    train,
    validate_annotations,
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("source path is missing or outside the dataset root")
    return path


def encode_cases(args, config):
    from audit_engine.readers.material_views import image_views
    from audit_engine.readers.qwen_encoder import FrozenQwenEncoder

    encoder = FrozenQwenEncoder(config, args.model_path, args.device)
    args.features.mkdir(parents=True, exist_ok=True)
    view_cache = args.features / "material_cache"
    view_cache.mkdir(exist_ok=True)
    tokenizer = encoder.processor.tokenizer
    chunk_size = config["reader"]["text_chunk_tokens"]
    overlap = config["reader"]["chunk_overlap_tokens"]
    if not 0 <= overlap < chunk_size:
        raise ValueError("invalid chunk overlap")
    shard_index, shard_count = map(int, args.shard.split("/"))
    if not 0 <= shard_index < shard_count:
        raise ValueError("invalid encoding shard")
    signature = {
        **encoder.signature,
        "kind": "frozen_backbone",
        "chunk_tokens": chunk_size,
        "overlap_tokens": overlap,
        "chunk_reduction": "mean_independent_views",
        "pdf_dpi": config["reader"].get("pdf_dpi", 120),
        "material_reduction": "mean_text_aggregate_and_whole_pages",
        "long_image_views": "automatic_long_image_tile_v1",
        "runtime_feature_contract": "currency_bound_v2",
        "policy_encoding": "normative_content_with_role_definitions_without_identifiers_or_review_metadata_v2",
    }

    def encode_text(text, image=None):
        key = hashlib.sha256(
            (
                encoder.cache_key(text, image) + json.dumps(signature, sort_keys=True)
            ).encode()
        ).hexdigest()
        cached = view_cache / (key + ".pt")
        if cached.is_file():
            return torch.load(cached, map_location="cpu", weights_only=True)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        chunks = [
            tokens[i : i + chunk_size]
            for i in range(0, len(tokens), chunk_size - overlap)
        ] or [[]]
        vectors = [encoder.encode(tokenizer.decode(chunk), image) for chunk in chunks]
        # Only views of this material enter its vector, never other attachments.
        vector = torch.stack(vectors).mean(0)
        temporary = cached.with_suffix("." + uuid.uuid4().hex + ".tmp")
        torch.save(vector, temporary)
        temporary.replace(cached)
        return vector

    for index, row in enumerate(read_jsonl(args.cases)):
        if index % shard_count != shard_index:
            continue
        case = AuditCase.model_validate(row)
        target = feature_path(args.features, case.workflow_id)
        if target.is_file():
            existing = load_features(target)
            existing.validate(case, config["model"]["hidden_size"])
            if existing.signature != signature:
                raise ValueError(
                    "feature cache encoding configuration changed; choose a new cache directory"
                )
            print(
                json.dumps({"workflow_id": case.workflow_id, "cache": "hit"}),
                flush=True,
            )
            continue
        vectors = []
        for material in case.evidence:
            raw = source_path(args.source_root, material.raw_ref)
            if sha256_file(raw) != material.source_sha256:
                raise ValueError("raw material changed; reparse and recompile the case")
            text = (
                "\n".join(f"[{s.segment_id}] {s.text}" for s in material.segments)
                or material.text
                or ""
            )
            material_vectors = [encode_text(text)] if text else []
            for view, image in image_views(
                material, raw, view_cache / "page_views", pdf_dpi=signature["pdf_dpi"]
            ):
                description = {
                    k: v
                    for k, v in view.items()
                    if k not in {"source_sha256", "source_rows"}
                }
                material_vectors.append(
                    encode_text(json.dumps(description, ensure_ascii=False), image)
                )
            if not material_vectors:
                raise ValueError(
                    "material has no readable views; populate reader output first"
                )
            vectors.append(torch.stack(material_vectors).mean(0))
        app = json.dumps(case.application.model_dump(), ensure_ascii=False)
        policies = [
            encode_text(
                json.dumps(
                    {
                        "question": c.question,
                        "kind": c.kind,
                        "roles": c.roles,
                        "role_origins": c.role_origins,
                        "role_definitions": c.role_definitions,
                        "operation": c.operation.model_dump() if c.operation else None,
                        "scope": c.scope,
                        "consequence": c.consequence.model_dump(),
                        "source_quotes": [
                            {"quote": r.quote, "kind": r.kind, "note": r.note}
                            for r in c.policy_refs
                        ],
                    },
                    ensure_ascii=False,
                )
            )
            for c in case.conditions
        ]
        hidden = config["model"]["hidden_size"]
        features = CaseFeatures(
            case.digest(),
            [e.evidence_id for e in case.evidence],
            [c.condition_id for c in case.conditions],
            torch.stack(vectors) if vectors else torch.empty(0, hidden),
            encode_text(app),
            torch.stack(policies) if policies else torch.empty(0, hidden),
            signature,
        )
        features.validate(case, hidden)
        temporary = target.with_suffix(".tmp")
        torch.save(asdict(features), temporary)
        temporary.replace(target)
        print(
            json.dumps({"workflow_id": case.workflow_id, "cache": "written"}),
            flush=True,
        )


def make_model(config, device):
    c = config["model"]
    model = AuditHeads(c["hidden_size"], c["dimension"], c["width"], c["role_slots"]).to(device)
    gain = config.get("heads", {}).get("initial_projection_gain", 1.0)
    if not 0 < gain <= 1:
        raise ValueError("initial projection gain must be in (0, 1]")
    with torch.no_grad():
        for parameter in model.shared.parameters():
            parameter.mul_(gain)
    return model


def predict(args):
    bundle = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    metadata = bundle["metadata"]
    config = metadata["config"]
    if not metadata["three_components_optimized"]:
        raise ValueError("checkpoint has not optimized all three components")
    model = make_model(config, args.device)
    model.load_state_dict(bundle["model"], strict=True)
    runner = AuditRunner(
        model,
        thresholds=config["thresholds"],
        max_rounds=0 if args.no_correction else config["correction"]["max_rounds"],
        qualification=metadata["qualification"],
        ablation=args.ablation,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects completed prediction sets from accidental replacement.
    with args.output.open("x", encoding="utf-8") as stream:
        for row in read_jsonl(args.cases):
            case = AuditCase.model_validate(row)
            try:
                features = load_features(feature_path(args.features, case.workflow_id))
                if (
                    metadata["qualification"] == "formal"
                    and features.signature.get("kind") != "frozen_backbone"
                ):
                    raise ValueError(
                        "formal prediction requires actual backbone features"
                    )
                if metadata[
                    "qualification"
                ] == "formal" and features.signature not in metadata.get(
                    "feature_signatures", []
                ):
                    raise ValueError(
                        "prediction feature encoder differs from the checkpoint training version"
                    )
                result = runner.run(case, features)
            except (ValueError, FileNotFoundError, RuntimeError) as error:
                result = {
                    "workflow_id": case.workflow_id,
                    "status": "failed",
                    "error": str(error),
                }
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--cases", type=Path, required=True)
    validate.add_argument("--annotations", type=Path)
    validate.add_argument("--partitions", type=Path)
    split = commands.add_parser("split")
    split.add_argument("--cases", type=Path, required=True)
    split.add_argument("--partitions", type=Path, required=True)
    split.add_argument("--output", type=Path, required=True)
    schemas = commands.add_parser("schemas")
    schemas.add_argument("--output", type=Path, required=True)
    encode = commands.add_parser("encode")
    training = commands.add_parser("train")
    inference = commands.add_parser("predict")
    for cmd in (encode, training, inference):
        cmd.add_argument("--cases", type=Path, required=True)
        cmd.add_argument("--features", type=Path, required=True)
        cmd.add_argument("--device", default="cuda:0")
    for cmd in (encode, training):
        cmd.add_argument(
            "--config", type=Path, default=Path("configs/patent_delivery.yaml")
        )
    encode.add_argument("--model-path", type=Path, required=True)
    encode.add_argument("--shard", default="0/1")
    encode.add_argument(
        "--source-root", type=Path, default=Path("data_raw/ReimburseBench")
    )
    training.add_argument("--annotations", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--initial-checkpoint", type=Path)
    training.add_argument("--adapt-reader-encoder", action="store_true",
                          help="Retrain all heads after a same-backbone reader LoRA change")
    inference.add_argument("--checkpoint", type=Path, required=True)
    inference.add_argument("--output", type=Path, required=True)
    inference.add_argument("--no-correction", action="store_true")
    inference.add_argument(
        "--ablation",
        choices=["full", "no_policy_comparison", "no_contribution"],
        default="full",
    )
    args = parser.parse_args()
    if args.command == "validate":
        cases = [AuditCase.model_validate(v) for v in read_jsonl(args.cases)]
        annotations = (
            [
                ConditionAnnotation.model_validate(v)
                for v in read_jsonl(args.annotations)
            ]
            if args.annotations
            else []
        )
        partitions = (
            {v["workflow_id"]: v for v in read_jsonl(args.partitions)}
            if args.partitions
            else None
        )
        validate_annotations(cases, annotations, partitions)
        print(
            json.dumps(
                {
                    "cases": len(cases),
                    "conditions": sum(len(c.conditions) for c in cases),
                    "annotations": len(annotations),
                    "approved": sum(
                        a.status in {"approved", "corrected"} for a in annotations
                    ),
                }
            )
        )
    elif args.command == "split":
        partitions = {v["workflow_id"]: v for v in read_jsonl(args.partitions)}
        cases = [AuditCase.model_validate(row) for row in read_jsonl(args.cases)]
        if {c.workflow_id for c in cases} != set(partitions) or len(cases) != len(
            partitions
        ):
            raise ValueError("runtime case set differs from fixed partitions")
        args.output.mkdir(parents=True, exist_ok=True)
        for name in ("train", "validation", "test"):
            target = args.output / (name + "_cases.jsonl")
            temporary = target.with_suffix(".tmp")
            temporary.write_text(
                "".join(
                    c.model_dump_json() + "\n"
                    for c in cases
                    if partitions[c.workflow_id]["split"] == name
                ),
                encoding="utf-8",
            )
            temporary.replace(target)
    elif args.command == "schemas":
        args.output.mkdir(parents=True, exist_ok=True)
        for name, cls in (
            ("runtime_case", AuditCase),
            ("condition_annotation", ConditionAnnotation),
        ):
            (args.output / f"{name}.schema.json").write_text(
                json.dumps(cls.model_json_schema(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    elif args.command == "encode":
        encode_cases(args, yaml.safe_load(args.config.read_text()))
    elif args.command == "train":
        config = yaml.safe_load(args.config.read_text())
        torch.manual_seed(config["project"]["seed"])
        partitions = Path(config["data"]["partitions"])
        dataset = load_dataset(args.cases, args.annotations, args.features, partitions)
        model = make_model(config, args.device)
        if args.initial_checkpoint:
            initial = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=True)
            signatures = initial["metadata"].get("feature_signatures", [])
            current_signatures = [item.features.signature for rows in dataset.values() for item in rows]
            mismatch = any(signature not in signatures for signature in current_signatures)
            if mismatch:
                if not args.adapt_reader_encoder or not reader_adapter_transfer_allowed(signatures, current_signatures):
                    raise ValueError("warm-start checkpoint uses a different backbone feature encoder")
                config["reader_encoder_transfer"] = {
                    "mode": "same_backbone_reader_adapter_retraining",
                    "previous_feature_signatures": signatures,
                    "initial_checkpoint": str(args.initial_checkpoint.resolve()),
                    "initial_checkpoint_sha256": sha256_file(args.initial_checkpoint),
                    "prediction_signature_checks_preserved": True,
                }
            model.load_state_dict(initial["model"], strict=True)
            gain = config.get("heads", {}).get("warm_start_projection_gain", 1.0)
            if not 0 < gain <= 1:
                raise ValueError("warm-start projection gain must be in (0, 1]")
            if gain != 1:
                with torch.no_grad():
                    for parameter in model.shared.parameters():
                        parameter.mul_(gain)
        train(model, dataset, config, args.output)
    else:
        predict(args)


if __name__ == "__main__":
    main()
