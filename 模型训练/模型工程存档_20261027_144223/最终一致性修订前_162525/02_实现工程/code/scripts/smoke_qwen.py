"""Exercise real local Qwen weights, a train-only image, cache and LoRA gradients."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from importlib import metadata
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit_engine.readers.qwen_encoder import FrozenQwenEncoder
from audit_engine.training.smoke import fixture_spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/smoke5090.yaml"))
    parser.add_argument(
        "--model-path", type=Path, default=Path("models/Qwen3-VL-8B-Instruct")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/smoke5090"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(args.config.read_text())
    torch.manual_seed(config["project"]["seed"])
    torch.set_num_threads(8)
    started = time.monotonic()
    encoder = FrozenQwenEncoder(config, args.model_path, args.device)
    print("real_checkpoint_loaded", flush=True)
    spec = fixture_spec()
    (args.output_dir / "fixture_spec.json").write_text(json.dumps(spec, indent=2))
    vectors = {}

    def encode(text, image=None):
        key = encoder.cache_key(text, image)
        if key not in vectors:
            vectors[key] = encoder.encode(text, image)
        value = vectors[key]
        if value.shape != (4096,) or not torch.isfinite(value).all():
            raise RuntimeError("invalid encoded features")
        return value

    raw_e, raw_x, raw_p = [], [], []
    for case in spec["cases"]:
        raw_e.append(torch.stack([encode(text) for text in case["evidence"]]))
        raw_x.append(encode(case["application"]))
        raw_p.append(encode(case["policy"]))
    cached = {
        "raw_e": torch.stack(raw_e),
        "raw_x": torch.stack(raw_x),
        "raw_p": torch.stack(raw_p),
        "label_method": spec["label_method"],
        "signature": encoder.signature,
        "feature_cache_keys": list(vectors),
    }
    feature_path = args.output_dir / "fixture_features.pt"
    torch.save(cached, feature_path)
    loaded = torch.load(feature_path, weights_only=True)
    projection = torch.nn.Linear(4096, 512).to(args.device)
    projection(loaded["raw_e"].to(args.device)).square().mean().backward()
    feature_gradient_ok = bool(projection.weight.grad.abs().sum() > 0)
    if not feature_gradient_ok:
        raise RuntimeError("frozen feature projection received no gradient")
    print("frozen_features_and_projection_gradient_ok", feature_gradient_ok, flush=True)
    pilot_ids = {
        json.loads(line)["workflow_id"]
        for line in Path("data/pilot/pilot20_selection.jsonl").open()
    }
    bundle = next(
        json.loads(line)
        for line in Path("data/normalized_full/train_inputs.jsonl").open()
        if json.loads(line)["workflow_id"] in pilot_ids
        and json.loads(line)["workflow_id"] != "1.1.1"
    )
    image_record = next(e for e in bundle["evidence"] if e["type"] == "image")
    root = Path(config["project"]["data_root"]).resolve()
    image_path = (root / image_record["raw_ref"]).resolve()
    if not image_path.is_relative_to(root):
        raise ValueError("invalid evidence path")
    image_features = encode("Read the visible facts in this document.", image_path)
    torch.save(
        {
            "vector": image_features,
            "signature": encoder.signature,
            "workflow_id": bundle["workflow_id"],
            "evidence_id": image_record["evidence_id"],
            "cache_key": encoder.cache_key(
                "Read the visible facts in this document.", image_path
            ),
        },
        args.output_dir / "image_features.pt",
    )
    prompt = 'Read this image. Return only a JSON object with keys "document_type", "date_text", "total_amount_text", "currency_text", "source_quote", and "uncertainties". Copy visible facts exactly. Values you cannot confirm must be null; uncertainties is a list. Do not infer currency from location, and do not make a reimbursement decision.'
    reading = encoder.read(prompt, image_path, config["reader"]["max_new_tokens"])
    try:
        candidate = json.loads(reading)
        if not isinstance(candidate, dict) or set(candidate) != {
            "document_type",
            "date_text",
            "total_amount_text",
            "currency_text",
            "source_quote",
            "uncertainties",
        }:
            raise ValueError("unexpected reading schema")
        if not isinstance(candidate["uncertainties"], list) or any(
            candidate[k] is not None and not isinstance(candidate[k], str)
            for k in candidate
            if k != "uncertainties"
        ):
            raise ValueError("invalid reading value types")
        schema_ok = True
    except (ValueError, TypeError):
        candidate, schema_ok = None, False
    (args.output_dir / "reading_candidate.json").write_text(
        json.dumps(
            {
                "workflow_id": bundle["workflow_id"],
                "evidence_id": image_record["evidence_id"],
                "raw_ref": image_record["raw_ref"],
                "raw_output": reading,
                "candidate": candidate,
                "schema_ok": schema_ok,
                "annotation_status": "pending",
                "approved_training_label": False,
                "signature": encoder.signature,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("real_train_image_forward_and_generation", schema_ok, flush=True)
    # Optional adapter compatibility test; no adapter training checkpoint is used.
    from peft import LoraConfig, get_peft_model

    layer_count = encoder.model.config.text_config.num_hidden_layers
    layer_ids = "|".join(
        str(i)
        for i in range(layer_count - config["lora"]["last_text_layers"], layer_count)
    )
    targets = rf".*language_model\.layers\.({layer_ids})\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    names = [
        name for name, _ in encoder.model.named_modules() if re.fullmatch(targets, name)
    ]
    if len(names) != 4 * config["lora"]["last_text_layers"]:
        raise RuntimeError("LoRA target module mismatch")
    adapter = get_peft_model(
        encoder.model,
        LoraConfig(
            r=config["lora"]["rank"],
            lora_alpha=config["lora"]["alpha"],
            lora_dropout=0.05,
            target_modules=targets,
            task_type="CAUSAL_LM",
        ),
    )
    inputs = encoder.inputs(
        "Read this material: The approved accommodation budget is USD 100. Return the budget."
    )
    adapter.train()
    loss = adapter(**inputs, labels=inputs.input_ids.clone(), use_cache=False).loss
    loss.backward()
    gradient_modules = [
        name
        for name, p in adapter.named_parameters()
        if p.requires_grad and p.grad is not None and bool(p.grad.abs().sum() > 0)
    ]
    if not gradient_modules or not torch.isfinite(loss):
        raise RuntimeError("LoRA gradient test failed")
    if any(
        p.grad is not None
        for name, p in adapter.named_parameters()
        if "lora_" not in name
    ):
        raise RuntimeError("backbone unexpectedly received gradients")
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
    memory_ok = peak_memory <= config.get("environment", {}).get(
        "max_gpu_memory_gib", 72
    )
    report = {
        "status": "passed" if schema_ok else "reading_schema_failed",
        "formal_training_started": False,
        "within_gpu_memory_budget": memory_ok,
        "revision": config["model"]["revision"],
        "device": args.device,
        "dtype": "bfloat16",
        "hidden_size": 4096,
        "image_forward": True,
        "reading_generation": True,
        "reading_json_schema": schema_ok,
        "reading_fact_accuracy": "pending_review",
        "frozen_feature_training_gradient": feature_gradient_ok,
        "lora_gradient": True,
        "lora_target_module_count": len(names),
        "lora_nonzero_gradient_module_count": len(gradient_modules),
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "elapsed_seconds": time.monotonic() - started,
        "packages": {
            p: metadata.version(p)
            for p in [
                "torch",
                "transformers",
                "peft",
                "accelerate",
                "Pillow",
                "safetensors",
                "tokenizers",
            ]
        },
        "limits": {
            "total_input_tokens": config["reader"]["total_input_tokens"],
            "image_max_pixels": config["reader"]["image_max_pixels"],
        },
    }
    (args.output_dir / "backbone_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report), flush=True)
    if not schema_ok or not memory_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
