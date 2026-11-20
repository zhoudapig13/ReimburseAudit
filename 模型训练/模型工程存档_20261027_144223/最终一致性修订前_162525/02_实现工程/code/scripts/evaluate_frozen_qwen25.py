"""Run an untouched Qwen2.5-VL checkpoint on the fixed validation workflows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluate_official_adapter import score

from audit_engine.models.chunked_mlp import (
    enable_decoder_mlp_chunks,
    enable_final_norm_chunks,
)
from audit_engine.training.official_sft import workflow_messages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-path", type=Path, default=Path("models/Qwen2.5-VL-7B-Instruct")
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    folder = args.output_dir
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "predictions.jsonl").exists():
        raise ValueError("predictions already exist; choose a new output directory")
    started = time.time()
    torch.set_num_threads(8)
    config = yaml.safe_load(Path("configs/official_sft5090_corrected.yaml").read_text())
    pinned = yaml.safe_load(Path("configs/qwen25_comparison.yaml").read_text())["model"]
    manifest = json.loads((args.model_path / ".range_parts/manifest.json").read_text())
    if manifest["sha"] != pinned["revision"]:
        raise ValueError("checkpoint revision mismatch")
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        local_files_only=True,
        max_pixels=config["reader"]["image_max_pixels"],
    )
    processor.image_processor.size["longest_edge"] = config["reader"][
        "image_max_pixels"
    ]
    native = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    limit = min(131072, native.text_config.max_position_embeddings)
    inputs_path = Path("data/normalized_full/validation_inputs.jsonl")
    bundles = [json.loads(line) for line in inputs_path.open()]
    prepared = []
    for bundle in bundles:
        messages = workflow_messages(
            bundle, Path(config["project"]["data_root"]).resolve(), True, True
        )
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        count = inputs.input_ids.shape[1]
        if count + 1024 > limit:
            raise ValueError(
                f"complete input exceeds native context: {bundle['workflow_id']}"
            )
        if "image_grid_thw" in inputs:
            pixels = (
                inputs.image_grid_thw[:, 1:].prod(-1)
                * processor.image_processor.patch_size**2
            )
            if bool((pixels > config["reader"]["image_max_pixels"]).any()):
                raise ValueError("image pixel budget exceeded")
        record = {
            "workflow_id": bundle["workflow_id"],
            "source_family_id": bundle["source_family_id"],
            "input_tokens": count,
            "source_truncated": False,
            "variant": "new_model",
        }
        prepared.append((record, dict(inputs)))
    (folder / "input_manifest.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record, _ in prepared)
    )
    print(
        json.dumps(
            {
                "prepared": len(prepared),
                "max_input_tokens": max(r["input_tokens"] for r, _ in prepared),
            }
        ),
        flush=True,
    )
    if args.prepare_only:
        return
    download = json.loads(
        Path("runs/three_model_comparison_20261020/new_model_download.json").read_text()
    )
    if (
        download["revision"] != pinned["revision"]
        or not download["all_shard_sha256_verified"]
    ):
        raise ValueError("verified checkpoint weights are required")
    device_map = {
        "model.visual": "cuda:4",
        "model.language_model.embed_tokens": "cuda:4",
        "model.language_model.rotary_emb": "cuda:4",
        "model.language_model.norm": "cuda:6",
        "lm_head": "cuda:6",
    }
    group = math.ceil(native.text_config.num_hidden_layers / 3)
    for i in range(native.text_config.num_hidden_layers):
        device_map[f"model.language_model.layers.{i}"] = f"cuda:{4 + i // group}"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="sdpa",
    ).eval()
    model.requires_grad_(False)
    if enable_decoder_mlp_chunks(model, 8192) != native.text_config.num_hidden_layers:
        raise ValueError("incomplete pointwise MLP chunks")
    if enable_final_norm_chunks(model, 8192) != 1:
        raise ValueError("incomplete pointwise norm chunks")
    outputs = []
    for record, inputs in prepared:
        inputs = {key: value.to("cuda:4") for key, value in inputs.items()}
        with torch.inference_mode():
            result = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        output = {
            **record,
            "raw_output": processor.batch_decode(
                result[:, record["input_tokens"] :],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0],
        }
        outputs.append(output)
        with (folder / "predictions.jsonl").open("a") as stream:
            stream.write(json.dumps(output, ensure_ascii=False) + "\n")
        print(
            json.dumps(
                {
                    "workflow_id": record["workflow_id"],
                    "generated_tokens": result.shape[1] - record["input_tokens"],
                }
            ),
            flush=True,
        )
        del inputs, result
    targets = {
        r["workflow_id"]: r
        for r in map(
            json.loads,
            Path("data/labels_private/validation_reference_targets.jsonl")
            .read_text()
            .splitlines(),
        )
    }
    for output in outputs:
        target = targets[output["workflow_id"]]
        if (
            target["split"] != "validation"
            or target["source_family_id"] != output["source_family_id"]
        ):
            raise ValueError("validation target mismatch")
        output["score"] = score(output["raw_output"], target["target"])
    metrics = {
        key: sum(r["score"][key] for r in outputs)
        for key in [
            "schema_valid",
            "flag_correct",
            "amount_correct",
            "explanation_exact_match",
        ]
    }
    metrics["joint_flag_amount_correct"] = sum(
        r["score"]["schema_valid"]
        and r["score"]["flag_correct"]
        and r["score"]["amount_correct"]
        for r in outputs
    )
    report = {
        "model": pinned,
        "finetuned_on_project": False,
        "validation_available": len(bundles),
        "validation_evaluated": len(outputs),
        "deferred_workflow_ids": [],
        "source_truncated": False,
        "test_targets_loaded": False,
        "source_sha256": hashlib.sha256(inputs_path.read_bytes()).hexdigest(),
        "native_context_limit": limit,
        "outputs": outputs,
        "new_model": metrics,
        "generation": {"do_sample": False, "max_new_tokens": 1024},
        "elapsed_seconds": time.time() - started,
        "image_max_pixels": config["reader"]["image_max_pixels"],
        "scoring": "unchanged strict JSON/FLAG/numeric amount scoring shared with the original and finetuned Qwen3-VL",
    }
    (folder / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
