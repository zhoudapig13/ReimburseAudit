"""Evaluate a frozen InternVL checkpoint using the unchanged Qwen audit task."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

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
        "--model-path", type=Path, default=Path("models/InternVL3_5-8B-HF")
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    folder = args.output_dir
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "predictions.jsonl").exists():
        raise ValueError("predictions already exist; use a new output directory")
    started = time.time()
    torch.set_num_threads(8)
    torch.manual_seed(3407)
    config = yaml.safe_load(Path("configs/official_sft5090_corrected.yaml").read_text())
    pinned = yaml.safe_load(Path("configs/internvl_comparison.yaml").read_text())[
        "model"
    ]
    manifest_path = args.model_path / ".range_parts/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["sha"] != pinned["revision"]:
        raise ValueError("checkpoint revision mismatch")
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    native = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    limit = native.text_config.max_position_embeddings
    # Four native 448px crops plus a thumbnail stay below the Qwen 1MP budget.
    processor.image_processor.max_patches = 4
    inputs_path = Path("data/normalized_full/validation_inputs.jsonl")
    bundles = [json.loads(line) for line in inputs_path.open()]
    prepared, input_records = [], []
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
            max_patches=4,
            crop_to_patches=True,
        )
        count = inputs.input_ids.shape[1]
        record = {
            "workflow_id": bundle["workflow_id"],
            "source_family_id": bundle["source_family_id"],
            "input_tokens": count,
            "source_truncated": False,
            "context_limit": limit,
            "context_fits": count + 1024 <= limit,
        }
        input_records.append(record)
        prepared.append((record, dict(inputs)))
    (folder / "input_manifest.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in input_records)
    )
    print(
        json.dumps(
            {
                "prepared": len(prepared),
                "context_failures": sum(not r["context_fits"] for r in input_records),
            }
        ),
        flush=True,
    )
    if args.prepare_only:
        return
    download = json.loads(
        Path("runs/three_model_comparison_20261020/internvl_download.json").read_text()
    )
    if (
        download["revision"] != pinned["revision"]
        or not download["all_shard_sha256_verified"]
    ):
        raise ValueError("verified weights are required")
    device_map = {
        "model.vision_tower": "cuda:4",
        "model.multi_modal_projector": "cuda:4",
        "model.language_model.embed_tokens": "cuda:4",
        "model.language_model.rotary_emb": "cuda:4",
        "model.language_model.norm": "cuda:6",
        "lm_head": "cuda:6",
    }
    for i in range(native.text_config.num_hidden_layers):
        device_map[f"model.language_model.layers.{i}"] = f"cuda:{4 + i // 12}"
    model = AutoModelForImageTextToText.from_pretrained(
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
        output = {**record, "variant": "internvl", "raw_output": ""}
        if not record["context_fits"]:
            output["failure"] = "full_input_exceeds_native_context_limit"
        else:
            inputs = {key: value.to("cuda:4") for key, value in inputs.items()}
            try:
                with torch.inference_mode():
                    result = model.generate(
                        **inputs, max_new_tokens=1024, do_sample=False
                    )
                output["raw_output"] = processor.batch_decode(
                    result[:, record["input_tokens"] :],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                output["generated_tokens"] = result.shape[1] - record["input_tokens"]
                del result
            except torch.OutOfMemoryError:
                output["failure"] = "gpu_out_of_memory"
                torch.cuda.empty_cache()
            del inputs
        outputs.append(output)
        with (folder / "predictions.jsonl").open("a") as stream:
            stream.write(json.dumps(output, ensure_ascii=False) + "\n")
        print(
            json.dumps(
                {
                    key: output[key]
                    for key in [
                        "workflow_id",
                        "input_tokens",
                        "failure",
                        "generated_tokens",
                    ]
                    if key in output
                }
            ),
            flush=True,
        )
    # Read labels only after all model predictions are fixed.
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
            raise ValueError("validation labels do not match")
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
        "validation_attempted": len(outputs),
        "generation_failures": [r["workflow_id"] for r in outputs if r.get("failure")],
        "source_sha256": hashlib.sha256(inputs_path.read_bytes()).hexdigest(),
        "source_truncated": False,
        "test_targets_loaded": False,
        "outputs": outputs,
        "internvl": metrics,
        "generation": {
            "do_sample": False,
            "max_new_tokens": 1024,
            "thinking_system_prompt": False,
        },
        "native_context_limit": limit,
        "image_max_patches": 4,
        "elapsed_seconds": time.time() - started,
        "scoring": "same strict JSON/FLAG/numeric amount scoring as Qwen; failures count as incorrect among all 23 cases",
    }
    (folder / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
