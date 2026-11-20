"""Compare saved LoRA and base model on complete validation workflows only."""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from decimal import Decimal, InvalidOperation
from pathlib import Path

import torch
import yaml
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit_engine.models.chunked_mlp import (
    enable_decoder_mlp_chunks,
    enable_final_norm_chunks,
)
from audit_engine.readers.qwen_encoder import FrozenQwenEncoder
from audit_engine.training.official_sft import workflow_messages


def score(text: str, target: dict) -> dict:
    try:
        prediction = json.loads(text)
        if not isinstance(prediction, dict):
            raise ValueError("answer is not a JSON object")
        flag = prediction.get("FLAG")
        flag_valid = type(flag) is int and flag in (0, 1)
        amount_value = prediction.get("max_legal")
        try:
            if isinstance(amount_value, bool) or not isinstance(
                amount_value, (str, int, float)
            ):
                raise ValueError("invalid amount type")
            amount = Decimal(str(amount_value))
            amount_valid = amount.is_finite()
            amount_match = amount_valid and amount == Decimal(
                target["reference_max_legal"]
            )
        except (ValueError, InvalidOperation):
            amount_valid, amount_match = False, False
        problems = prediction.get("problem_reports")
        problems_valid = isinstance(problems, list) and all(
            isinstance(p, str) for p in problems
        )
        schema_valid = (
            set(prediction) == {"FLAG", "max_legal", "problem_reports"}
            and flag_valid
            and isinstance(amount_value, str)
            and amount_valid
            and problems_valid
        )
        return {
            "schema_valid": schema_valid,
            "flag_correct": flag_valid and flag == target["reference_flag"],
            "amount_correct": amount_match,
            "explanation_exact_match": problems_valid
            and problems == target["problem_reports"],
            "prediction": prediction if schema_valid else None,
        }
    except (ValueError, TypeError, KeyError, InvalidOperation):
        return {
            "schema_valid": False,
            "flag_correct": False,
            "amount_correct": False,
            "explanation_exact_match": False,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/official_sft5090_extended.yaml")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/official_sft_validation")
    )
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--token-budget", type=int, default=32768)
    parser.add_argument("--adapter-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    torch.set_num_threads(8)
    config = yaml.safe_load(args.config.read_text())
    # Evaluation only: allow complete longer sources; keep image resolution fixed.
    config["reader"]["total_input_tokens"] = args.token_budget
    encoder = FrozenQwenEncoder(
        config, Path("models/Qwen3-VL-8B-Instruct"), args.device
    )
    model = PeftModel.from_pretrained(encoder.model, args.adapter).eval()
    if config["model"].get("final_norm_chunk_tokens"):
        if (
            enable_final_norm_chunks(model, config["model"]["final_norm_chunk_tokens"])
            != 1
        ):
            raise ValueError("incomplete final norm chunk configuration")
    if config["model"].get("mlp_chunk_tokens"):
        count = enable_decoder_mlp_chunks(model, config["model"]["mlp_chunk_tokens"])
        if count != model.config.text_config.num_hidden_layers:
            raise ValueError("incomplete decoder MLP chunk configuration")
    bundles = [
        json.loads(line)
        for line in Path("data/normalized_full/validation_inputs.jsonl").open()
    ]
    selected, deferred = [], []
    for bundle in bundles:
        messages = workflow_messages(
            bundle,
            Path(config["project"]["data_root"]).resolve(),
            config["reader"].get("compact_tables", False),
            config["reader"].get("compact_json_records", False),
        )
        text = encoder.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        length = len(encoder.processor.tokenizer.encode(text, add_special_tokens=False))
        if (
            length + 1024 > args.token_budget
            or sum(e["type"] == "image" for e in bundle["evidence"])
            > config["reader"]["max_images_per_call"]
        ):
            deferred.append(bundle["workflow_id"])
            continue
        inputs = encoder.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        if inputs.input_ids.shape[1] + 1024 > args.token_budget:
            deferred.append(bundle["workflow_id"])
            continue
        if "image_grid_thw" in inputs:
            pixels = (
                inputs.image_grid_thw[:, 1:].prod(-1)
                * encoder.processor.image_processor.patch_size**2
            )
            if bool((pixels > config["reader"]["image_max_pixels"]).any()):
                raise ValueError("processor exceeded image pixel budget")
        selected.append((bundle, inputs))
    outputs = []
    # Official validation answers are loaded only after predictions are produced.
    for bundle, inputs in selected:
        inputs = inputs.to(args.device)
        for variant in ("adapter",) if args.adapter_only else ("base", "adapter"):
            context = model.disable_adapter() if variant == "base" else nullcontext()
            with (
                context,
                torch.inference_mode(),
                torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=config["model"].get("training_autocast_bf16", False),
                ),
            ):
                result = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
            raw = encoder.processor.batch_decode(
                result[:, inputs.input_ids.shape[1] :],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            record = {
                "workflow_id": bundle["workflow_id"],
                "source_family_id": bundle["source_family_id"],
                "variant": variant,
                "input_tokens": inputs.input_ids.shape[1],
                "source_truncated": False,
                "raw_output": raw,
            }
            outputs.append(record)
            with (args.output_dir / "predictions.jsonl").open("a") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                json.dumps(
                    {
                        "workflow_id": bundle["workflow_id"],
                        "variant": variant,
                        "generated_tokens": result.shape[1] - inputs.input_ids.shape[1],
                    }
                ),
                flush=True,
            )
    targets = {
        r["workflow_id"]: r
        for r in (
            json.loads(line)
            for line in Path(
                "data/labels_private/validation_reference_targets.jsonl"
            ).open()
        )
    }
    for record in outputs:
        target = targets[record["workflow_id"]]
        if (
            target["split"] != "validation"
            or target["source_family_id"] != record["source_family_id"]
        ):
            raise ValueError("validation target mismatch")
        record["score"] = score(record["raw_output"], target["target"])
    report = {
        "adapter_checkpoint": str(args.adapter),
        "validation_available": len(bundles),
        "validation_evaluated": len(selected),
        "deferred_workflow_ids": deferred,
        "test_targets_loaded": False,
        "source_truncated": False,
        "input_token_budget": args.token_budget,
        "compact_tables": config["reader"].get("compact_tables", False),
        "compact_json_records": config["reader"].get("compact_json_records", False),
        "outputs": outputs,
        "peak_gpu_gib": torch.cuda.max_memory_allocated(args.device) / 1024**3,
        "elapsed_seconds": time.time() - started,
        "explanation_metric": "exact match only, not semantic official rubric",
        "field_metrics": "fields scored independently of overall schema; finite numeric amounts accepted for amount matching; strict schema requires decimal strings",
        "subset_selection": "complete sources fitting budget; not representative of full benchmark",
    }
    for variant in ("base", "adapter"):
        subset = [r for r in outputs if r["variant"] == variant]
        report[variant] = {
            key: sum(r["score"][key] for r in subset)
            for key in (
                "schema_valid",
                "flag_correct",
                "amount_correct",
                "explanation_exact_match",
            )
        }
        report[variant]["joint_flag_amount_correct"] = sum(
            r["score"]["schema_valid"]
            and r["score"]["flag_correct"]
            and r["score"]["amount_correct"]
            for r in subset
        )
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "validation_evaluated": len(selected),
                "base": report["base"],
                "adapter": report["adapter"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
