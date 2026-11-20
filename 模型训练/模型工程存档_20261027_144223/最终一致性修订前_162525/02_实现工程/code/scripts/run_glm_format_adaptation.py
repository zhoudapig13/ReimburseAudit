"""Check GLM inference adaptation on train examples, then evaluate validation once."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import yaml
from lmformatenforcer import CharacterLevelParserConfig, JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_token_enforcer_tokenizer_data,
    build_transformers_prefix_allowed_tokens_fn,
)
from transformers import (
    AutoConfig,
    AutoProcessor,
    Glm4vForConditionalGeneration,
    StoppingCriteria,
    StoppingCriteriaList,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluate_official_adapter import score  # noqa: E402

from audit_engine.models.chunked_mlp import (  # noqa: E402
    enable_decoder_mlp_chunks,
    enable_final_norm_chunks,
)
from audit_engine.training.glm_format_adapter import (  # noqa: E402
    ANSWER_SCHEMA,
    FINAL_REMINDER,
    SYSTEM_PROMPT,
    adapted_chat_template,
    adapted_messages,
)
from audit_engine.training.glm_output import ANSWER_START, THINK_END  # noqa: E402
from audit_engine.training.official_sft import workflow_messages  # noqa: E402

PILOT_IDS = ("2.3.3", "5.3.8", "3.7.1")
THINKING_BUDGET = 4096
ANSWER_BUDGET = 1024


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def metrics(outputs):
    result = {
        key: sum(row["score"][key] for row in outputs)
        for key in (
            "schema_valid",
            "flag_correct",
            "amount_correct",
            "explanation_exact_match",
        )
    }
    result["joint_flag_amount_correct"] = sum(
        row["score"]["schema_valid"]
        and row["score"]["flag_correct"]
        and row["score"]["amount_correct"]
        for row in outputs
    )
    return result


class StopAtAnswer(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        done = int(input_ids[0, -1]) in (THINK_END, ANSWER_START)
        return torch.tensor([done], dtype=torch.bool, device=input_ids.device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    folder = args.output_dir.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "state.json").exists():
        raise ValueError("adaptation run already exists")
    state = {
        "pid": os.getpid(),
        "status": "running",
        "stage": "preparing",
        "started_unix": time.time(),
        "finetuned": False,
    }

    def save(**updates):
        state.update(updates, updated_unix=time.time())
        write_json(folder / "state.json", state)

    save()
    try:
        torch.set_num_threads(8)
        config = yaml.safe_load(
            (ROOT / "configs/official_sft5090_corrected.yaml").read_text()
        )
        pinned = yaml.safe_load((ROOT / "configs/glm41v_comparison.yaml").read_text())[
            "model"
        ]
        local = ROOT / "models/GLM-4.1V-9B-Thinking"
        verified = json.loads(
            (
                ROOT / "runs/three_model_comparison_20261020/new_model_download.json"
            ).read_text()
        )
        if (
            verified["revision"] != pinned["revision"]
            or not verified["all_shard_sha256_verified"]
        ):
            raise ValueError("verified original checkpoint required")
        processor = AutoProcessor.from_pretrained(
            local, local_files_only=True, use_fast=True
        )
        processor.chat_template = adapted_chat_template(processor.chat_template)
        processor.image_processor.size["longest_edge"] = (
            2 * config["reader"]["image_max_pixels"]
        )
        native = AutoConfig.from_pretrained(local, local_files_only=True)
        tokenizer = processor.tokenizer
        if (
            tokenizer.convert_tokens_to_ids("<answer>") != ANSWER_START
            or tokenizer.convert_tokens_to_ids("</think>") != THINK_END
        ):
            raise ValueError("native channel token IDs changed")
        train_path = ROOT / "data/normalized_full/train_inputs.jsonl"
        validation_path = ROOT / "data/normalized_full/validation_inputs.jsonl"
        train = {
            r["workflow_id"]: r
            for r in map(json.loads, train_path.read_text().splitlines())
        }
        validation = list(map(json.loads, validation_path.read_text().splitlines()))
        if set(PILOT_IDS) - set(train) or set(PILOT_IDS) & {
            r["workflow_id"] for r in validation
        }:
            raise ValueError("pilot split isolation failed")
        protocol = {
            "model": pinned,
            "weights_changed": False,
            "training_performed": False,
            "pilot_workflow_ids": PILOT_IDS,
            "pilot_split": "train",
            "pilot_gate": "3/3 strict schema valid; no validation-label tuning",
            "system_prompt": SYSTEM_PROMPT,
            "final_reminder": FINAL_REMINDER,
            "answer_schema": ANSWER_SCHEMA,
            "decoding": {
                "do_sample": False,
                "thinking_budget": THINKING_BUDGET,
                "answer_budget": ANSWER_BUDGET,
                "grammar": "lm-format-enforcer",
                "grammar_version": importlib.metadata.version("lm-format-enforcer"),
                "strict_field_order": True,
                "reasoning_budget_may_close_native_channel": True,
            },
            "image_max_pixels": config["reader"]["image_max_pixels"],
            "source_truncated": False,
            "input_sha256": {
                str(path.relative_to(ROOT)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in (train_path, validation_path)
            },
            "code_sha256": {
                str(path.relative_to(ROOT)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in (
                    Path(__file__),
                    ROOT / "audit_engine/training/glm_format_adapter.py",
                )
            },
            "validation_targets_read_only_after_predictions": True,
        }
        write_json(folder / "protocol.json", protocol)

        def prepare(bundle):
            messages = adapted_messages(
                workflow_messages(
                    bundle,
                    (ROOT / config["project"]["data_root"]).resolve(),
                    True,
                    True,
                )
            )
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            if (
                inputs.input_ids.shape[1] + THINKING_BUDGET + ANSWER_BUDGET + 4
                > native.text_config.max_position_embeddings
            ):
                raise ValueError(
                    f"full workflow exceeds context: {bundle['workflow_id']}"
                )
            if "image_grid_thw" in inputs:
                pixels = (
                    inputs.image_grid_thw[:, 1:].prod(-1)
                    * processor.image_processor.patch_size**2
                )
                if bool((pixels > config["reader"]["image_max_pixels"]).any()):
                    raise ValueError("image budget exceeded")
            return (bundle, dict(inputs))

        pilot_prepared = [prepare(train[wid]) for wid in PILOT_IDS]
        validation_prepared = [prepare(bundle) for bundle in validation]
        write_json(
            folder / "input_manifest.json",
            {
                "pilot": list(PILOT_IDS),
                "validation": [
                    {
                        "workflow_id": bundle["workflow_id"],
                        "input_tokens": inputs["input_ids"].shape[1],
                        "source_truncated": False,
                    }
                    for bundle, inputs in validation_prepared
                ],
            },
        )
        save(stage="loading_frozen_model")
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
        model = Glm4vForConditionalGeneration.from_pretrained(
            local,
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map=device_map,
            attn_implementation="sdpa",
        ).eval()
        model.requires_grad_(False)
        if (
            enable_decoder_mlp_chunks(model, 8192)
            != native.text_config.num_hidden_layers
            or enable_final_norm_chunks(model, 8192) != 1
        ):
            raise ValueError("pointwise memory configuration incomplete")
        grammar_data = build_token_enforcer_tokenizer_data(tokenizer)

        def predict(bundle, cpu_inputs):
            case_started = time.time()
            inputs = {key: value.to("cuda:4") for key, value in cpu_inputs.items()}
            prompt_length = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                reasoning = model.generate(
                    **inputs,
                    max_new_tokens=THINKING_BUDGET,
                    do_sample=False,
                    stopping_criteria=StoppingCriteriaList([StopAtAnswer()]),
                )
            thought = reasoning[0, prompt_length:].tolist()
            eos_ids = model.generation_config.eos_token_id
            eos_ids = [eos_ids] if isinstance(eos_ids, int) else eos_ids
            while thought and thought[-1] in eos_ids:
                thought.pop()
            forced = not thought or thought[-1] not in (THINK_END, ANSWER_START)
            injected = ([THINK_END] if forced else []) + (
                [] if thought and thought[-1] == ANSWER_START else [ANSWER_START]
            )
            prefix = torch.cat(
                [
                    inputs["input_ids"],
                    torch.tensor(
                        [thought + injected], device="cuda:4", dtype=torch.long
                    ),
                ],
                dim=1,
            )
            continuation = {
                key: value
                for key, value in inputs.items()
                if key not in ("input_ids", "attention_mask")
            }
            continuation.update(
                input_ids=prefix, attention_mask=torch.ones_like(prefix)
            )
            grammar = build_transformers_prefix_allowed_tokens_fn(
                grammar_data,
                JsonSchemaParser(
                    ANSWER_SCHEMA,
                    CharacterLevelParserConfig(
                        force_json_field_order=True, max_consecutive_whitespaces=4
                    ),
                ),
            )
            del reasoning
            with torch.inference_mode():
                answer = model.generate(
                    **continuation,
                    max_new_tokens=ANSWER_BUDGET,
                    do_sample=False,
                    prefix_allowed_tokens_fn=grammar,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                )
            final = answer[0, prefix.shape[1] :].tolist()
            record = {
                "workflow_id": bundle["workflow_id"],
                "source_family_id": bundle["source_family_id"],
                "variant": "glm_adapted",
                "input_tokens": prompt_length,
                "source_truncated": False,
                "raw_output": tokenizer.decode(
                    final, skip_special_tokens=True, clean_up_tokenization_spaces=False
                ),
                "raw_reasoning": tokenizer.decode(thought, skip_special_tokens=False),
                "injected_channel_tokens": injected,
                "raw_final_generation": tokenizer.decode(
                    final, skip_special_tokens=False
                ),
                "thinking_tokens": len(thought),
                "answer_tokens": len(final),
                "thinking_budget_forced_close": forced,
                "elapsed_seconds": time.time() - case_started,
            }
            del inputs, continuation, prefix, answer
            return record

        train_targets = {
            r["workflow_id"]: r
            for r in map(
                json.loads,
                (ROOT / "data/labels_private/train_reference_targets.jsonl")
                .read_text()
                .splitlines(),
            )
        }
        pilot_outputs = []
        save(stage="pilot", pilot_completed=0)
        for bundle, inputs in pilot_prepared:
            record = predict(bundle, inputs)
            target = train_targets[bundle["workflow_id"]]
            if target["split"] != "train":
                raise ValueError("pilot label split mismatch")
            record["score"] = score(record["raw_output"], target["target"])
            pilot_outputs.append(record)
            with (folder / "pilot_predictions.jsonl").open("a") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            save(
                pilot_completed=len(pilot_outputs),
                last_workflow_id=bundle["workflow_id"],
            )
            print(
                json.dumps(
                    {
                        "stage": "pilot",
                        "workflow_id": bundle["workflow_id"],
                        "schema_valid": record["score"]["schema_valid"],
                    }
                ),
                flush=True,
            )
        write_json(
            folder / "pilot_report.json",
            {"outputs": pilot_outputs, "metrics": metrics(pilot_outputs)},
        )
        if metrics(pilot_outputs)["schema_valid"] != len(PILOT_IDS):
            raise RuntimeError("pilot format gate failed; validation was not run")
        save(stage="validation", validation_completed=0, protocol_frozen=True)
        validation_started = time.time()
        outputs = []
        for bundle, inputs in validation_prepared:
            record = predict(bundle, inputs)
            outputs.append(record)
            with (folder / "predictions.jsonl").open("a") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            save(
                validation_completed=len(outputs),
                last_workflow_id=bundle["workflow_id"],
            )
            print(
                json.dumps(
                    {
                        "stage": "validation",
                        "workflow_id": bundle["workflow_id"],
                        "completed": len(outputs),
                    }
                ),
                flush=True,
            )
        targets = {
            r["workflow_id"]: r
            for r in map(
                json.loads,
                (ROOT / "data/labels_private/validation_reference_targets.jsonl")
                .read_text()
                .splitlines(),
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
            "model": pinned,
            "adaptation": protocol,
            "validation_available": len(validation),
            "validation_evaluated": len(outputs),
            "test_targets_loaded": False,
            "source_truncated": False,
            "outputs": outputs,
            "adapted_glm": metrics(outputs),
            "validation_elapsed_seconds": time.time() - validation_started,
            "total_elapsed_seconds": time.time() - state["started_unix"],
        }
        write_json(folder / "report.json", report)
        save(stage="exporting", metrics=report["adapted_glm"])
        from export_glm_adaptation import export

        export(folder)
        save(status="completed", stage="adaptation_complete", finished_unix=time.time())
        print(json.dumps(report["adapted_glm"]), flush=True)
    except Exception as error:
        save(status="failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
