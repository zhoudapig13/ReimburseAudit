"""Train local Qwen LoRA on complete workflows and unchanged official answers."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit_engine.models.checkpointing import enable_device_local_decoder_checkpoints
from audit_engine.models.chunked_mlp import (
    enable_decoder_mlp_chunks,
    enable_final_norm_chunks,
)
from audit_engine.readers.qwen_encoder import FrozenQwenEncoder
from audit_engine.training.continuous import next_cursor, timed_learning_rate
from audit_engine.training.official_sft import (
    official_answer,
    response_labels,
    workflow_messages,
)
from audit_engine.training.targeted_loss import decision_amount_weights


def write_json(path: Path, value: dict):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/official_sft5090.yaml")
    )
    parser.add_argument(
        "--model-path", type=Path, default=Path("models/Qwen3-VL-8B-Instruct")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/official_sft"))
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--initial-adapter", type=Path)
    parser.add_argument("--workflow-id", action="append", default=[])
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--training-seconds", type=float)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--schedule-seconds", type=float, default=28800)
    parser.add_argument("--learning-rate-scale", type=float, default=1.0)
    parser.add_argument("--checkpoint-every-steps", type=int, default=10)
    parser.add_argument("--skip-train-loss-evaluation", action="store_true")
    parser.add_argument("--require-all-workflows", action="store_true")
    parser.add_argument(
        "--training-objective",
        choices=["standard", "decision_amount_balanced"],
        default="standard",
    )
    args = parser.parse_args()
    if args.initial_adapter and args.resume_checkpoint:
        parser.error("initial-adapter and resume-checkpoint are mutually exclusive")
    if args.training_seconds is not None and args.training_seconds <= 0:
        parser.error("training-seconds must be positive")
    if args.schedule_seconds <= 0 or args.learning_rate_scale <= 0:
        parser.error("schedule and learning-rate scale must be positive")
    if args.checkpoint_every_steps < 1:
        parser.error("checkpoint-every-steps must be positive")
    if args.resume_checkpoint:
        args.initial_adapter = args.resume_checkpoint
    config = yaml.safe_load(args.config.read_text())
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.glob("checkpoint_*")):
        raise ValueError(
            "output directory already contains checkpoints; choose a new directory"
        )
    state_path = args.output_dir / "state.json"
    started = time.time()
    state = {
        "pid": os.getpid(),
        "status": "preparing",
        "started_unix": started,
        "device": args.device,
        "supervision": "official_whole_workflow_answers",
        "component_labels_generated": False,
        "optimizer_steps": 0,
    }
    write_json(state_path, state)
    try:
        torch.set_num_threads(8)
        torch.manual_seed(config["project"]["seed"])
        random.seed(config["project"]["seed"])
        inputs_path = Path("data/normalized_full/train_inputs.jsonl")
        targets_path = Path("data/labels_private/train_reference_targets.jsonl")
        bundles = [json.loads(line) for line in inputs_path.open()]
        target_rows = [json.loads(line) for line in targets_path.open()]
        targets = {row["workflow_id"]: row for row in target_rows}
        if len(targets) != len(target_rows) or set(targets) != {
            row["workflow_id"] for row in bundles
        }:
            raise ValueError("training sources and targets do not match uniquely")
        if set(args.workflow_id) - set(targets):
            raise ValueError("requested workflow is not in the training split")
        processor = AutoProcessor.from_pretrained(
            args.model_path,
            local_files_only=True,
            max_pixels=config["reader"]["image_max_pixels"],
        )
        processor.image_processor.size["longest_edge"] = config["reader"][
            "image_max_pixels"
        ]
        root = Path(config["project"]["data_root"]).resolve()
        prepared, records = [], []
        for bundle in bundles:
            if args.workflow_id and bundle["workflow_id"] not in args.workflow_id:
                records.append(
                    {
                        "workflow_id": bundle["workflow_id"],
                        "status": "outside_requested_workflows",
                        "source_truncated": False,
                    }
                )
                continue
            target = targets[bundle["workflow_id"]]
            if target["source_family_id"] != bundle["source_family_id"]:
                raise ValueError("source family mismatch")
            answer = official_answer(target)
            images = sum(row["type"] == "image" for row in bundle["evidence"])
            messages = workflow_messages(
                bundle,
                root,
                config["reader"].get("compact_tables", False),
                config["reader"].get("compact_json_records", False),
            )
            # Cheap rejection avoids decoding images for already oversized sources.
            prompt_text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            text_length = len(
                processor.tokenizer.encode(prompt_text, add_special_tokens=False)
            )
            record = {
                "workflow_id": bundle["workflow_id"],
                "source_family_id": bundle["source_family_id"],
                "official_answer": answer,
                "official_annotation": target["annotation"],
                "source_material_count": len(bundle["evidence"]),
                "image_count": images,
                "text_prompt_tokens": text_length,
                "source_truncated": False,
                "status": "deferred_context_budget",
            }
            if images > config["reader"]["max_images_per_call"]:
                record["status"] = "deferred_image_budget"
            elif text_length < config["reader"]["total_input_tokens"]:
                prompt = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                full = processor.apply_chat_template(
                    messages
                    + [
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": answer}],
                        }
                    ],
                    tokenize=True,
                    add_generation_prompt=False,
                    return_dict=True,
                    return_tensors="pt",
                )
                labels = response_labels(prompt.input_ids, full.input_ids)
                record["full_tokens"] = full.input_ids.shape[1]
                record["response_tokens"] = int((labels != -100).sum())
                if "image_grid_thw" in full:
                    pixels = (
                        full.image_grid_thw[:, 1:].prod(-1)
                        * processor.image_processor.patch_size**2
                    )
                    if bool((pixels > config["reader"]["image_max_pixels"]).any()):
                        raise ValueError("processor exceeded image pixel budget")
                if full.input_ids.shape[1] <= config["reader"]["total_input_tokens"]:
                    record["status"] = "selected_complete_workflow"
                    weights = None
                    if args.training_objective == "decision_amount_balanced":
                        weights = decision_amount_weights(
                            processor.tokenizer, answer, labels[labels != -100]
                        )
                    prepared.append((record, dict(full), labels, weights))
            records.append(record)
        (args.output_dir / "dataset_manifest.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)
        )
        manifest = {
            "source_sha256": {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (inputs_path, targets_path)
            },
            "train_workflows_available": len(bundles),
            "selected_workflows": len(prepared),
            "requested_workflows": len(args.workflow_id)
            if args.workflow_id
            else len(bundles),
            "outside_requested_workflows": len(bundles) - len(args.workflow_id)
            if args.workflow_id
            else 0,
            "deferred_workflows": (
                len(args.workflow_id) if args.workflow_id else len(bundles)
            )
            - len(prepared),
            "official_targets_rewritten": False,
            "source_truncated": False,
            "table_serialization": "xlsx_dense_cells_v1"
            if config["reader"].get("compact_tables", False)
            else "original_parsed_text",
            "json_record_serialization": "json_records_columns_v1"
            if config["reader"].get("compact_json_records", False)
            else "original_parsed_text",
            "validation_or_test_targets_loaded": False,
            "model_revision": config["model"]["revision"],
            "config": config,
            "user_direction": "继续训练，按照官方答案来",
            "initial_adapter": str(args.initial_adapter)
            if args.initial_adapter
            else None,
            "optimizer_restart": bool(args.initial_adapter)
            and not bool(args.resume_checkpoint),
            "resume_checkpoint": str(args.resume_checkpoint)
            if args.resume_checkpoint
            else None,
            "training_seconds_requested": args.training_seconds,
            "learning_rate_scale": args.learning_rate_scale,
            "training_objective": args.training_objective,
            "loss_mass": {"FLAG": 0.25, "max_legal": 0.25, "remaining_response": 0.5}
            if args.training_objective == "decision_amount_balanced"
            else None,
            "requested_workflow_ids": args.workflow_id,
            "command": sys.argv,
            "initial_adapter_sha256": (
                hashlib.sha256(
                    (args.initial_adapter / "adapter_model.safetensors").read_bytes()
                ).hexdigest()
                if args.initial_adapter
                else None
            ),
        }
        write_json(args.output_dir / "stage_manifest.json", manifest)
        print(
            json.dumps(
                {
                    "prepared": len(prepared),
                    "deferred": manifest["deferred_workflows"],
                    "outside_requested": manifest["outside_requested_workflows"],
                }
            ),
            flush=True,
        )
        if not prepared:
            raise ValueError("no complete workflows fit the configured budget")
        if args.require_all_workflows and len(prepared) != len(bundles):
            raise ValueError(
                "continuous training requires every complete training workflow"
            )
        if args.prepare_only:
            state.update(status="prepared", selected_workflows=len(prepared))
            write_json(state_path, state)
            return
        state.update(
            status="loading_checkpoint",
            selected_workflows=len(prepared),
            epochs=args.epochs,
        )
        write_json(state_path, state)
        encoder = FrozenQwenEncoder(config, args.model_path, args.device)
        count = encoder.model.config.text_config.num_hidden_layers
        layers = "|".join(
            str(i) for i in range(count - config["lora"]["last_text_layers"], count)
        )
        pattern = rf".*language_model\.layers\.({layers})\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"
        modules = [
            name
            for name, _ in encoder.model.named_modules()
            if re.fullmatch(pattern, name)
        ]
        if len(modules) != 4 * config["lora"]["last_text_layers"]:
            raise ValueError("unexpected LoRA module count")
        if args.initial_adapter:
            saved = json.loads(
                (args.initial_adapter / "adapter_config.json").read_text()
            )
            if (
                saved["r"] != config["lora"]["rank"]
                or saved["target_modules"] != pattern
                or saved["lora_alpha"] != config["lora"]["alpha"]
            ):
                raise ValueError(
                    "initial adapter does not match configured LoRA modules"
                )
            model = PeftModel.from_pretrained(
                encoder.model, args.initial_adapter, is_trainable=True
            )
        else:
            model = get_peft_model(
                encoder.model,
                LoraConfig(
                    r=config["lora"]["rank"],
                    lora_alpha=config["lora"]["alpha"],
                    lora_dropout=config["lora"]["dropout"],
                    target_modules=pattern,
                    task_type="CAUSAL_LM",
                ),
            )
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        device_local_modules = 0
        if config["model"].get("device_local_checkpointing", False):
            device_local_modules = enable_device_local_decoder_checkpoints(model)
            if device_local_modules != count:
                raise ValueError(
                    "incomplete device-local decoder checkpoint configuration"
                )
        chunked_modules = 0
        if config["model"].get("mlp_chunk_tokens"):
            chunked_modules = enable_decoder_mlp_chunks(
                model, config["model"]["mlp_chunk_tokens"]
            )
            if chunked_modules != count:
                raise ValueError("did not chunk all text decoder MLP modules")
        model.config.use_cache = False
        final_norm_modules = 0
        if config["model"].get("final_norm_chunk_tokens"):
            final_norm_modules = enable_final_norm_chunks(
                model, config["model"]["final_norm_chunk_tokens"]
            )
            if final_norm_modules != 1:
                raise ValueError("final norm chunk configuration failed")
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            parameters, lr=config["lora"]["learning_rate"], weight_decay=0.01
        )
        initial = {
            name: p.detach().cpu().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        accum = config["lora"]["gradient_accumulation"]
        total_steps = math.ceil(len(prepared) / accum) * args.epochs
        warmup_steps = math.ceil(total_steps * config["lora"].get("warmup_ratio", 0))

        accumulated_training_seconds = 0.0
        optimization_started = None

        def training_seconds():
            return accumulated_training_seconds + (
                time.monotonic() - optimization_started
                if optimization_started is not None
                else 0.0
            )

        def lr_multiplier(step):
            if args.training_seconds is not None:
                return timed_learning_rate(
                    step,
                    training_seconds(),
                    args.schedule_seconds,
                    args.learning_rate_scale,
                )
            if warmup_steps and step < warmup_steps:
                return (step + 1) / warmup_steps
            if config["lora"].get("schedule") == "cosine":
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * min(1, max(0, progress))))
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
        resume_state = None
        if args.resume_checkpoint:
            resume_state = torch.load(
                args.resume_checkpoint / "training_state.pt",
                map_location="cpu",
                weights_only=False,
            )
            if resume_state.get("format_version") != 2:
                raise ValueError("checkpoint has no exact data cursor")
            if resume_state["source_sha256"] != manifest["source_sha256"]:
                raise ValueError("training sources changed since checkpoint")
            if resume_state["config"] != config:
                raise ValueError("configuration changed since checkpoint")
            if resume_state["schedule_seconds"] != args.schedule_seconds:
                raise ValueError("learning-rate schedule horizon changed")
            optimizer.load_state_dict(resume_state["optimizer"])
            scheduler.load_state_dict(resume_state["scheduler"])
            accumulated_training_seconds = resume_state["training_seconds"]
            state["optimizer_steps"] = resume_state["optimizer_steps"]
            torch.set_rng_state(resume_state["torch_rng"])
            random.setstate(resume_state["python_rng"])
            for device, rng in resume_state["cuda_rng_by_device"].items():
                torch.cuda.set_rng_state(rng, device)
            for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
                group["lr"] = base_lr * lr_multiplier(scheduler.last_epoch)

        def loss_for(full, labels, weights):
            inputs = {key: value.to(args.device) for key, value in full.items()}
            # Generate logits only for the answer and its preceding prompt token.
            keep = int((labels != -100).sum()) + 1
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=config["model"].get("training_autocast_bf16", False),
            ):
                outputs = model(**inputs, use_cache=False, logits_to_keep=keep)
            response = labels[:, -keep:].to(args.device)
            losses = F.cross_entropy(
                outputs.logits[:, :-1].float().reshape(-1, outputs.logits.size(-1)),
                response[:, 1:].reshape(-1),
                reduction="none",
            )
            if weights is None:
                return losses.mean()
            if losses.numel() != weights.numel():
                raise ValueError("loss weights and supervised tokens differ")
            return (losses * weights.to(losses.device)).sum()

        def evaluate(stage):
            state.update(status=stage, evaluated_workflows=0)
            write_json(state_path, state)
            model.eval()
            losses = []
            with torch.no_grad():
                for record, full, labels, weights in prepared:
                    value = float(loss_for(full, labels, weights))
                    if not math.isfinite(value):
                        raise ValueError("non-finite evaluation loss")
                    losses.append({"workflow_id": record["workflow_id"], "loss": value})
                    state.update(
                        evaluated_workflows=len(losses),
                        last_workflow_id=record["workflow_id"],
                        updated_unix=time.time(),
                    )
                    write_json(state_path, state)
            return losses

        baseline = (
            [] if args.skip_train_loss_evaluation else evaluate("baseline_evaluation")
        )
        write_json(args.output_dir / "train_loss_before.json", {"losses": baseline})
        state.update(
            status="training", selected_workflows=len(prepared), epochs=args.epochs
        )
        cursor = (
            resume_state["cursor"]
            if resume_state
            else {
                "epoch_index": 0,
                "order": None,
                "next_offset": 0,
            }
        )
        optimizer_steps_before = state["optimizer_steps"]
        optimization_started = time.monotonic()
        training_started_unix = time.time()
        state.update(
            training_started_unix=training_started_unix,
            training_seconds_requested=args.training_seconds,
            evaluation_due_unix=(training_started_unix + args.training_seconds)
            if args.training_seconds is not None
            else None,
            resumed_training_seconds=accumulated_training_seconds,
        )
        write_json(state_path, state)
        grad_verified = False
        last_saved_step = -1

        def save_checkpoint(current_cursor, epoch):
            nonlocal last_saved_step
            if last_saved_step == state["optimizer_steps"]:
                return
            name = (
                f"checkpoint_step_{state['optimizer_steps']:08d}"
                if args.training_seconds is not None
                else f"checkpoint_epoch_{epoch + 1}"
            )
            checkpoint = args.output_dir / name
            model.save_pretrained(checkpoint)
            torch.save(
                {
                    "format_version": 2,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "optimizer_steps": state["optimizer_steps"],
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(args.device),
                    "cuda_rng_by_device": {
                        device: torch.cuda.get_rng_state(device)
                        for device in sorted(
                            set(
                                config["model"]
                                .get("device_map", {"": args.device})
                                .values()
                            )
                        )
                    },
                    "python_rng": random.getstate(),
                    "cursor": current_cursor,
                    "training_seconds": training_seconds(),
                    "source_sha256": manifest["source_sha256"],
                    "config": config,
                    "schedule_seconds": args.schedule_seconds,
                    "training_objective": args.training_objective,
                },
                checkpoint / "training_state.pt",
            )
            last_saved_step = state["optimizer_steps"]
            state["checkpoint"] = str(checkpoint)
            write_json(state_path, state)

        epoch_indices = (
            itertools.count(cursor["epoch_index"])
            if args.training_seconds is not None
            else range(cursor["epoch_index"], args.epochs)
        )
        budget_reached = False
        for epoch in epoch_indices:
            model.train()
            if cursor["order"] is None:
                order = list(range(len(prepared)))
                random.shuffle(order)
                offset_start = 0
            else:
                order = cursor["order"]
                offset_start = cursor["next_offset"]
            epoch_losses = []
            for offset in range(offset_start, len(order), accum):
                group = order[offset : offset + accum]
                optimizer.zero_grad(set_to_none=True)
                for index in group:
                    record, full, labels, weights = prepared[index]
                    loss = loss_for(full, labels, weights)
                    if not torch.isfinite(loss):
                        raise ValueError("non-finite training loss")
                    epoch_losses.append(float(loss.detach()))
                    (loss / len(group)).backward()
                    state.update(
                        last_workflow_id=record["workflow_id"],
                        last_microbatch_loss=float(loss.detach()),
                        training_seconds_this_run=time.monotonic()
                        - optimization_started,
                        updated_unix=time.time(),
                    )
                    write_json(state_path, state)
                    if not grad_verified:
                        nonzero = [
                            name
                            for name, p in model.named_parameters()
                            if p.requires_grad
                            and p.grad is not None
                            and bool(p.grad.abs().sum() > 0)
                        ]
                        if not nonzero or any(
                            p.grad is not None
                            for p in model.parameters()
                            if not p.requires_grad
                        ):
                            raise ValueError("LoRA/base gradient isolation failed")
                        grad_verified = True
                        print(
                            json.dumps(
                                {
                                    "gradient_nonzero_parameters": len(nonzero),
                                    "base_gradients_present": False,
                                }
                            ),
                            flush=True,
                        )
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                if not torch.isfinite(grad_norm):
                    raise ValueError("non-finite gradient norm")
                optimizer.step()
                used_learning_rate = optimizer.param_groups[0]["lr"]
                scheduler.step()
                state["optimizer_steps"] += 1
                entry = {
                    "epoch": epoch + 1,
                    "step": state["optimizer_steps"],
                    "loss": sum(epoch_losses[-len(group) :]) / len(group),
                    "gradient_norm": float(grad_norm),
                    "learning_rate": used_learning_rate,
                    "training_seconds_total": training_seconds(),
                    "unix": time.time(),
                }
                with (args.output_dir / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(entry) + "\n")
                cursor = next_cursor(epoch, order, offset + len(group))
                state.update(
                    last_loss=entry["loss"],
                    epoch=epoch + 1,
                    training_seconds_total=training_seconds(),
                )
                write_json(state_path, state)
                print(json.dumps(entry), flush=True)
                budget_reached = args.training_seconds is not None and (
                    time.monotonic() - optimization_started >= args.training_seconds
                )
                if (
                    args.training_seconds is not None
                    and state["optimizer_steps"] % args.checkpoint_every_steps == 0
                ) or budget_reached:
                    save_checkpoint(cursor, epoch)
                if budget_reached:
                    break
            save_checkpoint(cursor, epoch)
            if budget_reached:
                break
        elapsed_training = time.monotonic() - optimization_started
        total_training = training_seconds()
        optimization_started = None
        final = (
            []
            if args.skip_train_loss_evaluation
            else evaluate("final_train_evaluation")
        )
        changed = sum(
            not torch.equal(initial[name], p.detach().cpu())
            for name, p in model.named_parameters()
            if p.requires_grad
        )
        if not changed:
            raise ValueError("optimization did not change any adapter parameters")
        report = {
            **manifest,
            "status": "passed",
            "epochs": state.get("epoch", args.epochs),
            "optimizer_steps": state["optimizer_steps"],
            "optimizer_steps_this_run": state["optimizer_steps"]
            - optimizer_steps_before,
            "training_seconds_this_run": elapsed_training,
            "training_seconds_total": total_training,
            "training_seconds_requested": args.training_seconds,
            "final_checkpoint": state["checkpoint"],
            "train_loss_evaluation_skipped": args.skip_train_loss_evaluation,
            "lora_target_modules": len(modules),
            "chunked_text_mlp_modules": chunked_modules,
            "device_local_checkpoint_modules": device_local_modules,
            "chunked_final_norm_modules": final_norm_modules,
            "trainable_parameters": sum(p.numel() for p in parameters),
            "changed_parameter_tensors": changed,
            "base_gradients_present": False,
            "response_only_loss": True,
            "initial_train_loss": sum(r["loss"] for r in baseline) / len(baseline)
            if baseline
            else None,
            "final_train_loss": sum(r["loss"] for r in final) / len(final)
            if final
            else None,
            "final_per_workflow_loss": final,
            "peak_gpu_gib": torch.cuda.max_memory_allocated(args.device) / 1024**3,
            "peak_gpu_by_device_gib": {
                device: torch.cuda.max_memory_allocated(device) / 1024**3
                for device in sorted(
                    set(config["model"].get("device_map", {"": args.device}).values())
                )
            },
            "elapsed_seconds": time.time() - started,
            "heldout_accuracy_evaluated": False,
        }
        write_json(args.output_dir / "report.json", report)
        state.update(
            status="completed",
            report=str(args.output_dir / "report.json"),
            finished_unix=time.time(),
        )
        write_json(state_path, state)
        print(
            json.dumps(
                {
                    "status": "completed",
                    "initial_loss": report["initial_train_loss"],
                    "final_loss": report["final_train_loss"],
                }
            ),
            flush=True,
        )
    except Exception as error:
        state.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            finished_unix=time.time(),
        )
        write_json(state_path, state)
        raise


if __name__ == "__main__":
    main()
