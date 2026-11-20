"""Continue the verified three-component checkpoint until an explicit deadline."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from audit_engine.models.heads import AuditHeads
from audit_engine.training.patent import (
    component_validation,
    correction_validation,
    load_dataset,
    validate,
    workflow_loss,
)
from audit_engine.training.stages import configure_trainable_stage


def atomic_save(path, value):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-run", type=Path, required=True)
    parser.add_argument("--switch-run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None:
        parser.error("deadline requires an explicit UTC offset")
    stop_at = deadline.timestamp()
    if time.time() >= stop_at:
        parser.error("deadline is already past")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    stopped = []

    def request_stop(signum, _frame):
        stopped.append(signal.Signals(signum).name)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    started = time.time()
    state = {"pid": os.getpid(), "stage": "loading", "deadline": args.deadline,
             "automatic_accuracy_stop": False, "final_delivery_closed": False,
             "optimizer_steps": 0, "cycles": 0, "device": args.device,
             "official_answers_used": False}

    def status(**values):
        state.update(values, updated_at=datetime.now(timezone.utc).isoformat())
        atomic_json(args.output / "state.json", state)

    def load_input(root):
        checkpoint = root / "weights/audit_heads.pt"
        bundle = torch.load(checkpoint, map_location="cpu", weights_only=True)
        metadata = copy.deepcopy(bundle["metadata"])
        if metadata.get("qualification") != "formal" or not metadata.get("three_components_optimized"):
            raise ValueError("continuous training requires a verified three-component checkpoint")
        config = metadata["config"]
        dataset = load_dataset(root / "reader_cases.jsonl", root / "annotations.jsonl",
                               root / "features", Path(config["data"]["partitions"]))
        if len(dataset["train"]) < config["data"].get("minimum_approved_train_workflows", 20) or not dataset["validation"]:
            raise ValueError("continuous training needs the approved fixed train/validation split")
        for items in dataset.values():
            for item in items:
                if metadata["case_digests"].get(item.case.workflow_id) != item.case.digest():
                    raise ValueError("continuous inputs differ from their source checkpoint")
                item.features.validate(item.case, config["model"]["hidden_size"])
                if item.features.signature not in metadata["feature_signatures"]:
                    raise ValueError("continuous inputs use a different feature encoder")
        c = config["model"]
        model = AuditHeads(c["hidden_size"], c["dimension"], c["width"], c["role_slots"]).to(args.device)
        model.load_state_dict(bundle["model"], strict=True)
        optimizers = {}
        stages = ["B1", "B2", "B3"]
        if any(any(a.augmentation_of for a in item.annotations) for item in dataset["train"]):
            stages.append("B4")
        stages.append("C")
        for stage in stages:
            configure_trainable_stage(model, stage)
            lr = config["correction"]["learning_rate"] if stage == "C" else config["heads"]["learning_rate"]
            optimizers[stage] = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                                 lr=lr * 0.1, weight_decay=config["heads"]["weight_decay"])
        return model, metadata, config, dataset, optimizers

    active_run = args.input_run
    model, metadata, config, dataset, optimizers = load_input(active_run)
    torch.manual_seed(config["project"]["seed"])
    switched = False
    best = None
    last_saved = 0.0

    def measure():
        model.eval()
        metrics = component_validation(model, dataset["validation"], config)
        corrections = correction_validation(model, dataset["validation"], config)
        loss = validate(model, dataset["validation"], "B3", config)
        score = (-(metrics["state_macro_f1"] or 0.0), -(metrics["reference_recall"] or 0.0),
                 metrics["reference_false_positive_fraction"],
                 -(corrections["workflow_mean_net_correction"] or 0.0),
                 corrections["unwanted_changes"], loss if loss is not None else 1e30)
        return score, {"component_validation": metrics, "correction_validation": corrections,
                       "validation_loss": loss, "selection_score": score}

    def snapshot(name):
        nonlocal last_saved
        details = copy.deepcopy(metadata)
        details["continuous_training"] = {**state, "input_run": str(active_run)}
        atomic_save(args.output / name, {"model": model.state_dict(), "metadata": details,
                    "optimizers": {key: opt.state_dict() for key, opt in optimizers.items()},
                    "torch_rng_state": torch.get_rng_state()})
        last_saved = time.time()

    best, initial = measure()
    status(stage="training", input_run=str(active_run), baseline_validation=initial)
    snapshot("best.pt")
    snapshot("latest.pt")
    try:
        while time.time() < stop_at and not stopped:
            if not switched and args.switch_run and (args.switch_run / "weights/audit_heads.pt").is_file():
                ready = args.switch_run / "pipeline_status.json"
                if ready.is_file() and json.loads(ready.read_text()).get("stage") == "pilot_checkpoint_ready":
                    snapshot("before_input_switch.pt")
                    (args.output / "best.pt").replace(args.output / "best_before_input_switch.pt")
                    active_run = args.switch_run
                    model, metadata, config, dataset, optimizers = load_input(active_run)
                    switched = True
                    best, baseline = measure()
                    status(input_run=str(active_run), input_switched=True, baseline_validation=baseline)
                    snapshot("best.pt")
            for stage in optimizers:
                if time.time() >= stop_at or stopped:
                    break
                configure_trainable_stage(model, stage)
                optimizer = optimizers[stage]
                base_lr = config["correction"]["learning_rate"] if stage == "C" else config["heads"]["learning_rate"]
                progress = min(1.0, max(0.0, (time.time() - started) / (stop_at - started)))
                lr = base_lr * 0.1 * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * progress)))
                for group in optimizer.param_groups:
                    group["lr"] = lr
                total, count = 0.0, 0
                training_items = dataset["train"]
                if stage in {"B1", "B2", "B3"}:
                    training_items = [item for item in training_items if not any(a.augmentation_of for a in item.annotations)]
                for index in torch.randperm(len(training_items)).tolist():
                    if time.time() >= stop_at or stopped:
                        break
                    optimizer.zero_grad(set_to_none=True)
                    loss, _ = workflow_loss(model, training_items[index], stage, config, training=True)
                    if loss is None:
                        continue
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("nonfinite continuous training loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["heads"]["gradient_clip"], error_if_nonfinite=True)
                    optimizer.step()
                    state["optimizer_steps"] += 1
                    total += float(loss.detach())
                    count += 1
                status(stage="training", training_stage=stage, learning_rate=lr,
                       train_loss=total / count if count else None)
                if time.time() - last_saved >= 60:
                    snapshot("latest.pt")
            if time.time() >= stop_at or stopped:
                break
            state["cycles"] += 1
            score, metrics = measure()
            status(stage="training", **metrics)
            with (args.output / "training.jsonl").open("a") as stream:
                stream.write(json.dumps(state, ensure_ascii=False) + "\n")
            if score < best:
                best = score
                snapshot("best.pt")
            if time.time() - last_saved >= 60:
                snapshot("latest.pt")
        reason = stopped[-1] if stopped else "user_deadline_reached"
        status(stage="saving_at_stop", stop_reason=reason)
        snapshot("latest.pt")
        status(stage="stopped_with_checkpoint", checkpoint=str(args.output / "latest.pt"),
               best_checkpoint=str(args.output / "best.pt"), stop_reason=reason)
    except Exception as error:
        status(stage="saving_after_error", error=f"{type(error).__name__}: {error}")
        snapshot("latest.pt")
        status(stage="requires_repair")
        raise


if __name__ == "__main__":
    main()
