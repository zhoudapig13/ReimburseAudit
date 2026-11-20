"""Train a source-reading LoRA from reviewed material QA, never audit answers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F
import yaml
from peft import LoraConfig, PeftModel, get_peft_model

from audit_engine.readers.qwen_encoder import FrozenQwenEncoder
from audit_engine.training.official_sft import response_labels
from audit_engine.training.patent import read_jsonl


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--source-root", type=Path, default=Path("data_raw/ReimburseBench")
    )
    parser.add_argument(
        "--partitions", type=Path, default=Path("data/patent_full/partitions.jsonl")
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--initial-adapter", type=Path,
                        help="Continue source QA from an existing task adapter")
    args = parser.parse_args()
    if args.resume and args.initial_adapter:
        parser.error("choose exact resume or adapter initialization")
    initialization = None
    if args.initial_adapter:
        initialization = {"path": str(args.initial_adapter.resolve()),
            "adapter_sha256": hashlib.sha256((args.initial_adapter / "adapter_model.safetensors").read_bytes()).hexdigest(),
            "config_sha256": hashlib.sha256((args.initial_adapter / "adapter_config.json").read_bytes()).hexdigest()}
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "state.json").exists() and not args.resume:
        raise ValueError("use a new run directory or an explicit resume checkpoint")
    config = yaml.safe_load(args.config.read_text())
    partitions = {v["workflow_id"]: v for v in read_jsonl(args.partitions)}
    rows = {
        "train": list(read_jsonl(args.train)),
        "validation": list(read_jsonl(args.validation)),
    }
    for split, data in rows.items():
        if not data:
            raise ValueError("both approved reading-QA partitions are required")
        for row in data:
            partition = partitions[row["workflow_id"]]
            if (
                row["split"] != split
                or partition["split"] != split
                or row["source_family_id"] != partition["source_family_id"]
                or row["status"] != "approved"
                or not row["reviewer"]
                or row["reference_answer_read"] is not False
            ):
                raise ValueError("unapproved or incorrectly partitioned reading QA")
    hashes = {
        split: hashlib.sha256(path.read_bytes()).hexdigest()
        for split, path in (("train", args.train), ("validation", args.validation))
    }
    torch.manual_seed(config["project"]["seed"])
    random.seed(config["project"]["seed"])
    torch.set_num_threads(4)
    state = {
        "pid": os.getpid(),
        "stage": "loading_backbone",
        "optimizer_steps": 0,
        "supervision": "reviewed_source_reading_qa",
        "official_answers_used": False,
        "data_sha256": hashes,
        "initial_adapter": initialization,
    }
    write_json(args.output / "state.json", state)
    encoder = FrozenQwenEncoder(config, args.model_path, args.device)
    processor = encoder.processor
    count = encoder.model.config.text_config.num_hidden_layers
    settings = config["lora"]
    layers = "|".join(
        str(i) for i in range(count - settings["last_text_layers"], count)
    )
    pattern = rf".*language_model\.layers\.({layers})\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    targets = [
        name for name, _ in encoder.model.named_modules() if re.fullmatch(pattern, name)
    ]
    if len(targets) != settings["last_text_layers"] * 4:
        raise ValueError("unexpected decoder LoRA target modules")
    if args.resume or args.initial_adapter:
        source_adapter = args.resume or args.initial_adapter
        adapter_config = json.loads((source_adapter / "adapter_config.json").read_text())
        if (adapter_config["r"], adapter_config["lora_alpha"], adapter_config["lora_dropout"], adapter_config["target_modules"]) != (
            settings["rank"], settings["alpha"], settings["dropout"], pattern):
            raise ValueError("initial adapter topology differs from the configured reader LoRA")
        model = PeftModel.from_pretrained(encoder.model, source_adapter, is_trainable=True)
    else:
        model = get_peft_model(
            encoder.model,
            LoraConfig(
                r=settings["rank"],
                lora_alpha=settings["alpha"],
                lora_dropout=settings["dropout"],
                target_modules=pattern,
                task_type="CAUSAL_LM",
            ),
        )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.config.use_cache = False
    parameters = [p for p in model.parameters() if p.requires_grad]
    for parameter in parameters:
        parameter.data = parameter.data.float()
    optimizer = torch.optim.AdamW(
        parameters, lr=settings["learning_rate"], weight_decay=0.01
    )
    accumulation = settings["gradient_accumulation"]
    epochs = settings["max_epochs"]
    total_steps = math.ceil(len(rows["train"]) / accumulation) * epochs
    warmup = max(1, math.ceil(total_steps * 0.05))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (
            (step + 1) / warmup
            if step < warmup
            else 0.5
            * (
                1
                + math.cos(
                    math.pi * min(1.0, (step - warmup) / max(1, total_steps - warmup))
                )
            )
        ),
    )
    prepared = {"train": [], "validation": []}
    for split, data in rows.items():
        for row in data:
            raw = (args.source_root / row["raw_ref"]).resolve()
            if (
                not raw.is_relative_to(args.source_root.resolve())
                or hashlib.sha256(raw.read_bytes()).hexdigest() != row["source_sha256"]
            ):
                raise ValueError("source QA raw material changed")
            content = []
            if row["image_ref"]:
                if row["image_ref"] != row["raw_ref"]:
                    raise ValueError("reading QA requires a source-bound image")
                content.append({"type": "image", "image": str(raw)})
            content.append({"type": "text", "text": row["prompt"]})
            messages = [{"role": "user", "content": content}]
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
                        "content": [{"type": "text", "text": row["response"]}],
                    }
                ],
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
            )
            if full.input_ids.size(1) > config["reader"]["total_input_tokens"]:
                raise ValueError(
                    "reviewed QA exceeds the input budget; no silent truncation"
                )
            labels = response_labels(prompt.input_ids, full.input_ids)
            prepared[split].append((row["qa_id"], full, labels))
    manifest = {
        "config": config,
        "data_sha256": hashes,
        "train_qa": len(prepared["train"]),
        "validation_qa": len(prepared["validation"]),
        "train_workflows": sorted({r["workflow_id"] for r in rows["train"]}),
        "validation_workflows": sorted({r["workflow_id"] for r in rows["validation"]}),
        "lora_modules": targets,
        "trainable_parameters": sum(p.numel() for p in parameters),
        "official_answers_used": False,
        "initial_adapter": initialization,
    }
    write_json(args.output / "manifest.json", manifest)

    def loss_for(item):
        _, full, labels = item
        inputs = {key: value.to(args.device) for key, value in full.items()}
        keep = int((labels != -100).sum()) + 1
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(**inputs, use_cache=False, logits_to_keep=keep)
        wanted = labels[:, -keep:].to(output.logits.device)
        return F.cross_entropy(
            output.logits[:, :-1].float().reshape(-1, output.logits.size(-1)),
            wanted[:, 1:].reshape(-1),
        )

    @torch.no_grad()
    def validation():
        model.eval()
        return sum(float(loss_for(item)) for item in prepared["validation"]) / len(
            prepared["validation"]
        )

    cursor = {"epoch": 0, "offset": 0, "order": None}
    best = float("inf")
    if args.resume:
        saved = torch.load(
            args.resume / "training_state.pt", map_location="cpu", weights_only=False
        )
        if saved["config"] != config or saved["data_sha256"] != hashes:
            raise ValueError("resume config or QA data differs")
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        torch.set_rng_state(saved["torch_rng"])
        random.setstate(saved["python_rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"], args.device)
        cursor, best = saved["cursor"], saved["best_validation"]
        state["optimizer_steps"] = saved["optimizer_steps"]
        initialization = saved.get("initial_adapter")
        manifest["initial_adapter"] = state["initial_adapter"] = initialization
        write_json(args.output / "manifest.json", manifest)

    def checkpoint(name):
        target = args.output / name
        model.save_pretrained(target)
        torch.save(
            {
                "config": config,
                "data_sha256": hashes,
                "initial_adapter": initialization,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "python_rng": random.getstate(),
                "cuda_rng": torch.cuda.get_rng_state(args.device),
                "cursor": cursor,
                "optimizer_steps": state["optimizer_steps"],
                "best_validation": best,
            },
            target / "training_state.pt",
        )
        return target

    baseline = validation()
    write_json(args.output / "validation_before.json", {"loss": baseline})
    started = time.monotonic()
    stop = {"requested": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(requested=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(requested=True))
    optimizer.zero_grad(set_to_none=True)
    while cursor["epoch"] < epochs:
        if cursor["order"] is None:
            cursor["order"] = torch.randperm(len(prepared["train"])).tolist()
        losses = []
        while cursor["offset"] < len(cursor["order"]):
            model.train()
            start = cursor["offset"]
            group = cursor["order"][start : start + accumulation]
            for index in group:
                loss = loss_for(prepared["train"][index])
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite reading-QA loss")
                (loss / len(group)).backward()
                losses.append(float(loss.detach()))
            if not any(
                p.grad is not None and bool(p.grad.ne(0).any()) for p in parameters
            ):
                raise RuntimeError("source reading adapter has no nonzero gradients")
            if any(
                p.grad is not None
                for name, p in model.named_parameters()
                if "lora_" not in name
            ):
                raise RuntimeError("frozen base or vision tower received gradients")
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            cursor["offset"] += len(group)
            state.update(
                stage="training_source_reader",
                optimizer_steps=state["optimizer_steps"] + 1,
                epoch=cursor["epoch"] + 1,
                qa_offset=cursor["offset"],
                elapsed_seconds=time.monotonic() - started,
                last_loss=losses[-1],
                peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(args.device),
            )
            write_json(args.output / "state.json", state)
            if stop["requested"]:
                target = checkpoint("user_stop")
                state.update(
                    stage="stopped_after_user_signal",
                    adapter=str(target),
                    final_delivery_closed=False,
                )
                write_json(args.output / "state.json", state)
                return
            if state["optimizer_steps"] % 5 == 0:
                checkpoint("latest")
        score = validation()
        row = {
            "epoch": cursor["epoch"] + 1,
            "train_loss": sum(losses) / len(losses) if losses else None,
            "validation_loss": score,
            "optimizer_steps": state["optimizer_steps"],
            "elapsed_seconds": time.monotonic() - started,
        }
        if not math.isfinite(score):
            raise FloatingPointError("nonfinite validation loss")
        cursor.update(epoch=cursor["epoch"] + 1, offset=0, order=None)
        checkpoint(f"epoch_{cursor['epoch']}")
        if score < best:
            best = score
            checkpoint("best")
        with (args.output / "training.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row), flush=True)
    state.update(
        stage="source_reader_adapter_ready",
        adapter=str(args.output / "best"),
        validation_before=baseline,
        best_validation_loss=best,
        automatic_accuracy_stop=False,
        final_delivery_closed=False,
    )
    write_json(args.output / "state.json", state)


if __name__ == "__main__":
    main()
