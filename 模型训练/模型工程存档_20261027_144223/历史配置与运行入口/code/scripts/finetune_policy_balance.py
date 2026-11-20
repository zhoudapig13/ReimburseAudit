"""Bounded policy/semantic scale tuning using source-level validation only."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import yaml

from audit_engine.models.heads import AuditHeads
from audit_engine.training.patent import load_dataset, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gain", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.gain <= 1:
        parser.error("initial projection gain must be in (0, 1]")
    root = Path("runs/patent_targeted_source_20261027_0930")
    torch.set_num_threads(4)
    config = yaml.safe_load((root / "config.yaml").read_text())
    config["heads"]["warm_start_projection_gain"] = args.gain
    config["fast_training"] = {"epochs": {"B1": 12, "B2": 8, "B3": 12, "B4": 6, "C": 8}, "patience": 4}
    torch.manual_seed(config["project"]["seed"])
    dataset = load_dataset(root / "dataset/reader_cases.jsonl", root / "dataset/annotations.jsonl",
                           root / "features", Path(config["data"]["partitions"]))
    c = config["model"]
    model = AuditHeads(c["hidden_size"], c["dimension"], c["width"], c["role_slots"])
    initial = torch.load(root / "weights/audit_heads.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(initial["model"], strict=True)
    # A recorded warm-start parameter transformation; no rule outcomes, model
    # equations, raw facts, held-out answers, or runtime logits are replaced.
    with torch.no_grad():
        for parameter in model.shared.parameters():
            parameter.mul_(args.gain)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "initialization.json").write_text(json.dumps({
        "checkpoint": str(root / "weights/audit_heads.pt"),
        "shared_projection_parameter_gain": args.gain,
        "selection": "source_condition_validation_only", "benchmark_answers_read": False}, indent=2))
    train(model, dataset, config, args.output)


if __name__ == "__main__":
    main()
