"""Exercise staged training/checkpoint/inference on a labelled engineering fixture."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import yaml

from audit_engine.examples import patent_fixture as fixture
from audit_engine.inference.runner import AuditRunner
from audit_engine.training.patent import TrainingCase, feature_path, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("choose a new engineering verification directory")
    torch.set_num_threads(4)
    torch.manual_seed(3407)
    config = yaml.safe_load(Path("configs/patent_delivery.yaml").read_text())
    # The engineering fixture sets a small learning rate to retain a
    # detectable relation error for the C-stage gradient check.
    config["heads"]["learning_rate"] = 0.000001
    from audit_engine.models.heads import AuditHeads

    model = AuditHeads(4096, 512, 256).to(args.device)
    with torch.no_grad():
        model.relation.classifier.bias.copy_(
            torch.tensor([-8.0, 8.0, -8.0], device=args.device)
        )
    dataset = {"train": [], "validation": [], "test": []}
    features_root = args.output / "features"
    features_root.mkdir(parents=True)
    sources = args.output / "source"
    sources.mkdir()
    (sources / "receipt.txt").write_text("Lodging 90.00")
    (sources / "policy.txt").write_text("Nightly cap 100")
    cases, annotations = [], []
    for split in dataset:
        case = fixture.sample_case("engineering-" + split)
        features = fixture.fixture_features(case, 4096)
        annotation = fixture.fixture_annotation(
            case, split, "engineering-family-" + split
        )
        dataset[split].append(TrainingCase(case, features, [annotation]))
        torch.save(asdict(features), feature_path(features_root, case.workflow_id))
        cases.append(case)
        annotations.append(annotation)
    (args.output / "cases.jsonl").write_text(
        "\n".join(c.model_dump_json() for c in cases) + "\n"
    )
    (args.output / "annotations.jsonl").write_text(
        "\n".join(a.model_dump_json() for a in annotations) + "\n"
    )
    summary = train(
        model, dataset, config, args.output / "training", qualification="engineering"
    )
    result = AuditRunner(model).run(
        dataset["test"][0].case, dataset["test"][0].features
    )
    (args.output / "prediction.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "qualification": "engineering",
                "feature_kind": "synthetic_engineering_fixture",
                "three_components_optimized": summary["three_components_optimized"],
                "module_calls": result["module_calls"],
            }
        )
    )


if __name__ == "__main__":
    main()
