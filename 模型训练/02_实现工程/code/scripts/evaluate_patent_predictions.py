"""Score saved predictions only; this process never invokes a model or reader."""

from __future__ import annotations

import argparse
import hashlib
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path


def score(predictions, gold):
    by_id = {}
    for row in predictions:
        key = row["workflow_id"]
        if key in by_id:
            raise ValueError("duplicate prediction")
        by_id[key] = row
    expected = [g["workflow_id"] for g in gold]
    if len(expected) != len(set(expected)):
        raise ValueError("duplicate gold workflow")
    if set(by_id) - set(expected):
        raise ValueError("prediction contains workflows outside the evaluation split")
    correct = 0
    details = []
    for reference in gold:
        prediction = by_id.get(reference["workflow_id"], {})
        answer = prediction.get("answer", {})
        if "target" in reference:
            target = {
                "FLAG": reference["target"]["reference_flag"],
                "max_legal": reference["target"]["reference_max_legal"],
            }
        else:
            target = reference["answer"]
        try:
            hit = (
                prediction.get("status") != "failed"
                and type(answer.get("FLAG")) is int
                and answer["FLAG"] == target["FLAG"]
                and Decimal(str(answer["max_legal"])).is_finite()
                and Decimal(str(answer["max_legal"]))
                == Decimal(str(target["max_legal"]))
            )
        except (KeyError, InvalidOperation, TypeError, ValueError):
            hit = False
        correct += int(hit)
        details.append(
            {
                "workflow_id": reference["workflow_id"],
                "correct": bool(hit),
                "prediction": answer or None,
                "reference": target,
            }
        )
    return {
        "total": len(gold),
        "correct": correct,
        "accuracy": correct / len(gold) if gold else None,
        "details": details,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    saved = args.predictions.read_bytes()
    predictions = [json.loads(line) for line in saved.splitlines() if line.strip()]
    # Reference answers are opened only after all saved predictions are loaded.
    gold = [
        json.loads(line) for line in args.gold.read_text().splitlines() if line.strip()
    ]
    report = score(predictions, gold)
    report["predictions_sha256"] = hashlib.sha256(saved).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "details"}))


if __name__ == "__main__":
    main()
