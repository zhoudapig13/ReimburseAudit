import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "scripts"
    / "align_benchmark_dataset.py"
)


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


class DatasetAlignmentTests(unittest.TestCase):
    def test_writes_only_usable_train_pairs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            inputs_path = root / "inputs.jsonl"
            targets_path = root / "targets.jsonl"
            output_dir = root / "training"
            bundle = {
                "schema_version": "1.0",
                "input_version": "normalized_bundle_v1",
                "workflow_id": "1.1.1",
                "source_family_id": "sf_001",
                "input": {"claimed_amount": "100", "currency": "USD"},
                "evidence": [],
            }
            target = {
                "schema_version": "1.0",
                "target_version": "benchmark_reference_v1",
                "input_version": "bundle_v1",
                "workflow_id": "1.1.1",
                "source_family_id": "sf_001",
                "split": "train",
                "target": {
                    "reference_flag": 0,
                    "reference_max_legal": "90",
                    "problem_reports": [],
                    "logic_types": [],
                    "scoring_points": [],
                    "reference_score_sum": "0.50",
                },
                "annotation": {
                    "status": "reference_only",
                    "method": "benchmark_reference_conversion",
                    "approved_for_component_training": False,
                    "issues": [],
                },
            }
            write_jsonl(inputs_path, [bundle])
            write_jsonl(targets_path, [target])

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--inputs",
                    str(inputs_path),
                    "--targets",
                    str(targets_path),
                    "--output-dir",
                    str(output_dir),
                    "--dataset-name",
                    "train",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            pair = json.loads(
                (output_dir / "train_benchmark_pairs.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(pair["workflow_id"], "1.1.1")
            self.assertFalse(pair["annotation"]["approved_for_component_training"])
            summary = json.loads(
                (output_dir / "train_benchmark_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(summary["benchmark_pair_count"], 1)
            self.assertEqual(summary["workflow_count"], 1)
            self.assertNotIn("pilot_workflow_count", summary)


if __name__ == "__main__":
    unittest.main()
