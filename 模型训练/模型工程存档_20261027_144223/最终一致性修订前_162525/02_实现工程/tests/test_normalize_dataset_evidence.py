import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "scripts"
    / "normalize_dataset_evidence.py"
)


class DatasetNormalizationTests(unittest.TestCase):
    def test_normalizes_every_input_without_adding_targets(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw_root = root / "raw"
            image_path = raw_root / "workflow" / "1.1.1" / "receipt.png"
            image_path.parent.mkdir(parents=True)
            Image.new("RGB", (8, 6), "white").save(image_path)
            inputs_path = root / "train_inputs.jsonl"
            bundle = {
                "schema_version": "1.0",
                "input_version": "bundle_v1",
                "workflow_id": "1.1.1",
                "source_family_id": "sf_001",
                "input": {"claimed_amount": "12.30", "currency": "USD"},
                "evidence": [
                    {
                        "evidence_id": "e001",
                        "raw_ref": "workflow/1.1.1/receipt.png",
                        "type": "image",
                        "role_hint": "receipt",
                        "owners": [],
                    }
                ],
            }
            inputs_path.write_text(
                json.dumps(bundle, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            output_dir = root / "normalized"

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--inputs",
                    str(inputs_path),
                    "--raw-root",
                    str(raw_root),
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
            normalized = json.loads(
                (output_dir / "train_inputs.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(normalized["workflow_id"], "1.1.1")
            self.assertEqual(normalized["evidence"][0]["parse_status"], "visual_only")
            self.assertNotIn("target", normalized)
            queue = json.loads(
                (output_dir / "train_review_queue.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(queue["actions"], ["multimodal_reading"])

    def test_rejects_dataset_name_that_could_escape_output_directory(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--dataset-name", "../outside"],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid dataset name", result.stderr)


if __name__ == "__main__":
    unittest.main()
