import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

SCRIPT = Path(__file__).resolve().parents[1] / "code/scripts/merge_pilot_features.py"


class FrozenFeatureMergeTests(unittest.TestCase):
    def fixture(self, root, missing=False):
        encoded = root / "encoded"
        signature = {
            "revision": "pinned_fixture_revision",
            "encoding_prompt": "facts_only",
        }
        jobs = []
        for index in range(2):
            folder = encoded / f"shard_{index:04d}"
            folder.mkdir(parents=True)
            summary = {
                "status": "passed",
                "feature_dimension": 4096,
                "projection_applied": False,
                "labels_read": False,
                "base_signature": signature,
            }
            (folder / "summary.json").write_text(json.dumps(summary))
            job = {
                "workflow_id": "fixture",
                "evidence_id": "e001",
                "view_id": f"view{index}",
                "view_count": 2,
                "source_sha256": "unchanged_source",
                "split": "train",
                "base_revision": signature["revision"],
            }
            jobs.append(job)
            vector_path = folder / "vector.pt"
            torch.save(
                {
                    "vector": torch.full((4096,), float(2 + index * 2)),
                    "signature": signature,
                    "cache_key": str(index),
                },
                vector_path,
            )
            record = {
                key: value
                for key, value in job.items()
                if key not in {"split", "base_revision"}
            }
            record.update({"path": str(vector_path), "cache_key": str(index)})
            (folder / "manifest.jsonl").write_text(
                "" if missing and index == 1 else json.dumps(record) + "\n"
            )
        jobs_path = root / "jobs.jsonl"
        jobs_path.write_text("".join(json.dumps(job) + "\n" for job in jobs))
        return jobs_path, encoded

    def run_merge(self, root, jobs, encoded):
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--jobs",
                str(jobs),
                "--encoded-root",
                str(encoded),
                "--output-dir",
                str(root / "pooled"),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_all_views_are_pooled_before_trainable_projection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs, encoded = self.fixture(root)
            result = self.run_merge(root, jobs, encoded)
            self.assertEqual(result.returncode, 0, result.stderr)
            stored = torch.load(root / "pooled/fixture/e001.pt", weights_only=True)
            self.assertTrue(
                torch.equal(stored["raw_evidence"], torch.full((4096,), 3.0))
            )
            self.assertEqual(stored["view_ids"], ["view0", "view1"])

    def test_incomplete_material_is_rejected_before_any_pooled_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs, encoded = self.fixture(root, missing=True)
            result = self.run_merge(root, jobs, encoded)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("coverage incomplete", result.stderr)
            self.assertFalse((root / "pooled").exists())


if __name__ == "__main__":
    unittest.main()
