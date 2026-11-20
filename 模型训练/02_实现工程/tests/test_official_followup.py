import json
import os
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

namespace = runpy.run_path(
    str(
        Path(__file__).resolve().parents[1]
        / "audit_engine/training/job_wait.py"
    )
)
wait_completed = namespace["wait_completed"]


class FollowupStartupTests(unittest.TestCase):
    def test_live_process_can_start_before_its_state_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            control, job = root / "control", root / "job"
            control.mkdir()
            job.mkdir()
            state = {}

            def finish_startup(_):
                (job / "state.json").write_text(json.dumps({"status": "completed"}))
                (job / "report.json").write_text(
                    json.dumps({"status": "passed", "optimizer_steps": 1})
                )

            with (
                patch.dict(
                    wait_completed.__globals__,
                    {
                        "ROOT": root,
                        "CONTROL": control,
                        "live_training": lambda s, d: s["pid"] == os.getpid(),
                    },
                ),
                patch("time.sleep", side_effect=finish_startup),
            ):
                _, report = wait_completed(job, state)
            self.assertEqual(report["optimizer_steps"], 1)
            self.assertEqual(state["verified_live_pid"], os.getpid())
            self.assertEqual(state["prerequisite_stage"], "starting_before_state_file")

    def test_missing_state_without_live_job_fails_instead_of_waiting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.dict(
                wait_completed.__globals__,
                {"ROOT": root, "live_training": lambda s, d: False},
            ):
                with self.assertRaises(RuntimeError):
                    wait_completed(root / "missing_job", {})
