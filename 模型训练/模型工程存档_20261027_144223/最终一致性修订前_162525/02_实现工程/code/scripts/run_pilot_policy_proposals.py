"""Run local policy proposal shards and assemble a pending review queue."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    logs_root = Path("runs/policy_candidates")
    logs_root.mkdir(parents=True, exist_ok=True)
    output = Path("data/policy_candidates")
    children = []
    started_at = datetime.now(timezone.utc).isoformat()

    def record(status, **extra):
        payload = {
            "controller_pid": os.getpid(),
            "child_pids": children,
            "status": status,
            "started_at": started_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **extra,
        }
        temporary = logs_root / "state.tmp"
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(logs_root / "state.json")
        print(json.dumps(payload), flush=True)

    try:
        processes, logs = [], []
        for index in range(4):
            log = (logs_root / f"shard_{index}.log").open("a")
            logs.append(log)
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "code/scripts/propose_pilot_policies.py",
                    "--shard",
                    f"{index}/4",
                    "--device",
                    f"cuda:{index}",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            processes.append(process)
        children = [p.pid for p in processes]
        record("running", stage="generating_policy_candidates")
        results = [p.wait() for p in processes]
        for log in logs:
            log.close()
        children = []
        if any(results):
            raise RuntimeError(f"candidate workers exited with codes {results}")
        planned = {
            (j["workflow_id"], j["evidence_id"], j["view_id"])
            for j in (
                json.loads(line)
                for line in Path("data/features_pilot/encoding_jobs.jsonl").open()
            )
            if j["kind"] == "text_view"
        }
        seen, signature, review, failures = set(), None, [], []
        for index in range(4):
            shard = output / f"shard_{index:04d}"
            report = json.loads((shard / "report.json").read_text())
            if report["approved_labels"] or report["reference_answers_read"]:
                raise ValueError("candidate source/approval contract failed")
            if signature is None:
                signature = report["signature"]
            elif signature != report["signature"]:
                raise ValueError("candidate signature mismatch")
            for line in (shard / "manifest.jsonl").open():
                item = json.loads(line)
                key = item["workflow_id"], item["evidence_id"], item["view_id"]
                if key not in planned or key in seen:
                    raise ValueError("unknown or duplicate source view")
                seen.add(key)
                path = Path(item["path"]).resolve()
                if not path.is_relative_to(output.resolve()):
                    raise ValueError("unsafe proposal path")
                candidate = json.loads(path.read_text())
                if (
                    candidate["approved_training_label"]
                    or candidate["annotation_status"] != "pending"
                ):
                    raise ValueError("unapproved candidate changed status")
                if candidate["schema_status"] != "passed":
                    failures.append(
                        {
                            "workflow_id": key[0],
                            "evidence_id": key[1],
                            "view_id": key[2],
                            "path": item["path"],
                            "issue": candidate["schema_error"],
                        }
                    )
                for condition in candidate["conditions"]:
                    review.append(
                        {
                            "workflow_id": key[0],
                            "evidence_id": key[1],
                            "view_id": key[2],
                            "input_version": candidate["input_version"],
                            "source_sha256": candidate["source_sha256"],
                            "proposal_path": item["path"],
                            **condition,
                            "review_action": "pending",
                            "review_reason": None,
                            "reviewer": None,
                        }
                    )
        if seen != planned:
            raise ValueError("not all source views have a recorded result")
        (output / "review_queue.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in review)
        )
        (output / "schema_failures.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in failures)
        )
        manifest = {
            "status": "completed_with_pending_review",
            "input_plan_sha256": hashlib.sha256(
                Path("data/features_pilot/encoding_jobs.jsonl").read_bytes()
            ).hexdigest(),
            "signature": signature,
            "workflow_count": len({x[0] for x in seen}),
            "text_views": len(seen),
            "condition_proposals": len(review),
            "schema_failures": len(failures),
            "conditions_with_all_literal_refs": sum(
                all(ref["existence_verified"] for ref in r["policy_refs"])
                for r in review
            ),
            "approved_labels": 0,
            "reference_answers_read": False,
            "semantic_coverage_verified": False,
            "scope": "policy_proposals_only_not_relation_or_amount_labels",
            "outputs": ["review_queue.jsonl", "schema_failures.jsonl"],
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        (output / "stage_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        record(
            "completed_with_pending_review",
            stage="review_queue_ready",
            manifest="data/policy_candidates/stage_manifest.json",
        )
    except Exception as error:
        record("failed", error_type=type(error).__name__, message=str(error))
        raise


if __name__ == "__main__":
    main()
