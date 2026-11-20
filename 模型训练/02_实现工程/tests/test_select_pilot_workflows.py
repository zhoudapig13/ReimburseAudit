import importlib.util
import unittest
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "scripts"
    / "select_pilot_workflows.py"
)


def load_selection_module():
    spec = importlib.util.spec_from_file_location("select_pilot_workflows", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load selection module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assignment(
    workflow_id,
    family_id,
    split="train",
    difficulty="medium",
    reference_flag=0,
    expected_types=None,
):
    return {
        "workflow_id": workflow_id,
        "source_family_id": family_id,
        "split": split,
        "difficulty": difficulty,
        "reference_flag": reference_flag,
        "expected_types": expected_types or [],
    }


def bundle(workflow_id, evidence_types):
    return {
        "workflow_id": workflow_id,
        "evidence": [
            {"evidence_id": f"e{index:03d}", "type": evidence_type}
            for index, evidence_type in enumerate(evidence_types, start=1)
        ],
    }


class PilotSelectionTests(unittest.TestCase):
    def test_selection_is_deterministic_train_only_and_family_unique(self):
        selector = load_selection_module()
        assignments = [
            assignment("1.1.1", "sf_001", difficulty="hard", reference_flag=1),
            assignment("1.1.2", "sf_001", difficulty="easy"),
            assignment("2.1.1", "sf_002", difficulty="medium"),
            assignment(
                "3.1.1",
                "sf_003",
                difficulty="easy",
                expected_types=["Duplicate reimbursement"],
            ),
            assignment("4.1.1", "sf_004", split="validation"),
        ]
        bundles = [
            bundle("1.1.1", ["pdf", "image"]),
            bundle("1.1.2", ["xlsx", "image"]),
            bundle("2.1.1", ["docx", "image"]),
            bundle("3.1.1", ["json", "image"]),
            bundle("4.1.1", ["pdf"]),
        ]

        first = selector.select_pilot_workflows(
            assignments, bundles, limit=3, seed=3407
        )
        second = selector.select_pilot_workflows(
            assignments, bundles, limit=3, seed=3407
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertEqual(len({row["source_family_id"] for row in first}), 3)
        self.assertNotIn("4.1.1", {row["workflow_id"] for row in first})
        self.assertEqual([row["selection_rank"] for row in first], [1, 2, 3])

    def test_selection_requires_enough_train_families(self):
        selector = load_selection_module()
        assignments = [assignment("1.1.1", "sf_001")]
        bundles = [bundle("1.1.1", ["pdf"])]

        with self.assertRaisesRegex(ValueError, "train source families"):
            selector.select_pilot_workflows(assignments, bundles, limit=2, seed=3407)

    def test_selection_includes_required_workflow(self):
        selector = load_selection_module()
        assignments = [
            assignment("1.1.1", "sf_001"),
            assignment("1.1.2", "sf_001"),
            assignment("2.1.1", "sf_002"),
        ]
        bundles = [
            bundle("1.1.1", ["pdf"]),
            bundle("1.1.2", ["xlsx"]),
            bundle("2.1.1", ["docx"]),
        ]

        selected = selector.select_pilot_workflows(
            assignments,
            bundles,
            limit=2,
            seed=3407,
            required_workflow_ids=["1.1.1"],
        )

        self.assertEqual(selected[0]["workflow_id"], "1.1.1")
        self.assertEqual(
            {row["source_family_id"] for row in selected}, {"sf_001", "sf_002"}
        )


if __name__ == "__main__":
    unittest.main()
