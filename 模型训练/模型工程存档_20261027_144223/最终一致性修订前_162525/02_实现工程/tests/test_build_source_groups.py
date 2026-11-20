import importlib.util
import unittest
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "code" / "scripts" / "build_source_groups.py"
)


def load_split_module():
    spec = importlib.util.spec_from_file_location("build_source_groups", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load split module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def workflow(
    workflow_id,
    *,
    difficulty="easy",
    flag=0,
    expected_types=None,
    files=None,
):
    return {
        "workflow_id": workflow_id,
        "difficulty": difficulty,
        "reference_flag": flag,
        "expected_types": expected_types or [],
        "files": files or [],
    }


def file_record(category, sha256, relative_path):
    return {
        "category": category,
        "sha256": sha256,
        "relative_path": relative_path,
        "exists": True,
    }


class SourceFamilyTests(unittest.TestCase):
    def test_extracts_two_part_scenario_prefix(self):
        split = load_split_module()

        self.assertEqual(split.scenario_prefix("4.14.2"), "4.14")

    def test_same_prefix_workflows_stay_in_one_family(self):
        split = load_split_module()
        rows = [workflow("1.1.1"), workflow("1.1.2"), workflow("1.2.1")]

        _, workflow_to_family = split.build_source_families(rows)

        self.assertEqual(workflow_to_family["1.1.1"], workflow_to_family["1.1.2"])
        self.assertNotEqual(workflow_to_family["1.1.1"], workflow_to_family["1.2.1"])

    def test_exact_receipt_hash_links_different_prefixes(self):
        split = load_split_module()
        rows = [
            workflow(
                "1.1.1",
                files=[file_record("receipt", "same", "ticket-a.png")],
            ),
            workflow(
                "2.2.1",
                files=[file_record("receipt", "same", "ticket-b.png")],
            ),
        ]

        _, workflow_to_family = split.build_source_families(rows)

        self.assertEqual(workflow_to_family["1.1.1"], workflow_to_family["2.2.1"])

    def test_shared_background_hash_does_not_link_prefixes(self):
        split = load_split_module()
        rows = [
            workflow(
                "1.1.1",
                files=[file_record("background", "policy", "policy.pdf")],
            ),
            workflow(
                "2.2.1",
                files=[file_record("background", "policy", "policy.pdf")],
            ),
        ]

        _, workflow_to_family = split.build_source_families(rows)

        self.assertNotEqual(workflow_to_family["1.1.1"], workflow_to_family["2.2.1"])

    def test_verified_visual_receipt_match_links_prefixes(self):
        split = load_split_module()
        rows = [workflow("1.1.1"), workflow("2.2.1")]
        visual_links = [
            {
                "left_workflow": "1.1.1",
                "right_workflow": "2.2.1",
                "left_path": "ticket-a.jpg",
                "right_path": "ticket-b.jpg",
                "pixel_mae": 0.003,
                "gray_correlation": 0.99,
                "edge_correlation": 0.98,
            }
        ]

        _, workflow_to_family = split.build_source_families(
            rows, visual_links=visual_links
        )

        self.assertEqual(workflow_to_family["1.1.1"], workflow_to_family["2.2.1"])


class SplitAssignmentTests(unittest.TestCase):
    def test_assignment_is_deterministic_and_family_atomic(self):
        split = load_split_module()
        rows = []
        for index in range(1, 13):
            rows.append(
                workflow(
                    f"{index}.1.1",
                    difficulty=("easy", "medium", "hard")[index % 3],
                    flag=index % 2,
                    expected_types=[f"L{index % 4}"],
                )
            )
            rows.append(
                workflow(
                    f"{index}.1.2",
                    difficulty=("easy", "medium", "hard")[index % 3],
                    flag=index % 2,
                    expected_types=[f"L{index % 4}"],
                )
            )

        families, workflow_to_family = split.build_source_families(rows)
        first = split.assign_splits(families, rows, seed=3407)
        second = split.assign_splits(families, rows, seed=3407)

        self.assertEqual(first, second)
        self.assertEqual(set(first.values()), {"train", "validation", "test"})
        for family in families:
            assigned = {first[workflow_id] for workflow_id in family["workflows"]}
            self.assertEqual(len(assigned), 1)
        self.assertEqual(
            split.validate_assignments(rows, families, first),
            [],
        )
        self.assertEqual(len(workflow_to_family), len(rows))


if __name__ == "__main__":
    unittest.main()
