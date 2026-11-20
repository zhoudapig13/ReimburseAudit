import importlib.util
import unittest
from decimal import Decimal
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "code" / "scripts" / "pack_training_data.py"
)


def load_pack_module():
    spec = importlib.util.spec_from_file_location("pack_training_data", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load packing module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_task():
    return {
        "id": "1.1.1",
        "required_amount": Decimal("1476.15"),
        "currency": "USD",
        "reference_outputs": ["reference_answer.json"],
        "source_files": {
            "Background_information": ["policy.pdf", "approval.xlsx"],
            "Receipts": {
                "standard": "explanation only",
                "Zhang_Ming": "hotel.jpg",
                "Li_Hua": ["taxi.png", "meal.png"],
            },
        },
        "context": "Conference travel expenses",
    }


def sample_manifest():
    files = []
    for relative_path, category, owner, extension in [
        ("policy.pdf", "background", None, ".pdf"),
        ("approval.xlsx", "background", None, ".xlsx"),
        ("hotel.jpg", "receipt", "Zhang_Ming", ".jpg"),
        ("taxi.png", "receipt", "Li_Hua", ".png"),
        ("meal.png", "receipt", "Li_Hua", ".png"),
    ]:
        files.append(
            {
                "relative_path": relative_path,
                "category": category,
                "owner": owner,
                "extension": extension,
                "exists": True,
                "sha256": relative_path,
            }
        )
    return {
        "workflow_id": "1.1.1",
        "difficulty": "hard",
        "expected_types": ["Amount-cap violation"],
        "files": files,
    }


def sample_assignment():
    return {
        "workflow_id": "1.1.1",
        "source_family_id": "sf_001",
        "split": "train",
    }


class InputBundleTests(unittest.TestCase):
    def test_builds_neutral_evidence_and_receipt_owner_mapping(self):
        pack = load_pack_module()

        bundle = pack.build_input_bundle(
            sample_task(), sample_manifest(), sample_assignment()
        )

        self.assertEqual(
            [item["evidence_id"] for item in bundle["evidence"]],
            ["e001", "e002", "e003", "e004", "e005"],
        )
        self.assertEqual(
            bundle["input"]["receipt_owners"],
            [
                {"owner": "Zhang_Ming", "evidence_ids": ["e003"]},
                {"owner": "Li_Hua", "evidence_ids": ["e004", "e005"]},
            ],
        )
        self.assertEqual(bundle["input"]["claimed_amount"], "1476.15")
        self.assertEqual(pack.validate_input_bundle(bundle), [])

    def test_input_bundle_excludes_answer_and_label_fields(self):
        pack = load_pack_module()

        bundle = pack.build_input_bundle(
            sample_task(), sample_manifest(), sample_assignment()
        )
        serialized = pack.json_text(bundle)

        self.assertNotIn("reference_answer", serialized)
        self.assertNotIn("expected_types", serialized)
        self.assertNotIn('"target"', serialized)
        self.assertNotIn('"split"', serialized)


class ReferenceTargetTests(unittest.TestCase):
    def test_reference_target_is_private_and_not_component_approved(self):
        pack = load_pack_module()
        answer = {
            "Scoring_point_1_Flag": {"answer": 0, "score": Decimal("0.25")},
            "Scoring_point_2_Max_legal": {
                "answer": Decimal("340"),
                "score": Decimal("0.25"),
            },
            "Scoring_point_3_Problem_1": {
                "answer": "Hotel rate exceeds policy.",
                "score": Decimal("0.50"),
            },
        }

        target = pack.build_reference_target(
            sample_task(), answer, sample_manifest(), sample_assignment()
        )

        self.assertEqual(target["target"]["reference_flag"], 0)
        self.assertEqual(target["target"]["reference_max_legal"], "340")
        self.assertEqual(target["target"]["logic_types"], ["Amount-cap violation"])
        self.assertFalse(target["annotation"]["approved_for_component_training"])
        self.assertEqual(target["annotation"]["status"], "needs_review")
        self.assertTrue(target["annotation"]["issues"])


if __name__ == "__main__":
    unittest.main()
