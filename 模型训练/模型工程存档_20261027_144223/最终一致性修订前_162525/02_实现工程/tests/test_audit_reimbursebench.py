import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "code" / "scripts" / "audit_reimbursebench.py"
)


def load_audit_module():
    spec = importlib.util.spec_from_file_location("audit_reimbursebench", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load audit module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReferencePathTests(unittest.TestCase):
    def test_accepts_file_inside_workflow(self):
        audit = load_audit_module()
        root = Path("C:/dataset/workflow/1.1.1")

        resolved, error = audit.resolve_reference(root, "receipts/hotel.jpg")

        self.assertIsNone(error)
        self.assertEqual(resolved, (root / "receipts" / "hotel.jpg").resolve())

    def test_rejects_parent_escape(self):
        audit = load_audit_module()
        root = Path("C:/dataset/workflow/1.1.1")

        resolved, error = audit.resolve_reference(root, "../reference_answer.json")

        self.assertIsNone(resolved)
        self.assertIn("escapes workflow directory", error)


class ReceiptNormalizationTests(unittest.TestCase):
    def test_normalizes_string_and_list_values(self):
        audit = load_audit_module()

        self.assertEqual(audit.normalize_receipt_value("one.jpg"), ["one.jpg"])
        self.assertEqual(
            audit.normalize_receipt_value(["one.jpg", "two.jpg"]),
            ["one.jpg", "two.jpg"],
        )

    def test_rejects_unsupported_receipt_value(self):
        audit = load_audit_module()

        with self.assertRaises(TypeError):
            audit.normalize_receipt_value({"file": "one.jpg"})


class FlagNormalizationTests(unittest.TestCase):
    def test_normalizes_supported_flag_values(self):
        audit = load_audit_module()

        self.assertEqual(audit.normalize_flag(1), 1)
        self.assertEqual(audit.normalize_flag("yes"), 1)
        self.assertEqual(audit.normalize_flag(0), 0)
        self.assertEqual(audit.normalize_flag("no"), 0)

    def test_returns_none_for_unknown_flag(self):
        audit = load_audit_module()

        self.assertIsNone(audit.normalize_flag("maybe"))


class JsonIntegrityTests(unittest.TestCase):
    def test_accepts_valid_json_input(self):
        audit = load_audit_module()
        with tempfile.TemporaryDirectory() as temporary_dir:
            input_path = Path(temporary_dir) / "receipt.json"
            input_path.write_text(
                json.dumps({"status": "reimbursed"}), encoding="utf-8"
            )

            status, error = audit.validate_file(input_path)

        self.assertEqual(status, "passed")
        self.assertIsNone(error)


class DuplicateClassificationTests(unittest.TestCase):
    def test_separates_receipt_reuse_from_background_only_reuse(self):
        audit = load_audit_module()
        groups = [
            {"locations": [{"category": "receipt"}, {"category": "receipt"}]},
            {"locations": [{"category": "background"}, {"category": "background"}]},
        ]

        counts = audit.classify_duplicate_groups(groups)

        self.assertEqual(counts["with_receipt"], 1)
        self.assertEqual(counts["background_only"], 1)


if __name__ == "__main__":
    unittest.main()
