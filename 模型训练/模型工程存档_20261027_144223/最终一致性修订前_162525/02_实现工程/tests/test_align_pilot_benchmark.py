import importlib.util
import unittest
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "scripts"
    / "align_pilot_benchmark.py"
)


def load_alignment_module():
    spec = importlib.util.spec_from_file_location("align_pilot_benchmark", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load alignment module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bundle(claimed_amount="100"):
    return {
        "schema_version": "1.0",
        "input_version": "normalized_bundle_v1",
        "workflow_id": "1.1.1",
        "source_family_id": "sf_001",
        "input": {"claimed_amount": claimed_amount, "currency": "USD"},
        "evidence": [
            {
                "evidence_id": "e001",
                "text": "Hotel total 1,476.15 USD. Policy cap 220.00 USD.",
                "metadata": {},
            }
        ],
    }


def target(flag=0, max_legal="90", annotation_status="reference_only"):
    return {
        "schema_version": "1.0",
        "target_version": "benchmark_reference_v1",
        "input_version": "bundle_v1",
        "workflow_id": "1.1.1",
        "source_family_id": "sf_001",
        "split": "train",
        "target": {
            "reference_flag": flag,
            "reference_max_legal": max_legal,
            "problem_reports": [
                "The bill is 1476.15 USD and the policy cap is 220 USD."
            ],
            "logic_types": ["Amount-cap violation"],
            "scoring_points": [],
            "reference_score_sum": "1.00",
        },
        "annotation": {
            "status": annotation_status,
            "method": "benchmark_reference_conversion",
            "approved_for_component_training": False,
            "issues": [],
        },
    }


class AlignmentValidationTests(unittest.TestCase):
    def test_requires_one_target_for_every_pilot_bundle(self):
        alignment = load_alignment_module()

        with self.assertRaisesRegex(ValueError, "missing benchmark targets"):
            alignment.validate_pair_coverage([bundle()], [])

    def test_detects_flag_that_disagrees_with_amount_comparison(self):
        alignment = load_alignment_module()

        check = alignment.check_flag_consistency(bundle(), target(flag=1))

        self.assertFalse(check["consistent"])
        self.assertEqual(check["expected_flag"], 0)

    def test_numeric_traceability_ignores_comma_and_decimal_formatting(self):
        alignment = load_alignment_module()

        traceability = alignment.check_problem_numeric_traceability(bundle(), target())

        self.assertEqual(traceability["status"], "full")
        self.assertEqual(traceability["unmatched"], [])
        self.assertEqual(traceability["matched"], ["1476.15", "220"])


class BenchmarkPairTests(unittest.TestCase):
    def test_pair_never_claims_component_label_approval(self):
        alignment = load_alignment_module()

        pair = alignment.build_benchmark_pair(bundle(), target())

        self.assertTrue(pair["annotation"]["use_for_end_to_end_supervision"])
        self.assertFalse(pair["annotation"]["approved_for_component_training"])
        self.assertEqual(pair["annotation"]["label_source"], "benchmark_reference")

    def test_known_reference_issue_is_held_out(self):
        alignment = load_alignment_module()
        held_target = target(annotation_status="needs_review")
        held_target["annotation"]["issues"] = ["known inconsistency"]

        record = alignment.build_alignment_record(bundle(), held_target)

        self.assertEqual(record["training_status"], "hold")
        self.assertIn("benchmark_reference_needs_review", record["hold_reasons"])


if __name__ == "__main__":
    unittest.main()
