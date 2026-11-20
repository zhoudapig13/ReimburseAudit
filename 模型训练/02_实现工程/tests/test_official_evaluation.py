import json
import runpy
import unittest
from pathlib import Path

score = runpy.run_path(
    str(
        Path(__file__).resolve().parents[1]
        / "code/scripts/evaluate_official_adapter.py"
    )
)["score"]


class OfficialEvaluationTests(unittest.TestCase):
    def test_amount_format_error_does_not_hide_correct_flag(self):
        target = {
            "reference_flag": 0,
            "reference_max_legal": "340",
            "problem_reports": [],
        }
        result = score('{"FLAG":0,"max_legal":340,"problem_reports":[]}', target)
        self.assertFalse(result["schema_valid"])
        self.assertTrue(result["flag_correct"])
        self.assertTrue(result["amount_correct"])
        result = score('{"FLAG":0,"max_legal":"NaN","problem_reports":[]}', target)
        self.assertFalse(result["schema_valid"])
        self.assertTrue(result["flag_correct"])
        self.assertFalse(result["amount_correct"])

    def test_decimal_equivalence_and_explanation_metric(self):
        result = score(
            '{"FLAG":0,"max_legal":"340.00","problem_reports":["paraphrase"]}',
            {
                "reference_flag": 0,
                "reference_max_legal": "340",
                "problem_reports": ["original"],
            },
        )
        self.assertTrue(result["flag_correct"])
        self.assertTrue(result["amount_correct"])
        self.assertFalse(result["explanation_exact_match"])

    def test_invalid_flag_and_nonfinite_amount_fail_schema(self):
        target = {
            "reference_flag": 0,
            "reference_max_legal": "340",
            "problem_reports": [],
        }
        for flag, amount in [(True, "340"), (0, "NaN"), (0, "Infinity"), (2, "340")]:
            result = score(
                json.dumps({"FLAG": flag, "max_legal": amount, "problem_reports": []}),
                target,
            )
            self.assertFalse(result["schema_valid"])
