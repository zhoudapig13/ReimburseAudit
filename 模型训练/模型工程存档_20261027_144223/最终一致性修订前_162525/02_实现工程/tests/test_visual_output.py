import json
import unittest

from audit_engine.readers.visual import (
    recover_completed_visual_fields,
    validate_visual_output,
)


class VisualOutputTests(unittest.TestCase):
    def reading(self):
        return {
            "document_kind": "receipt",
            "reading_status": "partial",
            "transcription": None,
            "facts": [
                {
                    "field": "total_amount",
                    "value": "$ 100.00",
                    "currency": None,
                    "source_text": "TOTAL $ 100.00",
                    "confidence": "medium",
                }
            ],
            "visual_observations": [],
            "uncertainties": ["Dollar sign does not identify an ISO currency."],
        }

    def test_keeps_raw_amount_and_unknown_currency(self):
        raw = self.reading()
        self.assertEqual(validate_visual_output(json.dumps(raw)), raw)

    def test_rejects_teacher_approval_field(self):
        raw = self.reading()
        raw["approved"] = True
        with self.assertRaises(ValueError):
            validate_visual_output(json.dumps(raw))

    def test_rejects_numeric_value_instead_of_coercing_it(self):
        raw = self.reading()
        raw["facts"][0]["value"] = 100.0
        with self.assertRaises(ValueError):
            validate_visual_output(json.dumps(raw))

    def test_does_not_invent_facts_for_unreadable_image(self):
        raw = self.reading()
        raw.update(reading_status="needs_review", facts=[])
        self.assertEqual(validate_visual_output(json.dumps(raw))["facts"], [])

    def test_null_value_is_preserved_without_a_guessed_replacement(self):
        raw = self.reading()
        raw["facts"][0]["value"] = None
        self.assertIsNone(validate_visual_output(json.dumps(raw))["facts"][0]["value"])

    def test_visible_blank_string_is_preserved_without_turning_it_into_zero(self):
        raw = self.reading()
        raw["facts"][0]["value"] = ""
        self.assertEqual(
            validate_visual_output(json.dumps(raw))["facts"][0]["value"], ""
        )

    def test_unknown_document_kind_remains_null(self):
        raw = self.reading()
        raw["document_kind"] = None
        self.assertIsNone(validate_visual_output(json.dumps(raw))["document_kind"])

    def truncated_after_facts(self):
        reading = self.reading()
        prefix = {
            k: reading[k]
            for k in ("document_kind", "reading_status", "transcription", "facts")
        }
        return json.dumps(prefix)[:-1] + ', "visual_observations":'

    def test_recovers_complete_facts_as_partial_without_creating_source_values(self):
        recovery = recover_completed_visual_fields(self.truncated_after_facts())
        self.assertEqual(recovery["candidate"]["facts"], self.reading()["facts"])
        self.assertEqual(recovery["candidate"]["reading_status"], "partial")
        self.assertFalse(recovery["source_output_complete"])
        self.assertEqual(
            recovery["unavailable_fields"], ["uncertainties", "visual_observations"]
        )

    def test_does_not_recover_an_incomplete_fact_array(self):
        self.assertIsNone(
            recover_completed_visual_fields(
                '{"document_kind":"invoice","reading_status":"clear","transcription":null,"facts":[{"field":"amount","value":"100'
            )
        )

    def test_unknown_trailing_field_is_not_silently_dropped(self):
        raw = self.truncated_after_facts().replace(
            '"visual_observations"', '"annotation_issue"'
        )
        self.assertIsNone(recover_completed_visual_fields(raw))

    def test_partial_recovery_does_not_coerce_invalid_numeric_facts(self):
        raw = self.truncated_after_facts().replace('"$ 100.00"', "100.00")
        self.assertIsNone(recover_completed_visual_fields(raw))

    def test_partial_recovery_does_not_hide_invalid_original_status(self):
        raw = self.truncated_after_facts().replace('"partial"', '"finished"')
        self.assertIsNone(recover_completed_visual_fields(raw))


if __name__ == "__main__":
    unittest.main()
