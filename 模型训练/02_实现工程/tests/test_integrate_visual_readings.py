import importlib.util
import unittest
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "scripts"
    / "integrate_visual_readings.py"
)


def load_integration_module():
    spec = importlib.util.spec_from_file_location(
        "integrate_visual_readings", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load integration module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def queue_row():
    return {
        "workflow_id": "1.1.1",
        "evidence_id": "e001",
        "raw_ref": "workflow/1.1.1/receipt.png",
        "type": "image",
        "actions": ["multimodal_reading"],
    }


def reading(transcription="Total USD 12.30", observations=None):
    return {
        "reading_version": "visual_reading_v1",
        "workflow_id": "1.1.1",
        "evidence_id": "e001",
        "raw_ref": "workflow/1.1.1/receipt.png",
        "document_kind": "invoice",
        "reading_status": "clear",
        "transcription": transcription,
        "facts": [
            {
                "field": "total_amount",
                "value": "12.30",
                "normalized": "12.30",
                "unit": None,
                "currency": "USD",
                "source_text": "Total USD 12.30",
                "confidence": "high",
                "notes": None,
            }
        ],
        "visual_observations": observations or [],
        "uncertainties": [],
        "review_status": "pending_human_review",
    }


class ReadingValidationTests(unittest.TestCase):
    def test_queue_coverage_requires_one_matching_reading(self):
        integration = load_integration_module()

        with self.assertRaisesRegex(ValueError, "missing visual readings"):
            integration.validate_queue_coverage([queue_row()], [])

    def test_amount_values_must_remain_strings(self):
        integration = load_integration_module()
        invalid = reading()
        invalid["facts"][0]["value"] = 12.3

        with self.assertRaisesRegex(ValueError, "fact value must be a string"):
            integration.validate_reading(invalid)


class ReadingIntegrationTests(unittest.TestCase):
    def test_textual_reading_adds_segment_and_clears_multimodal_warning(self):
        integration = load_integration_module()
        bundles = [
            {
                "workflow_id": "1.1.1",
                "evidence": [
                    {
                        "evidence_id": "e001",
                        "raw_ref": "workflow/1.1.1/receipt.png",
                        "type": "image",
                        "text": None,
                        "segments": [],
                        "views": [{"view_id": "e001:v001"}],
                        "parse_status": "visual_only",
                        "warnings": ["multimodal_reading_required"],
                        "metadata": {},
                    }
                ],
            }
        ]

        merged = integration.apply_visual_readings(bundles, [reading()])
        evidence = merged[0]["evidence"][0]

        self.assertEqual(evidence["parse_status"], "multimodal_read_pending_review")
        self.assertEqual(evidence["text"], "Total USD 12.30")
        self.assertEqual(evidence["segments"][0]["kind"], "multimodal_text")
        self.assertNotIn("multimodal_reading_required", evidence["warnings"])
        self.assertIn("human_review_required", evidence["warnings"])

    def test_photo_observation_does_not_become_invented_transcription(self):
        integration = load_integration_module()
        photo_reading = reading(
            transcription=None,
            observations=["The image shows a marked project site office."],
        )
        photo_reading["document_kind"] = "site_photo"
        photo_reading["facts"] = []
        bundles = [
            {
                "workflow_id": "1.1.1",
                "evidence": [
                    {
                        "evidence_id": "e001",
                        "raw_ref": "workflow/1.1.1/receipt.png",
                        "type": "image",
                        "text": None,
                        "segments": [],
                        "views": [{"view_id": "e001:v001"}],
                        "parse_status": "visual_only",
                        "warnings": ["multimodal_reading_required"],
                        "metadata": {},
                    }
                ],
            }
        ]

        merged = integration.apply_visual_readings(bundles, [photo_reading])
        evidence = merged[0]["evidence"][0]

        self.assertIsNone(evidence["text"])
        self.assertEqual(evidence["segments"][0]["kind"], "visual_observation")
        self.assertEqual(evidence["segments"][0]["text"], "")


if __name__ == "__main__":
    unittest.main()
