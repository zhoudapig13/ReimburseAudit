import json
import unittest

from audit_engine.policy.candidates import (
    annotate_source_view,
    attach_source_checks,
    split_existing_quote,
    validate_policy_output,
)


class PolicyCandidateTests(unittest.TestCase):
    def proposal(self):
        return {
            "conditions": [
                {
                    "question": "Is the receipt within the per-night limit?",
                    "kind": "amount",
                    "roles": ["nightly_cost"],
                    "operation": "LE",
                    "policy_quote": "The limit is USD 100 per night.",
                    "policy_refs": [
                        {
                            "segment_id": "e001:t001",
                            "quote": "The limit is USD 100 per night.",
                        }
                    ],
                    "scope": "per night",
                    "amount_unit": "USD/night",
                    "exceptions": None,
                    "consequence": None,
                }
            ],
            "uncertainties": [],
        }

    def material(self):
        return {
            "evidence_id": "e001",
            "segments": [
                {"segment_id": "e001:t001", "text": "The limit is USD 100 per night."}
            ],
        }

    def test_literal_source_does_not_approve_semantics_or_training(self):
        proposal = validate_policy_output(json.dumps(self.proposal()))
        candidate = attach_source_checks(proposal, self.material())[0]
        self.assertTrue(candidate["policy_refs"][0]["existence_verified"])
        self.assertFalse(candidate["policy_refs"][0]["semantic_verified"])
        self.assertFalse(candidate["approved_training_label"])
        self.assertEqual(candidate["compile_status"], "pending")

    def test_teacher_approval_field_is_rejected(self):
        proposal = self.proposal()
        proposal["conditions"][0]["approved"] = True
        with self.assertRaisesRegex(ValueError, "unexpected condition"):
            validate_policy_output(json.dumps(proposal))

    def test_cross_segment_quote_is_kept_as_a_review_issue(self):
        material = self.material()
        material["segments"] = [
            {"segment_id": "e001:t001", "text": "The limit is USD"},
            {"segment_id": "e001:t002", "text": "100 per night."},
        ]
        candidate = attach_source_checks(self.proposal(), material)[0]
        self.assertFalse(candidate["policy_refs"][0]["existence_verified"])
        self.assertIn(
            "policy_quote_not_found_in_one_source_segment", candidate["issues"]
        )

    def test_no_policy_and_uncertainty_are_valid_without_invented_conditions(self):
        proposal = {
            "conditions": [],
            "uncertainties": ["No policy stated in this view."],
        }
        self.assertEqual(validate_policy_output(json.dumps(proposal)), proposal)

    def test_no_roles_or_unlisted_operation_is_rejected(self):
        for key, value in (("roles", []), ("operation", "GUESS")):
            proposal = self.proposal()
            proposal["conditions"][0][key] = value
            with self.assertRaises(ValueError):
                validate_policy_output(json.dumps(proposal))

    def test_marking_preserves_text_and_leaves_partial_segments_untagged(self):
        material = self.material()
        text = "Header\n" + material["segments"][0]["text"] + "\nFooter"
        marked = annotate_source_view(text, material)
        self.assertIn("[segment e001:t001]", marked)
        self.assertEqual(
            marked.replace("[segment e001:t001]", "").replace("[/segment]", ""), text
        )
        self.assertEqual(
            annotate_source_view("The limit is USD", material), "The limit is USD"
        )

    def test_two_separate_literal_references_are_valid_but_still_pending(self):
        material = self.material()
        material["segments"] = [
            {"segment_id": "e001:t001", "text": "The limit is USD"},
            {"segment_id": "e001:t002", "text": "100 per night."},
        ]
        proposal = self.proposal()
        proposal["conditions"][0]["policy_refs"] = [
            {"segment_id": s["segment_id"], "quote": s["text"]}
            for s in material["segments"]
        ]
        candidate = attach_source_checks(proposal, material)[0]
        self.assertTrue(all(r["existence_verified"] for r in candidate["policy_refs"]))
        self.assertFalse(candidate["approved_training_label"])

    def test_existing_full_quote_splits_into_independently_verified_fragments(self):
        material = self.material()
        material["text"] = "The limit is USD\n\n100 per night."
        material["segments"] = [
            {"segment_id": "e001:t001", "text": "The limit is USD"},
            {"segment_id": "e001:t002", "text": "100 per night."},
        ]
        candidate = attach_source_checks(self.proposal(), material)[0]
        self.assertEqual(len(candidate["policy_refs"]), 2)
        self.assertTrue(all(r["existence_verified"] for r in candidate["policy_refs"]))
        self.assertTrue(
            all(
                not r["original_reference"]["existence_verified"]
                for r in candidate["policy_refs"]
            )
        )
        self.assertFalse(candidate["approved_training_label"])
        self.assertEqual(
            split_existing_quote("The limit is USD 10 per night.", material), []
        )

    def test_does_not_construct_global_quote_from_disconnected_segments(self):
        material = self.material()
        material["text"] = "Different original text."
        self.assertEqual(
            split_existing_quote("The limit is USD 100 per night.", material), []
        )

    def test_repeated_global_quote_has_no_automatic_location_resolution(self):
        material = self.material()
        material["text"] = "The limit is USD 100 per night. " * 2
        self.assertEqual(
            split_existing_quote("The limit is USD 100 per night.", material), []
        )


if __name__ == "__main__":
    unittest.main()
