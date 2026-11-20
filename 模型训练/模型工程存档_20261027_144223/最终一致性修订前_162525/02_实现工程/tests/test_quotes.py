import unittest

from audit_engine.readers.quotes import verify_text_quote


class LiteralQuoteTests(unittest.TestCase):
    def evidence(self, text):
        return [
            {
                "evidence_id": "e001",
                "segments": [{"segment_id": "e001:t001", "text": text}],
            }
        ]

    def reference(self, quote, **extra):
        return {
            "evidence_id": "e001",
            "segment_id": "e001:t001",
            "quote": quote,
            **extra,
        }

    def test_whitespace_is_normalized_and_teacher_approval_is_not_trusted(self):
        result = verify_text_quote(
            self.reference("Travel dates: 11 October 2026", semantic_verified=True),
            self.evidence("Travel dates:\n11  October 2026"),
        )
        self.assertTrue(result.existence_verified)
        self.assertFalse(result.semantic_verified)
        self.assertEqual(result.status, "semantic_review_pending")

    def test_money_date_and_name_changes_do_not_get_fuzzy_matched(self):
        for source, quote in [
            ("Total amount: 150.00", "Total amount: 15"),
            ("Date: 2026-10-13", "Date: 2026-10"),
            ("Employee: Johnny", "Employee: John"),
            ("Amount: -100", "100"),
        ]:
            result = verify_text_quote(self.reference(quote), self.evidence(source))
            self.assertFalse(result.existence_verified, (source, quote))

    def test_caller_semantic_approval_requires_a_matching_source(self):
        result = verify_text_quote(
            self.reference("Total: 150.00"),
            self.evidence("Total: 150.00"),
            semantic_verified=True,
        )
        self.assertTrue(result.semantic_verified)
        mismatch = verify_text_quote(
            self.reference("Total: 100.00"),
            self.evidence("Total: 150.00"),
            semantic_verified=True,
        )
        self.assertFalse(mismatch.semantic_verified)

    def test_a_quote_cannot_be_joined_across_two_segments(self):
        source = self.evidence("Travel dates:")
        source[0]["segments"].append(
            {"segment_id": "e001:t002", "text": "11 October 2026"}
        )
        self.assertFalse(
            verify_text_quote(
                self.reference("Travel dates: 11 October 2026"), source
            ).existence_verified
        )

    def test_missing_source_and_nontext_observation_stay_unverified(self):
        self.assertFalse(
            verify_text_quote(self.reference("text"), []).existence_verified
        )
        self.assertFalse(
            verify_text_quote(
                self.reference("text", kind="visual"), self.evidence("text")
            ).existence_verified
        )


if __name__ == "__main__":
    unittest.main()
