import json
import unittest

import torch

from audit_engine.training.official_sft import official_answer, response_labels


class OfficialSupervisionTests(unittest.TestCase):
    def test_official_inconsistency_preserved_and_private_metadata_excluded(self):
        target = {
            "split": "train",
            "target": {
                "reference_flag": 0,
                "reference_max_legal": "340",
                "problem_reports": ["220 USD/night for two nights"],
            },
            "annotation": {"answer_hint": "do not include"},
        }
        answer = json.loads(official_answer(target))
        self.assertEqual(
            answer,
            {
                "FLAG": 0,
                "max_legal": "340",
                "problem_reports": ["220 USD/night for two nights"],
            },
        )

    def test_heldout_answer_rejected(self):
        with self.assertRaises(ValueError):
            official_answer({"split": "test"})

    def test_only_response_tokens_supervised(self):
        full = torch.tensor([[1, 2, 3, 4, 5]])
        self.assertEqual(
            response_labels(full[:, :3], full).tolist(), [[-100, -100, -100, 4, 5]]
        )
        with self.assertRaises(ValueError):
            response_labels(torch.tensor([[1, 9]]), full)
