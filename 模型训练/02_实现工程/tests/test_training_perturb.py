import unittest

import torch

from audit_engine.models.heads import AuditHeads
from audit_engine.training.forward import ScalarCondition
from audit_engine.training.perturb import generate_detectable_errors


class PerturbationSelectionTests(unittest.TestCase):
    def setUp(self):
        # A controlled boundary makes a genuine gradient-based 3% perturbation
        # flip the relation. These weights are a unit fixture, not learned data.
        self.model = AuditHeads(base_dimension=16, dimension=8, width=4)
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.zero_()
                parameter.requires_grad_(False)
            self.model.relation.semantic[0].weight[0, 0] = 1
            self.model.relation.semantic[2].weight[0, 0] = 1
            self.model.relation.classifier.weight[0, 0] = 1
            self.model.relation.classifier.weight[1, 0] = -1
            self.model.relation.classifier.bias[2] = -4
            self.model.contribution.support.bias.fill_(8)
            self.model.contribution.conflict.bias.fill_(-8)
        self.condition = torch.tensor([[1.5, 100.0, 0, 0, 0, 0, 0, 0]])
        self.evidence = torch.zeros(1, 2, 8)
        self.policies = [ScalarCondition("LE", (("90", None), (None, "100")), (0, 1))]
        self.active = torch.ones(1, 2, dtype=torch.bool)
        self.labels = {
            "relation": torch.zeros(1, 2, dtype=torch.long),
            "relation_mask": self.active,
            "reference": torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]),
            "reference_mask": torch.ones(1, 2, 2, dtype=torch.bool),
            "state": torch.tensor([0]),
            "state_mask": torch.tensor([True]),
        }

    def generate(self, **kwargs):
        return generate_detectable_errors(
            self.model,
            self.condition,
            self.evidence,
            self.policies,
            self.active,
            self.labels,
            **kwargs,
        )

    def test_accepts_detected_error_with_unchanged_facts_and_bounded_vector(self):
        facts_before = self.evidence.clone()
        original = self.condition.clone()
        weights_before = {
            name: value.clone() for name, value in self.model.state_dict().items()
        }
        perturbed, accepted, report = self.generate()
        self.assertEqual(accepted.tolist(), [True])
        self.assertEqual(report["original_correct_count"], 1)
        self.assertEqual(report["accepted_detectable_count"], 1)
        self.assertLessEqual(report["max_relative_norm"], 0.030001)
        self.assertFalse(perturbed.requires_grad)
        self.assertTrue(torch.equal(original, self.condition))
        self.assertTrue(torch.equal(facts_before, self.evidence))
        self.assertTrue(
            all(
                torch.equal(weights_before[name], value)
                for name, value in self.model.state_dict().items()
            )
        )
        self.assertFalse(any(p.grad is not None for p in self.model.parameters()))

    def test_rejects_condition_without_approved_state(self):
        self.labels["state_mask"].fill_(False)
        _, accepted, report = self.generate()
        self.assertFalse(accepted.any())
        self.assertEqual(report["original_correct_count"], 0)

    def test_rejects_original_state_error_despite_correct_relations(self):
        # Relations still have the right argmax but support is below this limit.
        _, accepted, report = self.generate(
            thresholds={"reference": 0.6, "support": 0.95, "conflict": 0.7}
        )
        self.assertFalse(accepted.any())
        self.assertEqual(report["original_correct_count"], 0)

    def test_requires_condition_state_supervision(self):
        del self.labels["state"]
        with self.assertRaisesRegex(ValueError, "approved condition-state"):
            self.generate()

    def test_runtime_citation_verification_contract_is_enforced(self):
        with self.assertRaisesRegex(ValueError, "verification mask"):
            self.generate(reference_verified=torch.ones(1, 3, 2, dtype=torch.bool))
        _, accepted, report = self.generate(
            reference_verified=torch.ones(1, 2, 2, dtype=torch.bool)
        )
        self.assertTrue(accepted.any())
        self.assertEqual(report["accepted_detectable_count"], 1)


if __name__ == "__main__":
    unittest.main()
