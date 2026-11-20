import unittest

import torch

from audit_engine.models.heads import AuditHeads
from audit_engine.models.pooling import pool_last_input
from audit_engine.training.forward import (
    ScalarCondition,
    correction_inputs,
    forward_condition,
)
from audit_engine.training.perturb import bounded_perturbation


class MaskingTests(unittest.TestCase):
    def test_open_text_reference_error_is_eligible_without_a_deterministic_comparison(
        self,
    ):
        model = AuditHeads(base_dimension=16, dimension=8, width=4)
        with torch.no_grad():
            model.relation.classifier.weight.zero_()
            model.relation.classifier.bias.copy_(torch.tensor([8.0, -8.0, -8.0]))
            model.contribution.support.weight.zero_()
            model.contribution.support.bias.fill_(-8.0)
            model.contribution.conflict.weight.zero_()
            model.contribution.conflict.bias.fill_(-8.0)
        evidence, condition = model.shared(
            torch.randn(1, 1, 16), torch.randn(1, 16), torch.randn(1, 16)
        )
        policy = ScalarCondition(
            None,
            (("A project meeting with the client",),),
            (0,),
            comparison_required=False,
        )
        active = torch.tensor([[True]])
        output = forward_condition(model, condition, evidence, [policy], active)
        self.assertFalse(output["complete"].any())
        errors, _, _, eligible = correction_inputs(evidence, output, active, [policy])
        self.assertEqual(float(errors[0, 0]), 1.0)
        self.assertTrue(eligible[0])
        with torch.no_grad():
            model.contribution.support.bias.fill_(8.0)
        covered = forward_condition(model, condition, evidence, [policy], active)
        covered_errors, _, _, covered_eligible = correction_inputs(
            evidence, covered, active, [policy]
        )
        self.assertFalse(covered_errors.any())
        self.assertFalse(covered_eligible.any())
        missing = forward_condition(model, condition, evidence, [policy], ~active)
        self.assertFalse(correction_inputs(evidence, missing, ~active, [policy])[3][0])

    def test_unverified_quotes_do_not_cover_roles_and_read_failures_bypass_correction(
        self,
    ):
        model = AuditHeads(base_dimension=16, dimension=8, width=4)
        with torch.no_grad():
            model.relation.classifier.weight.zero_()
            model.relation.classifier.bias.copy_(torch.tensor([8.0, -8.0, -8.0]))
            model.contribution.support.weight.zero_()
            model.contribution.support.bias.fill_(8.0)
            model.contribution.conflict.weight.zero_()
            model.contribution.conflict.bias.fill_(-8.0)
        evidence, condition = model.shared(
            torch.randn(1, 2, 16), torch.randn(1, 16), torch.randn(1, 16)
        )
        active = torch.ones(1, 2, dtype=torch.bool)
        policies = [ScalarCondition("LE", (("90", None), (None, "100")), (0, 1))]
        output = forward_condition(model, condition, evidence, policies, active)
        verified = torch.ones_like(output["references"], dtype=torch.bool)
        errors, _, _, eligible = correction_inputs(
            evidence, output, active, policies, reference_verified=verified
        )
        self.assertFalse(errors.any())
        self.assertFalse(eligible.any())
        errors, _, _, eligible = correction_inputs(
            evidence, output, active, policies, reference_verified=~verified
        )
        self.assertEqual(float(errors[0, 0]), 1.0)
        self.assertTrue(eligible[0])
        _, _, _, eligible = correction_inputs(
            evidence,
            output,
            active,
            policies,
            reference_verified=~verified,
            upstream_read_failure=torch.tensor([True]),
        )
        self.assertFalse(eligible[0])

    def test_internal_perturbation_respects_relative_radius(self):
        original = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
        perturbed = bounded_perturbation(
            original, torch.tensor([[1.0, 2.0], [1.0, 0.0]])
        )
        self.assertLessEqual(float((perturbed[0] - original[0]).norm()), 0.150001)
        self.assertTrue(torch.equal(original[1], perturbed[1]))
        self.assertFalse(perturbed.requires_grad)

    def test_masking_invalidates_derived_comparison(self):
        policy = ScalarCondition("LE", (("90", None), (None, "100")), (0, 1))
        active = torch.tensor([True, True])
        roles, _, comparison, complete = policy.rebuild(active, 8)
        self.assertEqual(comparison.argmax(-1).tolist(), [0, 0])
        self.assertTrue(complete.bool().all())
        active[0] = False
        roles, _, comparison, complete = policy.rebuild(active, 8)
        self.assertFalse(roles[:, 0].any())
        self.assertFalse(complete.any())
        self.assertEqual(comparison.argmax(-1).tolist(), [2, 2])

    def test_equivalent_source_preserves_comparison_and_ambiguity_blocks_it(self):
        policy = ScalarCondition(
            "LE", (("90", "90", None), (None, None, "100")), (0, 1)
        )
        self.assertTrue(
            policy.rebuild(torch.tensor([False, True, True]), 8)[3][1:].bool().all()
        )
        ambiguous = ScalarCondition(
            "LE", (("90", "120", None), (None, None, "100")), (0, 1)
        )
        self.assertFalse(ambiguous.rebuild(torch.ones(3, dtype=torch.bool), 8)[3].any())

    def test_missing_material_bypasses_correction_and_masked_gradient_survives(self):
        torch.manual_seed(4)
        model = AuditHeads(base_dimension=16, dimension=8, width=4)
        raw = torch.randn(2, 2, 16)
        evidence, condition = model.shared(raw, torch.randn(2, 16), torch.randn(2, 16))
        policies = [ScalarCondition("LE", (("90", None), (None, "100")), (0, 1))] * 2
        active = torch.tensor([[True, True], [True, False]])
        output = forward_condition(model, condition, evidence, policies, active)
        errors, fact, slots, eligible = correction_inputs(
            evidence, output, active, policies
        )
        self.assertFalse(eligible[1])
        corrected = model.corrector(
            condition, output["context"], errors, fact, slots, eligible
        )
        self.assertTrue(torch.equal(corrected[1], condition[1]))
        # Difference through both full and masked relation computations is live.
        output["delta"].square().sum().backward()
        self.assertGreater(model.shared.evidence.weight.grad.abs().sum().item(), 0)

    def test_pooling_handles_both_padding_directions(self):
        hidden = torch.arange(12).reshape(2, 3, 2)
        mask = torch.tensor([[0, 1, 1], [1, 1, 0]])
        self.assertTrue(
            torch.equal(pool_last_input(hidden, mask), hidden[[0, 1], [2, 1]])
        )
        with self.assertRaises(ValueError):
            pool_last_input(hidden, torch.zeros_like(mask))


if __name__ == "__main__":
    unittest.main()
