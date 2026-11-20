import unittest

import torch

from audit_engine.models.heads import (
    AuditHeads,
    aggregate_strengths,
    parameter_count,
)


class AuditHeadShapeTests(unittest.TestCase):
    def test_parameter_counts_match_the_implementation_guide(self):
        for dimension, expected in (
            (256, 4_968_457),
            (512, 13_541_385),
            (1024, 41_500_681),
        ):
            model = AuditHeads(dimension=dimension, width=dimension // 2)
            self.assertEqual(parameter_count(model), expected)

    def test_forward_shapes_and_relation_probabilities(self):
        torch.manual_seed(3407)
        model = AuditHeads()
        batch_size, evidence_count, role_count = 2, 3, 8
        raw_evidence = torch.randn(batch_size, evidence_count, 4096)
        raw_application = torch.randn(batch_size, 4096)
        raw_policy = torch.randn(batch_size, 4096)
        evidence, condition = model.shared(raw_evidence, raw_application, raw_policy)
        active = torch.tensor([[True, True, False], [True, True, True]])
        role_map = torch.zeros(batch_size, role_count, evidence_count, dtype=torch.bool)
        role_map[:, 0, 0] = True
        role_map[:, 1, 1] = True
        required = torch.zeros(batch_size, role_count, dtype=torch.bool)
        required[:, :2] = True
        role_context = model.relation.context(evidence, role_map, active)
        comparison = torch.zeros(batch_size, evidence_count, 3)
        comparison[..., 0] = 1
        complete = torch.ones(batch_size, evidence_count, 1)

        fused, logits, probabilities = model.relation(
            condition,
            evidence,
            role_context,
            comparison,
            complete,
        )
        strengths = aggregate_strengths(probabilities, role_map, active, required)
        reference_logits = model.contribution(
            fused, probabilities, torch.zeros(batch_size, evidence_count, 2)
        )

        self.assertEqual(tuple(fused.shape), (2, 3, 512))
        self.assertEqual(tuple(logits.shape), (2, 3, 3))
        self.assertEqual(tuple(strengths.shape), (2, 2))
        self.assertEqual(tuple(reference_logits.shape), (2, 3, 2))
        self.assertTrue(torch.allclose(probabilities.sum(dim=-1), torch.ones(2, 3)))

    def test_condition_without_required_role_is_rejected(self):
        probabilities = torch.full((1, 1, 3), 1 / 3)
        role_map = torch.zeros(1, 8, 1, dtype=torch.bool)
        active = torch.ones(1, 1, dtype=torch.bool)
        required = torch.zeros(1, 8, dtype=torch.bool)

        with self.assertRaisesRegex(ValueError, "required role"):
            aggregate_strengths(probabilities, role_map, active, required)


class CorrectorGradientTests(unittest.TestCase):
    def test_ineligible_rows_bypass_and_gradient_reaches_only_corrector(self):
        torch.manual_seed(3407)
        model = AuditHeads()
        batch_size, evidence_count, dimension, role_count = 2, 3, 512, 8
        raw_evidence = torch.randn(batch_size, evidence_count, 4096)
        raw_application = torch.randn(batch_size, 4096)
        raw_policy = torch.randn(batch_size, 4096)
        evidence, condition = model.shared(raw_evidence, raw_application, raw_policy)
        active = torch.ones(batch_size, evidence_count, dtype=torch.bool)
        role_map = torch.zeros(batch_size, role_count, evidence_count, dtype=torch.bool)
        role_map[:, 0, 0] = True
        role_context = model.relation.context(evidence, role_map, active)
        comparison = torch.zeros(batch_size, evidence_count, 3)
        comparison[..., 0] = 1
        complete = torch.ones(batch_size, evidence_count, 1)

        for parameter in model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        for parameter in model.corrector.parameters():
            parameter.requires_grad_(True)

        evidence = evidence.detach()
        condition = condition.detach()
        role_context = role_context.detach()
        errors = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        fact_encoding = torch.tensor([[0.8, 0.1, 0.0, 1.0, 0.0, 0.0, 0.0]]).repeat(
            batch_size, 1
        )
        error_slots = torch.randn(batch_size, role_count, dimension + 4)
        eligible = torch.tensor([True, False])

        corrected = model.corrector(
            condition,
            role_context,
            errors,
            fact_encoding,
            error_slots,
            eligible,
        )
        self.assertTrue(torch.equal(corrected[1], condition[1]))
        fused, logits, probabilities = model.relation(
            corrected,
            evidence,
            role_context,
            comparison,
            complete,
        )
        reference_logits = model.contribution(
            fused, probabilities, torch.zeros(batch_size, evidence_count, 2)
        )
        loss = logits[0, :, 0].square().mean() + reference_logits[0].square().mean()
        loss.backward()

        self.assertTrue(
            any(
                parameter.grad is not None and parameter.grad.abs().sum() > 0
                for parameter in model.corrector.parameters()
            )
        )
        for frozen_module in (model.shared, model.relation, model.contribution):
            self.assertTrue(
                all(parameter.grad is None for parameter in frozen_module.parameters())
            )


if __name__ == "__main__":
    unittest.main()
