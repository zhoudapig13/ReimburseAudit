import unittest

import torch

from audit_engine.training.losses import (
    masked_reference_loss,
    masked_relation_loss,
    masked_state_loss,
    state_probabilities,
)


class MaskedLossTests(unittest.TestCase):
    def test_class_weights_emphasize_conflict_without_training_masked_labels(self):
        logits = torch.zeros(1, 3, 3, requires_grad=True)
        targets = torch.tensor([[0, 1, 2]])
        mask = torch.tensor([[True, True, False]])
        loss = masked_relation_loss(logits, targets, mask, torch.tensor([1.0, 3.0, 0.25]))
        self.assertTrue(torch.allclose(loss, 2 * torch.log(torch.tensor(3.0))))
        loss.backward()
        self.assertTrue(torch.equal(logits.grad[0, 2], torch.zeros(3)))
        self.assertTrue(torch.allclose(logits.grad[0, 1].abs().sum(),
                                       3 * logits.grad[0, 0].abs().sum()))

    def test_invalid_class_weights_are_rejected(self):
        for weights in ([1.0, 0.0, 1.0], [1.0, float("nan"), 1.0], [1.0, 1.0]):
            with self.assertRaisesRegex(ValueError, "finite positive"):
                masked_relation_loss(torch.zeros(1, 1, 3), torch.zeros(1, 1, dtype=torch.long),
                                     torch.ones(1, 1, dtype=torch.bool), torch.tensor(weights))

    def test_relation_loss_excludes_unapproved_and_padding_labels(self):
        logits = torch.tensor(
            [[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]],
            requires_grad=True,
        )
        targets = torch.tensor([[0, 2, -100]])
        approved = torch.tensor([[True, False, True]])

        loss = masked_relation_loss(logits, targets, approved)

        self.assertIsNotNone(loss)
        expected = torch.nn.functional.cross_entropy(logits[:, 0, :], targets[:, 0])
        self.assertTrue(torch.allclose(loss, expected))

    def test_reference_loss_averages_each_condition_before_batch(self):
        logits = torch.zeros(2, 2, 2, requires_grad=True)
        targets = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [0.0, 0.0]]])
        approved = torch.tensor(
            [[[True, False], [False, False]], [[True, True], [True, True]]]
        )

        loss = masked_reference_loss(logits, targets, approved)

        self.assertIsNotNone(loss)
        self.assertTrue(torch.allclose(loss, torch.tensor(0.6931472)))

    def test_no_approved_labels_returns_none(self):
        relation = masked_relation_loss(
            torch.zeros(1, 1, 3),
            torch.full((1, 1), -100),
            torch.zeros(1, 1, dtype=torch.bool),
        )
        reference = masked_reference_loss(
            torch.zeros(1, 1, 2),
            torch.zeros(1, 1, 2),
            torch.zeros(1, 1, 2, dtype=torch.bool),
        )

        self.assertIsNone(relation)
        self.assertIsNone(reference)


class StateLossTests(unittest.TestCase):
    def test_state_mapping_is_normalized_in_required_order(self):
        strengths = torch.tensor([[0.8, 0.25]])

        probabilities = state_probabilities(strengths)

        expected = torch.tensor([[0.6, 0.05, 0.15, 0.2]])
        self.assertTrue(torch.allclose(probabilities, expected))
        self.assertTrue(torch.allclose(probabilities.sum(dim=-1), torch.ones(1)))

    def test_state_loss_ignores_unapproved_conditions(self):
        probabilities = torch.tensor(
            [[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1]], requires_grad=True
        )
        targets = torch.tensor([0, 1])
        approved = torch.tensor([True, False])

        loss = masked_state_loss(probabilities, targets, approved)

        self.assertIsNotNone(loss)
        self.assertTrue(torch.allclose(loss, -torch.log(torch.tensor(0.7))))


if __name__ == "__main__":
    unittest.main()
