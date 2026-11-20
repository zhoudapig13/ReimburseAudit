import copy
import unittest

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from audit_engine.models.chunked_mlp import chunked_forward


class PointwiseMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(4, 11)
        self.up = nn.Linear(4, 11)
        self.down = nn.Linear(11, 4)

    def forward(self, inputs):
        return self.down(torch.nn.functional.silu(self.gate(inputs)) * self.up(inputs))


class ChunkedMLPTests(unittest.TestCase):
    def test_outputs_and_gradients_match_with_nested_checkpoint(self):
        torch.manual_seed(19)
        baseline = PointwiseMLP().double()
        chunked = copy.deepcopy(baseline)
        x = torch.randn(2, 13, 4, dtype=torch.float64, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        expected = baseline(x)
        actual = checkpoint(
            lambda v: chunked_forward(chunked, v, 3), y, use_reentrant=False
        )
        self.assertTrue(torch.allclose(expected, actual, atol=1e-12, rtol=1e-12))
        expected.square().sum().backward()
        actual.square().sum().backward()
        self.assertTrue(torch.allclose(x.grad, y.grad, atol=1e-11, rtol=1e-11))
        for first, second in zip(
            baseline.parameters(), chunked.parameters(), strict=True
        ):
            self.assertTrue(
                torch.allclose(first.grad, second.grad, atol=1e-11, rtol=1e-11)
            )

    def test_frozen_parameters_still_propagate_input_gradient(self):
        mlp = PointwiseMLP().requires_grad_(False)
        x = torch.randn(1, 7, 4, requires_grad=True)
        chunked_forward(mlp, x, 2).sum().backward()
        self.assertGreater(float(x.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in mlp.parameters()))
