import copy

import pytest
import torch
from torch import nn

from audit_engine.models.chunked_mlp import enable_decoder_mlp_chunks
from audit_engine.training.glm_output import final_answer_ids


@pytest.mark.parametrize(
    "ids, expected",
    [
        ([151345, 5, 151346, 151347, 11, 12, 151348], [11, 12]),
        ([151345, 5, 151346, 11, 12], [11, 12]),
        ([151345, 5, 11, 12], []),
        ([151347, 11, 12], [11, 12]),
        ([11, 12], [11, 12]),
    ],
)
def test_native_answer_channel(ids, expected):
    assert final_answer_ids(ids) == expected


class FusedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Linear(4, 14)
        self.down_proj = nn.Linear(7, 4)
        self.activation_fn = torch.nn.functional.silu

    def forward(self, inputs):
        gate, up = self.gate_up_proj(inputs).chunk(2, dim=-1)
        return self.down_proj(self.activation_fn(gate) * up)


def test_glm_fused_mlp_chunking_preserves_values_and_gradients():
    model = nn.Module()
    model.language_model = nn.Module()
    layer = nn.Module()
    layer.mlp = FusedMLP().double()
    model.language_model.layers = nn.ModuleList([layer])
    baseline = copy.deepcopy(layer.mlp)
    assert enable_decoder_mlp_chunks(model, 3) == 1
    x = torch.randn(1, 13, 4, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    expected, actual = baseline(x), layer.mlp(y)
    assert torch.allclose(expected, actual, atol=1e-12)
    expected.sum().backward()
    actual.sum().backward()
    assert torch.allclose(x.grad, y.grad, atol=1e-12)
