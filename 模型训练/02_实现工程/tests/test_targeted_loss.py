import json

import pytest
import torch

from audit_engine.training.targeted_loss import decision_amount_weights


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        return {
            "input_ids": [ord(c) for c in text],
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }


@pytest.mark.parametrize("amount", ["0", "29.55", "12800.00"])
@pytest.mark.parametrize(
    "explanation", ["", "evidence and calculation " * 100], ids=["short", "long"]
)
def test_field_mass_is_independent_of_amount_length_and_explanation(
    amount, explanation
):
    answer = json.dumps(
        {"FLAG": 0, "max_legal": amount, "problem_reports": [explanation]},
        separators=(",", ":"),
    )
    ids = torch.tensor([ord(c) for c in answer] + [999])
    weights = decision_amount_weights(CharacterTokenizer(), answer, ids)
    flag_at = answer.index('"FLAG":') + len('"FLAG":')
    amount_at = answer.index('"max_legal":"') + len('"max_legal":"')
    assert weights[flag_at].item() == pytest.approx(0.25)
    assert weights[amount_at : amount_at + len(amount)].sum().item() == pytest.approx(
        0.25
    )
    assert weights.sum().item() == pytest.approx(1)
    assert torch.all(weights > 0)
    losses = torch.ones_like(weights, requires_grad=True)
    (losses * weights).sum().backward()
    assert torch.allclose(losses.grad, weights)


def test_changed_response_prefix_is_rejected():
    answer = '{"FLAG":1,"max_legal":"1","problem_reports":[]}'
    with pytest.raises(ValueError, match="offsets"):
        decision_amount_weights(CharacterTokenizer(), answer, torch.tensor([0]))
