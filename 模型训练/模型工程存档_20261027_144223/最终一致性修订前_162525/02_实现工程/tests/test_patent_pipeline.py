import copy
import hashlib
import importlib.util
from decimal import Decimal
from pathlib import Path

import pytest
import torch

from audit_engine.contracts import AuditCase, ConditionAnnotation
from audit_engine.inference.runner import AuditRunner, CaseFeatures
from audit_engine.models.heads import AuditHeads
from audit_engine.policy.amounts import project_amounts
from audit_engine.policy.money_ast import evaluate_money_ast
from audit_engine.training.forward import ScalarCondition
from audit_engine.training.patent import (
    TrainingCase,
    label_tensors,
    load_dataset,
    train,
)


def sample_case(name="example"):
    return AuditCase.model_validate(
        {
            "workflow_id": name,
            "input_version": "fixture-v1",
            "application": {
                "application": "Lodging",
                "claimed_amount": "90.00",
                "currency": "CNY",
                "attachment_ids": ["receipt", "policy"],
            },
            "evidence": [
                {
                    "evidence_id": "receipt",
                    "raw_ref": "receipt.txt",
                    "source_sha256": hashlib.sha256(b"Lodging 90.00").hexdigest(),
                    "type": "text",
                    "text": "Lodging 90.00",
                    "segments": [{"segment_id": "r1", "text": "Lodging 90.00"}],
                    "read_status": "obtained",
                },
                {
                    "evidence_id": "policy",
                    "raw_ref": "policy.txt",
                    "source_sha256": hashlib.sha256(b"Nightly cap 100").hexdigest(),
                    "type": "text",
                    "text": "Nightly cap 100",
                    "segments": [{"segment_id": "p1", "text": "Nightly cap 100"}],
                    "read_status": "obtained",
                },
            ],
            "conditions": [
                {
                    "condition_id": "cap",
                    "object_id": "stay",
                    "question": "Is nightly lodging within cap?",
                    "kind": "amount",
                    "roles": ["nightly"],
                    "policy_refs": [
                        {
                            "evidence_id": "policy",
                            "segment_id": "p1",
                            "quote": "cap 100",
                        }
                    ],
                    "operation": {
                        "name": "LE",
                        "left_role": "nightly",
                        "right_constant": "100",
                        "right_from_policy": True,
                        "value_type": "money",
                        "currency": "CNY",
                    },
                    "scope": "item",
                    "consequence": {
                        "on_failed": "cap_excess",
                        "on_gap": "request_material",
                    },
                    "compile_status": "verified",
                    "verification_method": "synthetic fixture definition",
                }
            ],
            "operands": [
                {
                    "condition_id": "cap",
                    "role": "nightly",
                    "evidence_id": "receipt",
                    "value": "90.00",
                    "value_type": "money",
                    "currency": "CNY",
                    "status": "obtained",
                    "refs": [
                        {"evidence_id": "receipt", "segment_id": "r1", "quote": "90.00"}
                    ],
                    "depends_on": ["receipt"],
                }
            ],
            "amount_units": [
                {
                    "object_id": "stay",
                    "scope": "item",
                    "claimed_amount": "90.00",
                    "currency": "CNY",
                    "limit_status": "known",
                    "limit": "100",
                    "material_supported_max": "90",
                }
            ],
            "policy_coverage": "verified",
        }
    )


def fixture_features(case, hidden=16):
    return CaseFeatures(
        case.digest(),
        ["receipt", "policy"],
        ["cap"],
        torch.randn(2, hidden),
        torch.randn(hidden),
        torch.randn(1, hidden),
        {
            "kind": "synthetic_engineering_fixture",
            "pooling": "last_non_padding_input_v1",
        },
    )


def fixture_annotation(case, split="train", family="fixture-train"):
    return ConditionAnnotation.model_validate(
        {
            "workflow_id": case.workflow_id,
            "condition_id": "cap",
            "split": split,
            "source_family_id": family,
            "case_sha256": case.digest(),
            "status": "approved",
            "method": "fixture_definition",
            "reviewer": "automated_fixture_test",
            "review_reason": "Known synthetic input; not benchmark supervision",
            "state": "satisfied",
            "relations": [
                {"evidence_id": "receipt", "relation": "support", "approved": True}
            ],
            "references": [
                {
                    "evidence_id": "receipt",
                    "polarity": "support",
                    "label": 1,
                    "approved": True,
                },
                {
                    "evidence_id": "receipt",
                    "polarity": "conflict",
                    "label": 0,
                    "approved": True,
                },
            ],
        }
    )


def test_source_quotes_and_answer_isolation():
    raw = sample_case().model_dump()
    raw["operands"][0]["refs"][0]["quote"] = "invented"
    with pytest.raises(ValueError, match="literal citation"):
        AuditCase.model_validate(raw)
    raw = sample_case().model_dump()
    raw["target"] = {"FLAG": 1}
    with pytest.raises(ValueError):
        AuditCase.model_validate(raw)
    raw = sample_case().model_dump()
    raw["operands"][0]["value"] = "999"
    with pytest.raises(ValueError, match="numeric read differs"):
        AuditCase.model_validate(raw)


def test_money_partition_maximum_and_shared_cap():
    unit = sample_case().amount_units[0]
    unit = unit.model_copy(update={"material_supported_max": "200", "limit": "150"})
    result = project_amounts([unit], {"stay": "accept"}, "90.00", "CNY")
    assert Decimal(result["A_max"]) == 150
    assert Decimal(result["A_ok"]) == 90
    missing = unit.model_copy(update={"limit_status": "unknown", "limit": None})
    result = project_amounts([missing], {"stay": "accept"}, "90.00", "CNY")
    assert result["A_max"] is None and Decimal(result["A_review"]) == 90
    with pytest.raises(ValueError, match="exactly partition"):
        project_amounts([unit], {"stay": "accept"}, "91", "CNY")


def test_masking_removes_cross_material_operands_and_policy_constant():
    p = ScalarCondition(
        "LE",
        (("90", None, None),),
        (0,),
        right_from_policy=True,
        right_constant="100",
        dependencies=(((0, 1), (), ()),),
        policy_sources=(2,),
    )
    assert p.rebuild(torch.tensor([True, True, True]), 8)[3][0].item() == 1
    assert not p.rebuild(torch.tensor([True, False, True]), 8)[0].any()
    assert not p.rebuild(torch.tensor([True, True, False]), 8)[3].any()


def test_runtime_really_calls_modules_and_logs_only_actual_correction():
    torch.manual_seed(18)
    case = sample_case()
    model = AuditHeads(16, 8, 4)
    with torch.no_grad():
        model.relation.classifier.weight.zero_()
        model.relation.classifier.bias.copy_(torch.tensor([-8.0, 8.0, -8.0]))
    output = AuditRunner(model).run(case, fixture_features(case))
    assert output["module_calls"]["relation"] >= 3
    assert output["module_calls"]["contribution"] >= 1
    assert output["module_calls"]["corrector"] >= 1
    assert (
        len([t for t in output["T"] if t["op"] == "local_correction"])
        == output["module_calls"]["corrector"]
    )
    assert output["module_calls"]["corrector"] <= 2
    assert sum(
        Decimal(output["B"][k]) for k in ("A_ok", "A_no", "A_review")
    ) == Decimal("90")
    bad = fixture_features(case)
    bad.case_sha256 = "different"
    with pytest.raises(ValueError, match="different input"):
        AuditRunner(model).run(case, bad)


def test_ablations_disable_the_selected_runtime_component():
    case = sample_case()
    model = AuditHeads(16, 8, 4)
    features = fixture_features(case)
    removed = AuditRunner(model, ablation="no_contribution", max_rounds=0).run(
        case, features
    )
    assert removed["module_calls"]["contribution"] == 0
    assert removed["module_calls"]["corrector"] == 0
    assert all(
        t["details"].get("source_mask_recomputations", 0) == 0 for t in removed["T"]
    )
    from audit_engine.inference.runner import tensors_for_case
    from audit_engine.training.forward import forward_condition

    h, e, policies, active = tensors_for_case(model, case, features)
    result = forward_condition(
        model, h, e, policies, active, ablation="no_policy_comparison"
    )
    assert not result["complete"].any()
    assert (result["comparison"].argmax(-1) == 2).all()


def test_local_correction_reuses_other_conditions_without_recomputing(monkeypatch):
    import audit_engine.inference.runner as runtime

    raw = sample_case().model_dump()
    missing = copy.deepcopy(raw["conditions"][0])
    missing.update(condition_id="missing", roles=["missing_amount"])
    missing["operation"]["left_role"] = "missing_amount"
    raw["conditions"].append(missing)
    case = AuditCase.model_validate(raw)
    features = fixture_features(sample_case())
    features.case_sha256 = case.digest()
    features.condition_ids = ["cap", "missing"]
    features.policies = features.policies.repeat(2, 1)
    model = AuditHeads(16, 8, 4)
    with torch.no_grad():
        model.relation.classifier.weight.zero_()
        model.relation.classifier.bias.copy_(torch.tensor([-8.0, 8.0, -8.0]))
    sizes = []
    original = runtime.forward_condition

    def counted(*args, **kwargs):
        sizes.append(len(args[3]))
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime, "forward_condition", counted)
    result = AuditRunner(model).run(case, features)
    assert sizes == [2, 1, 1]
    reuse = [event for event in result["T"] if event["op"] == "reuse_condition_result"]
    assert len(reuse) == 2 and all(event["condition_id"] == "missing" for event in reuse)
    assert all(event["details"]["source_mask_recomputations"] == 0 for event in reuse)
    gap = next(fact for fact in result["F"] if fact["condition_id"] == "missing")
    assert gap["gaps"] == [{"role": "missing_amount", "status": "not_read", "source_ids": []}]
    task = next(task for task in result["review"] if task.get("condition_id") == "missing")
    assert Decimal(task["review_amount"]) == 90 and task["required_material_types"]


def test_pending_labels_are_masked_and_family_leakage_is_rejected(tmp_path):
    c1, c2 = sample_case("a"), sample_case("b")
    a1 = fixture_annotation(c1)
    a2 = fixture_annotation(c2, "validation", "fixture-train")
    cases = tmp_path / "cases.jsonl"
    labels = tmp_path / "annotations.jsonl"
    cases.write_text("\n".join(c.model_dump_json() for c in (c1, c2)))
    labels.write_text("\n".join(a.model_dump_json() for a in (a1, a2)))
    with pytest.raises(ValueError, match="source-family leakage"):
        load_dataset(cases, labels, tmp_path)
    pending = a1.model_copy(
        update={"status": "pending", "state": None, "relations": [], "references": []}
    )
    targets = label_tensors(TrainingCase(c1, fixture_features(c1), [pending]), "cpu")
    assert not any(v.any() for k, v in targets.items() if k.endswith("_mask"))


def test_staged_training_changes_each_component_and_saves_all_weights(tmp_path):
    torch.manual_seed(3407)
    torch.set_num_threads(1)
    c1, c2 = sample_case("a"), sample_case("b")
    dataset = {
        "train": [TrainingCase(c1, fixture_features(c1), [fixture_annotation(c1)])],
        "validation": [
            TrainingCase(
                c2,
                fixture_features(c2),
                [fixture_annotation(c2, "validation", "fixture-val")],
            )
        ],
        "test": [],
    }
    model = AuditHeads(16, 8, 4)
    with torch.no_grad():
        model.relation.classifier.bias.copy_(torch.tensor([-3.0, 3.0, -3.0]))
    initial = copy.deepcopy(model.state_dict())
    config = {
        "project": {"seed": 3407},
        "model": {"hidden_size": 16, "dimension": 8, "width": 4, "role_slots": 8},
        "heads": {"learning_rate": 0.0003, "weight_decay": 0.01, "gradient_clip": 1.0},
        "correction": {"learning_rate": 0.0001, "rho_relative": 0.03},
        "thresholds": {"reference": 0.6, "support": 0.7, "conflict": 0.7},
        "fast_training": {"epochs": {"B1": 1, "B2": 1, "B3": 1, "C": 1}, "patience": 2},
    }
    with pytest.raises(ValueError, match="actual frozen-backbone"):
        train(model, dataset, config, tmp_path / "formal")
    status = train(
        model, dataset, config, tmp_path / "engineering", qualification="engineering"
    )
    assert status["three_components_optimized"]
    saved = torch.load(tmp_path / "engineering/audit_heads.pt", weights_only=True)
    for component in ("shared", "relation", "contribution", "corrector"):
        keys = [k for k in initial if k.startswith(component + ".")]
        assert any(not torch.equal(initial[k], saved["model"][k]) for k in keys), (
            component
        )


def test_money_ast_is_bounded_and_cannot_execute_code():
    expr = {
        "op": "CAP",
        "args": [
            {"op": "VALUE", "source": "rate", "quantity_source": "nights"},
            {"op": "VALUE", "source": "cap"},
        ],
    }
    assert evaluate_money_ast(expr, {"rate": "100", "nights": "2", "cap": "150"}) == 150
    with pytest.raises(ValueError):
        evaluate_money_ast({"op": "__import__", "args": []}, {})
    with pytest.raises(ValueError):
        evaluate_money_ast({"op": "VALUE", "source": "rate"}, {"rate": "NaN"})


def test_accuracy_denominator_includes_missing_predictions():
    path = (
        Path(__file__).resolve().parents[1]
        / "code/scripts/evaluate_patent_predictions.py"
    )
    spec = importlib.util.spec_from_file_location("patent_evaluation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    gold = [
        {"workflow_id": "a", "answer": {"FLAG": 1, "max_legal": "90"}},
        {"workflow_id": "b", "answer": {"FLAG": 0, "max_legal": "0"}},
    ]
    result = module.score(
        [{"workflow_id": "a", "answer": {"FLAG": 1, "max_legal": "90.00"}}], gold
    )
    assert result["correct"] == 1 and result["total"] == 2 and result["accuracy"] == 0.5


def test_reader_adapter_transfer_requires_same_frozen_backbone_contract():
    from audit_engine.training.patent import reader_adapter_transfer_allowed

    old = {"kind": "frozen_backbone", "revision": "pinned", "pooling": "last",
           "max_pixels": 1024, "adapter_sha256": "a" * 64,
           "adapter_config_sha256": "b" * 64}
    new = dict(old, adapter_sha256="c" * 64)
    assert reader_adapter_transfer_allowed([old], [new])
    for key, value in [("revision", "other"), ("pooling", "mean"),
                       ("max_pixels", 2048), ("kind", "synthetic"),
                       ("adapter_sha256", "incomplete")]:
        assert not reader_adapter_transfer_allowed([old], [dict(new, **{key: value})])
    assert not reader_adapter_transfer_allowed([], [new])
    assert not reader_adapter_transfer_allowed([old], [])
