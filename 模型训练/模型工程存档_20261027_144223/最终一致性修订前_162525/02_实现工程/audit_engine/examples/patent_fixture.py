"""Synthetic fixtures for engineering verification, separate from benchmark data."""

import hashlib

import torch

from audit_engine.contracts import AuditCase, ConditionAnnotation
from audit_engine.inference.runner import CaseFeatures


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
