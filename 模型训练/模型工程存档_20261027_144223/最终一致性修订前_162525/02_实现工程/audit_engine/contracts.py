"""Versioned runtime and supervision contracts for the patent pipeline.

Runtime cases intentionally have no target/answer/annotation fields. Supervision
is stored separately and is bound to an exact case digest and source-group split.
"""

from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Money = Annotated[str, StringConstraints(pattern=r"^(0|[1-9][0-9]*)(\.[0-9]+)?$")]
Identifier = Annotated[str, StringConstraints(min_length=1, max_length=256)]
State = Literal["satisfied", "failed", "insufficient", "contradictory"]
Action = Literal[
    "reject_object",
    "reject_application",
    "cap_excess",
    "request_material",
    "manual_review",
    "no_action",
]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Citation(Contract):
    evidence_id: Identifier
    segment_id: str | None = None
    quote: str = ""
    note: str = ""
    kind: Literal["text", "visual"] = "text"

    @model_validator(mode="after")
    def check_location(self):
        if self.kind == "text" and (not self.segment_id or not self.quote.strip()):
            raise ValueError("text citation requires segment and literal quote")
        if self.kind == "visual" and (self.segment_id or self.quote or not self.note):
            raise ValueError("visual citation requires a whole-file observation note")
        return self


class Segment(Contract):
    segment_id: Identifier
    text: str


class Evidence(Contract):
    evidence_id: Identifier
    raw_ref: Identifier
    source_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    type: Literal["image", "pdf", "docx", "xlsx", "json", "text"]
    text: str | None = None
    segments: list[Segment] = Field(default_factory=list)
    read_status: Literal["obtained", "partial", "not_obtained"] = "not_obtained"

    @model_validator(mode="after")
    def unique_segments(self):
        ids = [s.segment_id for s in self.segments]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate segment id")
        return self


class Application(Contract):
    application: str
    claimed_amount: Money
    currency: Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")]
    attachment_ids: list[Identifier]
    receipt_owners: list[dict[str, str | list[str]]] = Field(default_factory=list)


class Consequence(Contract):
    on_failed: Action
    on_gap: Action
    on_contradictory: Action = "manual_review"


class Operation(Contract):
    name: Literal["EQ", "LE", "GE", "WITHIN", "IN_SET", "NOT_IN_SET", "AUTHORIZED", "CLOSED_ROUTE"]
    left_role: Identifier
    right_role: str | None = None
    right_constant: str | list[str] | None = None
    right_from_policy: bool = False
    value_type: Literal["text", "decimal", "money", "date", "datetime"]
    inclusive: bool = True
    grace_days: int = Field(default=0, ge=0)
    currency: Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")] | None = None
    text_normalization: Literal["exact", "casefold_whitespace", "casefold_alnum"] = (
        "exact"
    )

    @model_validator(mode="after")
    def operand_layout(self):
        if self.right_from_policy:
            if self.right_constant is None or self.right_role is not None:
                raise ValueError("policy constant and evidence role are exclusive")
        elif not self.right_role or self.right_constant is not None:
            raise ValueError("evidence comparison requires a right role")
        return self


class PolicyNumber(Contract):
    parameter_id: Identifier
    value: Money
    currency: Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")] | None = None
    refs: list[Citation] = Field(min_length=1)
    compile_status: Literal["verified", "pending", "unresolved"]
    verification_method: str = Field(min_length=1)


class NumberBinding(Contract):
    kind: Literal["operand", "policy_number", "count_members", "table_sum"]
    condition_id: str | None = None
    role: str | None = None
    evidence_id: str | None = None
    parameter_id: str | None = None
    match_suffix: str | None = None
    sheet: str | None = None
    column: str | None = None
    row_keys: list[str] = Field(default_factory=list)
    date_key: str | None = None
    group_key: str | None = None
    required_nonempty: list[str] = Field(default_factory=list)
    required_contains: dict[str, str] = Field(default_factory=dict)
    allowed_values: list[Money] = Field(default_factory=list)
    period_condition_id: str | None = None
    period_role: str | None = None
    members_condition_id: str | None = None
    members_role: str | None = None

    @model_validator(mode="after")
    def binding_shape(self):
        if self.kind == "table_sum":
            if not self.evidence_id or not self.sheet or not self.column or not self.row_keys or any((self.condition_id, self.role, self.parameter_id, self.match_suffix)):
                raise ValueError("table sum needs a source, sheet, column and explicit record keys")
            if bool(self.period_condition_id) != bool(self.period_role) or (self.period_role and not self.date_key):
                raise ValueError("table period filter needs a source date column and factual period role")
            if bool(self.members_condition_id) != bool(self.members_role) or (self.members_role and not self.group_key):
                raise ValueError("authorized table members need a source role and group identity column")
        elif self.kind == "policy_number":
            if not self.parameter_id or any(
                (self.condition_id, self.role, self.evidence_id, self.match_suffix)
            ):
                raise ValueError(
                    "policy number binding requires only a parameter identifier"
                )
        elif not self.condition_id or not self.role or self.parameter_id:
            raise ValueError("source binding requires a condition and factual role")
        if self.match_suffix is not None and self.kind != "count_members":
            raise ValueError("only member counting accepts a suffix selector")
        return self


class MoneyExpression(Contract):
    bindings: dict[Identifier, NumberBinding] = Field(min_length=1, max_length=128)
    expression: dict
    member_partitions: list[list[Identifier]] = Field(default_factory=list)
    table_agreements: list[list[Identifier]] = Field(default_factory=list)

    @model_validator(mode="after")
    def safe_expression(self):
        budget = [128]

        def visit(node, depth=0):
            budget[0] -= 1
            if depth > 16 or budget[0] < 0 or not isinstance(node, dict):
                raise ValueError("amount expression exceeds structural budget")
            op = node.get("op")
            if not isinstance(op, str):
                raise ValueError("amount operation must be a whitelist name")
            if op == "VALUE":
                if (
                    not set(node).issubset({"op", "source", "quantity_source"})
                    or not isinstance(node.get("source"), str)
                    or node.get("source") not in self.bindings
                ):
                    raise ValueError("VALUE requires a declared source binding")
                if (
                    "quantity_source" in node
                    and (not isinstance(node["quantity_source"], str) or node["quantity_source"] not in self.bindings)
                ):
                    raise ValueError("unknown quantity source binding")
                return
            if op not in {"SUM", "MIN", "CAP", "EXCLUDE", "CONVERT"}:
                raise ValueError("amount operation is outside the whitelist")
            fields = (
                {"op", "args", "from_currency", "to_currency", "mode"}
                if op == "CONVERT"
                else {"op", "args"}
            )
            if (
                set(node) != fields
                or not isinstance(node["args"], list)
                or not node["args"]
            ):
                raise ValueError("invalid amount operation fields")
            if op in {"CAP", "EXCLUDE", "CONVERT"} and len(node["args"]) != 2:
                raise ValueError("invalid amount operation arity")
            if op == "CONVERT" and (
                node["mode"] not in {"multiply", "divide"}
                or any(
                    not isinstance(node[key], str)
                    or not re.fullmatch(r"[A-Z]{3}", node[key])
                    for key in ("from_currency", "to_currency")
                )
            ):
                raise ValueError("invalid currency conversion declaration")
            for child in node["args"]:
                visit(child, depth + 1)

        visit(self.expression)
        for partition in self.member_partitions:
            if not partition or any(
                key not in self.bindings or self.bindings[key].kind != "count_members"
                for key in partition
            ):
                raise ValueError(
                    "member partition requires declared member-count bindings"
                )
            groups = {
                (
                    self.bindings[key].condition_id,
                    self.bindings[key].role,
                    self.bindings[key].evidence_id,
                )
                for key in partition
            }
            if len(groups) != 1:
                raise ValueError(
                    "member partition must cover the same original member set"
                )
        for agreement in self.table_agreements:
            if len(agreement) < 2 or any(key not in self.bindings or self.bindings[key].kind != "table_sum" or not self.bindings[key].group_key for key in agreement):
                raise ValueError("table agreement requires grouped source sums")
        return self


class AmountProgram(Contract):
    object_id: Identifier
    basis: MoneyExpression
    limit: MoneyExpression | None = None
    unlimited_refs: list[Citation] = Field(default_factory=list)
    quantum: Literal["1", "0.1", "0.01", "0.001", "0.0001"] = "0.01"
    compile_status: Literal["verified", "pending", "unresolved"] = "pending"
    verification_method: str = ""

    @model_validator(mode="after")
    def limit_basis(self):
        if (self.limit is not None) == bool(self.unlimited_refs):
            raise ValueError(
                "amount program needs a verified limit or explicit unlimited-cost policy"
            )
        if not any(
            binding.kind == "operand" for binding in self.basis.bindings.values()
        ):
            raise ValueError(
                "material-supported maximum needs an actual source operand"
            )
        if self.compile_status == "verified" and not self.verification_method:
            raise ValueError("verified amount program requires an applicability review")
        return self


class Condition(Contract):
    condition_id: Identifier
    object_id: Identifier
    question: str = Field(min_length=1)
    kind: Literal["required", "prohibited", "amount", "open_text"]
    roles: list[Identifier] = Field(min_length=1, max_length=8)
    policy_refs: list[Citation] = Field(min_length=1)
    operation: Operation | None = None
    scope: Literal["item", "group", "application"]
    consequence: Consequence
    compile_status: Literal["verified", "pending", "unresolved"]
    verification_method: str
    role_origins: dict[str, Literal["receipt", "background", "any"]] = Field(
        default_factory=dict
    )
    role_definitions: dict[str, str] = Field(default_factory=dict)
    role_types: dict[str, Literal["text", "decimal", "money", "date", "datetime", "interval", "set"]] = Field(default_factory=dict)
    derived_roles: dict[str, MoneyExpression] = Field(default_factory=dict)

    @model_validator(mode="after")
    def role_layout(self):
        if len(self.roles) != len(set(self.roles)):
            raise ValueError("duplicate role")
        if not set(self.role_origins).issubset(self.roles):
            raise ValueError("role origin refers to an undeclared role")
        if not set(self.role_definitions).issubset(self.roles):
            raise ValueError("definition refers to an undeclared factual role")
        if not set(self.role_types).issubset(self.roles):
            raise ValueError("type declaration refers to an undeclared factual role")
        if not set(self.derived_roles).issubset(self.roles):
            raise ValueError("derived expression refers to an undeclared role")
        if self.operation:
            op = self.operation
            needed = [op.left_role] + ([] if op.right_from_policy else [op.right_role])
            if not set(needed).issubset(self.roles):
                raise ValueError("comparison refers to undeclared roles")
        return self


class Operand(Contract):
    condition_id: Identifier
    role: Identifier
    evidence_id: Identifier
    value: str | list[str] | None
    value_type: Literal[
        "text", "decimal", "money", "date", "datetime", "interval", "set"
    ]
    status: Literal["obtained", "ambiguous", "not_obtained"]
    refs: list[Citation]
    depends_on: list[Identifier]
    currency: str | None = None
    dependency_groups: list[list[Identifier]] = Field(default_factory=list)

    @model_validator(mode="after")
    def source_layout(self):
        if self.status == "obtained" and (self.value is None or not self.refs):
            raise ValueError("obtained value requires source citations")
        if self.evidence_id not in self.depends_on:
            raise ValueError("operand dependency must include its source material")
        if self.dependency_groups:
            if any(not group for group in self.dependency_groups) or {
                i for group in self.dependency_groups for i in group
            } != set(self.depends_on):
                raise ValueError(
                    "dependency alternatives must cover exactly the declared sources"
                )
            if [self.evidence_id] not in self.dependency_groups:
                raise ValueError(
                    "the material containing an observation is a mandatory dependency"
                )
        if any(ref.evidence_id != self.evidence_id for ref in self.refs):
            raise ValueError("operand citation must point to its material")
        return self


class AmountUnit(Contract):
    object_id: Identifier
    scope: Literal["item", "group", "application"]
    claimed_amount: Money
    currency: str
    limit_status: Literal["known", "unlimited", "unknown"]
    limit: Money | None = None
    material_supported_max: Money | None = None
    shared_group_id: str | None = None

    @model_validator(mode="after")
    def cap_layout(self):
        if (self.limit_status == "known") != (self.limit is not None):
            raise ValueError("known cap requires a value; other cap states omit it")
        return self


class SourceDocumentAbsence(Contract):
    evidence_id: Identifier
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checked_segment_ids: list[str] = Field(default_factory=list)
    whole_image: bool = False
    checked_pages: list[int] = Field(default_factory=list)
    page_count: int = Field(default=0, ge=0)
    notes: list[str] = Field(min_length=1)
    raw_call_sha256: list[Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]] = Field(min_length=1)


class DocumentAbsenceProof(Contract):
    condition_id: Identifier
    role: Identifier
    method: Literal["full_submission_document_absence_v1"]
    reader_signature_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sources: list[SourceDocumentAbsence] = Field(min_length=1)


class AuditCase(Contract):
    schema_version: Literal["patent-runtime-v1"] = "patent-runtime-v1"
    workflow_id: Identifier
    input_version: Identifier
    application: Application
    evidence: list[Evidence]
    conditions: list[Condition]
    operands: list[Operand]
    amount_units: list[AmountUnit]
    policy_coverage: Literal["verified", "pending", "unresolved"]
    unresolved: list[str] = Field(default_factory=list)
    policy_numbers: list[PolicyNumber] = Field(default_factory=list)
    amount_programs: list[AmountProgram] = Field(default_factory=list)
    document_absences: list[DocumentAbsenceProof] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_links(self):
        evidence = {e.evidence_id: e for e in self.evidence}
        conditions = {c.condition_id: c for c in self.conditions}
        units = {u.object_id for u in self.amount_units}
        if len(evidence) != len(self.evidence) or len(conditions) != len(
            self.conditions
        ):
            raise ValueError("duplicate material or condition id")
        if len(units) != len(self.amount_units):
            raise ValueError("duplicate amount unit")
        parameters = {p.parameter_id: p for p in self.policy_numbers}
        if len(parameters) != len(self.policy_numbers):
            raise ValueError("duplicate policy numeric parameter")
        for parameter in self.policy_numbers:
            for ref in parameter.refs:
                self.check_citation(ref, evidence)
            numbers = re.findall(
                r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\w|\.\d)",
                " ".join(r.quote or r.note for r in parameter.refs),
            )
            if Decimal(parameter.value) not in {
                Decimal(v.replace(",", "")) for v in numbers
            }:
                raise ValueError(
                    "policy numeric parameter is not present in its source quotes"
                )
        expressions = [
            expr for c in self.conditions for expr in c.derived_roles.values()
        ]
        if len({p.object_id for p in self.amount_programs}) != len(
            self.amount_programs
        ):
            raise ValueError("duplicate amount program")
        for program in self.amount_programs:
            if program.object_id not in units:
                raise ValueError("amount program refers to unknown amount unit")
            expressions.extend(
                [program.basis] + ([program.limit] if program.limit is not None else [])
            )
            for ref in program.unlimited_refs:
                self.check_citation(ref, evidence)
        for expression in expressions:
            for binding in expression.bindings.values():
                if binding.kind == "policy_number":
                    if binding.parameter_id not in parameters:
                        raise ValueError(
                            "amount expression has an unknown policy parameter"
                        )
                elif binding.kind == "table_sum":
                    source = evidence.get(binding.evidence_id)
                    if source is None or source.type != "xlsx":
                        raise ValueError("table aggregation requires a declared spreadsheet source")
                    if binding.period_role:
                        context = conditions.get(binding.period_condition_id)
                        if context is None or binding.period_role not in context.roles or binding.period_role in context.derived_roles:
                            raise ValueError("table period requires a primitive factual date interval")
                    if binding.members_role:
                        context = conditions.get(binding.members_condition_id)
                        if context is None or binding.members_role not in context.roles or binding.members_role in context.derived_roles:
                            raise ValueError("table members require a primitive factual identity set")
                else:
                    condition = conditions.get(binding.condition_id)
                    if (
                        condition is None
                        or binding.role not in condition.roles
                        or binding.role in condition.derived_roles
                    ):
                        raise ValueError(
                            "amount expression needs a declared primitive factual role"
                        )
                    if (
                        binding.evidence_id is not None
                        and binding.evidence_id not in evidence
                    ):
                        raise ValueError("amount binding has an unknown source")
        if set(self.application.attachment_ids) != set(evidence):
            raise ValueError("attachment list and evidence set differ")
        for c in self.conditions:
            if c.object_id not in units:
                raise ValueError("condition refers to an unknown amount unit")
            for ref in c.policy_refs:
                self.check_citation(ref, evidence)
        proof_keys = set()
        for proof in self.document_absences:
            key = (proof.condition_id, proof.role)
            condition = conditions.get(proof.condition_id)
            if key in proof_keys or condition is None or proof.role not in condition.roles or proof.role in condition.derived_roles:
                raise ValueError("document absence needs a unique declared primitive role")
            proof_keys.add(key)
            checked = {source.evidence_id: source for source in proof.sources}
            if len(checked) != len(proof.sources) or set(checked) != set(evidence):
                raise ValueError("document absence must cover every submitted material")
            if any(o.status in {"obtained", "ambiguous"} and o.condition_id == proof.condition_id and o.role == proof.role for o in self.operands):
                raise ValueError("document absence conflicts with a factual observation")
            for identifier, source in checked.items():
                material = evidence[identifier]
                if source.source_sha256 != material.source_sha256:
                    raise ValueError("document absence source hash changed")
                if material.type == "image":
                    if not source.whole_image:
                        raise ValueError("document absence requires the whole image")
                elif material.type == "pdf":
                    if source.page_count < 1 or sorted(source.checked_pages) != list(range(1, source.page_count + 1)):
                        raise ValueError("document absence requires all PDF pages")
                elif not material.segments or set(source.checked_segment_ids) != {segment.segment_id for segment in material.segments}:
                    raise ValueError("document absence requires all text segments")
        for operand in self.operands:
            c = conditions.get(operand.condition_id)
            if c is None or operand.role not in c.roles:
                raise ValueError("operand refers to an unknown condition role")
            if operand.role in c.derived_roles:
                raise ValueError(
                    "derived role values must be recomputed from their source program"
                )
            if operand.evidence_id not in evidence or not set(
                operand.depends_on
            ).issubset(evidence):
                raise ValueError("operand has dangling material dependencies")
            if (
                operand.status == "obtained"
                and evidence[operand.evidence_id].read_status == "not_obtained"
            ):
                raise ValueError("obtained operand requires a source reading")
            for ref in operand.refs:
                self.check_citation(ref, evidence)
            if operand.status == "obtained" and operand.value_type in {
                "money",
                "decimal",
            }:
                # Literal numeric reads must agree with the cited observation.
                # Applicability and units remain part of condition review.
                text = " ".join(
                    ref.quote if ref.kind == "text" else ref.note
                    for ref in operand.refs
                )
                numbers = re.findall(
                    r"(?<![\w.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\w|\.\d)", text
                )
                try:
                    value = Decimal(operand.value)
                    found = {Decimal(v.replace(",", "")) for v in numbers}
                except (InvalidOperation, TypeError):
                    raise ValueError("invalid typed numeric read") from None
                if not value.is_finite() or value not in found:
                    raise ValueError("numeric read differs from cited source value")
        return self

    @staticmethod
    def check_citation(ref: Citation, evidence: dict[str, Evidence]):
        material = evidence.get(ref.evidence_id)
        if material is None:
            raise ValueError("unknown citation source")
        if ref.kind == "text":
            segments = {s.segment_id: s.text for s in material.segments}
            text = segments.get(ref.segment_id)

            def normalize(s):
                return "".join(s.split())

            if text is None or normalize(ref.quote) not in normalize(text):
                raise ValueError("literal citation does not exist in source segment")
        elif material.type not in {"image", "pdf"}:
            raise ValueError("whole-image observation requires image or PDF")

    def digest(self) -> str:
        payload = self.model_dump()
        for condition in payload["conditions"]:
            if not condition.get("role_origins"):
                condition.pop("role_origins", None)
            if not condition.get("role_definitions"):
                condition.pop("role_definitions", None)
            if not condition.get("role_types"):
                condition.pop("role_types", None)
            if not condition.get("derived_roles"):
                condition.pop("derived_roles", None)
            operation = condition["operation"]
            if operation is not None and operation.get("currency") is None:
                operation.pop("currency", None)
            if operation is not None and operation.get("text_normalization") == "exact":
                operation.pop("text_normalization", None)
        for optional in ("policy_numbers", "amount_programs", "document_absences"):
            if not payload.get(optional):
                payload.pop(optional, None)
        for operand in payload["operands"]:
            if not operand.get("dependency_groups"):
                operand.pop("dependency_groups", None)
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(encoded.encode()).hexdigest()


class RelationTarget(Contract):
    evidence_id: Identifier
    relation: Literal["support", "conflict", "unknown"]
    approved: bool


class ReferenceTarget(Contract):
    evidence_id: Identifier
    polarity: Literal["support", "conflict"]
    label: Literal[0, 1]
    approved: bool


class ConditionAnnotation(Contract):
    schema_version: Literal["patent-supervision-v1"] = "patent-supervision-v1"
    workflow_id: Identifier
    condition_id: Identifier
    split: Literal["train", "validation", "test"]
    source_family_id: Identifier
    augmentation_of: Identifier | None = None
    case_sha256: str
    status: Literal["approved", "corrected", "pending", "excluded"]
    method: str
    reviewer: str | None
    review_reason: str
    state: State | None
    relations: list[RelationTarget]
    references: list[ReferenceTarget]

    @model_validator(mode="after")
    def require_review(self):
        approved = self.status in {"approved", "corrected"}
        if approved and (
            not self.reviewer or not self.method or not self.review_reason
        ):
            raise ValueError("approved supervision requires recorded verification")
        if not approved and any(
            x.approved for x in [*self.relations, *self.references]
        ):
            raise ValueError("pending/excluded records cannot supply approved labels")
        return self
