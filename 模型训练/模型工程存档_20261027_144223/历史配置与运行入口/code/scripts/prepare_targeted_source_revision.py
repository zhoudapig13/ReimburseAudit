"""Source-only policy revision and additional meal training families.

This script never opens reference answers. The original source reviews remain
unchanged; revised checks and their scope decisions are saved in a new version.
"""

import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit_engine.contracts import AuditCase
from audit_engine.training.patent import read_jsonl
from compile_reviewed_policies import compile_case


def rule(identifier, question, refs, roles, definitions, *, operation="EQ", constant=None,
         value_type="text", origins=None, types=None, consequence="reject_object", state=None,
         support=None, conflict=None, derived=None):
    result = {"id": identifier, "question": question, "policy_refs": refs,
              "operation": operation, "roles": roles, "role_definitions": definitions,
              "role_origins": origins or {}, "role_types": types or {},
              "value_type": value_type, "text_normalization": "casefold_whitespace",
              "on_failed": consequence}
    if constant is not None:
        result["right_constant"] = constant
    if derived:
        result["derived_roles"] = derived
    if state:
        result.update(candidate_state=state, positive_sources=support or [],
                      conflict_sources=conflict or [], negative_sources=[])
    return result


def main():
    root = Path("runs/patent_targeted_source_20261027_0930")
    root.mkdir(exist_ok=True)
    reviews = Path("data/patent_source_review_v14")
    reviews.mkdir(exist_ok=True)
    for path in Path("data/patent_source_review_v13").glob("*.json"):
        target = reviews / path.name
        if not target.exists():
            target.write_bytes(path.read_bytes())
    cases = {row["workflow_id"]: AuditCase.model_validate(row)
             for row in read_jsonl(Path("data/patent_full/cases.jsonl"))}
    now = datetime.now(timezone.utc).isoformat()
    changed = []
    for identifier in ["1.2.1", "1.2.2", "1.2.3", "1.2.4"]:
        review = json.loads((Path("data/patent_source_review_v13") / (identifier + ".json")).read_text())
        review["reviewed_at"] = now
        review["method"] = "source_review_meal_fields_category_location_and_shared_cap_v14"
        review["confirmed_policy"] = [r for r in review["confirmed_policy"]
                                      if r["id"] != "basic_meal_food_present"]
        review["confirmed_policy"].append(rule(
            "expense_is_a_meal", "Does the original itemization include food as a meal expense?",
            ["e002:t028", "e002:t029", "e002:t030", "e002:t032"],
            ["actual_e006_consumption_category"],
            {"actual_e006_consumption_category":
             "Classify the actual purchased itemization as meal_with_food, beverage_only, or non_dining. "
             "Use meal_with_food when any actual food dish is present, including mixed food and beverages. "
             "Use beverage_only only when every purchased item is a drink. Do not classify compliance, "
             "use a blanket alcohol prohibition, or substitute the claimant's description."},
            constant="meal_with_food", origins={"actual_e006_consumption_category": "receipt"}))
        review["confirmed_policy"].append(rule(
            "merchant_city_matches_travel", "Is the original merchant in the documented travel city?",
            ["e002:t074"], ["actual_e006_merchant_city", "documented_e001_travel_city"],
            {"actual_e006_merchant_city": "Read the city from the actual merchant address. Normalize an airport or borough to its printed metropolitan city, for example Queens/JFK to New York City. Do not take it from background documents.",
             "documented_e001_travel_city": "Read the city of this meeting and delayed return trip, not the future meeting city or home city. Normalize the actual source city to its English city name."},
            origins={"actual_e006_merchant_city": "receipt", "documented_e001_travel_city": "background"},
            consequence="manual_review"))
        review["confirmed_policy"].append(rule(
            "documented_force_majeure_reason", "Is a return-flight disruption documented in the original situation explanation?",
            ["e002:t030", "e002:t043", "e002:t072"], ["documented_e001_return_disruption"],
            {"documented_e001_return_disruption": "Classify the actual stated cause of the return delay as weather_flight_delay, unrelated_personal_delay, or no_explanation. Read the original return travel situation explanation, not a receipt or a later meeting deadline."},
            constant="weather_flight_delay", origins={"documented_e001_return_disruption": "background"},
            consequence="request_material"))
        review["policy_coverage"] = "verified"
        review["coverage_review"] = {
            "method": "original_financial_eligibility_scope_inspection",
            "reviewer": "codex_source_inspection", "reviewed_at": now,
            "included": ["actual receipt format and all required printed fields", "meal expense scope without an invented alcohol ban", "actual city and delay dates", "documented force-majeure explanation", "participant count consistency", "source-roster sum of individual grade limits", "original-currency amount basis"],
            "administrative_scope": "The review is the pre-payment financial decision. Future Finance approval, payment vouchers and bank transfers are subsequent workflow actions. The submitted situation explanation records its department-head reviewer. A count or background discrepancy remains a review task.",
            "inapplicable": ["employee-training policy: this activity is a partner business meeting", "foreign-exchange conversion when the original receipt and requested settlement are both USD"]}
        (reviews / (identifier + ".json")).write_text(json.dumps(review, ensure_ascii=False, indent=2))
        changed.append(identifier)

    # Four neighbouring meal workflows belong to the original TRAIN partition.
    # Their original images were inspected independently: 2825 is preserved,
    # and the taxi slip is not relabelled as a meal receipt.
    for identifier in ["2.1.1", "2.1.2", "2.1.3", "2.1.4"]:
        case = cases[identifier]
        review = {"workflow_id": identifier, "source_sha256": {e.evidence_id: e.source_sha256 for e in case.evidence},
                  "reviewer": "codex_source_inspection", "reviewed_at": now,
                  "method": "original_training_meal_images_and_policy_inspection_v14",
                  "runtime_operands_injected": False, "reference_answer_read": False,
                  "policy_coverage": "pending", "confirmed_policy": []}
        receipt_ids = [e.evidence_id for e in case.evidence if e.type == "image"]
        for evidence_id in receipt_ids:
            taxi = identifier == "2.1.3"
            actual = "actual_" + evidence_id + "_"
            source = [evidence_id]
            for suffix, definition, constant, state in [
                ("receipt_format", "Classify the visible receipt body as machine_printed or handwritten, independently of the merchant header.", "machine_printed", "satisfied"),
                ("required_fields", "Return complete only if this actual dining receipt has a readable merchant name, date, detailed purchased items and amount. Return incomplete for a generic Invoice Header without a merchant name or for a card/taxi charge slip without dining itemization. Do not borrow fields from other documents.", "complete", "failed"),
                ("consumption_category", "Classify the actual purchased items as meal_with_food, beverage_only, or non_dining. A taxi transportation slip is non_dining. Mixed dishes and drinks are meal_with_food. Return the source category, not a compliance verdict.", "meal_with_food", "failed" if taxi else "satisfied")]:
                refs = ["e003:t083", "e003:t084", "e003:t092", "e003:t093"] if suffix != "consumption_category" else ["e003:t028", "e003:t029", "e003:t094"]
                review["confirmed_policy"].append(rule(
                    suffix + "_" + evidence_id, "Check the original dining receipt " + suffix.replace("_", " ") + ".",
                    refs, [actual + suffix], {actual + suffix: definition}, constant=constant,
                    origins={actual + suffix: "receipt"}, state=state,
                    support=source if state == "satisfied" else [], conflict=source if state == "failed" else []))
            # Keep the individually inspected source year, including unusual years.
            if not taxi:
                date_failed = identifier == "2.1.2" or (identifier == "2.1.1" and evidence_id in {"e007", "e008"})
                review["confirmed_policy"].append(rule(
                    "expense_date_" + evidence_id, "Does the actual receipt date fall within the approved negotiation period?",
                    ["e003:t079"], [actual + "meal_date", "documented_e005_negotiation_interval"],
                    {actual + "meal_date": "Transcribe the actual date printed on this receipt, preserving every year digit. Do not change an unusual future year to the approved year.",
                     "documented_e005_negotiation_interval": "The start and end dates of the approved Washington negotiation trip, excluding document preparation and later action deadlines."},
                    operation="WITHIN", value_type="date", origins={actual + "meal_date": "receipt", "documented_e005_negotiation_interval": "background"},
                    state="failed" if date_failed else "satisfied",
                    support=[] if date_failed else [evidence_id, "e005"],
                    conflict=[evidence_id, "e005"] if date_failed else []))
        for condition in review["confirmed_policy"]:
            providers = set(condition.get("positive_sources", []) + condition.get("conflict_sources", []))
            condition["negative_sources"] = sorted(set(review["source_sha256"]) - providers)
        (reviews / (identifier + ".json")).write_text(json.dumps(review, ensure_ascii=False, indent=2))
        changed.append(identifier)
    output = Path("data/patent_reviewed/policy_cases_v14.jsonl")
    temporary = output.with_suffix(".tmp")
    with temporary.open("w") as stream:
        for case in cases.values():
            path = reviews / (case.workflow_id + ".json")
            if path.exists():
                case = compile_case(case, json.loads(path.read_text()))
            stream.write(case.model_dump_json() + "\n")
    temporary.replace(output)
    selection = {"workflow_ids": changed, "reference_answers_read": False}
    (root / "selection.json").write_text(json.dumps(selection, indent=2))
    print(json.dumps({"changed": changed, "compiled": str(output)}))


if __name__ == "__main__":
    main()
