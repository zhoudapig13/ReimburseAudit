import unittest

import torch

from audit_engine.policy.comparators import evaluate
from audit_engine.training.forward import ScalarCondition


class PolicyComparatorTests(unittest.TestCase):
    def test_fixed_policy_limit_is_not_an_artificial_evidence_role(self):
        policy = ScalarCondition(
            "LE", (("120",),), (0,), right_from_policy=True, right_constant="100"
        )
        role_map, required, comparison, complete = policy.rebuild(
            torch.tensor([True]), 8
        )
        self.assertEqual(int(required.sum()), 1)
        self.assertEqual(int(role_map.sum()), 1)
        self.assertTrue(complete.bool().all())
        self.assertEqual(comparison.argmax(-1).tolist(), [1])
        masked = policy.rebuild(torch.tensor([False]), 8)
        self.assertFalse(masked[3].any())
        self.assertEqual(masked[2].argmax(-1).tolist(), [2])

    def test_decimal_comparison_is_exact_and_currency_is_required_for_money(self):
        self.assertTrue(evaluate("EQ", "0.10", "0.1", value_type="decimal").passed)
        self.assertFalse(evaluate("LE", "100.01", "100.00").passed)
        self.assertTrue(evaluate("GE", "100.00", "100.00").passed)
        self.assertFalse(evaluate("LE", "90", "100", value_type="money").complete)
        self.assertFalse(
            evaluate(
                "LE",
                "90",
                "100",
                value_type="money",
                left_currency="USD",
                right_currency="CNY",
            ).complete
        )
        self.assertTrue(
            evaluate(
                "LE",
                "90",
                "100",
                value_type="money",
                left_currency="USD",
                right_currency="USD",
            ).passed
        )

    def test_date_boundaries_and_declared_grace_period(self):
        interval = ("2026-10-11", "2026-10-13")
        self.assertTrue(evaluate("WITHIN", "2026-10-11", interval).passed)
        self.assertFalse(
            evaluate("WITHIN", "2026-10-11", interval, inclusive=False).passed
        )
        self.assertFalse(evaluate("WITHIN", "2026-10-14", interval).passed)
        self.assertTrue(evaluate("WITHIN", "2026-10-14", interval, grace_days=1).passed)
        self.assertFalse(evaluate("WITHIN", "10/12/2026", interval).complete)
        self.assertFalse(evaluate("WITHIN", "2026-10-12", interval[::-1]).complete)

    def test_set_and_authorization_use_explicit_verified_membership(self):
        self.assertTrue(evaluate("IN_SET", "San Diego", ("San Diego", "Boston")).passed)
        self.assertFalse(evaluate("IN_SET", "San Diego", ("Boston",)).passed)
        self.assertTrue(
            evaluate("AUTHORIZED", "Finance director", ("Finance director",)).passed
        )
        self.assertFalse(
            evaluate("AUTHORIZED", "Project manager", ("Finance director",)).passed
        )
        self.assertFalse(
            evaluate("AUTHORIZED", "Project manager", "Manager or director").complete
        )

    def test_bad_values_do_not_become_a_deterministic_rule_result(self):
        for invalid in ["NaN", "Infinity", "1,000", "USD 100", ""]:
            self.assertFalse(evaluate("LE", invalid, "100").complete)
        self.assertFalse(evaluate("LE", None, "100").complete)
        self.assertFalse(evaluate("eval", "1", "1").complete)

    def test_previous_reimbursement_exclusion_preserves_compliance_polarity(self):
        self.assertFalse(evaluate("NOT_IN_SET", "G123456", ("G123456", "G123457")).passed)
        self.assertTrue(evaluate("NOT_IN_SET", "G999999", ("G123456", "G123457")).passed)
        self.assertFalse(evaluate("NOT_IN_SET", "G999999", None).complete)
        policy = ScalarCondition("NOT_IN_SET", (("G123456", None), (None, ("G123456",))), (0, 1))
        self.assertEqual(policy.rebuild(torch.tensor([True, True]), 8)[2].argmax(-1).tolist(), [1, 1])
        masked = policy.rebuild(torch.tensor([True, False]), 8)
        self.assertFalse(masked[3].any())
        self.assertEqual(masked[2].argmax(-1).tolist(), [2, 2])

    def test_closed_business_route_requires_every_leg_and_project_base(self):
        self.assertTrue(evaluate("CLOSED_ROUTE", ("Potenza -> Napoli", "Napoli -> Potenza"), "Potenza").passed)
        self.assertFalse(evaluate("CLOSED_ROUTE", ("Roma -> Firenze", "Firenze -> Napoli", "Napoli -> Potenza", "Potenza -> Salerno"), "Potenza").passed)
        self.assertFalse(evaluate("CLOSED_ROUTE", ("Potenza -> Napoli", "Napoli -> Potenza", "Roma -> Firenze", "Firenze -> Roma"), "Potenza").passed)
        self.assertFalse(evaluate("CLOSED_ROUTE", ("Napoli -> Roma", "Roma -> Napoli"), "Potenza").passed)
        self.assertFalse(evaluate("CLOSED_ROUTE", ("unclear route",), "Potenza").complete)
        policy = ScalarCondition("CLOSED_ROUTE",
            (("Potenza -> Napoli", None, None), (None, "Napoli -> Potenza", None), (None, None, "Potenza")),
            (0, 1, 2), right_role=2)
        self.assertEqual(policy.rebuild(torch.tensor([True, True, True]), 8)[2].argmax(-1).tolist(), [0, 0, 0])
        self.assertFalse(policy.rebuild(torch.tensor([True, False, True]), 8)[3].any())

    def test_masked_interval_source_invalidates_all_derived_comparisons(self):
        policy = ScalarCondition(
            "WITHIN",
            (("2026-10-12", None), (None, ("2026-10-11", "2026-10-13"))),
            (0, 1),
        )
        self.assertTrue(policy.rebuild(torch.tensor([True, True]), 8)[3].bool().all())
        masked = policy.rebuild(torch.tensor([True, False]), 8)
        self.assertFalse(masked[3].any())
        self.assertEqual(masked[2].argmax(-1).tolist(), [2, 2])


if __name__ == "__main__":
    unittest.main()
