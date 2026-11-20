import unittest

from audit_engine.models.heads import AuditHeads
from audit_engine.training.stages import configure_trainable_stage


def trainable_parameter_names(model):
    return {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }


class TrainingStageTests(unittest.TestCase):
    def test_b1_trains_shared_and_relation_only(self):
        model = AuditHeads(base_dimension=16, dimension=8, width=4, role_count=2)

        configure_trainable_stage(model, "B1")
        names = trainable_parameter_names(model)

        self.assertTrue(any(name.startswith("shared.") for name in names))
        self.assertTrue(any(name.startswith("relation.") for name in names))
        self.assertFalse(any(name.startswith("contribution.") for name in names))
        self.assertFalse(any(name.startswith("corrector.") for name in names))

    def test_b2_trains_contribution_only(self):
        model = AuditHeads(base_dimension=16, dimension=8, width=4, role_count=2)

        configure_trainable_stage(model, "B2")
        names = trainable_parameter_names(model)

        self.assertTrue(names)
        self.assertTrue(all(name.startswith("contribution.") for name in names))

    def test_c_trains_corrector_only(self):
        model = AuditHeads(base_dimension=16, dimension=8, width=4, role_count=2)

        configure_trainable_stage(model, "C")
        names = trainable_parameter_names(model)

        self.assertTrue(names)
        self.assertTrue(all(name.startswith("corrector.") for name in names))

    def test_unknown_stage_is_rejected(self):
        model = AuditHeads(base_dimension=16, dimension=8, width=4, role_count=2)

        with self.assertRaisesRegex(ValueError, "unknown training stage"):
            configure_trainable_stage(model, "D")


if __name__ == "__main__":
    unittest.main()
