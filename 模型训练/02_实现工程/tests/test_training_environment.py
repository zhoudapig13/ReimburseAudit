import importlib.util
import unittest
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "scripts"
    / "check_training_environment.py"
)


def load_environment_module():
    spec = importlib.util.spec_from_file_location(
        "check_training_environment", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load environment module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ready_probe():
    return {
        "operating_system": "Linux",
        "python_version": "3.12.3",
        "packages": {
            name: "1.0"
            for name in (
                "torch",
                "transformers",
                "peft",
                "accelerate",
                "safetensors",
                "tokenizers",
                "Pillow",
                "openpyxl",
                "python-docx",
                "PyMuPDF",
                "pydantic",
                "PyYAML",
                "scikit-learn",
            )
        },
        "cuda_available": True,
        "cuda_device_count": 1,
        "cuda_bf16_supported": True,
        "gpu_memory_gib": [32.0],
        "qwen_interface_available": True,
        "model_name": "Qwen/Qwen3-VL-8B-Instruct",
        "model_revision": "abc123",
    }


class EnvironmentAssessmentTests(unittest.TestCase):
    def test_delivery_profile_checks_single_5090_memory_budget(self):
        import yaml

        environment = load_environment_module()
        config = yaml.safe_load((SCRIPT_PATH.parents[2] / "configs/patent_delivery.yaml").read_text())
        probe = ready_probe()
        probe["environment_profile"] = config["environment"]
        probe["gpu_memory_gib"] = [16.0]
        self.assertIn("gpu_memory_below_profile_budget", environment.assess_environment(probe)["blocking_reasons"])
        probe["gpu_memory_gib"] = [32.0]
        assessment = environment.assess_environment(probe)
        self.assertEqual(assessment["status"], "ready_for_model_smoke_test")
        self.assertFalse(assessment["environment_lock_allowed"])

    def test_compatible_server_is_ready(self):
        environment = load_environment_module()

        assessment = environment.assess_environment(ready_probe())

        self.assertEqual(assessment["status"], "ready_for_model_smoke_test")
        self.assertEqual(assessment["blocking_reasons"], [])

    def test_cpu_machine_reports_actionable_blockers(self):
        environment = load_environment_module()
        probe = ready_probe()
        probe.update(
            {
                "operating_system": "Windows",
                "python_version": "3.12.7",
                "cuda_available": False,
                "cuda_device_count": 0,
                "cuda_bf16_supported": False,
                "gpu_memory_gib": [],
                "qwen_interface_available": False,
                "model_revision": None,
            }
        )
        probe["packages"]["transformers"] = None

        assessment = environment.assess_environment(probe)

        self.assertEqual(assessment["status"], "blocked")
        self.assertIn(
            "linux_required_for_formal_training", assessment["blocking_reasons"]
        )
        self.assertIn("cuda_unavailable", assessment["blocking_reasons"])
        self.assertIn("model_revision_unpinned", assessment["blocking_reasons"])
        self.assertIn("missing_package:transformers", assessment["blocking_reasons"])


if __name__ == "__main__":
    unittest.main()
