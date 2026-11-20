"""Probe whether the current machine can run the formal Qwen training plan."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

REQUIRED_DISTRIBUTIONS = (
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
REQUIRED_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def _package_versions() -> dict[str, str | None]:
    versions = {}
    for distribution in REQUIRED_DISTRIBUTIONS:
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _model_config(path: Path) -> tuple[str | None, str | None]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise ValueError(f"model configuration is missing from {path}")
    model = config["model"]
    return model.get("name"), model.get("revision")


def probe_environment(config_path: Path) -> dict[str, Any]:
    model_name, model_revision = _model_config(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    packages = _package_versions()
    probe: dict[str, Any] = {
        "operating_system": platform.system(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "packages": packages,
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_bf16_supported": False,
        "gpu_names": [],
        "gpu_memory_gib": [],
        "qwen_interface_available": False,
        "qwen_interface_error": None,
        "model_name": model_name,
        "model_revision": model_revision,
        "environment_profile": config.get("environment", {}),
    }
    if packages["torch"] is not None:
        torch = importlib.import_module("torch")
        probe["cuda_available"] = bool(torch.cuda.is_available())
        probe["cuda_device_count"] = int(torch.cuda.device_count())
        if probe["cuda_available"]:
            probe["cuda_bf16_supported"] = bool(torch.cuda.is_bf16_supported())
            for index in range(probe["cuda_device_count"]):
                properties = torch.cuda.get_device_properties(index)
                probe["gpu_names"].append(properties.name)
                probe["gpu_memory_gib"].append(
                    round(properties.total_memory / (1024**3), 2)
                )
    if packages["transformers"] is not None:
        try:
            transformers = importlib.import_module("transformers")
            getattr(transformers, "Qwen3VLForConditionalGeneration")
            getattr(transformers, "AutoProcessor")
            probe["qwen_interface_available"] = True
        except (AttributeError, ImportError) as error:
            probe["qwen_interface_error"] = f"{type(error).__name__}: {error}"
    return probe


def assess_environment(probe: dict[str, Any]) -> dict[str, Any]:
    blockers = []
    profile = probe.get("environment_profile", {})
    smoke = profile.get("profile") == "smoke"
    if probe.get("operating_system") != "Linux":
        blockers.append("linux_required_for_formal_training")
    allowed_python = profile.get("python_versions", ["3.11"]) if smoke else ["3.11"]
    if not any(
        str(probe.get("python_version", "")).startswith(v + ".") for v in allowed_python
    ):
        blockers.append("python_version_outside_profile")
    for package, version in probe.get("packages", {}).items():
        if version is None:
            blockers.append(f"missing_package:{package}")
    if not probe.get("cuda_available"):
        blockers.append("cuda_unavailable")
    required_gpus = int(profile.get("required_gpu_count", 1)) if smoke else 2
    if int(probe.get("cuda_device_count", 0)) < required_gpus:
        blockers.append("insufficient_gpu_count")
    if not probe.get("cuda_bf16_supported"):
        blockers.append("bf16_unavailable")
    memory = probe.get("gpu_memory_gib", [])
    minimum_memory = float(profile.get("minimum_gpu_memory_gib", 24)) if smoke else 72
    if len(memory) >= required_gpus and any(
        float(value) < minimum_memory for value in memory[:required_gpus]
    ):
        blockers.append("gpu_memory_below_profile_budget")
    if not probe.get("qwen_interface_available"):
        blockers.append("qwen3_vl_interface_unavailable")
    if probe.get("model_name") != REQUIRED_MODEL:
        blockers.append("unexpected_model_name")
    if not probe.get("model_revision"):
        blockers.append("model_revision_unpinned")
    return {
        "status": "blocked" if blockers else "ready_for_model_smoke_test",
        "blocking_reasons": blockers,
        "formal_training_started": False,
        "profile": "smoke" if smoke else "formal_a800",
        "environment_lock_allowed": False,
        "remaining_gate": "actual_model_image_feature_and_gradient_smoke_required",
    }


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(content, encoding="utf-8", newline="\n")
    temporary_path.replace(path)


def render_report(probe: dict[str, Any], assessment: dict[str, Any]) -> str:
    blockers = (
        "\n".join(f"- `{reason}`" for reason in assessment["blocking_reasons"])
        or "- 无"
    )
    missing = [name for name, version in probe["packages"].items() if version is None]
    return f"""# 训练环境门禁报告

- 状态：`{assessment["status"]}`
- 操作系统：{probe["operating_system"]}
- Python：{probe["python_version"]}
- CUDA 可用：{probe["cuda_available"]}
- GPU 数量：{probe["cuda_device_count"]}
- BF16 可用：{probe["cuda_bf16_supported"]}
- 缺少依赖：{json.dumps(missing, ensure_ascii=False)}
- 模型：{probe["model_name"]}
- 模型 revision：{probe["model_revision"]}

## 阻塞原因

{blockers}

正式训练尚未启动。只有在兼容服务器完成模型加载、图像前向、隐藏状态和 LoRA 梯度冒烟后，才能生成 `environment.lock`。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base512.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/readiness"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    probe = probe_environment(args.config)
    assessment = assess_environment(probe)
    result = {"probe": probe, "assessment": assessment}
    _atomic_write(
        args.output_dir / "training_environment.json",
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write(
        args.output_dir / "training_environment_report.md",
        render_report(probe, assessment),
    )
    print(assessment["status"])
    return 0 if assessment["status"] == "ready_for_model_smoke_test" else 2


if __name__ == "__main__":
    raise SystemExit(main())
