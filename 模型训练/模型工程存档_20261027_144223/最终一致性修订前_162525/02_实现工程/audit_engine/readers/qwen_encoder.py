"""Pinned local Qwen3-VL reader and frozen final-input-position encoder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import Tensor

from audit_engine.models.pooling import pool_last_input


class FrozenQwenEncoder:
    def __init__(self, config: dict, local_path: Path, device: str = "cuda:0"):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        if not config["model"]["revision"]:
            raise ValueError("model revision must be pinned")
        download_record = (
            local_path / ".cache/huggingface/download/config.json.metadata"
        )
        snapshot_matches = (
            download_record.is_file()
            and download_record.read_text().splitlines()[0]
            == config["model"]["revision"]
        )
        range_manifest = local_path / ".range_parts/manifest.json"
        if range_manifest.is_file():
            manifest = json.loads(range_manifest.read_text())
            if manifest["sha"] != config["model"]["revision"]:
                raise ValueError("local checkpoint range revision mismatch")
            for record in manifest["siblings"]:
                name = record["rfilename"]
                if name.endswith((".md", ".png", ".jpg")) or name == ".gitattributes":
                    continue
                artifact = local_path / name
                if not artifact.is_file() or artifact.stat().st_size != record["size"]:
                    raise ValueError(
                        f"checkpoint artifact is missing or incomplete: {name}"
                    )
                if not record.get("lfs"):
                    data = artifact.read_bytes()
                    blob = hashlib.sha1(
                        f"blob {len(data)}\0".encode() + data
                    ).hexdigest()
                    if blob != record["blobId"]:
                        raise ValueError(
                            f"checkpoint tokenizer/config blob mismatch: {name}"
                        )
        elif not snapshot_matches:
            raise ValueError("local checkpoint does not match pinned revision metadata")
        self.config = config
        self.processor = AutoProcessor.from_pretrained(
            local_path,
            local_files_only=True,
            max_pixels=config["reader"]["image_max_pixels"],
        )
        self.processor.image_processor.size["longest_edge"] = config["reader"][
            "image_max_pixels"
        ]
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            local_path,
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map=config["model"].get("device_map", {"": device}),
            attn_implementation=config["model"]["attention_backend"],
        ).eval()
        adapter = config["model"].get("adapter_path")
        adapter_signature = {}
        if adapter:
            from peft import PeftModel

            adapter_path = Path(adapter).resolve()
            weights = adapter_path / "adapter_model.safetensors"
            adapter_config = adapter_path / "adapter_config.json"
            if not weights.is_file() or not adapter_config.is_file():
                raise ValueError("configured frozen reader adapter is incomplete")
            self.model = PeftModel.from_pretrained(
                self.model, adapter_path, is_trainable=False
            ).eval()
            self.backbone = self.model.get_base_model().model
            adapter_signature = {
                "adapter_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                "adapter_config_sha256": hashlib.sha256(
                    adapter_config.read_bytes()
                ).hexdigest(),
            }
        else:
            self.backbone = self.model.model
        self.model.requires_grad_(False)
        if self.model.config.text_config.hidden_size != config["model"]["hidden_size"]:
            raise ValueError("unexpected backbone hidden size")
        self.device = device
        self.signature = {
            "model": config["model"]["name"],
            "revision": config["model"]["revision"],
            "attention": config["model"]["attention_backend"],
            "max_pixels": config["reader"]["image_max_pixels"],
            "encoding_prompt": "Read this material and preserve factual information. Do not make a final audit decision.",
            "pooling": "last_non_padding_input_v1",
            **adapter_signature,
        }

    def inputs(self, text: str, image_path: Path | None = None):
        content = []
        if image_path is not None:
            content.append({"type": "image", "image": str(image_path.resolve())})
        content.append({"type": "text", "text": text})
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        if "image_grid_thw" in inputs:
            patch = self.processor.image_processor.patch_size
            pixels = inputs.image_grid_thw[:, 1:].prod(dim=-1) * patch**2
            if bool((pixels > self.config["reader"]["image_max_pixels"]).any()):
                raise ValueError("processor exceeded configured image pixel budget")
        if inputs.input_ids.size(1) > self.config["reader"]["total_input_tokens"]:
            raise ValueError(
                "input exceeds token budget; explicit document chunking is required"
            )
        return inputs

    def encode(self, text: str, image_path: Path | None = None) -> Tensor:
        inputs = self.inputs(
            self.signature["encoding_prompt"] + "\n" + text, image_path
        )
        with torch.inference_mode():
            output = self.backbone(**inputs, use_cache=False, return_dict=True)
            pooled = (
                pool_last_input(
                    output.last_hidden_state,
                    inputs.attention_mask.to(output.last_hidden_state.device),
                )
                .float()
                .cpu()
            )
        # Clone outside inference mode so trainable projections can save inputs.
        return pooled.clone().squeeze(0)

    def read(
        self, prompt: str, image_path: Path | None = None, max_new_tokens: int = 256
    ) -> str:
        inputs = self.inputs(prompt, image_path)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        return self.processor.batch_decode(
            generated[:, inputs.input_ids.size(1) :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def cache_key(self, text: str, image_path: Path | None = None) -> str:
        digest = hashlib.sha256(json.dumps(self.signature, sort_keys=True).encode())
        digest.update(text.encode())
        if image_path is not None:
            digest.update(image_path.read_bytes())
        return digest.hexdigest()
