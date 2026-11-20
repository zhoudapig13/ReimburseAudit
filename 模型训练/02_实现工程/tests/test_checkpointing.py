import copy
import unittest
from functools import partial

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from audit_engine.models.checkpointing import (
    enable_device_local_decoder_checkpoints,
    move_arguments,
)
from audit_engine.models.chunked_mlp import chunked_forward


class CheckpointingTests(unittest.TestCase):
    def test_nested_arguments_preserve_metadata_and_tensor_dtype(self):
        source = {
            "x": [torch.ones(2, dtype=torch.float64)],
            "metadata": ("currency", "USD"),
        }
        moved = move_arguments(source, torch.device("cpu"))
        self.assertEqual(moved["metadata"], source["metadata"])
        self.assertEqual(moved["x"][0].dtype, torch.float64)

    def test_rms_norm_chunk_outputs_and_gradients_match(self):
        torch.manual_seed(2)
        first = nn.RMSNorm(4).double()
        second = copy.deepcopy(first)
        x = torch.randn(2, 13, 4, dtype=torch.float64, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        weights = torch.randn_like(x)
        a = first(x)
        b = chunked_forward(second, y, 3)
        (a * weights).sum().backward()
        (b * weights).sum().backward()
        self.assertTrue(torch.allclose(a, b, atol=1e-12, rtol=1e-12))
        self.assertTrue(torch.allclose(x.grad, y.grad, atol=1e-11, rtol=1e-11))
        self.assertTrue(
            torch.allclose(
                first.weight.grad, second.weight.grad, atol=1e-11, rtol=1e-11
            )
        )

    @unittest.skipUnless(
        torch.cuda.device_count() >= 6, "requires two free CUDA test devices 4/5"
    )
    def test_real_qwen_layer_dropout_gradients_match_across_devices(self):
        from accelerate.hooks import AlignDevicesHook, add_hook_to_module
        from peft import LoraConfig, inject_adapter_in_model
        from transformers import Qwen3VLTextConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLTextDecoderLayer,
        )

        config = Qwen3VLTextConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
        )
        config._attn_implementation = "sdpa"
        layer = Qwen3VLTextDecoderLayer(config, 0).to("cuda:5").requires_grad_(False)
        inject_adapter_in_model(
            LoraConfig(
                r=4,
                lora_alpha=8,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            ),
            layer,
        )
        for name, p in layer.named_parameters():
            if "lora_B" in name:
                torch.nn.init.normal_(p, std=0.01)
        add_hook_to_module(
            layer, AlignDevicesHook(execution_device=torch.device("cuda:5"))
        )
        layer.train()
        layer._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
        root = nn.Module()
        root.language_model = nn.Module()
        root.language_model.layers = nn.ModuleList([layer])
        source = torch.randn(1, 31, 64, device="cuda:4")
        positions = (
            torch.ones(1, 31, 16, device="cuda:4"),
            torch.zeros(1, 31, 16, device="cuda:4"),
        )

        def run(enabled):
            for device in [4, 5]:
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(541)
            layer.zero_grad(set_to_none=True)
            layer.gradient_checkpointing = enabled
            x = source.detach().clone().requires_grad_()
            out = layer(
                x, position_embeddings=positions, attention_mask=None, use_cache=False
            )
            out.square().mean().backward()
            return (
                out.detach().cpu(),
                {
                    n: p.grad.detach().cpu().clone()
                    for n, p in layer.named_parameters()
                    if p.requires_grad
                },
                x.grad.detach().cpu(),
            )

        expected = run(False)
        self.assertEqual(enable_device_local_decoder_checkpoints(root), 1)
        actual = run(True)
        self.assertTrue(torch.allclose(expected[0], actual[0], atol=1e-7, rtol=1e-5))
        self.assertTrue(torch.allclose(expected[2], actual[2], atol=1e-7, rtol=1e-5))
        for name in expected[1]:
            self.assertTrue(
                torch.allclose(
                    expected[1][name], actual[1][name], atol=1e-7, rtol=1e-5
                ),
                name,
            )
