# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare full and pre-sharded HSDP loading of a reduced MiniMax-H3 DiT end to end."""

from __future__ import annotations

import gc
import json
import sys
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors.torch import load_file, save_file
from torch.distributed.tensor import DTensor
from vllm.utils.network_utils import get_file_store_init_method

from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader

pytestmark = [pytest.mark.diffusion, pytest.mark.parallel, pytest.mark.cpu, pytest.mark.core_model]

_NUM_LAYERS = 2
_HEADS = 2
_HEAD_DIM = 8
_ARCH: dict[str, object] = {
    "num_layers": _NUM_LAYERS,
    "token_refiner_num_layers": 1,
    "hidden_size": 16,
    "num_attention_heads": _HEADS,
    "attention_head_dim": _HEAD_DIM,
    "ffn_hidden_size": 32,
    "latents_dim": 2,
    "audio_latents_dim": 2,
    "patch_size": (1, 2, 2),
    "text_dim": 6,
    "timestep_input_dim": 4,
    "time_embed_hidden_size": 16,
    "time_embed_dim": 8,
    "adaln_out_features": 18 * 16,
    "final_adaln_out_features": 2 * 16,
    "rope_inv_freq_len": 2,
}


class _NoAttention(nn.Module):
    def __init__(self, **kwargs):
        del kwargs
        super().__init__()

    def add_schedule_candidate_check(self, check):
        del check


class _TextEncoder(nn.Module):
    """Stand-in for a non-DiT weight source, loaded through the ordinary path."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2, bias=False)

    def load_weights(self, weights):
        loaded = set()
        params = dict(self.named_parameters())
        for name, tensor in weights:
            params[name].data.copy_(tensor)
            loaded.add(name)
        return loaded


@pytest.fixture(scope="module")
def cpu_process_group():
    if dist.is_initialized():
        yield
        return
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=get_file_store_init_method())
    try:
        yield
    finally:
        dist.destroy_process_group()
        gc.collect()


def _patched_h3(stack: ExitStack) -> None:
    from tests.model_executor.helpers import bootstrap_vllm_layer_custom_op_modules

    bootstrap_vllm_layer_custom_op_modules()

    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear

    from vllm_omni.diffusion.distributed import hsdp
    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_blocks as blocks
    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as h3

    stack.enter_context(set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))))
    for module in (linear, parameter):
        stack.enter_context(patch.object(module, "get_tensor_model_parallel_rank", return_value=0))
        stack.enter_context(patch.object(module, "get_tensor_model_parallel_world_size", return_value=1))
    stack.enter_context(patch.object(h3, "get_tensor_model_parallel_world_size", return_value=1))
    stack.enter_context(patch.object(blocks, "Attention", _NoAttention))
    stack.enter_context(
        patch.object(hsdp, "get_world_group", return_value=SimpleNamespace(world_size=1, rank_in_group=0))
    )
    stack.enter_context(patch.object(hsdp, "current_omni_platform", SimpleNamespace(device_type="cpu")))


def _write_checkpoint(model: nn.Module, directory: Path, *, diffusers: bool) -> None:
    """Serialize the reference DiT the way the matching release stores it."""
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import _reorder_grouped_qkv_to_qkv

    heads, head_dim = _HEADS, _HEAD_DIM
    state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    if not diffusers:
        for name in [n for n in state if n.endswith(".attn.qkv_proj.weight")]:
            # Native H3 stores Q/K/V grouped per head; invert the runtime reorder.
            runtime = state[name]
            q, k, v = runtime.split(heads * head_dim)
            grouped = torch.stack(
                [
                    torch.cat(
                        (
                            q[i * head_dim : (i + 1) * head_dim],
                            k[i * head_dim : (i + 1) * head_dim],
                            v[i * head_dim : (i + 1) * head_dim],
                        )
                    )
                    for i in range(heads)
                ]
            ).reshape_as(runtime)
            assert torch.equal(
                _reorder_grouped_qkv_to_qkv(grouped, num_query_groups=heads, heads_per_group=1, head_dim=head_dim),
                runtime,
            )
            state[name] = grouped
        save_file({k: v.contiguous() for k, v in state.items()}, str(directory / "model.safetensors"))
        return

    native_to_diffusers = {
        "attn.out_proj": "attn.to_out.0",
        "attn.q_norm": "attn.norm_q",
        "attn.k_norm": "attn.norm_k",
        "mlp.fc2": "ff.net.2",
        "adaln_proj.linear": "adaln_proj.linear",
        "norm1": "norm1",
        "norm2": "norm2",
    }
    model_level = {
        "video_patch_proj": "proj_in",
        "final_layer.video_out": "proj_out",
        "audio_patch_proj": "audio_proj_in",
        "final_layer.audio_out": "audio_proj_out",
        "condition_proj": "context_embedder",
        "time_embedder.proj_in": "time_embedder.linear_1",
        "time_embedder.proj_out": "time_embedder.linear_2",
        "final_layer.adaln_proj.linear": "norm_out.linear",
        "final_layer.norm": "norm_out.norm",
        "token_refiner.final_norm": "token_refiner.final_norm",
    }
    out = {}
    for name, tensor in state.items():
        if name == "rope.inv_freq":
            continue  # Diffusers reconstructs RoPE from config.
        module, _, kind = name.rpartition(".")
        if module in model_level:
            out[f"{model_level[module]}.{kind}"] = tensor
            continue
        for native_prefix, diffusers_prefix in (
            ("blocks.", "transformer_blocks."),
            ("token_refiner.blocks.", "token_refiner.refiner_blocks."),
        ):
            if not module.startswith(native_prefix) or (
                native_prefix == "blocks." and module.startswith("token_refiner.")
            ):
                continue
            index, _, suffix = module[len(native_prefix) :].partition(".")
            base = f"{diffusers_prefix}{index}."
            if suffix == "attn.qkv_proj":
                for part, chunk in zip("qkv", tensor.chunk(3), strict=True):
                    out[f"{base}attn.to_{part}.{kind}"] = chunk
            elif suffix == "mlp.fc1":
                gate, up = tensor.chunk(2)
                out[f"{base}ff.net.0.proj.{kind}"] = torch.cat((up, gate))  # Diffusers stores [up, gate].
            else:
                out[f"{base}{native_to_diffusers[suffix]}.{kind}"] = tensor
            break
        else:
            if not module.startswith(("blocks.", "token_refiner.")):
                raise AssertionError(f"unmapped tensor {name}")
    save_file({k: v.contiguous() for k, v in out.items()}, str(directory / "model.safetensors"))


def _pipeline(diffusers: bool, checkpoint: Path, encoder_dir: Path, strategy: str, *, quant_config=None, device="cpu"):
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    config = SimpleNamespace(
        tf_model_config=dict(_ARCH),
        dtype=torch.bfloat16,
        quantization_config=quant_config,
        lora_path=None,
        cache_config={"minimax_h3_adaln_cache": False},
        num_weight_load_threads=1,
        hsdp_weight_load_strategy=strategy,
        parallel_config=SimpleNamespace(
            use_hsdp=True, hsdp_replicate_size=1, hsdp_shard_size=1, tensor_parallel_size=1, ulysses_degree=1
        ),
    )
    pipeline = object.__new__(MiniMaxH3Pipeline)
    nn.Module.__init__(pipeline)
    pipeline.od_config = config
    # Like DiffusersPipelineLoader._init_from_load_format: HSDP defers to CPU
    # only without a quant config; quantized models are built on the device.
    with torch.device(device):
        pipeline.transformer = MiniMaxH3DiTModel(config, quant_config, diffusers_weights=diffusers)
        pipeline.text_encoder = _TextEncoder()
    pipeline.transformers_ref = None
    pipeline.video_vae = None
    pipeline.audio_vae = None
    pipeline._vdn = None
    pipeline._regenerate = None
    pipeline._fasth3 = None
    pipeline._fasth3_checkpoint = None
    pipeline._dit_modules = ["transformer"]
    pipeline._encoder_modules = ["text_encoder"]
    pipeline._vae_modules = []
    pipeline.weights_sources = [
        DiffusersPipelineLoader.ComponentSource(str(checkpoint), None, None, "transformer.", False),
        DiffusersPipelineLoader.ComponentSource(str(encoder_dir), None, None, "text_encoder.", False),
    ]
    return pipeline, config


# Packed NVFP4 needs N % 128 == 0 and K % 64 == 0 per projection.
NVFP4_ARCH: dict[str, object] = {
    **_ARCH,
    "hidden_size": 128,
    "attention_head_dim": 64,
    "ffn_hidden_size": 256,
    "time_embed_hidden_size": 128,
    "time_embed_dim": 64,
    "adaln_out_features": 18 * 128,
    "final_adaln_out_features": 2 * 128,
}
NVFP4_PROJECTIONS = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out", "ff.fc_in", "ff.fc_out")
# Like the released FastH3 QAD export, only the FFN projections carry a static
# input scale. The value is not representable in BF16.
NVFP4_INPUT_SCALED = ("ff.fc_in", "ff.fc_out")
NVFP4_INPUT_SCALE = 427.9403076171875
_DIFFUSERS_NAMES = {"attn.to_out": "attn.to_out.0", "ff.fc_in": "ff.net.0.proj", "ff.fc_out": "ff.net.2"}

Packer = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor, float]]


def nvfp4_arch(stack: ExitStack) -> None:
    """Resize the reduced H3 so every block projection fits the packed NVFP4 tiles."""
    module = sys.modules[__name__]
    stack.enter_context(patch.dict(_ARCH, NVFP4_ARCH))
    stack.enter_context(patch.object(module, "_HEADS", NVFP4_ARCH["num_attention_heads"]))
    stack.enter_context(patch.object(module, "_HEAD_DIM", NVFP4_ARCH["attention_head_dim"]))


def write_qad_checkpoint(directory: Path, model: nn.Module, pack: Packer) -> None:
    """Write the FastH3 QAD layout: a BF16 index without block projections, plus nvfp4_weights.safetensors."""
    _write_checkpoint(model, directory, diffusers=True)
    state = load_file(str(directory / "model.safetensors"))
    packed = {}
    for index in range(_NUM_LAYERS):
        for projection in NVFP4_PROJECTIONS:
            source = f"transformer_blocks.{index}.{projection}"
            weight = state.pop(f"transformer_blocks.{index}.{_DIFFUSERS_NAMES.get(projection, projection)}.weight")
            qweight, scales, global_scale = pack(weight)
            packed[f"{source}::_nvfp4_weight"] = qweight
            packed[f"{source}::_nvfp4_weight_scale"] = scales
            packed[f"{source}::_nvfp4_alpha"] = torch.tensor(1 / global_scale, dtype=torch.float32)
            packed[f"{source}::_weight_global_sf"] = torch.tensor(global_scale, dtype=torch.bfloat16)
            if projection in NVFP4_INPUT_SCALED:
                packed[f"{source}::_nvfp4_input_global_sf"] = torch.tensor(NVFP4_INPUT_SCALE, dtype=torch.float32)
    (directory / "model.safetensors").unlink()
    shard = "diffusion_pytorch_model-00001-of-00001.safetensors"
    save_file(state, str(directory / shard))
    (directory / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(state, shard)})
    )
    save_file(packed, str(directory / "nvfp4_weights.safetensors"))


def add_qad_sidecar_source(pipeline, checkpoint: Path) -> None:
    """Use the same source list as MiniMaxH3Pipeline for a QAD checkpoint."""
    pipeline.weights_sources.insert(
        1,
        DiffusersPipelineLoader.ComponentSource(
            str(checkpoint), None, None, "transformer.", False, ["nvfp4_weights.safetensors"]
        ),
    )


def _random_pack(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Arbitrary packed bytes; loading never interprets them. The global scale is not BF16-exact."""
    rows, columns = weight.shape
    qweight = torch.randint(0, 256, (rows, columns // 2), dtype=torch.uint8)
    scales = torch.randint(0, 256, (rows, columns // 16), dtype=torch.uint8)
    return qweight, scales, 448 * 6 / 1.2345


@pytest.mark.parametrize("strategy", ["full", "pre_sharded"])
def test_minimax_h3_nvfp4_scales_stay_fp32_through_hsdp_gather(cpu_process_group, tmp_path, strategy):
    """FlashInfer's mm_fp4 needs an FP32 alpha; a BF16 HSDP gather would also round the scales."""
    from vllm.config.load import LoadConfig
    from vllm.utils.torch_utils import set_default_torch_dtype

    from vllm_omni.diffusion.models.minimax_h3.checkpoint_quantization import FastH3NVFP4Config, _Projection

    checkpoint, encoder_dir = tmp_path / "transformer", tmp_path / "text_encoder"
    with ExitStack() as stack:
        _patched_h3(stack)
        nvfp4_arch(stack)
        stack.enter_context(set_default_torch_dtype(torch.bfloat16))
        reference, _ = _pipeline(True, tmp_path, tmp_path, "full")
        checkpoint.mkdir()
        encoder_dir.mkdir()
        torch.manual_seed(0)
        write_qad_checkpoint(checkpoint, reference.transformer, _random_pack)
        save_file({"proj.weight": torch.ones(2, 2, dtype=torch.bfloat16)}, str(encoder_dir / "model.safetensors"))
        sidecar = load_file(str(checkpoint / "nvfp4_weights.safetensors"))

        quant = FastH3NVFP4Config.from_dit(checkpoint / "nvfp4_weights.safetensors", rank=0)
        pipeline, config = _pipeline(True, checkpoint, encoder_dir, strategy, quant_config=quant)
        add_qad_sidecar_source(pipeline, checkpoint)
        loader = DiffusersPipelineLoader(LoadConfig(), config)
        with patch.object(loader, "_init_from_load_format", return_value=pipeline):
            transformer = loader._load_model_with_hsdp(torch.device("cpu")).transformer

    checked = 0
    for block in transformer.blocks:
        block.unshard()
        try:
            for projection in block.modules():
                if not isinstance(projection, _Projection):
                    continue
                source = projection.spec.source
                expected = {"alpha": sidecar[f"{source}::_nvfp4_alpha"]}
                if f"{source}::_nvfp4_input_global_sf" in sidecar:
                    expected["input_scale"] = sidecar[f"{source}::_nvfp4_input_global_sf"]
                    checked += 1
                for field, value in expected.items():
                    gathered = getattr(projection, field)
                    assert gathered.dtype == torch.float32, (source, field, gathered.dtype)
                    torch.testing.assert_close(gathered, value.reshape(1), rtol=0, atol=0, msg=f"{source} {field}")
                assert (projection.alpha / projection.input_scale).dtype == torch.float32, source
        finally:
            block.reshard()
    assert checked == _NUM_LAYERS * len(NVFP4_INPUT_SCALED), checked


@pytest.mark.parametrize("diffusers", [False, True], ids=["native", "diffusers"])
def test_minimax_h3_pre_sharded_matches_full_hsdp(cpu_process_group, tmp_path, diffusers):
    from vllm.config.load import LoadConfig
    from vllm.utils.torch_utils import set_default_torch_dtype

    with ExitStack() as stack:
        _patched_h3(stack)
        stack.enter_context(set_default_torch_dtype(torch.bfloat16))
        reference, _ = _pipeline(diffusers, tmp_path, tmp_path, "full")
        torch.manual_seed(0)
        with torch.no_grad():
            for name, tensor in reference.transformer.named_parameters():
                tensor.copy_(torch.randn_like(tensor))
        checkpoint = tmp_path / "transformer"
        encoder_dir = tmp_path / "text_encoder"
        checkpoint.mkdir()
        encoder_dir.mkdir()
        _write_checkpoint(reference.transformer, checkpoint, diffusers=diffusers)
        encoder_weight = torch.arange(4, dtype=torch.bfloat16).reshape(2, 2)
        save_file({"proj.weight": encoder_weight}, str(encoder_dir / "model.safetensors"))

        results = {}
        for strategy in ("full", "pre_sharded"):
            pipeline, config = _pipeline(diffusers, checkpoint, encoder_dir, strategy)
            loader = DiffusersPipelineLoader(LoadConfig(), config)
            with patch.object(loader, "_init_from_load_format", return_value=pipeline):
                results[strategy] = loader._load_model_with_hsdp(torch.device("cpu"))

    full, pre_sharded = results["full"].transformer, results["pre_sharded"].transformer
    expected = dict(full.named_parameters())
    for name, actual in pre_sharded.named_parameters():
        want = expected[name]
        actual = actual.full_tensor() if isinstance(actual, DTensor) else actual
        want = want.full_tensor() if isinstance(want, DTensor) else want
        assert actual.dtype == want.dtype, name
        torch.testing.assert_close(actual, want, rtol=0, atol=0, msg=name)
    torch.testing.assert_close(pre_sharded.rope.inv_freq, full.rope.inv_freq, rtol=0, atol=0)
    # The loaded weights must equal the reference model that wrote the checkpoint.
    reference_state = dict(reference.transformer.named_parameters())
    for name, actual in pre_sharded.named_parameters():
        actual = actual.full_tensor() if isinstance(actual, DTensor) else actual
        torch.testing.assert_close(actual, reference_state[name].to(actual.dtype), rtol=0, atol=0, msg=name)
    # The non-DiT source is loaded through the ordinary path in both strategies.
    for strategy in ("full", "pre_sharded"):
        weight = results[strategy].text_encoder.proj.weight
        torch.testing.assert_close(weight, encoder_weight.to(weight.dtype), rtol=0, atol=0)
