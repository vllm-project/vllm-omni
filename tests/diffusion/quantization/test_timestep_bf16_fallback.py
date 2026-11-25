# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for generic timestep BF16 fallback, streaming ring, and H3 offline MX integration."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch import nn
from vllm.model_executor.layers.linear import LinearMethodBase

from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import _reorder_grouped_qkv_to_qkv
from vllm_omni.diffusion.models.minimax_h3.native_mx import (
    H3NativeMXConfig,
    NativeMXLinearMethod,
    load_h3_bf16_prefix_weights,
)
from vllm_omni.diffusion.models.minimax_h3.offline_quantization import (
    iter_bf16_transformer_delta,
    resolve_offline_components,
)
from vllm_omni.diffusion.offloader.prefix_bf16_streamer import PrefixBF16FallbackManager
from vllm_omni.quantization.timestep_bf16_fallback import (
    BF16_FALLBACK_MANAGED_ATTR,
    BF16_FALLBACK_WEIGHT_ATTR,
    TimestepBF16FallbackLinearMethod,
    TimestepBF16Policy,
    attach_bf16_fallback_weight,
    unwrap_linear_method,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeQuantLinearMethod(LinearMethodBase):
    """Deterministic fake quantized linear method returning scaled output."""

    def __init__(self, scale_factor: float = 0.25) -> None:
        self.scale_factor = scale_factor
        self.quant_calls = 0

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size, params_dtype, extra_weight_attrs
        out_features = sum(output_partition_sizes)
        layer.register_parameter(
            "weight",
            nn.Parameter(torch.ones(out_features, input_size_per_partition, dtype=torch.uint8), requires_grad=False),
        )
        layer.register_parameter(
            "weight_scale",
            nn.Parameter(torch.ones(out_features, 1, dtype=torch.uint8), requires_grad=False),
        )

    def apply(self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        self.quant_calls += 1
        w = layer.weight.to(dtype=x.dtype) * self.scale_factor
        out = F.linear(x, w, bias)
        return out


class _WrappedLinear(nn.Module):
    """Minimal LinearBase-like module using any ``LinearMethodBase``."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        quant_method: LinearMethodBase,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.input_size_per_partition = in_features
        self.output_size_per_partition = out_features
        self.quant_method = quant_method
        self.quant_method.create_weights(
            self,
            input_size_per_partition=in_features,
            output_partition_sizes=[out_features],
            input_size=in_features,
            output_size=out_features,
            params_dtype=torch.bfloat16,
            weight_loader=self._default_loader,
        )
        self.bias = (
            nn.Parameter(torch.randn(out_features, dtype=torch.bfloat16) * 0.05, requires_grad=False) if bias else None
        )

    @staticmethod
    def _default_loader(param: nn.Parameter, loaded_weight: torch.Tensor, *args) -> None:
        if args and isinstance(args[0], int):
            # Simulate MergedColumnParallelLinear gate/up shard loading.
            shard_id = args[0]
            half = param.shape[0] // 2
            param.data[shard_id * half : (shard_id + 1) * half].copy_(loaded_weight)
            return
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quant_method.apply(self, x, self.bias)


class _ToyBlock(nn.Module):
    def __init__(self, dim: int, policy: TimestepBF16Policy) -> None:
        super().__init__()
        self.inner1 = _FakeQuantLinearMethod(scale_factor=0.1)
        self.inner2 = _FakeQuantLinearMethod(scale_factor=0.2)
        self.fc1 = _WrappedLinear(dim, dim * 2, TimestepBF16FallbackLinearMethod(self.inner1, policy))
        self.fc2 = _WrappedLinear(dim * 2, dim, TimestepBF16FallbackLinearMethod(self.inner2, policy))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(x)))


class _ToyDiT(nn.Module):
    _layerwise_offload_blocks_attrs = ["blocks"]

    def __init__(
        self, num_blocks: int, dim: int, policy: TimestepBF16Policy, *, include_out_proj: bool = False
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_ToyBlock(dim, policy) for _ in range(num_blocks)])
        self.head_inner = _FakeQuantLinearMethod(scale_factor=0.3) if include_out_proj else None
        self.head = (
            _WrappedLinear(dim, dim, TimestepBF16FallbackLinearMethod(self.head_inner, policy))
            if include_out_proj and self.head_inner is not None
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        if self.head is not None:
            x = self.head(x)
        return x


class _ToyPipeline(nn.Module):
    _dit_modules = ["transformer"]

    def __init__(self, transformer: _ToyDiT) -> None:
        super().__init__()
        self.transformer = transformer


def _attach_random_bf16(dit: _ToyDiT, seed: int = 123) -> dict[int, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    truth: dict[int, torch.Tensor] = {}
    for module in dit.modules():
        if isinstance(getattr(module, "quant_method", None), TimestepBF16FallbackLinearMethod):
            shape = module._bf16_fallback_weight_shape
            w = torch.randn(shape, generator=gen, dtype=torch.bfloat16) * 0.2
            truth[id(module)] = w.clone()
            attach_bf16_fallback_weight(module, w)
    return truth


def _run_bf16_reference(dit: _ToyDiT, truth: dict[int, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    out = x
    for block in dit.blocks:
        h = F.linear(out, truth[id(block.fc1)], block.fc1.bias)
        h = F.silu(h)
        out = F.linear(h, truth[id(block.fc2)], block.fc2.bias)
    if dit.head is not None:
        out = F.linear(out, truth[id(dit.head)], dit.head.bias)
    return out


def test_policy_validation_and_step_sets() -> None:
    p = TimestepBF16Policy.from_prefix(5, "stream")
    assert p.enabled
    assert p.prefix_steps == 5
    assert p.is_active(0) and p.is_active(4) and not p.is_active(5) and not p.is_active(None)

    p_arb = TimestepBF16Policy.from_steps([0, 1, 49], "stream")
    assert p_arb.prefix_steps == 0
    assert p_arb.is_active(0) and p_arb.is_active(49) and not p_arb.is_active(2)

    with pytest.raises(ValueError, match="contiguous"):
        TimestepBF16Policy.from_steps([0, 2], "dlo")
    with pytest.raises(ValueError, match="Invalid prefix_bf16_master_mode"):
        TimestepBF16Policy.from_prefix(3, "bogus")
    with pytest.raises(ValueError, match="non-negative"):
        TimestepBF16Policy.from_prefix(-1, "stream")


def test_wrapper_fail_closed_and_dlo_contract() -> None:
    policy_dlo = TimestepBF16Policy.from_prefix(3, "dlo")
    inner = _FakeQuantLinearMethod()
    wrapper = TimestepBF16FallbackLinearMethod(inner, policy_dlo)
    layer = _WrappedLinear(8, 12, wrapper)

    assert unwrap_linear_method(wrapper) is inner
    with pytest.raises(RuntimeError, match="without a materialized CPU BF16"):
        wrapper.get_layerwise_prefix_weights(layer)

    w_bf16 = torch.randn(12, 8, dtype=torch.bfloat16)
    attach_bf16_fallback_weight(layer, w_bf16)
    k, aux, replaced = wrapper.get_layerwise_prefix_weights(layer)
    assert k == 3
    assert set(aux.keys()) == {BF16_FALLBACK_WEIGHT_ATTR}
    assert set(replaced) == {"weight", "weight_scale"}

    # Missing weight on active step fails closed.
    setattr(layer, BF16_FALLBACK_WEIGHT_ATTR, None)
    x = torch.randn(2, 8, dtype=torch.bfloat16)
    with set_forward_context(denoise_step_idx=1):
        with pytest.raises(RuntimeError, match="no materialized BF16 weight"):
            layer(x)

    # Inactive step delegates directly to inner method even when BF16 weight is detached.
    with set_forward_context(denoise_step_idx=3):
        out = layer(x)
    assert out.shape == (2, 12)
    assert inner.quant_calls == 1


@pytest.mark.parametrize("num_blocks", [4, 3, 1])
def test_prefix_block_streamer_exact_parity_even_odd_and_multi_request(num_blocks: int) -> None:
    policy = TimestepBF16Policy.from_prefix(2, "stream")
    dit = _ToyDiT(num_blocks=num_blocks, dim=16, policy=policy, include_out_proj=True)
    truth = _attach_random_bf16(dit, seed=42 + num_blocks)
    pipe = _ToyPipeline(dit)

    manager = PrefixBF16FallbackManager(device=torch.device("cpu"))
    assert manager.enable(pipe)
    assert len(manager.streamers) == 1
    streamer = manager.streamers[0]

    # Streamed per-layer CPU masters must have been released; non-block head promoted to resident.
    for block in dit.blocks:
        assert getattr(block.fc1, BF16_FALLBACK_WEIGHT_ATTR) is None
        assert getattr(block.fc1, BF16_FALLBACK_MANAGED_ATTR) == "stream"
    assert dit.head is not None
    assert getattr(dit.head, BF16_FALLBACK_MANAGED_ATTR) == "resident"

    x = torch.randn(2, 16, dtype=torch.bfloat16)
    expected_bf16 = _run_bf16_reference(dit, truth, x)

    # Run 2 full requests back-to-back (steps 0..3 each) to verify multi-request reset.
    for _req in range(2):
        for step in range(4):
            prev_prefetches = streamer.prefetch_count
            with set_forward_context(denoise_step_idx=step):
                out = dit(x)
            if step < 2:
                assert torch.equal(out, expected_bf16)
                # Ensure views are detached after post_forward.
                for block in dit.blocks:
                    assert getattr(block.fc1, BF16_FALLBACK_WEIGHT_ATTR) is None
            else:
                # Zero prefetches on quantized steps!
                assert streamer.prefetch_count == prev_prefetches
                assert not torch.equal(out, expected_bf16)

    manager.disable()


def test_prefix_block_streamer_simulated_4rank_allgather() -> None:
    policy = TimestepBF16Policy.from_prefix(2, "stream")
    shard_size = 4
    dits = [_ToyDiT(num_blocks=4, dim=16, policy=policy) for _ in range(shard_size)]
    truth = _attach_random_bf16(dits[0], seed=99)
    # Give all 4 ranks identical initial BF16 weights before sharding.
    for rank in range(1, shard_size):
        for b0, br in zip(dits[0].blocks, dits[rank].blocks, strict=True):
            for l0, lr in ((b0.fc1, br.fc1), (b0.fc2, br.fc2)):
                lr.bias.data.copy_(l0.bias.data)
                attach_bf16_fallback_weight(lr, truth[id(l0)].clone())

    # First verify each rank holds only 1/4 shard and reconstructs via simulated AllGather.
    rank_shards: list[list[torch.Tensor]] = []
    managers: list[PrefixBF16FallbackManager] = []

    def _make_all_gather(rank: int):
        def _ag(dst_full: torch.Tensor, src_shard: torch.Tensor) -> None:
            # Look up matching block shards across all 4 ranks by length/identity.
            # Find which payload owns src_shard's values on this rank.
            streamer = managers[rank].streamers[0]
            matched_idx = None
            for p in streamer._payloads:
                if p.shard_numel == src_shard.numel() and torch.equal(p.cpu_shard, src_shard):
                    matched_idx = p.block_idx
                    break
            assert matched_idx is not None
            chunks = [managers[r].streamers[0]._payloads[matched_idx].cpu_shard for r in range(shard_size)]
            dst_full.copy_(torch.cat(chunks, dim=0))

        return _ag

    for rank in range(shard_size):
        mgr = PrefixBF16FallbackManager(
            device=torch.device("cpu"),
            shard_size=shard_size,
            shard_rank=rank,
            all_gather_fn=_make_all_gather(rank),
        )
        assert mgr.enable(_ToyPipeline(dits[rank]))
        managers.append(mgr)
        rank_shards.append([p.cpu_shard for p in mgr.streamers[0]._payloads])

    # Verify each rank only stores 1/4 of total elements per block.
    full_block_numel = truth[id(dits[0].blocks[0].fc1)].numel() + truth[id(dits[0].blocks[0].fc2)].numel()
    assert rank_shards[0][0].numel() == (full_block_numel + shard_size - 1) // shard_size

    x = torch.randn(2, 16, dtype=torch.bfloat16)
    expected_bf16 = _run_bf16_reference(dits[0], truth, x)

    for rank in range(shard_size):
        with set_forward_context(denoise_step_idx=0):
            out0 = dits[rank](x)
        with set_forward_context(denoise_step_idx=1):
            out1 = dits[rank](x)
        assert torch.equal(out0, expected_bf16)
        assert torch.equal(out1, expected_bf16)

    for mgr in managers:
        mgr.disable()


def test_h3_offline_prefix_config_and_delta_loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.minimax_h3.offline_quantization.resolve_npu_quant_family",
        lambda device=None: "mx",
    )
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1)
    # Build minimal synthetic offline MX checkpoint (1 MXFP4 block + 1 MXFP8 block + FLOAT adaln).
    quant_dir = tmp_path / "quant_transformer"
    bf16_dir = tmp_path / "bf16_transformer"
    quant_dir.mkdir()
    bf16_dir.mkdir()

    # Shapes: hidden=64, heads=2, head_dim=32 -> qkv_out=192 (3*64), fc1_out=128 (2*64)
    q_tensors = {
        "blocks.0.attn.qkv_proj.weight": torch.zeros((192, 32), dtype=torch.uint8),
        "blocks.0.attn.qkv_proj.weight_scale": torch.full((192, 2), 127, dtype=torch.uint8),
        "blocks.0.mlp.fc1.weight": torch.zeros((128, 32), dtype=torch.uint8),
        "blocks.0.mlp.fc1.weight_scale": torch.full((128, 2), 127, dtype=torch.uint8),
        "blocks.0.adaln_proj.linear.weight": torch.randn((32, 64), dtype=torch.bfloat16),
        "blocks.1.mlp.fc2.weight": torch.zeros((64, 64), dtype=torch.float8_e4m3fn),
        "blocks.1.mlp.fc2.weight_scale": torch.full((64, 2), 127, dtype=torch.uint8),
    }
    q_desc = {
        "blocks.0.attn.qkv_proj.weight": "W4A4_MXFP4",
        "blocks.0.attn.qkv_proj.weight_scale": "W4A4_MXFP4",
        "blocks.0.mlp.fc1.weight": "W4A4_MXFP4",
        "blocks.0.mlp.fc1.weight_scale": "W4A4_MXFP4",
        "blocks.0.adaln_proj.linear.weight": "FLOAT",
        "blocks.1.mlp.fc2.weight": "W8A8_MXFP8",
        "blocks.1.mlp.fc2.weight_scale": "W8A8_MXFP8",
    }
    save_file(q_tensors, str(quant_dir / "model.safetensors"))
    (quant_dir / "quant_model_description.json").write_text(json.dumps(q_desc), encoding="utf-8")

    # Original BF16 checkpoint contains full weights including unquantized adaln_proj
    # and FP32 boundary layers (e.g. time_embedder) that offline exporters may downcast.
    raw_qkv = torch.arange(192 * 64, dtype=torch.float32).reshape(192, 64).to(torch.bfloat16)
    raw_fc1 = torch.randn((128, 64), dtype=torch.bfloat16)
    raw_fc2 = torch.randn((64, 64), dtype=torch.bfloat16)
    raw_fp32_time = torch.tensor([0.12345678, -0.98765432, 0.31415926, 0.27182818], dtype=torch.float32)
    bf16_tensors = {
        "blocks.0.attn.qkv_proj.weight": raw_qkv,
        "blocks.0.mlp.fc1.weight": raw_fc1,
        "blocks.0.adaln_proj.linear.weight": torch.randn((32, 64), dtype=torch.bfloat16),
        "blocks.1.mlp.fc2.weight": raw_fc2,
        "time_embedder.proj_in.bias": raw_fp32_time,
    }
    save_file(bf16_tensors, str(bf16_dir / "model.safetensors"))

    # 1. Verify resolve_offline_components parses prefix config & filters w4_only vs all.
    ckpts_all, _ = resolve_offline_components(
        {
            "execution_mode": "native_mx",
            "prefix_bf16_steps": 5,
            "prefix_bf16_master_mode": "stream",
            "prefix_bf16_bits": "all",
            "prefix_bf16_source": str(bf16_dir),
            "components": {"transformer": str(quant_dir)},
        }
    )
    cfg_all = H3NativeMXConfig.from_checkpoint(ckpts_all["transformer"])
    assert cfg_all.prefix_target_modules() == {
        "blocks.0.attn.qkv_proj",
        "blocks.0.mlp.fc1",
        "blocks.1.mlp.fc2",
    }

    ckpts_w4, _ = resolve_offline_components(
        {
            "execution_mode": "native_mx",
            "prefix_bf16_steps": 5,
            "prefix_bf16_bits": "w4_only",
            "components": {"transformer": str(quant_dir)},
        }
    )
    cfg_w4 = H3NativeMXConfig.from_checkpoint(ckpts_w4["transformer"])
    assert cfg_w4.prefix_target_modules() == {"blocks.0.attn.qkv_proj", "blocks.0.mlp.fc1"}

    # Reject prefix under dequantize mode.
    with pytest.raises(ValueError, match="not execution_mode='dequantize'"):
        resolve_offline_components(
            {
                "execution_mode": "dequantize",
                "prefix_bf16_steps": 5,
                "components": {"transformer": str(quant_dir)},
            }
        )

    # 2. Verify iter_bf16_transformer_delta skips adaln_proj and streams only targets.
    delta_names = [name for name, _ in iter_bf16_transformer_delta(bf16_dir, cfg_all.prefix_target_modules())]
    assert set(delta_names) == {
        "blocks.0.attn.qkv_proj.weight",
        "blocks.0.mlp.fc1.weight",
        "blocks.1.mlp.fc2.weight",
    }

    # 3. Verify load_h3_bf16_prefix_weights reorders grouped QKV and splits fc1 identically to H3.
    class _MiniH3Transformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.arch = SimpleNamespace(num_attention_heads=2, attention_head_dim=32)
            policy = TimestepBF16Policy.from_prefix(5, "stream")
            blk0 = nn.Module()
            blk0.attn = nn.Module()
            blk0.attn.qkv_proj = _WrappedLinear(
                64,
                192,
                TimestepBF16FallbackLinearMethod(NativeMXLinearMethod(4), policy, quant_config=cfg_all),
                bias=False,
            )
            blk0.mlp = nn.Module()
            blk0.mlp.fc1 = _WrappedLinear(
                64,
                128,
                TimestepBF16FallbackLinearMethod(NativeMXLinearMethod(4), policy, quant_config=cfg_all),
                bias=False,
            )
            blk1 = nn.Module()
            blk1.mlp = nn.Module()
            blk1.mlp.fc2 = _WrappedLinear(
                64,
                64,
                TimestepBF16FallbackLinearMethod(NativeMXLinearMethod(8), policy, quant_config=cfg_all),
                bias=False,
            )
            self.blocks = nn.ModuleList([blk0, blk1])
            self.time_embedder = nn.Module()
            self.time_embedder.proj_in = nn.Module()
            # Simulate ModelSlim BF16-downcast loaded into an FP32 boundary parameter.
            self.time_embedder.proj_in.register_parameter(
                "bias",
                nn.Parameter(raw_fp32_time.to(torch.bfloat16).to(torch.float32), requires_grad=False),
            )

    mini = _MiniH3Transformer()
    assert not torch.equal(mini.time_embedder.proj_in.bias, raw_fp32_time)
    cfg_all.validate_model_targets(mini)
    loaded_count = load_h3_bf16_prefix_weights(mini, bf16_dir)
    assert loaded_count == 3

    expected_qkv = _reorder_grouped_qkv_to_qkv(raw_qkv, num_query_groups=2, heads_per_group=1, head_dim=32)
    assert torch.equal(getattr(mini.blocks[0].attn.qkv_proj, BF16_FALLBACK_WEIGHT_ATTR), expected_qkv)
    assert torch.equal(getattr(mini.blocks[0].mlp.fc1, BF16_FALLBACK_WEIGHT_ATTR), raw_fc1)
    assert torch.equal(getattr(mini.blocks[1].mlp.fc2, BF16_FALLBACK_WEIGHT_ATTR), raw_fc2)
    assert torch.equal(mini.time_embedder.proj_in.bias, raw_fp32_time)


def test_offline_prefix_dlo_validation_in_od_config() -> None:
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    with pytest.raises(ValueError, match="prefix_bf16_master_mode='dlo' requires distributed layerwise offload"):
        OmniDiffusionConfig(
            model="dummy-model",
            additional_config={
                "offline_quantization": {
                    "execution_mode": "native_mx",
                    "prefix_bf16_steps": 5,
                    "prefix_bf16_master_mode": "dlo",
                    "components": {"transformer": "/tmp/nonexistent"},
                }
            },
        )

    # Stream mode does NOT require DLO.
    cfg = OmniDiffusionConfig(
        model="dummy-model",
        additional_config={
            "offline_quantization": {
                "execution_mode": "native_mx",
                "prefix_bf16_steps": 5,
                "prefix_bf16_master_mode": "stream",
                "components": {"transformer": "/tmp/nonexistent"},
            }
        },
    )
    assert cfg.additional_config["offline_quantization"]["prefix_bf16_master_mode"] == "stream"


def test_scheme_b_dlo_mode_end_to_end_with_distributed_layerwise_hook(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify TimestepBF16FallbackLinearMethod plugs directly into DistributedLayerwiseOffloadBackend."""
    import vllm_omni.diffusion.offloader.distributed_layerwise_backend as backend_module
    from tests.diffusion.offloader.helpers import patch_offload_runtime
    from vllm_omni.diffusion.offloader.base import OffloadConfig, OffloadStrategy
    from vllm_omni.diffusion.offloader.distributed_layerwise_backend import DistributedLayerwiseOffloadBackend

    patch_offload_runtime(monkeypatch, backend_module.current_omni_platform, synchronize=True)

    policy = TimestepBF16Policy.from_prefix(2, "dlo")
    dit = _ToyDiT(num_blocks=3, dim=16, policy=policy)
    truth = _attach_random_bf16(dit, seed=77)
    pipe = _ToyPipeline(dit)
    x = torch.randn(2, 16, dtype=torch.bfloat16)
    expected_bf16 = _run_bf16_reference(dit, truth, x)

    # Before DLO claims the layers, PrefixBF16FallbackManager in dlo mode must fail closed.
    with pytest.raises(RuntimeError, match="not claimed by DistributedLayerwiseOffloadBackend"):
        PrefixBF16FallbackManager(device=torch.device("cpu")).enable(pipe)

    # Now enable DLO backend on the block ring.
    backend = DistributedLayerwiseOffloadBackend(
        OffloadConfig(strategy=OffloadStrategy.DISTRIBUTED_LAYER_WISE, pin_cpu_memory=False, dlo_use_allgather=False),
        torch.device("cpu"),
    )
    hooks = backend._install_hook_group(dit.blocks, "dit")
    buffers = backend._allocate_shared_buffers(hooks)
    tracker = [-1, -1]
    for hook in hooks:
        hook.gpu_buffers = buffers
        hook._shared_slot_group = tracker
        hook._group_id = 0

    # After DLO claims the layers, manager validates and succeeds.
    mgr = PrefixBF16FallbackManager(device=torch.device("cpu"))
    assert mgr.enable(pipe)

    for step in (0, 1, 2, 3, 0, 2):
        with set_forward_context(denoise_step_idx=step):
            out = dit(x)
        if step < 2:
            assert torch.equal(out, expected_bf16)
        else:
            assert not torch.equal(out, expected_bf16)
    mgr.disable()
