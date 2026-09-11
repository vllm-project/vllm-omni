# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType

import pytest
import torch

from vllm_omni.diffusion.models.hunyuan_image3 import layers as selected_layers
from vllm_omni.diffusion.models.hunyuan_image3.layers.native.transformer_blocks import (
    ResBlock as NativeResBlock,
)
from vllm_omni.diffusion.models.hunyuan_image3.layers.nvidia import (
    transformer_blocks as nvidia_blocks,
)
from vllm_omni.diffusion.models.hunyuan_image3.layers.nvidia.transformer_blocks import (
    ResBlock as NvidiaResBlock,
)
from vllm_omni.platforms import current_omni_platform

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    pytest.mark.cuda,
    pytest.mark.skipif(
        not current_omni_platform.is_cuda(),
        reason="HunyuanImage3 NVIDIA ResBlock qualification requires CUDA",
    ),
]

_SEED = 20260911
_TOLERANCES = {
    torch.float32: (2e-5, 2e-5),
    torch.float16: (5e-3, 5e-3),
    torch.bfloat16: (2e-2, 2e-2),
}
_FUSED_OP_MODULE_NAMES = (
    "vllm_omni.model_executor.models.common.ops.fused_group_norm_silu",
    "vllm_omni.model_executor.models.common.ops.fused_adaptive_group_norm_silu",
)


@dataclass(frozen=True)
class BlockCase:
    name: str
    batch: int
    in_channels: int
    out_channels: int
    height: int
    width: int
    use_conv: bool = False
    up: bool = False
    down: bool = False


_CASES = (
    BlockCase("identity", 1, 128, 128, 16, 16),
    BlockCase("batched_identity", 2, 128, 128, 32, 32),
    BlockCase("channel_change", 1, 128, 256, 16, 16),
    BlockCase("convolutional_skip", 1, 128, 256, 16, 16, use_conv=True),
    BlockCase("production_down_branch", 1, 128, 128, 32, 32, down=True),
    BlockCase("production_up_branch", 1, 128, 128, 16, 16, up=True),
)


def _load_fused_op_modules() -> tuple[ModuleType, ...]:
    return tuple(importlib.import_module(name) for name in _FUSED_OP_MODULE_NAMES)


@pytest.fixture(scope="module", autouse=True)
def _require_triton_fused_ops() -> None:
    unavailable = [module.__name__ for module in _load_fused_op_modules() if not getattr(module, "HAS_TRITON", False)]
    assert unavailable == [], (
        "HunyuanImage3 NVIDIA ResBlock qualification must execute the Triton "
        f"implementations, but fallback is active for: {', '.join(unavailable)}"
    )


def _constructor_kwargs(case: BlockCase, dtype: torch.dtype) -> dict[str, object]:
    return {
        "in_channels": case.in_channels,
        "emb_channels": 512,
        "out_channels": case.out_channels,
        "dropout": 0.0,
        "use_conv": case.use_conv,
        "dims": 2,
        "up": case.up,
        "down": case.down,
        "device": torch.device("cuda"),
        "dtype": dtype,
    }


def _randomize_parameters(module: torch.nn.Module, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(_SEED)
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            values = torch.randn(
                parameter.shape,
                generator=generator,
                device=parameter.device,
                dtype=torch.float32,
            )
            if parameter.ndim == 1 and name.endswith("weight"):
                values = 1.0 + values * 0.05
            else:
                values = values * 0.02
            parameter.copy_(values.to(dtype=dtype))
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _make_blocks(case: BlockCase, dtype: torch.dtype) -> tuple[NativeResBlock, NvidiaResBlock]:
    kwargs = _constructor_kwargs(case, dtype)
    native = NativeResBlock(**kwargs).eval()
    nvidia = NvidiaResBlock(**kwargs).eval()
    state = _randomize_parameters(native, dtype)
    incompatible = nvidia.load_state_dict(state, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    return native, nvidia


def _make_inputs(case: BlockCase, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(_SEED + 1)
    x = torch.randn(
        case.batch,
        case.in_channels,
        case.height,
        case.width,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    ).to(dtype)
    emb = torch.randn(
        case.batch,
        512,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    ).to(dtype)
    return x, emb


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
def test_resblock_state_dict_contract(case: BlockCase) -> None:
    native, nvidia = _make_blocks(case, torch.float32)

    assert tuple(native.state_dict()) == tuple(nvidia.state_dict())
    required = {
        "in_layers.0.weight",
        "in_layers.0.bias",
        "in_layers.2.weight",
        "in_layers.2.bias",
        "emb_layers.1.weight",
        "emb_layers.1.bias",
        "out_layers.0.weight",
        "out_layers.0.bias",
        "out_layers.3.weight",
        "out_layers.3.bias",
    }
    assert required <= set(native.state_dict())
    if case.in_channels != case.out_channels:
        assert "skip_connection.weight" in native.state_dict()


@pytest.mark.parametrize("dtype", tuple(_TOLERANCES), ids=lambda dtype: str(dtype).removeprefix("torch."))
@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
def test_native_and_nvidia_resblocks_match(case: BlockCase, dtype: torch.dtype) -> None:
    native, nvidia = _make_blocks(case, dtype)
    x, emb = _make_inputs(case, dtype)

    with torch.inference_mode():
        native_output = native(x, emb)
        nvidia_output = nvidia(x, emb)
        skip_output = native.skip_connection(x)

    assert native_output.shape == nvidia_output.shape
    assert native_output.dtype == nvidia_output.dtype == dtype
    assert torch.isfinite(native_output).all()
    assert torch.isfinite(nvidia_output).all()
    assert (native_output - skip_output).abs().max().item() > 1e-4

    rtol, atol = _TOLERANCES[dtype]
    torch.testing.assert_close(nvidia_output, native_output, rtol=rtol, atol=atol)


@pytest.mark.parametrize("autocast_dtype", (torch.float16, torch.bfloat16))
def test_native_and_nvidia_resblocks_match_under_autocast(autocast_dtype: torch.dtype) -> None:
    case = _CASES[1]
    native, nvidia = _make_blocks(case, torch.float32)
    x, emb = _make_inputs(case, torch.float32)

    with torch.inference_mode(), torch.autocast("cuda", dtype=autocast_dtype):
        native_output = native(x, emb)
        nvidia_output = nvidia(x, emb)

    assert native_output.shape == nvidia_output.shape
    assert native_output.dtype == nvidia_output.dtype
    rtol, atol = _TOLERANCES[autocast_dtype]
    torch.testing.assert_close(nvidia_output, native_output, rtol=rtol, atol=atol)


def test_cuda_dispatch_selects_nvidia_resblock() -> None:
    assert selected_layers.ResBlock is NvidiaResBlock


def test_nvidia_resblock_calls_both_fused_operations(monkeypatch: pytest.MonkeyPatch) -> None:
    case = _CASES[0]
    native, nvidia = _make_blocks(case, torch.float32)
    x, emb = _make_inputs(case, torch.float32)
    calls = {"group_norm": 0, "adaptive_group_norm": 0}
    original_group_norm = nvidia_blocks.fused_group_norm_silu
    original_adaptive = nvidia_blocks.fused_adaptive_group_norm_silu

    def counted_group_norm(*args: object, **kwargs: object) -> torch.Tensor:
        calls["group_norm"] += 1
        return original_group_norm(*args, **kwargs)

    def counted_adaptive(*args: object, **kwargs: object) -> torch.Tensor:
        calls["adaptive_group_norm"] += 1
        return original_adaptive(*args, **kwargs)

    monkeypatch.setattr(nvidia_blocks, "fused_group_norm_silu", counted_group_norm)
    monkeypatch.setattr(nvidia_blocks, "fused_adaptive_group_norm_silu", counted_adaptive)

    with torch.inference_mode():
        nvidia(x, emb)
        assert calls == {"group_norm": 1, "adaptive_group_norm": 1}
        native(x, emb)
        assert calls == {"group_norm": 1, "adaptive_group_norm": 1}


@pytest.mark.parametrize("benchmark,deterministic", ((False, False), (True, False)))
def test_nvidia_resblock_restores_cudnn_state(benchmark: bool, deterministic: bool) -> None:
    case = _CASES[0]
    _, nvidia = _make_blocks(case, torch.float32)
    x, emb = _make_inputs(case, torch.float32)
    original_benchmark = torch.backends.cudnn.benchmark
    original_deterministic = torch.backends.cudnn.deterministic

    try:
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = deterministic
        with torch.inference_mode():
            nvidia(x, emb)
        assert torch.backends.cudnn.benchmark is benchmark
        assert torch.backends.cudnn.deterministic is deterministic
    finally:
        torch.backends.cudnn.benchmark = original_benchmark
        torch.backends.cudnn.deterministic = original_deterministic


def test_nvidia_resblock_restores_cudnn_state_after_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class SentinelError(RuntimeError):
        pass

    case = _CASES[0]
    _, nvidia = _make_blocks(case, torch.float32)
    x, emb = _make_inputs(case, torch.float32)
    original_benchmark = torch.backends.cudnn.benchmark
    original_deterministic = torch.backends.cudnn.deterministic

    def fail_group_norm(*args: object, **kwargs: object) -> torch.Tensor:
        raise SentinelError("forced fused GroupNorm failure")

    monkeypatch.setattr(nvidia_blocks, "fused_group_norm_silu", fail_group_norm)

    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        with pytest.raises(SentinelError, match="forced fused GroupNorm failure"), torch.inference_mode():
            nvidia(x, emb)
        assert torch.backends.cudnn.benchmark is False
        assert torch.backends.cudnn.deterministic is False
    finally:
        torch.backends.cudnn.benchmark = original_benchmark
        torch.backends.cudnn.deterministic = original_deterministic
