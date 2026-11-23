# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import importlib
import sys
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.attention.backends.ring import ring_globals, ring_kernels
from vllm_omni.diffusion.attention.parallel import factory, ring

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def disable_vllm_fa_ring_path(monkeypatch):
    """Keep fallback-selection tests independent of the installed vLLM wheel."""
    monkeypatch.setattr(ring, "_can_use_vllm_flash_attn", lambda _device: False)


@pytest.mark.parametrize(
    ("ppu_predicate", "uses_tuple_window"),
    [(None, False), (True, False), (True, True)],
)
def test_source_build_fa3_interface_follows_platform(monkeypatch, ppu_predicate, uses_tuple_window):
    def tuple_window_forward(*args, window_size=(-1, -1)):
        return None

    fake_interface = SimpleNamespace(
        _flash_attn_forward=tuple_window_forward if uses_tuple_window else lambda *args, **kwargs: None,
        flash_attn_func=lambda *args, **kwargs: None,
    )
    is_ppu = None if ppu_predicate is None else lambda: ppu_predicate
    try:
        with monkeypatch.context() as context:
            context.setitem(sys.modules, "flash_attn_interface", fake_interface)
            context.setattr("vllm.platforms.current_platform", SimpleNamespace(is_ppu=is_ppu))
            reloaded = importlib.reload(ring_globals)
            assert reloaded.HAS_FA3
            assert reloaded.HAS_PPU_FA3 is bool(ppu_predicate)
            assert reloaded.FA3_USES_TUPLE_WINDOW is uses_tuple_window
            assert reloaded.FA3_SUPPORTED_CUDA_MAJORS == (None if ppu_predicate else frozenset({9}))
    finally:
        importlib.reload(ring_globals)


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        ((8, 0), True),
        ((8, 9), True),
        ((9, 0), True),
        ((10, 0), False),
        ((10, 3), False),
        ((12, 0), False),
    ],
)
def test_can_use_fa3_checks_device_arch(monkeypatch, capability, expected):
    monkeypatch.setattr(ring, "HAS_FA3", True)
    monkeypatch.setattr(ring, "FA3_SUPPORTED_CUDA_MAJORS", frozenset({8, 9}))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: capability)

    assert ring._can_use_fa3(torch.device("cuda")) is expected


def test_can_use_fa3_preserves_unknown_extension_contract(monkeypatch):
    monkeypatch.setattr(ring, "HAS_FA3", True)
    monkeypatch.setattr(ring, "FA3_SUPPORTED_CUDA_MAJORS", None)

    assert ring._can_use_fa3(torch.device("cuda"))


@pytest.mark.parametrize(("has_ppu_fa3", "expected"), [(False, "vllm_fa"), (True, "fa3")])
def test_ring_prefers_platform_flash_attention(monkeypatch, has_ppu_fa3, expected):
    captured = {}

    def fake_ring_flash_attn_func(*args, **kwargs):
        captured.update(kwargs)
        return "output"

    monkeypatch.setattr(ring, "_can_use_vllm_flash_attn", lambda _device: True)
    monkeypatch.setattr(ring, "_can_use_fa4", lambda _device: True)
    monkeypatch.setattr(ring, "_can_use_fa3", lambda _device: True)
    monkeypatch.setattr(ring, "_can_use_fa2", lambda _device: True)
    monkeypatch.setattr(ring, "HAS_PPU_FA3", has_ppu_fa3)
    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.diffusion.attention.backends.ring_flash_attn",
        SimpleNamespace(ring_flash_attn_func=fake_ring_flash_attn_func),
    )
    query = SimpleNamespace(
        dtype=torch.bfloat16,
        shape=(1, 16, 4, 128),
        device=torch.device("cuda:0"),
    )

    strategy = ring.RingParallelAttention(SimpleNamespace(ring_group=object()))

    assert strategy.run_attention(query, "key", "value", None) == "output"
    assert captured["attn_type"].value == expected


@pytest.mark.parametrize("uses_tuple_window", [False, True])
@pytest.mark.parametrize("window", [None, (7, 2)])
def test_fa3_forward_uses_platform_signature(monkeypatch, uses_tuple_window, window):
    query, key, value = object(), object(), object()
    output, lse = object(), object()
    left, right = window or (-1, -1)

    def fake_forward(*args, **kwargs):
        assert kwargs.pop("softmax_scale") == 0.125
        assert kwargs.pop("causal") is True
        assert kwargs.pop("softcap") == 0.0
        if uses_tuple_window:
            assert not args
            assert kwargs.pop("q") is query
            assert kwargs.pop("k") is key
            assert kwargs.pop("v") is value
            assert kwargs.pop("window_size") == (left, right)
            assert len(kwargs) == 20
            assert all(argument is None for argument in kwargs.values())
        else:
            assert args == (query, key, value)
            assert kwargs == {"window_size_left": left, "window_size_right": right}
        return output, lse, None, None

    monkeypatch.setattr(ring_kernels, "HAS_FA3", True)
    monkeypatch.setattr(ring_kernels, "FA3_USES_TUPLE_WINDOW", uses_tuple_window)
    monkeypatch.setattr(ring_kernels, "fa3_fwd_func", fake_forward)

    assert ring_kernels.fa3_forward(query, key, value, 0.0, 0.125, True, window, None, None, False) == (output, lse)


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        ((8, 0), True),
        ((9, 0), True),
        ((10, 0), False),
        ((10, 3), False),
        ((12, 0), False),
    ],
)
def test_can_use_fa2_checks_device_arch(monkeypatch, capability, expected):
    monkeypatch.setattr(ring, "HAS_FLASH_ATTN", True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: capability)

    assert ring._can_use_fa2(torch.device("cuda")) is expected


def test_ring_falls_back_to_sdpa_when_fa3_extension_has_no_device_kernel(monkeypatch):
    monkeypatch.setattr(ring, "HAS_FA3", True)
    monkeypatch.setattr(ring, "HAS_FA4", False)
    monkeypatch.setattr(ring, "HAS_FLASH_ATTN", False)
    monkeypatch.setattr(ring, "HAS_AITER", False)
    monkeypatch.setattr(ring, "_can_use_fa3", lambda _device: False)

    expected = torch.randn(1, 2, 1, 8)

    def fake_ring_sdpa(*args, **kwargs):
        return expected

    module_name = "vllm_omni.diffusion.attention.backends.ring_pytorch_attn"
    monkeypatch.setitem(sys.modules, module_name, SimpleNamespace(ring_pytorch_attn_func=fake_ring_sdpa))

    strategy = ring.RingParallelAttention(SimpleNamespace(ring_group=object()))
    query = torch.randn(1, 2, 1, 8, dtype=torch.bfloat16)
    actual = strategy.run_attention(query, query, query, attn_metadata=None)

    assert actual is expected


def test_explicit_ring_backend_does_not_fallback_to_sdpa(monkeypatch):
    monkeypatch.setattr(ring, "HAS_FA3", True)
    monkeypatch.setattr(ring, "HAS_FA4", False)
    monkeypatch.setattr(ring, "HAS_FLASH_ATTN", False)
    monkeypatch.setattr(ring, "HAS_AITER", False)
    monkeypatch.setattr(ring, "_can_use_fa3", lambda _device: False)

    strategy = ring.RingParallelAttention(
        SimpleNamespace(ring_group=object()),
        attn_backend_pref="FLASH_ATTN",
        attn_backend_explicit=True,
    )
    query = torch.randn(1, 2, 1, 8, dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="explicitly selected"):
        strategy.run_attention(query, query, query, attn_metadata=None)


@pytest.mark.parametrize(
    "backend_pref",
    [
        "CUDNN_ATTN",
        "FLASH_ATTN_HUB",
        "FLASH_ATTN_3_HUB",
    ],
)
def test_explicit_non_ring_backend_is_rejected(monkeypatch, backend_pref):
    # Local FA is available so a missing rejection would silently run FA4/FA3/FA2
    # instead of the requested Hub/cuDNN backend.
    monkeypatch.setattr(ring, "HAS_FA4", True)
    monkeypatch.setattr(ring, "_can_use_fa4", lambda _device: True)
    module_name = "vllm_omni.diffusion.attention.backends.ring_pytorch_attn"
    monkeypatch.setitem(
        sys.modules,
        module_name,
        SimpleNamespace(ring_pytorch_attn_func=lambda *args, **kwargs: pytest.fail("unexpected SDPA fallback")),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.diffusion.attention.backends.ring_flash_attn",
        SimpleNamespace(ring_flash_attn_func=lambda *args, **kwargs: pytest.fail("unexpected local FA ring")),
    )
    strategy = ring.RingParallelAttention(
        SimpleNamespace(ring_group=object()),
        attn_backend_pref=backend_pref,
        attn_backend_explicit=True,
    )
    query = torch.randn(1, 2, 1, 8, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="ring sequence parallelism has no implementation"):
        strategy.run_attention(query, query, query, attn_metadata=None)


def test_explicit_flash_ring_rejects_float32_instead_of_falling_back(monkeypatch):
    module_name = "vllm_omni.diffusion.attention.backends.ring_pytorch_attn"
    monkeypatch.setitem(
        sys.modules,
        module_name,
        SimpleNamespace(ring_pytorch_attn_func=lambda *args, **kwargs: pytest.fail("unexpected SDPA fallback")),
    )
    strategy = ring.RingParallelAttention(
        SimpleNamespace(ring_group=object()),
        attn_backend_pref="FLASH_ATTN",
        attn_backend_explicit=True,
    )
    query = torch.randn(1, 2, 1, 8, dtype=torch.float32)

    with pytest.raises(ValueError, match="does not support float32"):
        strategy.run_attention(query, query, query, attn_metadata=None)


def test_pytorch_ring_flash_op_rejects_float32():
    query = torch.randn(1, 2, 1, 8, dtype=torch.float32)

    with pytest.raises(ValueError, match="does not support float32"):
        ring_kernels.pytorch_attn_forward(query, query, query, op_type="flash")


def test_configured_sp_does_not_fallback_when_group_is_unavailable(monkeypatch):
    parallel_config = SimpleNamespace(ulysses_degree=2, ring_degree=1, allgather_degree=1)
    forward_context = SimpleNamespace(omni_diffusion_config=SimpleNamespace(parallel_config=parallel_config))
    monkeypatch.setattr(factory, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(factory, "get_forward_context", lambda: forward_context)
    monkeypatch.setattr(
        factory,
        "get_sp_group",
        lambda: (_ for _ in ()).throw(RuntimeError("SP group is not initialized")),
    )

    with pytest.raises(RuntimeError, match="SP is configured"):
        factory.build_parallel_attention_strategy(scatter_idx=2, gather_idx=1, use_sync=False)


def test_configured_sp_rejects_single_rank_group(monkeypatch):
    parallel_config = SimpleNamespace(ulysses_degree=2, ring_degree=1, allgather_degree=1)
    forward_context = SimpleNamespace(omni_diffusion_config=SimpleNamespace(parallel_config=parallel_config))
    monkeypatch.setattr(factory, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(factory, "get_forward_context", lambda: forward_context)
    monkeypatch.setattr(factory, "get_sp_group", lambda: SimpleNamespace())
    monkeypatch.setattr(factory, "get_sequence_parallel_world_size", lambda: 1)

    with pytest.raises(RuntimeError, match="world size is not greater than one"):
        factory.build_parallel_attention_strategy(scatter_idx=2, gather_idx=1, use_sync=False)
