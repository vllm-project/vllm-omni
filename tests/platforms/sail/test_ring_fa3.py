# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import importlib
import sys
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.attention.backends.ring import ring_globals, ring_kernels
from vllm_omni.diffusion.attention.parallel import ring

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("ppu_predicate", "uses_tuple_window"),
    [(None, False), (True, False), (True, True)],
)
def test_fa3_interface_detection(monkeypatch, ppu_predicate, uses_tuple_window):
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
            assert reloaded.SAIL_FA3_USES_TUPLE_WINDOW is uses_tuple_window
            assert reloaded.FA3_SUPPORTED_CUDA_MAJORS == (None if ppu_predicate else frozenset({9}))
    finally:
        importlib.reload(ring_globals)


@pytest.mark.parametrize(("has_ppu_fa3", "expected"), [(False, ring.AttnType.VLLM_FA), (True, ring.AttnType.FA3)])
def test_ring_flash_attention_priority(monkeypatch, has_ppu_fa3, expected):
    monkeypatch.setattr(ring, "_can_use_vllm_flash_attn", lambda _device: True)
    monkeypatch.setattr(ring, "_can_use_fa4", lambda _device: True)
    monkeypatch.setattr(ring, "_can_use_fa3", lambda _device: True)
    monkeypatch.setattr(ring, "_can_use_fa2", lambda _device: True)
    monkeypatch.setattr(ring, "HAS_PPU_FA3", has_ppu_fa3)
    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.diffusion.attention.backends.ring_flash_attn",
        SimpleNamespace(ring_flash_attn_func=lambda *args, **kwargs: kwargs["attn_type"]),
    )
    query = SimpleNamespace(dtype=torch.bfloat16, shape=(1, 16, 4, 128), device=torch.device("cuda:0"))
    strategy = ring.RingParallelAttention(SimpleNamespace(ring_group=object()))

    assert strategy.run_attention(query, query, query, None) is expected


@pytest.mark.parametrize("uses_tuple_window", [False, True])
@pytest.mark.parametrize("window", [None, (7, 2)])
def test_fa3_forward_passes_window_and_returns_lse(monkeypatch, uses_tuple_window, window):
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
    monkeypatch.setattr(ring_kernels, "SAIL_FA3_USES_TUPLE_WINDOW", uses_tuple_window)
    monkeypatch.setattr(ring_kernels, "fa3_fwd_func", fake_forward)

    assert ring_kernels.fa3_forward(query, key, value, 0.0, 0.125, True, window, None, None, False) == (output, lse)
