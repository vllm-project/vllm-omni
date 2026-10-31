# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tests for the whole-solve CUDA graph wrapper around the Qwen2.5-Omni DiT
flow-matching solve (``dit_solve_graph_wrapper.py``).

Graph replay must be bit-exact against the eager solve, consume fresh inputs
on every replay (no stale static buffers), key entries on exact input shapes,
fall back to eager when disabled / cache-full / mid-capture, and tolerate
capture and replay running under different autograd contexts.
"""

import socket

import pytest
import torch
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniDiTConfig,
)

from vllm_omni.model_executor.models.qwen2_5_omni.dit_solve_graph_wrapper import (
    DEFAULT_MAX_ENTRIES,
    DiTSolveCUDAGraphWrapper,
    dit_solve_cudagraph_enabled,
    dit_solve_cudagraph_max_entries,
)
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model]

_requires_cuda = pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="needs CUDA")


# ---------------------------------------------------------------------------
# Env-var handling (CPU)
# ---------------------------------------------------------------------------


@pytest.mark.cpu
@pytest.mark.parametrize("value", ["0", "false", "off", "FALSE", " 0 "])
def test_env_flag_disables(monkeypatch, value):
    monkeypatch.setenv("VLLM_OMNI_QWEN2_5_DIT_SOLVE_CUDAGRAPH", value)
    assert not dit_solve_cudagraph_enabled()


@pytest.mark.cpu
def test_env_flag_defaults_on(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_QWEN2_5_DIT_SOLVE_CUDAGRAPH", raising=False)
    assert dit_solve_cudagraph_enabled()


@pytest.mark.cpu
def test_env_max_entries(monkeypatch):
    name = "VLLM_OMNI_QWEN2_5_DIT_SOLVE_CUDAGRAPH_MAX_ENTRIES"
    monkeypatch.delenv(name, raising=False)
    assert dit_solve_cudagraph_max_entries() == DEFAULT_MAX_ENTRIES
    monkeypatch.setenv(name, "3")
    assert dit_solve_cudagraph_max_entries() == 3
    monkeypatch.setenv(name, "0")  # clamped to at least one entry
    assert dit_solve_cudagraph_max_entries() == 1
    monkeypatch.setenv(name, "not-a-number")
    assert dit_solve_cudagraph_max_entries() == DEFAULT_MAX_ENTRIES


@pytest.mark.cpu
def test_solve_skips_cpu_tensors():
    wrapper = DiTSolveCUDAGraphWrapper(build_ode_fn=lambda *args: None, integrate_fn=lambda *args: None)
    mel = torch.zeros(1, 8, 8)
    assert (
        wrapper.solve(
            torch.zeros(1, 32),
            mel,
            torch.zeros(1, 8, dtype=torch.long),
            mel,
            torch.linspace(0, 1, 4),
            0.5,
        )
        is None
    )


# ---------------------------------------------------------------------------
# Graph capture / replay against the real DiT (CUDA)
# ---------------------------------------------------------------------------

_MEL_DIM = 8
_EMB_DIM = 32
_ENC_EMB_DIM = 192  # Qwen2_5OmniDiTConfig.enc_emb_dim default (talker hidden)
_NUM_EMBEDS = 100


def _make_inputs(t_code=24, t_ref=40, seed=42):
    """Deterministic solve inputs; ``t_mel = t_code * repeats = t_code * 2``."""
    device = "cuda"
    torch.manual_seed(seed)
    cond = torch.randn(1, _ENC_EMB_DIM, device=device)
    ref_mel = torch.randn(1, t_ref, _MEL_DIM, device=device)
    code = torch.randint(0, _NUM_EMBEDS - 10, (1, t_code), device=device, dtype=torch.long)
    y0 = torch.randn(1, t_code * 2, _MEL_DIM, device=device)
    return cond, ref_mel, code, y0


def _sample(model, inputs):
    cond, ref_mel, code, y0 = inputs
    return model.fast_block_sample(
        conditioning_vector=cond,
        reference_mel_spectrogram=ref_mel,
        quantized_code=code,
        y0=y0,
        num_steps=4,
        guidance_scale=0.5,
        sway_coefficient=-1.0,
    )


def _eager_sample(model, inputs):
    """Run the eager solve by disabling the graph wrapper for one call."""
    wrapper = model._solve_graph_wrapper
    assert wrapper is not None
    wrapper.enabled = False
    try:
        return _sample(model, inputs)
    finally:
        wrapper.enabled = True


@pytest.fixture(scope="module")
def dit_model():
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        local_rank=0,
    )
    set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cuda"))).__enter__()
    initialize_model_parallel(tensor_model_parallel_size=1)

    from vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_token2wav import (
        Qwen2_5OmniToken2WavDiTModel,
    )

    config = Qwen2_5OmniDiTConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=16,
        ff_mult=2,
        emb_dim=_EMB_DIM,
        block_size=8,
        look_ahead_layers=(0,),
        look_backward_layers=(0,),
        repeats=2,
        num_embeds=_NUM_EMBEDS,
        mel_dim=_MEL_DIM,
        dropout=0.0,
    )
    torch.manual_seed(1234)
    model = Qwen2_5OmniToken2WavDiTModel(config).to("cuda").eval()
    assert model._solve_graph_wrapper is not None  # enabled by default
    return model


@_requires_cuda
def test_graph_replay_is_bit_exact_and_captures_one_entry(dit_model):
    inputs = _make_inputs()
    with torch.no_grad():
        out_graph = _sample(dit_model, inputs)
    assert len(dit_model._solve_graph_wrapper.entries) == 1
    with torch.no_grad():
        out_eager = _eager_sample(dit_model, inputs)
    assert torch.equal(out_graph, out_eager)


@_requires_cuda
def test_graph_replay_consumes_new_inputs(dit_model):
    """Replays must read the freshly copied statics, not stale capture values."""
    inputs_a = _make_inputs(seed=1)
    inputs_b = _make_inputs(seed=2)
    with torch.no_grad():
        out_a = _sample(dit_model, inputs_a)  # captures
        out_b = _sample(dit_model, inputs_b)  # replays
    assert not torch.equal(out_a, out_b)
    with torch.no_grad():
        assert torch.equal(out_b, _eager_sample(dit_model, inputs_b))


@_requires_cuda
def test_distinct_shapes_get_distinct_entries(dit_model):
    dit_model._solve_graph_wrapper.entries.clear()
    for t_code in (24, 32):
        inputs = _make_inputs(t_code=t_code)
        with torch.no_grad():
            out_graph = _sample(dit_model, inputs)
        with torch.no_grad():
            out_eager = _eager_sample(dit_model, inputs)
        assert torch.equal(out_graph, out_eager)
    assert len(dit_model._solve_graph_wrapper.entries) == 2
    dit_model._solve_graph_wrapper.entries.clear()


@_requires_cuda
def test_full_cache_falls_back_to_eager(dit_model):
    wrapper = dit_model._solve_graph_wrapper
    wrapper.entries.clear()
    wrapper.max_entries = 1
    try:
        with torch.no_grad():
            first = _sample(dit_model, _make_inputs(t_code=24))
        assert len(wrapper.entries) == 1
        # A second shape cannot be captured; the solve must still be correct.
        inputs = _make_inputs(t_code=40)
        with torch.no_grad():
            second = _sample(dit_model, inputs)
        with torch.no_grad():
            second_eager = _eager_sample(dit_model, inputs)
        assert torch.equal(second, second_eager)
        assert len(wrapper.entries) == 1
        assert not torch.equal(first, second)
    finally:
        wrapper.max_entries = DEFAULT_MAX_ENTRIES
        wrapper.entries.clear()


@_requires_cuda
def test_capture_and_replay_under_mixed_autograd_contexts(dit_model):
    """Capture under inference_mode, replay under no_grad (and vice versa).

    The statics must be normal tensors regardless of the context that
    triggered the capture: in-place updates to inference tensors outside
    inference mode are not allowed.
    """
    dit_model._solve_graph_wrapper.entries.clear()
    inputs = _make_inputs(seed=7)
    with torch.inference_mode():
        out_inference = _sample(dit_model, inputs)  # capture happens here
    with torch.no_grad():
        out_no_grad = _sample(dit_model, inputs)  # replay under no_grad
    assert torch.equal(out_inference, out_no_grad)
    with torch.inference_mode():
        out_inference_2 = _sample(dit_model, _make_inputs(seed=8))
    with torch.no_grad():
        out_no_grad_2 = _sample(dit_model, _make_inputs(seed=8))
    assert torch.equal(out_inference_2, out_no_grad_2)
    dit_model._solve_graph_wrapper.entries.clear()


@_requires_cuda
def test_solve_falls_back_to_eager_inside_outer_graph_capture(dit_model, monkeypatch):
    wrapper = dit_model._solve_graph_wrapper
    wrapper.entries.clear()
    inputs = _make_inputs(t_code=48)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    try:
        with torch.no_grad():
            out_capturing = _sample(dit_model, inputs)
    finally:
        monkeypatch.undo()
    assert len(wrapper.entries) == 0  # no capture attempted mid-capture
    with torch.no_grad():
        out_eager = _eager_sample(dit_model, inputs)
    assert torch.equal(out_capturing, out_eager)
    wrapper.entries.clear()
