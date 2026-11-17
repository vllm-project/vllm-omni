# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tests for the per-solve loop-invariant hoist in the Qwen2.5-Omni DiT
flow-matching solve (``DiTEvalInvariants`` in ``qwen2_5_omni_token2wav.py``).

An RK4 solve evaluates the DiT ``(num_steps - 1) * 4`` times with a fixed
code sequence, reference mel and mel length, so the codec text embedding, the
ECAPA speaker encoding of the reference mel and the rotary tables are
identical in every eval. The hoist computes them once per solve and threads
them through ``forward``. These tests pin down the two properties that make
it safe: bit-exactness against the legacy per-eval path (single eval and
whole solve, CFG and non-CFG), and exactly-once computation per solve.
"""

import socket

import pytest
import torch
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniDiTConfig,
)

from tests.helpers.mark import hardware_test
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model]

_requires_cuda = pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="needs CUDA")

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


def _eager_sample(model, inputs, num_steps=4, guidance_scale=0.5):
    """Run the eager solve by disabling the graph wrapper for one call."""
    cond, ref_mel, code, y0 = inputs
    wrapper = model._solve_graph_wrapper
    assert wrapper is not None
    wrapper.enabled = False
    try:
        return model.fast_block_sample(
            conditioning_vector=cond,
            reference_mel_spectrogram=ref_mel,
            quantized_code=code,
            y0=y0,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            sway_coefficient=-1.0,
        )
    finally:
        wrapper.enabled = True


def _time_points(num_steps, device, dtype, sway_coefficient=-1.0):
    """Replicate ``fast_block_sample``'s time grid (needed for a like-for-like
    comparison against a hand-built legacy ODE function)."""
    t = torch.linspace(0, 1, num_steps, device=device, dtype=dtype)
    if sway_coefficient is not None:
        t = t + sway_coefficient * (torch.cos(torch.pi / 2 * t) - 1 + t)
    return t


@pytest.fixture(scope="module")
def dit_model():
    from vllm.config import (
        DeviceConfig,
        VllmConfig,
        get_current_vllm_config_or_none,
        set_current_vllm_config,
    )
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )

    # Guarded so this file can run in the same pytest process as
    # test_dit_solve_cudagraph.py, whose fixture performs the same setup.
    if not torch.distributed.is_initialized():
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            local_rank=0,
        )
    if get_current_vllm_config_or_none() is None:
        set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cuda"))).__enter__()
    if not model_parallel_is_initialized():
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
    # vLLM 0.33's QKVParallelLinear leaves weights uninitialized (torch.empty;
    # load_weights normally fills them) and HF-style _init_weights never touches
    # vLLM layers, so the tiny model can start with NaN weights. Re-seed every
    # parameter so the random-weight model is finite and deterministic.
    torch.manual_seed(4321)
    with torch.no_grad():
        for param in model.parameters():
            torch.nn.init.normal_(param, mean=0.0, std=0.02)
    return model


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@_requires_cuda
@pytest.mark.parametrize("apply_cfg", [True, False])
def test_single_eval_bit_exact_with_invariants(dit_model, apply_cfg):
    """One DiT eval with precomputed invariants must equal the legacy path."""
    cond, ref_mel, code, y0 = _make_inputs()
    time_step = torch.tensor(0.3, device=y0.device)
    speaker_embedding = cond.unsqueeze(1).repeat(1, y0.shape[1], 1)

    invariants = dit_model._build_eval_invariants(ref_mel, code, y0, apply_cfg=apply_cfg)
    with torch.no_grad():
        out_legacy = dit_model(
            hidden_states=y0,
            condition_vector=ref_mel,
            speaker_embedding=speaker_embedding,
            quantized_code=code,
            time_step=time_step,
            drop_audio_conditioning=False,
            drop_code=False,
            apply_cfg=apply_cfg,
        )
        out_hoisted = dit_model(
            hidden_states=y0,
            condition_vector=ref_mel,
            speaker_embedding=speaker_embedding,
            quantized_code=code,
            time_step=time_step,
            drop_audio_conditioning=False,
            drop_code=False,
            apply_cfg=apply_cfg,
            eval_invariants=invariants,
        )
    assert torch.equal(out_legacy, out_hoisted)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@_requires_cuda
@pytest.mark.parametrize("guidance_scale", [0.5, 0.0])
def test_hoisted_solve_bit_exact_vs_legacy_ode_function(dit_model, guidance_scale):
    """The whole hoisted eager solve must equal a hand-built legacy (per-eval
    recomputation) ODE function integrated over the same time grid."""
    from vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_token2wav import (
        _integrate_rk4,
    )

    cond, ref_mel, code, y0 = _make_inputs(seed=7)
    num_steps = 4
    time_points = _time_points(num_steps, y0.device, y0.dtype)
    speaker_embedding = cond.unsqueeze(1).repeat(1, y0.shape[1], 1)

    def legacy_ode_function(time_step, hidden_states):
        out = dit_model(
            hidden_states=hidden_states,
            condition_vector=ref_mel,
            speaker_embedding=speaker_embedding,
            quantized_code=code,
            time_step=time_step,
            apply_cfg=guidance_scale >= 1e-5,
        )
        if guidance_scale < 1e-5:
            return out
        guided, null = torch.chunk(out, 2, dim=0)
        return guided + (guided - null) * guidance_scale

    with torch.no_grad():
        out_legacy = _integrate_rk4(legacy_ode_function, y0, time_points)
        out_hoisted = _eager_sample(
            dit_model, (cond, ref_mel, code, y0), num_steps=num_steps, guidance_scale=guidance_scale
        )
    assert torch.equal(out_legacy.permute(0, 2, 1), out_hoisted)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@_requires_cuda
@pytest.mark.parametrize(
    ("guidance_scale", "expected_spk", "expected_text", "expected_rotary"),
    [(0.5, 1, 2, 1), (0.0, 1, 1, 1)],
)
def test_invariants_computed_once_per_solve(
    dit_model, monkeypatch, guidance_scale, expected_spk, expected_text, expected_rotary
):
    """A 4-step RK4 solve runs 12 evals; the hoisted path must run the ECAPA
    encoder, codec embedding and rotary table exactly once (vs 12/24/12 in
    the legacy path, asserted here as a control for the counters)."""
    calls = {"spk": 0, "text": 0, "rotary": 0}

    spk = dit_model.input_embed.spk_encoder
    text = dit_model.text_embed
    rotary = dit_model.rotary_embed

    def counting(module, key):
        original = module.forward

        def wrapper(*args, **kwargs):
            calls[key] += 1
            return original(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(spk, "forward", counting(spk, "spk"))
    monkeypatch.setattr(text, "forward", counting(text, "text"))
    monkeypatch.setattr(rotary, "forward", counting(rotary, "rotary"))

    inputs = _make_inputs(seed=11)
    with torch.no_grad():
        _eager_sample(dit_model, inputs, num_steps=4, guidance_scale=guidance_scale)
    assert calls == {"spk": expected_spk, "text": expected_text, "rotary": expected_rotary}

    # Control: the legacy per-eval path really does recompute per eval, so
    # the counters above measure what they claim.
    from vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_token2wav import (
        _integrate_rk4,
    )

    cond, ref_mel, code, y0 = inputs
    speaker_embedding = cond.unsqueeze(1).repeat(1, y0.shape[1], 1)
    apply_cfg = guidance_scale >= 1e-5
    calls.update(spk=0, text=0, rotary=0)
    time_points = _time_points(4, y0.device, y0.dtype)

    def legacy_ode_function(time_step, hidden_states):
        out = dit_model(
            hidden_states=hidden_states,
            condition_vector=ref_mel,
            speaker_embedding=speaker_embedding,
            quantized_code=code,
            time_step=time_step,
            apply_cfg=apply_cfg,
        )
        if not apply_cfg:
            return out
        guided, null = torch.chunk(out, 2, dim=0)
        return guided + (guided - null) * guidance_scale

    with torch.no_grad():
        _integrate_rk4(legacy_ode_function, y0, time_points)
    assert calls["spk"] == 12
    assert calls["text"] == (24 if apply_cfg else 12)
    assert calls["rotary"] == 12


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@_requires_cuda
def test_graph_replay_with_hoist_is_bit_exact(dit_model):
    """The whole-solve CUDA graph (captured around the hoisted solve) must
    still be bit-exact against the eager hoisted solve."""
    inputs = _make_inputs(seed=21)
    dit_model._solve_graph_wrapper.entries.clear()
    with torch.no_grad():
        out_graph = dit_model.fast_block_sample(
            conditioning_vector=inputs[0],
            reference_mel_spectrogram=inputs[1],
            quantized_code=inputs[2],
            y0=inputs[3],
            num_steps=4,
            guidance_scale=0.5,
            sway_coefficient=-1.0,
        )
    with torch.no_grad():
        out_eager = _eager_sample(dit_model, inputs, num_steps=4, guidance_scale=0.5)
    assert torch.equal(out_graph, out_eager)
    dit_model._solve_graph_wrapper.entries.clear()
