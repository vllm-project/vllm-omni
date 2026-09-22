# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The causal flow draws its initial noise from a fixed, position-indexed buffer."""

import types

import pytest
import torch
from omegaconf import DictConfig
from torch import nn

from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.cfm import (
    _FIXED_NOISE_FRAMES,
    CausalConditionalCFM,
)
from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav import CosyVoice3Code2Wav

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

CFM_PARAMS = DictConfig(
    {
        "sigma_min": 1e-06,
        "solver": "euler",
        "t_scheduler": "cosine",
        "training_cfg_rate": 0.2,
        "inference_cfg_rate": 0.7,
    }
)


class _Estimator(nn.Module):
    """Returns something input-dependent so noise differences reach the output."""

    def forward(self, x, mask, mu, t, spks, cond):
        return x * 0.5 + mu


def _cfm() -> CausalConditionalCFM:
    return CausalConditionalCFM(
        in_channels=240, cfm_params=CFM_PARAMS, n_spks=1, spk_emb_dim=80, estimator=_Estimator()
    )


def test_buffer_matches_upstream_seed_zero_draw():
    """Upstream: ``set_all_random_seed(0); torch.randn([1, 80, 50 * 300])``."""
    cfm = _cfm()
    state = torch.random.get_rng_state()
    try:
        torch.manual_seed(0)
        expected = torch.randn([1, 80, 50 * 300])
    finally:
        torch.random.set_rng_state(state)
    assert torch.equal(cfm.rand_noise, expected)
    assert "rand_noise" not in cfm.state_dict()  # not a checkpoint key


def test_default_offset_is_the_buffer_prefix():
    cfm = _cfm()
    mu = torch.zeros(2, 80, 96)
    z = cfm.fixed_noise(mu, temperature=0.7)
    assert z.shape == mu.shape
    assert torch.equal(z[0], cfm.rand_noise[0, :, :96] * 0.7)
    assert torch.equal(z[1], z[0])


def test_left_context_reuses_the_noise_it_was_generated_with():
    """Two consecutive windows of one stream agree on their overlapping positions."""
    cfm = _cfm()
    prompt = 10
    # First chunk: tokens [0, 30) after the prompt, no left context.
    first = cfm.fixed_noise(torch.zeros(1, 80, prompt + 30), prompt_len=prompt, noise_offset=0)
    # Second chunk resends [20, 30) as left context and adds [30, 45).
    second = cfm.fixed_noise(torch.zeros(1, 80, prompt + 25), prompt_len=prompt, noise_offset=20)
    assert torch.equal(second[..., :prompt], first[..., :prompt])  # prompt is always position 0
    assert torch.equal(second[..., prompt : prompt + 10], first[..., prompt + 20 : prompt + 30])
    # And the new frames continue the buffer instead of restarting it.
    assert torch.equal(second[0, :, prompt + 10 :], cfm.rand_noise[0, :, prompt + 30 : prompt + 45])


def test_per_row_offsets_and_wraparound():
    cfm = _cfm()
    mu = torch.zeros(2, 80, 8)
    z = cfm.fixed_noise(mu, prompt_len=2, noise_offset=torch.tensor([0, _FIXED_NOISE_FRAMES - 4]))
    assert torch.equal(z[0], cfm.rand_noise[0, :, :8])
    assert torch.equal(z[1, :, :2], cfm.rand_noise[0, :, :2])
    wrapped = torch.cat([cfm.rand_noise[0, :, -2:], cfm.rand_noise[0, :, :4]], dim=-1)
    assert torch.equal(z[1, :, 2:], wrapped)


def test_forward_is_deterministic_and_does_not_touch_global_rng():
    cfm = _cfm()
    mu = torch.randn(1, 80, 40)
    mask = torch.ones(1, 1, 40)
    spks = torch.randn(1, 80)
    cond = torch.randn(1, 80, 40)
    torch.manual_seed(123)
    before = torch.randn(3)
    torch.manual_seed(123)
    out1, _ = cfm(mu, mask, n_timesteps=2, spks=spks, cond=cond, prompt_len=4, noise_offset=6)
    after = torch.randn(3)
    out2, _ = cfm(mu, mask, n_timesteps=2, spks=spks, cond=cond, prompt_len=4, noise_offset=6)
    assert torch.equal(out1, out2)
    assert torch.equal(before, after)  # the fixed noise consumed no global randomness
    out3, _ = cfm(mu, mask, n_timesteps=2, spks=spks, cond=cond, prompt_len=4, noise_offset=7)
    assert not torch.equal(out1, out3)


def _stubbed_code2wav(calls: list):
    model = object.__new__(CosyVoice3Code2Wav)
    nn.Module.__init__(model)
    model.flow_model = types.SimpleNamespace(token_mel_ratio=2, pre_lookahead_len=3)

    def fake_forward_mel(self, token, prompt_token, prompt_feat, embedding, **kw):
        calls.append(kw["noise_offset_tokens"])
        return torch.zeros(1, 80, 2 * (int(token.shape[1]) - 3 - int(kw["token_offset_tokens"])))

    def fake_hift(self, feat, *, cache_state=None, finalize=False):
        return feat, (None if finalize else {"mel": feat})

    model._forward_mel = types.MethodType(fake_forward_mel, model)
    model._stream_hift_from_feat = types.MethodType(fake_hift, model)
    return model


def test_streaming_tracks_the_absolute_offset_across_windows():
    """emitted-so-far minus the resent left context is the first token's position."""
    calls: list = []
    model = _stubbed_code2wav(calls)
    common = dict(prompt_token=torch.zeros(1, 2, dtype=torch.int32), prompt_feat=torch.zeros(1, 4, 80))
    common["embedding"] = torch.zeros(1, 192)

    # Chunk 1: 15 new tokens + 3 lookahead, nothing resent.
    _, state = model.forward_streaming(token=torch.zeros(1, 18, dtype=torch.int32), **common, token_offset_tokens=0)
    assert state["flow_emitted_tokens"] == 15
    # Chunk 2: resend all 15 as left context (window not yet full), 30 new.
    _, state = model.forward_streaming(
        token=torch.zeros(1, 48, dtype=torch.int32), **common, cache_state=state, token_offset_tokens=15
    )
    assert state["flow_emitted_tokens"] == 45
    # Chunk 3: window of 25 slides; the first resent token is absolute 20.
    _, state = model.forward_streaming(
        token=torch.zeros(1, 28, dtype=torch.int32), **common, cache_state=state, token_offset_tokens=25
    )
    assert calls == [0, 0, 20]
    assert state["flow_emitted_tokens"] == 45  # 28 - 3 lookahead - 25 context: no new tokens yet
    # Finalize releases the lookahead and returns no state.
    _, state = model.forward_streaming(
        token=torch.zeros(1, 28, dtype=torch.int32),
        **common,
        cache_state=state,
        token_offset_tokens=25,
        finalize=True,
    )
    assert calls[-1] == 20
    assert state is None
