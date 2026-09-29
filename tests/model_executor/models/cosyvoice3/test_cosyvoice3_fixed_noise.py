# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The causal flow draws its initial noise from a fixed, position-indexed buffer."""

from collections import defaultdict
from dataclasses import dataclass, field

import pytest
import torch
from omegaconf import DictConfig
from torch import nn

from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.cfm import (
    _FIXED_NOISE_FRAMES,
    CausalConditionalCFM,
)
from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav import CosyVoice3Code2Wav
from vllm_omni.model_executor.stage_input_processors.cosyvoice3 import talker2code2wav_async_chunk

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
    z = cfm.fixed_noise(torch.zeros(2, 80, 8), prompt_len=2, noise_offset=torch.tensor([0, _FIXED_NOISE_FRAMES - 4]))
    # Offset 0 is the plain buffer prefix, as upstream slices it.
    assert torch.equal(z[0], cfm.rand_noise[0, :, :8])
    assert torch.equal(z[1, :, :2], cfm.rand_noise[0, :, :2])
    wrapped = torch.cat([cfm.rand_noise[0, :, -2:], cfm.rand_noise[0, :, :4]], dim=-1)
    assert torch.equal(z[1, :, 2:], wrapped)


def test_scalar_offset_and_dtype_apply_to_every_row():
    cfm = _cfm()
    z = cfm.fixed_noise(torch.zeros(2, 80, 6, dtype=torch.bfloat16), noise_offset=3, temperature=0.7)
    expected = cfm.rand_noise[0, :, 3:9].to(torch.bfloat16) * 0.7
    assert z.dtype == torch.bfloat16
    assert torch.equal(z[0], expected)
    assert torch.equal(z[1], expected)


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


PRE_LOOKAHEAD = 3


class _FlowStub(nn.Module):
    token_mel_ratio = 2
    pre_lookahead_len = PRE_LOOKAHEAD


@dataclass
class _Connector:
    config: dict[str, dict[str, int]]


@dataclass
class _TransferManager:
    connector: _Connector
    code_prompt_token_ids: defaultdict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    request_payload: dict[str, dict[str, object]] = field(default_factory=dict)


@dataclass
class _Request:
    external_req_id: str
    output_token_ids: list[int] = field(default_factory=list)
    additional_information: dict[str, object] = field(default_factory=dict)

    def is_finished(self) -> bool:
        return False


def _stubbed_code2wav(calls: list[int | torch.Tensor]) -> CosyVoice3Code2Wav:
    """Code2wav with the flow and HiFT replaced by shape-only fakes."""
    model = object.__new__(CosyVoice3Code2Wav)
    nn.Module.__init__(model)
    model.flow_model = _FlowStub()

    def fake_forward_mel(token, prompt_token, prompt_feat, embedding, **kw):
        calls.append(kw["noise_offset_tokens"])
        lookahead = 0
        if not kw["finalize"]:
            lookahead = PRE_LOOKAHEAD
        valid = int(token.shape[1]) - lookahead - int(kw["token_offset_tokens"])
        return torch.zeros(int(token.shape[0]), 80, 2 * valid)

    def fake_hift(feat, *, cache_state=None, finalize=False):
        if finalize:
            return feat, None
        return feat, {"mel": feat}

    model._forward_mel = fake_forward_mel
    model._stream_hift_from_feat = fake_hift
    return model


def _prompt(batch: int = 1) -> dict[str, torch.Tensor]:
    return {
        "prompt_token": torch.zeros(batch, 2, dtype=torch.int32),
        "prompt_feat": torch.zeros(batch, 4, 80),
        "embedding": torch.zeros(batch, 192),
    }


def test_streaming_tracks_the_absolute_offset_across_windows():
    """emitted-so-far minus the resent left context is the first token's position."""
    calls: list[int | torch.Tensor] = []
    model = _stubbed_code2wav(calls)

    # Chunk 1: 15 new tokens + 3 lookahead, nothing resent.
    _, state = model.forward_streaming(
        token=torch.zeros(1, 18, dtype=torch.int32),
        **_prompt(),
        token_offset_tokens=0,
    )
    assert state is not None
    assert state["flow_emitted_tokens"] == 15
    # Chunk 2: resend all 15 as left context (window not yet full), 30 new.
    _, state = model.forward_streaming(
        token=torch.zeros(1, 48, dtype=torch.int32),
        **_prompt(),
        cache_state=state,
        token_offset_tokens=15,
    )
    assert state is not None
    assert state["flow_emitted_tokens"] == 45
    # Chunk 3: window of 25 slides; the first resent token is absolute 20.
    _, state = model.forward_streaming(
        token=torch.zeros(1, 28, dtype=torch.int32),
        **_prompt(),
        cache_state=state,
        token_offset_tokens=25,
    )
    assert state is not None
    assert calls == [0, 0, 20]
    assert state["flow_emitted_tokens"] == 45
    # Finalize releases the lookahead and returns no state.
    _, state = model.forward_streaming(
        token=torch.zeros(1, 28, dtype=torch.int32),
        **_prompt(),
        cache_state=state,
        token_offset_tokens=25,
        finalize=True,
    )
    assert calls[-1] == 20
    assert state is None


def test_batched_streaming_tracks_offsets_per_row():
    calls: list[int | torch.Tensor] = []
    model = _stubbed_code2wav(calls)
    items = [
        # Fresh stream: 10 new tokens + lookahead.
        {"token": torch.zeros(1, 13, dtype=torch.int32), "cache_state": None, "token_offset_tokens": 0},
        # 40 emitted, window resends the last 25, then 5 new tokens + lookahead.
        {
            "token": torch.zeros(1, 33, dtype=torch.int32),
            "cache_state": {"flow_emitted_tokens": 40},
            "token_offset_tokens": 25,
        },
    ]
    results = model.forward_streaming_batch([{**item, **_prompt()} for item in items])

    assert len(calls) == 1  # both rows went through one batched flow call
    assert isinstance(calls[0], torch.Tensor)
    assert calls[0].tolist() == [0, 15]
    assert [state["flow_emitted_tokens"] for _, state in results] == [10, 45]


def test_code2wav_offset_matches_the_processor_window():
    """The offset code2wav rebuilds from its state is the processor's window start."""
    transfer_manager = _TransferManager(
        connector=_Connector(
            config={
                "extra": {
                    "codec_chunk_frames": 4,
                    "codec_pre_lookahead_frames": PRE_LOOKAHEAD,
                    "codec_max_chunk_frames": 8,
                    "codec_stream_scale_factor": 2,
                    "codec_left_context_frames": 5,
                    "codec_vocab_size": 6561,
                },
            },
        ),
    )
    total = 40
    # Token ``k`` of the stream has value ``k + 1``, so a payload's first code
    # tells which absolute position the processor's window starts at.
    request = _Request(external_req_id="rid")
    calls: list[int | torch.Tensor] = []
    model = _stubbed_code2wav(calls)
    state = None
    window_starts: list[int] = []
    emitted_frames = 0
    for count in range(1, total + 1):
        request.output_token_ids = list(range(1, count + 1))
        payload = talker2code2wav_async_chunk(transfer_manager, None, request, is_finished=count == total)
        if payload is None or payload.codes.audio.numel() == 0:
            continue
        window_starts.append(int(payload.codes.audio[0]) - 1)
        speech, state = model.forward_streaming(
            token=payload.codes.audio.to(torch.int32).unsqueeze(0),
            **_prompt(),
            cache_state=state,
            token_offset_tokens=int(payload.meta.left_context_size),
            finalize=bool(payload.meta.finished),
        )
        emitted_frames += speech.shape[-1] // 2

    assert max(window_starts) > 0  # the window actually slid
    assert calls == window_starts
    assert emitted_frames == total  # every token becomes audio exactly once
