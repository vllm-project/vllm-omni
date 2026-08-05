# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.experimental.fullduplex.personaplex.policy import (
    SILENCE_TOKENS,
    SINE_TOKENS,
)
from vllm_omni.experimental.fullduplex.personaplex.stage0 import (
    PersonaPlexStage0DuplexRuntime,
)
from vllm_omni.model_executor.models.personaplex.personaplex_talker import (
    PersonaPlexTalkerForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeCodec:
    def __init__(self) -> None:
        self.encode_calls = 0
        self.reset_calls = 0

    def streaming_init(self, batch_size: int) -> None:
        assert batch_size == 1

    def encode_frame(self, pcm: torch.Tensor) -> torch.Tensor:
        assert pcm.shape == (1, 1920)
        self.encode_calls += 1
        return torch.full((1, 8), self.encode_calls, dtype=torch.long)

    def reset_streaming(self) -> None:
        self.reset_calls += 1


class _FakeTalker:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.frame_calls: list[dict[str, torch.Tensor | None]] = []

    def _build_prefill_embed(
        self,
        tokens,
        offset,
        span,
        device,
        silence=None,
        user_sine=None,
    ):
        del offset, silence, user_sine
        values = tokens[:span].to(device=device, dtype=torch.float32)
        return values[:, None].expand(-1, 4).contiguous()

    def _build_frame_embed(
        self,
        text_token,
        last_agent,
        prev_agent,
        device,
        user_d0=None,
        user_d1=None,
    ):
        self.frame_calls.append(
            {
                "text_token": text_token.clone(),
                "last_agent": None if last_agent is None else last_agent.clone(),
                "prev_agent": None if prev_agent is None else prev_agent.clone(),
                "user_d0": None if user_d0 is None else user_d0.clone(),
                "user_d1": None if user_d1 is None else user_d1.clone(),
            }
        )
        value = float(user_d0[0].item()) if user_d0 is not None else -1.0
        return torch.full((1, 4), value, device=device)


def _duplex_info(*, seq: int, session_id: str = "session", incarnation: int = 1):
    pcm = np.zeros(1920, dtype="<f4")
    return {
        "data_plane": True,
        "session_id": session_id,
        "incarnation": incarnation,
        "epoch": 0,
        "seq": seq,
        "payload": {
            "format": "pcm_f32le",
            "sample_rate_hz": 24000,
            "audio": base64.b64encode(pcm.tobytes()).decode("ascii"),
        },
        "runtime_config": {
            "personaplex_model_path": "/unused",
            "personaplex_voice_prompt": "NATF2.pt",
            "personaplex_persona": "Be concise.",
        },
    }


def _runtime(*codecs: _FakeCodec) -> PersonaPlexStage0DuplexRuntime:
    voice_embeddings = torch.arange(8, dtype=torch.float32).reshape(2, 1, 1, 4)
    codec_args = {"codec": codecs[0]}
    if len(codecs) > 1:
        available = iter(codecs)
        codec_args = {
            "codec_factory": lambda: next(available),
            "max_sessions": len(codecs),
        }
    return PersonaPlexStage0DuplexRuntime(
        _FakeTalker(),
        model_path="/unused",
        device="cpu",
        **codec_args,
        tokenizer=lambda _text: [7, 8, 9],
        voice_loader=lambda _voice: {
            "embeddings": voice_embeddings,
            "cache": torch.zeros((1, 17, 4), dtype=torch.long),
        },
    )


def test_first_append_prepends_voice_and_persona_once() -> None:
    codec = _FakeCodec()
    runtime = _runtime(codec)

    first = runtime.prepare_append(_duplex_info(seq=1), prompt_len=18)
    second = runtime.prepare_append(_duplex_info(seq=2), prompt_len=18)

    assert first.prefill_applied is True
    assert first.prompt_offset == 0
    assert first.user_codes.shape == (1, 8)
    assert first.inputs_embeds.shape == (18, 4)
    assert second.prefill_applied is False
    assert second.prompt_offset == 17
    assert second.user_codes.shape == (2, 8)
    assert second.inputs_embeds.shape == (1, 4)
    assert codec.encode_calls == 2


@pytest.mark.parametrize(
    ("append_count", "expected_tokens", "expected_provided"),
    [
        (
            1,
            [*SILENCE_TOKENS, 1, *SINE_TOKENS[1:]],
            [False, *([True] * 15)],
        ),
        (
            2,
            [*SILENCE_TOKENS, 2, *([1] * 7)],
            [*([False] * 8), *([True] * 8)],
        ),
    ],
)
def test_prepare_append_preserves_native_depformer_teacher_forcing(
    append_count: int,
    expected_tokens: list[int],
    expected_provided: list[bool],
) -> None:
    runtime = _runtime(_FakeCodec())

    result = None
    for seq in range(1, append_count + 1):
        result = runtime.prepare_append(_duplex_info(seq=seq), prompt_len=18)

    assert result is not None
    assert result.info_update["pplex_depformer_audio_tokens"].tolist() == expected_tokens
    assert result.info_update["pplex_depformer_audio_provided"].tolist() == expected_provided


def test_frame_embed_uses_previous_effective_agent_frame() -> None:
    fake_talker = SimpleNamespace(
        config=SimpleNamespace(
            num_audio_codebooks=16,
            audio_vocab_size=2048,
        ),
        input_embeddings=lambda stack: stack,
    )
    last_agent = torch.arange(10, 18)

    embeds = PersonaPlexTalkerForConditionalGeneration._build_frame_embed(
        fake_talker,
        torch.tensor([3]),
        last_agent,
        torch.arange(8),
        torch.device("cpu"),
    )

    assert embeds.shape == (1, 17)
    assert torch.equal(embeds[0, 1:9], last_agent)


def test_repeated_append_identity_does_not_advance_codec() -> None:
    codec = _FakeCodec()
    runtime = _runtime(codec)

    info = _duplex_info(seq=1)
    first = runtime.prepare_append(info, prompt_len=18, request_id="req")
    retry = runtime.prepare_append(info, prompt_len=18, request_id="req")

    assert codec.encode_calls == 1
    assert torch.equal(first.inputs_embeds, retry.inputs_embeds)
    assert torch.equal(first.user_codes, retry.user_codes)


def test_next_append_uses_prior_sample_and_causally_delayed_user_frame() -> None:
    runtime = _runtime(_FakeCodec())

    runtime.prepare_append(_duplex_info(seq=1), prompt_len=18, request_id="req")
    first_agent = torch.arange(8, dtype=torch.long)
    runtime.record_sample(request_id="req", text_token=torch.tensor(101), agent_codes=first_agent)
    runtime.prepare_append(_duplex_info(seq=2), prompt_len=19, request_id="req")
    second_agent = torch.arange(10, 18, dtype=torch.long)
    runtime.record_sample(request_id="req", text_token=torch.tensor(102), agent_codes=second_agent)
    runtime.prepare_append(_duplex_info(seq=3), prompt_len=20, request_id="req")

    silence = torch.tensor(SILENCE_TOKENS, dtype=torch.long)
    sine = torch.tensor(SINE_TOKENS, dtype=torch.long)
    first_call = runtime.stage_model.frame_calls[0]
    assert all(torch.equal(first_call[key], silence) for key in ("last_agent", "prev_agent"))
    assert all(torch.equal(first_call[key], sine) for key in ("user_d0", "user_d1"))

    second_call = runtime.stage_model.frame_calls[1]
    assert second_call["text_token"].tolist() == [101]
    expected_first_effective = torch.cat([first_agent[:1], silence[1:]])
    assert torch.equal(second_call["last_agent"], expected_first_effective)
    assert torch.equal(second_call["prev_agent"], expected_first_effective)
    assert torch.equal(second_call["user_d0"], torch.full((8,), 1, dtype=torch.long))
    assert torch.equal(second_call["user_d1"], sine)

    third_call = runtime.stage_model.frame_calls[2]
    assert third_call["text_token"].tolist() == [102]
    assert torch.equal(third_call["last_agent"], second_agent)
    assert torch.equal(third_call["prev_agent"], second_agent)
    assert torch.equal(third_call["user_d0"], torch.full((8,), 2, dtype=torch.long))
    assert torch.equal(third_call["user_d1"], torch.full((8,), 1, dtype=torch.long))


def test_decoded_pcm_is_writable_for_torch_zero_copy() -> None:
    pcm = PersonaPlexStage0DuplexRuntime._decode_pcm(_duplex_info(seq=1)["payload"])

    assert pcm.flags.writeable


def _prepare_two_sessions(
    runtime: PersonaPlexStage0DuplexRuntime,
) -> tuple[object, object, object, object]:
    first_1 = runtime.prepare_append(_duplex_info(seq=1), prompt_len=18)
    second_1 = runtime.prepare_append(
        _duplex_info(seq=1, session_id="other", incarnation=2),
        prompt_len=18,
    )
    first_2 = runtime.prepare_append(_duplex_info(seq=2), prompt_len=19)
    second_2 = runtime.prepare_append(
        _duplex_info(seq=2, session_id="other", incarnation=2),
        prompt_len=19,
    )
    return first_1, second_1, first_2, second_2


def test_live_sessions_keep_independent_streaming_encoders() -> None:
    first_codec = _FakeCodec()
    second_codec = _FakeCodec()
    runtime = _runtime(first_codec, second_codec)

    first_1, second_1, first_2, second_2 = _prepare_two_sessions(runtime)

    assert first_1.user_codes[:, 0].tolist() == [1]
    assert second_1.user_codes[:, 0].tolist() == [1]
    assert first_2.user_codes[:, 0].tolist() == [1, 2]
    assert second_2.user_codes[:, 0].tolist() == [1, 2]


def test_stage0_session_capacity_fails_before_codec_state_is_shared() -> None:
    runtime = _runtime(_FakeCodec(), _FakeCodec())
    _prepare_two_sessions(runtime)

    with pytest.raises(RuntimeError, match="capacity 2"):
        runtime.prepare_append(
            _duplex_info(seq=1, session_id="third", incarnation=3),
            prompt_len=18,
        )


def test_close_session_resets_and_reuses_released_codec() -> None:
    first_codec = _FakeCodec()
    second_codec = _FakeCodec()
    runtime = _runtime(first_codec, second_codec)
    _prepare_two_sessions(runtime)

    runtime.close_session("session", 1)

    assert first_codec.reset_calls == 1
    assert second_codec.reset_calls == 0
    replacement = runtime.prepare_append(
        _duplex_info(seq=1, session_id="replacement", incarnation=4),
        prompt_len=18,
    )
    assert replacement.user_codes[:, 0].tolist() == [3]
