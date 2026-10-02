# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import base64
import io
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.model_executor.models.personaplex.duplex.policy import (
    AUDIO_SILENCE_FRAME_CNT,
    ZERO_TEXT_TOKEN,
    wrap_with_system_tags,
)
from vllm_omni.model_executor.models.personaplex.duplex.stage0 import (
    PersonaPlexStage0DuplexRuntime,
    load_personaplex_voice_state,
)
from vllm_omni.model_executor.models.personaplex.personaplex_talker import (
    PersonaPlexTalkerForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_PERSONA = "Be concise."


class _FakeCodec:
    def streaming_init(self, batch_size: int, *, decode: bool = True) -> None:
        self.batch_size = batch_size

    def encode_frame(self, pcm: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        return torch.ones((pcm.shape[0], 8), dtype=torch.long)

    def reset_slot(self, row: int) -> None:
        del row


class _FakeTalker:
    """Deterministic prefill and frame embeddings; one float per token and frame."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.dtype = torch.float32

    def _build_prefill_embed(self, tokens, offset, span, device, silence=None, user_sine=None):
        del offset, silence, user_sine
        values = tokens[:span].to(device=device, dtype=torch.float32)
        return values[:, None].expand(-1, 4).contiguous()

    def _build_frame_embeds(self, text_tokens, last_agent, *, user_d0, user_d1):
        del text_tokens, last_agent, user_d1
        return user_d0[:, :1].to(torch.float32).expand(-1, 4).contiguous()


def _voice_embeddings(voice: str) -> torch.Tensor:
    base = float(sum(voice.encode()))
    return (torch.arange(8, dtype=torch.float32) + base).reshape(2, 1, 1, 4)


def _persona_tokens(text: str) -> list[int]:
    return [100 + len(text) % 50, 7, 8]


def _runtime(max_sessions: int = 4):
    return PersonaPlexStage0DuplexRuntime(
        _FakeTalker(),
        model_path="/unused",
        device="cpu",
        codec=_FakeCodec(),
        max_sessions=max_sessions,
        tokenizer=_persona_tokens,
        voice_loader=lambda voice: {"embeddings": _voice_embeddings(voice)},
    )


def _duplex_info(*, session_id: str, seq: int = 1, voice: str = "NATF2.pt", persona: str = _PERSONA):
    pcm = np.zeros(1920, dtype="<f4")
    return {
        "data_plane": True,
        "session_id": session_id,
        "epoch": 0,
        "seq": seq,
        "payload": {
            "format": "pcm_f32le",
            "sample_rate_hz": 24000,
            "audio": base64.b64encode(pcm.tobytes()).decode("ascii"),
        },
        "runtime_config": {
            "personaplex_voice_prompt": voice,
            "personaplex_persona": persona,
        },
    }


def _uncached_prefill(voice: str, persona: str) -> torch.Tensor:
    """The first-append prefill as prepare_append built it before caching."""
    tokens = torch.tensor(
        [
            *([ZERO_TEXT_TOKEN] * AUDIO_SILENCE_FRAME_CNT),
            *_persona_tokens(wrap_with_system_tags(persona)),
            *([ZERO_TEXT_TOKEN] * AUDIO_SILENCE_FRAME_CNT),
        ],
        dtype=torch.long,
    )
    voice_rows = _voice_embeddings(voice).reshape(-1, 4)
    token_rows = _FakeTalker()._build_prefill_embed(tokens, 0, int(tokens.numel()), torch.device("cpu"))
    return torch.cat([voice_rows, token_rows], dim=0)


def test_cached_prefill_is_bit_identical_to_the_uncached_build() -> None:
    runtime = _runtime()
    expected = _uncached_prefill("NATF2.pt", _PERSONA)

    first = runtime.prepare_append(_duplex_info(session_id="a"), prompt_len=64)
    assert torch.equal(first.inputs_embeds[:-1], expected)
    # Changing a prepared append does not change the cached prefill.
    first.inputs_embeds.fill_(-1.0)
    second = runtime.prepare_append(_duplex_info(session_id="b"), prompt_len=64)

    assert first.prefill_applied is True and second.prefill_applied is True
    assert torch.equal(second.inputs_embeds[:-1], expected)
    assert runtime.sessions[("a", 0)].prefill_slots == expected.shape[0]
    assert runtime.sessions[("b", 0)].prefill_slots == expected.shape[0]


def test_a_changed_voice_or_persona_gets_its_own_prefill() -> None:
    runtime = _runtime()
    runtime.prepare_append(_duplex_info(session_id="a"), prompt_len=64)

    for session_id, voice, persona in (("b", "NATF2.pt", "Be brief."), ("c", "NATM1.pt", _PERSONA)):
        prepared = runtime.prepare_append(
            _duplex_info(session_id=session_id, voice=voice, persona=persona), prompt_len=64
        )
        assert torch.equal(prepared.inputs_embeds[:-1], _uncached_prefill(voice, persona))


def _voice_bundle(tmp_path, voices: dict[str, torch.Tensor]):
    archive = tmp_path / "voices.tgz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, embeddings in voices.items():
            buffer = io.BytesIO()
            torch.save({"embeddings": embeddings}, buffer)
            data = buffer.getvalue()
            info = tarfile.TarInfo(f"voices/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return archive


def test_voices_load_from_the_archive(tmp_path) -> None:
    voices = {"A.pt": torch.arange(4.0).reshape(1, 4), "B.pt": torch.arange(8.0).reshape(2, 4)}
    _voice_bundle(tmp_path, voices)

    for name, embeddings in voices.items():
        assert torch.equal(load_personaplex_voice_state(str(tmp_path), name)["embeddings"], embeddings)
    with pytest.raises(FileNotFoundError):
        load_personaplex_voice_state(str(tmp_path), "missing.pt")


def test_talker_prefill_warmup_is_best_effort() -> None:
    def missing_voice():
        raise FileNotFoundError("no voices.tgz")

    fake = SimpleNamespace(_duplex_stage0_runtime=lambda: SimpleNamespace(warm_prefill=missing_voice))

    PersonaPlexTalkerForConditionalGeneration._warm_duplex_prefill(fake)
