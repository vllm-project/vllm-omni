# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""What an open-session ``instructions`` field builds into the MiniCPM-o 4.5 Stage-0 context.

E3 contract: the system string has ONE resolution point
(``MiniCPMO45DuplexPolicy.session_context_texts``), so an omitted, empty, or
explicitly default ``instructions`` field all build byte-identical Stage-0
contexts (system piece + reference-audio block + suffix), and an explicit
non-default string is honored as the system prompt. The serving-side first
append reservation (``_apply_first_append_context_tokens``) and the worker-side
context builder (``_prepare_session_context``) must agree on those counts,
because the scheduler reserves the parked listening request's slots from the
serving number while the worker embeds the pieces from its own encoding.
"""

from __future__ import annotations

import base64
from typing import cast

import numpy as np
import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.minicpmo_4_5.duplex.plugin import (
    _apply_first_append_context_tokens,
)
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.policy import (
    MiniCPMO45DuplexPolicy,
)
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.stage0 import (
    MiniCPMO45Stage0DuplexRuntime,
    _MiniCPMO45Stage0SessionState,
)
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.window_plan import (
    DuplexWindowGeometry,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

DEFAULT_INSTRUCTIONS = "Streaming Omni Conversation."
CUSTOM_INSTRUCTIONS = "You are a pirate. Answer in one sentence."
#: The bundled ``assets/HT_ref_audio.wav`` budget: 96256 float32 samples pool
#: to 60 ``<unit>`` embeddings, so the checkpoint tokenizer's 8 system + 2
#: suffix ids make the 70 context ids a duplex session always embeds.
_REF_SAMPLES = 96256
_REF_TOKENS = 60

_SPECIAL_TOKEN_IDS: dict[str, int] = {
    "<unit>": 1,
    "</unit>": 2,
    "<|listen|>": 3,
    "<|speak|>": 4,
    "<|tts_bos|>": 5,
    "<|tts_eos|>": 6,
    "<|tts_pad|>": 7,
    "<|chunk_eos|>": 8,
    "<|chunk_tts_eos|>": 9,
    "<|turn_eos|>": 10,
    "<|audio|>": 11,
    "<|im_start|>": 12,
    "<|im_end|>": 13,
    "<|audio_start|>": 14,
    "<|audio_end|>": 15,
}


class _FakeTokenizer:
    """Word-level stand-in for the checkpoint tokenizer (special tokens stay single ids)."""

    unk_token_id = None
    all_special_ids = tuple(_SPECIAL_TOKEN_IDS.values())

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return _SPECIAL_TOKEN_IDS.get(token)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = []
        for word in text.split(" "):
            if word in _SPECIAL_TOKEN_IDS:
                ids.append(_SPECIAL_TOKEN_IDS[word])
            else:
                ids.extend(100 + ord(ch) % 90 for ch in word)
        return ids


class _FakeEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.arange(300 * 8, dtype=torch.float32).reshape(300, 8))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight.index_select(0, token_ids.reshape(-1)).unsqueeze(0)


class _FakeThinker(nn.Module):
    """The two entry points ``_prepare_session_context`` consumes, no weights."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = _FakeEmbedding()

    def get_input_embeddings(self) -> _FakeEmbedding:
        return self.embedding

    def get_audio_embedding(self, batch_feature: object) -> list[torch.Tensor]:
        # The real thinker pools to ``audio_token_count`` rows; mirror that shape.
        rows = _REF_TOKENS
        return [torch.arange(rows * 8, dtype=torch.float32).reshape(rows, 8)]


class _FakeProcessor:
    tokenizer = _FakeTokenizer()

    def process_audio(self, wavs: list[np.ndarray]) -> dict[str, torch.Tensor]:
        frames = (len(wavs[0]) + 159) // 160
        return {"audio_features": torch.arange(frames * 4, dtype=torch.float32).reshape(1, 4, frames)}


class _FakeStageModel:
    def __init__(self) -> None:
        self.thinker = _FakeThinker()
        self.processor = _FakeProcessor()
        self.tokenizer = _FakeTokenizer()


def _build_stage0_context(instructions: object) -> _MiniCPMO45Stage0SessionState:
    runtime = MiniCPMO45Stage0DuplexRuntime(_FakeStageModel(), model_path=None, device="cpu")
    state = _MiniCPMO45Stage0SessionState(session_id="duplex-test")
    ref_audio: np.ndarray = np.zeros(_REF_SAMPLES, dtype=np.float32)
    runtime_config: dict[str, object] = {
        "ref_audio_data": base64.b64encode(ref_audio.tobytes()).decode("ascii"),
        "ref_audio_format": "pcm_f32le",
        "ref_audio_sample_rate_hz": 16000,
    }
    runtime._prepare_session_context(state, {"instructions": instructions}, runtime_config=runtime_config)
    return state


def _serving_reservation(instructions: object, ref_sample_count: int | None = _REF_SAMPLES) -> dict[str, object]:
    runtime_config: dict[str, object] = {"instructions": instructions}
    _apply_first_append_context_tokens(
        runtime_config,
        tokenizer=_FakeTokenizer(),
        instructions=instructions,
        initial_user_text=None,
        ref_sample_count=ref_sample_count,
    )
    return runtime_config


@pytest.mark.parametrize("instructions", [None, "", DEFAULT_INSTRUCTIONS], ids=["omitted", "empty", "explicit-default"])
def test_default_substitution_builds_identical_stage0_context(instructions: object) -> None:
    """Omitted, empty and explicit-default instructions are the same context, piece by piece."""
    state = _build_stage0_context(instructions)
    prefix, suffix = MiniCPMO45DuplexPolicy.session_context_texts(instructions, True, None)

    expected_prefix_ids = _FakeTokenizer().encode(prefix)
    expected_suffix_ids = _FakeTokenizer().encode(suffix)
    ref_unit_ids = [_SPECIAL_TOKEN_IDS["<unit>"]] * MiniCPMO45DuplexPolicy.audio_token_count(_REF_SAMPLES)
    # The window head is the system text plus the reference audio block; the suffix trails it.
    assert state.context_prefix_token_ids == expected_prefix_ids + ref_unit_ids
    assert state.context_suffix_token_ids == expected_suffix_ids
    assert state.context_token_ids == expected_prefix_ids + ref_unit_ids + expected_suffix_ids
    # Text pieces embed one row per id; the reference block appends one (rows × dim) tensor.
    assert len(state.context_embeds) == len(state.context_token_ids) - len(ref_unit_ids) + 1
    assert state.context_embeds[-1 - len(expected_suffix_ids)].shape == (len(ref_unit_ids), 8)
    assert all(isinstance(embed, torch.Tensor) for embed in state.context_embeds)


def test_omitted_and_explicit_default_instructions_are_byte_identical() -> None:
    omitted = _build_stage0_context(None)
    explicit = _build_stage0_context(DEFAULT_INSTRUCTIONS)
    assert omitted.context_token_ids == explicit.context_token_ids
    assert len(omitted.context_embeds) == len(explicit.context_embeds)
    assert all(torch.equal(a, b) for a, b in zip(omitted.context_embeds, explicit.context_embeds))


def test_custom_instructions_are_honored_without_touching_ref_audio_or_suffix() -> None:
    custom = _build_stage0_context(CUSTOM_INSTRUCTIONS)
    default = _build_stage0_context(None)
    prefix, suffix = MiniCPMO45DuplexPolicy.session_context_texts(CUSTOM_INSTRUCTIONS, True, None)

    assert (
        custom.context_prefix_token_ids
        == _FakeTokenizer().encode(prefix) + [_SPECIAL_TOKEN_IDS["<unit>"]] * _REF_TOKENS
    )
    assert custom.context_suffix_token_ids == default.context_suffix_token_ids
    custom_units = [token_id for token_id in custom.context_token_ids if token_id == _SPECIAL_TOKEN_IDS["<unit>"]]
    assert custom_units == [_SPECIAL_TOKEN_IDS["<unit>"]] * _REF_TOKENS
    # Same skeleton, different system text: only the system piece moved.
    suffix_len = len(default.context_suffix_token_ids)
    assert custom.context_token_ids[-suffix_len:] == default.context_token_ids[-suffix_len:]
    default_prefix, _ = MiniCPMO45DuplexPolicy.session_context_texts(None, True, None)
    assert (
        custom.context_prefix_token_ids
        != _FakeTokenizer().encode(default_prefix) + [_SPECIAL_TOKEN_IDS["<unit>"]] * _REF_TOKENS
    )


@pytest.mark.parametrize("ref_sample_count", [_REF_SAMPLES, None], ids=["ref-audio", "no-ref-audio"])
@pytest.mark.parametrize("instructions", [None, "", DEFAULT_INSTRUCTIONS, CUSTOM_INSTRUCTIONS])
def test_serving_reservation_matches_worker_context_counts(instructions: object, ref_sample_count: int | None) -> None:
    """The parked listening request's budget (serving) fits the pieces (worker) for every variant."""
    has_ref_audio = ref_sample_count is not None
    prefix, suffix = MiniCPMO45DuplexPolicy.session_context_texts(instructions, has_ref_audio, None)
    prefix_tokens = len(_FakeTokenizer().encode(prefix))
    suffix_tokens = len(_FakeTokenizer().encode(suffix))
    ref_tokens = _REF_TOKENS if has_ref_audio else 0

    reservation = _serving_reservation(instructions, ref_sample_count)
    assert reservation["duplex_first_append_context_tokens"] == prefix_tokens + ref_tokens + suffix_tokens
    assert reservation["duplex_window_prefix_tokens"] == prefix_tokens + ref_tokens
    assert reservation["duplex_window_suffix_token_ids"] == _FakeTokenizer().encode(suffix)

    window_prefix_tokens = cast("int", reservation["duplex_window_prefix_tokens"])
    geometry = DuplexWindowGeometry(
        prefix_tokens=window_prefix_tokens,
        window_tokens=512,
        block_size=16,
        max_model_len=8192,
    )
    assert geometry.prefix_tokens == prefix_tokens + ref_tokens


def test_omitted_and_explicit_default_reservations_are_identical() -> None:
    omitted = _serving_reservation(None)
    explicit = _serving_reservation(DEFAULT_INSTRUCTIONS)
    functional = {
        key: value
        for key, value in omitted.items()
        if key != "instructions"  # the raw field is update-time bookkeeping only
    }
    assert functional == {key: value for key, value in explicit.items() if key != "instructions"}
