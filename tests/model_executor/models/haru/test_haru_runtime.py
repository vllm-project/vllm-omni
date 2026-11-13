# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""HaruRuntime step protocol on a tiny stand-in model (CPU): prompt spans, live frames, ordering, release."""

from __future__ import annotations

import base64
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from tests.model_executor.models.haru.conftest import EOS, TINY_SPEC, build_tiny_codec
from vllm_omni.model_executor.models.haru import runtime as runtime_module
from vllm_omni.model_executor.models.haru.haru import AudioProjector, HaruForConditionalGeneration
from vllm_omni.model_executor.models.haru.meanflow_head import MeanFlowHead
from vllm_omni.model_executor.models.haru.prompt import prompt_length
from vllm_omni.model_executor.models.haru.runtime import HaruRuntime

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

HIDDEN, VOCAB = 8, 16
CARD = "be brief"


class _TinyHaru(nn.Module):
    """The attributes HaruRuntime reads from HaruForConditionalGeneration, at toy size."""

    embed_streams = HaruForConditionalGeneration.embed_streams
    scale_latents = HaruForConditionalGeneration.scale_latents
    unscale_latents = HaruForConditionalGeneration.unscale_latents

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.haru_config = TINY_SPEC
        self.embedding = nn.Embedding(VOCAB, HIDDEN)
        self.model = SimpleNamespace(embed_input_ids=self.embedding)
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
        self.model_audio_proj = AudioProjector(TINY_SPEC.latent_dim, HIDDEN)
        self.user_audio_proj = AudioProjector(TINY_SPEC.latent_dim, HIDDEN)
        self.head = MeanFlowHead(TINY_SPEC.head)
        self.codec = build_tiny_codec()
        self.register_buffer("latent_scale", torch.ones(()))
        self.register_buffer("latent_bias", torch.zeros(()))


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        return SimpleNamespace(ids=[1] * len(text.split()))


PROMPT_LEN = prompt_length(_Tokenizer(), CARD, 0, False, TINY_SPEC.frame_samples)


@pytest.fixture
def runtime(monkeypatch) -> HaruRuntime:
    monkeypatch.setattr(runtime_module, "load_tokenizer", lambda _path: _Tokenizer())
    return HaruRuntime(_TinyHaru(), model_path="unused")


def _frame(value: float = 0.0) -> dict[str, object]:
    pcm = np.full(TINY_SPEC.frame_samples, value, dtype=np.float32)
    return {"audio": base64.b64encode(pcm.tobytes()).decode()}


def _info(request_id: str, computed: int, prompt_len: int, *, first: bool = False, seed: int = 7) -> dict:
    duplex: dict[str, object] = {"payload": _frame(0.1)}
    if first:
        duplex["runtime_config"] = {"haru": {"system_prompt": CARD, "seed": seed}}
    return {
        "request_id": request_id,
        "_omni_num_computed_tokens": computed,
        "_omni_prompt_len": prompt_len,
        "duplex": duplex,
    }


def _step(runtime: HaruRuntime, rows: list[tuple[str, int, dict]], token: int = 3) -> dict:
    """One engine step: add_row per request -> logits -> sampled tokens -> post-sample audio."""
    runtime.begin_step()
    embeds = [runtime.add_row(torch.zeros(span, dtype=torch.long), None, info) for _, span, info in rows]
    hidden = torch.stack([e[-1] for e in embeds])
    logits = runtime.adjust_logits(hidden, torch.zeros(len(rows), VOCAB))
    assert logits.dtype == torch.float32
    runtime.record_sampled([[token] for _ in rows])
    return runtime.finish_step([request_id for request_id, _, _ in rows])


def test_prompt_then_live_frames_each_return_one_audio_frame(runtime: HaruRuntime) -> None:
    prompt_len = PROMPT_LEN  # card only: no reference, no opening turn
    out = _step(runtime, [("r1", prompt_len, _info("r1", 0, prompt_len, first=True))])
    assert out["meta"]["req_id"] == ["r1"]
    (audio,) = out["model_outputs"]
    assert audio.shape == (TINY_SPEC.frame_samples,) and audio.abs().max() <= 1.0
    assert int(out["sr"][0]) == TINY_SPEC.sample_rate

    for computed in (prompt_len, prompt_len + 1):
        out = _step(runtime, [("r1", 1, _info("r1", computed, computed + 1))])
        assert [a.shape[0] for a in out["model_outputs"]] == [TINY_SPEC.frame_samples]


def test_a_prompt_chunk_that_does_not_reach_the_end_emits_no_audio(runtime: HaruRuntime) -> None:
    out = _step(runtime, [("r1", 1, _info("r1", 0, PROMPT_LEN, first=True))])
    assert out["model_outputs"] == [] and out["meta"]["req_id"] == []


def test_sessions_with_the_same_seed_and_input_produce_the_same_audio(runtime: HaruRuntime) -> None:
    rows = [(rid, PROMPT_LEN, _info(rid, 0, PROMPT_LEN, first=True, seed=5)) for rid in ("a", "b")]
    first = _step(runtime, rows)
    torch.testing.assert_close(first["model_outputs"][0], first["model_outputs"][1])
    live = _step(runtime, [(rid, 1, _info(rid, PROMPT_LEN, PROMPT_LEN + 1)) for rid in ("a", "b")])
    torch.testing.assert_close(live["model_outputs"][0], live["model_outputs"][1])


def test_out_of_order_or_lost_conversations_fail_loudly(runtime: HaruRuntime) -> None:
    _step(runtime, [("r1", PROMPT_LEN, _info("r1", 0, PROMPT_LEN, first=True))])
    runtime.begin_step()
    with pytest.raises(RuntimeError, match="out of order"):
        runtime.add_row(torch.zeros(1, dtype=torch.long), None, _info("r1", PROMPT_LEN + 2, PROMPT_LEN + 3))
    with pytest.raises(RuntimeError, match="lost its state"):
        runtime.add_row(torch.zeros(1, dtype=torch.long), None, _info("unknown", 4, 5))
    with pytest.raises(RuntimeError, match="exactly one position"):
        runtime.add_row(torch.zeros(2, dtype=torch.long), None, _info("r1", PROMPT_LEN, PROMPT_LEN + 2))


def test_release_drops_the_conversation_state(runtime: HaruRuntime) -> None:
    _step(runtime, [("r1", PROMPT_LEN, _info("r1", 0, PROMPT_LEN, first=True))])
    runtime.release({"r1"})
    runtime.begin_step()
    with pytest.raises(RuntimeError, match="lost its state"):
        runtime.add_row(torch.zeros(1, dtype=torch.long), None, _info("r1", PROMPT_LEN, PROMPT_LEN + 1))


def test_rows_without_a_conversation_use_plain_token_embeddings(runtime: HaruRuntime) -> None:
    runtime.begin_step()
    ids = torch.tensor([1, 2, 3])
    embeds = runtime.add_row(ids, None, {"request_id": "warmup", "_omni_num_computed_tokens": 0, "_omni_prompt_len": 3})
    torch.testing.assert_close(embeds, runtime._model.embedding(ids))
    runtime.adjust_logits(torch.zeros(1, HIDDEN), torch.zeros(1, VOCAB))
    runtime.record_sampled([[EOS]])
    assert runtime.finish_step(["warmup"])["model_outputs"] == []


def test_mismatched_step_rows_are_rejected(runtime: HaruRuntime) -> None:
    runtime.begin_step()
    runtime.add_row(torch.zeros(2, dtype=torch.long), None, _info("r1", 0, PROMPT_LEN, first=True))
    with pytest.raises(RuntimeError, match="logit rows"):
        runtime.adjust_logits(torch.zeros(2, HIDDEN), torch.zeros(2, VOCAB))
