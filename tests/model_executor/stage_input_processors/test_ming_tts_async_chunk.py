# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.ming_tts.constants import LATENT_DIM, PATCH_SIZE
from vllm_omni.model_executor.stage_input_processors.ming_tts import (
    MING_EMIT_PATCH_COUNT_KEY,
    llm2audio_vae_async_chunk,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_transfer_manager(*, chunk_size: int = 5, initial_chunk_size: int = 2):
    return SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        put_req_chunk=defaultdict(int),
        request_payload={},
        connector=SimpleNamespace(
            config={
                "extra": {
                    "latent_chunk_size": chunk_size,
                    "initial_latent_chunk_size": initial_chunk_size,
                    "latent_left_context": 0,
                }
            }
        ),
    )


def _make_request(req_id: str = "req", *, finished: bool = False):
    return SimpleNamespace(
        external_req_id=req_id,
        is_finished=lambda: finished,
    )


def _make_output(value: float):
    return {
        "ming_has_patch": torch.tensor([1], dtype=torch.bool),
        "ming_latent_patch": torch.full((1, PATCH_SIZE, LATENT_DIM), value),
    }


def _append_patch(tm, req_id: str, idx: int, *, finished: bool = False):
    return llm2audio_vae_async_chunk(
        tm,
        _make_output(float(idx)),
        _make_request(req_id, finished=finished),
        is_finished=finished,
    )


def test_ming_async_chunk_emits_small_initial_chunk_then_steady_chunks():
    tm = _make_transfer_manager(chunk_size=5, initial_chunk_size=2)

    assert _append_patch(tm, "req", 0) is None

    first = _append_patch(tm, "req", 1)
    assert first is not None
    assert first.latent.shape == (2, PATCH_SIZE, LATENT_DIM)
    assert first.kv_metadata[MING_EMIT_PATCH_COUNT_KEY] == 2
    assert first.meta.finished.item() is False

    for idx in range(2, 6):
        assert _append_patch(tm, "req", idx) is None

    second = _append_patch(tm, "req", 6)
    assert second is not None
    assert second.latent.shape == (5, PATCH_SIZE, LATENT_DIM)
    assert second.kv_metadata[MING_EMIT_PATCH_COUNT_KEY] == 5


def test_ming_async_chunk_flushes_short_final_chunk():
    tm = _make_transfer_manager(chunk_size=5, initial_chunk_size=2)

    payload = _append_patch(tm, "req", 0, finished=True)

    assert payload is not None
    assert payload.latent.shape == (1, PATCH_SIZE, LATENT_DIM)
    assert payload.kv_metadata[MING_EMIT_PATCH_COUNT_KEY] == 1
    assert payload.meta.finished.item() is True


def test_ming_async_chunk_flushes_leftover_after_initial_chunk():
    tm = _make_transfer_manager(chunk_size=5, initial_chunk_size=2)

    assert _append_patch(tm, "req", 0) is None
    assert _append_patch(tm, "req", 1) is not None
    assert _append_patch(tm, "req", 2) is None

    final = _append_patch(tm, "req", 3, finished=True)

    assert final is not None
    assert final.latent.shape == (2, PATCH_SIZE, LATENT_DIM)
    assert final.kv_metadata[MING_EMIT_PATCH_COUNT_KEY] == 2
    assert final.meta.finished.item() is True
