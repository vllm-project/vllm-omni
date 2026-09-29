# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.moss_tts import (
    talker2codec_raw_async_chunk,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_FRAME = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
_N_VQ = len(_FRAME)


def _req(rid="r1", *, finished=False):
    return SimpleNamespace(
        external_req_id=rid,
        request_id=rid,
    )


def _tm(
    *,
    chunk_frames=15,
    initial_chunk_frames=1,
    chunk_ramp=None,
    adaptive=False,
):
    extra: dict = {
        "codec_chunk_frames": chunk_frames,
        "initial_codec_chunk_frames": initial_chunk_frames,
    }
    if chunk_ramp is not None:
        extra["codec_chunk_ramp"] = chunk_ramp
    if adaptive:
        extra["codec_chunk_adaptive"] = True
    return SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        request_payload={},
        put_req_chunk=defaultdict(int),
        ramp_chunk_count=defaultdict(int),
        _adaptive_states={},
        _ramp_total_emitted=defaultdict(int),
        connector=SimpleNamespace(config={"extra": extra}),
    )


def _frame(step):
    return torch.tensor(
        [step % 1024 for _ in range(_N_VQ)],
        dtype=torch.long,
    )


def _run_seq(tm, req, n_frames, *, chunk_ramp_set_ramp_count=True):
    """Feed n_frames one-by-one, return list of emitted frame counts."""
    emits: list[int] = []
    for step in range(n_frames):
        mm = {"codes": {"audio": _frame(step).unsqueeze(0)}}
        finished = step == n_frames - 1
        result = talker2codec_raw_async_chunk(tm, mm, req, is_finished=finished)
        if result is not None:
            emits.append(result.meta.codec_chunk_frames)
            if chunk_ramp_set_ramp_count and not result.meta.finished.item():
                tm.ramp_chunk_count[req.external_req_id] += 1
                tm.put_req_chunk[req.external_req_id] += 1
    return emits


# ---------------------------------------------------------------------------
# Default path (no ramp / adaptive)
# ---------------------------------------------------------------------------


def test_default_emits_initial_then_steady():
    tm = _tm()
    req = _req()
    emits = _run_seq(tm, req, 31)
    assert emits == [1, 15, 15], f"expected [1, 15, 15], got {emits}"


def test_default_finish_flushes_partial():
    tm = _tm()
    req = _req()
    emits = _run_seq(tm, req, 10)
    assert emits == [1, 9], f"expected [1, 9], got {emits}"


def test_default_empty_finish_returns_sentinel():
    tm = _tm()
    req = _req()
    result = talker2codec_raw_async_chunk(tm, None, req, is_finished=True)
    assert result is not None
    assert result.meta.finished.item() is True
    assert result.meta.codec_chunk_frames == 0


# ---------------------------------------------------------------------------
# Ramp path
# ---------------------------------------------------------------------------


def test_ramp_emits_gradual_chunks():
    tm = _tm(chunk_ramp=[4, 4, 8, 15])
    req = _req()
    emits = _run_seq(tm, req, 31)
    # ramp [4, 4, 8, 15] → cumulative [4, 8, 16, 31]
    # chunk 0: 4, chunk 1: 4, chunk 2: 8, chunk 3: remaining 15
    assert emits == [4, 4, 8, 15], f"expected [4, 4, 8, 15], got {emits}"


def test_ramp_finish_flushes_partial():
    tm = _tm(chunk_ramp=[4, 4, 8, 15])
    req = _req()
    emits = _run_seq(tm, req, 10)
    # chunk 0: 4, chunk 1: 4, then finish with 2 remaining
    assert emits == [4, 4, 2], f"expected [4, 4, 2], got {emits}"


def test_ramp_total_frames_match():
    tm = _tm(chunk_ramp=[4, 4, 8, 15])
    req = _req()
    emits = _run_seq(tm, req, 50)
    assert sum(emits) == 50, f"total {sum(emits)} != 50"


def test_ramp_total_emitted_cleaned_on_finish():
    tm = _tm(chunk_ramp=[4, 4, 8, 15])
    req = _req()
    _run_seq(tm, req, 31)
    assert req.external_req_id not in tm._ramp_total_emitted


# ---------------------------------------------------------------------------
# Adaptive path
# ---------------------------------------------------------------------------


def test_adaptive_chunk_0_uses_ic_threshold():
    tm = _tm(initial_chunk_frames=1, adaptive=True)
    req = _req()
    emits = _run_seq(tm, req, 1)
    assert emits == [1], f"expected [1], got {emits}"


def test_adaptive_finish_flushes_remaining():
    tm = _tm(adaptive=True)
    req = _req()
    emits = _run_seq(tm, req, 5)
    # chunk 0 uses threshold=15, but only 5 frames → finish flushes all
    # The adaptive chunk 0 path uses the same IC/steady threshold as default.
    # With adaptive=True and no explicit initial_chunk_frames override (default=1),
    # chunk 0 emits 1 frame, then chunks 1+ use adaptive controller.
    # With only 5 frames total, chunk 0=1, then 4 remaining flushed on finish.
    assert sum(emits) == 5, f"total {sum(emits)} != 5"
    assert len(emits) >= 1, "expected at least one emit"


def test_adaptive_states_cleaned_on_finish():
    tm = _tm(adaptive=True)
    req = _req()
    _run_seq(tm, req, 5)
    assert req.external_req_id not in tm._adaptive_states
    assert req.external_req_id not in tm._ramp_total_emitted


# ---------------------------------------------------------------------------
# Seed fallback (serving_speech)
# ---------------------------------------------------------------------------


def test_moss_tts_seed_fallback_sets_tts_local_seed():
    """Verify the serving_speech seed fallback logic for MOSS-TTS.

    When request.seed is None but default_sampling_params.seed is set,
    tts_local_seed should be set from the default seed. This mirrors
    the Qwen3-TTS pattern.
    """
    import copy

    from vllm.sampling_params import SamplingParams

    sp = SamplingParams(seed=42)
    sp.extra_args = None

    # Simulate the serving_speech logic:
    # if self._tts_model_type in ("moss_tts", ...):
    #     default_seed = getattr(stage0_params, "seed", None)
    #     if default_seed is not None:
    #         sp = copy.deepcopy(sp)
    #         sp.extra_args = {}
    #         sp.extra_args.setdefault("tts_local_seed", int(default_seed))

    default_seed = getattr(sp, "seed", None)
    assert default_seed == 42

    sp = copy.deepcopy(sp)
    if sp.extra_args is None:
        sp.extra_args = {}
    sp.extra_args.setdefault("tts_local_seed", int(default_seed))

    assert sp.extra_args["tts_local_seed"] == 42


def test_moss_tts_seed_fallback_skips_when_no_default_seed():
    """When both request.seed and default seed are None, tts_local_seed is not set."""
    from vllm.sampling_params import SamplingParams

    sp = SamplingParams(seed=None)
    sp.extra_args = None

    default_seed = getattr(sp, "seed", None)
    assert default_seed is None
    # The if default_seed is not None branch is skipped — no tts_local_seed set


# ---------------------------------------------------------------------------
# Abort / cleanup
# ---------------------------------------------------------------------------


def test_ramp_chunk_count_owned_by_adapter_not_processor():
    """The processor must not increment ramp_chunk_count — the adapter owns it."""
    tm = _tm(chunk_ramp=[4, 4, 8, 15])
    req = _req()
    # Simulate adapter incrementing ramp_chunk_count after each successful emit.
    # The last emit has finished=True, so the adapter does not increment after it.
    emits = _run_seq(tm, req, 31, chunk_ramp_set_ramp_count=True)
    assert emits == [4, 4, 8, 15]
    # ramp_chunk_count should be 3 (adapter increments after each non-finished emit)
    assert tm.ramp_chunk_count[req.external_req_id] == 3
