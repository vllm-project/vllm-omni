# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""E2E accuracy guard for the LMCache MP offload path (Qwen2.5-Omni).

Same contract as the in-process consistency test, against a standalone LMCache
cache server: a hit must reproduce what a fresh prefill would have produced.

The two paths store hidden states differently -- in-process files one entry per
(chunk, layer), MP one object per chunk covering every layer -- so passing the
in-process test says nothing about this one. What is shared is the requirement:
the talker must see the same conditioning either way.
"""

import os

import pytest

from tests.engine import kv_offload_helpers as helpers

pytestmark = [pytest.mark.advanced_model, pytest.mark.omni, pytest.mark.cuda]

# See the in-process test: batch composition alone flips greedy decoding on a
# near-tie without any cache involved.
os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")

MODEL = "Qwen/Qwen2.5-Omni-3B"

# The MP server is a fourth process on the same box, but it holds its cache in
# CPU memory, so only the three stages claim GPU.
#
# These fractions are of total card memory, so the in-process test's 0.5/0.1/0.05
# leaves the two downstream stages too little on a 24 GB card to load their
# weights. This split holds on both 24 GB and 40 GB, and still leaves room for
# three CUDA contexts -- each stage is its own process.
_THINKER = {
    "max_model_len": 1024,
    "max_num_batched_tokens": 1024,
    "gpu_memory_utilization": 0.45,
    "devices": "0",
    "enforce_eager": True,
    "async_chunk": False,
}
_DOWNSTREAM = {
    "1": {"devices": "0", "gpu_memory_utilization": 0.20, "enforce_eager": True},
    "2": {"devices": "0", "gpu_memory_utilization": 0.10, "enforce_eager": True},
}

# autostart lets the connector own the server process for the duration of the
# run, so the test needs no external service.
_MP = {"mode": "mp", "mp.autostart": True}


def _run(*, lmcache: bool, rounds: int) -> dict[str, dict]:
    return helpers.run(
        model=MODEL,
        overrides=helpers.stage_overrides(
            lmcache=lmcache,
            hidden_states=True,
            thinker_extra=_THINKER,
            downstream_extra=_DOWNSTREAM,
            lmcache_extra=_MP if lmcache else None,
        ),
        rounds=rounds,
    )


def test_mp_offload_matches_baseline():
    """A hit served by the MP cache server must match a no-offload run."""
    pytest.importorskip("lmcache", reason="lmcache not installed")

    # Round 1 populates the cache; round 2 is served from it.
    baseline = _run(lmcache=False, rounds=2)
    cached = _run(lmcache=True, rounds=2)

    assert baseline, "baseline produced no output"
    assert cached, "MP offload run produced no output"
    assert set(baseline) == set(cached), "the two runs answered different prompts"
    assert any(helpers.audio_len(e) for e in baseline.values()), (
        "baseline produced no audio; the HS restore path is untested without it"
    )

    problems = helpers.compare(baseline, cached, expect_audio=True)
    assert not problems, "MP offload run diverged from the no-offload baseline:\n" + "\n".join(problems)
