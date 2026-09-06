# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Minimal repro: does serving a prompt from LMCache change its own answer?

Every earlier comparison ran two engines and differed in more than one way. Here
a single engine answers the same prompt twice: round 1 is cold and round 2 is
served from LMCache, so the cache hit is the only variable.

Round 1's answer is the reference -- it is what this very engine produces with
no cache involved.
"""

import os

import pytest

from tests.engine import kv_offload_helpers as helpers

pytestmark = [pytest.mark.advanced_model, pytest.mark.omni, pytest.mark.cuda]

# Batch composition shifts the bf16 accumulation order, which flips greedy
# decoding on a near-tie -- enough that a multi-request comparison fails with no
# cache involved at all. vLLM's batch-invariant kernels remove that, so the
# comparison below tests the restore rather than the scheduler. Set before any
# engine process is spawned, since each reads it at import.
os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")

MODEL = "Qwen/Qwen2.5-Omni-3B"

# Every stage shares one card; which card is overridable so two variants of
# this test can occupy separate GPUs at the same time.
_DEVICE = os.environ.get("OMNI_TEST_DEVICE", "0")
_THINKER = {
    "max_model_len": 1024,
    "max_num_batched_tokens": 1024,
    "gpu_memory_utilization": 0.8,
    "devices": _DEVICE,
    "enforce_eager": True,
    "async_chunk": False,
}
_DOWNSTREAM = {
    "1": {"devices": _DEVICE, "gpu_memory_utilization": 0.1, "enforce_eager": True},
    "2": {"devices": _DEVICE, "gpu_memory_utilization": 0.05, "enforce_eager": True},
}


@pytest.mark.parametrize("mode", ["off", "kv_only", "kv_and_hs"])
def test_second_round_matches_first(mode):
    """Round 2 must answer exactly what round 1 answered.

    ``off`` is the control: it shows whether repeating a prompt is stable at all
    here, so a failure elsewhere can be attributed to the cache hit.

    ``kv_only`` disables the hidden-state store. Text is produced by the thinker,
    which reads KV and not hidden states, so text still diverging there puts the
    fault in LMCache's KV restore rather than in the hidden-state path. Audio is
    then required to be *absent*: the hit skips the prefill that would have
    produced the hidden states, and with no store to restore them from, the
    talker has nothing to condition on. That arm is what shows the hidden-state
    offload is load-bearing rather than an optimisation.
    """
    pytest.importorskip("lmcache", reason="lmcache not installed")

    rounds = helpers.run_rounds(
        model=MODEL,
        overrides=helpers.stage_overrides(
            lmcache=mode != "off",
            prefix_caching=False,
            hidden_states=mode == "kv_and_hs",
            thinker_extra=_THINKER,
            downstream_extra=_DOWNSTREAM,
        ),
        rounds=2,
        # Several requests, so a restore that targets the wrong request's slots
        # shows up here and not only in the unit tests.
        num_prompts=int(os.environ.get("OMNI_TEST_NUM_PROMPTS", "3")),
    )
    cold, served = rounds[0], rounds[1]

    for label, result in (("round 1", cold), ("round 2", served)):
        for prompt, entry in result.items():
            print(f"{label}: text={entry.get('text')!r} audio={helpers.audio_len(entry)}")
            del prompt

    assert cold and served, "a round produced no output"
    assert set(cold) == set(served), "the two rounds answered different prompts"

    # Round 1 is this engine's first inference, so its waveform carries startup
    # state the later rounds do not; the cross-engine comparison in
    # test_kv_offload_consistency is where the waveform is asserted.
    problems = helpers.compare(cold, served, expect_audio=mode != "kv_only", assert_waveform=False)
    assert not problems, "round 2 diverged from this engine's own cold round:\n" + "\n".join(problems)
