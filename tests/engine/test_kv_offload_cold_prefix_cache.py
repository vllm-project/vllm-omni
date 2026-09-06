# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Does a KV hit that only LMCache can serve leave the talker with stale hidden states?

``_update_states`` marks a request as an in-GPU prefix-cache hit whenever
``num_computed_tokens > 0``, and ``_get_merged_tensors`` then reads the hidden
states for those blocks straight out of ``hidden_states_cache``. That is sound
when the in-GPU cache is what supplied the tokens, but LMCache can supply them
too -- and then those slots were never written.

The earlier consistency runs could not reach that state: round 1 populated the
in-GPU cache, so round 2 read valid hidden states and LMCache loaded nothing.
Here LMCache is given a disk backend so the prefix survives into a *fresh*
engine whose in-GPU cache starts empty.
"""

import pathlib
import tempfile

import pytest

from tests.engine import kv_offload_helpers as helpers

pytestmark = [pytest.mark.advanced_model, pytest.mark.omni, pytest.mark.cuda]

MODEL = "Qwen/Qwen2.5-Omni-3B"

_THINKER = {
    "max_model_len": 1024,
    "max_num_batched_tokens": 1024,
    "gpu_memory_utilization": 0.8,
    "devices": "0",
    "enforce_eager": True,
    "async_chunk": False,
}
_DOWNSTREAM = {
    "1": {"devices": "0", "gpu_memory_utilization": 0.1, "enforce_eager": True},
    "2": {"devices": "0", "gpu_memory_utilization": 0.05, "enforce_eager": True},
}


def _run(*, lmcache: bool, disk: pathlib.Path | None = None) -> dict[str, dict]:
    lmcache_extra = None
    if disk is not None:
        lmcache_extra = {"local_disk": f"file://{disk}/", "max_local_disk_size": 5.0}
    return helpers.run(
        model=MODEL,
        overrides=helpers.stage_overrides(
            lmcache=lmcache,
            prefix_caching=True,
            thinker_extra=_THINKER,
            downstream_extra=_DOWNSTREAM,
            lmcache_extra=lmcache_extra,
        ),
        rounds=1,
    )


def test_lmcache_only_hit_does_not_serve_stale_hidden_states():
    pytest.importorskip("lmcache", reason="lmcache not installed")

    baseline = _run(lmcache=False)

    with tempfile.TemporaryDirectory() as tmp:
        disk = pathlib.Path(tmp)
        # First engine fills the LMCache disk tier, then goes away with its
        # in-GPU cache.
        _run(lmcache=True, disk=disk)
        # Fresh engine: nothing in the in-GPU cache, but LMCache still has the
        # prefix, so a hit here can only come from LMCache.
        cached = _run(lmcache=True, disk=disk)

    assert baseline, "baseline produced no output"
    assert cached, "offload run produced no output"
    assert set(baseline) == set(cached), "the two runs answered different prompts"

    problems = helpers.compare(baseline, cached)
    assert not problems, "a KV hit served only by LMCache diverged from the uncached baseline:\n" + "\n".join(problems)
