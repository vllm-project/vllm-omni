# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Admission policy for requests that carry an explicit sampling seed.

A seeded request is only bit-reproducible when it never shares an autoregressive
batch with another request: batched GEMM reductions change the logits at bf16
level, which flips sampled tokens (and EOS) even under greedy decoding. Measured
on Qwen3-TTS-1.7B, one RTX 4080 SUPER: the same (text, seed) is byte-identical at
concurrency 1, but at concurrency 2/4/8 it produces 2-3 distinct outputs - with
CUDA graphs off, without the streaming connector, and with greedy decoding, so
neither the graph path, the connector, nor the sampler RNG is responsible.

With ``VLLM_OMNI_ISOLATE_SEEDED_REQUESTS=1`` the scheduler keeps seeded requests
out of every co-batch: a seeded request waits for the engine to drain, then runs
alone, and no other request is admitted while it runs. "Waiting" covers both the
regular wait queue and ``kv_holding_waiting``, so a seeded request that resumes
from KV retention cannot join a running unseeded request either. Unseeded traffic keeps
batching as before. Measured cost on the same card: 8 concurrent seeded requests
take 3.25 s instead of 0.77 s, while 8 unseeded requests stay at 0.83 s.

The policy is off by default; this module only decides *when* to isolate, so it
can be unit-tested without a GPU or a scheduler instance.
"""

from __future__ import annotations

import os
from typing import Final

ENV_VAR: Final = "VLLM_OMNI_ISOLATE_SEEDED_REQUESTS"
_DISABLED: Final = frozenset({"", "0", "false", "off", "no"})


def seeded_isolation_enabled() -> bool:
    """Whether seeded request isolation is switched on for this process."""
    return os.environ.get(ENV_VAR, "0").strip().lower() not in _DISABLED


def isolation_plan(
    *,
    has_running: bool,
    seeded_running: bool,
    seeded_waiting: bool,
    enabled: bool | None = None,
) -> tuple[bool, bool]:
    """Decide this step's admission behaviour.

    Returns ``(defer_all_admission, promote_seeded_request)``:

    * ``defer_all_admission`` - admit nothing this step, so the running set can
      drain (a waiting seeded request must not join it) or so the running
      seeded request stays alone.
    * ``promote_seeded_request`` - move the oldest waiting seeded request to the
      front of the queue and admit exactly one request, so it starts alone.
    """
    if enabled is None:
        enabled = seeded_isolation_enabled()
    if not enabled:
        return False, False
    if seeded_running:
        # The seeded request owns the engine until it finishes.
        return True, False
    if not seeded_waiting:
        return False, False
    if has_running:
        # Wait for the current occupants; admitting now would co-batch it.
        return True, False
    return False, True


def request_is_seeded(sampling_params: object | None) -> bool:
    """Whether a request asked for an explicit seed."""
    return sampling_params is not None and getattr(sampling_params, "seed", None) is not None
