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

The scheduler keeps seeded requests out of every co-batch: a seeded request
waits for the engine to drain, then runs alone, and no other request is admitted
while it runs. "Waiting" covers both the regular wait queue and
``kv_holding_waiting``, so a seeded request that resumes from KV retention cannot
join a running unseeded request either.

Unseeded traffic batches as before when no seeded request is in flight; while a
seeded request waits for the engine to drain, unseeded requests that arrive
alongside it wait too. Measured cost: 8 concurrent seeded requests take seconds
instead of under one (3.6 s on an RTX 4080 SUPER; 5.9 s on an RTX 6000 Ada in an
independent run), while 8 unseeded requests alone stay at 0.8-1.0 s.

The request opts in through the ``omni_user_seed`` marker that the speech
serving path sets when the client supplies a seed; stages that copy a
``default_sampling_params`` seed, and the other entrypoints that do the same,
stay unmarked. There is no process or engine switch. This module only decides *when* to isolate, so it can be
unit-tested without a GPU or a scheduler instance.
"""

from __future__ import annotations


def isolation_plan(
    *,
    has_running: bool,
    seeded_running: bool,
    seeded_waiting: bool,
) -> tuple[bool, bool]:
    """Decide this step's admission behaviour.

    Returns ``(defer_all_admission, promote_seeded_request)``:

    * ``defer_all_admission`` - admit nothing this step, so the running set can
      drain (a waiting seeded request must not join it) or so the running
      seeded request stays alone.
    * ``promote_seeded_request`` - move the oldest waiting seeded request to the
      front of the queue and admit exactly one request, so it starts alone.
    """
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
    """Whether the client asked for this request to be reproducible.

    A request whose ``seed`` came from a stage's ``default_sampling_params`` is
    not an opt-in: those seeds apply to every request on that deployment, so
    isolating them would serialize the whole engine. The serving layer marks a
    client-supplied seed with ``extra_args["omni_user_seed"]``; anything
    unmarked keeps normal batching.
    """
    if sampling_params is None:
        return False
    extra_args = getattr(sampling_params, "extra_args", None)
    if not isinstance(extra_args, dict):
        return False
    # Only an explicit client marker opts in, and only together with a seed:
    # every other path - stage defaults, realtime responses, offline fallbacks -
    # leaves the seed unmarked and keeps normal batching, so one entrypoint
    # forgetting to mark cannot serialize an engine. Requiring the seed also
    # rejects a marker smuggled through ``extra_params`` without one.
    return bool(extra_args.get("omni_user_seed")) and getattr(sampling_params, "seed", None) is not None
