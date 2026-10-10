# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Platform-agnostic pieces of the MiniCPM-o Talker multi-frame (K-step) decode.

One Talker decode step is almost all vLLM host work wrapped around a small
device forward, and none of that host work (scheduling, input preparation,
attention metadata, sampling, output assembly, engine-core IPC) is per codec
*frame*. Both multi-frame runners schedule K query positions per request
through stage 1's ``speculative_config`` (n-gram, ``num_speculative_tokens ==
K - 1``) and produce K frames inside one ``execute_model``:

* the NPU runner (``platforms/npu/worker/talker_multiframe.py``) samples the
  codec stream in-model and collapses the vLLM-level head to a two-wide
  continue/stop row (PR #7929);
* the CUDA runner (``worker/gpu_talker_multiframe.py``) keeps the Talker's
  single-frame codec head and vLLM's own sampler, and drives the frames itself.

What both need -- the gate that recognises a uniform multi-frame decode step,
the fail-loud check for one that slipped past it, and the all-or-nothing draft
proposal that gets the next step scheduled K positions wide -- lives here.
"""

from __future__ import annotations

from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

_LOGGED_BLOCKS: set[str] = set()


def _log_block_once(reason: str, detail: str = "") -> None:
    """Say once per distinct reason why the loop did not engage.

    Silence is the worse failure. ``detail`` carries the numbers of the first
    occurrence; later ones with the same reason are not logged again.
    """
    if reason in _LOGGED_BLOCKS:
        return
    _LOGGED_BLOCKS.add(reason)
    suffix = f" ({detail})" if detail else ""
    logger.info("[minicpmo] multi-frame Talker decode not engaged: %s%s", reason, suffix)


def _block(reason: str, detail: str = "") -> int:
    _log_block_once(reason, detail)
    return 0


def applies(
    model: Any,
    model_kwargs_extra: dict[str, Any],
    *,
    require_sampled_frame: bool = True,
) -> int:
    """Frames this step should run, or 0 when the loop must not engage.

    Returns the uniform per-request query length of a pure decode step over a
    Talker that can build its own decode embedding. Anything else -- a prefill,
    a mixed step, a batch whose requests were scheduled different token counts
    -- returns 0 and the caller takes the ordinary single-forward path.

    ``require_sampled_frame`` is the NPU in-model contract: frame 0 embeds the
    codec id the model recorded for the previous frame, so a request that has
    not sampled one yet cannot take the loop. The CUDA runner reads frame 0's
    id from the scheduled tokens instead and passes ``False``.

    Every refusal says so once. A step that schedules several tokens per
    request and then takes the single-forward path is not merely slow: that
    path samples one frame from the *last* row of each span and reports one
    stop row for a step that needs one per position, so it corrupts the codec
    stream and the scheduler's accounting alike. The runners turn those into an
    error rather than letting them run (``is_multi_token_decode``).
    """
    if not getattr(model, "supports_multi_frame_decode", False):
        # Every other stage: not a refusal, just not this model.
        return 0
    spans = model_kwargs_extra.get("request_token_spans")
    infos = model_kwargs_extra.get("model_intermediate_buffer")
    if not spans or not infos or len(spans) != len(infos):
        return _block(
            "no request_token_spans/model_intermediate_buffer for this step",
            f"{len(spans or ())} spans, {len(infos or ())} buffers",
        )
    frames = int(spans[0][1]) - int(spans[0][0])
    if frames <= 1:
        return 0
    for start, end in spans:
        if int(end) - int(start) != frames:
            # A mixed step (one request prefilling, another decoding) has no
            # single frame count, and the captured graph is not a uniform
            # decode either. The first refusal dumps its spans so it can be
            # attributed to an exact scheduling state instead of guessed at.
            return _block(
                "request token spans are not uniform",
                f"frames {frames}, spans {[(int(s), int(e)) for s, e in spans]}",
            )
    for index, info in enumerate(infos):
        if not isinstance(info, dict):
            return _block("a request carries no intermediate buffer", f"request {index} of {len(infos)}")
        if bool(info.get("_omni_is_prefill", False)):
            return 0
        state = info.get("audio_state")
        if not isinstance(state, dict):
            return _block("a request carries no audio_state", f"request {info.get('request_id', index)}")
        if require_sampled_frame and int(state.get("step", 0)) <= 0:
            # The request has not sampled a codec token yet, so there is no
            # previous code for frame 0 to embed and this is not a decode.
            return _block("a request has not sampled a codec token yet", f"request {info.get('request_id', index)}")
    return frames


def is_multi_token_decode(model: Any, model_kwargs_extra: dict[str, Any]) -> bool:
    """True when this step schedules several tokens for a request that is decoding.

    Scoped to the Talker. Stage 0 drafts the Thinker's text with n-grams and so
    schedules multi-token decode steps of its own, which vLLM handles perfectly
    well -- it is only the Talker's one-frame-per-position sampler that cannot.
    """
    if not getattr(model, "supports_multi_frame_decode", False):
        return False
    spans = model_kwargs_extra.get("request_token_spans")
    infos = model_kwargs_extra.get("model_intermediate_buffer")
    if not spans or not infos or len(spans) != len(infos):
        return False
    for (start, end), info in zip(spans, infos):
        if int(end) - int(start) <= 1:
            continue
        if isinstance(info, dict) and not bool(info.get("_omni_is_prefill", False)):
            return True
    return False


def drafts_this_step(runner: Any) -> int:
    """Frames the *next* step should be scheduled for, or 0 to stay generic.

    The count travels to the scheduler as speculative tokens because that is
    vLLM V1's only way of saying "this request advanced by more than one token",
    which is what grows the block table and reserves the KV slots the loop
    writes into. Nothing about it is speculative: the frames are generated
    sequentially rather than verified.
    """
    if not getattr(getattr(runner, "model", None), "supports_multi_frame_decode", False):
        return 0
    num_spec = int(getattr(runner, "num_spec_tokens", 0) or 0)
    if num_spec <= 0:
        return 0
    return num_spec + 1


def constant_drafts(
    valid_sampled_token_ids: Any,
    frames: int,
    num_reqs: int,
    *,
    draft_token: int | None,
) -> list[list[int]]:
    """``frames - 1`` drafts per request -- for the whole batch, or for none of it.

    ``draft_token`` is the id every draft carries; ``None`` repeats each
    request's last sampled id instead, so the scheduled span reads
    ``[last, last, ...]`` and a single-frame preprocess of that span embeds the
    real previous frame (the CUDA runner relies on this).

    The drafts are how the next step gets K query positions per request, and
    the multi-frame loop can only run a step whose requests all have the *same*
    span: it replays one captured uniform-decode graph and samples one codec
    frame per position. So drafting per request is not an option. A request
    that sampled nothing this step (discarded, or finished on this step's stop
    token) would be scheduled one token while its neighbours got K, and
    `applies` would refuse the resulting step -- which the runners turn into a
    fatal error rather than let the single-forward path sample one frame from
    the last row of a K-row span.

    So: every request drafts, or nobody does. A step with no drafts is an
    ordinary one-frame-per-request decode, and the frames resume on the step
    after it.
    """
    rows: list[list[int]] = []
    for index in range(num_reqs):
        sampled = None
        if isinstance(valid_sampled_token_ids, list) and index < len(valid_sampled_token_ids):
            sampled = valid_sampled_token_ids[index]
        if not sampled:
            _log_block_once(
                "a request in this batch sampled nothing; no drafts this step",
                f"request {index} of {num_reqs}",
            )
            return [[] for _ in range(num_reqs)]
        token = int(sampled[-1]) if draft_token is None else int(draft_token)
        rows.append([token] * (frames - 1))
    return rows
