import logging

import torch
from vllm.inputs import TextPrompt

from vllm_omni.data_entry_keys import (
    EmbeddingsStruct,
    HiddenStatesStruct,
    IdsStruct,
    OmniPayload,
    OmniPayloadStruct,
    to_dict,
)
from vllm_omni.inputs.data import OmniTokensPrompt

logger = logging.getLogger(__name__)

TALKER_CODEC_PAD_TOKEN_ID = 8292
TALKER_CODEC_START_TOKEN_ID = 8293
TALKER_CODEC_END_TOKEN_ID = 8294


def thinker2talker(
    source_outputs,
    prompt: OmniTokensPrompt | TextPrompt = None,
    requires_multimodal_data: bool = False,
):
    thinker_outputs = source_outputs
    talker_inputs = []
    if not isinstance(prompt, list):
        prompt = [prompt]
    multi_modal_data = {
        thinker_output.request_id: p.get("multi_modal_data", None) for thinker_output, p in zip(thinker_outputs, prompt)
    }

    for i, thinker_output in enumerate(thinker_outputs):
        output = thinker_output.outputs[0]
        prompt_token_ids = thinker_output.prompt_token_ids
        thinker_output_ids = output.cumulative_token_ids
        prompt_token_ids_len = len(prompt_token_ids)
        mm: OmniPayload = output.multimodal_output
        latent = mm["latent"]
        thinker_hidden_states = latent.clone().detach().to(latent.device)
        decode_hidden = thinker_hidden_states[prompt_token_ids_len:].to(torch.float32)
        prefill_hidden = thinker_hidden_states[:prompt_token_ids_len].to(torch.float32)
        additional_information = to_dict(
            OmniPayloadStruct(
                hidden_states=HiddenStatesStruct(output=decode_hidden, output_shape=list(decode_hidden.shape)),
                embed=EmbeddingsStruct(prefill=prefill_hidden, prefill_shape=list(prefill_hidden.shape)),
                ids=IdsStruct(prompt=list(prompt_token_ids), output=list(thinker_output_ids)),
            )
        )
        talker_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[TALKER_CODEC_START_TOKEN_ID]
                + [TALKER_CODEC_PAD_TOKEN_ID] * (len(prompt_token_ids))
                + [TALKER_CODEC_END_TOKEN_ID],
                additional_information=additional_information,
                multi_modal_data=(
                    multi_modal_data[thinker_output.request_id]
                    if requires_multimodal_data and multi_modal_data is not None
                    else None
                ),
                mm_processor_kwargs=None,
            )
        )
    return talker_inputs


def talker2code2wav(
    source_outputs,
    _prompt: OmniTokensPrompt | TextPrompt = None,
    _requires_multimodal_data: bool = False,
):
    code2wav_inputs = []
    for talker_output in source_outputs:
        output = talker_output.outputs[0]
        token_ids = list(output.cumulative_token_ids)
        if token_ids and token_ids[0] == TALKER_CODEC_START_TOKEN_ID:
            token_ids = token_ids[1:]
        if token_ids and token_ids[-1] == TALKER_CODEC_END_TOKEN_ID:
            token_ids = token_ids[:-1]
        if not token_ids:
            continue
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=token_ids,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return code2wav_inputs


# ============================================================================
# Worker-connector data plane (non-async-chunk path).
# Both transitions ship payloads via the worker connector
# (registered in ``_FULL_PAYLOAD_INPUT_STAGES`` in
# omni_scheduling_coordinator):
# - thinker->talker reads accumulated ``pooling_output["hidden"]`` and
#   packs an OmniPayload-shaped dict (embed.prefill /
#   hidden_states.output / ids.prompt / ids.output) for the talker, which
#   the talker's ``talker_preprocess`` reads from
#   ``model_intermediate_buffer``.  The shape matches what legacy
#   ``thinker2talker`` writes into ``additional_information`` as a debug
#   fallback; ``thinker2talker_token_only`` only allocates prompt slots.
# - talker->code2wav strips TALKER_CODEC_{START,END} boundary tokens
#   and ships the codec token ids.
# ============================================================================

# Per-model REPLACE-keys for the full-payload accumulator.  qwen2_5_omni's
# producer side does not emit model_outputs through pooler_output (it ships
# token_ids on the request directly), so the empty set preserves correctness.
_FULL_PAYLOAD_REPLACE_KEYS: frozenset[str] = frozenset()


def _strip_codec_boundaries(token_ids: list[int]) -> list[int]:
    """Keep only real codec ids for the code2wav stage.

    The talker stream can contain prompt/control ids (START/PAD/END/MASK) in
    addition to sampled codec ids.  Code2wav expects codec ids only; carrying
    the prompt PAD span forward can inflate the sequence enough to OOM on L4.
    Async scheduling may also leave trailing ``-1`` placeholders, so preserve
    their length by repeating the last valid codec id.
    """
    tids = list(token_ids)
    trailing_placeholder_count = 0
    while trailing_placeholder_count < len(tids) and tids[-1 - trailing_placeholder_count] == -1:
        trailing_placeholder_count += 1

    if tids and tids[-1] == TALKER_CODEC_END_TOKEN_ID:
        tids = tids[:-1]
        trailing_placeholder_count = 0

    codec_ids = [tid for tid in tids if 0 <= tid < TALKER_CODEC_PAD_TOKEN_ID]
    if trailing_placeholder_count > 0 and codec_ids:
        codec_ids.extend([codec_ids[-1]] * trailing_placeholder_count)
    return codec_ids


def talker2code2wav_token_only(
    source_outputs,
    _prompt: OmniTokensPrompt | TextPrompt = None,
    _requires_multimodal_data: bool = False,
):
    """Sync-side placeholder for Stage-2 input (code2wav).

    Returns OmniTokensPrompt sized to the stripped codec token count.
    Actual codec ids are delivered via the worker connector payload built
    by ``talker2code2wav_full_payload``.
    """
    code2wav_inputs = []
    for talker_output in source_outputs:
        output = talker_output.outputs[0]
        token_ids = _strip_codec_boundaries(list(output.cumulative_token_ids))
        if not token_ids:
            continue
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0] * len(token_ids),
                additional_information=None,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return code2wav_inputs


def talker2code2wav_full_payload(
    transfer_manager,
    pooling_output: dict,
    request,
) -> dict | None:
    """Producer-side payload builder: ship the stripped codec ids via connector.

    Token-ids-only shape.  The talker stage's output already
    carries the codec ids on ``request.output_token_ids``; we strip the
    boundary tokens and pack a minimal payload.
    """
    del transfer_manager
    rid = getattr(request, "request_id", "?")
    token_ids = list(getattr(request, "output_token_ids", None) or [])
    if not token_ids:
        logger.warning(
            "qwen2_5_omni.talker2code2wav_full_payload: empty output_token_ids "
            "for req=%s; consumer wait gate may hang.",
            rid,
        )
        return None
    token_ids = _strip_codec_boundaries(token_ids)
    if not token_ids:
        logger.warning(
            "qwen2_5_omni.talker2code2wav_full_payload: codec ids empty after "
            "stripping boundary tokens for req=%s; consumer wait gate may hang.",
            rid,
        )
        return None
    return {
        "codes": {"audio": token_ids},
        "meta": {"finished": torch.tensor(True, dtype=torch.bool)},
    }


# ============================================================================
# Worker-connector data plane (non-async-chunk path) -- thinker->talker.
#
# qwen2_5_omni's talker consumes the thinker's last-layer hidden state
# via Linear(3584, 896).  The AR runner publishes those hidden states
# per decode step on ``pooling_output["hidden"]`` (unpacked from
# ``OmniOutput.text_hidden_states``); the full-payload accumulator
# concatenates them so ``thinker2talker_full_payload`` sees the full
# prefill+decode trajectory and packs an OmniPayload-shaped dict that
# the talker's ``talker_preprocess`` reads from
# ``model_intermediate_buffer``.  ``thinker2talker_token_only`` only
# allocates the talker's codec prompt slots; legacy
# ``thinker2talker`` above remains as a debug fallback that bundles the
# same shape into ``additional_information``.
# ============================================================================


def thinker2talker_token_only(
    source_outputs,
    prompt: OmniTokensPrompt | TextPrompt = None,
    requires_multimodal_data: bool = False,
):
    """Placeholder builder for the connector-driven thinker->talker path.

    Allocates the TALKER_CODEC_{START,PAD,END} prompt slots sized to the
    thinker prompt length and forwards ``multi_modal_data``.  The bulk
    payload (hidden_states / embed / ids) ships exclusively through
    ``thinker2talker_full_payload`` via the worker connector and lands
    in ``model_intermediate_buffer`` before the talker's forward() runs.

    Consumer-wait gating is whitelist-driven via
    ``_FULL_PAYLOAD_INPUT_STAGES`` (see the mixin
    ``should_accumulate_full_payload_output`` docstring).
    """
    thinker_outputs = source_outputs
    talker_inputs = []
    if not isinstance(prompt, list):
        prompt = [prompt]
    multi_modal_data = {
        thinker_output.request_id: p.get("multi_modal_data", None) for thinker_output, p in zip(thinker_outputs, prompt)
    }

    for thinker_output in thinker_outputs:
        prompt_token_ids = thinker_output.prompt_token_ids
        talker_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[TALKER_CODEC_START_TOKEN_ID]
                + [TALKER_CODEC_PAD_TOKEN_ID] * (len(prompt_token_ids))
                + [TALKER_CODEC_END_TOKEN_ID],
                additional_information=None,
                multi_modal_data=(
                    multi_modal_data[thinker_output.request_id]
                    if requires_multimodal_data and multi_modal_data is not None
                    else None
                ),
                mm_processor_kwargs=None,
            )
        )

    return talker_inputs


def thinker2talker_full_payload(
    transfer_manager,
    pooling_output,
    request,
):
    """Producer-side payload builder for the worker-connector data plane.

    The AR runner emits per-step ``pooling_output["hidden"]`` (the
    thinker's last-layer hidden states for the request span, unpacked
    from ``OmniOutput.text_hidden_states``).  The full-payload
    accumulator concatenates those per-step rows across decode steps, so
    by the time this builder fires the materialized
    ``pooling_output["hidden"]`` contains the full prefill+decode
    hidden-state trajectory of size
    ``len(prompt_token_ids) + len(output_token_ids)``.

    We split it at ``len(prompt_token_ids)`` into prefill embeddings and
    decode hidden states, then pack the ``OmniPayload``-shaped dict that
    the talker's ``thinker_to_talker_process`` reads from
    ``model_intermediate_buffer`` (keys ``embed.prefill`` /
    ``hidden_states.output`` / ``ids.prompt`` / ``ids.output``).  Shape
    matches what legacy ``thinker2talker`` writes into
    ``additional_information`` as a debug fallback, so the talker
    consumes the same payload layout from either path.

    Like ``qwen3_omni.thinker2talker_full_payload``, we apply a
    finish-reason-aware stop-row trim: vLLM v1 appends the sampled
    token to ``output_token_ids`` before ``check_stop``, so a request
    that finished via ``FINISHED_STOPPED`` has one extra accumulated
    hidden-state row that the talker must not consume.  Max-token
    finishes need no drop.  Status is read from the request when
    available; otherwise we fall back to a last-token-in-stop-set
    heuristic.
    """
    del transfer_manager
    rid = getattr(request, "request_id", "?")
    if not isinstance(pooling_output, dict):
        logger.warning(
            "qwen2_5_omni.thinker2talker_full_payload: pooling_output not a dict "
            "(type=%s) for req=%s; consumer wait gate may hang.",
            type(pooling_output).__name__,
            rid,
        )
        return None

    hidden = pooling_output.get("hidden")
    if not isinstance(hidden, torch.Tensor):
        logger.warning(
            "qwen2_5_omni.thinker2talker_full_payload: missing 'hidden' tensor "
            "(keys=%s) for req=%s; consumer wait gate may hang.",
            list(pooling_output.keys()),
            rid,
        )
        return None

    def _ensure_list(x):
        if x is None:
            return []
        if hasattr(x, "_x"):
            # vLLM wraps cached token-id lists in ConstantList-like objects.
            return list(x._x)
        if isinstance(x, list):
            return list(x)
        return list(x)

    prompt_token_ids = _ensure_list(getattr(request, "prompt_token_ids", None))
    output_token_ids = _ensure_list(getattr(request, "output_token_ids", None))
    all_token_ids = _ensure_list(getattr(request, "all_token_ids", None) or [])
    if not all_token_ids:
        all_token_ids = list(prompt_token_ids) + list(output_token_ids)

    # Length-aware trim of accumulated thinker output, finish-reason-aware.
    # Mirror qwen3_omni.thinker2talker_full_payload's logic so a stop-finish
    # does not leak an extra hidden-state row to the talker.
    status = getattr(request, "status", None)
    status_name = getattr(status, "name", None) or ""
    if not status_name and status is not None:
        status_name = str(status).rsplit(".", 1)[-1]
    stop_emission_drop = 1 if status_name == "FINISHED_STOPPED" else 0
    if stop_emission_drop == 0 and not status_name and output_token_ids:
        # Worker-side CachedRequestState has no `.status` field in vLLM v1;
        # fall back to a last-token-in-stop-set heuristic.
        sampling_params = getattr(request, "sampling_params", None)
        if sampling_params is not None:
            stop_ids: set[int] = set()
            ignore_eos = bool(getattr(sampling_params, "ignore_eos", False))
            for sid in getattr(sampling_params, "stop_token_ids", None) or ():
                if isinstance(sid, int):
                    stop_ids.add(sid)
            if not ignore_eos:
                for eos in (
                    getattr(sampling_params, "eos_token_id", None),
                    getattr(sampling_params, "_eos_token_id", None),
                ):
                    if isinstance(eos, int):
                        stop_ids.add(eos)
                for sid in (
                    getattr(sampling_params, "all_stop_token_ids", None)
                    or getattr(sampling_params, "_all_stop_token_ids", None)
                    or ()
                ):
                    if isinstance(sid, int):
                        stop_ids.add(sid)
            if stop_ids and output_token_ids[-1] in stop_ids:
                stop_emission_drop = 1

    # Trim accumulated thinker output based on stop_emission_drop computed
    # above.  Mirror qwen3_omni.thinker2talker_full_payload's contract:
    #   target_rows = len(all_token_ids) - stop_emission_drop
    # which excludes the stop-emission row for FINISHED_STOPPED but keeps
    # all rows for FINISHED_LENGTH_CAPPED (max_tokens) finishes.
    if stop_emission_drop > 0 and len(output_token_ids) >= stop_emission_drop:
        output_token_ids = output_token_ids[:-stop_emission_drop]
    h = hidden.detach().cpu().to(torch.float32)
    target_rows = max(0, len(all_token_ids) - stop_emission_drop)
    if target_rows <= 0:
        logger.warning(
            "qwen2_5_omni.thinker2talker_full_payload: target_rows<=0 "
            "(all_token_ids=%d, stop_drop=%d) for req=%s; nothing to ship.",
            len(all_token_ids),
            stop_emission_drop,
            getattr(request, "request_id", "?"),
        )
        return None
    if h.dim() >= 1 and h.shape[0] > target_rows:
        logger.warning(
            "qwen2_5_omni.thinker2talker_full_payload: excess hidden rows "
            "(got %d, target %d, stop_drop %d) for req=%s; trimming",
            int(h.shape[0]),
            target_rows,
            stop_emission_drop,
            getattr(request, "request_id", None),
        )
        h = h[:target_rows]

    prompt_len = len(prompt_token_ids)
    if h.shape[0] < prompt_len:
        # Under-captured prefill -- defensively skip rather than ship a
        # truncated payload that would confuse the talker's prefill path.
        logger.warning(
            "qwen2_5_omni.thinker2talker_full_payload: hidden rows=%d < prompt_len=%d "
            "for req=%s; under-captured prefill, skipping payload.",
            int(h.shape[0]),
            prompt_len,
            getattr(request, "request_id", "?"),
        )
        return None

    prefill_hidden = h[:prompt_len]
    decode_hidden = h[prompt_len:]

    payload: OmniPayload = to_dict(
        OmniPayloadStruct(
            hidden_states=HiddenStatesStruct(
                output=decode_hidden,
                output_shape=list(decode_hidden.shape),
            ),
            embed=EmbeddingsStruct(
                prefill=prefill_hidden,
                prefill_shape=list(prefill_hidden.shape),
            ),
            ids=IdsStruct(
                prompt=list(prompt_token_ids),
                output=list(output_token_ids),
            ),
        )
    )
    # Intentionally omit payload["meta"]: the thinker->talker transition
    # carries no scheduler-relevant metadata (next_stage_prompt_len /
    # left_context_size are not set on this edge).
    return payload
