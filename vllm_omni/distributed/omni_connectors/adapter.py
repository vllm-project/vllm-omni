# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import time
from collections.abc import Callable
from typing import Any

from vllm.v1.request import Request

from vllm_omni.metrics import OrchestratorAggregator

from .utils.logging import get_connector_logger

logger = get_connector_logger(__name__)


def try_send_via_connector(
    connector: Any,
    stage_id: int,
    next_stage_id: int,
    req_id: str,
    next_inputs: Any,
    sampling_params: Any,
    original_prompt: Any,
    next_stage_queue_submit_fn: Callable[[dict[str, Any]], None],
    metrics: OrchestratorAggregator,
) -> bool:
    """
    Attempts to send data via OmniConnector.
    Returns True if successful, False otherwise.
    Encapsulates the logic of preparing payload, sending via connector,
    sending notification, and recording metrics.
    """
    try:
        t0 = time.time()

        # Strip non-serializable multimodal feature fields from original_prompt
        # before including it in metadata.  After stage-0 runs, the TokPrompt
        # returned by render_chat_async may carry processed multimodal features
        # (mm_kwargs, mm_placeholders, mm_hashes) that contain MultiModalKwargsItems
        # objects, which are not supported by OmniMsgpackEncoder.  The receiving
        # side (try_recv_via_connector) only extracts "engine_inputs" from the
        # payload and never uses "original_prompt", so stripping these fields
        # only affects debug metadata and is safe.
        _MM_FEATURE_KEYS = frozenset({"mm_kwargs", "mm_placeholders", "mm_hashes"})
        if isinstance(original_prompt, dict) and any(k in original_prompt for k in _MM_FEATURE_KEYS):
            safe_prompt = {k: v for k, v in original_prompt.items() if k not in _MM_FEATURE_KEYS}
        else:
            safe_prompt = original_prompt

        # Prepare data for connector
        payload_data = {
            "engine_inputs": next_inputs,
            "sampling_params": sampling_params,
            "metadata": {
                "original_prompt": safe_prompt,
                "stage_transition": f"{stage_id}->{next_stage_id}",
                "timestamp": time.time(),
            },
        }

        # Send data via connector
        success, serialized_size, metadata = connector.put(str(stage_id), str(next_stage_id), str(req_id), payload_data)

        if success:
            # Send lightweight notification via queue
            notify_payload = {
                "type": "generate",
                "request_id": req_id,
                "sampling_params": sampling_params,
                "from_connector": True,
                "from_stage": str(stage_id),
                "to_stage": str(next_stage_id),
                "sent_ts": time.time(),
            }
            # Merge connector metadata (e.g. shm handle or inline data) into queue payload
            if metadata:
                notify_payload["connector_metadata"] = metadata

            next_stage_queue_submit_fn(notify_payload)

            t1 = time.time()
            tx_ms = (t1 - t0) * 1000.0

            metrics.on_forward(
                stage_id,
                next_stage_id,
                req_id,
                serialized_size,  # Use size from connector
                float(tx_ms),
                True,  # Mark as using connector
            )
            return True
        else:
            # If put returned False, we let the caller handle fallback
            return False

    except Exception as e:
        logger.warning(
            "[Orchestrator] OmniConnector failed for req %s: %s; falling back to queue",
            req_id,
            e,
        )
        return False


def try_recv_via_connector(
    task: dict[str, Any],
    connectors: dict[Any, Any],
    stage_id: int,
) -> tuple[Any, dict[str, Any] | None]:
    """
    Attempts to resolve input data from either connector or IPC.
    Returns (engine_inputs, rx_metrics) or (None, None) if failed/skipped.
    """
    rid = task["request_id"]

    if task.get("from_connector"):
        from_stage = task.get("from_stage")
        to_stage = str(stage_id)

        if not from_stage:
            logger.error(
                "[Stage-%s] 'from_connector' is true but 'from_stage' is missing for request %s", stage_id, rid
            )
            return None, None

        # Get connector for this edge
        connector_key = (from_stage, to_stage)
        connector = connectors.get(connector_key)

        if connector:
            try:
                # Get data from connector with timeout
                _t_start = time.time()
                connector_metadata = task.get("connector_metadata")
                payload = connector.get(from_stage, to_stage, str(rid), metadata=connector_metadata)
                _t_end = time.time()

                if payload:
                    if isinstance(payload, tuple):
                        payload_data, serialized_size = payload
                    else:
                        payload_data = payload
                        serialized_size = len(connector.serialize_obj(payload_data))
                else:
                    payload_data = None
                    serialized_size = 0

                if payload_data and isinstance(payload_data, dict):
                    ein = payload_data.get("engine_inputs")
                    decode_ms = (_t_end - _t_start) * 1000.0

                    rx_metrics = {"rx_decode_time_ms": decode_ms, "rx_transfer_bytes": serialized_size}
                    return ein, rx_metrics
                else:
                    logger.error(
                        "[Stage-%s] Failed to get data from connector for request %s or payload is empty", stage_id, rid
                    )
                    return None, None
            except Exception as e:
                logger.error("[Stage-%s] Error retrieving data from connector for request %s: %s", stage_id, rid, e)
                return None, None
        else:
            logger.error(
                "[Stage-%s] No connector found for edge %s -> %s for request %s", stage_id, from_stage, to_stage, rid
            )
            return None, None
    else:
        # Data comes from queue as usual (e.g. seed request for Stage-0)
        # Since fallback logic is deprecated, we assume this is a direct inputs payload.
        # We still need to decode it if it used SHM (via legacy stage_utils logic, or new shm_connector format)
        # For Stage-0 specifically, 'engine_inputs' is often directly in the task dict.

        # Try to use the new stage_utils which uses OmniSerializer
        from vllm_omni.entrypoints.stage_utils import maybe_load_from_ipc_with_metrics

        try:
            ein, metrics = maybe_load_from_ipc_with_metrics(task, "engine_inputs", "engine_inputs_shm")
            # If metrics are empty or zero, we might want to populate dummy metrics
            return ein, metrics
        except Exception:
            # If engine_inputs is missing, it might be a different kind of payload,
            # but for Stage-0 seed it should be there.
            # We'll return None to let caller handle error if strictly required.
            return None, None


def compute_talker_prompt_ids_length(prompt_ids: list[int]) -> int:
    """Compute the length of the talker prompt ids.

    Args:
        prompt_ids: The prompt ids tensor.

    Returns:
        The length of the talker prompt ids.
    """
    im_start_token_id = 151644
    system_token_id = 8948
    user_token_id = 872
    assistant_token_id = 77091
    im_start_indexes = [i for i in range(len(prompt_ids)) if prompt_ids[i] == im_start_token_id]
    im_start_indexes.append(len(prompt_ids))
    sum_user_len = 0
    assistant_len = 0
    for i in range(len(im_start_indexes) - 1):
        s = im_start_indexes[i]
        e = im_start_indexes[i + 1]
        role = prompt_ids[s + 1]
        if role == system_token_id:
            continue
        elif role == user_token_id:
            sum_user_len += e - s
        elif role == assistant_token_id and i == len(im_start_indexes) - 2:
            assistant_len += 9  # 3 + 4 + 1 + 1
        else:
            pass

    return sum_user_len + assistant_len


def construct_next_stage_streaming_input_prompt(
    payload_data: dict[str, Any],
    request: Request,
    *,
    max_model_len: int | None = None,
    previous_condition_len: int | None = None,
    previous_condition_seq: int | None = None,
    condition_seq: int | None = None,
    recompute_previous_chunks: int = 0,
) -> bool:
    """Update a downstream streaming request prompt from connector payload ids.

    Async-chunk downstream stages are prewarmed before the real Talker prompt is
    known. When a Thinker payload carries ``ids.prompt``, this helper:

    * Preserves ``num_computed_tokens`` while extending a non-window prompt.
      Explicit replacements and one-condition window recomputes reset it.
    * Moves already-computed output tokens into ``prompt_token_ids``.
    * Appends a new placeholder prompt slice sized from the upstream ids.
    * Refreshes block hashes so the scheduler allocates KV slots for the
      extended prompt without discarding prior computed state.
    """
    ids = payload_data.get("ids")
    if not isinstance(ids, dict):
        ids = {}
        payload_data["ids"] = ids
    meta = payload_data.get("meta")
    if not isinstance(meta, dict):
        meta = {}
        payload_data["meta"] = meta
    # This flag is transported through a merge-based runtime buffer. Set it on
    # every condition so a prior rollover cannot be replayed by a later append.
    meta["streaming_prompt_recompute"] = False
    next_stage_prompt_len = meta.get("next_stage_prompt_len")
    if "next_stage_generation_tokens" in meta:
        next_generation_tokens = meta["next_stage_generation_tokens"]
        if (
            not isinstance(next_generation_tokens, int)
            or isinstance(next_generation_tokens, bool)
            or next_generation_tokens <= 0
        ):
            raise ValueError("next_stage_generation_tokens must be a positive integer")
        generation_reserve = next_generation_tokens
    else:
        generation_reserve = 0
    capacity_limit = (
        max_model_len
        if isinstance(max_model_len, int) and not isinstance(max_model_len, bool) and max_model_len > 0
        else None
    )
    managed_prompt_len: int | None = None
    if generation_reserve > 0 and capacity_limit is not None:
        if (
            not isinstance(next_stage_prompt_len, int)
            or isinstance(next_stage_prompt_len, bool)
            or next_stage_prompt_len <= 0
        ):
            raise ValueError("capacity-managed streaming prompt requires a positive next_stage_prompt_len")
        managed_prompt_len = next_stage_prompt_len
        if managed_prompt_len + generation_reserve > capacity_limit:
            raise ValueError(
                "fresh streaming prompt plus generation reserve exceeds max_model_len: "
                f"prompt={managed_prompt_len}, reserve={generation_reserve}, limit={capacity_limit}"
            )
    explicit_replacement = meta.get("replace_streaming_prompt") is True
    if isinstance(recompute_previous_chunks, bool) or recompute_previous_chunks not in (0, 1):
        raise ValueError("streaming prompt recompute supports exactly one previous chunk")
    has_window_contract = recompute_previous_chunks == 1
    window_recompute = not explicit_replacement and has_window_contract and previous_condition_seq is not None
    exceeds_accumulated_capacity = (
        managed_prompt_len is not None
        and capacity_limit is not None
        and request.num_computed_tokens + managed_prompt_len + generation_reserve > capacity_limit
    )
    if not explicit_replacement and exceeds_accumulated_capacity and not window_recompute:
        raise ValueError("capacity-managed streaming prompt rollover requires window_size=1")
    replacement_prompt_len = next_stage_prompt_len
    if window_recompute:
        if managed_prompt_len is None or capacity_limit is None:
            raise ValueError("streaming prompt window requires a positive generation reserve and max_model_len")
        if (
            not isinstance(previous_condition_len, int)
            or isinstance(previous_condition_len, bool)
            or previous_condition_len <= 0
            or not isinstance(previous_condition_seq, int)
            or isinstance(previous_condition_seq, bool)
            or previous_condition_seq < 0
            or not isinstance(condition_seq, int)
            or isinstance(condition_seq, bool)
            or condition_seq < 0
        ):
            raise ValueError("streaming prompt recompute requires condition lengths and monotonic sequence metadata")
        if condition_seq != previous_condition_seq + 1:
            raise ValueError(
                "streaming prompt recompute skipped a Talker condition: "
                f"previous={previous_condition_seq}, current={condition_seq}"
            )

        num_placeholders = int(getattr(request, "num_output_placeholders", 0) or 0)
        confirmed_num_computed_tokens = max(0, int(request.num_computed_tokens) - num_placeholders)
        if confirmed_num_computed_tokens < request.num_prompt_tokens:
            raise ValueError(
                "confirmed streaming output ends before the current condition: "
                f"confirmed={confirmed_num_computed_tokens}, prompt={request.num_prompt_tokens}"
            )
        previous_codec_ids = list(request._all_token_ids[request.num_prompt_tokens : confirmed_num_computed_tokens])
        max_confirmed_codec_tokens = generation_reserve - 1
        if len(previous_codec_ids) > max_confirmed_codec_tokens:
            raise ValueError(
                "streaming prompt recompute retained too many codec tokens: "
                f"retained={len(previous_codec_ids)}, limit={max_confirmed_codec_tokens}"
            )

        replacement_prompt_len = previous_condition_len + len(previous_codec_ids) + managed_prompt_len
        if replacement_prompt_len + generation_reserve > capacity_limit:
            raise ValueError(
                "sliding streaming prompt plus generation reserve exceeds max_model_len: "
                f"previous={previous_condition_len}, codec={len(previous_codec_ids)}, "
                f"current={managed_prompt_len}, reserve={generation_reserve}, limit={capacity_limit}"
            )
        ids["streaming_prompt_previous_codes"] = previous_codec_ids
        meta.update(
            {
                "streaming_prompt_recompute": True,
                "streaming_condition_seq": condition_seq,
            }
        )
        logger.debug(
            "Recomputing a one-chunk streaming prompt window: previous=%d, codec=%d, current=%d, reserve=%d, limit=%d",
            previous_condition_len,
            len(previous_codec_ids),
            managed_prompt_len,
            generation_reserve,
            capacity_limit,
        )

    replace_prompt = explicit_replacement or window_recompute
    if replace_prompt and isinstance(replacement_prompt_len, int) and replacement_prompt_len > 0:
        # Some downstream stages consume complete, independently conditioned
        # segments instead of extending an existing KV prefix. The producer
        # declares that transport behavior explicitly in payload metadata.
        new_prompt = [0] * replacement_prompt_len
        request._output_token_ids.clear()
        request._all_token_ids.clear()
        request._all_token_ids.extend(new_prompt)
        request.prompt_token_ids = new_prompt
        request.num_computed_tokens = 0
        request.num_prompt_tokens = replacement_prompt_len
        request.update_block_hashes()
        return True
    prompt_token_ids = ids.get("prompt", None)
    if not prompt_token_ids:
        return False
    num_computed_tokens = request.num_computed_tokens
    kept_output_tokens = request._all_token_ids[request.num_prompt_tokens : num_computed_tokens]
    del request._all_token_ids[num_computed_tokens:]
    request._output_token_ids.clear()
    assert request.prompt_token_ids is not None
    # Extend prompt with kept output tokens.
    request.prompt_token_ids.extend(kept_output_tokens)
    if isinstance(next_stage_prompt_len, int) and next_stage_prompt_len > 0:
        next_prompt_len = next_stage_prompt_len
    else:
        next_prompt_len = max(1, compute_talker_prompt_ids_length(prompt_token_ids))
    new_prompt = [0] * next_prompt_len
    request._all_token_ids.extend(new_prompt or ())
    request.prompt_token_ids.extend(new_prompt or ())
    request.update_block_hashes()
    request.num_prompt_tokens = len(request.prompt_token_ids)
    return False
