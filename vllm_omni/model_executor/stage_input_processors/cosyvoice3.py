# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import defaultdict
from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
from vllm.inputs import TextPrompt
from vllm.logger import init_logger

from vllm_omni.data_entry_keys import (
    CodesStruct,
    EmbeddingsStruct,
    MetaStruct,
    OmniPayloadStruct,
)
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.models.cosyvoice3.utils import unpad_prompt_conditioning

logger = init_logger(__name__)


def _build_prompt_embed_struct(prompt_payload: dict[str, Any]) -> EmbeddingsStruct | None:
    """Wrap prompt_payload's flat speech_token/speech_feat/embedding tensors into EmbeddingsStruct."""
    speech_token = prompt_payload.get("speech_token")
    speech_feat = prompt_payload.get("speech_feat")
    embedding = prompt_payload.get("embedding")
    if speech_token is None and speech_feat is None and embedding is None:
        return None
    return EmbeddingsStruct(
        speech_token=speech_token,
        speech_feat=speech_feat,
        embedding=embedding,
    )


def _ensure_list(x: Any) -> list[Any]:
    if hasattr(x, "_x"):
        return list(x._x)
    if isinstance(x, list):
        return list(x)
    if isinstance(x, tuple):
        return list(x)
    if x is None:
        return []
    try:
        return list(x)
    except TypeError:
        return [x]


def _to_token_id_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        value = value.detach().to("cpu").reshape(-1).tolist()
    token_ids: list[int] = []
    for item in _ensure_list(value):
        if isinstance(item, torch.Tensor):
            token_ids.extend(_to_token_id_list(item))
            continue
        if isinstance(item, (list, tuple)):
            token_ids.extend(_to_token_id_list(item))
            continue
        token_ids.append(int(item))
    return token_ids


def _strip_prompt_prefix(output_ids: list[Any], prefix_ids: list[Any]) -> list[Any]:
    if prefix_ids and len(output_ids) >= len(prefix_ids) and output_ids[: len(prefix_ids)] == prefix_ids:
        return output_ids[len(prefix_ids) :]
    return output_ids


def _prompt_speech_token_ids(multi_modal_data: dict[str, Any]) -> list[int]:
    speech_token = multi_modal_data.get("speech_token")
    if speech_token is None:
        embed = multi_modal_data.get("embed")
        if isinstance(embed, dict):
            speech_token = embed.get("speech_token")
    return _to_token_id_list(speech_token)


def _to_cpu_tensor(x: Any) -> torch.Tensor | None:
    if isinstance(x, list):
        if not x:
            return None
        x = x[0]
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return None


def _decode_additional_information(raw_info: Any) -> dict[str, Any]:
    if raw_info is None:
        return {}
    if isinstance(raw_info, dict):
        return raw_info

    entries = getattr(raw_info, "entries", None)
    if not isinstance(entries, dict):
        return {}

    decoded: dict[str, Any] = {}
    for key, entry in entries.items():
        tensor_data = getattr(entry, "tensor_data", None)
        if tensor_data is not None:
            dtype_name = getattr(entry, "tensor_dtype", "float32")
            tensor_shape = getattr(entry, "tensor_shape", None)
            if tensor_shape is None:
                continue
            dt = np.dtype(dtype_name)
            arr = np.frombuffer(tensor_data, dtype=dt).reshape(tensor_shape)
            decoded[key] = torch.from_numpy(arr.copy())
        else:
            decoded[key] = getattr(entry, "list_data", None)
    return decoded


def text2flow(
    source_outputs: list[Any],
    prompt: OmniTokensPrompt | TextPrompt = None,
    _requires_multimodal_data: bool = True,
):
    """Build stage-1 inputs by prefixing stage-0 prompt ids to its outputs."""
    engine_inputs: list[OmniTokensPrompt] = []
    for source_output in source_outputs:
        output = source_output.outputs[0]
        multi_modal_data = output.multimodal_output
        if multi_modal_data is None:
            raise RuntimeError(f"Missing multimodal_output for request {source_output.request_id}")

        prefix_ids = _ensure_list(source_output.prompt_token_ids)
        raw_output_ids = _ensure_list(output.cumulative_token_ids)
        prompt_speech_ids = _prompt_speech_token_ids(multi_modal_data)
        output_ids = _strip_prompt_prefix(raw_output_ids, prefix_ids)
        output_ids = _strip_prompt_prefix(output_ids, prompt_speech_ids)
        additional_info = dict(multi_modal_data)
        additional_info.setdefault("ids", {})["prompt"] = prefix_ids
        engine_inputs.append(OmniTokensPrompt(prompt_token_ids=output_ids, additional_information=additional_info))
    return engine_inputs


def talker2code2wav_async_chunk(
    transfer_manager: Any,
    multimodal_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """CosyVoice3 async_chunk processor: talker token stream -> code2wav chunks."""
    with nullcontext():
        request_id = request.external_req_id
        finished = bool(is_finished or request.is_finished())

        connector = getattr(transfer_manager, "connector", None)
        raw_cfg = getattr(connector, "config", {}) or {}
        cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
        chunk_size = int(cfg.get("codec_chunk_frames", 25))
        code_vocab_size = int(cfg.get("codec_vocab_size", 6561))
        pre_lookahead_len = int(cfg.get("codec_pre_lookahead_frames", 3))
        max_chunk_size = int(cfg.get("codec_max_chunk_frames", 4 * chunk_size))
        stream_scale_factor = int(cfg.get("codec_stream_scale_factor", 2))
        if chunk_size <= 0 or pre_lookahead_len < 0 or max_chunk_size <= 0 or stream_scale_factor <= 0:
            raise ValueError(
                f"Invalid codec chunk config: codec_chunk_frames={chunk_size}, "
                f"codec_pre_lookahead_frames={pre_lookahead_len}, "
                f"codec_max_chunk_frames={max_chunk_size}, "
                f"codec_stream_scale_factor={stream_scale_factor}"
            )

        request_state = transfer_manager.request_payload.get(request_id)
        if not isinstance(request_state, dict) or "_cosyvoice3_async_state" not in request_state:
            with nullcontext():
                info = _decode_additional_information(getattr(request, "additional_information", None))
                info_embed = info.get("embed", {}) if isinstance(info, dict) else {}
                prompt_payload = {}
                cond_keys = ("speech_token", "speech_feat", "embedding", "speech_token_len")
                for key in cond_keys:
                    value = _to_cpu_tensor(info_embed.get(key))
                    if value is not None:
                        prompt_payload[key] = value
                if isinstance(multimodal_output, Mapping):
                    mm_embed = multimodal_output.get("embed", {})
                    if not isinstance(mm_embed, Mapping):
                        mm_embed = multimodal_output
                    for key in cond_keys:
                        if key in prompt_payload:
                            continue
                        value = _to_cpu_tensor(mm_embed.get(key))
                        if value is not None:
                            prompt_payload[key] = value
                # Drop any right-padding carried from batched talker emission so
                # the chunk-routing math (prompt_token_len) and the conditioning
                # sent to code2wav use the true prompt length.
                if "speech_token" in prompt_payload:
                    st_unpad, sf_unpad = unpad_prompt_conditioning(
                        prompt_payload.get("speech_token"),
                        prompt_payload.get("speech_feat"),
                        prompt_payload.pop("speech_token_len", None),
                    )
                    prompt_payload["speech_token"] = st_unpad
                    if sf_unpad is not None:
                        prompt_payload["speech_feat"] = sf_unpad
                prompt_token = prompt_payload.get("speech_token")
                prompt_token_len = (
                    int(prompt_token.shape[1])
                    if isinstance(prompt_token, torch.Tensor) and prompt_token.ndim >= 2
                    else 0
                )
                prompt_token_pad = (
                    ((prompt_token_len + chunk_size - 1) // chunk_size) * chunk_size - prompt_token_len
                    if prompt_token_len > 0
                    else 0
                )
            request_state = {
                "_cosyvoice3_async_state": {
                    "seen_len": 0,
                    "sent_prompt": False,
                    "emitted_chunks": 0,
                    "emitted_token_len": 0,
                    "token_hop_len": chunk_size,
                    "prompt_token_pad": prompt_token_pad,
                    "pre_lookahead_len": pre_lookahead_len,
                    "token_max_hop_len": max(chunk_size, max_chunk_size),
                    "stream_scale_factor": stream_scale_factor,
                    "terminal_sent": False,
                    "prompt_payload": prompt_payload,
                }
            }
            transfer_manager.request_payload[request_id] = request_state

        state = request_state["_cosyvoice3_async_state"]
        if bool(state.get("terminal_sent", False)):
            return None

        with nullcontext():
            output_token_ids = _ensure_list(getattr(request, "output_token_ids", []))
            seen_len = int(state.get("seen_len", 0))
            new_tokens = output_token_ids[seen_len:] if seen_len < len(output_token_ids) else []
            state["seen_len"] = len(output_token_ids)

        if not hasattr(transfer_manager, "code_prompt_token_ids"):
            transfer_manager.code_prompt_token_ids = defaultdict(list)
        token_frames = transfer_manager.code_prompt_token_ids[request_id]
        for tok in new_tokens:
            tok_int = int(tok)
            if 0 <= tok_int < code_vocab_size:
                token_frames.append([tok_int])

        length = len(token_frames)
        if length <= 0:
            if not finished:
                return None
            embed_struct = None
            if not state.get("sent_prompt", False):
                embed_struct = _build_prompt_embed_struct(state.get("prompt_payload", {}))
                state["sent_prompt"] = True
            state["terminal_sent"] = True
            return OmniPayloadStruct(
                codes=CodesStruct(audio=torch.empty(0, dtype=torch.long)),
                meta=MetaStruct(finished=torch.tensor(True, dtype=torch.bool)),
                embed=embed_struct,
            )

        emitted_token_len = int(state.get("emitted_token_len", 0))
        if finished and length <= emitted_token_len:
            embed_struct = None
            if not state.get("sent_prompt", False):
                embed_struct = _build_prompt_embed_struct(state.get("prompt_payload", {}))
                state["sent_prompt"] = True
            state["terminal_sent"] = True
            return OmniPayloadStruct(
                codes=CodesStruct(audio=torch.empty(0, dtype=torch.long)),
                meta=MetaStruct(finished=torch.tensor(True, dtype=torch.bool)),
                embed=embed_struct,
            )

        with nullcontext():
            token_hop_len = max(1, int(state.get("token_hop_len", chunk_size)))
            prompt_token_pad = max(0, int(state.get("prompt_token_pad", 0)))
            pre_lookahead_len = max(0, int(state.get("pre_lookahead_len", pre_lookahead_len)))
            available = max(0, length - emitted_token_len)
            this_token_hop_len = token_hop_len + prompt_token_pad if emitted_token_len == 0 else token_hop_len
            required = this_token_hop_len + pre_lookahead_len

            if not finished:
                if available < required:
                    return None
                prefix_len = emitted_token_len + required
                token_offset = emitted_token_len
            else:
                if available <= 0:
                    return None
                prefix_len = length
                token_offset = emitted_token_len

        with nullcontext():
            code_predictor_codes = [int(frame[0]) for frame in token_frames[:prefix_len]]

        embed_struct = None
        if not state.get("sent_prompt", False):
            embed_struct = _build_prompt_embed_struct(state.get("prompt_payload", {}))
            state["sent_prompt"] = True

        payload = OmniPayloadStruct(
            codes=CodesStruct(audio=torch.tensor(code_predictor_codes, dtype=torch.long)),
            meta=MetaStruct(
                finished=torch.tensor(finished, dtype=torch.bool),
                stream_finished=torch.tensor(finished, dtype=torch.bool),
                req_id=[request_id],
                left_context_size=token_offset,
            ),
            embed=embed_struct,
        )

        if not finished:
            state["emitted_token_len"] = emitted_token_len + this_token_hop_len
            state["token_hop_len"] = min(
                int(state.get("token_max_hop_len", chunk_size)),
                max(chunk_size, token_hop_len * int(state.get("stream_scale_factor", 1))),
            )
        else:
            state["terminal_sent"] = True

        state["emitted_chunks"] = int(state.get("emitted_chunks", 0)) + 1
        return payload


# ============================================================================
# Worker-connector data plane (non-async-chunk path).
# cosyvoice3 talker emits `multimodal_outputs={"embed": {"speech_token": t,
# "speech_feat": t, "embedding": t}}` ONLY at prefill (decode steps emit
# `{}`).  After flatten_payload these become flat top-level keys
# `embed.speech_token` etc., persisted across decode steps by the
# full-payload accumulator (decode doesn't re-emit them).  Shipping via the connector
# keeps the orchestrator off the heavy-tensor path.
# ============================================================================

# All three embed tensors are emitted once at prefill and must REPLACE-not-
# CONCAT across the (already trivial) per-request accumulator history so a
# regression where decode unexpectedly re-emits them does not silently
# duplicate the prefill tensor.  See mixin._FULL_PAYLOAD_REPLACE_KEYS.
_FULL_PAYLOAD_REPLACE_KEYS: frozenset[str] = frozenset({"embed.speech_token", "embed.speech_feat", "embed.embedding"})


def text2flow_token_only(
    source_outputs: list,
    prompt: OmniTokensPrompt | TextPrompt = None,
    _requires_multimodal_data: bool = True,
):
    """Sync-side builder for the non-async-chunk text→flow path.

    CosyVoice3 sync keeps codec ids on the legacy token path.  Some vLLM v1
    histories include the source prompt prefix, so strip it only when it is an
    exact leading match.
    """
    del prompt
    engine_inputs: list[OmniTokensPrompt] = []
    for source_output in source_outputs:
        if not source_output.finished:
            continue
        output = source_output.outputs[0]
        prefix_ids = _ensure_list(source_output.prompt_token_ids)
        raw_output_ids = _ensure_list(output.cumulative_token_ids)
        output_ids = _strip_prompt_prefix(raw_output_ids, prefix_ids)
        multi_modal_data = output.multimodal_output
        if multi_modal_data is None:
            raise RuntimeError(f"Missing multimodal_output for request {source_output.request_id}")
        prompt_speech_ids = _prompt_speech_token_ids(multi_modal_data)
        output_ids = _strip_prompt_prefix(output_ids, prompt_speech_ids)
        additional_info: dict[str, Any] = dict(multi_modal_data)
        additional_info.setdefault("ids", {})["prompt"] = prefix_ids
        engine_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=output_ids,
                additional_information=additional_info,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return engine_inputs


def text2flow_full_payload(
    transfer_manager,
    pooling_output,
    request,
):
    """Producer-side payload builder.

    Reads prefill-emitted `embed.{speech_token, speech_feat, embedding}` from
    the accumulator and ships prompt conditioning as a connector payload.
    The downstream flow stage reads these from `model_intermediate_buffer`
    (see cosyvoice3.py:671 in the code2wav forward — runtime_info pickup).
    """
    del transfer_manager
    rid = getattr(request, "external_req_id", None) or getattr(request, "request_id", "?")
    if not isinstance(pooling_output, dict):
        logger.warning(
            "cosyvoice3.text2flow_full_payload: pooling_output not a dict "
            "(type=%s) for req=%s; consumer wait gate may hang.",
            type(pooling_output).__name__,
            rid,
        )
        return None
    embed_out: dict[str, Any] = {}
    for key in ("speech_token", "speech_feat", "embedding"):
        v = pooling_output.get(f"embed.{key}")
        if v is None:
            nested = pooling_output.get("embed")
            if isinstance(nested, dict):
                v = nested.get(key)
        if isinstance(v, torch.Tensor) and v.numel() > 0:
            embed_out[key] = v
    if not embed_out:
        logger.warning(
            "cosyvoice3.text2flow_full_payload: no embed.{speech_token,speech_feat,embedding} "
            "found in pooling_output (keys=%s) for req=%s; consumer wait gate may hang.",
            list(pooling_output.keys()),
            rid,
        )
        return None
    return {
        "meta": {
            "finished": torch.tensor(True, dtype=torch.bool),
        },
        "embed": embed_out,
    }
