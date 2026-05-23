# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FunCineForge stage input processors.

Stage 0 (talker) streams codec tokens → Stage 1 (code2wav) converts to mel/audio.
``text2flow`` handles synchronous (full-sequence) handoff.
``talker2code2wav_async_chunk`` handles async streaming handoff.
"""

from collections import defaultdict
from contextlib import nullcontext
from typing import Any

import torch
from vllm.inputs import TextPrompt

from vllm_omni.data_entry_keys import (
    CodesStruct,
    MetaStruct,
    OmniPayloadStruct,
)
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.stage_input_processors.cosyvoice3 import (
    _build_prompt_embed_struct,
    _decode_additional_information,
    _ensure_list,
    _to_cpu_tensor,
)

__all__ = ["text2flow", "talker2code2wav_async_chunk"]

_STATE_KEY = "_funcineforge_async_state"


def text2flow(
    source_outputs: list[Any] | None = None,
    prompt: OmniTokensPrompt | TextPrompt = None,
    _requires_multimodal_data: bool = True,
    streaming_context: Any | None = None,
    **legacy_kwargs: Any,
):
    """Build stage-1 inputs by prefixing stage-0 prompt ids to its outputs."""
    if source_outputs is None:
        stage_list = legacy_kwargs.get("stage_list")
        engine_input_source = legacy_kwargs.get("engine_input_source")
        if stage_list is None or not engine_input_source:
            raise TypeError("text2flow requires source_outputs or legacy stage_list/engine_input_source")
        source_stage = stage_list[int(engine_input_source[0])]
        source_outputs = list(getattr(source_stage, "engine_outputs", []))

    engine_inputs: list[OmniTokensPrompt] = []
    for source_output in source_outputs:
        output = source_output.outputs[0]
        multi_modal_data = output.multimodal_output
        if multi_modal_data is None:
            raise RuntimeError(f"Missing multimodal_output for request {source_output.request_id}")

        output_ids = _ensure_list(getattr(output, "cumulative_token_ids", None))
        if not output_ids:
            output_ids = _ensure_list(getattr(output, "token_ids", None))
        prefix_ids = _ensure_list(source_output.prompt_token_ids)
        additional_info = dict(multi_modal_data)
        additional_info.setdefault("ids", {})["prompt"] = prefix_ids
        engine_inputs.append(OmniTokensPrompt(prompt_token_ids=output_ids, additional_information=additional_info))
    return engine_inputs


def talker2code2wav_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """Async-chunk processor: talker token stream -> code2wav chunks."""
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
    if not isinstance(request_state, dict) or _STATE_KEY not in request_state:
        with nullcontext():
            info = _decode_additional_information(getattr(request, "additional_information", None))
            info_embed = info.get("embed", {}) if isinstance(info, dict) else {}
            prompt_payload = {}
            for key in ("speech_token", "speech_feat", "embedding"):
                value = _to_cpu_tensor(info_embed.get(key))
                if value is not None:
                    prompt_payload[key] = value
            if isinstance(pooling_output, dict):
                po_embed = pooling_output.get("embed", {}) if isinstance(pooling_output.get("embed"), dict) else {}
                for key in ("speech_token", "speech_feat", "embedding"):
                    if key in prompt_payload:
                        continue
                    value = _to_cpu_tensor(po_embed.get(key))
                    if value is not None:
                        prompt_payload[key] = value
            prompt_token = prompt_payload.get("speech_token")
            prompt_token_len = (
                int(prompt_token.shape[1]) if isinstance(prompt_token, torch.Tensor) and prompt_token.ndim >= 2 else 0
            )
            prompt_token_pad = (
                ((prompt_token_len + chunk_size - 1) // chunk_size) * chunk_size - prompt_token_len
                if prompt_token_len > 0
                else 0
            )
        request_state = {
            _STATE_KEY: {
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

    state = request_state[_STATE_KEY]
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
