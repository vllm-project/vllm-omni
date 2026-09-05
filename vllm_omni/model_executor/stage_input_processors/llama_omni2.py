# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage handoffs for LLaMA-Omni 2."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from vllm_omni.data_entry_keys import (
    CodesStruct,
    EmbeddingsStruct,
    HiddenStatesStruct,
    IdsStruct,
    MetaStruct,
    OmniPayloadStruct,
)
from vllm_omni.inputs.data import OmniTokensPrompt

_FULL_PAYLOAD_REPLACE_KEYS: frozenset[str] = frozenset(
    {
        "embed.decode",
        "hidden_states.output",
        "codes.audio",
    }
)
TALKER_EOS_TOKEN_ID = 151643
TALKER_SEPARATOR_TOKEN_ID = 151665
TALKER_CODEC_TOKEN_OFFSET = 151666
TALKER_CODEC_VOCAB_SIZE = 6561
_STREAM_STATE_KEY = "_llama_omni2_stream_state"
_PENDING_THINKER_ROWS_KEY = "_llama_omni2_pending_thinker_rows"


def _new_items(name: str, previous: list[int], current: list[int]) -> list[int]:
    if len(current) < len(previous) or current[: len(previous)] != previous:
        raise ValueError(f"{name} must be a cumulative monotonic sequence")
    return current[len(previous) :]


def _decode_codec_token_ids(token_ids: list[int]) -> list[int]:
    codec_end = TALKER_CODEC_TOKEN_OFFSET + TALKER_CODEC_VOCAB_SIZE
    invalid = [token_id for token_id in token_ids if not TALKER_CODEC_TOKEN_OFFSET <= token_id < codec_end]
    if invalid:
        raise ValueError(
            f"LLaMA-Omni 2 codec token IDs must be in [{TALKER_CODEC_TOKEN_OFFSET}, {codec_end}), got {invalid[:8]}"
        )
    return [token_id - TALKER_CODEC_TOKEN_OFFSET for token_id in token_ids]


def _decode_terminal_codec_token_ids(
    token_ids: list[int],
    *,
    finished: bool,
) -> list[int]:
    del finished
    if TALKER_EOS_TOKEN_ID in token_ids:
        eos_index = token_ids.index(TALKER_EOS_TOKEN_ID)
        terminal_tokens = token_ids[eos_index:]
        if any(token_id != TALKER_EOS_TOKEN_ID for token_id in terminal_tokens):
            raise ValueError("LLaMA-Omni 2 codec tokens cannot follow the Talker EOS token")
        token_ids = token_ids[:eos_index]
    return _decode_codec_token_ids(token_ids)


@dataclass
class LlamaOmni2StreamState:
    stream_text_tokens: int = 3
    stream_unit_tokens: int = 10
    separator_token_id: int | None = None
    talker_eos_token_id: int | None = None
    max_drain_units: int = 100
    thinker_tokens: list[int] = field(default_factory=list)
    talker_tokens: list[int] = field(default_factory=list)
    codec_tokens: list[int] = field(default_factory=list)
    pending_thinker_tokens: list[int] = field(default_factory=list)
    separator_scheduled: bool = False
    thinker_finished: bool = False
    talker_finished: bool = False
    codec_finished: bool = False
    codec_chunk_seq: int = 0
    drain_units: int = 0

    def consume_thinker_tokens(
        self,
        token_ids: list[int],
        *,
        finished: bool = False,
    ) -> list[list[int]]:
        current = list(token_ids)
        new_tokens = _new_items("thinker token stream", self.thinker_tokens, current)
        self.thinker_tokens = current
        self.pending_thinker_tokens.extend(new_tokens)

        bursts: list[list[int]] = []
        while len(self.pending_thinker_tokens) >= self.stream_text_tokens:
            bursts.append(self.pending_thinker_tokens[: self.stream_text_tokens])
            del self.pending_thinker_tokens[: self.stream_text_tokens]

        if finished:
            self.thinker_finished = True
            if not self.separator_scheduled:
                terminal_tokens = list(self.pending_thinker_tokens)
                self.pending_thinker_tokens.clear()
                if self.separator_token_id is not None:
                    terminal_tokens.append(self.separator_token_id)
                if terminal_tokens:
                    if bursts and len(terminal_tokens) == 1:
                        bursts[-1].extend(terminal_tokens)
                    else:
                        bursts.append(terminal_tokens)
                self.separator_scheduled = True
        return bursts

    def consume_talker_tokens(self, token_ids: list[int]) -> list[int]:
        current = list(token_ids)
        new_tokens = _new_items("talker token stream", self.talker_tokens, current)
        self.talker_tokens = current

        if self.thinker_finished:
            remaining = max(0, self.max_drain_units - self.drain_units)
            new_tokens = new_tokens[:remaining]
            self.drain_units += len(new_tokens)

        if self.talker_eos_token_id is not None and self.talker_eos_token_id in new_tokens:
            eos_index = new_tokens.index(self.talker_eos_token_id)
            new_tokens = new_tokens[: eos_index + 1]
            self.talker_finished = True
        elif self.thinker_finished and self.drain_units >= self.max_drain_units:
            self.talker_finished = True
        return new_tokens

    def consume_codec_tokens(
        self,
        token_ids: list[int],
        *,
        finished: bool = False,
    ) -> list[int]:
        current = list(token_ids)
        new_tokens = _new_items("codec token stream", self.codec_tokens, current)
        self.codec_tokens = current
        if finished:
            self.codec_finished = True
        return new_tokens

    @property
    def should_continue_drain(self) -> bool:
        return self.thinker_finished and not self.talker_finished


class LlamaOmni2StreamStateStore:
    def __init__(
        self,
        request_payload: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._request_payload = request_payload if request_payload is not None else {}

    def get(self, request_id: str) -> LlamaOmni2StreamState:
        container = self._request_payload.setdefault(request_id, {})
        state = container.get(_STREAM_STATE_KEY)
        if not isinstance(state, LlamaOmni2StreamState):
            state = LlamaOmni2StreamState()
            container[_STREAM_STATE_KEY] = state
        return state

    def cancel(self, request_id: str) -> None:
        container = self._request_payload.get(request_id)
        if not isinstance(container, dict):
            return
        container.pop(_STREAM_STATE_KEY, None)
        container.pop(_PENDING_THINKER_ROWS_KEY, None)
        if not container:
            self._request_payload.pop(request_id, None)

    def __contains__(self, request_id: object) -> bool:
        container = self._request_payload.get(request_id)
        return isinstance(container, dict) and _STREAM_STATE_KEY in container


def _request_id(request: Any, fallback: str | None = None) -> str:
    request_id = getattr(request, "external_req_id", None)
    if request_id is None:
        request_id = getattr(request, "request_id", None)
    if request_id is None:
        request_id = fallback
    if request_id is None:
        raise ValueError("LLaMA-Omni 2 stage handoff requires a request ID")
    return str(request_id)


def _state_store(transfer_manager: Any) -> LlamaOmni2StreamStateStore:
    request_payload = getattr(transfer_manager, "request_payload", None)
    if not isinstance(request_payload, dict):
        request_payload = {}
        transfer_manager.request_payload = request_payload
    return LlamaOmni2StreamStateStore(request_payload)


def _request_state_container(
    transfer_manager: Any,
    request_id: str,
) -> dict[str, Any]:
    request_payload = getattr(transfer_manager, "request_payload", None)
    if not isinstance(request_payload, dict):
        request_payload = {}
        transfer_manager.request_payload = request_payload
    container = request_payload.get(request_id)
    if not isinstance(container, dict):
        container = {}
        request_payload[request_id] = container
    return container


def _tensor_from_payload(
    payload: Mapping[str, Any],
    category: str,
    key: str,
) -> torch.Tensor | None:
    if not category:
        value = payload.get(key)
        if value is None:
            return None
        return value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    nested = payload.get(category)
    value = nested.get(key) if isinstance(nested, Mapping) else None
    if value is None:
        value = payload.get(f"{category}.{key}")
    if value is None:
        return None
    return value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)


def _aligned_new_rows(
    tensor: torch.Tensor | None,
    *,
    previous_count: int,
    current_count: int,
) -> torch.Tensor | None:
    new_count = current_count - previous_count
    if tensor is None or new_count <= 0:
        return None
    if tensor.ndim == 0:
        raise ValueError("LLaMA-Omni 2 handoff tensors must have a row dimension")
    if tensor.shape[0] == current_count:
        return tensor[previous_count:]
    if tensor.shape[0] == new_count:
        return tensor
    raise ValueError(
        "LLaMA-Omni 2 handoff tensor rows must match either the cumulative "
        f"token count ({current_count}) or new token count ({new_count}), got {tensor.shape[0]}"
    )


def _runner_hidden_new_rows(
    tensor: torch.Tensor | None,
    *,
    previous_count: int,
    current_count: int,
) -> torch.Tensor | None:
    new_count = current_count - previous_count
    if tensor is None or new_count <= 0:
        return None
    if tensor.ndim == 0:
        raise ValueError("LLaMA-Omni 2 handoff tensors must have a row dimension")
    if tensor.shape[0] < new_count:
        raise ValueError(
            "LLaMA-Omni 2 runner hidden rows must include every new token row, "
            f"expected at least {new_count}, got {tensor.shape[0]}"
        )
    return tensor[-new_count:]


def _pending_tensor_rows(transfer_manager: Any, request_id: str) -> dict[str, list[torch.Tensor]]:
    container = _request_state_container(transfer_manager, request_id)
    pending = container.get(_PENDING_THINKER_ROWS_KEY)
    if not isinstance(pending, dict):
        pending = {"embed": [], "hidden": []}
        container[_PENDING_THINKER_ROWS_KEY] = pending
    return pending


def _pop_rows(
    rows: dict[str, list[torch.Tensor]],
    count: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if count <= 0:
        return None, None
    if len(rows["hidden"]) < count:
        raise ValueError("LLaMA-Omni 2 Thinker payload is missing hidden-state rows")
    if rows["embed"] and len(rows["embed"]) < count:
        raise ValueError("LLaMA-Omni 2 Thinker payload has partially aligned decode embedding rows")
    embed = torch.cat(rows["embed"][:count], dim=0) if rows["embed"] else None
    hidden = torch.cat(rows["hidden"][:count], dim=0)
    if rows["embed"]:
        del rows["embed"][:count]
    del rows["hidden"][:count]
    return embed, hidden


def thinker2talker_async_chunk(
    transfer_manager: Any,
    multimodal_output: Mapping[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    request_id = _request_id(request)
    state = _state_store(transfer_manager).get(request_id)
    state.separator_token_id = TALKER_SEPARATOR_TOKEN_ID
    current_tokens = list(getattr(request, "output_token_ids", None) or [])
    previous_count = len(state.thinker_tokens)

    if isinstance(multimodal_output, Mapping):
        embed = _aligned_new_rows(
            _tensor_from_payload(multimodal_output, "embed", "decode"),
            previous_count=previous_count,
            current_count=len(current_tokens),
        )
        hidden = _aligned_new_rows(
            _tensor_from_payload(multimodal_output, "hidden_states", "output"),
            previous_count=previous_count,
            current_count=len(current_tokens),
        )
        if hidden is None:
            hidden = _runner_hidden_new_rows(
                _tensor_from_payload(multimodal_output, "", "hidden"),
                previous_count=previous_count,
                current_count=len(current_tokens),
            )
        if embed is not None or hidden is not None:
            if hidden is None or (embed is not None and embed.shape[0] != hidden.shape[0]):
                raise ValueError(
                    "LLaMA-Omni 2 Thinker handoff requires hidden-state rows "
                    "and any decode embedding rows must be aligned"
                )
            pending_rows = _pending_tensor_rows(transfer_manager, request_id)
            if embed is not None:
                pending_rows["embed"].extend(embed[i : i + 1] for i in range(embed.shape[0]))
            pending_rows["hidden"].extend(hidden[i : i + 1] for i in range(hidden.shape[0]))

    bursts = state.consume_thinker_tokens(current_tokens, finished=is_finished)
    if not bursts:
        return None
    if len(bursts) != 1:
        raise ValueError("LLaMA-Omni 2 async Thinker handoff received multiple scheduling bursts in one callback")

    burst = bursts[0]
    has_separator = burst[-1:] == [TALKER_SEPARATOR_TOKEN_ID]
    thinker_row_count = len(burst) - int(has_separator)
    rows = _pending_tensor_rows(transfer_manager, request_id)
    embed, hidden = _pop_rows(rows, thinker_row_count)
    return OmniPayloadStruct(
        ids=IdsStruct(output=burst),
        embed=EmbeddingsStruct(decode=embed),
        hidden_states=HiddenStatesStruct(output=hidden),
        meta=MetaStruct(
            finished=torch.tensor(is_finished, dtype=torch.bool),
            next_stage_prompt_len=len(burst),
            replace_streaming_prompt=True,
        ),
    )


def thinker2talker_full_payload(
    transfer_manager: Any,
    pooling_output: Mapping[str, Any],
    request: Any,
) -> dict[str, Any] | None:
    del transfer_manager
    if not isinstance(pooling_output, Mapping):
        return None
    payload_token_ids = _tensor_from_payload(pooling_output, "ids", "output")
    output_token_ids = (
        [int(token_id) for token_id in payload_token_ids.reshape(-1).tolist()]
        if payload_token_ids is not None
        else list(getattr(request, "output_token_ids", None) or [])
    )
    embed = _tensor_from_payload(pooling_output, "embed", "decode")
    hidden = _tensor_from_payload(pooling_output, "hidden_states", "output")
    if not output_token_ids or embed is None or hidden is None:
        return None
    if embed.shape[0] != len(output_token_ids) or hidden.shape[0] != len(output_token_ids):
        raise ValueError(
            "LLaMA-Omni 2 full Thinker payload rows must match output token IDs: "
            f"token_ids={len(output_token_ids)}, "
            f"embed={embed.shape[0]}, hidden={hidden.shape[0]}"
        )
    return {
        "ids": {"output": output_token_ids + [TALKER_SEPARATOR_TOKEN_ID]},
        "embed": {"decode": embed},
        "hidden_states": {"output": hidden},
        "meta": {
            "finished": torch.tensor(True, dtype=torch.bool),
            "next_stage_prompt_len": len(output_token_ids) + 1,
        },
    }


def thinker2talker_token_only(
    source_outputs: list[Any],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any = None,
) -> list[OmniTokensPrompt]:
    del prompt, requires_multimodal_data, streaming_context
    talker_inputs: list[OmniTokensPrompt] = []
    for source_output in source_outputs:
        cumulative = list(source_output.outputs[0].cumulative_token_ids)
        finished = bool(getattr(source_output, "finished", False))
        prompt_len = len(cumulative) + int(finished)
        if prompt_len == 0:
            continue
        talker_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0] * prompt_len,
                additional_information=None,
                multi_modal_data=None,
                mm_processor_kwargs=None,
                sampling_params_override={"max_tokens": 100},
            )
        )
    return talker_inputs


def talker2code2wav_async_chunk(
    transfer_manager: Any,
    multimodal_output: Mapping[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    if not isinstance(multimodal_output, Mapping):
        return None
    codes = _tensor_from_payload(multimodal_output, "codes", "audio")
    if codes is None:
        return None
    codes = codes.to(torch.long).reshape(-1)

    state = _state_store(transfer_manager).get(_request_id(request))
    was_finished = state.codec_finished
    current = codes.tolist()
    delta = _new_items("codec token stream", state.codec_tokens, current)
    decoded_delta = _decode_terminal_codec_token_ids(
        delta,
        finished=is_finished,
    )
    state.codec_tokens = current
    if is_finished:
        state.codec_finished = True
    if not decoded_delta and not (is_finished and not was_finished):
        return None
    chunk_seq = state.codec_chunk_seq
    state.codec_chunk_seq += 1
    return OmniPayloadStruct(
        codes=CodesStruct(
            audio=torch.tensor(
                decoded_delta,
                dtype=torch.long,
            )
        ),
        meta=MetaStruct(
            finished=torch.tensor(is_finished, dtype=torch.bool),
            request_id=_request_id(request),
            chunk_seq=chunk_seq,
        ),
    )


talker2code2wav_async_chunk.manages_output_dedup = True


def talker2code2wav_full_payload(
    transfer_manager: Any,
    pooling_output: Mapping[str, Any],
    request: Any,
    request_id: str | None = None,
) -> dict[str, Any] | None:
    del transfer_manager
    if not isinstance(pooling_output, Mapping):
        return None
    codes = _tensor_from_payload(pooling_output, "codes", "audio")
    if codes is None or codes.numel() == 0:
        return None
    decoded_codes = _decode_terminal_codec_token_ids(
        codes.to(torch.long).reshape(-1).tolist(),
        finished=True,
    )
    return {
        "codes": {"audio": torch.tensor(decoded_codes, dtype=torch.long)},
        "meta": {
            "finished": torch.tensor(True, dtype=torch.bool),
            "request_id": _request_id(request, request_id),
            "chunk_seq": 0,
        },
    }
