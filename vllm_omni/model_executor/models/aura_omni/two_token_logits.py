# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""AURA_v2 first-token silent/respond force, as in the reference streaming inference.

Opt in per request with ``SamplingParams.extra_args={"aura_mode": "streaming"}``.
Register on Stage1 via deploy yaml ``logits_processors``. Prefer
``async_scheduling: false`` on that stage so the live output list is not stale.

Model Runner V2 loads this module's ``AuraTwoTokenLogitsProcessor`` (the
``vllm.v1.worker.gpu.sample.logits_processor.LogitsProcessor`` interface).
``AuraTwoTokenLogitsProcessorV1`` keeps the legacy runner API for unit tests
and any V1 deploy that still registers it.
"""

from __future__ import annotations

from pathlib import Path

import torch
from vllm.config import VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import BatchUpdate
from vllm.v1.sample.logits_processor import LogitsProcessor as V1LogitsProcessor
from vllm.v1.sample.logits_processor.builtin import process_dict_updates
from vllm.v1.worker.gpu.sample.logits_processor import (
    LogitsContext,
    LogitsProcessor,
    LogitsProcRequestState,
)

from vllm_omni.model_executor.models.aura_omni.control_tokens import AURA_V2

_SILENT = "<|silent|>"
_RESPOND = "<|respond|>"


def _resolve_control_ids(tokenizer_path: str) -> tuple[int, int]:
    """Prefer the stage tokenizer; fall back to the known v2 checkpoint ids."""
    path = Path(tokenizer_path)
    tok_file = path / "tokenizer.json" if path.is_dir() else path
    if tok_file.is_file():
        try:
            from tokenizers import Tokenizer

            tok = Tokenizer.from_file(str(tok_file))
            silent_ids = tok.encode(_SILENT, add_special_tokens=False).ids
            respond_ids = tok.encode(_RESPOND, add_special_tokens=False).ids
            if len(silent_ids) == 1 and len(respond_ids) == 1 and silent_ids != respond_ids:
                return int(silent_ids[0]), int(respond_ids[0])
        except Exception:
            pass
    if AURA_V2.respond_id is None:
        raise ValueError("AURA_v2 respond id is not configured")
    return AURA_V2.silent_id, AURA_V2.respond_id


def _tokenizer_path(vllm_config: VllmConfig) -> str:
    model_config = vllm_config.model_config
    return str(getattr(model_config, "tokenizer", None) or getattr(model_config, "model", ""))


def _validate_streaming_params(params: SamplingParams) -> None:
    mode = (params.extra_args or {}).get("aura_mode")
    if mode is None or mode == "offline_qa":
        return
    if mode != "streaming":
        raise ValueError("aura_mode must be 'streaming' or 'offline_qa'")
    if params.min_tokens != 0:
        raise ValueError("AURA two-token streaming requires min_tokens=0")
    if (
        params.repetition_penalty != 1
        or params.presence_penalty != 0
        or params.frequency_penalty != 0
        or params.logit_bias
        or params.allowed_token_ids
        or params.bad_words
        or params.stop
        or params.ignore_eos
    ):
        raise ValueError(
            "Penalties, biases, static constraints, string stops and ignore_eos "
            "are incompatible with AURA two-token streaming"
        )


def _validate_control_stops(params: SamplingParams, silent: int, respond: int) -> None:
    stop_ids = set(params.stop_token_ids or [])
    if silent not in stop_ids or respond in stop_ids:
        raise ValueError("Only Silent must be a control stop token (Respond must not stop)")


def _force_control_pair(row: torch.Tensor, silent: int, respond: int) -> None:
    pair = row[[silent, respond]].clone()
    if not torch.isfinite(pair).all():
        raise ValueError("Non-finite control logits")
    row.fill_(float("-inf"))
    row[silent] = pair[0]
    row[respond] = pair[1]
    # argmax ties choose the lower ID (Silent in checkpoint-1318).
    if silent > respond and pair[0] == pair[1]:
        row[respond] = float("-inf")


def _mask_control_pair(row: torch.Tensor, silent: int, respond: int) -> None:
    row[[silent, respond]] = float("-inf")


class AuraTwoTokenLogitsProcessor(LogitsProcessor):
    """MRV2 control pair → body: first step only silent/respond; then mask both."""

    def __init__(self, vllm_config: VllmConfig, req_states: LogitsProcRequestState) -> None:
        self.silent, self.respond = _resolve_control_ids(_tokenizer_path(vllm_config))
        self.req_states = req_states
        self.enabled = [False] * req_states.max_num_reqs

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams) -> None:
        _validate_streaming_params(sampling_params)

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
        self.validate_params(sampling_params)
        if (sampling_params.extra_args or {}).get("aura_mode") != "streaming":
            self.enabled[req_idx] = False
            return False
        _validate_control_stops(sampling_params, self.silent, self.respond)
        self.enabled[req_idx] = True
        return True

    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        prompt_lens = self.req_states.prompt_len.np
        total_lens = self.req_states.total_len.gpu
        token_ids = self.req_states.all_token_ids.gpu
        for row in range(logits.shape[0]):
            req_idx = int(ctx.expanded_idx_mapping[row].item())
            if not self.enabled[req_idx]:
                continue
            prompt_len = int(prompt_lens[req_idx])
            total_len = int(total_lens[req_idx].item())
            if total_len < prompt_len:
                raise ValueError("AURA two-token request has total_len < prompt_len")
            if total_len == prompt_len:
                _force_control_pair(logits[row], self.silent, self.respond)
                continue
            first_out = int(token_ids[req_idx, prompt_len].item())
            if first_out != self.respond:
                raise ValueError("Non-Respond request unexpectedly reached body decoding")
            _mask_control_pair(logits[row], self.silent, self.respond)
        return logits


class AuraTwoTokenLogitsProcessorV1(V1LogitsProcessor):
    """Legacy Model Runner V1 control-pair logits processor."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool) -> None:
        del device, is_pin_memory
        self.silent, self.respond = _resolve_control_ids(_tokenizer_path(vllm_config))
        self.entries: dict[int, list[int]] = {}

    @classmethod
    def validate_params(cls, params: SamplingParams) -> None:
        _validate_streaming_params(params)

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        def add(params: SamplingParams, prompt: list[int] | None, output: list[int]) -> list[int] | None:
            del prompt
            self.validate_params(params)
            if (params.extra_args or {}).get("aura_mode") != "streaming":
                return None
            _validate_control_stops(params, self.silent, self.respond)
            # Keep vLLM's live output list; never infer stage from prompt/history.
            return output

        process_dict_updates(self.entries, batch_update, add)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        for index, output in self.entries.items():
            if index >= logits.shape[0]:
                continue
            if not output:
                _force_control_pair(logits[index], self.silent, self.respond)
            else:
                if output[0] != self.respond:
                    raise ValueError("Non-Respond request unexpectedly reached body decoding")
                _mask_control_pair(logits[index], self.silent, self.respond)
        return logits


__all__ = ["AuraTwoTokenLogitsProcessor", "AuraTwoTokenLogitsProcessorV1"]
