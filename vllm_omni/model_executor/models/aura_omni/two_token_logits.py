# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""AURA_v2 first-token silent/respond force, as in the reference streaming inference.

Opt in per request with ``SamplingParams.extra_args={"aura_mode": "streaming"}``.
Register on Stage1 via deploy yaml ``logits_processors``.

Model Runner V2 loads this module's ``AuraTwoTokenLogitsProcessor`` (the
``vllm.v1.worker.gpu.sample.logits_processor.LogitsProcessor`` interface). It
decides the first step from ``total_len == prompt_len`` on device, so it does
not read vLLM's output list. ``AuraTwoTokenLogitsProcessorV1`` keeps the legacy
runner API: it reads the live output list, which is why its deploy needs
``async_scheduling: false``. The V2 deploys keep that setting; whether V2
still needs it is not measured.
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
    """MRV2 control pair → body: first step only silent/respond; then mask both.

    ``apply()`` runs as whole-batch tensor ops with no per-row host reads. The
    first step is ``total_len == prompt_len``, read from the device mirrors the
    runner refreshes each step. Per-request enablement is a host list flushed
    to a device bool table only when a request is admitted.
    """

    def __init__(self, vllm_config: VllmConfig, req_states: LogitsProcRequestState) -> None:
        self.silent, self.respond = _resolve_control_ids(_tokenizer_path(vllm_config))
        self.req_states = req_states
        self._enabled_host = [False] * req_states.max_num_reqs
        self._enabled_gpu = torch.zeros(req_states.max_num_reqs, dtype=torch.bool, device=req_states.device)
        self._enabled_dirty = False

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams) -> None:
        _validate_streaming_params(sampling_params)

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
        self.validate_params(sampling_params)
        enabled = (sampling_params.extra_args or {}).get("aura_mode") == "streaming"
        if enabled:
            _validate_control_stops(sampling_params, self.silent, self.respond)
        self._enabled_host[req_idx] = enabled
        self._enabled_dirty = True
        return enabled

    def apply_staged_writes(self) -> None:
        self._flush_enabled()

    def _flush_enabled(self) -> None:
        if not self._enabled_dirty:
            return
        self._enabled_gpu.copy_(torch.tensor(self._enabled_host, dtype=torch.bool))
        self._enabled_dirty = False

    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        self._flush_enabled()
        slot = ctx.expanded_idx_mapping.long()
        enabled = self._enabled_gpu[slot]
        first = self.req_states.total_len.gpu[slot] == self.req_states.prompt_len.gpu[slot]
        force_rows = (enabled & first).unsqueeze(1)
        body_rows = (enabled & ~first).unsqueeze(1)

        silent, respond = self.silent, self.respond
        neg_inf = float("-inf")
        # First step: keep only the control pair, as _force_control_pair does.
        silent_logit = logits[:, silent].clone()
        respond_logit = logits[:, respond].clone()
        logits.masked_fill_(force_rows, neg_inf)
        force = force_rows[:, 0]
        logits[:, silent] = torch.where(force, silent_logit, logits[:, silent])
        logits[:, respond] = torch.where(force, respond_logit, logits[:, respond])
        if silent > respond:
            # Argmax ties choose the lower id; Silent should win ties.
            tie = force & (silent_logit == respond_logit)
            logits[:, respond] = logits[:, respond].masked_fill(tie, neg_inf)
        # Body: mask both control ids, as _mask_control_pair does.
        body = body_rows[:, 0]
        logits[:, silent] = logits[:, silent].masked_fill(body, neg_inf)
        logits[:, respond] = logits[:, respond].masked_fill(body, neg_inf)
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
