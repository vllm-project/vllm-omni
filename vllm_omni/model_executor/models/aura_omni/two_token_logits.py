# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""AURA_v2 first-token silent/respond force, as in the reference streaming inference.

Opt in per request with ``SamplingParams.extra_args={"aura_mode": "streaming"}``.
Register on Stage1 via deploy yaml ``logits_processors``. Prefer
``async_scheduling: false`` on that stage so the live output list is not stale.
"""

from __future__ import annotations

from pathlib import Path

import torch
from vllm.config import VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor
from vllm.v1.sample.logits_processor.builtin import process_dict_updates

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


class AuraTwoTokenLogitsProcessor(LogitsProcessor):
    """Control pair → body: first step only silent/respond; then mask both."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool) -> None:
        del device, is_pin_memory
        tokenizer = getattr(vllm_config.model_config, "tokenizer", None) or getattr(
            vllm_config.model_config, "model", ""
        )
        self.silent, self.respond = _resolve_control_ids(str(tokenizer))
        self.entries: dict[int, list[int]] = {}

    @classmethod
    def validate_params(cls, params: SamplingParams) -> None:
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

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        def add(params: SamplingParams, prompt: list[int] | None, output: list[int]) -> list[int] | None:
            del prompt
            self.validate_params(params)
            if (params.extra_args or {}).get("aura_mode") != "streaming":
                return None
            stop_ids = set(params.stop_token_ids or [])
            if self.silent not in stop_ids or self.respond in stop_ids:
                raise ValueError("Only Silent must be a control stop token (Respond must not stop)")
            # Keep vLLM's live output list; never infer stage from prompt/history.
            return output

        process_dict_updates(self.entries, batch_update, add)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        for index, output in self.entries.items():
            if index >= logits.shape[0]:
                continue
            if not output:
                pair = logits[index, [self.silent, self.respond]].clone()
                if not torch.isfinite(pair).all():
                    raise ValueError("Non-finite control logits")
                logits[index].fill_(float("-inf"))
                logits[index, self.silent] = pair[0]
                logits[index, self.respond] = pair[1]
                # argmax ties choose the lower ID (Silent in checkpoint-1318).
                if self.silent > self.respond and pair[0] == pair[1]:
                    logits[index, self.respond] = float("-inf")
            else:
                if output[0] != self.respond:
                    raise ValueError("Non-Respond request unexpectedly reached body decoding")
                logits[index, [self.silent, self.respond]] = float("-inf")
        return logits


__all__ = ["AuraTwoTokenLogitsProcessor"]
