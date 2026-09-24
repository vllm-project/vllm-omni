# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""AuK Speech API adapter for the current multi-stage serving contract."""

from __future__ import annotations

import copy
import json
import math
from typing import TYPE_CHECKING, Any

import numpy as np
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest
from vllm_omni.model_extras.auk import DEFAULT_SWAY, auk_prompt

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest


@register_tts_adapter
class AuKAdapter(ARTTSAdapter):
    """Translate Speech API fields into the existing AuK encoder/DiT prompt."""

    name = "auk"
    stage_keys = frozenset({"encoder"})
    model_archs = frozenset({"AuKForConditionalGeneration"})

    def normalize(self, request: OpenAICreateSpeechRequest) -> None:
        if request.task_type is None:
            return
        if request.task_type in ("CustomVoice", "Base"):
            logger.warning_once(
                "AuK `task_type` is a compatibility shortcut; prefer a complete `instructions` prompt. "
                'CustomVoice applies `Say the following: "<input>"`; Base applies '
                '`Say the following with the same voice: "<input>"`.'
            )
        if request.instructions and request.instructions.strip():
            logger.warning(
                "AuK request contains both `task_type` and a complete `instructions` prompt; "
                "using `instructions` without applying another task template."
            )
            request.input = ""
            request.task_type = None
            return
        instruction = request.input.strip()
        if instruction.startswith(("Say the following:", "Say the following with the same voice:")):
            logger.warning(
                "AuK request `input` already contains a complete task instruction; "
                "using it without applying another task template."
            )
            request.instructions = instruction
            request.input = ""
            request.task_type = None
            return
        if not instruction:
            return
        quoted_instruction = json.dumps(instruction, ensure_ascii=False)
        if request.task_type == "Base":
            request.instructions = f"Say the following with the same voice: {quoted_instruction}"
        elif request.task_type == "CustomVoice":
            request.instructions = f"Say the following: {quoted_instruction}"
        else:
            return
        request.input = ""
        # task_type is a compatibility shortcut for AuK, not a model-side
        # parameter. Consuming it makes normalization idempotent when the
        # shared batch path validates and later prepares the same request.
        request.task_type = None

    def validate(self, request: OpenAICreateSpeechRequest) -> str | None:
        if not request.input.strip() and not (request.instructions and request.instructions.strip()):
            return "AuK requires input text or a complete instructions prompt"
        if request.duration_seconds is not None:
            if not math.isfinite(request.duration_seconds) or request.duration_seconds <= 0:
                return "AuK requires duration_seconds to be a positive finite value"
        if request.duration_seconds is None and request.ref_audio is None:
            return "AuK requires duration_seconds for requests without ref_audio"
        if request.instructions is not None:
            if len(request.instructions) > self.ctx.server._max_instructions_length:
                return f"instructions exceeds max length {self.ctx.server._max_instructions_length}"
        if request.ref_audio is not None:
            if not isinstance(request.ref_audio, str):
                return "AuK supports one ref_audio string per request"
            error = self.ctx.server._validate_ref_audio_format(request.ref_audio)
            if error:
                return error
        extra = request.extra_params or {}
        for key in extra:
            if key not in {"num_inference_steps", "guidance_scale", "sway", "t_grid", "vae_sample"}:
                return f"AuK does not support extra_params.{key}"
        t_grid = extra.get("t_grid")
        if t_grid is not None:
            t_grid_error = "AuK extra_params.t_grid must contain at least two finite, strictly increasing values"
            if not isinstance(t_grid, (list, tuple)) or len(t_grid) < 2:
                return t_grid_error
            try:
                values = [float(value) for value in t_grid]
            except (TypeError, ValueError):
                return t_grid_error
            if not all(math.isfinite(value) for value in values) or any(
                right <= left for left, right in zip(values, values[1:])
            ):
                return t_grid_error
        return None

    async def build(
        self,
        request: OpenAICreateSpeechRequest,
        sampling_params_list: list,
        has_inline_ref_audio: bool,
    ) -> PreparedRequest:
        del sampling_params_list, has_inline_ref_audio
        reference: tuple[np.ndarray, int] | None = None
        if request.ref_audio is not None:
            samples, sample_rate, _ = await self.ctx.server._resolve_ref_audio(request.ref_audio)
            reference = (np.asarray(samples, dtype=np.float32), int(sample_rate))

        extra = request.extra_params or {}
        # The caller supplies the complete cookbook prompt in instructions;
        # this adapter deliberately does not invent or compose a template.
        if request.instructions and request.instructions.strip():
            if request.input.strip():
                logger.warning(
                    "AuK request contains both non-empty `input` and `instructions`; "
                    "using the complete `instructions` prompt and ignoring `input`."
                )
            instruction = request.instructions.strip()
        else:
            logger.warning(
                "AuK request only provides `input`; treating it as the complete AuK instruction. "
                'Prefer `input=""` with the instruction in `instructions`.'
            )
            instruction = request.input.strip()
        prompt = auk_prompt(
            instruction,
            reference,
            gen_seconds=request.duration_seconds,
            sway=float(extra.get("sway", DEFAULT_SWAY)),
            t_grid=extra.get("t_grid"),
            vae_sample=extra.get("vae_sample", False),
        )
        return PreparedRequest(prompt=prompt, tts_params={}, model_type=self.name)

    def apply_sampling_overrides(
        self,
        sampling_params_list: list,
        request: OpenAICreateSpeechRequest,
        prompt: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> list:
        if len(sampling_params_list) < 2:
            return sampling_params_list
        updated = copy.deepcopy(sampling_params_list)
        stage1 = updated[1]
        extra = request.extra_params or {}
        if "num_inference_steps" in extra:
            stage1.num_inference_steps = int(extra["num_inference_steps"])
        if "guidance_scale" in extra:
            stage1.guidance_scale = float(extra["guidance_scale"])
        if request.seed is not None:
            stage1.seed = int(request.seed)
        return updated


__all__ = ["AuKAdapter"]
