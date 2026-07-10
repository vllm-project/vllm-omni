# SPDX-License-Identifier: Apache-2.0
"""Audex (Nemotron-Labs-Audex-2B) text-to-audio serving adapter.

Caption → general audio through the ``audex_tta`` pipeline.
The adapter builds the TTA prompt; the serving layer completes the request
after the final request id exists: the RVQ phase-mask contract
(``extra_args["tta_rvq"]``) and the CFG pair contract (official default
scale 3.0 — guidance is effectively mandatory for TTA quality).
"""

from typing import TYPE_CHECKING

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest
from vllm_omni.model_executor.models.audex.prompt import build_tta_cond_prompt

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

AUDEX_TTA_DEFAULT_CFG_SCALE = 3.0
AUDEX_TTA_CFG_SCALE_MIN = 1.0
AUDEX_TTA_CFG_SCALE_MAX = 10.0

# Internal plumbing keys; injected by the serving layer, never accepted
# from callers.
_INTERNAL_KEYS = ("cfg_role", "cfg_pair_id", "cfg_null_prompt", "tta_rvq")


@register_tts_adapter
class AudexTTAAdapter(ARTTSAdapter):
    """Caption-conditioned general audio: no voices, no reference audio."""

    stage_keys = frozenset({"audex_tta_thinker"})
    name = "audex_tta"

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        if not request.input or not request.input.strip():
            return "Audex TTA requires a non-empty caption"
        voice = (request.voice or "").strip().lower()
        if voice not in ("", "default"):
            return f"Audex TTA generates general audio and has no voices; got voice={request.voice!r}."
        if request.ref_audio is not None:
            return "Audex TTA does not support reference audio."
        extra = request.extra_params or {}
        for key in _INTERNAL_KEYS:
            if key in extra:
                return f"extra_params.{key} is managed internally by the server; only cfg_scale may be set."
        cfg_scale = extra.get("cfg_scale")
        if cfg_scale is not None:
            try:
                cfg_value = float(cfg_scale)
            except (TypeError, ValueError):
                return (
                    f"extra_params.cfg_scale must be a number in "
                    f"[{AUDEX_TTA_CFG_SCALE_MIN}, {AUDEX_TTA_CFG_SCALE_MAX}]; got {cfg_scale!r}."
                )
            if not (AUDEX_TTA_CFG_SCALE_MIN <= cfg_value <= AUDEX_TTA_CFG_SCALE_MAX):
                return (
                    f"extra_params.cfg_scale must be within "
                    f"[{AUDEX_TTA_CFG_SCALE_MIN}, {AUDEX_TTA_CFG_SCALE_MAX}]; got {cfg_scale!r}. "
                    f"The official TTA setting is {AUDEX_TTA_DEFAULT_CFG_SCALE}; 1.0 disables guidance."
                )
        return None

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        prompt = {"prompt": build_tta_cond_prompt(request.input)}
        return PreparedRequest(prompt=prompt, tts_params={}, model_type="audex_tta")
