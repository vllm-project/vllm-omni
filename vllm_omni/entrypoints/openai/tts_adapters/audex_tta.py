# SPDX-License-Identifier: Apache-2.0
"""Audex (Nemotron-Labs-Audex-2B) text-to-audio serving adapter.

Caption → general audio through the ``audex_tta`` pipeline.
The adapter builds the TTA prompt and, after receiving the final request id,
completes the sampling parameters with the RVQ phase-mask contract
(``extra_args["tta_rvq"]``) and the CFG pair contract (official default
scale 3.0 — guidance is effectively mandatory for TTA quality).
"""

from typing import TYPE_CHECKING, Any

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest, apply_max_new_tokens
from vllm_omni.model_executor.models.audex.prompt import build_tta_cond_prompt

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

AUDEX_TTA_DEFAULT_CFG_SCALE = 3.0
AUDEX_TTA_CFG_SCALE_MIN = 1.0
AUDEX_TTA_CFG_SCALE_MAX = 10.0

# Internal plumbing keys; injected by this adapter, never accepted from callers.
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

    def apply_sampling_overrides(
        self,
        sampling_params_list: list,
        request: "OpenAICreateSpeechRequest",
        prompt: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> list:
        if request_id is None:
            raise ValueError("Audex TTA requires a request id")
        stage0_params = sampling_params_list[0]
        sampling_params_list[0] = self._inject_audex_tta_args(
            request_id=request_id, prompt=prompt, stage0_params=stage0_params
        )
        return apply_max_new_tokens(sampling_params_list, request)

    def _inject_audex_tta_args(self, request_id: str, prompt: Any, stage0_params: Any) -> Any:
        """Complete the Audex TTA contract on a CLONE of the stage-0 params.

        Always attaches the RVQ phase-mask contract (required for token
        validity) and the CFG pair contract at the official default scale
        3.0 unless the caller passed ``cfg_scale`` (1.0 disables guidance).

        Returns the cloned params; the input is never mutated. On the
        no-extra_params online path the input is an element of the engine's
        SHARED ``default_sampling_params_list`` — writing per-request pair
        state (``cfg_pair_id``/``cfg_null_prompt``) into it would race
        concurrent requests and leak stale CFG metadata into later ones.
        """
        server = self.ctx.server
        import copy

        from vllm_omni.model_executor.models.audex.prompt import build_tta_null_prompt
        from vllm_omni.model_executor.models.audex.tta import build_tta_phase_token_ids

        cond_prompt = prompt.get("prompt") if isinstance(prompt, dict) else prompt
        if not isinstance(cond_prompt, str) or not cond_prompt:
            raise ValueError("Audex TTA requires the adapter-built caption prompt")

        stage0_params = copy.deepcopy(stage0_params)
        tokenizer = server._get_audex_tokenizer()
        if server._audex_tta_rvq is None:
            phase_token_ids, start_tid, end_tid = build_tta_phase_token_ids(tokenizer)
            server._audex_tta_rvq = {
                "phase_token_ids": phase_token_ids,
                "start_tid": start_tid,
                "end_tid": end_tid,
                # Official generation cap (decode truncates at 500 frames).
                "codec_cap": 4000,
                # The TTA prompt ends with <audiogen_start>.
                "start_in_prompt": True,
            }

        extra_args = getattr(stage0_params, "extra_args", None)
        if extra_args is None:
            extra_args = {}
            stage0_params.extra_args = extra_args
        extra_args["tta_rvq"] = server._audex_tta_rvq

        cfg_scale = extra_args.get("cfg_scale")
        cfg_scale = 3.0 if cfg_scale is None else float(cfg_scale)
        if cfg_scale <= 1.0:
            extra_args.pop("cfg_scale", None)
            return stage0_params
        extra_args.update(
            {
                "cfg_scale": cfg_scale,
                "cfg_role": "cond",
                "cfg_pair_id": request_id,
                "cfg_null_prompt": build_tta_null_prompt(cond_prompt, tokenizer),
            }
        )
        return stage0_params
