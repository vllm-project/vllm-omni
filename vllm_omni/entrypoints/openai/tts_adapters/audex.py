# SPDX-License-Identifier: Apache-2.0
"""Audex (Nemotron-Labs-Audex-2B) TTS serving adapter."""

from typing import TYPE_CHECKING, Any

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest, apply_max_new_tokens
from vllm_omni.model_executor.models.audex.prompt import build_cond_prompt

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest


# Classifier-free guidance strength accepted on /v1/audio/speech via
# extra_params. 1.0 disables guidance (identical to omitting it); the
# official TTS quality setting is 1.5.
AUDEX_CFG_SCALE_MIN = 1.0
AUDEX_CFG_SCALE_MAX = 10.0

# Internal CFG pair-plumbing keys; injected by this adapter once it receives
# the final request id, never accepted from callers.
_AUDEX_INTERNAL_CFG_KEYS = ("cfg_role", "cfg_pair_id", "cfg_null_prompt")


@register_tts_adapter
class AudexAdapter(ARTTSAdapter):
    """Plain English TTS: single built-in voice, no reference audio, optional CFG."""

    stage_keys = frozenset({"audex_thinker", "audex_omni"})
    name = "audex"

    @classmethod
    def stage_serves_speech(cls, model_stage: str | None, all_stage_keys: frozenset[str]) -> bool:
        """``audex_omni`` covers the audex_s2s deployment, whose TTS pass uses
        the same ``/v1/audio/speech`` surface. Without the speech decoder the
        same thinker is text-final and must not accept speech requests."""
        if model_stage == "audex_omni":
            return "audex_code2wav" in all_stage_keys
        return True

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        if not request.input or not request.input.strip():
            return "Audex TTS requires non-empty input text"
        voice = (request.voice or "").strip().lower()
        if voice not in ("", "default"):
            return (
                f"Audex has a single built-in voice and no voice cloning; got voice={request.voice!r}. "
                "Omit 'voice' or pass 'default'."
            )
        if request.ref_audio is not None:
            return "Audex does not support reference audio (no voice cloning)."
        extra = request.extra_params or {}
        for key in _AUDEX_INTERNAL_CFG_KEYS:
            if key in extra:
                return f"extra_params.{key} is managed internally by the server; only cfg_scale may be set."
        cfg_scale = extra.get("cfg_scale")
        if cfg_scale is not None:
            try:
                cfg_value = float(cfg_scale)
            except (TypeError, ValueError):
                return (
                    f"extra_params.cfg_scale must be a number in "
                    f"[{AUDEX_CFG_SCALE_MIN}, {AUDEX_CFG_SCALE_MAX}]; got {cfg_scale!r}."
                )
            if not (AUDEX_CFG_SCALE_MIN <= cfg_value <= AUDEX_CFG_SCALE_MAX):
                return (
                    f"extra_params.cfg_scale must be within [{AUDEX_CFG_SCALE_MIN}, {AUDEX_CFG_SCALE_MAX}]; "
                    f"got {cfg_scale!r}. 1.0 disables guidance; 1.5 is the recommended quality setting."
                )
        return None

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        prompt = {"prompt": build_cond_prompt(request.input)}
        return PreparedRequest(prompt=prompt, tts_params={}, model_type="audex")

    def apply_sampling_overrides(
        self,
        sampling_params_list: list,
        request: "OpenAICreateSpeechRequest",
        prompt: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> list:
        if request_id is None:
            raise ValueError("Audex CFG requires a request id")
        stage0_params = sampling_params_list[0]
        sampling_params_list[0] = self._inject_audex_cfg_pair_args(
            request_id=request_id, prompt=prompt, stage0_params=stage0_params
        )
        return apply_max_new_tokens(sampling_params_list, request)

    def _inject_audex_cfg_pair_args(self, request_id: str, prompt: Any, stage0_params: Any) -> Any:
        """Complete the Audex CFG contract on a CLONE of the stage-0 params.

        The adapter validated ``cfg_scale``; here (where the final request id
        exists) the cond role, pair id, and the length-matched null prompt are
        attached. ``cfg_scale`` absent/1.0 returns a clone value-identical to
        the non-CFG flow (the key is dropped rather than forwarded).

        Returns the cloned params; the input is never mutated (it may be an
        element of the engine's shared ``default_sampling_params_list`` —
        see ``_inject_audex_tta_args``).
        """
        server = self.ctx.server
        import copy

        stage0_params = copy.deepcopy(stage0_params)
        extra_args = getattr(stage0_params, "extra_args", None)
        if not extra_args or extra_args.get("cfg_scale") is None:
            if extra_args:
                extra_args.pop("cfg_scale", None)
            return stage0_params
        cfg_scale = float(extra_args["cfg_scale"])
        if cfg_scale <= 1.0:
            extra_args.pop("cfg_scale", None)
            return stage0_params

        from vllm_omni.model_executor.models.audex.prompt import build_null_prompt

        cond_prompt = prompt.get("prompt") if isinstance(prompt, dict) else prompt
        if not isinstance(cond_prompt, str) or not cond_prompt:
            raise ValueError("Audex CFG requires the adapter-built text prompt")
        null_prompt = build_null_prompt(cond_prompt, server._get_audex_tokenizer())
        extra_args.update(
            {
                "cfg_scale": cfg_scale,
                "cfg_role": "cond",
                "cfg_pair_id": request_id,
                "cfg_null_prompt": null_prompt,
            }
        )
        # Guided decoding sharpens the distribution; the unguided default
        # temperature (0.1) adds excess sampling noise under CFG. Measured
        # on the en-24 gate corpus at cfg 1.5: temp 0.1 -> CER 7.31%,
        # temp 0.05 -> CER 6.87% (unguided baseline 7.24%).
        stage0_params.temperature = 0.05
        return stage0_params
