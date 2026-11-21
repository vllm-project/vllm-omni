# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Sesame CSM serving adapter: one speaker turn of text to speech."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch
from vllm.inputs import tokens_input
from vllm.utils.async_utils import make_async

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest, apply_max_new_tokens

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

# ``speaker_id`` labels the turn; the sampling values apply to the first codebook only.
_EXTRA_PARAMS = frozenset({"speaker_id", "temperature", "top_k", "top_p"})
# Conditioning this adapter does not consume; a request that sets one is rejected rather than silently ignored.
_UNSUPPORTED_FIELDS = (
    "instructions",
    "language",
    "ref_audio",
    "ref_text",
    "ref_audio_2",
    "speaker_embedding",
    "x_vector_only_mode",
)


def _has_error_flag(value: Any) -> bool:
    """Whether any accumulated ``meta.error`` value marks a failed chunk."""
    if isinstance(value, torch.Tensor):
        return bool(value.any())
    if isinstance(value, (list, tuple)):
        return any(_has_error_flag(item) for item in value)
    return value in (True, 1, "1")


@register_tts_adapter
class CsmAdapter(ARTTSAdapter):
    name = "csm"
    stage_keys = frozenset({"csm_talker", "csm_code2wav"})
    # The talker produces one frame per token, so a length finish means the speech was cut off.
    reports_length_truncation = True
    max_new_tokens_max = 2048

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._tokenizer: Any = None
        self._added_token_ids: frozenset[int] = frozenset()
        self._build_prompt_async = make_async(self._build_prompt, executor=getattr(ctx.server, "_tts_executor", None))

    def _model_config(self) -> Any:
        engine_client = self.ctx.engine_client
        if engine_client is None:
            raise RuntimeError("CSM speech serving requires an engine client")
        return engine_client.model_config

    def _load_codec_frame_rate(self) -> float | None:
        codec_config = getattr(self._model_config().hf_config, "codec_config", None)
        rate = getattr(codec_config, "frame_rate", None)
        return float(rate) if rate else None

    def _get_tokenizer(self) -> Any:
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self._model_config().model)
            self._added_token_ids = frozenset(tokenizer.get_added_vocab().values())
            self._tokenizer = tokenizer
        return self._tokenizer

    def _build_prompt(self, text: str, speaker_id: int) -> dict[str, Any]:
        """Tokenize one turn the way the checkpoint's chat template renders it: ``[speaker]text`` in BOS and EOS."""
        tokenizer = self._get_tokenizer()
        rendered = f"{tokenizer.bos_token}[{speaker_id}]{text}{tokenizer.eos_token}"
        prompt_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        # Control tokens written into the text would be read as prompt structure.
        if any(token_id in self._added_token_ids for token_id in prompt_ids[1:-1]):
            raise ValueError("Input text must not contain the model's special tokens")
        return tokens_input(prompt_token_ids=prompt_ids)

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"
        for name in _UNSUPPORTED_FIELDS:
            if getattr(request, name, None) is not None:
                return f"CSM does not support '{name}'"
        extra = request.extra_params or {}
        if unknown := set(extra) - _EXTRA_PARAMS:
            return f"Unsupported CSM parameters: {sorted(unknown)}; supported: {sorted(_EXTRA_PARAMS)}"
        speaker_id = extra.get("speaker_id", 0)
        if isinstance(speaker_id, bool) or not isinstance(speaker_id, int) or speaker_id < 0:
            return "speaker_id must be a non-negative integer"
        return self._validate_max_new_tokens(request)

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        speaker_id = (request.extra_params or {}).get("speaker_id", 0)
        prompt = await self._build_prompt_async(request.input, speaker_id)
        return PreparedRequest(prompt=prompt, tts_params={"speaker": str(speaker_id)}, model_type=self.name)

    def apply_sampling_overrides(
        self,
        sampling_params_list: list,
        request: "OpenAICreateSpeechRequest",
        prompt: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> list:
        return apply_max_new_tokens(sampling_params_list, request)

    def collect_response_metadata(self, audio_output: Mapping[str, Any], collect: dict) -> None:
        """Reject the response when the codec stage marked the request as failed."""
        error = audio_output.get("meta.error")
        if error is None:
            meta = audio_output.get("meta")
            error = meta.get("error") if isinstance(meta, Mapping) else None
        if _has_error_flag(error):
            collect["audio_synthesis_error"] = True
