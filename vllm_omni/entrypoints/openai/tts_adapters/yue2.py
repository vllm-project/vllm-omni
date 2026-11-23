# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""YuE2-3B serving adapter for ``/v1/audio/speech``.

This is a text-to-music model on a speech endpoint, so most of the speech
contract does not apply. ``input`` carries the lyrics and ``instructions`` the
style caption (genre, instrumentation, tempo, mood). There is no speaker to
select, no reference audio, and no sampling knobs: the request-local preset is
pinned by the checkpoint's reference config. Model-specific controls travel in
``extra_params``:

- ``cot``: ``"off"`` (default) | ``"melody"`` | ``"full"`` — how the ABC
  score span in the prompt is handled.
- ``abc``: optional external score for ``cot=melody|full``; omitted scores
  are generated within the same request.
- ``abc_max_tokens``: automatic ABC budget, from 1 to 4096.
- ``cfg_scale``: paired semantic guidance, from 0 to 20; default 1 disables CFG.

Unsupported parameters are rejected rather than silently ignored, following
the MiniMax Music 3 precedent.
"""

import asyncio
import math
import secrets
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import (
    ARTTSAdapter,
    PreparedRequest,
)
from vllm_omni.model_executor.models.yue2.yue2 import (
    FRAMES_PER_SECOND,
    KEY_MAX_AUDIO_FRAMES,
    KEY_PHASE,
    SEMANTIC_SAMPLING,
    STOP_TOKEN_IDS,
)

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

_COT_MODES = ("off", "melody", "full")

# Without max_new_tokens the budget is the preset ceiling, so the model's
# own end token sets the song length (upstream reference behavior).
_DEFAULT_MAX_FRAMES = int(SEMANTIC_SAMPLING["max_tokens"])

# Sampling is fixed by the checkpoint's reference preset. Accepting these
# would imply a control the model does not honor.
_UNSUPPORTED_SAMPLING_PARAMS = ("temperature", "top_p", "top_k", "repetition_penalty")


@register_tts_adapter
class Yue2Adapter(ARTTSAdapter):
    """Lyrics plus a style caption in, one 48 kHz stereo song out."""

    name = "yue2"
    stage_keys = frozenset({"yue2"})
    model_archs = frozenset({"Yue2ForCausalLM"})
    max_new_tokens_min = int(SEMANTIC_SAMPLING["min_tokens"])
    max_new_tokens_max = int(SEMANTIC_SAMPLING["max_tokens"])

    def __init__(self, ctx: Any) -> None:
        super().__init__(ctx)
        self._cached_tokenizer: Any = None

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        if not request.input or not request.input.strip():
            return "YuE2 requires non-empty 'input' carrying the lyrics"

        instructions = request.instructions
        if not isinstance(instructions, str) or not instructions.strip():
            return (
                "YuE2 requires 'instructions' describing the music "
                "(genre, instrumentation, tempo, mood); it is what decides the arrangement"
            )

        extra: dict[str, Any] = request.extra_params or {}
        cot = extra.get("cot", "off")
        if cot not in _COT_MODES:
            return f"extra_params.cot must be one of {_COT_MODES}; got {cot!r}"
        abc = extra.get("abc")
        if cot == "off":
            if abc is not None:
                return "extra_params.abc requires cot=melody|full"
        elif abc is not None and (not isinstance(abc, str) or not abc.strip()):
            return "extra_params.abc must be a nonempty ABC score"
        budget = extra.get("abc_max_tokens", 4096)
        if type(budget) is not int or not 1 <= budget <= 4096:
            return "extra_params.abc_max_tokens must be an integer in [1,4096]"
        scale = extra.get("cfg_scale", 1.0)
        if (
            isinstance(scale, bool)
            or not isinstance(scale, (int, float))
            or not math.isfinite(scale)
            or not 0 <= scale <= 20
        ):
            return "extra_params.cfg_scale must be finite and in [0,20]"

        if request.max_new_tokens is not None:
            if request.max_new_tokens < self.max_new_tokens_min:
                return f"'max_new_tokens' must be at least {self.max_new_tokens_min}"
            if request.max_new_tokens > self.max_new_tokens_max:
                return (
                    f"'max_new_tokens' counts audio frames at {FRAMES_PER_SECOND} per second "
                    f"and cannot exceed {self.max_new_tokens_max}"
                )

        voice = (request.voice or "").strip().lower()
        if voice not in ("", "default"):
            return (
                "YuE2 has no speaker to select; the vocal comes from "
                f"'instructions'. Omit 'voice' or pass 'default'; got {request.voice!r}"
            )
        if request.ref_audio is not None or request.ref_text is not None:
            return "YuE2 does not support reference-audio conditioning"
        if request.language is not None:
            return "YuE2 has no language tag; the language follows the lyrics"
        if request.task_type is not None:
            return "YuE2 does not support 'task_type'"
        if request.speed is not None and float(request.speed) != 1.0:
            return "YuE2 only supports speed=1.0; put the tempo in 'instructions', e.g. 'at 120 BPM'"
        if request.is_streaming():
            return (
                "YuE2 does not support streaming: the song is decoded in one "
                "terminal NAR/VAE pass after AR decoding finishes. Use stream=false."
            )

        rejected = sorted(key for key in _UNSUPPORTED_SAMPLING_PARAMS if extra.get(key) is not None)
        if rejected:
            return "YuE2 has fixed sampling and does not accept: " + ", ".join(rejected)
        return None

    async def warmup(self) -> None:
        # Resolve (and possibly download) qwen.tiktoken at startup: build()
        # runs on the request path and must not block on a hub download, and a
        # bad path should surface before the first request.
        await asyncio.to_thread(self._tokenizer)

    async def build(
        self,
        request: "OpenAICreateSpeechRequest",
        sampling_params_list: list,
        has_inline_ref_audio: bool,
    ) -> PreparedRequest:
        from vllm_omni.model_executor.models.yue2.request import prepare_song

        extra = request.extra_params or {}
        seed = request.seed if request.seed is not None else secrets.randbelow(2**31)
        engine = self.ctx.engine_client
        assert engine is not None
        prompt, params = prepare_song(
            self._tokenizer().encode,
            request.instructions,
            request.input,
            cot=extra.get("cot", "off"),
            abc=extra.get("abc"),
            seed=seed,
            max_frames=int(request.max_new_tokens or _DEFAULT_MAX_FRAMES),
            max_abc_tokens=extra.get("abc_max_tokens", 4096),
            max_model_len=engine.vllm_config.model_config.max_model_len,
            cfg_scale=extra.get("cfg_scale", 1.0),
        )
        prompt["additional_information"] = {"yue2_song_sampling": params.extra_args}
        return PreparedRequest(prompt=prompt, tts_params={}, model_type=self.name)

    def apply_sampling_overrides(
        self,
        sampling_params_list: list,
        request: "OpenAICreateSpeechRequest",
        prompt: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> list:
        import copy

        assert prompt is not None
        params = copy.deepcopy(sampling_params_list)
        # Phase transitions mutate nested args; keep the prepared prompt reusable.
        args = copy.deepcopy(prompt["additional_information"]["yue2_song_sampling"])
        params[0].extra_args = args
        params[0].max_tokens = args[KEY_MAX_AUDIO_FRAMES] + (2 if args[KEY_PHASE] == "semantic" else 0)
        params[0].stop_token_ids = list(STOP_TOKEN_IDS)
        params[0].detokenize = False
        return params

    @staticmethod
    def _last_meta_flag(audio_output: Mapping[str, Any], name: str) -> Any:
        """Read a per-request 0/1 meta flag in all three wire shapes.

        The runner flattens nested meta.* to dotted keys (flatten_payload)
        and the wire payload is tensor-only (_ensure_tensor_values), so by
        serving time the flag is usually a 0-d "0"/"1" tensor under a dotted
        key; accept the nested/offline shapes too.
        """
        flags = audio_output.get(f"meta.{name}")
        if flags is None:
            meta = audio_output.get("meta")
            if isinstance(meta, dict):
                flags = meta.get(name)
        if isinstance(flags, torch.Tensor):
            flags = flags.reshape(-1)[-1].item() if flags.numel() else None
        elif isinstance(flags, (list, tuple)):
            flags = flags[-1] if flags else None
        return flags

    def collect_response_metadata(self, audio_output: Mapping[str, Any], collect: dict) -> None:
        """Surface meta.truncated / meta.error as response headers."""
        if self._last_meta_flag(audio_output, "abc_truncated") in (True, 1, "1"):
            collect["audio_truncated"] = True
        truncated = self._last_meta_flag(audio_output, "truncated")
        if truncated is not None:
            collect["audio_truncated"] = (
                collect.get("audio_truncated", False) or truncated is True or str(truncated) == "1"
            )
        # meta.error=1 marks a failed NAR/VAE pass (empty clip); serving turns
        # it into a 500 via this generic key. Absent/0 means success.
        error = self._last_meta_flag(audio_output, "error")
        if error is not None and (error is True or str(error) == "1"):
            collect["audio_synthesis_error"] = True

    def _tokenizer(self) -> Any:
        """The checkpoint-native tiktoken BPE, loaded once per process."""
        if self._cached_tokenizer is None:
            from vllm_omni.tokenizers.yue2_tokenizer import YuE2TextTokenizer

            engine = self.ctx.engine_client
            assert engine is not None
            vllm_config = engine.vllm_config
            assert vllm_config is not None
            config = vllm_config.model_config
            self._cached_tokenizer = YuE2TextTokenizer.from_pretrained(
                config.model,
                revision=config.revision,
                token=config.hf_token,
                cache_dir=vllm_config.load_config.download_dir,
            )
        return self._cached_tokenizer


__all__ = ["Yue2Adapter"]
