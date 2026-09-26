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
- ``abc``: an ABC score as text, required for ``cot=melody|full``. Serving v1
  does not run the model-generated ABC phase (that is two sequential engine
  requests; use the offline example for it).

Unsupported parameters are rejected rather than silently ignored, following
the MiniMax Music 3 precedent.
"""

import secrets
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import (
    ARTTSAdapter,
    PreparedRequest,
    resolve_stage_model_path,
)
from vllm_omni.model_executor.models.yue2.constants import (
    CONTEXT,
    FRAMES_PER_SECOND,
    KEY_MAX_AUDIO_FRAMES,
    KEY_MIN_TOKENS,
    KEY_PENALTY_WINDOW,
    KEY_PHASE,
    KEY_PREFIX_IDS,
    KEY_REPETITION_PENALTY,
    KEY_SEED,
    KEY_SKIP_SYNTHESIS,
    KEY_TEMPERATURE,
    KEY_TOP_K,
    KEY_TOP_P,
    SEMANTIC_SAMPLING,
    STOP_TOKEN_IDS,
)
from vllm_omni.model_executor.models.yue2.prompt import semantic_prefix_ids

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

_COT_MODES = ("off", "melody", "full")

# Same default as the deploy yaml's yue2_max_audio_frames placeholder.
_DEFAULT_MAX_FRAMES = 200

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
        elif not isinstance(abc, str) or not abc.strip():
            return (
                f"cot={cot} requires extra_params.abc carrying an ABC score. "
                "Model-generated ABC is two engine requests and is not served "
                "in v1; use examples/offline_inference/yue2 for that."
            )

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

        rejected = sorted(key for key in _UNSUPPORTED_SAMPLING_PARAMS if extra.get(key) is not None)
        if rejected:
            return "YuE2 has fixed sampling and does not accept: " + ", ".join(rejected)
        return None

    async def build(
        self,
        request: "OpenAICreateSpeechRequest",
        sampling_params_list: list,
        has_inline_ref_audio: bool,
    ) -> PreparedRequest:
        del sampling_params_list, has_inline_ref_audio
        extra: dict[str, Any] = request.extra_params or {}
        cot = extra.get("cot", "off")
        encode = self._tokenizer().encode
        abc_ids = encode(extra["abc"]) if cot != "off" else None
        # validate() has already rejected blank lyrics/caption.
        instructions = request.instructions or ""
        lyrics = request.input or ""
        prefix = semantic_prefix_ids(encode, instructions, lyrics, cot, abc_ids=abc_ids)
        max_frames = int(request.max_new_tokens or _DEFAULT_MAX_FRAMES)
        if len(prefix) + max_frames > CONTEXT:
            return_err = (
                f"YuE2 prompt is {len(prefix)} tokens and the {max_frames}-frame budget "
                f"exceeds the {CONTEXT}-token context. Shorten the lyrics/style or "
                "lower max_new_tokens."
            )
            raise ValueError(return_err)
        return PreparedRequest(
            prompt={"prompt_token_ids": prefix},
            tts_params={"max_audio_frames": [max_frames]},
            model_type=self.name,
        )

    def apply_sampling_overrides(
        self,
        sampling_params_list: list,
        request: "OpenAICreateSpeechRequest",
        prompt: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> list:
        import copy

        sampling_params_list = copy.deepcopy(sampling_params_list)
        params = sampling_params_list[0]
        max_frames = int(request.max_new_tokens or _DEFAULT_MAX_FRAMES)
        # A service must not give every caller the same song; without an
        # explicit seed, draw one per request.
        seed = request.seed if request.seed is not None else secrets.randbelow(2**31)
        assert prompt is not None  # the engine always passes the scheduled prompt
        params.extra_args = {
            **(params.extra_args or {}),
            KEY_PHASE: "semantic",
            KEY_SEED: seed,
            KEY_TEMPERATURE: SEMANTIC_SAMPLING["temperature"],
            KEY_TOP_P: SEMANTIC_SAMPLING["top_p"],
            KEY_TOP_K: SEMANTIC_SAMPLING["top_k"],
            KEY_REPETITION_PENALTY: SEMANTIC_SAMPLING["repetition_penalty"],
            KEY_PENALTY_WINDOW: SEMANTIC_SAMPLING["penalty_window"],
            KEY_MIN_TOKENS: SEMANTIC_SAMPLING["min_tokens"],
            KEY_MAX_AUDIO_FRAMES: max_frames,
            KEY_SKIP_SYNTHESIS: False,
            # Full prompt ids: under a KV prefix-cache hit the engine schedules
            # only the uncached tail, so the model cannot rebuild its NAR
            # conditioning prefix from the scheduled tokens alone.
            KEY_PREFIX_IDS: list(prompt["prompt_token_ids"]),
        }
        # The model needs one decode step beyond the last frame to deliver the
        # terminal NAR/VAE pass, so do not cap the engine at exactly max_frames.
        params.max_tokens = max_frames + 1
        params.stop_token_ids = list(STOP_TOKEN_IDS)
        params.detokenize = False
        return sampling_params_list

    def _tokenizer(self) -> Any:
        """The checkpoint-native tiktoken BPE, loaded once per process."""
        if self._cached_tokenizer is None:
            from vllm_omni.model_executor.models.yue2.tokenizer import YuE2TextTokenizer

            model_path = resolve_stage_model_path(self.ctx.engine_client)
            if model_path is None:
                raise RuntimeError("YuE2 tokenizer needs a resolvable stage model path")
            merge_file = Path(model_path).expanduser() / "qwen.tiktoken"
            if not merge_file.is_file():
                # model_path is an HF repo id: pull just the merge file into
                # the hub cache (vae.py does the same for the VAE snapshot).
                from vllm_omni.transformers_utils.repo_utils import hf_api

                merge_file = Path(hf_api().hf_hub_download(model_path, "qwen.tiktoken"))
            self._cached_tokenizer = YuE2TextTokenizer(merge_file)
        return self._cached_tokenizer


__all__ = ["Yue2Adapter"]
