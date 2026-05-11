# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import copy
import hashlib
import io
import re
import time
from collections.abc import AsyncIterator, MutableMapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import soundfile as sf
import torch
import yaml as _yaml
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams

from vllm_omni.model_executor.models.raon.raon_audio_encoder import (
    compute_num_audio_input_tokens,
)
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.tokenizers.raon_tokenizer import (
    AUDIO_END,
    AUDIO_INPUT_PLACEHOLDER,
    AUDIO_OUTPUT_END_PAD,
    AUDIO_OUTPUT_PAD,
    AUDIO_OUTPUT_PAD_TOKEN,
    AUDIO_OUTPUT_PLACEHOLDER,
    AUDIO_START,
    AUDIO_START_TOKEN,
    IM_END,
    SPEAKER_EMBEDDING_PLACEHOLDER,
    RaonChatTemplateBuilder,
    RaonResolvedIds,
    TaskType,
    normalize_token_ids,
    resolve_raon_special_ids,
)
from vllm_omni.transformers_utils.configs.raon import (
    AUDIO_SAMPLE_RATE,
    ENV,
    TTS_MAX_TOKENS_HARD_CAP,
)

logger = init_logger(__name__)

_DEFAULT_SPEAKER_NPY = Path(__file__).resolve().parent / "assets" / "default_speaker.npy"

# Lazily-cached default speaker embedding (192-dim ECAPA x-vector).
_default_speaker_embedding: torch.Tensor | None = None


_DEFAULT_RESOLVED_IDS = RaonResolvedIds(
    audio_start=AUDIO_START.id,
    audio_end=AUDIO_END.id,
    audio_input_placeholder=AUDIO_INPUT_PLACEHOLDER.id,
    audio_output_placeholder=AUDIO_OUTPUT_PLACEHOLDER.id,
    speaker_placeholder=SPEAKER_EMBEDDING_PLACEHOLDER.id,
    audio_output_pad=AUDIO_OUTPUT_PAD.id,
    audio_output_end_pad=AUDIO_OUTPUT_END_PAD.id,
)


def decode_audio_data_url(data_url: str) -> tuple[np.ndarray, int]:
    """Decode a ``data:audio/...;base64,...`` URL to (waveform, sample_rate).

    Only handles ``data:`` URIs.  Raises :class:`ValueError` for other schemes.
    """
    if not data_url.startswith("data:"):
        raise ValueError(f"Expected data: URI, got {data_url[:40]!r}...")
    b64_data = data_url.split(",", 1)[1] if "," in data_url else data_url
    audio_bytes = base64.b64decode(b64_data)
    audio_np, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if audio_np.ndim > 1:
        audio_np = audio_np[:, 0]  # mono
    return audio_np, int(sr)


def get_default_speaker_embedding() -> torch.Tensor | None:
    """Return the pre-computed 192-dim ECAPA embedding for the default speaker."""
    global _default_speaker_embedding
    if _default_speaker_embedding is not None:
        return _default_speaker_embedding.clone()
    if not _DEFAULT_SPEAKER_NPY.is_file():
        logger.warning("Default speaker embedding not found at %s", _DEFAULT_SPEAKER_NPY)
        return None
    _default_speaker_embedding = torch.from_numpy(np.load(str(_DEFAULT_SPEAKER_NPY))).to(dtype=torch.float32)
    logger.info(
        "Loaded default speaker embedding: shape=%s, norm=%.2f",
        tuple(_default_speaker_embedding.shape),
        _default_speaker_embedding.norm().item(),
    )
    return _default_speaker_embedding.clone()


class _NullTokenizer:
    """Sentinel used when no real tokenizer is available."""


MODALITY_TEXT = "text"
MODALITY_AUDIO = "audio"

OUTPUT_MODE_TEXT_ONLY = "text_only"
OUTPUT_MODE_AUDIO_ONLY = "audio_only"
OUTPUT_MODE_TEXT_AND_AUDIO = "text_and_audio"

_GLOBAL_TOP_K = 20


class _RaonTextOnlyEngineClient:
    _raon_text_only_adapter = True

    def __init__(self, engine_client: Any) -> None:
        self._engine_client = engine_client

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine_client, name)

    def generate(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        kwargs.setdefault("output_modalities", [MODALITY_TEXT])
        if "sampling_params_list" not in kwargs:
            sampling_params = args[1] if len(args) >= 2 else kwargs.get("sampling_params")
            if sampling_params is not None:
                default_params = getattr(self._engine_client, "default_sampling_params_list", None)
                if default_params is not None:
                    sampling_params_list = list(default_params)
                    if sampling_params_list:
                        sampling_params_list[0] = sampling_params
                    kwargs["sampling_params_list"] = sampling_params_list
        generator = self._engine_client.generate(*args, **kwargs)

        async def _unwrap_text_outputs() -> AsyncIterator[Any]:
            async for output in generator:
                if (
                    isinstance(output, OmniRequestOutput)
                    and output.final_output_type == MODALITY_TEXT
                    and output.request_output is not None
                ):
                    yield output.request_output
                else:
                    yield output

        return _unwrap_text_outputs()


def _is_raon_serving(serving: Any) -> bool:
    model_config = getattr(serving, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    return getattr(hf_config, "model_type", None) == "raon"


def _ensure_raon_text_engine_client(serving: Any) -> None:
    if not _is_raon_serving(serving):
        return
    engine_client = getattr(serving, "engine_client", None)
    if engine_client is None or getattr(engine_client, "_raon_text_only_adapter", False):
        return
    serving.engine_client = _RaonTextOnlyEngineClient(engine_client)


# ---------------------------------------------------------------------------
# Task-specific sampling params loaded from raon.yaml (lazy-cached).
# ---------------------------------------------------------------------------
_TASK_SAMPLING_PARAMS: dict[str, dict[str, Any]] | None = None


def _get_task_sampling_params() -> dict[str, dict[str, Any]]:
    """Load task-specific sampling defaults from raon.yaml (cached)."""
    global _TASK_SAMPLING_PARAMS
    if _TASK_SAMPLING_PARAMS is not None:
        return _TASK_SAMPLING_PARAMS
    yaml_path = Path(__file__).resolve().parents[2] / "stage_configs" / "raon.yaml"
    try:
        with open(yaml_path) as f:
            cfg = _yaml.safe_load(f)
        for stage in cfg.get("stage_args", []):
            if stage.get("is_comprehension"):
                _TASK_SAMPLING_PARAMS = stage.get("task_sampling_params", {})
                logger.info("Loaded Raon task_sampling_params: %s", list(_TASK_SAMPLING_PARAMS.keys()))
                return _TASK_SAMPLING_PARAMS
    except Exception:
        logger.warning("Failed to load task_sampling_params from %s", yaml_path, exc_info=True)
    _TASK_SAMPLING_PARAMS = {}
    return _TASK_SAMPLING_PARAMS


# Shared helpers: modalities, tokenizer invariants, TTS request seeding.


def canonicalize_modalities(
    requested_modalities: Sequence[str] | None,
    default_modalities: Sequence[str] | None,
) -> list[str]:
    source = requested_modalities if requested_modalities is not None else default_modalities
    if source is None:
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for modality in source:
        if not isinstance(modality, str):
            continue
        value = modality.strip().lower()
        if not value or value in seen:
            continue
        normalized.append(value)
        seen.add(value)
    return normalized


def modalities_to_output_mode(modalities: Sequence[str] | None) -> str | None:
    if modalities is None:
        return None

    normalized = set(canonicalize_modalities(modalities, None))
    if normalized == {MODALITY_TEXT}:
        return OUTPUT_MODE_TEXT_ONLY
    if normalized == {MODALITY_AUDIO}:
        return OUTPUT_MODE_AUDIO_ONLY
    if normalized == {MODALITY_TEXT, MODALITY_AUDIO}:
        return OUTPUT_MODE_TEXT_AND_AUDIO
    return None


def attach_output_mode_additional_information(
    prompt: MutableMapping[str, Any],
    output_mode: str | None,
) -> None:
    if output_mode is None:
        return

    additional_information = prompt.get("additional_information")
    if not isinstance(additional_information, dict):
        additional_information = {}
    additional_information["output_mode"] = [output_mode]
    prompt["additional_information"] = additional_information


def extract_audio_data_and_sample_rate(mm_output: MutableMapping[str, Any]) -> tuple[Any, int]:
    audio_data = mm_output.get("audio")
    if audio_data is None:
        audio_data = mm_output.get("model_outputs")

    sample_rate = mm_output.get("sr", AUDIO_SAMPLE_RATE)
    if isinstance(sample_rate, list) and sample_rate:
        sample_rate = sample_rate[-1]
    if isinstance(sample_rate, torch.Tensor):
        if sample_rate.numel() == 1:
            sample_rate = sample_rate.item()
        else:
            sample_rate = sample_rate.reshape(-1)[-1].item()
    return audio_data, int(sample_rate)


def compute_placeholder_count(
    audio_len_samples: int,
    *,
    sampling_rate: int,
    frame_rate: float,
) -> int:
    if audio_len_samples <= 0:
        return 0
    return compute_num_audio_input_tokens(
        audio_len_samples,
        sampling_rate=sampling_rate,
        frame_rate=frame_rate,
    )


def _stable_request_seed(
    *,
    request_id: str | None = None,
    text: str,
    task_type: str | None = None,
    ref_audio: str | None = None,
    base_seed: int = 0,
) -> int | None:
    seed_mode = ENV.tts_seed_mode
    if seed_mode in {"", "none", "off", "disabled"}:
        return None
    if seed_mode == "content":
        payload = f"{task_type or ''}\n{text or ''}\n{ref_audio or ''}"
    else:
        payload = str(request_id or "")
    digest = hashlib.blake2s(payload.encode("utf-8"), digest_size=4).digest()
    salt = int.from_bytes(digest, byteorder="little", signed=False)
    return int((int(base_seed) + salt) & 0x7FFFFFFF)


class RaonServingHooks:
    """Encapsulates all Raon-specific serving behaviour."""

    def __init__(self, model_config: Any) -> None:
        self._model_config = model_config
        self._hf_config = getattr(model_config, "hf_config", None)

        self._audio_stop_token_ids: list[int] = [IM_END.id, AUDIO_END.id]

    def get_default_chat_modalities(self) -> list[str]:
        return ["text"]

    def override_output_mode(self, output_mode: str | None) -> str | None:
        return output_mode

    def should_force_audio_first_token(self, output_modalities: list[str]) -> bool:
        return "audio" in output_modalities

    def apply_task_sampling_params(
        self,
        params: SamplingParams,
        *,
        task: str,
        request: Any = None,
    ) -> None:
        """Apply task defaults from YAML, then request overrides.

        Priority: YAML default_sampling_params < task_sampling_params < request
        """
        # 1. Apply task-specific defaults from raon.yaml
        task_defaults = _get_task_sampling_params().get(task, {})
        for k, v in task_defaults.items():
            if hasattr(params, k):
                setattr(params, k, v)
            else:
                logger.warning("Unknown sampling param in YAML task_sampling_params: %s", k)

        # 2. Request overrides take precedence
        if request is not None:
            for field in (
                "temperature",
                "top_p",
                "top_k",
                "max_tokens",
                "min_tokens",
                "seed",
                "repetition_penalty",
                "frequency_penalty",
                "presence_penalty",
            ):
                value = getattr(request, field, None)
                if value is not None:
                    setattr(params, field, value)

        # 3. TTS-specific: stop tokens and audio generation constraints
        if task == "tts":
            stop_ids = list(self._audio_stop_token_ids)
            params.stop = []
            if stop_ids:
                params.stop_token_ids = stop_ids
            params.ignore_eos = False
            params.min_tokens = max(getattr(params, "min_tokens", 0) or 0, 1)

    # Keep backward-compat alias
    def prepare_audio_sampling_params(self, params: SamplingParams) -> None:
        self.apply_task_sampling_params(params, task="tts")

    @staticmethod
    def detect_chat_task(messages: list) -> str:
        """Detect Raon task type from chat message content.

        - audio + text content in user messages → speechqa
        - audio only                            → spokenqa
        - text only                             → textqa
        """
        has_audio = False
        has_text = False
        for msg in messages:
            role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
            if role != "user":
                continue
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
            if isinstance(content, str):
                if content.strip():
                    has_text = True
                continue
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, dict):
                    t = item.get("type", "")
                else:
                    t = getattr(item, "type", "")
                if t in ("audio_url", "input_audio"):
                    has_audio = True
                elif t == "text":
                    text_val = item.get("text", "") if isinstance(item, dict) else getattr(item, "text", "")
                    if text_val and text_val.strip():
                        has_text = True
        if has_audio and has_text:
            return "speechqa"
        if has_audio:
            return "spokenqa"
        return "textqa"

    @staticmethod
    def normalize_audio_mm_item(item: Any) -> Any:
        if isinstance(item, list):
            return [RaonServingHooks.normalize_audio_mm_item(x) for x in item]
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], (int, float)):
            return item[0]
        return item

    def normalize_audio_prompt(
        self,
        engine_prompt: Any,
        tokenizer: Any,
    ) -> None:
        if not isinstance(engine_prompt, dict):
            return
        multi_modal_data = engine_prompt.get("multi_modal_data")
        if not isinstance(multi_modal_data, dict):
            return
        if "audio" not in multi_modal_data:
            return
        multi_modal_data["audio"] = self.normalize_audio_mm_item(multi_modal_data.get("audio"))
        engine_prompt["multi_modal_data"] = multi_modal_data

        prompt_token_ids = engine_prompt.get("prompt_token_ids")
        if isinstance(prompt_token_ids, list) and all(isinstance(t, int) for t in prompt_token_ids):
            special = _DEFAULT_RESOLVED_IDS
            if tokenizer is not None:
                try:
                    special = resolve_raon_special_ids(tokenizer)
                except Exception:
                    pass
            engine_prompt["prompt_token_ids"] = normalize_token_ids(
                prompt_token_ids,
                special=special,
            )

    async def build_tts_prompt(
        self,
        text: str,
        *,
        prepend_speaker_token: bool = False,
        engine_client: Any = None,
    ) -> str:
        tokenizer = None
        if engine_client is not None:
            try:
                tokenizer = await engine_client.get_tokenizer()
            except Exception:
                pass

        builder = RaonChatTemplateBuilder(tokenizer=tokenizer or _NullTokenizer())
        result = builder.build_prompt(
            task=TaskType.TTS,
            user_content=text,
            tts_prompt_style=ENV.tts_prompt_style,
            prepend_speaker_token=prepend_speaker_token,
            append_audio_start=ENV.tts_append_audio_start,
        )
        return result.prompt_text

    def estimate_tts_min_tokens(self, text: str, *, max_tokens: int | None) -> int:
        words = [w for w in text.split() if w]
        estimated = max(6, int(len(words) * 2))
        if max_tokens is not None:
            estimated = min(estimated, max(0, int(max_tokens) - 1))
        return max(0, estimated)

    def estimate_tts_max_tokens(self, text: str, *, hard_cap: int = TTS_MAX_TOKENS_HARD_CAP) -> int:
        fixed_override = ENV.tts_fixed_max_tokens
        if fixed_override is not None:
            return max(1, min(int(hard_cap), int(fixed_override)))

        # Char-based scaling with word-based cap for spaced scripts
        # to prevent over-allocation on short text.
        char_estimate = max(96, len(text) * 20)
        words = [w for w in text.split() if w]
        n_words = len(words)
        if n_words > 0 and len(text) / n_words < 8:
            word_estimate = max(96, n_words * 16 + 64)
            char_estimate = min(char_estimate, word_estimate)
        return max(1, min(int(hard_cap), char_estimate))

    def apply_sampling_parity(
        self,
        sampling_params_list: list[Any],
        comprehension_stage_index: int,
    ) -> None:
        if not sampling_params_list:
            return
        idx = max(0, min(comprehension_stage_index, len(sampling_params_list) - 1))
        self.prepare_audio_sampling_params(sampling_params_list[idx])

    def _configured_speaker_token_id(self) -> int | None:
        hf_config = self._hf_config
        speaker_token_id = getattr(hf_config, "speaker_token_id", None)
        if isinstance(speaker_token_id, (list, tuple)):
            speaker_token_id = speaker_token_id[0] if speaker_token_id else None
        return int(speaker_token_id) if isinstance(speaker_token_id, int) else None

    def has_speaker_token(self) -> bool:
        if self._configured_speaker_token_id() is not None:
            return True
        hf_config = self._hf_config
        return getattr(hf_config, "speaker_encoder_config", None) is not None

    def resolve_chat_modalities(
        self,
        request: Any,
        engine_prompts: list,
        tokenizer: Any,
    ) -> list[str]:
        requested = getattr(request, "modalities", None)
        if requested is not None and set(requested) != {"text"}:
            logger.info(
                "Raon chat: overriding client modalities=%s to ['text']; use /v1/audio/speech for speech synthesis.",
                list(requested),
            )
        default = self.get_default_chat_modalities()
        normalized = canonicalize_modalities(
            requested_modalities=None,
            default_modalities=default or getattr(request, "_engine_output_modalities", None),
        )
        output_modalities = normalized or getattr(request, "_engine_output_modalities", ["text"])
        request.modalities = output_modalities

        output_mode = modalities_to_output_mode(normalized)
        output_mode = self.override_output_mode(output_mode)
        force_audio = self.should_force_audio_first_token(output_modalities)

        for ep in engine_prompts:
            attach_output_mode_additional_information(ep, output_mode)
            if force_audio:
                ai = ep.get("additional_information")
                if not isinstance(ai, dict):
                    ai = {}
                ai["force_audio_first_token"] = [True]
                ep["additional_information"] = ai
            self.normalize_audio_prompt(ep, tokenizer)

        return output_modalities

    async def _build_icl_tts_prompt(
        self,
        *,
        target_text: str,
        ref_text: str,
        prepend_speaker_token: bool = True,
        engine_client: Any = None,
    ) -> str:
        """Build ICL prompt for voice cloning.

        Prompt layout::

            <|im_start|>user
            [speaker_embed] Speak the following text:
            {ref_text} {target_text}<|im_end|>
            <|im_start|>assistant
            <|audio_start|>[output_embed x N_ref]

        No trailing ``<audio_start>`` — the model continues generating
        from the last ref frame.
        """
        user_content = f"Speak the following text:\n{ref_text} {target_text}"
        if prepend_speaker_token:
            speaker_token = SPEAKER_EMBEDDING_PLACEHOLDER.text
            if speaker_token is not None and not user_content.startswith(speaker_token):
                user_content = f"{speaker_token}{user_content}"

        assistant_content = f"{AUDIO_START_TOKEN}{AUDIO_OUTPUT_PAD_TOKEN}"

        tokenizer = None
        if engine_client is not None:
            try:
                tokenizer = await engine_client.get_tokenizer()
            except Exception:
                pass

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
        prompt: str | None = None
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                if isinstance(prompt, str):
                    # Remove trailing <|im_end|> — model continues from here.
                    prompt = re.sub(r"<\|im_end\|>\s*$", "", prompt)
            except Exception:
                prompt = None

        if prompt is None:
            prompt = f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n{assistant_content}"

        return prompt

    @staticmethod
    def is_icl_request(request: Any) -> bool:
        """Return True if request is an ICL voice cloning request."""
        task_type = getattr(request, "task_type", None)
        if task_type is None:
            ref_audio = getattr(request, "ref_audio", None)
            if ref_audio is not None:
                task_type = "Base"
        if task_type != "Base":
            return False
        x_vector_only = bool(getattr(request, "x_vector_only_mode", False))
        if x_vector_only:
            return False
        ref_audio = getattr(request, "ref_audio", None)
        ref_text = getattr(request, "ref_text", None)
        return bool(isinstance(ref_audio, str) and ref_audio.strip() and isinstance(ref_text, str) and ref_text.strip())

    @staticmethod
    def validate_request(request: Any) -> str | None:
        """Validate Raon request. Returns error message or None."""
        # Auto-infer Base task when ref_audio, ref_text, or speaker_embedding is provided.
        has_speaker_embedding = getattr(request, "speaker_embedding", None) is not None
        if request.task_type is None and (
            request.ref_audio is not None or request.ref_text is not None or has_speaker_embedding
        ):
            request.task_type = "Base"

        task_type = request.task_type

        if task_type == "Base":
            if request.ref_audio is None and not has_speaker_embedding:
                return "Base task requires 'ref_audio' or 'speaker_embedding' for voice cloning"
            ref_audio = request.ref_audio
            if ref_audio is not None:
                is_valid_scheme = (
                    ref_audio.startswith(("http://", "https://"))
                    or ref_audio.startswith("data:")
                    or ref_audio.startswith("file://")
                )
                if not is_valid_scheme:
                    return "ref_audio must be a URL (http/https), base64 data URL (data:...), or file URI (file://...)"
                if not request.x_vector_only_mode:
                    if not request.ref_text or not request.ref_text.strip():
                        request.x_vector_only_mode = True

        if task_type != "Base":
            if request.ref_text is not None:
                return "'ref_text' is only valid for Base task"
            if request.x_vector_only_mode is not None:
                return "'x_vector_only_mode' is only valid for Base task"

        if not request.input or not request.input.strip():
            return "Input text cannot be empty"

        if request.max_new_tokens is not None:
            if request.max_new_tokens < 1:
                return "max_new_tokens must be at least 1"
            if request.max_new_tokens > 8192:
                return "max_new_tokens cannot exceed 8192"

        return None

    @staticmethod
    async def _resolve_ref_audio(ref_audio_str: str, model_config: Any) -> tuple:
        """Resolve ref_audio URI to (ndarray, sample_rate) using vLLM MediaConnector."""

        from vllm.multimodal.media import MediaConnector

        connector = MediaConnector(
            allowed_local_media_path=model_config.allowed_local_media_path,
            allowed_media_domains=model_config.allowed_media_domains,
        )
        wav_np, sr = await connector.fetch_audio_async(ref_audio_str)
        wav_np = np.asarray(wav_np, dtype=np.float32)
        if wav_np.ndim > 1:
            wav_np = np.mean(wav_np, axis=-1)
        return wav_np, int(sr)

    @classmethod
    def apply_default_modalities(cls) -> None:
        from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
        from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses

        from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

        if not getattr(OmniOpenAIServingChat, "_raon_default_modalities_applied", False):
            orig_preprocess = OmniOpenAIServingChat._preprocess_chat

            async def _preprocess_chat(self, request, *args, **kwargs):
                conversation, engine_prompts = await orig_preprocess(self, request, *args, **kwargs)
                hf = getattr(self.model_config, "hf_config", None)
                if getattr(hf, "model_type", None) == "raon":
                    request._engine_output_modalities = self.engine_client.output_modalities
                    tokenizer = await self.engine_client.get_tokenizer()
                    cls(self.model_config).resolve_chat_modalities(request, engine_prompts, tokenizer)
                return conversation, engine_prompts

            OmniOpenAIServingChat._preprocess_chat = _preprocess_chat
            OmniOpenAIServingChat._raon_default_modalities_applied = True

        if not getattr(OpenAIServingCompletion, "_raon_text_modalities_applied", False):
            orig_create_completion = OpenAIServingCompletion.create_completion

            async def _create_completion(self, *args, **kwargs):
                _ensure_raon_text_engine_client(self)
                return await orig_create_completion(self, *args, **kwargs)

            OpenAIServingCompletion.create_completion = _create_completion
            OpenAIServingCompletion._raon_text_modalities_applied = True

        if not getattr(OpenAIServingResponses, "_raon_text_modalities_applied", False):
            orig_create_responses = OpenAIServingResponses.create_responses

            async def _create_responses(self, *args, **kwargs):
                _ensure_raon_text_engine_client(self)
                return await orig_create_responses(self, *args, **kwargs)

            OpenAIServingResponses.create_responses = _create_responses
            OpenAIServingResponses._raon_text_modalities_applied = True


def _get_raon_serving_hooks(serving: Any) -> RaonServingHooks:
    hooks = getattr(serving, "_raon_hooks", None)
    if hooks is not None:
        return hooks
    return RaonServingHooks(serving.model_config)


def prepare_raon_tts_sampling_params(
    serving: Any,
    sampling_params_list: list[Any],
    request: Any,
) -> list[Any]:
    if not sampling_params_list:
        return sampling_params_list

    prepared = copy.deepcopy(sampling_params_list)
    hooks = _get_raon_serving_hooks(serving)
    hooks.apply_task_sampling_params(
        prepared[0],
        task="tts",
        request=request,
    )
    if getattr(request, "max_new_tokens", None) is not None:
        prepared[0].max_tokens = request.max_new_tokens
    return prepared


async def build_raon_speech_prompt(serving: Any, request: Any) -> dict[str, Any]:
    """Build the Raon speech prompt and serving metadata.

    Applies speaker selection, ICL-specific additional_information, and the
    serving-side max_new_tokens heuristic used by /v1/audio/speech.
    """
    hooks = _get_raon_serving_hooks(serving)
    additional_info: dict[str, Any] = {
        "force_audio_first_token": [True],
        "output_mode": ["audio_only"],
    }

    has_speaker = False
    is_icl = hooks.is_icl_request(request)

    if request.voice and request.voice.lower() in serving.uploaded_speakers and request.ref_audio is None:
        speaker_info = serving.uploaded_speakers[request.voice.lower()]
        if speaker_info.get("embedding_source") == "direct":
            stored_emb = serving._get_uploaded_speaker_embedding(request.voice)
            if stored_emb is not None:
                additional_info["cached_spk_embedding"] = [torch.tensor(stored_emb, dtype=torch.float32).tolist()]
                has_speaker = True
            else:
                logger.warning("Uploaded voice '%s' has no stored embedding", request.voice)
        else:
            get_uploaded_audio_data = getattr(serving, "_get_uploaded_audio_data", None)
            audio_data = get_uploaded_audio_data(request.voice) if callable(get_uploaded_audio_data) else None
            if audio_data:
                additional_info["speaker_ref_audio"] = [audio_data]
                has_speaker = True
            else:
                logger.warning(
                    "Uploaded voice '%s' audio data unavailable from %s",
                    request.voice,
                    speaker_info.get("file_path"),
                )
    elif request.speaker_embedding is not None:
        additional_info["cached_spk_embedding"] = [
            torch.tensor(request.speaker_embedding, dtype=torch.float32).tolist()
        ]
        has_speaker = True
    elif request.ref_audio is not None and isinstance(request.ref_audio, str):
        anchor_override = getattr(request, "_speaker_anchor_ref_audio", None)
        if isinstance(anchor_override, str) and anchor_override.strip():
            additional_info["speaker_ref_audio"] = [anchor_override]
        else:
            additional_info["speaker_ref_audio"] = [request.ref_audio]
        has_speaker = True

    if not has_speaker and not is_icl:
        default_emb = get_default_speaker_embedding()
        if default_emb is not None:
            additional_info["cached_spk_embedding"] = [default_emb.tolist()]
            has_speaker = True

    if is_icl:
        from vllm_omni.transformers_utils.configs.raon import ENV as _RAON_ENV_ICL

        additional_info["icl_mode"] = [True]
        additional_info["source_ref_text"] = [str(request.ref_text or "")]
        additional_info["continuation_silence_frames"] = [int(_RAON_ENV_ICL.continuation_silence_frames)]
        min_audio_steps = getattr(request, "_raon_min_audio_steps", None)
        if min_audio_steps is not None and int(min_audio_steps) > 0:
            additional_info["audio_min_steps"] = [int(min_audio_steps)]
        prompt_text = await hooks._build_icl_tts_prompt(
            target_text=request.input,
            ref_text=request.ref_text,
            prepend_speaker_token=has_speaker,
            engine_client=serving.engine_client,
        )
        prompt: dict[str, Any] = {"prompt": prompt_text}
        if request.ref_audio and isinstance(request.ref_audio, str):
            try:
                ref_np, ref_sr = await serving._resolve_ref_audio(request.ref_audio)
                ref_np = np.asarray(ref_np, dtype=np.float32)
                prompt["multi_modal_data"] = {"audio": [(ref_np, ref_sr)]}
            except Exception:
                logger.warning("Failed to resolve ref_audio for ICL prefill", exc_info=True)
    else:
        prompt_text = await hooks.build_tts_prompt(
            request.input,
            prepend_speaker_token=has_speaker,
            engine_client=serving.engine_client,
        )
        prompt = {"prompt": prompt_text}

    heuristic_max = hooks.estimate_tts_max_tokens(request.input)
    preserve_explicit_max_tokens = bool(getattr(request, "_rolling_plan_budget_explicit", False))
    if request.max_new_tokens is None:
        request.max_new_tokens = heuristic_max
    elif not preserve_explicit_max_tokens and int(request.max_new_tokens) > heuristic_max:
        logger.info(
            "Raon TTS: clamping client max_new_tokens=%d to heuristic cap %d for input_len=%d",
            int(request.max_new_tokens),
            heuristic_max,
            len(request.input),
        )
        request.max_new_tokens = heuristic_max

    prompt["additional_information"] = additional_info
    return prompt


async def collect_request_pcm(serving: Any, request: Any) -> tuple[np.ndarray, int]:
    """Run one unary speech request and return final PCM + sample rate."""
    request_id, generator, _ = await serving._prepare_speech_generation(request)

    final_output = None
    async for res in generator:
        final_output = res

    if final_output is None:
        raise ValueError("No output generated from the model.")

    audio_output, audio_key = serving._extract_audio_output(final_output)
    if audio_key is None:
        raise ValueError("TTS model did not produce audio output.")

    audio_tensor = audio_output[audio_key]
    sr_raw = audio_output.get("sr", 24000)
    sr_val = sr_raw[-1] if isinstance(sr_raw, list) and sr_raw else sr_raw
    sample_rate = sr_val.item() if hasattr(sr_val, "item") else int(sr_val)

    if isinstance(audio_tensor, list):
        async_chunk = bool(getattr(serving.engine_client.model_config, "async_chunk", False))
        if async_chunk:
            non_empty_chunks = [candidate for candidate in audio_tensor if candidate.numel() > 0]
            audio_tensor = torch.cat(non_empty_chunks, dim=-1) if non_empty_chunks else np.zeros((0,), dtype=np.float32)
        else:
            audio_history = audio_tensor
            audio_tensor = np.zeros((0,), dtype=np.float32)
            for candidate in reversed(audio_history):
                if candidate.numel() > 0:
                    audio_tensor = candidate
                    break
    if hasattr(audio_tensor, "float"):
        audio_tensor = audio_tensor.float().detach().cpu().numpy()

    if audio_tensor.ndim > 1:
        audio_tensor = audio_tensor.squeeze()
    return audio_tensor, int(sample_rate)


async def collect_best_of_k_request_pcm(
    serving: Any,
    request: Any,
    *,
    k: int,
    score_mode: str = "duration",
    target_words: int | None = None,
    expected_wps: float | None = None,
    early_exit_ratio: float | None = None,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    """Generate up to ``k`` Raon candidates and keep the best one.

    Candidates advance the deterministic seed one step at a time and can exit
    early once duration clears the expected threshold.
    """
    k_int = max(1, int(k))

    base_seed_opt = _stable_request_seed(
        text=request.input or "",
        task_type=getattr(request, "task_type", None),
        ref_audio=getattr(request, "ref_audio", None),
        base_seed=0,
    )
    base_seed = int(base_seed_opt) if base_seed_opt is not None else 0

    expected_duration_s: float | None = None
    if target_words is not None and expected_wps is not None and float(expected_wps) > 0 and int(target_words) > 0:
        expected_duration_s = float(target_words) / float(expected_wps)

    exit_ratio = float(early_exit_ratio) if early_exit_ratio is not None else 0.0
    early_exit_threshold_s: float | None = (
        expected_duration_s * exit_ratio if (expected_duration_s is not None and exit_ratio > 0) else None
    )

    mode = (score_mode or "duration").strip().lower()
    if mode not in ("duration", "duration_then_length"):
        logger.info(
            "Raon best-of-k score_mode=%r not implemented; falling back to 'duration'.",
            score_mode,
        )
        mode = "duration"

    candidates_meta: list[dict[str, Any]] = []
    results: list[tuple[np.ndarray, int]] = []
    early_exit_hit = False

    for idx in range(k_int):
        cand = request.model_copy(deep=True)
        if getattr(request, "_rolling_plan_budget_explicit", False):
            object.__setattr__(cand, "_rolling_plan_budget_explicit", True)
        object.__setattr__(cand, "seed", int((base_seed + idx) & 0x7FFFFFFF))

        pcm, sr = await collect_request_pcm(serving, cand)
        duration_s = float(np.asarray(pcm).size) / float(max(1, int(sr)))
        results.append((pcm, sr))
        candidates_meta.append({"idx": int(idx), "duration_s": duration_s, "sample_rate": int(sr)})

        if early_exit_threshold_s is not None and duration_s >= early_exit_threshold_s:
            early_exit_hit = True
            break

    if mode == "duration_then_length":
        selected_idx = max(
            range(len(results)),
            key=lambda i: (
                candidates_meta[i]["duration_s"],
                int(np.asarray(results[i][0]).size),
                -i,
            ),
        )
    else:
        selected_idx = max(
            range(len(results)),
            key=lambda i: (candidates_meta[i]["duration_s"], -i),
        )

    best_pcm, best_sr = results[selected_idx]
    selection_meta: dict[str, Any] = {
        "k_max": int(k_int),
        "generated_candidates_count": int(len(results)),
        "early_exit_hit": bool(early_exit_hit),
        "early_exit_threshold_s": early_exit_threshold_s,
        "score_mode": mode,
        "candidates": candidates_meta,
        "selected_idx": int(selected_idx),
        "selected_duration_s": float(candidates_meta[selected_idx]["duration_s"]),
        "expected_duration_s": expected_duration_s,
    }
    return np.asarray(best_pcm, dtype=np.float32), int(best_sr), selection_meta


def estimate_final_eos_min_steps(
    *,
    target_words: int,
    previous_chunk_stats: Sequence[tuple[int, float]],
    fallback_wps: float,
    min_duration_ratio: float,
    frame_rate_hz: float,
) -> int:
    if int(target_words) <= 0 or float(min_duration_ratio) <= 0 or float(frame_rate_hz) <= 0:
        return 0
    observed_words = 0
    observed_duration_s = 0.0
    for words, duration_s in previous_chunk_stats:
        if int(words) > 0 and float(duration_s) > 0:
            observed_words += int(words)
            observed_duration_s += float(duration_s)
    if observed_words > 0 and observed_duration_s > 0:
        wps = float(observed_words) / float(observed_duration_s)
    else:
        wps = float(fallback_wps)
    if wps <= 0:
        return 0
    expected_duration_s = float(target_words) / wps
    return max(0, int(round(expected_duration_s * float(min_duration_ratio) * float(frame_rate_hz))))


async def generate_raon_long_tts_rolling_icl(serving: Any, request: Any) -> tuple[np.ndarray, int]:
    """Run long-form Raon TTS with sentence-chunked rolling ICL.

    Later chunks feed the previous PCM back as ref_audio/ref_text, preserve the
    original speaker anchor when configured, and optionally use final best-of-k.
    """
    text = request.input or ""
    sentences = split_text_into_sentences(text)
    if not sentences:
        raise ValueError("rolling_icl received empty/unsegmentable input")
    raw_chunks = build_sentence_chunks(
        sentences,
        max_sentences_per_chunk=int(ENV.tts_long_max_sentences_per_chunk),
    )

    plans: list[ChunkPlan] = [ChunkPlan(sentences=tuple(rc), boundary_kind="sentence_end") for rc in raw_chunks]

    anchor_reset_every = max(0, int(ENV.tts_long_anchor_reset_every_chunks))
    keep_speaker_anchor = bool(ENV.tts_long_keep_original_speaker_anchor)
    min_ref_audio_s = float(ENV.tts_long_min_ref_audio_s)

    original_ref_audio_anchor: str | None = None
    if (
        keep_speaker_anchor
        and isinstance(request.ref_audio, str)
        and request.ref_audio.strip()
        and not request.voice
        and getattr(request, "speaker_embedding", None) is None
    ):
        original_ref_audio_anchor = request.ref_audio
        try:
            _anchor_wav, _anchor_sr = await serving._resolve_ref_audio(original_ref_audio_anchor)
            logger.info(
                "Raon rolling-ICL: cached speaker anchor from ref_audio (sr=%d samples=%d)",
                int(_anchor_sr),
                int(len(_anchor_wav)),
            )
        except Exception as _anchor_err:
            logger.warning(
                "Raon rolling-ICL: speaker anchor probe failed: %s",
                _anchor_err,
            )

    total_budget = request.max_new_tokens
    per_plan_budget: list[int | None] = [None] * len(plans)
    if total_budget is not None and total_budget > 0:
        if total_budget < len(plans):
            logger.info(
                "Raon rolling-ICL fallback: max_new_tokens=%d < plans=%d; "
                "routing to one-shot path to honour the explicit cap.",
                total_budget,
                len(plans),
            )
            return await collect_request_pcm(serving, request)
        plan_word_counts = [count_words(p.text) for p in plans]
        total_words = max(1, sum(plan_word_counts))
        remaining = int(total_budget)
        for i, wc in enumerate(plan_word_counts):
            if i == len(plans) - 1:
                per_plan_budget[i] = max(1, remaining)
            else:
                share = max(1, int(total_budget * wc / total_words))
                share = min(share, remaining - (len(plans) - i - 1))
                per_plan_budget[i] = share
                remaining -= share

    logger.info(
        "Raon rolling-ICL: words=%d raw_chunks=%d plans=%d anchor_every=%d ref_mode=%s",
        len(text.split()),
        len(raw_chunks),
        len(plans),
        anchor_reset_every,
        ENV.tts_long_ref_text_mode,
    )

    segments: list[np.ndarray] = []
    sample_rate: int = 24000
    prev_pcm: np.ndarray | None = None
    prev_sr: int | None = None
    chunk_stats: list[tuple[int, float]] = []

    for chunk_idx, plan in enumerate(plans):
        chunk_text = plan.text
        chunk_req = request.model_copy(deep=True)
        chunk_req.input = chunk_text
        chunk_req.stream = False

        chunk_req.max_new_tokens = per_plan_budget[chunk_idx]
        if chunk_req.max_new_tokens is not None:
            object.__setattr__(chunk_req, "_rolling_plan_budget_explicit", True)

        is_anchor_reset = anchor_reset_every > 0 and chunk_idx > 0 and (chunk_idx % anchor_reset_every == 0)
        if chunk_idx == 0 or is_anchor_reset or prev_pcm is None:
            used_task_type = chunk_req.task_type or request.task_type or "<original>"
            ref_text_len = len(chunk_req.ref_text or "")
            speaker_source = "original_anchor"
        else:
            prev_plan = plans[chunk_idx - 1]
            ref_text = get_ref_text_for_next_chunk(prev_plan.text, mode=ENV.tts_long_ref_text_mode).strip()
            ref_audio_url = encode_pcm_to_wav_data_url(prev_pcm, prev_sr or sample_rate)
            chunk_req.task_type = "Base"
            chunk_req.ref_audio = ref_audio_url
            chunk_req.ref_text = ref_text or "..."
            if not keep_speaker_anchor:
                chunk_req.voice = None
                chunk_req.x_vector_only_mode = False
                chunk_req.speaker_embedding = None
            elif original_ref_audio_anchor is not None:
                object.__setattr__(chunk_req, "_speaker_anchor_ref_audio", original_ref_audio_anchor)
            used_task_type = "Base"
            ref_text_len = len(chunk_req.ref_text or "")
            speaker_source = "original_anchor" if keep_speaker_anchor else "prev_pcm"

        t0 = time.perf_counter()
        is_final_chunk = chunk_idx == len(plans) - 1
        target_words_final = count_words(chunk_text)
        final_expected_duration_s: float | None = None
        if is_final_chunk and ENV.tts_long_enable_final_eos_min_gate:
            min_steps = estimate_final_eos_min_steps(
                target_words=target_words_final,
                previous_chunk_stats=chunk_stats,
                fallback_wps=float(ENV.tts_long_final_best_of_k_expected_wps),
                min_duration_ratio=float(ENV.tts_long_final_eos_min_duration_ratio),
                frame_rate_hz=float(ENV.tts_long_final_eos_gate_frame_rate_hz),
            )
            if min_steps > 0:
                object.__setattr__(chunk_req, "_raon_min_audio_steps", int(min_steps))
                final_expected_duration_s = (
                    float(min_steps)
                    / max(1e-6, float(ENV.tts_long_final_eos_gate_frame_rate_hz))
                    / max(1e-6, float(ENV.tts_long_final_eos_min_duration_ratio))
                )
                logger.debug(
                    "Raon rolling-ICL final EOS min gate: target_words=%d min_steps=%d expected_duration_s=%.2f",
                    target_words_final,
                    min_steps,
                    final_expected_duration_s,
                )
        if is_final_chunk and ENV.tts_long_enable_final_best_of_k and int(ENV.tts_long_final_best_of_k) >= 2:
            k = int(ENV.tts_long_final_best_of_k)
            chunk_pcm, chunk_sr, best_meta = await collect_best_of_k_request_pcm(
                serving,
                chunk_req,
                k=k,
                score_mode=str(ENV.tts_long_final_best_of_k_score_mode),
                target_words=target_words_final,
                expected_wps=float(ENV.tts_long_final_best_of_k_expected_wps),
                early_exit_ratio=float(ENV.tts_long_final_best_of_k_early_exit_ratio),
            )
            logger.debug(
                "Raon rolling-ICL final best-of-%d: generated=%d early_exit_hit=%s "
                "selected_idx=%d selected_duration_s=%.2f expected_duration_s=%s candidates=%s",
                k,
                best_meta["generated_candidates_count"],
                best_meta["early_exit_hit"],
                best_meta["selected_idx"],
                best_meta["selected_duration_s"],
                None if best_meta["expected_duration_s"] is None else round(float(best_meta["expected_duration_s"]), 2),
                [round(c["duration_s"], 2) for c in best_meta["candidates"]],
            )
        else:
            chunk_pcm, chunk_sr = await collect_request_pcm(serving, chunk_req)
            if (
                is_final_chunk
                and ENV.tts_long_enable_final_eos_min_gate
                and ENV.tts_long_enable_final_eos_retry
                and final_expected_duration_s is not None
            ):
                retry_threshold_s = final_expected_duration_s * float(ENV.tts_long_final_eos_retry_duration_ratio)
                first_duration_s = float(np.asarray(chunk_pcm).size) / float(max(1, int(chunk_sr)))
                if first_duration_s < retry_threshold_s:
                    retry_req = chunk_req.model_copy(deep=True)
                    if getattr(chunk_req, "_rolling_plan_budget_explicit", False):
                        object.__setattr__(retry_req, "_rolling_plan_budget_explicit", True)
                    if getattr(chunk_req, "_raon_min_audio_steps", None) is not None:
                        min_audio_steps = getattr(chunk_req, "_raon_min_audio_steps")
                        object.__setattr__(
                            retry_req,
                            "_raon_min_audio_steps",
                            min_audio_steps,
                        )
                    base_seed_opt = _stable_request_seed(
                        text=retry_req.input or "",
                        task_type=getattr(retry_req, "task_type", None),
                        ref_audio=getattr(retry_req, "ref_audio", None),
                        base_seed=0,
                    )
                    object.__setattr__(retry_req, "seed", int(((base_seed_opt or 0) + 1) & 0x7FFFFFFF))
                    retry_pcm, retry_sr = await collect_request_pcm(serving, retry_req)
                    retry_duration_s = float(np.asarray(retry_pcm).size) / float(max(1, int(retry_sr)))
                    logger.debug(
                        "Raon rolling-ICL final EOS retry: first_duration_s=%.2f retry_duration_s=%.2f "
                        "threshold_s=%.2f selected=%s",
                        first_duration_s,
                        retry_duration_s,
                        retry_threshold_s,
                        "retry" if retry_duration_s > first_duration_s else "first",
                    )
                    if retry_duration_s > first_duration_s:
                        chunk_pcm, chunk_sr = retry_pcm, retry_sr
        duration_s = chunk_pcm.size / float(max(1, chunk_sr))
        elapsed = time.perf_counter() - t0
        target_words_log = count_words(chunk_text)
        ref_words_log = count_words(chunk_req.ref_text or "") if used_task_type == "Base" else 0

        logger.debug(
            "Raon rolling-ICL chunk %d/%d: sentences=%d boundary=%s ref_text_len=%d "
            "ref_words=%d target_words=%d task=%s speaker_source=%s max_new_tokens=%s "
            "duration_s=%.2f elapsed_s=%.2f",
            chunk_idx,
            len(plans) - 1,
            len(plan.sentences),
            plan.boundary_kind,
            ref_text_len,
            ref_words_log,
            target_words_log,
            used_task_type,
            speaker_source,
            chunk_req.max_new_tokens,
            duration_s,
            elapsed,
        )

        raw_pcm = np.asarray(chunk_pcm, dtype=np.float32)
        trimmed = trim_leading_trailing_silence(raw_pcm, int(chunk_sr))
        segments.append(trimmed)
        sample_rate = int(chunk_sr)

        if duration_s < min_ref_audio_s:
            logger.warning(
                "Raon rolling-ICL: chunk %d duration=%.2fs < min_ref=%.2fs; "
                "dropping from ref_audio chain (next chunk uses original path).",
                chunk_idx,
                duration_s,
                min_ref_audio_s,
            )
            prev_pcm = None
            prev_sr = None
        else:
            prev_pcm = raw_pcm
            prev_sr = sample_rate
        chunk_stats.append((target_words_log, duration_s))

    pauses_ms = [pause_ms_for_boundary(plans[k].boundary_kind) for k in range(len(plans) - 1)]
    logger.info("Raon rolling-ICL complete: plans=%d", len(plans))
    final_pcm = concat_audio_segments(segments, pauses_ms, sample_rate)
    return final_pcm, sample_rate


# Rolling-ICL helpers.
# Long-form Raon TTS can chunk text and feed prior PCM back as ref_audio when
# ENV.tts_long_mode == "rolling_icl".


def should_use_rolling_icl(
    text: str,
    *,
    mode: str | None = None,
    threshold: int | None = None,
) -> bool:
    """Return whether rolling-ICL should handle this text.

    ``mode`` and ``threshold`` override the frozen ENV snapshot for tests.
    """
    effective_mode = mode if mode is not None else ENV.tts_long_mode
    if effective_mode != "rolling_icl":
        return False
    effective_threshold = int(threshold) if threshold is not None else int(ENV.tts_long_word_threshold)
    n_words = len((text or "").split())
    return n_words >= effective_threshold


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_MIN_WORDS_PER_SENTENCE = 3


def split_text_into_sentences(text: str) -> list[str]:
    """Split text conservatively and merge tiny fragments."""
    text = (text or "").strip()
    if not text:
        return []
    raw = _SENTENCE_BOUNDARY.split(text)
    sentences = [s.strip() for s in raw if s and s.strip()]
    merged: list[str] = []
    for s in sentences:
        if merged and len(s.split()) < _MIN_WORDS_PER_SENTENCE:
            merged[-1] = f"{merged[-1]} {s}".strip()
        else:
            merged.append(s)
    return merged


def build_sentence_chunks(
    sentences: list[str],
    *,
    max_sentences_per_chunk: int = 2,
) -> list[list[str]]:
    """Chunk sentences and merge tiny trailing tails into the previous chunk."""
    if not sentences:
        return []
    step = max(1, int(max_sentences_per_chunk))
    chunks: list[list[str]] = [sentences[i : i + step] for i in range(0, len(sentences), step)]
    if len(chunks) >= 2 and len(chunks[-1]) == 1:
        tail_words = len(chunks[-1][0].split())
        if tail_words < _MIN_WORDS_PER_SENTENCE * 2:
            chunks[-2] = chunks[-2] + chunks[-1]
            chunks.pop()
    return chunks


class ChunkPlan(NamedTuple):
    """Rolling-ICL chunk plus downstream boundary kind.

    Current planning emits sentence_end boundaries, while other kinds remain
    reserved for pause/ref-text handling.
    """

    sentences: tuple[str, ...]
    boundary_kind: str

    @property
    def text(self) -> str:
        return " ".join(self.sentences)


def pause_ms_for_boundary(kind: str) -> int:
    """Return the stitch pause for a chunk boundary."""
    if kind == "clause_split":
        return int(ENV.tts_long_pause_ms_clause)
    return int(ENV.tts_long_pause_ms_period)


_CLAUSE_SPLIT_RE = re.compile(r"(?<=[,;:])\s+")


def get_ref_text_for_next_chunk(prev_chunk_text: str, mode: str | None = None) -> str:
    """Pick the ref_text fed into the next rolling chunk.

    ``mode`` mirrors ENV.tts_long_ref_text_mode and may select the full chunk,
    the last sentence, or the last clause.
    """
    effective_mode = (mode or ENV.tts_long_ref_text_mode or "last_sentence").strip().lower()
    text = (prev_chunk_text or "").strip()
    if not text:
        return ""
    if effective_mode == "full_prev_chunk":
        return text
    if effective_mode == "last_sentence":
        sents = split_text_into_sentences(text)
        return sents[-1] if sents else text
    if effective_mode == "last_clause":
        parts = _CLAUSE_SPLIT_RE.split(text)
        parts = [p.strip() for p in parts if p.strip()]
        return parts[-1] if parts else text
    # Unknown mode: fall through to safe default (last_sentence).
    sents = split_text_into_sentences(text)
    return sents[-1] if sents else text


def infer_pause_ms_from_chunk(chunk_text: str) -> int:
    """Compatibility shim for tests; prefer pause_ms_for_boundary()."""
    text = (chunk_text or "").rstrip()
    if text.endswith((".", "!", "?", "。", "！", "？")):
        return int(ENV.tts_long_pause_ms_period)
    return int(ENV.tts_long_pause_ms_clause)


def trim_leading_trailing_silence(
    audio: np.ndarray,
    sample_rate: int,
    *,
    threshold_dbfs: float = -45.0,
    max_trim_s: float = 0.25,
) -> np.ndarray:
    if not ENV.tts_long_enable_stitch_trim:
        return audio
    if audio.size == 0:
        return audio
    x = np.asarray(audio, dtype=np.float32)
    threshold = 10.0 ** (float(threshold_dbfs) / 20.0)
    nonzero = np.where(np.abs(x) > threshold)[0]
    if nonzero.size == 0:
        return x
    start = int(nonzero[0])
    end = int(nonzero[-1]) + 1
    max_trim = int(float(max_trim_s) * float(sample_rate))
    start = min(start, max_trim)
    end = max(end, x.size - max_trim)
    return x[start:end]


def concat_audio_segments(
    segments: list[np.ndarray],
    pauses_ms: list[int],
    sample_rate: int,
) -> np.ndarray:
    if not segments:
        return np.zeros(0, dtype=np.float32)
    parts: list[np.ndarray] = []
    for i, seg in enumerate(segments):
        parts.append(np.asarray(seg, dtype=np.float32).reshape(-1))
        if i < len(segments) - 1:
            pause_ms = int(pauses_ms[i]) if i < len(pauses_ms) else 0
            if pause_ms > 0:
                n_silence = int(sample_rate * pause_ms / 1000.0)
                if n_silence > 0:
                    parts.append(np.zeros(n_silence, dtype=np.float32))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def encode_pcm_to_wav_data_url(pcm: np.ndarray, sample_rate: int) -> str:
    """Encode mono PCM as a WAV data URL."""
    buf = io.BytesIO()
    audio = np.asarray(pcm, dtype=np.float32).reshape(-1)
    sf.write(buf, audio, int(sample_rate), format="WAV", subtype="PCM_16")
    return f"data:audio/wav;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def count_words(text: str) -> int:
    return len((text or "").split())


async def maybe_rolling_icl_audio_bytes(
    serving: Any,
    request: Any,
    base64_encode: bool = False,
) -> tuple[Any, str] | None:
    """If the request qualifies for rolling ICL, generate audio and return
    ``(audio_data, media_type)``.  Otherwise return ``None`` so the caller
    falls through to the normal one-shot path."""
    if not should_use_rolling_icl(request.input or ""):
        return None

    from vllm_omni.entrypoints.openai.protocol.audio import AudioResponse, CreateAudio

    audio_tensor, sample_rate = await generate_raon_long_tts_rolling_icl(serving, request)
    audio_obj = CreateAudio(
        audio_tensor=audio_tensor,
        sample_rate=sample_rate,
        response_format=request.response_format or "wav",
        speed=request.speed or 1.0,
        stream_format=request.stream_format,
        base64_encode=base64_encode,
    )
    audio_response: AudioResponse = serving.create_audio(audio_obj)
    return audio_response.audio_data, audio_response.media_type
