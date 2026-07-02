# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-TTS AR Model (Stage 0): Text → Speech Tokens.

Based on Llama architecture, generates speech token sequences from input text.
Analogous to Fish Speech Slow AR and Qwen3-TTS Talker models.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature
from transformers.utils.hub import cached_file
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import get_pp_group
from vllm.inputs import MultiModalDataDict
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    ProcessorInputs,
    PromptIndexTargets,
    PromptInsertion,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.models.common.nucleus_ras_sampling import ras_sample_one as _ras_sample_one
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.platforms import current_omni_platform
from vllm_omni.transformers_utils.configs.glm_tts import GLMTTSConfig
from vllm_omni.utils.speaker_cache import get_speaker_cache

from .text_frontend import GLMTTSTextFrontend
from .voice_clone import (
    extract_prompt_feat,
    extract_prompt_speech_token,
    extract_spk_embedding,
    load_voice_clone_frontend,
)

logger = init_logger(__name__)

_GLM_TTS_DEFAULT_REPO_ID = "zai-org/GLM-TTS"
_GLM_TTS_TOKENIZER_SUBDIR = "vq32k-phoneme-tokenizer"
_GLM_TTS_MAX_PROMPT_SPEECH_TOKENS = 1024


def _req_float(param: torch.Tensor | None, req_idx: int, default: float) -> float:
    if param is None or param.numel() == 0:
        return default
    index = min(req_idx, int(param.numel()) - 1)
    return float(param.reshape(-1)[index].item())


def _tensor_param_values(
    param: torch.Tensor | None,
    count: int,
    default: float,
) -> list[float]:
    if param is None or param.numel() == 0:
        return [default] * count
    values = param.reshape(-1).detach().cpu().tolist()
    if not values:
        return [default] * count
    if len(values) < count:
        values.extend([values[-1]] * (count - len(values)))
    return [float(value) for value in values[:count]]


def resolve_glm_tts_tokenizer_path(model_name_or_path: Any) -> str:
    """Resolve vq32k-phoneme-tokenizer directory to a local path.

    For local model dirs, scans for the tokenizer subfolder on disk.
    For HF hub names, downloads the tokenizer subfolder via
    ``snapshot_download`` (same pattern as CosyVoice3's
    ``_ensure_cached_runtime_components`` and
    ``arg_utils._TOKENIZER_SUBFOLDER_MAP``).
    """
    model_path = os.fspath(model_name_or_path)
    if os.path.exists(model_path):
        for candidate in [
            os.path.join(model_path, _GLM_TTS_TOKENIZER_SUBDIR),
            os.path.join(os.path.dirname(model_path), _GLM_TTS_TOKENIZER_SUBDIR),
        ]:
            if os.path.isdir(candidate):
                return candidate
        return model_path

    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(
        model_path,
        allow_patterns=[
            f"{_GLM_TTS_TOKENIZER_SUBDIR}/tokenizer*",
            f"{_GLM_TTS_TOKENIZER_SUBDIR}/tokenization_*",
            f"{_GLM_TTS_TOKENIZER_SUBDIR}/special_tokens*",
            f"{_GLM_TTS_TOKENIZER_SUBDIR}/vocab*",
            f"{_GLM_TTS_TOKENIZER_SUBDIR}/merges*",
            f"{_GLM_TTS_TOKENIZER_SUBDIR}/added_tokens*",
        ],
    )
    candidate = os.path.join(local_dir, _GLM_TTS_TOKENIZER_SUBDIR)
    if os.path.isdir(candidate):
        return candidate
    return model_path


def resolve_glm_tts_model_dir(
    model_name_or_path: Any,
    *,
    tokenizer_path: Any | None = None,
    required_files: Sequence[str] = (),
    optional_files: Sequence[str] = (),
) -> str:
    """Resolve GLM-TTS repo root: local dir → tokenizer parent → cached_file → snapshot."""
    model_path = os.fspath(model_name_or_path)
    if os.path.isdir(model_path):
        return os.path.abspath(model_path)

    def _has_files(root: str) -> bool:
        has_req = all(os.path.isfile(os.path.join(root, *f.split("/"))) for f in required_files)
        has_opt = not optional_files or any(os.path.isfile(os.path.join(root, *f.split("/"))) for f in optional_files)
        return has_req and has_opt

    # Try tokenizer parent directory
    if tokenizer_path:
        tp = os.path.abspath(os.path.normpath(os.fspath(tokenizer_path)))
        root = os.path.dirname(tp) if os.path.basename(tp) == _GLM_TTS_TOKENIZER_SUBDIR else tp
        if os.path.isdir(root) and _has_files(root):
            return root

    # Try cached_file to locate individual files without full snapshot
    for filename in list(required_files) + list(optional_files):
        try:
            resolved = cached_file(model_name_or_path, filename)
            if resolved:
                root = os.path.abspath(resolved)
                for _ in filename.split("/"):
                    root = os.path.dirname(root)
                if _has_files(root):
                    return root
        except Exception:
            pass

    from huggingface_hub import snapshot_download

    return snapshot_download(model_name_or_path)


def _first_glm_tts_value(value: Any) -> Any:
    return value[0] if isinstance(value, list) and value else value


def _glm_tts_int_value(value: Any) -> int | None:
    """Extract a scalar integer from vLLM multimodal/runtime wrappers."""
    value = _first_glm_tts_value(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return int(value.reshape(-1)[0].item())
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _decode_glm_tts_audio_data(value: Any) -> tuple[torch.Tensor | None, int | None]:
    """Extract (waveform_tensor, sample_rate) from multimodal input."""
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if isinstance(value, torch.Tensor):
        return value, None
    if isinstance(value, (tuple, list)) and len(value) == 2 and isinstance(value[1], (int, float)):
        return torch.as_tensor(value[0]), int(value[1])
    if hasattr(value, "shape"):  # numpy array
        return torch.as_tensor(value), None
    return None, None


_glm_tts_tokenizer_cache: dict[str, Any] = {}


def load_glm_tts_tokenizer(
    tokenizer_path: Any,
    *,
    model_name_or_path: Any | None = None,
    trust_remote_code: bool = False,
    **kwargs: Any,
) -> Any:
    """Load GLM-TTS phoneme tokenizer (ChatGLM4Tokenizer, slow-only).

    Expects a resolved local directory (use ``resolve_glm_tts_tokenizer_path``
    to handle HF hub names first).  Results are cached by *tokenizer_path*.
    """
    cache_key = str(tokenizer_path)
    cached = _glm_tts_tokenizer_cache.get(cache_key)
    if cached is not None:
        return cached

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=False,
        trust_remote_code=trust_remote_code,
        **kwargs,
    )
    _glm_tts_tokenizer_cache[cache_key] = tokenizer
    return tokenizer


def get_glm_tts_special_token_ids(tokenizer: Any) -> dict[str, int]:
    """Resolve GLM-TTS special IDs from the phoneme/audio tokenizer."""
    special_tokens = {
        "ats": "<|audio_0|>",
        "ate": "<|audio_32767|>",
        "boa": "<|begin_of_audio|>",
        "eoa": "<|user|>",
        "pad": "<|endoftext|>",
    }

    result: dict[str, int] = {}
    for key, token_str in special_tokens.items():
        token_ids = tokenizer.encode(token_str, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(f"Token '{key}' ({token_str}) should encode to single ID, got: {token_ids}")
        result[key] = int(token_ids[0])
    return result


def resolve_glm_tts_campplus_path(model_dir: str) -> str | None:
    """Resolve ``campplus.onnx`` for GLM-TTS voice cloning."""
    local = os.path.join(model_dir, "frontend", "campplus.onnx")
    if os.path.isfile(local):
        return local

    try:
        resolved = cached_file("FunAudioLLM/CosyVoice-300M", "campplus.onnx")
        if resolved is not None:
            logger.info("Resolved campplus.onnx from FunAudioLLM/CosyVoice-300M: %s", resolved)
            return resolved
    except Exception:
        logger.debug("cached_file could not fetch campplus.onnx", exc_info=True)

    logger.warning(
        "campplus.onnx not found locally or could not be downloaded. "
        "Voice cloning speaker embedding will not be available.",
    )
    return None


def _normalize_glm_tts_processor_text(
    frontend: GLMTTSTextFrontend,
    value: str | None,
    *,
    add_trailing_space: bool = False,
) -> str:
    if value is None:
        return ""
    normalized = frontend.text_normalize(value)
    normalized = (normalized or value).strip()
    if add_trailing_space and normalized:
        normalized = f"{normalized} "
    return normalized


_glm_tts_text_frontend: GLMTTSTextFrontend | None = None


def _get_text_frontend() -> GLMTTSTextFrontend:
    global _glm_tts_text_frontend
    if _glm_tts_text_frontend is None:
        _glm_tts_text_frontend = GLMTTSTextFrontend()
    return _glm_tts_text_frontend


def build_glm_tts_prefill_metadata(
    model_name_or_path: Any,
    text: str,
    prompt_text: str | None,
    *,
    tokenizer_path: Any | None = None,
    trust_remote_code: bool = False,
) -> dict[str, list[int]]:
    """Build request-local GLM-TTS length metadata for AR prefill.

    These scalars are mirrored into ``model_intermediate_buffer`` so the model
    preprocess hook can initialize request state from the metadata alone.
    """
    if tokenizer_path is None:
        tokenizer_path = resolve_glm_tts_tokenizer_path(model_name_or_path)
    tokenizer = load_glm_tts_tokenizer(
        tokenizer_path,
        model_name_or_path=model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    frontend = _get_text_frontend()
    text_len = max(1, len(tokenizer.encode(_normalize_glm_tts_processor_text(frontend, text))))
    prompt_text_len = 0
    if prompt_text:
        prompt_text_len = max(
            1,
            len(
                tokenizer.encode(
                    _normalize_glm_tts_processor_text(
                        frontend,
                        prompt_text,
                        add_trailing_space=True,
                    )
                )
            ),
        )
    return {
        "glm_tts_text_token_len": [text_len],
        "glm_tts_prompt_text_token_len": [prompt_text_len],
        "input_len": [prompt_text_len + text_len + 1],
    }


class GLMTTSMultiModalProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(GLMTTSConfig)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int] | None:
        if mm_counts.get("audio", 0) <= 0:
            return {}
        return {"audio": min(seq_len, _GLM_TTS_MAX_PROMPT_SPEECH_TOKENS)}

    def get_data_parser(self):
        return MultiModalDataParser(
            target_sr=24000,
            expected_hidden_size=self._get_expected_hidden_size(),
        )


class GLMTTSMultiModalProcessor(BaseMultiModalProcessor[GLMTTSMultiModalProcessingInfo]):
    """GLM-TTS voice-clone processor.

    Unlike CosyVoice3, GLM-TTS prompt speech tokens are normal Llama vocab IDs
    (``<|audio_N|>``).  The processor exposes them as multimodal embeddings for
    the AR prompt and also carries WhisperVQ/CampPlus outputs to the AR->DiT
    handoff.
    """

    def _ensure_cached_runtime_components(
        self,
        model_dir: str,
        config: GLMTTSConfig,
        *,
        load_voice_clone: bool = True,
    ) -> None:
        # Skip if already loaded for this model_dir + mode
        cache_key = (model_dir, load_voice_clone)
        if getattr(self, "_loaded_cache_key", None) == cache_key:
            return

        tokenizer_path = getattr(self.info.ctx.model_config, "tokenizer", None)
        required = ("speech_tokenizer/config.json", "speech_tokenizer/model.safetensors") if load_voice_clone else ()
        model_dir = resolve_glm_tts_model_dir(
            model_dir,
            tokenizer_path=tokenizer_path,
            required_files=required,
        )

        # Load text tokenizer + frontend (always needed)
        if getattr(self, "_text_model_dir", None) != model_dir:
            tok_path = tokenizer_path
            if not tok_path or not os.path.isdir(os.fspath(tok_path)):
                tok_path = os.path.join(model_dir, _GLM_TTS_TOKENIZER_SUBDIR)
                if not os.path.isdir(tok_path):
                    tok_path = model_dir
            trust_remote_code = bool(getattr(self.info.ctx.model_config, "trust_remote_code", False))
            self.tokenizer = load_glm_tts_tokenizer(
                tok_path,
                model_name_or_path=self.info.ctx.model_config.model,
                trust_remote_code=trust_remote_code,
            )
            self.special_ids = get_glm_tts_special_token_ids(self.tokenizer)
            self.text_frontend = GLMTTSTextFrontend()
            self._text_model_dir = model_dir

        # Load voice clone components (speech tokenizer + CampPlus)
        if load_voice_clone:
            device = current_omni_platform.get_torch_device()
            self.processor_device = device
            campplus_path = getattr(self, "_campplus_path", None) or resolve_glm_tts_campplus_path(model_dir)
            self._campplus_path = campplus_path
            self.speech_tokenizer, self.campplus_session = load_voice_clone_frontend(
                model_dir,
                device,
                speech_tokenizer_cache=getattr(self, "speech_tokenizer", None),
                campplus_cache=getattr(self, "campplus_session", None),
                campplus_path=campplus_path,
            )

        if not hasattr(self, "_speaker_cache"):
            self._speaker_cache = get_speaker_cache()

        self._loaded_cache_key = cache_key

    def _encode_text(self, text: str) -> torch.Tensor:
        token_ids = self.tokenizer.encode(text)
        return torch.tensor([token_ids], dtype=torch.long)

    def _get_audio(self, mm_data: Mapping[str, object]) -> tuple[torch.Tensor | None, int | None]:
        audio = mm_data.get("audio")
        if audio is None:
            audios = mm_data.get("audios")
            if isinstance(audios, (list, tuple)) and audios:
                # Match CosyVoice3's compatibility behavior and accept the
                # first item from the legacy plural field name.
                audio = audios[0]
            else:
                audio = audios
        wav, sr = _decode_glm_tts_audio_data(audio)
        if wav is not None and wav.ndim > 1:
            wav = wav.float()
            wav = wav.mean(dim=0) if wav.shape[0] <= wav.shape[-1] else wav.mean(dim=-1)
        return wav, int(sr or 24000) if wav is not None else sr

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        config = self.info.ctx.get_hf_config()
        model_dir = self.info.ctx.model_config.model
        wav, sr = self._get_audio(mm_data)
        if wav is None:
            audio_supplied = mm_data.get("audio") is not None or mm_data.get("audios") is not None
            assert not audio_supplied, (
                "GLM-TTS received an audio multimodal payload that could not be decoded. "
                "Runtime voice cloning must provide decodable multi_modal_data['audio']; "
                "the text-only processor path is reserved for profiling/cache."
            )
            self._ensure_cached_runtime_components(model_dir, config, load_voice_clone=False)
            normalized_text = _normalize_glm_tts_processor_text(self.text_frontend, prompt)
            text_ids = self._encode_text(normalized_text)
            dummy_audio_len = min(
                _GLM_TTS_MAX_PROMPT_SPEECH_TOKENS,
                max(1, int(getattr(config, "max_position_embeddings", _GLM_TTS_MAX_PROMPT_SPEECH_TOKENS))),
            )
            boa_tensor = torch.tensor([[int(self.special_ids["boa"])]], dtype=torch.long)
            input_ids = torch.cat([text_ids, boa_tensor], dim=1)
            # vLLM profiling/cache setup may call the processor without an
            # audio item. Do not fabricate a dummy wav here: for GLM-TTS that
            # would exercise the real voice-clone frontend during profiling.
            # This shape-only branch is not a supported runtime inference mode.
            return BatchFeature(
                {
                    "input_ids": input_ids,
                    "prompt_speech_token": torch.zeros((1, dummy_audio_len), dtype=torch.long),
                    "prompt_speech_token_len": [torch.tensor([dummy_audio_len], dtype=torch.long)],
                    "glm_tts_prompt_text_token_len": [torch.tensor([0], dtype=torch.long)],
                    "glm_tts_text_token_len": [torch.tensor([int(text_ids.shape[1])], dtype=torch.long)],
                }
            )

        self._ensure_cached_runtime_components(model_dir, config)
        normalized_text = _normalize_glm_tts_processor_text(self.text_frontend, prompt)
        text_ids = self._encode_text(normalized_text)
        prompt_text = mm_kwargs.get("prompt_text")
        if not isinstance(prompt_text, str) or not prompt_text.strip():
            raise ValueError("GLM-TTS voice cloning requires mm_processor_kwargs['prompt_text'].")

        normalized_prompt_text = _normalize_glm_tts_processor_text(
            self.text_frontend,
            prompt_text,
            add_trailing_space=True,
        )
        prompt_text_ids = self._encode_text(normalized_prompt_text)

        # Speaker cache: skip WhisperVQ + mel + CampPlus on hit
        voice_name = mm_kwargs.get("voice_name")
        cache_key = None
        if voice_name and isinstance(voice_name, str):
            cache_key = self._speaker_cache.make_cache_key(
                voice_name,
                model_type="glm_tts",
                created_at=int(mm_kwargs.get("voice_created_at") or 0),
            )
            cached = self._speaker_cache.get(cache_key)
            if cached is not None:
                boa_tensor = torch.tensor([[int(self.special_ids["boa"])]], dtype=torch.long)
                input_ids = torch.cat([prompt_text_ids, text_ids, boa_tensor], dim=1)
                return BatchFeature(
                    {
                        "input_ids": input_ids,
                        "prompt_speech_token": cached["prompt_speech_token"].clone(),
                        "prompt_speech_token_len": [cached["prompt_speech_token_len"].clone()],
                        "glm_tts_prompt_text_token_len": [
                            torch.tensor([int(prompt_text_ids.shape[1])], dtype=torch.long)
                        ],
                        "prompt_feat": cached["prompt_feat"].clone(),
                        "embedding": cached["embedding"].clone(),
                        "glm_tts_text_token_len": [torch.tensor([int(text_ids.shape[1])], dtype=torch.long)],
                    }
                )

        prompt_speech_token = extract_prompt_speech_token(wav, int(sr or 24000), self.speech_tokenizer)
        if not prompt_speech_token:
            raise RuntimeError("GLM-TTS failed to extract WhisperVQ prompt speech tokens from ref_audio.")

        prompt_speech_token_tensor = torch.tensor([prompt_speech_token], dtype=torch.long)
        boa_tensor = torch.tensor([[int(self.special_ids["boa"])]], dtype=torch.long)
        input_ids = torch.cat([prompt_text_ids, text_ids, boa_tensor], dim=1)
        logger.info(
            "GLM-TTS processor prompt: prompt_text_tokens=%d text_tokens=%d "
            "prompt_speech_tokens=%d input_tokens_before_audio=%d expected_prefill_tokens=%d",
            int(prompt_text_ids.shape[1]),
            int(text_ids.shape[1]),
            len(prompt_speech_token),
            int(input_ids.shape[1]),
            int(input_ids.shape[1]) + len(prompt_speech_token),
        )

        prompt_feat = extract_prompt_feat(wav, int(sr or 24000), self.processor_device)
        if prompt_feat is None:
            raise RuntimeError("GLM-TTS failed to extract prompt mel features from ref_audio.")
        embedding = extract_spk_embedding(wav, int(sr or 24000), self.campplus_session)
        if embedding is None:
            raise RuntimeError("GLM-TTS failed to extract CampPlus speaker embedding from ref_audio.")

        prompt_speech_token_len = torch.tensor([len(prompt_speech_token)], dtype=torch.long)
        prompt_feat_cpu = prompt_feat.detach().to("cpu").unsqueeze(0).contiguous()
        embedding_tensor = torch.tensor([embedding], dtype=torch.float32)

        if cache_key is not None:
            self._speaker_cache.put(
                cache_key,
                {
                    "prompt_speech_token": prompt_speech_token_tensor.detach().cpu(),
                    "prompt_speech_token_len": prompt_speech_token_len.detach().cpu(),
                    "prompt_feat": prompt_feat_cpu.detach().cpu(),
                    "embedding": embedding_tensor.detach().cpu(),
                },
            )

        return BatchFeature(
            {
                "input_ids": input_ids,
                "prompt_speech_token": prompt_speech_token_tensor,
                "prompt_speech_token_len": [prompt_speech_token_len],
                "glm_tts_prompt_text_token_len": [torch.tensor([int(prompt_text_ids.shape[1])], dtype=torch.long)],
                "prompt_feat": prompt_feat_cpu,
                "embedding": embedding_tensor,
                "glm_tts_text_token_len": [torch.tensor([int(text_ids.shape[1])], dtype=torch.long)],
            }
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return {
            key: MultiModalFieldConfig.batched("audio")
            for key in (
                "prompt_speech_token",
                "prompt_speech_token_len",
                "glm_tts_prompt_text_token_len",
                "prompt_feat",
                "embedding",
                "glm_tts_text_token_len",
            )
            if key in hf_inputs
        }

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return False

    def _cached_apply_hf_processor(self, inputs: ProcessorInputs, timing_ctx: Any):
        # GLM-TTS builds the actual AR prompt from both the request text and
        # mm_processor_kwargs["prompt_text"]. The base cache path separates text
        # processing from audio processing, which drops the reference transcript
        # and BOA token from the final prompt.
        return self._apply_hf_processor(inputs, timing_ctx)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        def insertion_end(item_idx: int) -> list[int]:
            audio_items = out_mm_kwargs["audio"]
            item = audio_items[item_idx] if item_idx < len(audio_items) else audio_items[0]
            token_len = item["prompt_speech_token_len"].data[0].item()
            return [1] * int(token_len)

        return [
            PromptInsertion(
                modality="audio",
                target=PromptIndexTargets.end(),
                insertion=insertion_end,
            )
        ]


class GLMTTSDummyInputsBuilder(BaseDummyInputsBuilder[GLMTTSMultiModalProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return "This is a test of the GLM-TTS voice cloning system."

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        return {}

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> ProcessorInputs:
        inputs = super().get_dummy_processor_inputs(seq_len, mm_counts, mm_options)
        inputs.hf_processor_mm_kwargs = {"prompt_text": "This is the reference voice."}
        return inputs


@MULTIMODAL_REGISTRY.register_processor(
    GLMTTSMultiModalProcessor,
    info=GLMTTSMultiModalProcessingInfo,
    dummy_inputs=GLMTTSDummyInputsBuilder,
)
class GLMTTSForConditionalGeneration(nn.Module, SupportsMultiModal):
    """vLLM model for GLM-TTS.

    Handles both stages via model_stage branching:
      - ``glm_tts`` (Stage 0): Text → Speech tokens (LLM AR, Llama backbone).
      - ``glm_tts_dit`` (Stage 1): Speech tokens → Audio (DiT flow-matching + vocoder).

    Attributes:
        have_multimodal_outputs: Signals scheduler to collect multimodal outputs.
        has_preprocess: Model has preprocess hook for input preparation (stage 0 only).
        has_postprocess: Model has postprocess hook for hidden state caching (stage 0 only).
    """

    supports_multimodal_raw_input_only = True
    supports_multimodal = True
    requires_raw_input_tokens = True
    prefer_model_sampler = True
    _sampling_eps = 1e-5
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "llama_embedding.": "model.embed_tokens.",
            "llama.model.": "model.",
            "llama.": "model.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model
        self.model_dir = self.model_path
        self.model_stage = getattr(vllm_config.model_config, "model_stage", "glm_tts")

        # Load configuration
        config: GLMTTSConfig = vllm_config.model_config.hf_config  # type: ignore[assignment]
        self.config = config

        # ---- Stage 1 (DiT): delegate to GLMTTSDiTForGeneration ----
        if self.model_stage == "glm_tts_dit":
            from .glm_tts_dit_wrapper import GLMTTSDiTForGeneration

            self._dit_gen = GLMTTSDiTForGeneration(vllm_config=vllm_config, prefix=prefix)
            # Expose the DiT as self.model for weight loading routing
            self.model = self._dit_gen
            self.have_multimodal_outputs = True
            self.enable_update_additional_information = True
            # DiT stage does not use preprocess/postprocess/sample
            self.has_preprocess = False
            self.has_postprocess = False
            # DiT loads all weights internally from flow/flow.pt etc.
            # Point DefaultModelLoader to flow/flow.pt (a .pt file) so it can
            # find and load it without triggering the vLLM#39699 bug where
            # subdirectory safetensors patterns cause pt_weights_iterator
            # to be used on .safetensors files.  load_weights() ignores the
            # yielded weights and loads flow/flow.pt itself.
            self.allow_patterns_overrides = ["flow/flow.pt"]
            self.fall_back_to_pt_during_load = False
            return
        sample_method = str(getattr(config, "sample_method", "ras")).lower()
        if sample_method != "ras":
            raise ValueError(f"Unsupported GLM-TTS sample_method={sample_method!r}; expected 'ras'.")

        # Resolve repo root to local path (CosyVoice3 pattern).
        # Model weights are in llm/ subdirectory; tokenizer and other resources
        # are siblings of llm/ under the repo root.
        self.model_dir = resolve_glm_tts_model_dir(
            self.model_dir,
            tokenizer_path=getattr(vllm_config.model_config, "tokenizer", None),
        )

        # Stage 0 weights live under llm/model-*.safetensors and are loaded
        # explicitly by _iter_llm_safetensors() during load_weights().
        #
        # We still point DefaultModelLoader at flow/flow.pt as a bootstrap
        # sentinel because the current upstream loader path does not reliably
        # target subdirectory safetensors patterns here without hitting the
        # vLLM#39699 iterator bug. Keep the real stage-0 source of truth in
        # _iter_llm_safetensors(); this override is only to get through the
        # generic loader bootstrap step.
        self.allow_patterns_overrides = ["flow/flow.pt"]
        self.fall_back_to_pt_during_load = False

        # Load tokenizer for special token ID resolution.
        # Prefer the tokenizer path auto-detected by arg_utils
        # (_TOKENIZER_SUBFOLDER_MAP).  Fall back to vq32k-phoneme-tokenizer/
        # under model_dir.
        tokenizer_path = getattr(vllm_config.model_config, "tokenizer", None)
        if not (tokenizer_path and os.path.isdir(tokenizer_path)):
            tokenizer_path = os.path.join(self.model_dir, _GLM_TTS_TOKENIZER_SUBDIR)
            if not os.path.exists(tokenizer_path):
                tokenizer_path = self.model_dir
        trust_remote_code = bool(getattr(vllm_config.model_config, "trust_remote_code", False))
        self._tokenizer = load_glm_tts_tokenizer(
            tokenizer_path,
            model_name_or_path=self.model_path,
            trust_remote_code=trust_remote_code,
        )
        special_ids = get_glm_tts_special_token_ids(self._tokenizer)
        self._ats = special_ids["ats"]
        self._ate = special_ids["ate"]
        self._boa = special_ids["boa"]
        self._eoa = special_ids["eoa"]
        self._pad = special_ids["pad"]
        self._bos = self._tokenizer.bos_token_id or self._pad

        logger.info(
            "GLM-TTS token IDs: ATS=%d, ATE=%d, BOA=%d, EOA=%d, PAD=%d, vocab_size=%d",
            self._ats,
            self._ate,
            self._boa,
            self._eoa,
            self._pad,
            config.vocab_size,
        )

        # Validate special token sanity with runtime exceptions (asserts can be
        # stripped under python -O).
        if self._ate - self._ats != 32767:
            raise ValueError(f"Audio token range should be 32768, got {self._ate - self._ats + 1}")
        if self._boa >= self._ats:
            raise ValueError(f"BOA={self._boa} should be < ATS={self._ats} (BOA is text token)")

        # Validate vocab_size covers all special tokens
        max_token = max(self._ats, self._ate, self._boa, self._eoa, self._pad)
        if max_token >= config.vocab_size:
            raise ValueError(
                f"vocab_size ({config.vocab_size}) must be > max token ID ({max_token}). "
                f"Check model's config.json has correct vocab_size."
            )

        # Update config with dynamic token IDs so vLLM uses correct eos_token
        # This enables proper stop detection when EOA is sampled
        config.eos_token_id = self._eoa
        config.eoa_token_id = self._eoa
        config.audio_token_start = self._ats
        config.audio_token_end = self._ate
        config.boa_token_id = self._boa
        config.pad_token_id = self._pad
        config.bos_token_id = self._bos

        # Model flags for AR scheduler
        self.have_multimodal_outputs = True
        self.has_preprocess = True
        self.has_postprocess = True
        # GLM-TTS async chunk transfer consumes speech_tokens plus voice-clone
        self.gpu_resident_buffer_keys: set[tuple[str, str]] = {("last_hidden", "last")}

        self.model = LlamaModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))

        # LM head for speech token prediction
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

        # Runtime validation: dynamic EOA must match the hardcoded pipeline
        # stop_token_ids constant (59253).  A mismatch means the upstream
        # tokenizer vocabulary has changed and pipeline.py needs updating.
        _PIPELINE_EOA = 59253
        if self._eoa != _PIPELINE_EOA:
            logger.warning(
                "GLM-TTS EOA token mismatch: tokenizer resolved %d but "
                "pipeline.py hardcodes stop_token_ids=[%d]. Update "
                "pipeline.py to match the current checkpoint.",
                self._eoa,
                _PIPELINE_EOA,
            )

        # RAS sampling config — stored as attributes for sample()
        self._ras_win_size = int(getattr(config, "ras_win_size", 10))
        self._ras_tau_r = float(getattr(config, "ras_tau_r", 0.1))
        self._ras_top_p = float(getattr(config, "ras_top_p", 0.8))
        self._ras_top_k = int(getattr(config, "ras_top_k", 25))

    def _model_dtype(self) -> torch.dtype:
        """Return the active parameter dtype for locally-created embeddings."""
        try:
            return next(self.model.parameters()).dtype
        except StopIteration:
            dtype = getattr(self.vllm_config.model_config, "dtype", None)
            if isinstance(dtype, torch.dtype):
                return dtype
            if isinstance(dtype, str):
                _aliases = {"bf16": "bfloat16", "fp16": "float16", "half": "float16", "fp32": "float32"}
                resolved = getattr(torch, _aliases.get(dtype.lower(), dtype.lower()), None)
                if resolved is not None:
                    return resolved
            return torch.get_default_dtype()

    def _resolve_prefill_target_text_len(
        self,
        span_len: int,
        info_dict: Mapping[str, Any],
    ) -> int | None:
        """Resolve the target-text token count before the first AR sample.

        The request builder mirrors the processor's length metadata into
        model_intermediate_buffer before the first prefill.  Prefer lengths
        inferred from that metadata, then fall back to the explicit target
        scalar.  Never silently use a prompt-text length as the target length.
        """

        provided_text_len = _glm_tts_int_value(info_dict.get("glm_tts_text_token_len"))
        prompt_text_len = _glm_tts_int_value(info_dict.get("glm_tts_prompt_text_token_len"))
        prompt_speech_len = _glm_tts_int_value(info_dict.get("prompt_speech_token_len"))
        input_len = _glm_tts_int_value(info_dict.get("input_len"))

        inferred_text_len: int | None = None
        if input_len is not None and prompt_text_len is not None:
            inferred_text_len = input_len - prompt_text_len - 1  # strip prompt text and BOA.
        elif prompt_text_len is not None and prompt_speech_len is not None:
            inferred_text_len = span_len - prompt_text_len - prompt_speech_len - 1

        if inferred_text_len is not None and inferred_text_len > 0:
            if provided_text_len is not None and provided_text_len != inferred_text_len:
                logger.warning(
                    "GLM-TTS target text token length mismatch: processor=%d inferred=%d; using inferred length.",
                    provided_text_len,
                    inferred_text_len,
                )
            return inferred_text_len

        return provided_text_len if provided_text_len is not None and provided_text_len > 0 else None

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Any | None = None,
        is_multimodal: Any | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if self.model_stage == "glm_tts_dit":
            return self._dit_gen.embed_input_ids(input_ids, **kwargs)
        embed_tokens = self.model.embed_tokens(input_ids)
        if multimodal_embeddings is None or is_multimodal is None:
            return embed_tokens

        mask = torch.as_tensor(is_multimodal, device=embed_tokens.device, dtype=torch.bool).reshape(-1)
        if not bool(mask.any()):
            return embed_tokens

        if isinstance(multimodal_embeddings, (list, tuple)):
            tensors = [
                torch.as_tensor(item, device=embed_tokens.device, dtype=embed_tokens.dtype)
                for item in multimodal_embeddings
                if item is not None
            ]
            if not tensors:
                return embed_tokens
            mm_embeds = torch.cat([item.reshape(-1, item.shape[-1]) for item in tensors], dim=0)
        else:
            mm_embeds = multimodal_embeddings
        if mm_embeds is None:
            return embed_tokens
        mm_embeds = torch.as_tensor(mm_embeds, device=embed_tokens.device, dtype=embed_tokens.dtype)
        if mm_embeds.ndim == 3 and int(mm_embeds.shape[0]) == 1:
            mm_embeds = mm_embeds.squeeze(0)
        if mm_embeds.ndim != 2:
            shape = tuple(mm_embeds.shape)
            raise ValueError(f"GLM-TTS multimodal embeddings should be 2D, got shape={shape}")

        flat_tokens = embed_tokens.reshape(-1, embed_tokens.shape[-1])
        if int(flat_tokens.shape[0]) != int(mask.numel()):
            raise ValueError(
                "GLM-TTS multimodal mask/token length mismatch: "
                f"mask={int(mask.numel())}, tokens={int(flat_tokens.shape[0])}"
            )
        mm_len = int(mask.sum().item())
        if int(mm_embeds.shape[0]) < mm_len:
            raise ValueError(
                "GLM-TTS multimodal embedding length mismatch: "
                f"embeddings={int(mm_embeds.shape[0])}, placeholders={mm_len}"
            )

        flat_tokens = flat_tokens.clone()
        flat_tokens[mask] = mm_embeds[:mm_len]
        return flat_tokens.reshape_as(embed_tokens)

    def embed_multimodal(self, **kwargs: Any) -> list[torch.Tensor] | None:
        if self.model_stage != "glm_tts":
            return None
        prompt_speech_token = kwargs.get("prompt_speech_token")
        if prompt_speech_token is None:
            return None

        def embed_one(value: Any) -> torch.Tensor | None:
            if value is None:
                return None
            speech_token = torch.as_tensor(value, device=next(self.model.parameters()).device)
            if speech_token.numel() == 0:
                return None
            speech_token = speech_token.to(dtype=torch.long)
            if speech_token.ndim == 0:
                speech_token = speech_token.reshape(1, 1)
            elif speech_token.ndim == 1:
                speech_token = speech_token.unsqueeze(0)
            elif speech_token.ndim > 2:
                speech_token = speech_token.reshape(int(speech_token.shape[0]), -1)
            if int(speech_token.min().item()) >= self._ats:
                speech_ids = speech_token
            else:
                speech_ids = speech_token + self._ats
            speech_embeds = self.model.embed_tokens(speech_ids)
            return speech_embeds.reshape(-1, speech_embeds.shape[-1])

        def is_flat_token_sequence(value: list[Any] | tuple[Any, ...]) -> bool:
            if not value:
                return False
            try:
                return all(torch.as_tensor(item).ndim == 0 for item in value)
            except (TypeError, ValueError):
                return False

        if isinstance(prompt_speech_token, (list, tuple)):
            if is_flat_token_sequence(prompt_speech_token):
                one_embed = embed_one(prompt_speech_token)
                return [one_embed] if one_embed is not None else None
            embeds = [embed_one(item) for item in prompt_speech_token]
            embeds = [item for item in embeds if item is not None]
            if not embeds:
                return None
            return embeds

        one_embed = embed_one(prompt_speech_token)
        return [one_embed] if one_embed is not None else None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors | OmniOutput:
        if self.model_stage == "glm_tts_dit":
            return self._dit_gen.forward(input_ids, positions, **kwargs)
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(
        self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None
    ) -> torch.Tensor | None:
        if self.model_stage == "glm_tts_dit":
            return self._dit_gen.compute_logits(hidden_states, sampling_metadata)
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            return None
        logits = self.logits_processor(self.lm_head, hidden_states)
        if logits is None:
            return None

        return logits

    def _glm_tts_ras_enabled(self, sampling_metadata: SamplingMetadata) -> bool:
        if self.model_stage != "glm_tts":
            return False
        if sampling_metadata.max_num_logprobs is not None:
            return False
        if sampling_metadata.temperature is None:
            return False
        if bool(sampling_metadata.bad_words_token_ids):
            return False
        if torch.any(sampling_metadata.frequency_penalties != 0):
            return False
        if torch.any(sampling_metadata.presence_penalties != 0):
            return False
        return True

    def _apply_allowed_mask(self, logits: torch.Tensor) -> torch.Tensor:
        """Restrict logits to valid audio tokens [ATS..ATE] ∪ {EOA}."""
        allowed_mask = getattr(self, "_ar_allowed_mask", None)
        if allowed_mask is None or allowed_mask.device != logits.device or allowed_mask.shape[0] != logits.shape[-1]:
            allowed_mask = torch.full((logits.shape[-1],), float("-inf"), device=logits.device)
            allowed_mask[self._ats : self._ate + 1] = 0.0
            allowed_mask[self._eoa] = 0.0
            self._ar_allowed_mask = allowed_mask
        return logits + allowed_mask

    def _select_allowed_audio_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Return compact logits for [audio tokens] plus EOA.

        The RAS path samples only from GLM-TTS audio tokens and EOA.  Keeping a
        compact view avoids top-k/logsumexp over the full LLM vocabulary on
        every AR step; this is equivalent to sampling after ``_apply_allowed_mask``.
        """
        audio_logits = logits[:, self._ats : self._ate + 1]
        if self._ats <= self._eoa <= self._ate:
            return audio_logits
        return torch.cat((audio_logits, logits[:, self._eoa : self._eoa + 1]), dim=-1)

    def _allowed_local_to_token_id(self, local_id: int) -> int:
        audio_vocab = self._ate - self._ats + 1
        return self._ats + local_id if local_id < audio_vocab else self._eoa

    def _recent_allowed_local_ids(self, decoded_tokens: Sequence[int], win_size: int) -> list[int]:
        if win_size <= 0 or not decoded_tokens:
            return []
        audio_vocab = self._ate - self._ats + 1
        recent: list[int] = []
        for token in decoded_tokens[-win_size:]:
            token_id = int(token)
            if self._ats <= token_id <= self._ate:
                recent.append(token_id - self._ats)
            elif token_id == self._eoa:
                recent.append(audio_vocab)
            else:
                recent.append(-1)
        return recent

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        """RAS sampler following CosyVoice3 pattern.

        Uses vLLM Sampler for logits processing (logit_bias_state handles
        min_tokens/max_tokens/stop_token_ids).  When RAS is enabled, applies
        per-request nucleus+repetition-aware sampling; otherwise falls back
        to standard vLLM sampling.
        """
        if logits is None or logits.numel() == 0:
            return None
        if self.model_stage != "glm_tts":
            return None

        sampler = getattr(self, "_ar_sampler", None)
        if sampler is None:
            sampler = Sampler()
            self._ar_sampler = sampler

        if not self._glm_tts_ras_enabled(sampling_metadata):
            logits = self._apply_allowed_mask(logits)
            return sampler(logits=logits, sampling_metadata=sampling_metadata)

        logits = logits.to(torch.float32)
        sampling_for_processors = replace(sampling_metadata, no_penalties=True)
        logits = sampler.apply_logits_processors(logits, sampling_for_processors, predict_bonus_token=False)
        allowed_logits = self._select_allowed_audio_logits(logits)

        default_top_p = self._ras_top_p
        default_top_k = self._ras_top_k
        ws = self._ras_win_size
        tr = self._ras_tau_r

        num_reqs = int(allowed_logits.shape[0])
        t_data = sampling_metadata.temperature
        tp_data = sampling_metadata.top_p
        tk_data = sampling_metadata.top_k
        cache_key = (
            t_data.data_ptr() if t_data is not None else 0,
            tp_data.data_ptr() if tp_data is not None else 0,
            tk_data.data_ptr() if tk_data is not None else 0,
            num_reqs,
        )
        _sp_cache = getattr(self, "_sampling_param_cache", None)
        if _sp_cache is not None and _sp_cache[0] == cache_key:
            temperatures, top_ps, top_ks = _sp_cache[1]
        else:
            temperatures = _tensor_param_values(t_data, num_reqs, 1.0)
            top_ps = _tensor_param_values(tp_data, num_reqs, default_top_p)
            top_ks = _tensor_param_values(tk_data, num_reqs, default_top_k)
            self._sampling_param_cache = (cache_key, (temperatures, top_ps, top_ks))

        sampled_ids: list[int] = []
        for req_idx in range(num_reqs):
            row_logits = allowed_logits[req_idx]
            temperature = temperatures[req_idx]
            if temperature < self._sampling_eps:
                local_id = int(torch.argmax(row_logits).item())
                sampled_ids.append(self._allowed_local_to_token_id(local_id))
                continue

            top_p = top_ps[req_idx]
            top_k = int(top_ks[req_idx])
            generator = sampling_metadata.generators.get(req_idx)
            weighted_scores = row_logits / max(temperature, self._sampling_eps)
            decoded_tokens = (
                sampling_metadata.output_token_ids[req_idx] if req_idx < len(sampling_metadata.output_token_ids) else []
            )
            local_id = _ras_sample_one(
                weighted_scores,
                self._recent_allowed_local_ids(decoded_tokens, ws),
                top_p=top_p,
                top_k=top_k,
                win_size=ws,
                tau_r=tr,
                generator=generator,
            )
            sampled_ids.append(self._allowed_local_to_token_id(local_id))

        sampled = torch.tensor(sampled_ids, device=logits.device, dtype=torch.int32)
        return SamplerOutput(sampled_token_ids=sampled.unsqueeze(-1), logprobs_tensors=None)

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        """Package hidden states, speech tokens, and voice clone data into OmniOutput.

        Streaming contract: **delta**.  Each decode step emits exactly one
        speech token (or a prefill placeholder).  The engine's output
        processor concatenates per-step deltas into the final tensor.
        """
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        hidden = model_outputs
        info_dicts = kwargs.get("model_intermediate_buffer") or []
        if isinstance(info_dicts, dict):
            info_dicts = [info_dicts]

        speech_tokens_list: list[torch.Tensor] = []
        multimodal_extras: dict[str, Any] = {}
        mm_conditioning_keys = (
            "prompt_speech_token",
            "prompt_speech_token_len",
            "glm_tts_prompt_text_token_len",
            "prompt_feat",
            "embedding",
            "glm_tts_text_token_len",
        )
        info_for_flags = next((info for info in info_dicts if isinstance(info, dict)), None)
        if info_for_flags is None or not info_for_flags.get("_voice_clone_emitted"):
            copied_from_kwargs = False
            for key in mm_conditioning_keys:
                val = kwargs.get(key)
                if val is not None:
                    multimodal_extras[key] = val
                    copied_from_kwargs = True
            if copied_from_kwargs and info_for_flags is not None:
                info_for_flags["_voice_clone_emitted"] = True

        for info in info_dicts:
            if not isinstance(info, dict):
                continue
            tokens = info.get("speech_tokens")
            if isinstance(tokens, torch.Tensor) and tokens.numel() > 0:
                speech_tokens_list.append(tokens)
            # Propagate voice cloning features from preprocess info_update.
            # IMPORTANT: Only emit once — the output processor accumulates
            # every decode step and concatenates tensors, which would corrupt
            # constant data (e.g. embedding [192] → [N*192]).
            if not info.get("_voice_clone_emitted"):
                for key in mm_conditioning_keys:
                    val = info.get(key)
                    if val is not None and key not in multimodal_extras:
                        multimodal_extras[key] = val
                if any(info.get(k) is not None for k in mm_conditioning_keys):
                    info["_voice_clone_emitted"] = True

        if not speech_tokens_list:
            return OmniOutput(text_hidden_states=hidden, multimodal_outputs=multimodal_extras)

        speech_tokens = torch.cat(speech_tokens_list, dim=0)
        span_len = int(speech_tokens.shape[0])
        hidden = hidden[:span_len]
        multimodal_extras["speech_tokens"] = speech_tokens
        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs=multimodal_extras,
        )

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Prepare inputs for GLM-TTS AR model.

        GLM-TTS only supports the multimodal processor path:
        text prompt + ``multi_modal_data["audio"]`` + ``mm_processor_kwargs["prompt_text"]``.
        Legacy placeholder prompts via ``additional_information`` are rejected.
        """
        if info_dict.get("additional_information") is not None:
            raise ValueError(
                "GLM-TTS no longer accepts legacy additional_information prompts; "
                "use prompt + multi_modal_data['audio'] + mm_processor_kwargs['prompt_text']."
            )

        span_len = int(input_ids.shape[0])
        logger.debug("preprocess: span_len=%d, input_ids.shape=%s", span_len, input_ids.shape)
        if span_len <= 0:
            return input_ids, input_embeds if input_embeds is not None else self.embed_input_ids(input_ids), {}

        device = input_ids.device

        if isinstance(info_dict.get("text"), list) and info_dict["text"]:
            raise ValueError(
                "GLM-TTS no longer accepts legacy text/additional_information prefill payloads; "
                "use the multimodal processor path."
            )

        mm_prefill_done = bool(info_dict.get("glm_tts_mm_prefill_done"))
        sampled_token: int | None = None
        if input_embeds is not None and span_len == 1:
            sampled_token = int(input_ids[0].item())
        one_token_prefill_tail = (
            input_embeds is not None
            and span_len == 1
            and mm_prefill_done
            and sampled_token is not None
            and sampled_token != self._eoa
            and not (self._ats <= sampled_token <= self._ate)
        )
        is_prefill_span = span_len > 1 or not mm_prefill_done or one_token_prefill_tail
        if input_embeds is None and is_prefill_span:
            raise ValueError("Missing GLM-TTS multimodal input embeddings.")
        if input_embeds is not None and is_prefill_span:
            input_ids_out = input_ids.clone()
            input_ids_out[:] = self._pad
            info_update: dict[str, Any] = {
                "glm_tts_mm_prefill_done": True,
                "speech_tokens": torch.full((span_len, 1), -1, device=device, dtype=torch.long),
            }
            # One-token prefill tails re-enter the prefill branch but
            # text_token_len was already resolved during the initial prefill.
            if not one_token_prefill_tail:
                text_token_len = self._resolve_prefill_target_text_len(span_len, info_dict)
                if text_token_len is not None:
                    info_update["glm_tts_text_token_len"] = torch.tensor(
                        [text_token_len], device=device, dtype=torch.long
                    )
                prompt_text_len = info_dict.get("glm_tts_prompt_text_token_len")
                if prompt_text_len is not None:
                    info_update["glm_tts_prompt_text_token_len"] = prompt_text_len
                prompt_speech_len = info_dict.get("prompt_speech_token_len")
                if prompt_speech_len is None:
                    input_len = _glm_tts_int_value(info_dict.get("input_len"))
                    if input_len is not None:
                        prompt_speech_len = torch.tensor(
                            [max(0, span_len - input_len)], device=device, dtype=torch.long
                        )
                if prompt_speech_len is not None:
                    info_update["prompt_speech_token_len"] = prompt_speech_len
            return input_ids_out, input_embeds.to(dtype=self._model_dtype()), info_update

        # Decode: span_len == 1
        # Standard autoregressive decode - use input_ids directly
        if input_embeds is not None and int(input_embeds.shape[0]) == 1:
            inputs_embeds_out = input_embeds.reshape(1, -1).to(dtype=self._model_dtype())
        else:
            inputs_embeds_out = self.embed_input_ids(input_ids.reshape(1, 1).to(torch.long))
            inputs_embeds_out = inputs_embeds_out.reshape(1, -1).to(dtype=self._model_dtype())

        # Convert sampled token to speech token (relative to ATS)
        # -1 = invalid/EOA, valid range = [0, ATE-ATS]
        if sampled_token is None:
            sampled_token = int(input_ids[0].item())
        if self._ats <= sampled_token <= self._ate:
            speech_token = sampled_token - self._ats
        else:
            # EOA or other non-audio token → mark as invalid (-1)
            speech_token = -1
            logger.debug("GLM-TTS decode: non-audio token %d (EOA=%d)", sampled_token, self._eoa)
        speech_tokens = torch.tensor([[speech_token]], device=device, dtype=torch.long)

        info_update = {"speech_tokens": speech_tokens}
        return input_ids, inputs_embeds_out, info_update

    def postprocess(self, hidden_states: torch.Tensor, **kwargs: Any) -> dict[str, Any]:
        """Cache last hidden state for next decode step."""
        if hidden_states.numel() == 0:
            return {}
        last = hidden_states[-1, :].detach()
        update: dict[str, Any] = {"last_hidden": {"last": last}}

        multimodal_outputs = kwargs.get("multimodal_outputs")
        if isinstance(multimodal_outputs, dict):
            copied_conditioning = False
            for key in ("prompt_speech_token", "prompt_speech_token_len", "prompt_feat", "embedding"):
                val = multimodal_outputs.get(key)
                if val is not None:
                    update[key] = _first_glm_tts_value(val)
                    copied_conditioning = True

            prompt_text_len = multimodal_outputs.get("glm_tts_prompt_text_token_len")
            if prompt_text_len is not None:
                update["glm_tts_prompt_text_token_len"] = _first_glm_tts_value(prompt_text_len)

            text_token_len = _glm_tts_int_value(multimodal_outputs.get("glm_tts_text_token_len"))
            if text_token_len is not None:
                update["glm_tts_text_token_len"] = torch.tensor([int(text_token_len)], dtype=torch.long)

            if copied_conditioning:
                update["_voice_clone_emitted"] = True

        return update

    def _iter_llm_safetensors(self) -> Iterable[tuple[str, torch.Tensor]]:
        """Yield (name, tensor) pairs from llm/model-*.safetensors."""
        import glob as glob_module

        from safetensors.torch import load_file

        llm_dir = os.path.join(self.model_dir, "llm")
        if os.path.isdir(llm_dir):
            sf_files = sorted(glob_module.glob(os.path.join(llm_dir, "model-*.safetensors")))
            if sf_files:
                for sf_path in sf_files:
                    yield from load_file(sf_path, device="cpu").items()
                return

        from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

        model_root = download_weights_from_hf_specific(
            self.model_path,
            self.vllm_config.load_config.download_dir,
            allow_patterns=["llm/model-*.safetensors"],
        )
        llm_dir = os.path.join(model_root, "llm")
        sf_files = sorted(glob_module.glob(os.path.join(llm_dir, "model-*.safetensors")))
        if not sf_files:
            raise RuntimeError(f"No LLM safetensors found under {model_root}. Expected llm/model-*.safetensors.")
        for sf_path in sf_files:
            yield from load_file(sf_path, device="cpu").items()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from checkpoint.

        Stage 0 (glm_tts): HuggingFace Llama-format checkpoint from llm/ subdir.
        Stage 1 (glm_tts_dit): DiT flow.pt + vocoder.
        """
        if self.model_stage == "glm_tts_dit":
            return self._dit_gen.load_weights(weights)

        def _glm_tts_weights() -> Iterable[tuple[str, torch.Tensor]]:
            for name, loaded_weight in self._iter_llm_safetensors():
                if "rotary_emb.inv_freq" in name:
                    continue
                if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                    continue
                yield name, loaded_weight

        loader = AutoWeightsLoader(self)
        loaded_params = loader.load_weights(_glm_tts_weights(), mapper=self.hf_to_vllm_mapper)

        params_dict = dict(self.named_parameters())
        # Handle tie_word_embeddings: copy embed_tokens to lm_head if not loaded.
        lm_head_key = "lm_head.weight"
        embed_key = "model.embed_tokens.weight"
        if lm_head_key not in loaded_params and embed_key in loaded_params:
            if lm_head_key in params_dict and embed_key in params_dict:
                lm_head_param = params_dict[lm_head_key]
                embed_param = params_dict[embed_key]
                weight_loader = getattr(lm_head_param, "weight_loader", default_weight_loader)
                weight_loader(lm_head_param, embed_param.data)
                loaded_params.add(lm_head_key)
                logger.info("Tied lm_head.weight to embed_tokens.weight")

        logger.info("Loaded %d weights for GLMTTSForConditionalGeneration", len(loaded_params))
        return loaded_params
