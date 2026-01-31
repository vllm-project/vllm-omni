# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import base64
import io
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoProcessor
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .configuration_qwen3_tts import Qwen3TTSConfig
from .modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
from .processing_qwen3_tts import Qwen3TTSProcessor

logger = init_logger(__name__)

AudioLike = (
    str  # wav path, URL, base64
    | np.ndarray  # waveform (requires sr)
    | tuple[np.ndarray, int]  # (waveform, sr)
)

MaybeList = Any | list[Any]


@dataclass
class VoiceClonePromptItem:
    """
    Container for one sample's voice-clone prompt information that can be fed to the model.

    Fields are aligned with `Qwen3TTSForConditionalGeneration.generate(..., voice_clone_prompt=...)`.
    """

    ref_code: torch.Tensor | None  # (T, Q) or (T,) depending on tokenizer 25Hz/12Hz
    ref_spk_embedding: torch.Tensor  # (D,)
    x_vector_only_mode: bool
    icl_mode: bool
    ref_text: str | None = None


class Qwen3TTSModelForGeneration(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        model_path = vllm_config.model_config.model

        # Check if flash-attn is installed
        try:
            import flash_attn  # noqa: F401

            attn_kwargs = {"attn_implementation": "flash_attention_2"}
        except ImportError:
            logger.warning("Flash-Attn is not installed. Using default PyTorch attention implementation.")
            attn_kwargs = {}

        self.model = Qwen3TTSModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            **attn_kwargs,
        )
        self.task_type = model_path.split("-")[-1].strip("/")
        # Mark that this model produces multimodal outputs
        self.have_multimodal_outputs = True

        # Store vllm_config for potential future use
        self.vllm_config = vllm_config

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        """
        Forward pass for TTS generation model.

        Args:
            input_ids: Input token IDs (required for TTS generation)
            positions: Position IDs (not used for TTS, but required by runner)
            intermediate_tensors: Intermediate tensors for pipeline parallelism (not used)
            inputs_embeds: Input embeddings (not used for TTS, but required by runner)
            **kwargs: Additional arguments including task_type, sampling_metadata, etc.

        Returns:
            OmniOutput: Contains multimodal outputs with audio tensors
        """

        # Extract additional parameters from kwargs that the generation methods expect

        runtime_additional_information = kwargs.get("runtime_additional_information", [{}])
        if isinstance(runtime_additional_information, list) and len(runtime_additional_information) > 0:
            runtime_additional_information = runtime_additional_information[0]
        text = runtime_additional_information.pop("text", [""])[0]
        # Extract task_type from kwargs, default to "instruct"
        task_type = runtime_additional_information.pop("task_type", [self.task_type])[0]
        speaker = runtime_additional_information.pop("speaker", ["uncle_fu"])[0]
        language = runtime_additional_information.pop("language", ["Auto"])[0]
        instruct = runtime_additional_information.pop("instruct", [""])[0]
        for key, value in runtime_additional_information.items():
            if isinstance(value, list) and len(value) > 0:
                runtime_additional_information[key] = value[0]

        # During profile/warmup runs, text is empty and no real inputs exist.
        # Cap generation steps so the full pipeline executes (preserving
        # KV-cache profiling behaviour) but exits quickly even if the model
        # cannot converge from degenerate dummy inputs.
        if not text:
            logger.info("Profile run detected (empty text). Capping max_new_tokens to 2.")
            runtime_additional_information["max_new_tokens"] = 2

        # Call the appropriate generation method based on task_type
        if task_type == "CustomVoice":
            result = self.model.generate_custom_voice(
                text, speaker=speaker, language=language, instruct=instruct, **runtime_additional_information
            )
        elif task_type == "VoiceDesign":
            result = self.model.generate_voice_design(
                text, instruct=instruct, language=language, **runtime_additional_information
            )
        elif task_type == "Base":
            result = self.model.generate_voice_clone(text, language=language, **runtime_additional_information)
        else:
            raise ValueError(f"Invalid task type: {task_type}")

        # Convert result to OmniOutput format
        return self.make_omni_output(result, **kwargs)

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput | tuple, **kwargs: Any) -> OmniOutput:
        """
        Make an OmniOutput object from model outputs.
        Args:
            model_outputs: Model outputs (either OmniOutput, tuple of (audio_tensors, sr), or tensor)
        """
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        # Handle tuple format: (audio_tensors, sample_rate)
        if isinstance(model_outputs, tuple) and len(model_outputs) == 2:
            audio_tensors, sr = model_outputs
            # audio_tensors is a list of numpy arrays, convert first one to tensor if needed
            if isinstance(audio_tensors, list) and len(audio_tensors) > 0:
                # Convert numpy array to tensor if needed
                audio_tensor = audio_tensors[0]
                if isinstance(audio_tensor, np.ndarray):
                    audio_tensor = torch.from_numpy(audio_tensor).float()
                elif not isinstance(audio_tensor, torch.Tensor):
                    audio_tensor = torch.tensor(audio_tensor, dtype=torch.float32)
                return OmniOutput(
                    text_hidden_states=None,
                    multimodal_outputs={"model_outputs": audio_tensor, "sr": torch.tensor(sr, dtype=torch.int)},
                )

        # If it's already a tensor, wrap it
        if isinstance(model_outputs, torch.Tensor):
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": model_outputs},
            )

        raise ValueError(f"Unsupported model_outputs type: {type(model_outputs)}")

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        """
        Create empty intermediate tensors for pipeline parallelism.

        For TTS generation models, pipeline parallelism is typically not used,
        so this returns an empty dict. However, this method is required by the
        runner infrastructure.

        Args:
            batch_size: Batch size for the intermediate tensors
            dtype: Data type for the tensors
            device: Device for the tensors

        Returns:
            IntermediateTensors: Empty dict (no PP support for TTS models)
        """
        # TTS generation models typically don't use pipeline parallelism
        # Return empty dict to satisfy the interface
        return IntermediateTensors({})

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Any = None,
        is_multimodal: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Embed input token IDs into embeddings.

        This method is called by the runner when inputs_embeds are needed.
        For TTS models, we typically work with input_ids directly, but this
        method provides a fallback for cases where embeddings are required.

        Args:
            input_ids: Input token IDs
            multimodal_embeddings: Optional multimodal embeddings (not used for TTS)
            is_multimodal: Optional mask indicating multimodal tokens (not used for TTS)
            **kwargs: Additional arguments

        Returns:
            torch.Tensor: Embedded representations of input_ids
        """
        # For TTS models, we don't have a separate embedding layer exposed,
        # so we return a dummy tensor. In practice, TTS models work with
        # input_ids directly in the forward pass.
        # This is a minimal implementation to bypass the function call.
        return torch.zeros(
            (input_ids.shape[0], input_ids.shape[1], 1024),  # Dummy hidden size
            dtype=torch.bfloat16,
            device=input_ids.device,
        )

    def embed_multimodal(self, **kwargs: Any) -> Any:
        """
        Embed multimodal inputs (e.g., images, audio).

        For TTS models, this is typically not used as they work with text input_ids.
        This method provides a stub to satisfy the interface.

        Args:
            **kwargs: Multimodal input arguments

        Returns:
            None or empty list: TTS models don't use multimodal embeddings
        """
        # TTS models work with text input_ids, not multimodal embeddings
        # Return None to indicate no multimodal embeddings
        return None

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        """Load weights into the wrapped HF model."""
        # params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            loaded_params.add(name)

        return loaded_params

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> torch.Tensor | None:
        """Non-autoregressive TTS models do not compute token logits."""
        return None


class Qwen3TTSModel:
    """
    A HuggingFace-style wrapper for Qwen3 TTS models (CustomVoice/VoiceDesign/Base) that provides:
      - from_pretrained() initialization via AutoModel/AutoProcessor
      - generation APIs for:
          * CustomVoice: generate_custom_voice()
          * VoiceDesign: generate_voice_design()
          * Base: generate_voice_clone() + create_voice_clone_prompt()
      - consistent output: (wavs: List[np.ndarray], sample_rate: int)

    Notes:
      - This wrapper expects the underlying model class to be `Qwen3TTSForConditionalGeneration`
      - Language / speaker validation is done via model methods:
          model.get_supported_languages(), model.get_supported_speakers()
    """

    def __init__(
        self, model: Qwen3TTSForConditionalGeneration, processor, generate_defaults: dict[str, Any] | None = None
    ):
        self.model = model
        self.processor = processor
        self.generate_defaults = generate_defaults or {}

        self.device = getattr(model, "device", None)
        if self.device is None:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs: Any,
    ) -> "Qwen3TTSModel":
        """
        Load a Qwen3 TTS model and its processor in HuggingFace `from_pretrained` style.

        This method:
          1) Loads config via AutoConfig (so your side can register model_type -> config/model).
          2) Loads the model via AutoModel.from_pretrained(...), forwarding `kwargs` unchanged.
          3) Loads the processor via AutoProcessor.from_pretrained(model_path).
          4) Loads optional `generate_config.json` from the model directory/repo snapshot if present.

        Args:
            pretrained_model_name_or_path (str):
                HuggingFace repo id or local directory of the model.
            **kwargs:
                Forwarded as-is into `AutoModel.from_pretrained(...)`.
                Typical examples: device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="flash_attention_2".

        Returns:
            Qwen3TTSModel:
                Wrapper instance containing `model`, `processor`, and generation defaults.
        """
        AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
        AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
        AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

        model = AutoModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if not isinstance(model, Qwen3TTSForConditionalGeneration):
            raise TypeError(f"AutoModel returned {type(model)}, expected Qwen3TTSForConditionalGeneration. ")

        processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path,
            fix_mistral_regex=True,
        )

        generate_defaults = model.generate_config
        return cls(model=model, processor=processor, generate_defaults=generate_defaults)

    def _supported_languages_set(self) -> set | None:
        langs = getattr(self.model, "get_supported_languages", None)
        if callable(langs):
            v = langs()
            if v is None:
                return None
            return set([str(x).lower() for x in v])
        return None

    def _supported_speakers_set(self) -> set | None:
        spks = getattr(self.model, "get_supported_speakers", None)
        if callable(spks):
            v = spks()
            if v is None:
                return None
            return set([str(x).lower() for x in v])
        return None

    def _validate_languages(self, languages: list[str]) -> None:
        """
        Validate that requested languages are supported by the model.

        Args:
            languages (List[str]): Language names for each sample.

        Raises:
            ValueError: If any language is not supported.
        """
        supported = self._supported_languages_set()
        if supported is None:
            return

        bad = []
        for lang in languages:
            if lang is None:
                bad.append(lang)
                continue
            if str(lang).lower() not in supported:
                bad.append(lang)
        if bad:
            raise ValueError(f"Unsupported languages: {bad}. Supported: {sorted(supported)}")

    def _validate_speakers(self, speakers: list[str | None]) -> None:
        """
        Validate that requested speakers are supported by the Instruct model.

        Args:
            speakers (List[Optional[str]]): Speaker names for each sample.

        Raises:
            ValueError: If any speaker is not supported.
        """
        supported = self._supported_speakers_set()
        if supported is None:
            return

        bad = []
        for spk in speakers:
            if spk is None or spk == "":
                continue
            if str(spk).lower() not in supported:
                bad.append(spk)
        if bad:
            raise ValueError(f"Unsupported speakers: {bad}. Supported: {sorted(supported)}")

    def _is_probably_base64(self, s: str) -> bool:
        if s.startswith("data:audio"):
            return True
        if ("/" not in s and "\\" not in s) and len(s) > 256:
            return True
        return False

    def _is_url(self, s: str) -> bool:
        try:
            u = urlparse(s)
            return u.scheme in ("http", "https") and bool(u.netloc)
        except Exception:
            return False

    def _decode_base64_to_wav_bytes(self, b64: str) -> bytes:
        if "," in b64 and b64.strip().startswith("data:"):
            b64 = b64.split(",", 1)[1]
        return base64.b64decode(b64)

    def _load_audio_to_np(self, x: str) -> tuple[np.ndarray, int]:
        if self._is_url(x):
            with urllib.request.urlopen(x) as resp:
                audio_bytes = resp.read()
            with io.BytesIO(audio_bytes) as f:
                audio, sr = sf.read(f, dtype="float32", always_2d=False)
        elif self._is_probably_base64(x):
            wav_bytes = self._decode_base64_to_wav_bytes(x)
            with io.BytesIO(wav_bytes) as f:
                audio, sr = sf.read(f, dtype="float32", always_2d=False)
        else:
            audio, sr = librosa.load(x, sr=None, mono=True)

        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)

        return audio.astype(np.float32), int(sr)

    def _normalize_audio_inputs(self, audios: AudioLike | list[AudioLike]) -> list[tuple[np.ndarray, int]]:
        """
        Normalize audio inputs into a list of (waveform, sr).

        Supported forms:
          - str: wav path / URL / base64 audio string
          - (np.ndarray, sr): waveform + sampling rate
          - list of the above

        Args:
            audios:
                Audio input(s).

        Returns:
            List[Tuple[np.ndarray, int]]:
                List of (float32 waveform, original sr).

        Raises:
            ValueError: If a numpy waveform is provided without sr.
        """
        if isinstance(audios, list):
            items = audios
        else:
            items = [audios]

        out: list[tuple[np.ndarray, int]] = []
        for a in items:
            if isinstance(a, str):
                out.append(self._load_audio_to_np(a))
            elif isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], np.ndarray):
                out.append((a[0].astype(np.float32), int(a[1])))
            elif isinstance(a, np.ndarray):
                raise ValueError("For numpy waveform input, pass a tuple (audio, sr).")
            else:
                raise TypeError(f"Unsupported audio input type: {type(a)}")
        for i, a in enumerate(out):
            if a[0].ndim > 1:
                a[0] = np.mean(a[0], axis=-1).astype(np.float32)
                out[i] = (a[0], a[1])
        return out

    def _ensure_list(self, x: MaybeList) -> list[Any]:
        return x if isinstance(x, list) else [x]

    def _build_assistant_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def _build_ref_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n"

    def _build_instruct_text(self, instruct: str) -> str:
        return f"<|im_start|>user\n{instruct}<|im_end|>\n"

    def _tokenize_texts(self, texts: list[str]) -> list[torch.Tensor]:
        input_ids = []
        for text in texts:
            input = self.processor(text=text, return_tensors="pt", padding=True)
            input_id = input["input_ids"].to(self.device)
            input_id = input_id.unsqueeze(0) if input_id.dim() == 1 else input_id
            input_ids.append(input_id)
        return input_ids

    def _merge_generate_kwargs(
        self,
        non_streaming_mode: bool | None = None,
        do_sample: bool | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        temperature: float | None = None,
        repetition_penalty: float | None = None,
        subtalker_dosample: bool | None = None,
        subtalker_top_k: int | None = None,
        subtalker_top_p: float | None = None,
        subtalker_temperature: float | None = None,
        max_new_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Merge user-provided generation arguments with defaults from `generate_config.json`.

        Rule:
          - If the user explicitly passes a value (not None), use it.
          - Otherwise, use the value from generate_config.json if present.
          - Otherwise, fall back to the hard defaults.

        Args:
            non_streaming_mode, do_sample, top_k, top_p, temperature, repetition_penalty,
            subtalker_dosample, subtalker_top_k, subtalker_top_p, subtalker_temperature, max_new_tokens:
                Common generation parameters.
            **kwargs:
                Other arguments forwarded to model.generate().

        Returns:
            Dict[str, Any]: Final kwargs to pass into model.generate().
        """
        hard_defaults = dict(
            non_streaming_mode=False,
            do_sample=True,
            top_k=50,
            top_p=1.0,
            temperature=0.9,
            repetition_penalty=1.05,
            subtalker_dosample=True,
            subtalker_top_k=50,
            subtalker_top_p=1.0,
            subtalker_temperature=0.9,
            max_new_tokens=2048,
        )

        def pick(name: str, user_val: Any) -> Any:
            if user_val is not None:
                return user_val
            if name in self.generate_defaults:
                return self.generate_defaults[name]
            return hard_defaults[name]

        merged = dict(kwargs)
        merged.update(
            non_streaming_mode=pick("non_streaming_mode", non_streaming_mode),
            do_sample=pick("do_sample", do_sample),
            top_k=pick("top_k", top_k),
            top_p=pick("top_p", top_p),
            temperature=pick("temperature", temperature),
            repetition_penalty=pick("repetition_penalty", repetition_penalty),
            subtalker_dosample=pick("subtalker_dosample", subtalker_dosample),
            subtalker_top_k=pick("subtalker_top_k", subtalker_top_k),
            subtalker_top_p=pick("subtalker_top_p", subtalker_top_p),
            subtalker_temperature=pick("subtalker_temperature", subtalker_temperature),
            max_new_tokens=pick("max_new_tokens", max_new_tokens),
        )
        return merged

    # voice clone model
    @torch.inference_mode()
    def create_voice_clone_prompt(
        self,
        ref_audio: AudioLike | list[AudioLike],
        ref_text: str | list[str | None] | None = None,
        x_vector_only_mode: bool | list[bool] = False,
    ) -> list[VoiceClonePromptItem]:
        """
        Build voice-clone prompt items from reference audio (and optionally reference text) using Base model.

        Modes:
          - x_vector_only_mode=True:
              Only speaker embedding is used to clone voice; ref_text/ref_code are ignored.
              This is mutually exclusive with ICL.
          - x_vector_only_mode=False:
              ICL mode is enabled automatically (icl_mode=True). In this case ref_text is required,
              because the model continues/conditions on the reference text + reference speech codes.

        Batch behavior:
          - ref_audio can be a single item or a list.
          - ref_text and x_vector_only_mode can be scalars or lists.
          - If any of them are lists with length > 1, lengths must match.

        Audio input:
          - str: local wav path / URL / base64
          - (np.ndarray, sr): waveform + sampling rate

        Args:
            ref_audio:
                Reference audio(s) used to extract:
                  - ref_code via `model.speech_tokenizer.encode(...)`
                  - ref_spk_embedding via `model.extract_speaker_embedding(...)` (resampled to 24k)
            ref_text:
                Reference transcript(s). Required when x_vector_only_mode=False (ICL mode).
            x_vector_only_mode:
                Whether to use speaker embedding only. If False, ICL mode will be used.

        Returns:
            List[VoiceClonePromptItem]:
                List of prompt items that can be converted into `voice_clone_prompt` dict.

        Raises:
            ValueError:
                - If x_vector_only_mode=False but ref_text is missing.
                - If batch lengths mismatch.
        """
        if self.model.tts_model_type != "base":
            raise ValueError(
                f"model with \ntokenizer_type: {self.model.tokenizer_type}\n"
                f"tts_model_size: {self.model.tts_model_size}\n"
                f"tts_model_type: {self.model.tts_model_type}\n"
                "does not support create_voice_clone_prompt, Please check Model Card or Readme for more details."
            )

        ref_audio_list = self._ensure_list(ref_audio)
        ref_text_list = (
            self._ensure_list(ref_text) if isinstance(ref_text, list) else ([ref_text] * len(ref_audio_list))
        )
        xvec_list = (
            self._ensure_list(x_vector_only_mode)
            if isinstance(x_vector_only_mode, list)
            else ([x_vector_only_mode] * len(ref_audio_list))
        )

        if len(ref_text_list) != len(ref_audio_list) or len(xvec_list) != len(ref_audio_list):
            raise ValueError(
                f"Batch size mismatch: ref_audio={len(ref_audio_list)}, "
                f"ref_text={len(ref_text_list)}, "
                f"x_vector_only_mode={len(xvec_list)}"
            )

        normalized = self._normalize_audio_inputs(ref_audio_list)

        ref_wavs_for_code: list[np.ndarray] = []
        ref_sr_for_code: list[int] = []
        for wav, sr in normalized:
            ref_wavs_for_code.append(wav)
            ref_sr_for_code.append(sr)

        if len(set(ref_sr_for_code)) == 1:
            enc = self.model.speech_tokenizer.encode(ref_wavs_for_code, sr=ref_sr_for_code[0])
            ref_codes = enc.audio_codes
        else:
            ref_codes = []
            for wav, sr in normalized:
                ref_codes.append(self.model.speech_tokenizer.encode(wav, sr=sr).audio_codes[0])

        items: list[VoiceClonePromptItem] = []
        for i, ((wav, sr), code, rtext, xvec_only) in enumerate(zip(normalized, ref_codes, ref_text_list, xvec_list)):
            if not xvec_only:
                if rtext is None or rtext == "":
                    rtext = "For profile run"
                    logger.warning(
                        f"ref_text is required when x_vector_only_mode=False (ICL mode). "
                        f"Bad index={i}. Please check if it is profile run or "
                        f"you missed to provide ref_text."
                    )
                    # raise ValueError(f"ref_text is required when x_vector_only_mode=False (ICL mode). Bad index={i}")

            wav_resample = wav
            if sr != self.model.speaker_encoder_sample_rate:
                wav_resample = librosa.resample(
                    y=wav_resample.astype(np.float32), orig_sr=int(sr), target_sr=self.model.speaker_encoder_sample_rate
                )

            spk_emb = self.model.extract_speaker_embedding(
                audio=wav_resample, sr=self.model.speaker_encoder_sample_rate
            )

            items.append(
                VoiceClonePromptItem(
                    ref_code=None if xvec_only else code,
                    ref_spk_embedding=spk_emb,
                    x_vector_only_mode=bool(xvec_only),
                    icl_mode=bool(not xvec_only),
                    ref_text=rtext,
                )
            )
        return items

    def _prompt_items_to_voice_clone_prompt(self, items: list[VoiceClonePromptItem]) -> dict[str, Any]:
        return dict(
            ref_code=[it.ref_code for it in items],
            ref_spk_embedding=[it.ref_spk_embedding for it in items],
            x_vector_only_mode=[it.x_vector_only_mode for it in items],
            icl_mode=[it.icl_mode for it in items],
        )

    # voice clone model
    @torch.no_grad()
    def generate_voice_clone(
        self,
        text: str | list[str],
        language: str | list[str] = None,
        ref_audio: AudioLike | list[AudioLike] | None = None,
        ref_text: str | list[str | None] | None = None,
        x_vector_only_mode: bool | list[bool] = False,
        voice_clone_prompt: dict[str, Any] | list[VoiceClonePromptItem] | None = None,
        **kwargs: Any,
    ) -> tuple[list[np.ndarray], int]:
        """
        Voice clone speech using the Base model.

        You can provide either:
          - (ref_audio, ref_text, x_vector_only_mode) and let this method build the prompt, OR
          - `VoiceClonePromptItem` returned by `create_voice_clone_prompt`, OR
          - a list of `VoiceClonePromptItem` returned by `create_voice_clone_prompt`.

        `ref_audio` Supported forms:
        - str: wav path / URL / base64 audio string
        - (np.ndarray, sr): waveform + sampling rate
        - list of the above

        Input flexibility:
          - text/language can be scalar or list.
          - prompt can be single or batch.
          - If batch mode (len(text)>1), lengths must match.

        Args:
            text:
                Text(s) to synthesize.
            language:
                Language(s) for each sample.
            ref_audio:
                Reference audio(s) for prompt building. Required if voice_clone_prompt is not provided.
            ref_text:
                Reference text(s) used for ICL mode (required when x_vector_only_mode=False).
            x_vector_only_mode:
                If True, only speaker embedding is used (ignores ref_text/ref_code).
                If False, ICL mode is used automatically.
            voice_clone_prompt:
                list[VoiceClonePromptItem] from `create_voice_clone_prompt`.
            **kwargs:
                Additional generation options. Common keys include `non_streaming_mode`, `do_sample`, `top_k`, `top_p`,
                `temperature`, `repetition_penalty`, `subtalker_dosample`, `subtalker_top_k`, `subtalker_top_p`,
                `subtalker_temperature`, and `max_new_tokens`. Any other keyword arguments supported by HuggingFace
                Transformers `generate()` can also be passed and will be forwarded to
                `Qwen3TTSForConditionalGeneration.generate(...)`.

        Returns:
            Tuple[List[np.ndarray], int]:
                (wavs, sample_rate)

        Raises:
            ValueError:
                If batch sizes mismatch or required prompt inputs are missing.
        """
        if self.model.tts_model_type != "base":
            raise ValueError(
                f"model with \ntokenizer_type: {self.model.tokenizer_type}\n"
                f"tts_model_size: {self.model.tts_model_size}\n"
                f"tts_model_type: {self.model.tts_model_type}\n"
                "does not support generate_voice_clone, Please check Model Card or Readme for more details."
            )

        texts = self._ensure_list(text)
        languages = (
            self._ensure_list(language)
            if isinstance(language, list)
            else ([language] * len(texts) if language is not None else ["Auto"] * len(texts))
        )
        if len(languages) == 1 and len(texts) > 1:
            languages = languages * len(texts)
        if len(texts) != len(languages):
            raise ValueError(f"Batch size mismatch: text={len(texts)}, language={len(languages)}")

        self._validate_languages(languages)

        if voice_clone_prompt is None:
            if ref_audio is None:
                # For profile run
                sample_rate = int(self.model.speaker_encoder_sample_rate)
                # Use a 1-second silent clip to satisfy padding requirements.
                ref_audio = (np.zeros(sample_rate, dtype=np.float32), sample_rate)
                logger.warning(
                    "ref_audio is not provided. Using a 1-second silent clip "
                    "to satisfy padding requirements. Please check if it is "
                    "profile run or you missed to provide ref_audio."
                )
            prompt_items = self.create_voice_clone_prompt(
                ref_audio=ref_audio, ref_text=ref_text, x_vector_only_mode=x_vector_only_mode
            )
            if len(prompt_items) == 1 and len(texts) > 1:
                prompt_items = prompt_items * len(texts)
            if len(prompt_items) != len(texts):
                raise ValueError(f"Batch size mismatch: prompt={len(prompt_items)}, text={len(texts)}")
            voice_clone_prompt_dict = self._prompt_items_to_voice_clone_prompt(prompt_items)
            ref_texts_for_ids = [it.ref_text for it in prompt_items]
        else:
            if isinstance(voice_clone_prompt, list):
                prompt_items = voice_clone_prompt
                if len(prompt_items) == 1 and len(texts) > 1:
                    prompt_items = prompt_items * len(texts)
                if len(prompt_items) != len(texts):
                    raise ValueError(f"Batch size mismatch: prompt={len(prompt_items)}, text={len(texts)}")
                voice_clone_prompt_dict = self._prompt_items_to_voice_clone_prompt(prompt_items)
                ref_texts_for_ids = [it.ref_text for it in prompt_items]
            else:
                voice_clone_prompt_dict = voice_clone_prompt
                ref_texts_for_ids = None

        input_texts = [self._build_assistant_text(t) for t in texts]
        input_ids = self._tokenize_texts(input_texts)

        ref_ids = None
        if ref_texts_for_ids is not None:
            ref_ids = []
            for i, rt in enumerate(ref_texts_for_ids):
                if rt is None or rt == "":
                    ref_ids.append(None)
                else:
                    ref_tok = self._tokenize_texts([self._build_ref_text(rt)])[0]
                    ref_ids.append(ref_tok)

        gen_kwargs = self._merge_generate_kwargs(**kwargs)

        talker_codes_list, _ = self.model.generate(
            input_ids=input_ids,
            ref_ids=ref_ids,
            voice_clone_prompt=voice_clone_prompt_dict,
            languages=languages,
            **gen_kwargs,
        )

        codes_for_decode = []
        for i, codes in enumerate(talker_codes_list):
            ref_code_list = voice_clone_prompt_dict.get("ref_code", None)
            if ref_code_list is not None and ref_code_list[i] is not None:
                codes_for_decode.append(torch.cat([ref_code_list[i].to(codes.device), codes], dim=0))
            else:
                codes_for_decode.append(codes)

        wavs_all, fs = self.model.speech_tokenizer.decode([{"audio_codes": c} for c in codes_for_decode])

        wavs_out: list[np.ndarray] = []
        for i, wav in enumerate(wavs_all):
            ref_code_list = voice_clone_prompt_dict.get("ref_code", None)
            if ref_code_list is not None and ref_code_list[i] is not None:
                ref_len = int(ref_code_list[i].shape[0])
                total_len = int(codes_for_decode[i].shape[0])
                cut = int(ref_len / max(total_len, 1) * wav.shape[0])
                wavs_out.append(wav[cut:])
            else:
                wavs_out.append(wav)

        return wavs_out, fs

    # voice design model
    @torch.no_grad()
    def generate_voice_design(
        self,
        text: str | list[str],
        instruct: str | list[str],
        language: str | list[str] = None,
        **kwargs: Any,
    ) -> tuple[list[np.ndarray], int]:
        """
        Generate speech with the VoiceDesign model using natural-language style instructions.

        Args:
            text:
                Text(s) to synthesize.
            language:
                Language(s) for each sample.
            instruct:
                Instruction(s) describing desired voice/style. Empty string is allowed (treated as no instruction).
            **kwargs:
                Additional generation options. Common keys include `non_streaming_mode`, `do_sample`, `top_k`, `top_p`,
                `temperature`, `repetition_penalty`, `subtalker_dosample`, `subtalker_top_k`, `subtalker_top_p`,
                `subtalker_temperature`, and `max_new_tokens`. Any other keyword arguments supported by HuggingFace
                Transformers `generate()` can also be passed and will be forwarded to
                `Qwen3TTSForConditionalGeneration.generate(...)`.

        Returns:
            Tuple[List[np.ndarray], int]:
                (wavs, sample_rate)
        """
        if self.model.tts_model_type != "voice_design":
            raise ValueError(
                f"model with \ntokenizer_type: {self.model.tokenizer_type}\n"
                f"tts_model_size: {self.model.tts_model_size}\n"
                f"tts_model_type: {self.model.tts_model_type}\n"
                "does not support generate_voice_design, Please check Model Card or Readme for more details."
            )

        texts = self._ensure_list(text)
        languages = (
            self._ensure_list(language)
            if isinstance(language, list)
            else ([language] * len(texts) if language is not None else ["Auto"] * len(texts))
        )
        instructs = self._ensure_list(instruct)

        if len(languages) == 1 and len(texts) > 1:
            languages = languages * len(texts)
        if len(instructs) == 1 and len(texts) > 1:
            instructs = instructs * len(texts)

        if not (len(texts) == len(languages) == len(instructs)):
            raise ValueError(
                f"Batch size mismatch: text={len(texts)}, language={len(languages)}, instruct={len(instructs)}"
            )

        self._validate_languages(languages)

        input_ids = self._tokenize_texts([self._build_assistant_text(t) for t in texts])

        instruct_ids: list[torch.Tensor | None] = []
        for ins in instructs:
            if ins is None or ins == "":
                instruct_ids.append(None)
            else:
                instruct_ids.append(self._tokenize_texts([self._build_instruct_text(ins)])[0])

        gen_kwargs = self._merge_generate_kwargs(**kwargs)

        talker_codes_list, _ = self.model.generate(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            languages=languages,
            **gen_kwargs,
        )

        wavs, fs = self.model.speech_tokenizer.decode([{"audio_codes": c} for c in talker_codes_list])
        return wavs, fs

    # custom voice model
    @torch.no_grad()
    def generate_custom_voice(
        self,
        text: str | list[str],
        speaker: str | list[str],
        language: str | list[str] = None,
        instruct: str | list[str] | None = None,
        **kwargs: Any,
    ) -> tuple[list[np.ndarray], int]:
        """
        Generate speech with the CustomVoice model using a predefined speaker id,
        optionally controlled by instruction text.

        Args:
            text:
                Text(s) to synthesize.
            language:
                Language(s) for each sample.
            speaker:
                Speaker name(s). Will be validated against `model.get_supported_speakers()` (case-insensitive).
            instruct:
                Optional instruction(s). If None, treated as empty (no instruction).
            **kwargs:
                Additional generation options. Common keys include `non_streaming_mode`, `do_sample`, `top_k`, `top_p`,
                `temperature`, `repetition_penalty`, `subtalker_dosample`, `subtalker_top_k`, `subtalker_top_p`,
                `subtalker_temperature`, and `max_new_tokens`. Any other keyword arguments supported by HuggingFace
                Transformers `generate()` can also be passed and will be forwarded to
                `Qwen3TTSForConditionalGeneration.generate(...)`.

        Returns:
            Tuple[List[np.ndarray], int]:
                (wavs, sample_rate)

        Raises:
            ValueError:
                If any speaker/language is unsupported or batch sizes mismatch.
        """
        if self.model.tts_model_type != "custom_voice":
            raise ValueError(
                f"model with \ntokenizer_type: {self.model.tokenizer_type}\n"
                f"tts_model_size: {self.model.tts_model_size}\n"
                f"tts_model_type: {self.model.tts_model_type}\n"
                "does not support generate_custom_voice, Please check Model Card or Readme for more details."
            )

        texts = self._ensure_list(text)
        languages = (
            self._ensure_list(language)
            if isinstance(language, list)
            else ([language] * len(texts) if language is not None else ["Auto"] * len(texts))
        )
        speakers = self._ensure_list(speaker)
        if self.model.tts_model_size in "0b6":  # for 0b6 model, instruct is not supported
            instruct = None
        instructs = (
            self._ensure_list(instruct)
            if isinstance(instruct, list)
            else ([instruct] * len(texts) if instruct is not None else [""] * len(texts))
        )

        if len(languages) == 1 and len(texts) > 1:
            languages = languages * len(texts)
        if len(speakers) == 1 and len(texts) > 1:
            speakers = speakers * len(texts)
        if len(instructs) == 1 and len(texts) > 1:
            instructs = instructs * len(texts)

        if not (len(texts) == len(languages) == len(speakers) == len(instructs)):
            raise ValueError(
                f"Batch size mismatch: text={len(texts)}, "
                f"language={len(languages)}, speaker={len(speakers)}, "
                f"instruct={len(instructs)}"
            )

        self._validate_languages(languages)
        self._validate_speakers(speakers)

        input_ids = self._tokenize_texts([self._build_assistant_text(t) for t in texts])

        instruct_ids: list[torch.Tensor | None] = []
        for ins in instructs:
            if ins is None or ins == "":
                instruct_ids.append(None)
            else:
                instruct_ids.append(self._tokenize_texts([self._build_instruct_text(ins)])[0])

        gen_kwargs = self._merge_generate_kwargs(**kwargs)

        talker_codes_list, _ = self.model.generate(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            languages=languages,
            speakers=speakers,
            **gen_kwargs,
        )

        wavs, fs = self.model.speech_tokenizer.decode([{"audio_codes": c} for c in talker_codes_list])
        return wavs, fs

    def get_supported_speakers(self) -> list[str] | None:
        """
        List supported speaker names for the current model.

        This is a convenience wrapper around `model.get_supported_speakers()`.
        If the underlying model does not expose speaker constraints (returns None),
        this method also returns None.

        Returns:
            Optional[List[str]]:
                - A sorted list of supported speaker names (lowercased), if available.
                - None if the model does not provide supported speakers.
        """
        supported = self._supported_speakers_set()
        if supported is None:
            return None
        return sorted(supported)

    def get_supported_languages(self) -> list[str] | None:
        """
        List supported language names for the current model.

        This is a convenience wrapper around `model.get_supported_languages()`.
        If the underlying model does not expose language constraints (returns None),
        this method also returns None.

        Returns:
            Optional[List[str]]:
                - A sorted list of supported language names (lowercased), if available.
                - None if the model does not provide supported languages.
        """
        supported = self._supported_languages_set()
        if supported is None:
            return None
        return sorted(supported)
