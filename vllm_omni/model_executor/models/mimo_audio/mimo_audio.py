# Copyright 2025 Xiaomi Corporation.
import os
from collections.abc import Iterable, Mapping, Sequence
from functools import cached_property
from typing import Any

import numpy as np
import torch
from torch import nn
from transformers import BatchFeature, Qwen2Config
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models import SupportsPP
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.utils import init_vllm_registered_model
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    AudioItem,
    ModalityData,
    MultiModalBatchedField,
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalFieldElem,
    MultiModalKwargsItem,
)
from vllm.multimodal.parse import AudioProcessorItems, MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    ProcessorInputs,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.utils import group_mm_kwargs_by_modality
from vllm.sequence import IntermediateTensors
from vllm.utils.cache import LRUCache
from vllm.utils.collection_utils import is_list_of
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.custom_process_mixin import CustomProcessMixin
from vllm_omni.model_executor.models.mimo_audio.config_mimo_audio import (
    NO_INTERLEAVE_NEXT_TOKEN_ID,
    PAD_GROUP_SIZE,
    SPAN_CODEC_END_TOKEN_ID,
    SPAN_CODEC_START_TOKEN_ID,
    TALKER_CODEC_PAD_TOKEN_ID,
    TEXT_GROUP_SIZE,
    MiMoAudioConfig,
)
from vllm_omni.model_executor.models.mimo_audio.mimo_audio_code2wav import get_tokenizer_worker
from vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni import OmniOutput

logger = init_logger(__name__)


def interleave_5_and_5_in_span(
    input_ids: list[int],
    *,
    span_start_token: int = SPAN_CODEC_START_TOKEN_ID,
    span_end_token: int = SPAN_CODEC_END_TOKEN_ID,
    pad_token_id: int = TALKER_CODEC_PAD_TOKEN_ID,
    text_group_size: int = TEXT_GROUP_SIZE,
    pad_group_size: int = PAD_GROUP_SIZE,
    no_interleave_next_token: int = NO_INTERLEAVE_NEXT_TOKEN_ID,
) -> list[int]:
    """
    Interleave text tokens and padding tokens within spans marked by special tokens.

    Finds spans delimited by span_start_token and span_end_token, then reorganizes
    tokens inside each span by grouping text tokens (excluding padding) into groups
    of text_group_size, followed by pad_group_size padding tokens. Interleaving is
    skipped if the token after span_end_token is no_interleave_next_token.

    Args:
        input_ids: Input token ID list
        span_start_token: Token marking span start (default: 151670)
        span_end_token: Token marking span end (default: 151672)
        pad_token_id: Padding token ID (default: 151667)
        text_group_size: Number of text tokens per group (default: 5)
        pad_group_size: Number of padding tokens per group (default: 5)
        no_interleave_next_token: Skip interleaving if this token follows span_end (default: 151671)

    Returns:
        Processed token list with same length as input
    """
    if not input_ids:
        return input_ids

    original_len = len(input_ids)
    output_ids: list[int] = []
    cursor = 0

    while cursor < original_len:
        # Non-span starting point: Copy as is
        if input_ids[cursor] != span_start_token:
            output_ids.append(input_ids[cursor])
            cursor += 1
            continue

        # ===== Enter span processing =====
        output_ids.append(input_ids[cursor])
        cursor += 1

        span_end_pos = cursor
        while span_end_pos < original_len and input_ids[span_end_pos] != span_end_token:
            span_end_pos += 1

        # If span_end_token is not found: Output the remaining as is and end
        if span_end_pos >= original_len:
            output_ids.extend(input_ids[cursor:])
            break

        next_pos = span_end_pos + 1
        next_token = input_ids[next_pos] if next_pos < original_len else None

        # Rule:
        # After "-end" is 151667(PAD) -> for interlacing
        # After "-end" is 151671 -> Do not interlace
        do_interleave = next_token == pad_token_id
        if next_token == no_interleave_next_token:
            do_interleave = False

        if not do_interleave:
            output_ids.extend(input_ids[cursor:span_end_pos])
            output_ids.append(span_end_token)
            cursor = span_end_pos + 1
            continue

        span_content = input_ids[cursor:span_end_pos]
        span_original_len = len(span_content)
        span_text_tokens = [t for t in span_content if t != pad_token_id]

        rebuilt_span: list[int] = []
        text_cursor = 0
        total_text = len(span_text_tokens)

        while text_cursor < total_text:
            take_text = min(text_group_size, total_text - text_cursor)
            rebuilt_span.extend(span_text_tokens[text_cursor : text_cursor + take_text])
            text_cursor += take_text

            if text_cursor < total_text:
                rebuilt_span.extend([pad_token_id] * pad_group_size)

        output_ids.extend(rebuilt_span)
        output_ids.append(span_end_token)

        extra_tokens_added = len(rebuilt_span) - span_original_len
        cursor = span_end_pos + 1

        while extra_tokens_added > 0 and cursor < original_len:
            if input_ids[cursor] == pad_token_id:
                extra_tokens_added -= 1
                cursor += 1
            else:
                output_ids.append(input_ids[cursor])
                cursor += 1

        if extra_tokens_added > 0:
            return output_ids[:original_len]

    if len(output_ids) > original_len:
        output_ids = output_ids[:original_len]
    elif len(output_ids) < original_len:
        output_ids.extend([pad_token_id] * (original_len - len(output_ids)))

    return output_ids


class MiMoAudioLLMProcessingInfo(
    BaseProcessingInfo,
):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen2Config)

    def get_hf_processor(
        self,
        *,
        # Ignored in initialization
        sampling_rate: int | None = None,
        **kwargs: object,
    ):
        # MiMoAudio doesn't use a standard HF processor like Qwen2Audio
        # Instead it uses MiMoAudioTokenizer directly
        # This method may return None or a dummy processor
        # depending on the actual implementation requirements
        return None

    def get_data_parser(self) -> MultiModalDataParser:
        sampling_rate = 24000
        return MiMoAudioDataParser(target_sr=sampling_rate)

    def get_feature_extractor(
        self,
        *,
        # Ignored in initialization
        sampling_rate: int | None = None,
    ):
        # MiMoAudio uses MiMoAudioTokenizer instead of WhisperFeatureExtractor
        # This method may need to return the tokenizer or None
        # depending on the actual implementation requirements
        return None

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        # MiMoAudio has different tokenization than Qwen2Audio
        # This is a placeholder - adjust based on actual MiMoAudio tokenizer limits
        # For now, return a large value or None
        # This should be adjusted based on MiMoAudio's actual constraints
        return {"audio": 1}


class MiMoAudioLLMDummyInputsBuilder(BaseDummyInputsBuilder[MiMoAudioLLMProcessingInfo]):
    _processor_inputs_cache: LRUCache = LRUCache(capacity=1024)

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        return (" <|empty|>" * num_audios).strip()

    def get_dummy_mm_data(
        self, seq_len: int, mm_counts: Mapping[str, int], mm_options: Mapping[str, BaseDummyOptions] | None = None
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        if num_audios == 0:
            return {}
        # Return dummy raw audio data (not encoded codes)
        # This will be processed by _parse_audio_data like real audio
        # Use 1 second of audio at target_sr (24000 Hz)
        dummy_audio_length = mm_options.get("audio").length if mm_options else 24000  # 1 second at 24kHz
        dummy_audio = np.zeros((dummy_audio_length,), dtype=np.float32)
        return {"audio": [(dummy_audio, 24000)] * num_audios}

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> ProcessorInputs:
        dummy_text = self.get_dummy_text(mm_counts)
        dummy_mm_data = self.get_dummy_mm_data(seq_len, mm_counts, mm_options)

        dummy_mm_items = self.info.parse_mm_data(dummy_mm_data)

        dummy_processor_inputs = ProcessorInputs(
            prompt=dummy_text,
            mm_items=dummy_mm_items,
        )

        return dummy_processor_inputs


class MiMoAudioDataParser(MultiModalDataParser):
    def __init__(self, target_sr: int, use_mono_channel: bool = True):
        super().__init__(target_sr=target_sr)
        self.use_mono_channel = use_mono_channel
        self.target_sr = target_sr

        tokenizer_device = os.environ.get("MIMO_AUDIO_TOKENIZER_DEVICE", None)

        if tokenizer_device:
            if tokenizer_device.lower() == "cpu":
                self.device = torch.device("cpu")
            else:
                # Support formats like "cuda", "cuda:0", "cuda:6", etc.
                self.device = torch.device(tokenizer_device)
        else:
            # Default to cuda (will use current GPU)
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")

        self.audio_tokenizer_path = os.environ.get("MIMO_AUDIO_TOKENIZER_PATH", None)
        if not self.audio_tokenizer_path:
            raise ValueError(
                "Audio tokenizer path is not set. Provide "
                "`model_config.audio_tokenizer_path` in the stage config "
                "or export MIMO_AUDIO_TOKENIZER_PATH."
            )

        if not os.path.exists(self.audio_tokenizer_path):
            raise ValueError(
                "Audio tokenizer not exists. Provide "
                "`model_config.audio_tokenizer_path` in the stage config "
                "or export MIMO_AUDIO_TOKENIZER_PATH."
            )

        self.tokenizer_config_path = os.environ.get("MIMO_AUDIO_TOKENIZER_PATH", None)

        self.mimo_tokenizer = get_tokenizer_worker(
            device=self.device,
            config_path=self.tokenizer_config_path,
            audio_tokenizer_path=self.audio_tokenizer_path,
        )

    def _parse_audio_data(
        self,
        data: ModalityData[AudioItem],
    ):
        if data is None:
            return AudioProcessorItems(None)

        # also check single audio item with sampling rate
        if self._is_empty(data) or (isinstance(data, tuple) and self._is_empty(data[0])):
            return None

        if (
            is_list_of(data, float)
            or isinstance(data, (np.ndarray, torch.Tensor))
            and data.ndim == 1
            or isinstance(data, tuple)
        ):
            data_items = [data]
        elif isinstance(data, (np.ndarray, torch.Tensor)):
            data_items = [elem for elem in data]
        else:
            data_items = data

        new_audios = list[np.ndarray]()
        for data_item in data_items:
            if not isinstance(data_item, tuple) or not isinstance(data_item[0], np.ndarray):
                new_audios.append(data_item)
                continue

            audio, orig_sr = self._get_audio_with_sr(data_item)
            if orig_sr is None:
                new_audio = audio
            else:
                new_audio = self.audio_resampler.resample(audio, orig_sr=orig_sr)
            wav_tensor = torch.from_numpy(new_audio.astype(np.float32))
            if wav_tensor.dim() == 1:
                wav_tensor = wav_tensor.unsqueeze(0)
            if self.use_mono_channel:
                if wav_tensor.size(0) > 1:
                    wav_mono = wav_tensor.mean(dim=0)  # [samples]
                else:
                    wav_mono = wav_tensor[0]
            audio_codes = self.mimo_tokenizer.encode(audio=(wav_mono, self.target_sr))
            new_audio = audio_codes.detach().cpu().numpy().astype(np.int64).tolist()
            new_audios.append(new_audio)

        return AudioProcessorItems(new_audios)


class MiMoAudioLLMMultiModalProcessor(BaseMultiModalProcessor[MiMoAudioLLMProcessingInfo]):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, Any],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        sampling_rate = 24000
        audios = mm_data.pop("audios", [])
        tokenizer = self.info.get_tokenizer()
        if audios:
            mm_data["audio"] = audios

        if isinstance(prompt, str):
            prompt_ids = tokenizer.encode(prompt)
        else:
            prompt_ids = prompt

        # Text-only input not supported in composite processor
        if not mm_data.get("audio", []):
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)

            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        # feature_extractor = self.info.get_feature_extractor(**mm_kwargs)
        mm_kwargs = dict(
            **mm_kwargs,
            sampling_rate=sampling_rate,
        )

        audio_lengths = [len(x) for x in audios]
        if all(audio_length == 8 for audio_length in audio_lengths):
            audio_lengths = [len(x[0]) for x in audios]

        return BatchFeature(
            data={
                "input_ids": torch.tensor([prompt_ids], dtype=torch.int64),
                "audio_embeds": audios,
                "audio_lengths": audio_lengths,
                "mm_data": mm_data,
                "mm_kwargs": mm_kwargs,
                "tok_kwargs": tok_kwargs,
            },
            tensor_type=None,
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        # MiMoAudio may have different field names than Qwen2Audio
        # Adjust based on actual MiMoAudio model requirements
        # For now, return empty dict or adjust based on actual needs
        return {
            "audio_embeds": MultiModalFieldConfig.batched("audio"),
            "audio_lengths": MultiModalFieldConfig.batched("audio"),
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs,
    ) -> Sequence[PromptUpdate]:
        # MiMoAudio doesn't use a standard HF processor like Qwen2Audio
        # Get tokenizer and vocabulary directly
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        # MiMoAudio may use different audio tokens
        # Adjust these based on actual tokenizer vocabulary
        # Common tokens in MiMoAudio might be different from Qwen2Audio
        audio_token = "<|empty|>"

        audio_bos_id = None
        audio_eos_id = None
        audio_token_id = vocab.get(audio_token, None)

        # If tokens don't exist, try alternative tokens or use None
        if audio_token_id is None:
            # Try alternative token names or skip audio token replacement
            return []

        audio_kwargs = out_mm_kwargs.get("audio", [])
        # feature_attention_mask = out_mm_kwargs.get("feature_attention_mask", None)

        if audio_kwargs:
            audio_output_lengths = [item["audio_lengths"].data for item in audio_kwargs]
        else:
            audio_output_lengths = []
            if mm_items and "audio" in mm_items:
                try:
                    audios = mm_items.get_items("audio", AudioProcessorItems)
                    if audios is not None:
                        num_audios = audios.get_count()
                        audio_output_lengths = [audios.get_audio_length(i) for i in range(num_audios)]
                except Exception as e:
                    logger.error(f"_get_prompt_updates failed: {e}")

        def get_replacement_mimo_audio(item_idx: int):
            if item_idx >= len(audio_output_lengths):
                # If no length info, use a default or skip
                return PromptUpdateDetails.select_token_id(
                    [audio_token_id],
                    embed_token_id=audio_token_id,
                )

            num_features = audio_output_lengths[item_idx] // 4
            if num_features == 0:
                try:
                    audios = mm_items.get_items("audio", AudioProcessorItems)
                    audio_len = audios.get_audio_length(item_idx)
                    raise ValueError(f"The audio (len={audio_len}) is too short to be represented inside the model")
                except (AttributeError, KeyError):
                    # If AudioProcessorItems is not available, use default
                    num_features = 1

            audio_tokens = [audio_token_id] * num_features

            # Build replacement tokens following Qwen2Audio pattern
            # [audio_bos_id] + audio_tokens + [audio_eos_id]
            replacement_tokens = []
            if audio_bos_id is not None:
                replacement_tokens.append(audio_bos_id)
            replacement_tokens.extend(audio_tokens)
            if audio_eos_id is not None:
                replacement_tokens.append(audio_eos_id)

            # If no bos/eos tokens, just use audio tokens
            if not replacement_tokens:
                replacement_tokens = audio_tokens

            return PromptUpdateDetails.select_token_id(
                replacement_tokens,
                embed_token_id=audio_token_id,
            )

        return [
            PromptReplacement(
                modality="audio",
                target=audio_token,
                replacement=get_replacement_mimo_audio,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    MiMoAudioLLMMultiModalProcessor,
    info=MiMoAudioLLMProcessingInfo,
    dummy_inputs=MiMoAudioLLMDummyInputsBuilder,
)
class MiMoAudioForConditionalGeneration(
    nn.Module,
    SupportsPP,
    SupportsMultiModal,
    CustomProcessMixin,
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|vision_start|><|IMAGE|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|VIDEO|><|vision_end|>"
        if modality.startswith("audio"):
            return "<|sosp|><|empty|><|eosp|>"

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.has_preprocess = False
        self.have_multimodal_outputs = True
        self.requires_raw_input_tokens = True
        config = vllm_config.model_config.hf_config
        config = MiMoAudioConfig(**vars(config)) if isinstance(config, Qwen2Config) else config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.config = config
        self.lora_config = lora_config

        self.quant_config = quant_config
        self.vllm_config = vllm_config

        self.llm_prev_hidden_state = None
        self.device = vllm_config.device_config.device
        self.pin_memory = is_pin_memory_available()

        if "model_stage" in os.environ:
            self.model_stage = os.environ["model_stage"]
        else:
            self.model_stage = vllm_config.model_config.model_stage

        if self.model_stage == "fused_thinker_talker":
            self.has_preprocess = True
            self.set_custom_preprocess(self.fused_thinker_talker_preprocess)
            # Initialize fused_thinker_talker llm model (multimodal processing)
            self.fused_thinker_talker = init_vllm_registered_model(
                vllm_config=vllm_config,
                # prefix=maybe_prefix(prefix, "model"),
                hf_config=vllm_config.model_config.hf_config,
                # Use registry architecture key
                architectures=["MiMoAudioLLMModel"],
            )
            self.token2wav = None
            self.model = self.fused_thinker_talker

        elif self.model_stage == "code2wav":
            self.fused_thinker_talker = None
            # Initialize token2wav (code->mel->wav) like thinker/talker
            self.token2wav = init_vllm_registered_model(
                vllm_config=vllm_config,
                # prefix=maybe_prefix(prefix, "token2wav"),
                # hf_config=self.token2wav_config,
                architectures=["MiMoAudioToken2WavModel"],
            )

            self.model = self.token2wav
        else:
            raise ValueError("Invalid model stage")

        # Set up intermediate tensors
        self.make_empty_intermediate_tensors = (
            (self.fused_thinker_talker.make_empty_intermediate_tensors)
            if self.model_stage == "fused_thinker_talker"
            else lambda: None
        )

    def fused_thinker_talker_preprocess(self, input_ids: torch.Tensor, input_embeds: torch.Tensor, **info_dict: dict):
        # Mixed-mode support: In a single step, both Prefill*n and Decode*n are supported.

        # Ensure we have base embeddings when only ids are provided
        if input_embeds is None and input_ids is not None:
            input_embeds = self.fused_thinker_talker.embed_input_ids(input_ids)

        span_len = input_ids.shape[0]

        if span_len > 1:
            # prefill
            return self.fused_thinker_talker_prefill(input_ids, input_embeds, **info_dict)
        else:
            # decode
            return input_ids, input_embeds, info_dict

    def fused_thinker_talker_prefill(self, input_ids: torch.Tensor, input_embeds: torch.Tensor, **info_dict: dict):
        empty_token_id = self.fused_thinker_talker.empty_token_id

        mm_kwargs = list[tuple[str, MultiModalKwargsItem]]()
        mm_features = info_dict.get("mm_features", [])
        mm_embeddings = []
        prompt_ids = torch.tensor(
            interleave_5_and_5_in_span(input_ids.tolist()),
            dtype=torch.int64,
            device=input_ids.device,
        )

        for mm_feature in mm_features:
            mm_item = mm_feature.data

            if mm_item is not None:
                mm_item["mm_offset"] = MultiModalFieldElem(
                    data=mm_feature.mm_position.offset,
                    field=MultiModalBatchedField(),
                )
            mm_kwargs.append((mm_feature.modality, mm_item))

        for modality, num_items, mm_kwargs_group in group_mm_kwargs_by_modality(
            mm_kwargs,
            device=self.device,
            pin_memory=self.pin_memory,
        ):
            # MiMoAudio is only supported audio modality
            mm_kwargs_group["prompt_ids"] = prompt_ids
            mm_kwargs_group["modality_preprocess"] = True
            mm_kwargs_group["mm_offset"] = torch.tensor(
                mm_kwargs_group["mm_offset"], dtype=torch.long, device=self.device
            )
            mm_embeddings = self.fused_thinker_talker.embed_multimodal(**mm_kwargs_group)

        input_embeds = self.fused_thinker_talker.embed_input_ids(
            prompt_ids, multimodal_embeddings=mm_embeddings, is_multimodal=(prompt_ids == empty_token_id)
        )

        return prompt_ids, input_embeds, info_dict

    @staticmethod
    def _module_device(module: nn.Module) -> torch.device:
        try:
            return next(module.parameters()).device
        except StopIteration:
            # No parameters; fall back to buffers or cpu
            for _, buf in module.named_buffers(recurse=True):
                return buf.device
            return torch.device("cpu")

    def move_submodules_to_devices(
        self,
        *,
        llm_device: str | torch.device | None = None,
        token2wav_device: str | torch.device | None = None,
    ) -> None:
        """Optionally move thinker/talker/token2wav to different devices.

        Example:
            model.move_submodules_to_devices(
                thinker_device='cuda:0',
                talker_device='cuda:1',
                token2wav_device='cpu',
            )
        """
        if llm_device is not None and self.fused_thinker_talker is not None:
            self.fused_thinker_talker.to(llm_device)
        if token2wav_device is not None and self.token2wav is not None:
            self.token2wav.to(token2wav_device)

    @cached_property
    def sampler(self):
        if hasattr(self.model, "sampler"):
            return self.model.sampler
        return Sampler()

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal: bool = False,
    ) -> torch.Tensor:
        if self.model_stage == "code2wav":
            return torch.zeros_like(input_ids).reshape(-1, 1).repeat(1, self.vllm_config.model_config.get_hidden_size())
        return self.model.embed_input_ids(input_ids, multimodal_embeddings, is_multimodal=is_multimodal)

    def embed_multimodal(self, **kwargs):
        # Delegate to thinker model for multimodal processing
        return self.model.embed_multimodal(**kwargs)

    def last_index_of(self, list, value):
        return len(list) - 1 - list[::-1].index(value)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        """Load weights for all components of the omni model."""
        loaded_weights = set()
        llm_weights = []
        token2wav_weights = []
        for k, v in weights:
            if (
                k.startswith("model.")
                or k.startswith("lm_head.")
                or k.startswith("input_local_transformer.")
                or k.startswith("hidden_states_downcast.")
                or k.startswith("local_transformer")
                or k.startswith("speech_embeddings")
                or k.startswith("speech_group_downcast")
            ):
                if self.fused_thinker_talker:
                    llm_weights.append((k, v))
            elif k.startswith("mel_transform."):
                if self.token2wav:
                    token2wav_weights.append((k, v))
            else:
                pass

        # Load fused_thinker_talker llm weights
        if self.fused_thinker_talker:
            if llm_weights:
                llm_loaded = self.fused_thinker_talker.load_weights(llm_weights)
            else:
                llm_loaded = set([k for k, v in llm_weights])
            # thinker_loaded = add_prefix_to_loaded_weights(thinker_loaded, "thinker")
            loaded_weights.update(llm_loaded)

        ## load code2wav
        if self.token2wav:
            token2wav_loaded = self.token2wav.load_weights(
                token2wav_weights,
                os.environ["MIMO_AUDIO_TOKENIZER_PATH"],
            )

            loaded_weights.update(token2wav_loaded)

    @staticmethod
    def insert_between(input_ids: torch.Tensor, group_size: int, value: int = -100):
        if group_size < 0:
            raise ValueError("group_size must be non-negative")

        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids)
        if input_ids.dim() != 1:
            raise ValueError("input_ids must be 1-D tensor or list")

        L = input_ids.numel()
        if group_size == 0 or L == 0:
            return input_ids.clone()

        new_len = L * (group_size + 1)
        # Create output tensor filled with value, dtype & device consistent with input_ids
        out = input_ids.new_full((new_len,), value)

        # Place original elements at the first position of each block: 0, group_size+1, 2*(group_size+1), ...
        positions = torch.arange(L, dtype=torch.long, device=input_ids.device) * (group_size + 1)
        out[positions] = input_ids

        return out

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        generate_audio: bool = True,
        codec: torch.Tensor | None = None,
        sampling_metadata: SamplingMetadata | None = None,
        logits_index: int | None = None,
        sampler=None,
        additional_information: dict[str, object] | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors | OmniOutput:
        """
        Workflow:
        1) llm: multimodal understanding → text hidden states.
        2) If audio requested and codec not provided, use talker to derive codec.
        3) If audio requested (or codec provided), use token2wav to synthesize waveform.
        4) Return text hidden states (and audio when applicable).
        """
        if self.model_stage == "fused_thinker_talker":
            next_speech_tokens, text_hidden_states = self.generate_codes(
                input_ids=input_ids,
                positions=positions,
                inputs_embeds=inputs_embeds,
                intermediate_tensors=intermediate_tensors,
                **kwargs,
            )

            return OmniOutput(
                text_hidden_states=text_hidden_states.reshape(-1, text_hidden_states.shape[-1]),
                multimodal_outputs={"code_predictor_codes": next_speech_tokens},
            )

        if self.model_stage == "code2wav":
            code = (
                input_ids  # tensor [seq_len]
                if input_ids is not None
                else torch.zeros(
                    inputs_embeds.shape[0],
                    dtype=torch.long,
                    device=inputs_embeds.device,
                )
            )

            audio_tensor = self.generate_audio(code)
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "model_outputs": audio_tensor.reshape(1, -1) if audio_tensor is not None else audio_tensor
                },
            )

    def generate_codes(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        **kwargs: object,
    ):
        llm_dev = self._module_device(self.fused_thinker_talker)

        # if input_ids is None, set it to a zero tensor, in the length of the
        # same as the embedding seq length
        if input_ids is None:
            input_ids = torch.zeros(inputs_embeds.shape[1], dtype=torch.long, device=llm_dev).unsqueeze(0)  # (1, 0)

        # 1) Thinker (ensure inputs on thinker's device)
        if input_ids is not None and input_ids.device != llm_dev:
            input_ids = input_ids.to(llm_dev)
        if positions is not None and positions.device != llm_dev:
            positions = positions.to(llm_dev)
        if inputs_embeds is not None and inputs_embeds.device != llm_dev:
            inputs_embeds = inputs_embeds.to(llm_dev)

        # Run llm
        llm_output = self.fused_thinker_talker(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        next_speech_tokens = None
        if isinstance(llm_output, tuple):
            if len(llm_output) == 2:
                next_speech_tokens, text_hidden_states = llm_output
            elif len(llm_output) == 3:
                ids, next_speech_tokens, text_hidden_states = llm_output
        else:
            text_hidden_states = llm_output

        return next_speech_tokens, text_hidden_states

    def generate_audio(self, code: torch.Tensor):
        token2wav_dev = self._module_device(self.token2wav)
        # Check if in CUDA graph capture phase
        is_capturing = torch.cuda.is_current_stream_capturing()

        if isinstance(code, torch.Tensor):
            if is_capturing:
                # During CUDA graph capture, avoid device movement operations
                # Assume tensor is already on the correct device, only convert dtype
                code_tensor = code.to(dtype=torch.long)
            else:
                # During non-capture phase, normally perform device movement and type conversion
                code_tensor = code.to(dtype=torch.long, device=token2wav_dev)
        else:
            # If code is not a Tensor, should avoid creating new tensors during capture
            # This case should be rare during capture, as code usually comes from input_ids
            if is_capturing:
                # During capture, if code is not a Tensor, try to use current stream device
                # But this case should theoretically not happen
                code_tensor = torch.as_tensor(code, dtype=torch.long)
            else:
                code_tensor = torch.as_tensor(code, dtype=torch.long, device=token2wav_dev)
        if code_tensor.ndim == 2 and code_tensor.shape[0] == 1:
            code_tensor = code_tensor.squeeze(0)

        with torch.inference_mode():
            audio_tensor = self.token2wav(codes=code_tensor)  # code_tensor tensor [seq_len]

        return audio_tensor

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata | None = None,
    ) -> torch.Tensor | None:
        # logits = self.logits_processor(self.lm_head, hidden_states,
        #                                sampling_metadata)
        # Handle OmniOutput type
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states

        # Use thinker model for logits computation
        return self.model.compute_logits(hidden_states)
