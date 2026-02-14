# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Inference-only Qwen3-Omni-Moe model (thinker part)."""

from collections.abc import Iterable, Iterator, Mapping, Sequence
from functools import partial
from typing import Any, Literal, cast

import numpy as np
import torch
import torch.nn as nn
from packaging.version import Version
from transformers import PretrainedConfig
from transformers import __version__ as TRANSFORMERS_VERSION
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeConfig,
    Qwen3OmniMoeThinkerConfig,
)
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
    Qwen3OmniMoeProcessor,
)
from transformers.models.whisper import WhisperFeatureExtractor
from vllm.compilation.decorators import support_torch_compile
from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.distributed import get_pp_group
from vllm.inputs.data import PromptType
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsPP,
    SupportsTranscription,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.qwen2_5_omni_thinker import (
    Qwen2_5OmniAudioFeatureInputs,
    Qwen2_5OmniThinkerDummyInputsBuilder,
    Qwen2_5OmniThinkerMultiModalProcessor,
    check_interleaved_audio_video,
    merge_interleaved_embeddings,
)
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VLProcessingInfo,
)
from vllm.model_executor.models.qwen2_audio import Qwen2AudioProcessingInfo
from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM
from vllm.model_executor.models.qwen3_moe import Qwen3MoeModel as _Qwen3MoeLLMModel
from vllm.model_executor.models.qwen3_omni_moe_thinker import (
    Qwen3Omni_VisionTransformer,
    Qwen3OmniMoeAudioEncoder,
    _get_feat_extract_output_lengths,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    _merge_multimodal_embeddings,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFeatureSpec, MultiModalKwargsItems
from vllm.multimodal.parse import AudioProcessorItems, MultiModalDataItems
from vllm.multimodal.processing.processor import (
    MultiModalPromptUpdates,
    PlaceholderFeaturesInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.processor import cached_processor_from_config

from vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_thinker import (
    Qwen2_5OmniConditionalGenerationMixin,
)

try:
    import flash_attn
except (ImportError, ModuleNotFoundError):
    flash_attn = None

logger = init_logger(__name__)

# Speech input languages supported by Qwen3-Omni
# From: https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct
ISO639_1_SUPPORTED_LANGS = {
    "en": "English",
    "zh": "Chinese",
    "ko": "Korean",
    "ja": "Japanese",
    "de": "German",
    "ru": "Russian",
    "it": "Italian",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "ms": "Malay",
    "nl": "Dutch",
    "id": "Indonesian",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "yue": "Cantonese",
    "ar": "Arabic",
    "ur": "Urdu",
}


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "deepstack_input_embeds": 0,
    }
)
class Qwen3MoeLLMModel(_Qwen3MoeLLMModel):
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        capture_layer_indices: Sequence[int] | None = None,
        return_hidden_states: bool = False,
        deepstack_input_embeds: IntermediateTensors | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        capture_set = set(capture_layer_indices) if capture_layer_indices else None
        captured_hidden_states: dict[str, torch.Tensor] | None = {} if return_hidden_states else None

        for layer_idx, layer in enumerate(self.layers[self.start_layer : self.end_layer]):
            layer_idx = layer_idx + self.start_layer

            if captured_hidden_states is not None and capture_set is not None:
                if layer_idx in capture_set:
                    captured_hidden_states[str(layer_idx)] = hidden_states.clone().view(-1, hidden_states.shape[-1])

            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )

            if deepstack_input_embeds is not None and layer_idx in range(0, len(deepstack_input_embeds)):
                hidden_states = hidden_states + deepstack_input_embeds[f"deepstack_input_embeds_{layer_idx}"]

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})
        hidden_states, _ = self.norm(hidden_states, residual)
        if captured_hidden_states is not None:
            return hidden_states, captured_hidden_states
        else:
            return hidden_states, None


class Qwen3MoeLLMForCausalLM(Qwen3MoeForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Qwen3MoeForCausalLM, self).__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = Qwen3MoeLLMModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, quant_config=quant_config)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors


class Qwen3OmniMoeThinkerProcessingInfo(Qwen2AudioProcessingInfo, Qwen2_5_VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen3OmniMoeConfig).thinker_config

    def get_hf_processor(self, **kwargs: object) -> Qwen3OmniMoeProcessor:
        processor = self.ctx.get_hf_processor(
            Qwen3OmniMoeProcessor,
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )
        if not hasattr(processor, "audio_token"):
            processor.audio_token = "<|audio_pad|>"
        if not hasattr(processor, "image_token"):
            processor.image_token = "<|image_pad|>"
        if not hasattr(processor, "video_token"):
            processor.video_token = "<|video_pad|>"
        return processor

    def get_feature_extractor(self, **kwargs: object):
        hf_processor = self.get_hf_processor(**kwargs)
        feature_extractor = hf_processor.feature_extractor  # type: ignore
        assert isinstance(feature_extractor, WhisperFeatureExtractor)
        return feature_extractor

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None, "image": None, "video": None}


Qwen3OmniMoeThinkerDummyInputsBuilder = Qwen2_5OmniThinkerDummyInputsBuilder


class Qwen3OmniMoeThinkerMultiModalProcessor(
    Qwen2_5OmniThinkerMultiModalProcessor,
):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        mm_data = dict(mm_data)
        audios = mm_data.pop("audios", [])

        def pad_to_hop_length(x: np.ndarray, hop_length: int) -> np.ndarray:
            length = x.shape[-1]
            if length % hop_length != 0:
                pad_length = hop_length - (length % hop_length)
                x = np.pad(x, (0, pad_length), mode="constant", constant_values=0)
            return x

        # NOTE: WhisperFeatureExtractor cannot handle empty list of audios
        feature_extractor = self.info.get_feature_extractor()
        hop_length = feature_extractor.hop_length
        if audios:
            # NOTE: Qwen3-Omni processor accept "audio"
            # To make sure the cache works with padding=True, we pre-padded
            # the audio to multiple of hop_length.
            mm_data["audio"] = [
                pad_to_hop_length(audio, hop_length)
                if isinstance(audio, np.ndarray)
                else (pad_to_hop_length(audio[0], hop_length), audio[1])
                for audio in audios
            ]

            # TODO(Isotr0py): Remove this patch after upstream fix PR
            # released and Transformers version update:
            # https://github.com/huggingface/transformers/pull/41473
            mm_kwargs = dict(mm_kwargs)
            tok_kwargs = dict(tok_kwargs)
            mm_kwargs["audio_kwargs"] = dict(mm_kwargs.get("audio_kwargs") or {})
            mm_kwargs["text_kwargs"] = dict(mm_kwargs.get("text_kwargs") or {})
            if Version(TRANSFORMERS_VERSION) < Version("4.58.0"):
                # Extract audio_sample_rate before restructuring
                audio_sample_rate = mm_kwargs.pop("audio_sample_rate", None)

                # move truncation to audio_kwargs level to avoid conflict
                # with tok_kwargs
                mm_kwargs["audio_kwargs"].setdefault("truncation", mm_kwargs.pop("truncation", False))
                mm_kwargs["text_kwargs"].setdefault("truncation", tok_kwargs.pop("truncation", False))

                # Validate and conditionally pass audio_sample_rate
                # WhisperFeatureExtractor has a fixed sampling rate, and vLLM's
                # audio loader already resamples audio to the target rate.
                # Only pass the value if it matches to avoid unexpected behavior.
                if audio_sample_rate is not None:
                    expected_sr = feature_extractor.sampling_rate
                    if audio_sample_rate != expected_sr:
                        logger.warning(
                            "[%s] audio_sample_rate mismatch: user provided %dHz "
                            "but model expects %dHz. Ignoring user value. "
                            "vLLM's audio loader already resampled to %dHz.",
                            self.__class__.__name__,
                            audio_sample_rate,
                            expected_sr,
                            expected_sr,
                        )
                    else:
                        # Sample rate matches, safe to pass
                        mm_kwargs["audio_kwargs"]["audio_sample_rate"] = audio_sample_rate

        hf_inputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

        if (
            "audio_feature_lengths" in hf_inputs
            and "feature_attention_mask" in hf_inputs
            and (audios := mm_data.get("audio", []))
        ):
            audio_num_frames = []
            for _, audio in enumerate(audios):
                audio_length = len(audio[0]) if isinstance(audio, tuple) else len(audio)
                num_frame = (
                    (audio_length // hop_length) if audio_length % hop_length == 0 else (audio_length // hop_length - 1)
                )
                if mm_kwargs.get("truncation", False):
                    num_frame = min(num_frame, feature_extractor.n_samples // hop_length)
                audio_num_frames.append(num_frame)
            hf_inputs["feature_attention_mask"] = [torch.ones(num_frame) for num_frame in audio_num_frames]
            hf_inputs["audio_feature_lengths"] = torch.tensor(audio_num_frames)
        return hf_inputs

    def _maybe_apply_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        prompt_ids: list[int],
        mm_kwargs: MultiModalKwargsItems,
        mm_prompt_updates: MultiModalPromptUpdates,
        is_update_applied: bool,
    ) -> tuple[list[int], str, Mapping[str, list[PlaceholderFeaturesInfo]]]:
        """
        Qwen3-Omni reimplements this function to handle `use_audio_in_video`.
        """
        mm_item_counts = mm_items.get_all_counts()
        self._validate_mm_kwargs(mm_kwargs, mm_item_counts)

        use_audio_in_video = False
        if "video" in mm_kwargs:
            for item in mm_kwargs["video"]:
                if item and item["use_audio_in_video"].data:
                    use_audio_in_video = True
                else:
                    use_audio_in_video = False

        # normal case with `use_audio_in_video=False`
        if is_update_applied:
            mm_placeholders = self._find_mm_placeholders(
                prompt_ids,
                mm_prompt_updates,
            )
            self._validate_mm_placeholders(
                mm_placeholders,
                mm_item_counts,
            )
        else:
            if use_audio_in_video:
                # When use_audio_in_video=True, audio is extracted from video and embedded
                # in the video placeholder tokens. We should:
                # 1. Filter out audio from prompt updates (audio has no separate placeholder)
                # 2. Apply remaining updates (video, image, etc.)
                # 3. Derive audio placeholders from video placeholders
                filtered_updates = {k: v for k, v in mm_prompt_updates.items() if k != "audio"}
                prompt_ids, mm_placeholders = self._apply_prompt_updates(
                    prompt_ids,
                    filtered_updates,
                )
                # Derive audio placeholders from video placeholders
                mm_placeholders = self._derive_audio_from_video_placeholders(mm_placeholders, mm_item_counts)
            else:
                prompt_ids, mm_placeholders = self._apply_prompt_updates(
                    prompt_ids,
                    mm_prompt_updates,
                )

            self._validate_mm_placeholders(
                mm_placeholders,
                mm_item_counts,
            )

        return prompt_ids, mm_placeholders

    def get_updates_use_audio_in_video(
        self,
        thinker_config: PretrainedConfig,
        audio_len: int,
        video_grid_thw: list[int] | torch.Tensor,
        video_second_per_grid_t: float,
    ) -> list[int]:
        shift = 0
        audio_token_id = thinker_config.audio_token_id
        video_token_id = thinker_config.video_token_id
        audio_start_token_id = thinker_config.audio_start_token_id
        audio_end_token_id = thinker_config.audio_end_token_id
        spatial_merge_size = thinker_config.vision_config.spatial_merge_size
        position_id_per_seconds = thinker_config.position_id_per_seconds
        audio_token_indices = np.arange(next(iter([audio_len])))
        curr_video_grid_thw = next(iter([video_grid_thw]))
        height = curr_video_grid_thw[1] // spatial_merge_size
        width = curr_video_grid_thw[2] // spatial_merge_size
        video_token_indices = np.arange(curr_video_grid_thw[0]).reshape(-1, 1, 1)
        video_token_indices = np.broadcast_to(
            video_token_indices, (video_token_indices.shape[0], height, width)
        ).reshape(-1)
        video_token_indices = (
            (video_token_indices + shift) * next(iter([video_second_per_grid_t])) * position_id_per_seconds
        )
        video_data_index, audio_data_index = 0, 0
        updates = [audio_start_token_id]
        while video_data_index < len(video_token_indices) and audio_data_index < len(audio_token_indices):
            if video_token_indices[video_data_index] <= audio_token_indices[audio_data_index]:
                updates += [video_token_id]
                video_data_index += 1
            else:
                updates += [audio_token_id]
                audio_data_index += 1
        if video_data_index < len(video_token_indices):
            updates += [video_token_id] * (len(video_token_indices) - video_data_index)
        if audio_data_index < len(audio_token_indices):
            updates += [audio_token_id] * (len(audio_token_indices) - audio_data_index)
        updates += [audio_end_token_id]
        return updates

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
        vocab = tokenizer.get_vocab()

        audio_token = processor.audio_token
        image_token = processor.image_token
        video_token = processor.video_token
        audio_token_id = vocab[audio_token]
        image_token_id = vocab[image_token]
        video_token_id = vocab[video_token]

        out_mm_data = out_mm_kwargs.get_data()
        audio_feature_lengths = out_mm_data.get("audio_feature_lengths")
        feature_attention_mask = out_mm_data.get("feature_attention_mask")
        if audio_feature_lengths is None and feature_attention_mask is None:
            audio_output_lengths = []
        elif audio_feature_lengths is not None:
            audio_output_lens = _get_feat_extract_output_lengths(audio_feature_lengths)
            audio_output_lengths = audio_output_lens.tolist()
        elif feature_attention_mask is not None:
            assert isinstance(feature_attention_mask, torch.Tensor)
            audio_output_lens = _get_feat_extract_output_lengths(feature_attention_mask.sum(-1))
            audio_output_lengths = audio_output_lens.tolist()

        # number of audios read from video.
        audio_in_video_item_idx = 0
        audio_item_idx = 0

        def get_replacement_qwen2_audio(item_idx: int):
            nonlocal audio_item_idx
            item_idx += audio_in_video_item_idx

            audio_item_idx += 1

            num_features = audio_output_lengths[item_idx]
            if num_features == 0:
                audios = mm_items.get_items("audio", AudioProcessorItems)
                audio = audios.get(item_idx)
                raise ValueError(
                    f"The audio {audio} (len={len(audio)}) is too short to be represented inside the model"
                )

            return [audio_token_id] * num_features

        def get_replacement_qwen2_vision(item_idx: int, modality: str):
            grid_thw = out_mm_data[f"{modality}_grid_thw"][item_idx]
            assert isinstance(grid_thw, torch.Tensor)
            merge_length = image_processor.merge_size**2

            token_id = image_token_id if modality == "image" else video_token_id
            return [token_id] * (int(grid_thw.prod()) // merge_length)

        use_audio_in_video = hf_processor_mm_kwargs.get("use_audio_in_video", False)
        thinker_config = self.info.get_hf_config()

        def get_replacement_qwen2_use_audio_in_video(item_idx: int):
            nonlocal audio_in_video_item_idx
            audio_num_features = audio_output_lengths[audio_in_video_item_idx + item_idx]
            video_grid_thw = out_mm_data["video_grid_thw"][item_idx]

            audio_in_video_item_idx += 1

            second_per_grid_ts = hf_processor_mm_kwargs.get("second_per_grid_ts", None)
            if second_per_grid_ts:
                video_second_per_grid_t = second_per_grid_ts[item_idx]
            else:
                video_second_per_grid_t = 2.0

            placeholder = self.get_updates_use_audio_in_video(
                thinker_config=thinker_config,
                audio_len=audio_num_features,
                video_grid_thw=video_grid_thw,
                video_second_per_grid_t=video_second_per_grid_t,
            )
            return PromptUpdateDetails.select_token_id(placeholder, embed_token_id=video_token_id)

        video_replacement_fn = (
            get_replacement_qwen2_use_audio_in_video
            if use_audio_in_video
            else partial(get_replacement_qwen2_vision, modality="video")
        )

        return [
            PromptReplacement(
                modality="audio",
                target=audio_token,
                replacement=get_replacement_qwen2_audio,
            ),
            PromptReplacement(
                modality="image",
                target=image_token,
                replacement=partial(get_replacement_qwen2_vision, modality="image"),
            ),
            PromptReplacement(
                modality="video",
                target=video_token,
                replacement=video_replacement_fn,
            ),
        ]

    def _derive_audio_from_video_placeholders(
        self,
        placeholders: Mapping[str, list[PlaceholderFeaturesInfo]],
        mm_item_counts: Mapping[str, int],
    ) -> Mapping[str, list[PlaceholderFeaturesInfo]]:
        """
        Helper to derive audio placeholders from video placeholders when
        use_audio_in_video=True.

        In use_audio_in_video mode, audio is extracted from video and embedded
        within the video placeholder tokens. This function creates audio placeholder
        info by extracting the audio token positions from video placeholders.

        Args:
            placeholders: Current placeholders (should contain "video")
            mm_item_counts: Counts of multimodal items from mm_items.get_all_counts()
        """
        if "video" not in placeholders:
            return placeholders

        # Validate audio and video counts match
        # In use_audio_in_video mode, audio count comes from mm_items (extracted from video)
        num_videos = len(placeholders["video"])
        num_audios = mm_item_counts.get("audio", 0)
        if num_audios != num_videos:
            raise ValueError(
                f"use_audio_in_video requires equal number of audio and video items, got {num_audios=}, {num_videos=}"
            )

        tokenizer = self.info.get_tokenizer()
        processor = self.info.get_hf_processor()
        audio_token_id = tokenizer.get_vocab()[processor.audio_token]

        result_placeholders = dict(placeholders)
        audio_placeholders = []

        # Each video is paired with one audio
        for video_idx, video_placeholder in enumerate(placeholders["video"]):
            # Create is_embed mask selecting only audio tokens
            audio_is_embed = torch.tensor(video_placeholder.tokens) == audio_token_id

            audio_placeholder = PlaceholderFeaturesInfo(
                modality="audio",
                item_idx=video_idx,
                start_idx=video_placeholder.start_idx,
                tokens=video_placeholder.tokens,
                is_embed=audio_is_embed,
            )
            audio_placeholders.append(audio_placeholder)

        result_placeholders["audio"] = audio_placeholders
        return result_placeholders

    def _get_raw_input_ids(
        self,
        token_ids: list[int],
        use_audio_in_video: bool = False,
    ) -> list[int]:
        tokenizer = self.info.get_tokenizer()
        vision_bos_token = tokenizer.encode(tokenizer.vision_bos_token)[0]
        vision_eos_token = tokenizer.encode(tokenizer.vision_eos_token)[0]
        audio_bos_token = tokenizer.encode(tokenizer.audio_bos_token)[0]
        audio_eos_token = tokenizer.encode(tokenizer.audio_eos_token)[0]
        audio_token = tokenizer.encode("<|audio_pad|>")[0]
        image_token = tokenizer.encode("<|image_pad|>")[0]
        video_token = tokenizer.encode("<|video_pad|>")[0]

        result = token_ids[:]
        if use_audio_in_video:
            while True:
                start = None
                for i in range(len(result) - 1):
                    if result[i : i + 2] == [vision_bos_token, audio_bos_token]:
                        start = i
                        break
                if start is not None:
                    end = None
                    for i in range(start + 2, len(result) - 1):
                        if result[i : i + 2] == [audio_eos_token, vision_eos_token]:
                            end = i
                            break
                    if end is not None:
                        result = result[:start] + [vision_bos_token, video_token, vision_eos_token] + result[end + 2 :]
                else:
                    break

        for mm_token in [audio_token, image_token, video_token]:
            compressed = []
            for x in result:
                if x != mm_token or (not compressed or compressed[-1] != mm_token):
                    compressed.append(x)
            result = compressed

        return result


class Qwen3OmniMoeConditionalGenerationMixin(Qwen2_5OmniConditionalGenerationMixin):
    def _process_audio_input(
        self,
        audio_input: Qwen2_5OmniAudioFeatureInputs,
        audio_hashes: list[str] | None = None,
        cached_audio_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        input_features = audio_input["input_features"]
        audio_feature_lengths = audio_input["audio_feature_lengths"]

        audio_output_lengths = _get_feat_extract_output_lengths(audio_feature_lengths)

        audio_outputs = self.audio_tower(
            input_features.to(self.audio_tower.dtype),
            feature_lens=audio_feature_lengths,
            aftercnn_lens=audio_output_lengths,
        )
        # OMNI: audio_tower.forward() returns hidden_states tensor directly
        audio_features = audio_outputs
        return audio_features.split(audio_output_lengths.tolist())


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3OmniMoeThinkerMultiModalProcessor,
    info=Qwen3OmniMoeThinkerProcessingInfo,
    dummy_inputs=Qwen3OmniMoeThinkerDummyInputsBuilder,
)
class Qwen3OmniMoeThinkerForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsMRoPE,
    Qwen3OmniMoeConditionalGenerationMixin,
    SupportsTranscription,
):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "thinker.lm_head.": "language_model.lm_head.",
            "thinker.model.": "language_model.model.",
            "thinker.": "",
        }
    )

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

    supported_languages = ISO639_1_SUPPORTED_LANGS

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|video_pad|><|vision_end|>"
        if modality.startswith("audio"):
            return "<|audio_start|><|audio_pad|><|audio_end|>"

        raise ValueError("Only image, video or audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config  # needed for torch compile forward context
        thinker_config: Qwen3OmniMoeThinkerConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = thinker_config
        self.multimodal_config = multimodal_config
        self.quant_config = quant_config

        with self._mark_tower_model(vllm_config, "audio"):
            self.audio_tower = Qwen3OmniMoeAudioEncoder(
                thinker_config.audio_config,
                prefix=maybe_prefix(prefix, "audio_tower"),
            )

        self.use_deepstack = hasattr(thinker_config.vision_config, "deepstack_visual_indexes")
        self.deepstack_num_level = (
            len(thinker_config.vision_config.deepstack_visual_indexes) if self.use_deepstack else 0
        )
        self.visual_dim = thinker_config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen3Omni_VisionTransformer(
                vision_config=thinker_config.vision_config,
                norm_eps=getattr(thinker_config.text_config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )

            # register buffer for deepstack
            if self.use_deepstack:
                self.deepstack_input_embeds = [
                    torch.zeros(
                        vllm_config.scheduler_config.max_num_batched_tokens,
                        thinker_config.text_config.hidden_size,
                    )
                    for _ in range(self.deepstack_num_level)
                ]

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3MoeLLMForCausalLM(
                vllm_config=vllm_config.with_hf_config(
                    thinker_config.text_config,
                    architectures=["Qwen3MoeForCausalLM"],
                ),
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

    def _get_deepstack_input_embeds(
        self,
        num_tokens: int,
    ) -> IntermediateTensors | None:
        if not getattr(self, "deepstack_input_embeds", None):
            return None  # If vision tower is skipped

        # get deepstack_input_embeds from buffer, and clear the buffer
        return IntermediateTensors(
            {
                f"deepstack_input_embeds_{idx}": self.deepstack_input_embeds[idx][:num_tokens]
                for idx in range(self.deepstack_num_level)
            }
        )

    def _set_deepstack_input_embeds(self, deepstack_input_embeds: torch.Tensor) -> None:
        if not getattr(self, "deepstack_input_embeds", None):
            return

        # set deepstack_input_embeds to buffer
        num_tokens = deepstack_input_embeds.size(1)
        if num_tokens > self.deepstack_input_embeds[0].size(0):
            self.deepstack_input_embeds = [
                torch.zeros(
                    num_tokens,
                    self.config.text_config.hidden_size,
                    device=self.deepstack_input_embeds[0].device,
                    dtype=self.deepstack_input_embeds[0].dtype,
                )
                for _ in range(self.deepstack_num_level)
            ]
        for idx in range(self.deepstack_num_level):
            self.deepstack_input_embeds[idx][:num_tokens].copy_(deepstack_input_embeds[idx])

    def _clear_deepstack_input_embeds(self, num_tokens: int) -> None:
        if not getattr(self, "deepstack_input_embeds", None):
            return

        # clear deepstack_input_embeds in buffer
        if num_tokens > 0:
            for idx in range(self.deepstack_num_level):
                self.deepstack_input_embeds[idx][:num_tokens].zero_()

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        mm_input_by_modality = {}

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if input_key in ("pixel_values", "image_embeds") and "image" not in mm_input_by_modality:
                mm_input_by_modality["image"] = self._parse_and_validate_image_input(**kwargs)
            if input_key in ("pixel_values_videos", "video_embeds") and "video" not in mm_input_by_modality:
                mm_input_by_modality["video"] = self._parse_and_validate_video_input(**kwargs)
            if input_key in ("input_audio_features",) and "audio" not in mm_input_by_modality:
                mm_input_by_modality["audio"] = self._parse_and_validate_audio_input(**kwargs)
        return mm_input_by_modality

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            return []

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor correspoending to a multimodal data item (image or video).
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "image":
                image_embeddings = self._process_image_input(multimodal_input)
                multimodal_embeddings += tuple(image_embeddings)
            if modality == "video":
                video_embeddings = self._process_video_input(multimodal_input)
                multimodal_embeddings += tuple(video_embeddings)
            if modality == "audio":
                audio_embeddings = self._process_audio_input(multimodal_input)
                multimodal_embeddings += tuple(audio_embeddings)
        return multimodal_embeddings

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.language_model.embed_input_ids,
            is_multimodal=is_multimodal,
            handle_oov_mm_token=handle_oov_mm_token,
        )

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        # Detect interleaved audio-in-video early, since it affects
        # both the deepstack path and the final embedding merge.
        video_token_id = self.config.video_token_id
        audio_token_id = self.config.audio_token_id
        is_video = is_multimodal & (input_ids == video_token_id)
        is_audio = is_multimodal & (input_ids == audio_token_id)
        num_video = is_video.sum().item()
        num_audio = is_audio.sum().item()

        is_interleaved = check_interleaved_audio_video(is_video, is_audio, num_video, num_audio)

        deepstack_input_embeds = None
        # split the feat dim to obtain multi-scale visual feature
        has_vision_embeddings = [
            embeddings.shape[-1] != self.config.text_config.hidden_size for embeddings in multimodal_embeddings
        ]
        if self.visual.deepstack_visual_indexes is not None and any(has_vision_embeddings):
            multiscale_len = len(self.visual.deepstack_visual_indexes)
            multimodal_embeddings_multiscale = []

            if is_interleaved:
                # Use input_ids-based mask for correct vision positions
                # when audio and video tokens are interleaved.
                is_vision = is_video.clone()
            else:
                is_vision = torch.zeros_like(is_multimodal)
                mm_positions = torch.nonzero(is_multimodal, as_tuple=True)[0]
                mm_position_idx = 0

            for index, embeddings in enumerate(multimodal_embeddings):
                num_tokens = embeddings.shape[0]

                # Vision embeddings
                if embeddings.shape[-1] != self.config.text_config.hidden_size:
                    visual_dim = embeddings.shape[-1] // (multiscale_len + 1)
                    multi_dim = visual_dim * multiscale_len
                    embeddings_main, embeddings_multiscale = torch.split(embeddings, [visual_dim, multi_dim], dim=-1)
                    multimodal_embeddings[index] = embeddings_main
                    multimodal_embeddings_multiscale.append(embeddings_multiscale)
                    if not is_interleaved:
                        current_positions = mm_positions[mm_position_idx : mm_position_idx + num_tokens]
                        is_vision[current_positions] = True

                # Audio embeddings
                else:
                    if not is_interleaved:
                        current_positions = mm_positions[mm_position_idx : mm_position_idx + num_tokens]
                        is_vision[current_positions] = False

                if not is_interleaved:
                    mm_position_idx += num_tokens

            deepstack_input_embeds = inputs_embeds.new_zeros(
                inputs_embeds.size(0), multiscale_len * inputs_embeds.size(1)
            )
            deepstack_input_embeds = _merge_multimodal_embeddings(
                inputs_embeds=deepstack_input_embeds,
                multimodal_embeddings=multimodal_embeddings_multiscale,
                is_multimodal=is_vision,
            )
            deepstack_input_embeds = (
                deepstack_input_embeds.view(inputs_embeds.shape[0], multiscale_len, visual_dim)
                .permute(1, 0, 2)
                .contiguous()
            )
            self._set_deepstack_input_embeds(deepstack_input_embeds)

        if is_interleaved:
            return merge_interleaved_embeddings(
                inputs_embeds,
                multimodal_embeddings,
                is_video,
                is_audio,
                is_multimodal,
                num_video,
                num_audio,
            )

        # Default: standard merge (no interleaving)
        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        capture_layer_indices: Sequence[int] | None = None,
        return_hidden_states: bool = False,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        if self.use_deepstack and inputs_embeds is not None and get_pp_group().is_first_rank:
            deepstack_input_embeds = self._get_deepstack_input_embeds(inputs_embeds.size(0))
        else:
            deepstack_input_embeds = None

        hidden_states, captured_hidden_states = self.language_model.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
            capture_layer_indices=capture_layer_indices,
            return_hidden_states=return_hidden_states,
            # args for deepstack
            deepstack_input_embeds=deepstack_input_embeds,
        )

        if inputs_embeds is not None and get_pp_group().is_first_rank:
            self._clear_deepstack_input_embeds(inputs_embeds.size(0))

        return hidden_states, captured_hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["talker.", "code2wav."],
        )
        loaded_weights = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

        return loaded_weights

    def _compute_audio_token_count(self, audio_feature_length: int) -> int:
        """Compute audio tokens from feature length using Qwen3-Omni formula."""
        return _get_feat_extract_output_lengths(torch.tensor([audio_feature_length])).item()

    def _get_audio_for_video_mapping(self, mm_features: list[MultiModalFeatureSpec]) -> tuple[dict[int, int], set[int]]:
        """
        Map video offset -> paired audio_feature_length for use_audio_in_video.

        When use_audio_in_video=True, audio is interleaved within video.
        The pairing is based on feature order in mm_features.

        Returns:
            Tuple of (video_offset -> audio_feature_length mapping,
                      set of paired audio offsets to skip)
        """
        videos_with_audio = [
            f
            for f in mm_features
            if f.modality == "video" and f.data.get("use_audio_in_video") and f.data["use_audio_in_video"].data.item()
        ]
        audios = [f for f in mm_features if f.modality == "audio"]

        mapping: dict[int, int] = {}
        paired_audio_offsets: set[int] = set()
        for i, video_f in enumerate(videos_with_audio):
            if i < len(audios):
                audio_len = audios[i].data["audio_feature_lengths"].data.item()
                mapping[video_f.mm_position.offset] = audio_len
                paired_audio_offsets.add(audios[i].mm_position.offset)
        return mapping, paired_audio_offsets

    def iter_mm_features(self, mm_features: list[MultiModalFeatureSpec]) -> Iterator[tuple[int, str, dict[str, Any]]]:
        """
        Iterate over multimodal features sorted by position offset.

        Yields: (offset, modality, feature_data) where feature_data contains:
        - image: {"grid_t", "grid_h", "grid_w", "t_factor"}
        - video: {"grid_t", "grid_h", "grid_w", "t_factor",
                  "use_audio_in_video", "audio_feature_length"}
        - audio: {"audio_feature_length"}
        """
        config = self.config
        spatial_merge_size = config.vision_config.spatial_merge_size
        position_id_per_seconds = config.position_id_per_seconds

        sorted_features = sorted(mm_features, key=lambda f: f.mm_position.offset)
        audio_for_video, paired_audio_offsets = self._get_audio_for_video_mapping(sorted_features)

        for mm_feature in sorted_features:
            offset = mm_feature.mm_position.offset
            modality = mm_feature.modality

            if modality == "image":
                t, h, w = mm_feature.data["image_grid_thw"].data.tolist()
                yield (
                    offset,
                    "image",
                    {
                        "grid_t": t,
                        "grid_h": h // spatial_merge_size,
                        "grid_w": w // spatial_merge_size,
                        "t_factor": position_id_per_seconds,
                    },
                )
            elif modality == "video":
                t, h, w = mm_feature.data["video_grid_thw"].data.tolist()
                second_per_grid_ts = 2.0
                if mm_feature.data.get("second_per_grid_ts"):
                    second_per_grid_ts = mm_feature.data["second_per_grid_ts"].data.item()
                use_audio_in_video = bool(
                    mm_feature.data.get("use_audio_in_video") and mm_feature.data["use_audio_in_video"].data.item()
                )

                yield (
                    offset,
                    "video",
                    {
                        "grid_t": t,
                        "grid_h": h // spatial_merge_size,
                        "grid_w": w // spatial_merge_size,
                        "t_factor": second_per_grid_ts * position_id_per_seconds,
                        "use_audio_in_video": use_audio_in_video,
                        "audio_feature_length": audio_for_video.get(offset),
                    },
                )
            elif modality == "audio":
                if offset not in paired_audio_offsets:
                    audio_len = mm_feature.data["audio_feature_lengths"].data.item()
                    yield offset, "audio", {"audio_feature_length": audio_len}

    def _compute_interleaved_positions(self, start_idx: int, data: dict[str, Any]) -> tuple[np.ndarray, int]:
        """
        Compute positions for interleaved video+audio using Qwen3 token-by-token
        interleaving logic.

        Returns: (position_ids [3, N], total_token_count)
        """
        grid_t = data["grid_t"]
        grid_h = data["grid_h"]
        grid_w = data["grid_w"]
        t_factor = data["t_factor"]
        audio_feature_length = data["audio_feature_length"]

        audio_len = self._compute_audio_token_count(audio_feature_length)

        h_index = np.tile(np.arange(grid_h).reshape(1, -1, 1), (grid_t, 1, grid_w)).flatten()
        w_index = np.tile(np.arange(grid_w).reshape(1, 1, -1), (grid_t, grid_h, 1)).flatten()
        t_index_raw = np.arange(grid_t)
        t_index_scaled = (t_index_raw * t_factor).astype(np.int64)
        t_index = np.repeat(t_index_scaled, grid_h * grid_w)

        video_pos = np.stack([t_index, h_index, w_index]) + start_idx
        audio_pos = np.broadcast_to(np.arange(audio_len), (3, audio_len)) + start_idx

        video_t_values = video_pos[0]
        audio_t_values = audio_pos[0]

        pos_ids_list: list[np.ndarray] = []
        video_idx, audio_idx = 0, 0
        num_video = grid_t * grid_h * grid_w

        while video_idx < num_video and audio_idx < audio_len:
            if video_t_values[video_idx] <= audio_t_values[audio_idx]:
                pos_ids_list.append(video_pos[:, video_idx : video_idx + 1])
                video_idx += 1
            else:
                pos_ids_list.append(audio_pos[:, audio_idx : audio_idx + 1])
                audio_idx += 1

        if video_idx < num_video:
            pos_ids_list.append(video_pos[:, video_idx:])
        if audio_idx < audio_len:
            pos_ids_list.append(audio_pos[:, audio_idx:])

        total_tokens = num_video + audio_len
        return np.concatenate(pos_ids_list, axis=1), total_tokens

    @classmethod
    def get_speech_to_text_config(cls, model_config: ModelConfig, task_type: str) -> SpeechToTextConfig:
        processor = cached_processor_from_config(model_config, processor_cls=Qwen3OmniMoeProcessor)
        return SpeechToTextConfig(
            max_audio_clip_s=processor.feature_extractor.chunk_length,
            sample_rate=processor.feature_extractor.sampling_rate,
            min_energy_split_window_size=None,
        )

    @classmethod
    def get_generation_prompt(
        cls,
        audio: np.ndarray,
        stt_config: SpeechToTextConfig,
        model_config: ModelConfig,
        language: str | None,
        task_type: Literal["transcribe", "translate"],
        request_prompt: str,
        to_language: str | None,
    ) -> PromptType:
        """
        Construct a transcription/translation prompt for Qwen3-Omni.
        """
        # Transcribe this audio [into <language>] | for transcription
        # Translate this audio [from <language> into <to_language>] | for translation
        instruction = "Transcribe" if task_type == "transcribe" else "Translate"
        instruction += " this audio"

        # Default to_language to English for translation
        if task_type == "translate" and to_language is None:
            to_language = "en"

        # Get full language names from supported_languages mapping
        full_lang_name = cls.supported_languages.get(language, "")
        full_lang_name_to = cls.supported_languages.get(to_language, "")

        if task_type == "transcribe" and full_lang_name:
            instruction += f" into {full_lang_name}"
        elif task_type == "translate":
            if full_lang_name:
                instruction += f" from {full_lang_name}"
            if full_lang_name_to:
                instruction += f" into {full_lang_name_to}"

        instruction += "."

        if request_prompt:
            instruction += f" {request_prompt}"

        processor = cached_processor_from_config(model_config, processor_cls=Qwen3OmniMoeProcessor)
        # Audio placeholder format: <|audio_start|><|audio_pad|><|audio_end|>
        audio_placeholder = "<|audio_start|><|audio_pad|><|audio_end|>"
        user_content = f"{audio_placeholder}{instruction}"

        messages = [{"role": "user", "content": user_content}]
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        audio_data = (audio, stt_config.sample_rate)
        prompts_dict = {"multi_modal_data": {"audio": audio_data}, "prompt": prompt}
        return cast(PromptType, prompts_dict)

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        """Compute M-RoPE input positions using mm_features directly."""
        seq_len = len(input_tokens)

        llm_pos_ids_list: list[np.ndarray] = []
        st = 0

        for offset, modality, data in self.iter_mm_features(mm_features):
            text_len = offset - st
            st_idx = int(llm_pos_ids_list[-1].max()) + 1 if llm_pos_ids_list else 0

            if text_len > 0:
                llm_pos_ids_list.append(np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx)
                st_idx += text_len

            bos_pos = np.broadcast_to(np.array([st_idx]), (3, 1))
            llm_pos_ids_list.append(bos_pos)
            st_idx += 1

            if modality == "audio":
                audio_tokens = self._compute_audio_token_count(data["audio_feature_length"])
                audio_pos = np.broadcast_to(np.arange(audio_tokens), (3, audio_tokens)) + st_idx
                llm_pos_ids_list.append(audio_pos)
                st_idx = int(audio_pos.max()) + 1

                eos_pos = np.broadcast_to(np.array([st_idx]), (3, 1))
                llm_pos_ids_list.append(eos_pos)
                st = offset + 1 + audio_tokens + 1

            elif modality == "image":
                grid_t = data["grid_t"]
                grid_h = data["grid_h"]
                grid_w = data["grid_w"]
                t_factor = data["t_factor"]

                grid_indices = np.indices((grid_t, grid_h, grid_w))
                if t_factor != 1.0:
                    grid_indices[0] = (grid_indices[0] * t_factor).astype(np.int64)
                llm_pos_ids_list.append(grid_indices.reshape(3, -1) + st_idx)

                image_len = grid_t * grid_h * grid_w
                st_idx = int(llm_pos_ids_list[-1].max()) + 1

                eos_pos = np.broadcast_to(np.array([st_idx]), (3, 1))
                llm_pos_ids_list.append(eos_pos)
                st = offset + 1 + image_len + 1

            elif modality == "video":
                grid_t = data["grid_t"]
                grid_h = data["grid_h"]
                grid_w = data["grid_w"]
                t_factor = data["t_factor"]

                if not data["use_audio_in_video"]:
                    grid_indices = np.indices((grid_t, grid_h, grid_w))
                    if t_factor != 1.0:
                        grid_indices[0] = (grid_indices[0] * t_factor).astype(np.int64)
                    llm_pos_ids_list.append(grid_indices.reshape(3, -1) + st_idx)

                    video_len = grid_t * grid_h * grid_w
                    st_idx = int(llm_pos_ids_list[-1].max()) + 1

                    eos_pos = np.broadcast_to(np.array([st_idx]), (3, 1))
                    llm_pos_ids_list.append(eos_pos)
                    st = offset + 1 + video_len + 1
                else:
                    audio_bos_pos = np.broadcast_to(np.array([st_idx - 1]), (3, 1))
                    llm_pos_ids_list.append(audio_bos_pos)

                    pos_ids, _ = self._compute_interleaved_positions(st_idx, data)
                    llm_pos_ids_list.append(pos_ids)
                    st_idx = int(pos_ids.max()) + 1

                    eos_pos = np.broadcast_to(np.array([st_idx]), (3, 1))
                    llm_pos_ids_list.append(eos_pos)
                    llm_pos_ids_list.append(eos_pos)

                    video_len = grid_t * grid_h * grid_w
                    audio_len = self._compute_audio_token_count(data["audio_feature_length"])
                    st = offset + 2 + video_len + audio_len + 2

        if st < seq_len:
            st_idx = int(llm_pos_ids_list[-1].max()) + 1 if llm_pos_ids_list else 0
            text_len = seq_len - st
            llm_pos_ids_list.append(np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx)

        llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
        if llm_positions.shape[1] != seq_len:
            raise RuntimeError("Position ids length mismatch with input ids length")

        mrope_position_delta = int(llm_positions.max()) + 1 - seq_len
        return torch.from_numpy(llm_positions), mrope_position_delta

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="visual.merger",
            tower_model=["visual.", "audio_tower."],
        )
