# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
# Copyright 2024 ANT Group and the HuggingFace Inc. team.
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

from typing import Any

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoProcessor
from transformers.feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

DEFAULT_IMAGE_PATCH_TOKEN = "<imagePatch>"
DEFAULT_IM_START_TOKEN = "<image>"
DEFAULT_IM_END_TOKEN = "</image>"
DEFAULT_VID_START_TOKEN = "<video>"
DEFAULT_VID_END_TOKEN = "</video>"
DEFAULT_FRAME_PATCH_TOKEN = "<framePatch>"

DEFAULT_AUDIO_PATCH_TOKEN = "<audioPatch>"
DEFAULT_AU_START_TOKEN = "<audio>"
DEFAULT_AU_END_TOKEN = "</audio>"

# High-level placeholders used in user prompts
PLACEHOLDER_IMAGE_TOKEN_IN_TEXT = "<IMAGE>"
PLACEHOLDER_VIDEO_TOKEN_IN_TEXT = "<VIDEO>"
PLACEHOLDER_AUDIO_TOKEN_IN_TEXT = "<AUDIO>"

# Chat template constants
USER_PREFIX = "<role>HUMAN</role>"
ASSISTANT_PREFIX = "<role>ASSISTANT</role>"
SYSTEM_PROMPT_NOTHINK = "<role>SYSTEM</role>你是一个友好的AI助手。\n\ndetailed thinking off"
SYSTEM_PROMPT_THINK = "<role>SYSTEM</role>你是一个友好的AI助手。\n\ndetailed thinking on"


_NORM_FACTOR_FOR_DTYPE = {
    torch.int8: 2**7,
    torch.int16: 2**15,
    torch.int32: 2**31,
    torch.int64: 2**63,
    torch.float32: 1,
    torch.float64: 1,
}


def _normalize_audio_tensor(
    waveform: torch.Tensor,
    sample_rate: int,
    target_sample_rate: int = 16000,
) -> torch.Tensor:
    """Normalize waveform to float32, mono, and optionally resample."""
    norm_factor = _NORM_FACTOR_FOR_DTYPE.get(waveform.dtype, 1)
    waveform = waveform.to(torch.float32) / norm_factor

    # Remove channel dimension
    while len(waveform.shape) > 1:
        waveform = waveform[0]

    # Resample if needed
    if sample_rate != target_sample_rate:
        import torchaudio

        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform.unsqueeze(0)).squeeze(0)

    return waveform


class MingWhisperFeatureExtractor(FeatureExtractionMixin):
    """Whisper log-mel feature extractor for Ming-flash-omni-2.0.

    Produces audio_feats in the time-first packed format.

    Adapted from Ming's WhisperAudioEncoder
    https://github.com/inclusionAI/Ming/blob/070dc3c13f95d97952ab7d22030df0c9e28a5122/modeling_whisper_encoder.py
    and HF transformers WhisperFeatureExtractor
    https://github.com/huggingface/transformers/blob/f842abaca95a7dbf3fc6e16122e7409109bc1431/src/transformers/models/whisper/feature_extraction_whisper.py#L33
    """

    model_input_names = ["audio_feats", "audio_feats_lengths"]

    def __init__(self, feature_size: int = 128, sampling_rate: int = 16000, **kwargs):
        # feature_size == n_mels; stored so to_dict() serialises it correctly.
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        super().__init__(**kwargs)

    @property
    def n_mels(self) -> int:
        return self.feature_size

    def __call__(
        self,
        audios: tuple | list,
        return_tensors: str | None = None,
        **kwargs,
    ) -> BatchFeature:
        """Preprocess audio(s) into Whisper log-mel spectrograms"""
        import whisper

        if not isinstance(audios, list):
            audios = [audios]

        audio_feat_list = []
        for waveform, sr in audios:
            if isinstance(waveform, np.ndarray):
                waveform = torch.from_numpy(waveform)
            waveform = _normalize_audio_tensor(waveform, sr, target_sample_rate=self.sampling_rate)
            mel = whisper.log_mel_spectrogram(waveform, n_mels=self.n_mels)
            audio_feat_list.append(mel.transpose(0, 1))  # [T, n_mels]

        audio_feats_lengths = torch.tensor([[feat.shape[0] for feat in audio_feat_list]], dtype=torch.long)
        # Two stride-2 convolutions in series:
        #   1. WhisperAudioEncoder conv2: kernel=3, stride=2, padding=1
        #      (conv1 has stride=1 and does not change T)
        #   2. AudioProjector Conv1d: kernel=3, stride=2, padding=1
        # Combined: T → ((T-1)//2 + 1 - 1)//2 + 1
        # See also: AudioProjector.compute_output_length()
        encoder_feats_lengths = ((audio_feats_lengths - 3 + 2 * 1) // 2 + 1 - 3 + 2 * 1) // 2 + 1
        audio_feats = torch.cat(audio_feat_list, dim=0).unsqueeze(0)  # [1, T_total, n_mels]

        data = {
            # [1, T_total, n_mels], all audio clips concatenated
            "audio_feats": audio_feats.numpy(),
            # [1, n_audios], actual frame count
            "audio_feats_lengths": audio_feats_lengths.numpy(),
            # [1, n_audios]
            "encoder_feats_lengths": encoder_feats_lengths,
        }
        return BatchFeature(data=data, tensor_type=return_tensors)


class MingFlashOmniProcessor(ProcessorMixin):
    """Top-level multimodal processor for Ming-flash-omni 2.0.

    Adapted from Ming's BailingMM2Processor
    https://github.com/inclusionAI/Ming/blob/3954fcb880ff5e61ff128bcf7f1ec344d46a6fe3/processing_bailingmm2.py

    Subprocessors include:
    - Qwen2VLImageProcessor (image/video)
    - MingWhisperFeatureExtractor (modified audio processor using Whisper's log-mel spectrogram)
    """

    attributes = ["image_processor", "audio_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    audio_processor_class = "AutoFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        audio_processor=None,
        tokenizer=None,
        merge_size: int = 2,
        **kwargs,
    ):
        # Enforce that all sub-processors exist
        # Keep None defaults in the signature for HF ProcessorMixin compatibility
        if image_processor is None:
            raise ValueError("MingFlashOmniProcessor requires `image_processor`.")
        if audio_processor is None:
            raise ValueError("MingFlashOmniProcessor requires `audio_processor`.")
        if tokenizer is None:
            raise ValueError("MingFlashOmniProcessor requires `tokenizer`.")

        self.spatial_merge_size = merge_size
        self.image_token = PLACEHOLDER_IMAGE_TOKEN_IN_TEXT
        self.video_token = PLACEHOLDER_VIDEO_TOKEN_IN_TEXT
        self.audio_token = PLACEHOLDER_AUDIO_TOKEN_IN_TEXT
        super().__init__(
            image_processor=image_processor,
            audio_processor=audio_processor,
            tokenizer=tokenizer,
        )

        # Fall back to the tokenizer's own chat_template.
        if self.chat_template is None:
            self.chat_template = getattr(tokenizer, "chat_template", None)

    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput],
        images: Any | None = None,
        videos: Any | None = None,
        audios: tuple[np.ndarray, int] | list[tuple[np.ndarray, int]] | None = None,
        **kwargs,
    ) -> BatchFeature:
        # This should always be parallel implementations that mirror
        # `_get_prompt_updates` logic in Ming processor, and vice versa.
        # Ensure text is a list
        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list):
            raise ValueError("text must be a string or list of strings")

        data: dict[str, Any] = {}

        if images is not None:
            image_outputs = self.image_processor(
                images=images,
                videos=None,
                return_tensors="pt",
                **kwargs.get("images_kwargs", {}),
            )
            data.update(image_outputs)
            if "image_grid_thw" in image_outputs:
                text = self._expand_image_tokens(text, image_outputs["image_grid_thw"])

        if videos is not None:
            video_outputs = self.image_processor(
                images=None,
                videos=videos,
                return_tensors="pt",
                **kwargs.get("videos_kwargs", {}),
            )
            if "pixel_values" in video_outputs:
                video_outputs["pixel_values_videos"] = video_outputs.pop("pixel_values")
            if "image_grid_thw" in video_outputs:
                video_outputs["video_grid_thw"] = video_outputs.pop("image_grid_thw")
            data.update(video_outputs)
            if "video_grid_thw" in video_outputs:
                text = self._expand_video_tokens(text, video_outputs["video_grid_thw"])

        if audios is not None:
            audio_outputs = self.audio_processor(
                audios,
                return_tensors="pt",
                **kwargs.get("audio_kwargs", {}),
            )
            data.update(audio_outputs)
            if "encoder_feats_lengths" in audio_outputs:
                text = self._expand_audio_tokens(text, audio_outputs["encoder_feats_lengths"])

        text_outputs = self.tokenizer(
            text,
            return_tensors="pt",
            **kwargs.get("text_kwargs", {}),
        )
        data.update(text_outputs)
        return BatchFeature(data=data)

    def _expand_image_tokens(
        self,
        text: list[str],
        image_grid_thw: torch.Tensor,
        special_token: str = PLACEHOLDER_IMAGE_TOKEN_IN_TEXT,
    ) -> list[str]:
        merge_size = self.spatial_merge_size
        num_patches_per_image = torch.prod(image_grid_thw, dim=1) // (merge_size**2)
        prompt_strings = []
        image_index = 0
        for sample in text:
            num_images = sample.count(special_token)
            if num_images > 0:
                for i in range(image_index, num_images + image_index):
                    num_patches = int(num_patches_per_image[i].item())
                    img_text = (
                        DEFAULT_IM_START_TOKEN + (DEFAULT_IMAGE_PATCH_TOKEN * num_patches) + DEFAULT_IM_END_TOKEN + "\n"
                    )
                    sample = sample.replace(special_token, img_text, 1)
            image_index += num_images
            prompt_strings.append(sample)
        return prompt_strings

    def _expand_video_tokens(
        self,
        text: list[str],
        video_grid_thw: torch.Tensor,
        special_token: str = PLACEHOLDER_VIDEO_TOKEN_IN_TEXT,
    ) -> list[str]:
        merge_size = self.spatial_merge_size
        num_patches_per_video = torch.prod(video_grid_thw, dim=1) // (merge_size**2)
        prompt_strings = []
        video_index = 0
        for sample in text:
            num_videos = sample.count(special_token)
            if num_videos > 0:
                for i in range(video_index, num_videos + video_index):
                    num_patches = int(num_patches_per_video[i].item())
                    video_text = (
                        DEFAULT_VID_START_TOKEN
                        + (DEFAULT_FRAME_PATCH_TOKEN * num_patches)
                        + DEFAULT_VID_END_TOKEN
                        + "\n"
                    )
                    sample = sample.replace(special_token, video_text, 1)
            video_index += num_videos
            prompt_strings.append(sample)
        return prompt_strings

    def _expand_audio_tokens(
        self,
        text: list[str],
        encoder_feats_lengths: torch.Tensor,
        special_token: str = PLACEHOLDER_AUDIO_TOKEN_IN_TEXT,
    ) -> list[str]:
        prompt_strings = []
        for sample, lengths_tensor in zip(text, encoder_feats_lengths):
            for length in lengths_tensor:
                num_patches = int(length.item())
                if num_patches == 0:
                    continue
                audio_text = DEFAULT_AU_START_TOKEN + (DEFAULT_AUDIO_PATCH_TOKEN * num_patches) + DEFAULT_AU_END_TOKEN
                if special_token in sample:
                    sample = sample.replace(special_token, audio_text, 1)
                else:
                    sample = sample + audio_text + "\n"
            prompt_strings.append(sample)
        return prompt_strings

    def apply_system_template(
        self,
        sys_prompt_exp: str | None = None,
        use_cot_system_prompt: bool = False,
    ) -> str:
        sys_prompt = SYSTEM_PROMPT_THINK if use_cot_system_prompt else SYSTEM_PROMPT_NOTHINK
        if sys_prompt_exp is not None:
            sys_prompt = sys_prompt.replace("你是一个友好的AI助手。", sys_prompt_exp)
        return sys_prompt

    def apply_chat_template(
        self,
        conversation: list[dict[str, Any]],
        sys_prompt_exp: str | None = None,
        use_cot_system_prompt: bool = False,
        **kwargs,
    ) -> str:
        eos = self.tokenizer.eos_token
        text = self.apply_system_template(sys_prompt_exp, use_cot_system_prompt) + eos

        for idx, message in enumerate(conversation):
            assert message["role"] in ["HUMAN", "ASSISTANT"], (
                f"Invalid role: {message['role']}. Must be 'HUMAN' or 'ASSISTANT'"
            )
            if idx == len(conversation) - 1:
                assert message["role"] == "HUMAN", "Last message must be from HUMAN"

            text += USER_PREFIX if message["role"] == "HUMAN" else ASSISTANT_PREFIX

            content = message["content"]
            if isinstance(content, str):
                # text-only
                text += content
            elif isinstance(content, list):
                # structured content with multimodal elements
                # Count existing placeholders from text items only
                image_placeholders = 0
                video_placeholders = 0
                audio_placeholders = 0
                for content_item in content:
                    if content_item.get("type", "text") == "text":
                        t = content_item.get("text", "")
                        image_placeholders += t.count(PLACEHOLDER_IMAGE_TOKEN_IN_TEXT)
                        video_placeholders += t.count(PLACEHOLDER_VIDEO_TOKEN_IN_TEXT)
                        audio_placeholders += t.count(PLACEHOLDER_AUDIO_TOKEN_IN_TEXT)

                if video_placeholders > 1:
                    raise ValueError("Video count must be at most 1 per message!")

                # Insert placeholders only for media items not already covered
                for content_item in content:
                    content_type = content_item.get("type", "text")

                    if content_type == "image":
                        image_data = content_item.get("image")
                        if image_data is not None:
                            from PIL import Image as PILImage

                            num_images = 1 if isinstance(image_data, (str, PILImage.Image)) else len(image_data)
                            for _ in range(num_images):
                                if image_placeholders > 0:
                                    image_placeholders -= 1
                                else:
                                    text += PLACEHOLDER_IMAGE_TOKEN_IN_TEXT

                    elif content_type == "video":
                        if video_placeholders > 0:
                            video_placeholders -= 1
                        else:
                            text += PLACEHOLDER_VIDEO_TOKEN_IN_TEXT
                    elif content_type == "audio":
                        audio_data = content_item.get("audio")
                        if audio_data is not None:
                            num_audios = 1 if isinstance(audio_data, str) else len(audio_data)
                            for _ in range(num_audios):
                                if audio_placeholders > 0:
                                    audio_placeholders -= 1
                                else:
                                    text += PLACEHOLDER_AUDIO_TOKEN_IN_TEXT

                    elif content_type == "text":
                        text += content_item.get("text", "")

            # Add EOS token after each message except the last one
            text += eos

        text += ASSISTANT_PREFIX
        return text

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        names = (
            self.tokenizer.model_input_names
            + self.image_processor.model_input_names
            + self.audio_processor.model_input_names
        )
        return list(dict.fromkeys(names))


AutoFeatureExtractor.register("MingWhisperFeatureExtractor", MingWhisperFeatureExtractor)
AutoProcessor.register("MingFlashOmniProcessor", MingFlashOmniProcessor)
