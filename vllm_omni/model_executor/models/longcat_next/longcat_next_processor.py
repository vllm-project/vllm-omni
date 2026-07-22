"""vLLM multimodal processor for LongCat-Next.

The checkpoint's ``__call__`` (not vendored — see
``vllm.transformers_utils.processors.longcat_next``) only accepts *file
paths* embedded in the prompt text (it regex-scans for
``<longcat_img_start>PATH<longcat_img_end>``), which does not fit vLLM's
in-memory multimodal data flow. This processor therefore drives the HF
sub-processors directly:

- images -> ``hf_processor.image_processor`` (Qwen2VLImageProcessor) with PIL
  images, producing ``pixel_values`` / ``image_grid_thw``;
- audio  -> the fbank pipeline of ``LongcatNextAudioProcessor``
  (split_with_overlap + extract_fbank_features + inference_output_length)
  applied to resampled 16 kHz waveforms.

Prompts use the checkpoint's placeholder grammar: each image is
``<longcat_img_start><longcat_img_pad><longcat_img_end>`` and each audio clip
``<longcat_audio_start><longcat_audio_pad><longcat_audio_end>``; the single
pad token is expanded to the item's real token count.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from transformers import BatchFeature
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)

from vllm.transformers_utils.processors.longcat_next import (
    LongcatNextAudioProcessor,
    LongcatNextProcessor,
)

from .longcat_next_utils import AUDIO_PAD_TOKEN_ID, IMG_PAD_TOKEN_ID

_IMAGE_PLACEHOLDER = "<longcat_img_start><longcat_img_pad><longcat_img_end>"
_AUDIO_PLACEHOLDER = "<longcat_audio_start><longcat_audio_pad><longcat_audio_end>"

_AUDIO_SAMPLING_RATE = 16000


class LongcatNextProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "audio": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        mm_max_tokens: dict[str, int] = {}
        if mm_counts.get("image", 0) > 0:
            # max_pixels=3211264 / (14*14 patch, 2x2 merge) ~= 4096 tokens
            mm_max_tokens["image"] = 4096
        if mm_counts.get("audio", 0) > 0:
            # 30 s chunk -> bridge_length 187; long audio splits into chunks
            mm_max_tokens["audio"] = 2048
        return mm_max_tokens

    def get_hf_processor(self, **kwargs: object) -> LongcatNextProcessor:
        return self.ctx.get_hf_processor(LongcatNextProcessor, **kwargs)

    def get_data_parser(self):
        return MultiModalDataParser(target_sr=_AUDIO_SAMPLING_RATE)


class LongcatNextDummyInputsBuilder(BaseDummyInputsBuilder[LongcatNextProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        num_audios = mm_counts.get("audio", 0)
        return _IMAGE_PLACEHOLDER * num_images + _AUDIO_PLACEHOLDER * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        num_audios = mm_counts.get("audio", 0)

        image_width, image_height = 448, 448
        audio_duration = 3.0
        audio_length = int(audio_duration * _AUDIO_SAMPLING_RATE)

        mm_data: MultiModalDataDict = {
            "image": [
                np.random.randint(0, 255, (image_height, image_width, 3), dtype=np.uint8)
                for _ in range(num_images)
            ],
            "audio": [
                (np.random.randn(audio_length).astype(np.float32), _AUDIO_SAMPLING_RATE)
                for _ in range(num_audios)
            ],
        }
        return mm_data


def _extract_audio_features(
    audio_processor: LongcatNextAudioProcessor,
    audios: Sequence[np.ndarray],
) -> dict[str, torch.Tensor]:
    """Run LongcatNextAudioProcessor's fbank pipeline on in-memory waveforms.

    Drives the vendored processor's split_with_overlap/extract_fbank_features/
    inference_output_length directly (its own process()/__call__, which read
    audio from file paths, are not vendored): each waveform is split into
    <=30 s chunks, each chunk becomes a (num_mel_bins, 3000) log-mel matrix
    with encoder/bridge lengths.
    """
    features: list[torch.Tensor] = []
    encoder_lengths: list[int] = []
    bridge_lengths: list[int] = []
    chunk_counts: list[int] = []

    for audio in audios:
        waveform = torch.as_tensor(np.asarray(audio), dtype=torch.float32)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.shape[0] > 2:
            waveform = waveform.T
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        num_chunks = 0
        for chunk in audio_processor.split_with_overlap(waveform):
            mel, input_length = audio_processor.extract_fbank_features(chunk)
            encoder_length, bridge_length = audio_processor.inference_output_length(
                input_length,
                audio_processor.kernel_size,
                audio_processor.stride_size,
                audio_processor.avg_pooler,
            )
            if bridge_length <= 0:
                continue
            features.append(torch.as_tensor(mel, dtype=torch.float32))
            encoder_lengths.append(int(encoder_length))
            bridge_lengths.append(int(bridge_length))
            num_chunks += 1
        chunk_counts.append(num_chunks)

    if not features:
        return {}

    return {
        "audio_features": torch.stack(features),
        "audio_encoder_lengths": torch.tensor(encoder_lengths, dtype=torch.long),
        "audio_bridge_lengths": torch.tensor(bridge_lengths, dtype=torch.long),
        "audio_chunk_counts": torch.tensor(chunk_counts, dtype=torch.long),
    }


class LongcatNextMultiModalProcessor(BaseMultiModalProcessor[LongcatNextProcessingInfo]):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, Any],
        mm_kwargs: Mapping[str, Any],
        tok_kwargs: Mapping[str, Any],
    ) -> BatchFeature:
        hf_processor = self.info.get_hf_processor(**mm_kwargs)
        tokenizer = self.info.get_tokenizer()

        data: dict[str, Any] = {}
        data["input_ids"] = torch.tensor(
            [tokenizer.encode(prompt, add_special_tokens=False)], dtype=torch.long
        )

        images = mm_data.get("images") or mm_data.get("image")
        if images:
            image_inputs = hf_processor.image_processor(images=images, return_tensors="pt")
            data["pixel_values"] = image_inputs["pixel_values"]
            data["image_grid_thw"] = image_inputs["image_grid_thw"]

        audios = mm_data.get("audios") or mm_data.get("audio")
        if audios:
            data.update(_extract_audio_features(hf_processor.audio_processor, audios))

        return BatchFeature(data=data)

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: Any,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        # _call_hf_processor never expands mm placeholders; that's done by
        # _get_prompt_updates below.
        return False

    def _get_prompt_updates(
        self,
        mm_items: Any,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        spatial_merge_size = int(getattr(hf_processor.image_processor, "spatial_merge_size", 2))

        def get_image_replacement(item_idx: int) -> list[int]:
            grid_thw = out_mm_kwargs["image"][item_idx]["image_grid_thw"].data
            t, h, w = (int(x) for x in grid_thw)
            num_tokens = t * (h // spatial_merge_size) * (w // spatial_merge_size)
            return [IMG_PAD_TOKEN_ID] * num_tokens

        def get_audio_replacement(item_idx: int) -> list[int]:
            bridge = out_mm_kwargs["audio"][item_idx]["audio_bridge_lengths"].data
            num_tokens = int(bridge.sum())
            return [AUDIO_PAD_TOKEN_ID] * num_tokens

        prompt_updates: list[PromptUpdate] = []
        if "image" in out_mm_kwargs:
            prompt_updates.append(
                PromptReplacement(
                    modality="image",
                    target=[IMG_PAD_TOKEN_ID],
                    replacement=get_image_replacement,
                )
            )
        if "audio" in out_mm_kwargs:
            prompt_updates.append(
                PromptReplacement(
                    modality="audio",
                    target=[AUDIO_PAD_TOKEN_ID],
                    replacement=get_audio_replacement,
                )
            )
        return prompt_updates

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, Any],
    ) -> Mapping[str, MultiModalFieldConfig]:
        config: dict[str, MultiModalFieldConfig] = {}

        image_grid_thw = hf_inputs.get("image_grid_thw")
        if image_grid_thw is not None:
            config["image_grid_thw"] = MultiModalFieldConfig.batched("image")
            config["pixel_values"] = MultiModalFieldConfig.flat_from_sizes(
                "image", image_grid_thw.prod(-1)
            )

        chunk_counts = hf_inputs.get("audio_chunk_counts")
        if chunk_counts is not None:
            config["audio_chunk_counts"] = MultiModalFieldConfig.batched("audio")
            config["audio_features"] = MultiModalFieldConfig.flat_from_sizes("audio", chunk_counts)
            config["audio_encoder_lengths"] = MultiModalFieldConfig.flat_from_sizes("audio", chunk_counts)
            config["audio_bridge_lengths"] = MultiModalFieldConfig.flat_from_sizes("audio", chunk_counts)

        return config
