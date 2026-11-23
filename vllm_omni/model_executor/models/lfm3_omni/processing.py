# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Checkpoint preprocessing, with feature lengths retained on the host."""

import math
from collections.abc import Mapping

import torch
from transformers.video_utils import VideoMetadata
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.parse import MultiModalDataParser, VideoProcessorItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdateDetails,
)


class Lfm3VideoItems(VideoProcessorItems):
    def get_num_frames(self, item_idx):
        return len(self.get(item_idx)[0])

    def get_frame_size(self, item_idx):
        return VideoProcessorItems([self.get(item_idx)[0]]).get_frame_size(0)

    def get_processor_data(self):
        pairs = [self.get(i) for i in range(self.get_count())]
        return {
            "videos": [frames for frames, _ in pairs],
            "video_metadata": [
                VideoMetadata(**{k: v for k, v in metadata.items() if k != "do_sample_frames"}) for _, metadata in pairs
            ],
        }


class Lfm3DataParser(MultiModalDataParser):
    def _parse_video_data(self, data):
        items = super()._parse_video_data(data)
        if isinstance(items, VideoProcessorItems):
            return Lfm3VideoItems(items.data, metadata=items.metadata)
        return items


class Lfm3OmniProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "audio": None, "video": None}

    def get_data_parser(self):
        return Lfm3DataParser(target_sr=16000, target_channels=1, video_needs_metadata=True)


class Lfm3OmniDummyInputsBuilder(BaseDummyInputsBuilder[Lfm3OmniProcessingInfo]):
    def get_dummy_text(self, mm_counts):
        return "".join(f"<|{kind}|>" * mm_counts.get(kind, 0) for kind in ("image", "audio", "video"))

    def get_dummy_mm_data(self, seq_len, mm_counts, mm_options):
        processor = self.info.get_hf_processor()
        image_pixels = processor.image_processor.size["longest_edge"]
        factor = processor.image_processor.patch_size * processor.image_processor.merge_size
        image_height = max(factor, 2 ** math.floor(math.log2(math.sqrt(image_pixels))))
        image_width = max(factor, image_pixels // (image_height * factor) * factor)
        video_frames = min(processor.video_processor.max_frames, processor.video_processor.max_video_tokens // 2)
        data = {
            "image": self._get_dummy_images(
                width=image_width,
                height=image_height,
                num_images=mm_counts.get("image", 0),
                overrides=mm_options.get("image"),
            ),
            "audio": self._get_dummy_audios(
                length=seq_len * 1280,
                num_audios=mm_counts.get("audio", 0),
                overrides=mm_options.get("audio"),
            ),
            "video": self._get_dummy_videos(
                width=64,
                height=64,
                num_frames=video_frames,
                num_videos=mm_counts.get("video", 0),
                overrides=mm_options.get("video"),
            ),
        }

        data["video"] = [
            (frames, {"fps": 2.0, "frames_indices": list(range(len(frames))), "total_num_frames": len(frames)})
            for frames in data["video"]
        ]
        return data


class Lfm3OmniMultiModalProcessor(BaseMultiModalProcessor[Lfm3OmniProcessingInfo]):
    def _call_hf_processor(self, hf_data, hf_kwargs):
        if hf_kwargs.get("use_audio_in_video"):
            raise ValueError("LFM3-Omni currently accepts silent video frames; pass audio as a separate modality")
        kwargs = {"sampling_rate": 16000, **hf_kwargs}
        if "videos" in hf_data:
            kwargs.update(return_metadata=True, do_sample_frames=False)
        output = super()._call_hf_processor(hf_data, kwargs)
        if "video_metadata" in output:
            processor = self.info.get_hf_processor(**kwargs)
            output["video_prompt_ids"] = [
                torch.tensor(
                    processor.tokenizer.encode(processor._video_expansion(output, i), add_special_tokens=False)
                )
                for i in range(len(output["video_metadata"]))
            ]
            del output["video_metadata"]
        return output

    def _postprocess_prompt(self, prompt):
        processor = self.info.get_hf_processor()
        tokenizer = processor.tokenizer
        start = tokenizer.convert_tokens_to_ids(processor.vision_start_token)
        end = tokenizer.convert_tokens_to_ids(processor.vision_end_token)
        config = self.info.get_hf_config()
        media = {config.image_token_id, config.video_token_id, config.audio_token_id}
        result, i = [], 0
        while i < len(prompt):
            if i + 2 < len(prompt) and prompt[i] == start and prompt[i + 1] in media and prompt[i + 2] == end:
                result.append(prompt[i + 1])
                i += 3
            else:
                result.append(prompt[i])
                i += 1
        return result

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        fields = {
            "input_features": MultiModalFieldConfig.batched("audio"),
            "feature_attention_mask": MultiModalFieldConfig.batched("audio"),
        }
        for modality, pixels, grid in (
            ("image", "pixel_values", "image_grid_thw"),
            ("video", "pixel_values_videos", "video_grid_thw"),
        ):
            if grid in hf_inputs:
                fields[grid] = MultiModalFieldConfig.batched(modality, keep_on_cpu=True)
                fields[pixels] = MultiModalFieldConfig.flat_from_sizes(modality, hf_inputs[grid].prod(-1))
        if "video_prompt_ids" in hf_inputs:
            fields["video_prompt_ids"] = MultiModalFieldConfig.batched("video", keep_on_cpu=True)
        return fields

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        config = self.info.get_hf_config()
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        data = out_mm_kwargs.get_data()
        replacements = []
        for modality in ("image", "audio", "video"):
            if modality not in out_mm_kwargs:
                continue
            token_id = getattr(config, f"{modality}_token_id")
            if modality == "audio":
                mask = data["feature_attention_mask"]
                lengths = [processor.feature_extractor.num_encoder_tokens(int(n)) for n in mask.sum(-1)]
            else:
                lengths = (data[f"{modality}_grid_thw"].prod(-1) // config.downsample_factor**2).tolist()

            if modality == "video":
                prompts = [ids.tolist() for ids in data["video_prompt_ids"]]
            else:
                markers = processor.audio_markers if modality == "audio" else processor.vision_markers
                token = getattr(processor, f"{modality}_token")
                prompts = [
                    processor.tokenizer.encode(processor._wrap(token * n, markers), add_special_tokens=False)
                    for n in lengths
                ]

            def replacement(index, prompts=prompts, token_id=token_id):
                return PromptUpdateDetails.select_token_id(prompts[index], embed_token_id=token_id)

            replacements.append(PromptReplacement(modality=modality, target=[token_id], replacement=replacement))
        return replacements
