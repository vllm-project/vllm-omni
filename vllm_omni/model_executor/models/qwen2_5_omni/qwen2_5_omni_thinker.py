"""Thin Omni wrapper: reuse upstream Qwen2.5-Omni thinker with minimal overrides."""

from collections.abc import Iterable, Iterator, Mapping, Sequence
from functools import partial
from typing import Any

import numpy as np
import PIL.Image as PILImage
import torch
from torch import nn
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniThinkerConfig,
)
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniAudioEncoder,
)
from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.qwen2_5_omni_thinker import (
    Qwen2_5OmniAudioFeatureInputs,
    Qwen2_5OmniThinkerDummyInputsBuilder,
    check_interleaved_audio_video,
    create_qwen2_5_omni_thinker_field_factory,
    merge_interleaved_embeddings,
)
from vllm.model_executor.models.qwen2_5_omni_thinker import (
    Qwen2_5OmniConditionalGenerationMixin as Qwen2_5OmniConditionalGenerationMixinBase,
)
from vllm.model_executor.models.qwen2_5_omni_thinker import (
    Qwen2_5OmniThinkerMultiModalDataParser as _Qwen2_5OmniThinkerMultiModalDataParserBase,
)
from vllm.model_executor.models.qwen2_5_omni_thinker import (
    Qwen2_5OmniThinkerMultiModalProcessor as _Qwen2_5OmniThinkerMultiModalProcessorBase,
)
from vllm.model_executor.models.qwen2_5_omni_thinker import (
    Qwen2_5OmniThinkerProcessingInfo as _Qwen2_5OmniThinkerProcessingInfoBase,
)
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VisionTransformer,
    Qwen2_5_VLImageEmbeddingInputs,
    Qwen2_5_VLImageInputs,
    Qwen2_5_VLImagePixelInputs,
    Qwen2_5_VLVideoEmbeddingInputs,
    Qwen2_5_VLVideoInputs,
    Qwen2_5_VLVideoPixelInputs,
)
from vllm.model_executor.models.qwen2_audio import (
    _get_feat_extract_output_lengths,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import AudioProcessorItems, MultiModalDataItems, VideoProcessorItems
from vllm.multimodal.processing.context import TimingContext
from vllm.multimodal.processing.inputs import ProcessorInputs
from vllm.multimodal.processing.processor import (
    MultiModalProcessingInfo,
    MultiModalPromptUpdates,
    PlaceholderFeaturesInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.collection_utils import is_list_of

from vllm_omni.quantization.component_config import (
    resolve_encoder_quant_config,
)

try:
    import flash_attn
except (ImportError, ModuleNotFoundError):
    flash_attn = None
logger = init_logger(__name__)

_PER_VIDEO_USE_AUDIO_IN_VIDEO_KEY = "_vllm_omni_per_video_use_audio_in_video"


def _normalize_use_audio_in_video(
    value: object,
    num_videos: int,
) -> list[bool]:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return [bool(value.item())] * num_videos
        values = value.flatten().tolist()
    elif isinstance(value, np.ndarray):
        if value.size == 1:
            return [bool(value.item())] * num_videos
        values = value.flatten().tolist()
    elif isinstance(value, bool):
        return [value] * num_videos
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
    else:
        return [bool(value)] * num_videos

    if len(values) != num_videos:
        raise ValueError(
            "use_audio_in_video must be a boolean or contain one boolean per "
            f"video, but found {len(values)} values for {num_videos} videos."
        )
    return [bool(v) for v in values]


def _normalize_per_video_use_audio_in_video(value: object) -> list[bool]:
    if isinstance(value, torch.Tensor):
        values = value.flatten().tolist()
    elif isinstance(value, np.ndarray):
        values = value.flatten().tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
    else:
        values = [value]
    return [bool(v) for v in values]


def _is_per_video_use_audio_in_video(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, torch.Tensor):
        return value.numel() != 1
    if isinstance(value, np.ndarray):
        return value.size != 1
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _get_request_video_use_audio_in_video(
    hf_processor_mm_kwargs: Mapping[str, object],
    num_videos: int,
) -> list[bool]:
    value = hf_processor_mm_kwargs.get(
        _PER_VIDEO_USE_AUDIO_IN_VIDEO_KEY,
        hf_processor_mm_kwargs.get("use_audio_in_video", False),
    )
    return _normalize_use_audio_in_video(value, num_videos)


def _prompt_update_has_audio_token(
    mm_prompt_updates: MultiModalPromptUpdates,
    item_idx: int,
    audio_token_id: int,
) -> bool:
    updates = mm_prompt_updates.get("video", [])
    if item_idx >= len(updates):
        return False
    return any(
        audio_token_id in update.content.full for update in updates[item_idx] if isinstance(update.content.full, list)
    )


def _get_video_second_per_grid_t(
    out_mm_data: Mapping[str, object],
    hf_processor_mm_kwargs: Mapping[str, object],
    item_idx: int,
    default: float,
) -> float:
    second_per_grid_ts = out_mm_data.get("second_per_grid_ts")
    if second_per_grid_ts is None:
        second_per_grid_ts = hf_processor_mm_kwargs.get("second_per_grid_ts", None)
    if second_per_grid_ts is None:
        return default

    if isinstance(second_per_grid_ts, torch.Tensor):
        if second_per_grid_ts.numel() == 1:
            return float(second_per_grid_ts.item())
        return float(second_per_grid_ts.flatten()[item_idx].item())

    if isinstance(second_per_grid_ts, np.ndarray):
        if second_per_grid_ts.size == 1:
            return float(second_per_grid_ts.item())
        return float(second_per_grid_ts.flatten()[item_idx].item())

    if isinstance(second_per_grid_ts, Sequence) and not isinstance(second_per_grid_ts, (str, bytes)):
        return float(second_per_grid_ts[item_idx])

    return float(second_per_grid_ts)


def _filter_video_use_audio_in_video_for_uncached_items(
    hf_processor_mm_kwargs: Mapping[str, object],
    video_is_cached: Sequence[bool] | None,
) -> Mapping[str, object]:
    use_audio_in_video = hf_processor_mm_kwargs.get(
        _PER_VIDEO_USE_AUDIO_IN_VIDEO_KEY,
        hf_processor_mm_kwargs.get("use_audio_in_video"),
    )
    if video_is_cached is None or not _is_per_video_use_audio_in_video(use_audio_in_video):
        return hf_processor_mm_kwargs

    mask = _normalize_use_audio_in_video(use_audio_in_video, len(video_is_cached))
    missing_mask = [use_audio for use_audio, is_cached in zip(mask, video_is_cached) if not is_cached]

    filtered_kwargs = dict(hf_processor_mm_kwargs)
    filtered_kwargs["use_audio_in_video"] = missing_mask
    filtered_kwargs[_PER_VIDEO_USE_AUDIO_IN_VIDEO_KEY] = missing_mask
    second_per_grid_ts = hf_processor_mm_kwargs.get("second_per_grid_ts")
    if (
        isinstance(second_per_grid_ts, Sequence)
        and not isinstance(second_per_grid_ts, (str, bytes))
        and len(second_per_grid_ts) == len(video_is_cached)
    ):
        filtered_kwargs["second_per_grid_ts"] = [
            second_per_grid_t
            for second_per_grid_t, is_cached in zip(second_per_grid_ts, video_is_cached)
            if not is_cached
        ]
    return filtered_kwargs


def _coerce_use_audio_in_video_for_hf_processor(
    mm_data: Mapping[str, object],
    mm_kwargs: Mapping[str, object],
) -> Mapping[str, object]:
    use_audio_in_video = mm_kwargs.get("use_audio_in_video")
    if not _is_per_video_use_audio_in_video(use_audio_in_video):
        return mm_kwargs

    video_use_audio_in_video = _normalize_per_video_use_audio_in_video(use_audio_in_video)

    hf_mm_kwargs = dict(mm_kwargs)
    # HF processors only support a global bool. For per-video masks, vLLM
    # consumes the list while building prompt updates and placeholders.
    hf_mm_kwargs["use_audio_in_video"] = False
    hf_mm_kwargs[_PER_VIDEO_USE_AUDIO_IN_VIDEO_KEY] = video_use_audio_in_video
    return hf_mm_kwargs


def _presampled_videos_hf_kwargs(
    mm_data: Mapping[str, object],
    mm_kwargs: Mapping[str, object],
) -> Mapping[str, object]:
    """Adjust HF video kwargs for videos pre-sampled by vLLM's video loader.

    When ``video_metadata`` is present, the frames were already sampled
    according to ``media_io_kwargs``. The HF processor should not resample
    them and should use the sampled fps when computing temporal metadata.
    """
    video_metadata = mm_data.get("video_metadata")
    if not video_metadata:
        return mm_kwargs

    mm_kwargs = dict(mm_kwargs)
    videos_kwargs = dict(mm_kwargs.get("videos_kwargs") or {})
    videos_kwargs["do_sample_frames"] = False

    def _compute_sampled_video_fps(metadata) -> float | None:
        duration = getattr(metadata, "duration", None)
        indices = getattr(metadata, "frames_indices", None)
        if not duration or not indices or duration <= 0:
            return None
        return len(indices) / float(duration)

    if "fps" not in videos_kwargs:
        fps_values = [_compute_sampled_video_fps(m) for m in video_metadata]
        known_fps = [fps for fps in fps_values if fps is not None]
        unique_fps = set(known_fps)
        if len(unique_fps) == 1:
            videos_kwargs["fps"] = known_fps[0]
        elif len(unique_fps) > 1:
            logger.warning(
                "Mixed sampled FPS %s in one request; HF accepts a single fps, using %s.",
                sorted(unique_fps),
                known_fps[0],
            )
            videos_kwargs["fps"] = known_fps[0]

    mm_kwargs["videos_kwargs"] = videos_kwargs
    return mm_kwargs


class Qwen2_5OmniVideoProcessorItems(VideoProcessorItems):
    """Video items that carry the loader's ``(frames, metadata)`` tuples."""

    def get_processor_data(self) -> Mapping[str, object]:
        from transformers.video_utils import VideoMetadata

        videos = [frames for frames, _ in self.data]
        video_metadata = [
            VideoMetadata(**{k: v for k, v in metadata.items() if k != "do_sample_frames"}) for _, metadata in self.data
        ]
        return {"videos": videos, "video_metadata": video_metadata}


class Qwen2_5OmniThinkerMultiModalDataParser(_Qwen2_5OmniThinkerMultiModalDataParserBase):
    def _parse_video_data(self, data):
        if data is None or isinstance(data, dict) or self.is_embeddings(data):
            return super()._parse_video_data(data)

        if (is_list_of(data, PILImage.Image) and len(data) > 0) or (
            isinstance(data, (np.ndarray, torch.Tensor)) and data.ndim == 4
        ):
            data_items = [data]
        elif isinstance(data, (np.ndarray, torch.Tensor)):
            data_items = [elem for elem in data]
        elif isinstance(data, tuple) and len(data) == 2:
            data_items = [data]
        else:
            data_items = data

        videos_with_metadata = [self._get_video_with_metadata(item) for item in data_items]
        if any(metadata is None for _, metadata in videos_with_metadata):
            return super()._parse_video_data(data)

        return Qwen2_5OmniVideoProcessorItems(
            videos_with_metadata,
            metadata=[metadata for _, metadata in videos_with_metadata],
        )


class Qwen2_5OmniThinkerProcessingInfo(_Qwen2_5OmniThinkerProcessingInfoBase):
    def get_data_parser(self):
        feature_extractor = self.get_feature_extractor()

        return Qwen2_5OmniThinkerMultiModalDataParser(
            spatial_merge_size=self.get_hf_config().vision_config.spatial_merge_size,
            target_sr=feature_extractor.sampling_rate,
            target_channels=self.get_target_channels(),
            expected_hidden_size=self._get_expected_hidden_size(),
        )


class Qwen2_5OmniThinkerMultiModalProcessor(
    _Qwen2_5OmniThinkerMultiModalProcessorBase,
):
    """Override to fix use_audio_in_video detection when mm cache returns None."""

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ):
        mm_kwargs = _presampled_videos_hf_kwargs(mm_data, mm_kwargs)
        mm_kwargs = _coerce_use_audio_in_video_for_hf_processor(mm_data, mm_kwargs)
        hf_inputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )
        per_video_mask = mm_kwargs.get(_PER_VIDEO_USE_AUDIO_IN_VIDEO_KEY)
        if per_video_mask is not None:
            hf_inputs["use_audio_in_video"] = torch.tensor(per_video_mask)
        return hf_inputs

    def _get_mm_fields_config(
        self,
        hf_inputs,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        fields_config = dict(
            create_qwen2_5_omni_thinker_field_factory(self.info.get_hf_config().vision_config.spatial_merge_size)(
                hf_inputs
            )
        )
        use_audio_in_video = hf_inputs.get("use_audio_in_video")
        is_batched_mask = isinstance(use_audio_in_video, torch.Tensor) and use_audio_in_video.numel() > 1
        if _PER_VIDEO_USE_AUDIO_IN_VIDEO_KEY in hf_processor_mm_kwargs or is_batched_mask:
            fields_config["use_audio_in_video"] = MultiModalFieldConfig.batched("video")
        return fields_config

    def _get_video_use_audio_in_video(
        self,
        mm_kwargs: MultiModalKwargsItems,
        mm_prompt_updates: MultiModalPromptUpdates,
    ) -> list[bool]:
        video_kwargs = mm_kwargs.get("video", [])
        if not video_kwargs:
            return []

        audio_token_id = self.info.get_hf_config().audio_token_id
        video_use_audio_in_video = []
        has_use_audio_in_video = any(item is not None and "use_audio_in_video" in item for item in video_kwargs)
        for item_idx, item in enumerate(video_kwargs):
            if has_use_audio_in_video:
                if item is None or "use_audio_in_video" not in item:
                    video_use_audio_in_video.append(
                        _prompt_update_has_audio_token(
                            mm_prompt_updates,
                            item_idx,
                            audio_token_id,
                        )
                    )
                    continue
                use_audio_in_video_tensor = item["use_audio_in_video"].data
                if use_audio_in_video_tensor.numel() > 0:
                    video_use_audio_in_video.append(bool(use_audio_in_video_tensor.item()))
                    continue
                video_use_audio_in_video.append(False)
                continue

            video_use_audio_in_video.append(
                _prompt_update_has_audio_token(
                    mm_prompt_updates,
                    item_idx,
                    audio_token_id,
                )
            )

        return video_use_audio_in_video

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
            _, audio_output_lens = _get_feat_extract_output_lengths(audio_feature_lengths)
            audio_output_lengths = audio_output_lens.tolist()
        elif feature_attention_mask is not None:
            assert isinstance(feature_attention_mask, torch.Tensor)
            _, audio_output_lens = _get_feat_extract_output_lengths(feature_attention_mask.sum(-1))
            audio_output_lengths = audio_output_lens.tolist()

        audio_in_video_item_idx = 0

        def get_replacement_qwen2_audio(item_idx: int):
            item_idx += audio_in_video_item_idx

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

        num_videos = mm_items.get_all_counts().get("video", 0)
        video_use_audio_in_video = _get_request_video_use_audio_in_video(
            hf_processor_mm_kwargs,
            num_videos,
        )
        thinker_config = self.info.get_hf_config()

        def get_replacement_qwen2_video(item_idx: int):
            nonlocal audio_in_video_item_idx

            if not video_use_audio_in_video[item_idx]:
                return get_replacement_qwen2_vision(item_idx, modality="video")

            audio_num_features = audio_output_lengths[audio_in_video_item_idx]
            video_grid_thw = out_mm_data["video_grid_thw"][item_idx]

            audio_in_video_item_idx += 1

            video_second_per_grid_t = _get_video_second_per_grid_t(
                out_mm_data,
                hf_processor_mm_kwargs,
                item_idx,
                default=1.0,
            )

            updates = self.omni_get_updates_use_audio_in_video(
                thinker_config=thinker_config,
                audio_len=audio_num_features,
                video_grid_thw=video_grid_thw,
                video_second_per_grid_t=video_second_per_grid_t,
            )

            return PromptUpdateDetails.select_token_id(
                seq=updates,
                embed_token_id=video_token_id,
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
                replacement=get_replacement_qwen2_video,
            ),
        ]

    def _apply_hf_processor_mm_only(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ):
        mm_counts = mm_items.get_all_counts()

        if "video" in mm_counts:
            video_use_audio_in_video = _get_request_video_use_audio_in_video(
                hf_processor_mm_kwargs,
                mm_counts["video"],
            )
            if any(video_use_audio_in_video):
                assert "audio" in mm_counts
                mm_counts["audio"] -= sum(video_use_audio_in_video)

        _, mm_processed_data, _ = self._apply_hf_processor_text_mm(
            prompt_text=self.dummy_inputs.get_dummy_text(mm_counts),
            mm_items=mm_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            tokenization_kwargs=tokenization_kwargs,
        )

        return mm_processed_data

    def _cached_apply_hf_processor(
        self,
        inputs: ProcessorInputs,
        timing_ctx: TimingContext,
    ) -> tuple[list[int], MultiModalProcessingInfo, bool]:
        cache = self.cache

        _, passthrough_data = self._get_hf_mm_data(inputs.mm_data_items)
        if cache is None or passthrough_data:
            return self._apply_hf_processor(inputs, timing_ctx)

        with timing_ctx.record("get_mm_hashes"):
            mm_hashes = inputs.get_mm_hashes(self.info.model_id)

        with timing_ctx.record("get_cache_missing_items"):
            mm_is_cached, mm_missing_data_items = self._get_cache_missing_items(
                cache=cache,
                mm_data_items=inputs.mm_data_items,
                mm_hashes=mm_hashes,
            )

        hf_processor_mm_kwargs = _filter_video_use_audio_in_video_for_uncached_items(
            inputs.hf_processor_mm_kwargs,
            mm_is_cached.get("video"),
        )

        with timing_ctx.record("apply_hf_processor"):
            (
                prompt_ids,
                mm_missing_processed_data,
                is_update_applied,
            ) = self._apply_hf_processor_main(
                prompt=inputs.prompt,
                mm_items=mm_missing_data_items,
                hf_processor_mm_kwargs=hf_processor_mm_kwargs,
                tokenization_kwargs=inputs.tokenization_kwargs,
                enable_hf_prompt_update=False,
            )

        mm_missing_kwargs = MultiModalKwargsItems.from_hf_inputs(
            mm_missing_processed_data,
            self._get_mm_fields_config(
                mm_missing_processed_data,
                hf_processor_mm_kwargs,
            ),
        )

        mm_missing_prompt_updates = self._get_mm_prompt_updates(
            mm_missing_data_items,
            hf_processor_mm_kwargs,
            mm_missing_kwargs,
        )

        with timing_ctx.record("merge_mm_kwargs"):
            mm_kwargs, mm_prompt_updates = self._merge_mm_kwargs(
                cache,
                mm_hashes=mm_hashes,
                mm_is_cached=mm_is_cached,
                mm_missing_kwargs=mm_missing_kwargs,
                mm_missing_prompt_updates=mm_missing_prompt_updates,
            )

        mm_info = MultiModalProcessingInfo(
            kwargs=mm_kwargs,
            hashes=mm_hashes,
            prompt_updates=mm_prompt_updates,
        )

        return prompt_ids, mm_info, is_update_applied

    def _derive_audio_from_video_placeholders(
        self,
        placeholders: Mapping[str, list[PlaceholderFeaturesInfo]],
        mm_prompt_updates: MultiModalPromptUpdates,
        video_use_audio_in_video: Sequence[bool] | None = None,
    ) -> Mapping[str, list[PlaceholderFeaturesInfo]]:
        if "video" not in placeholders:
            return placeholders

        num_videos = len(placeholders["video"])
        if video_use_audio_in_video is None:
            video_use_audio_in_video = [True] * num_videos
        elif len(video_use_audio_in_video) != num_videos:
            raise ValueError(
                "use_audio_in_video must contain one boolean per video, "
                f"but found {len(video_use_audio_in_video)} values for "
                f"{num_videos} videos."
            )

        num_audio_in_video = sum(video_use_audio_in_video)
        num_audios = len(mm_prompt_updates.get("audio", []))
        if num_audios != num_audio_in_video:
            raise ValueError(
                "use_audio_in_video requires equal number of audio and video "
                f"items using audio, got {num_audios=}, {num_audio_in_video=}"
            )

        tokenizer = self.info.get_tokenizer()
        processor = self.info.get_hf_processor()
        audio_token_id = tokenizer.get_vocab()[processor.audio_token]

        result_placeholders = dict(placeholders)
        audio_placeholders = []
        video_placeholders = []

        audio_idx = 0
        for video_idx, video_placeholder in enumerate(placeholders["video"]):
            audio_is_embed = torch.tensor(video_placeholder.tokens) == audio_token_id

            if video_use_audio_in_video[video_idx]:
                audio_placeholder = PlaceholderFeaturesInfo(
                    modality="audio",
                    item_idx=audio_idx,
                    start_idx=video_placeholder.start_idx,
                    tokens=video_placeholder.tokens,
                    is_embed=audio_is_embed,
                )
                audio_placeholders.append(audio_placeholder)
                audio_idx += 1

            video_placeholder_with_mask = PlaceholderFeaturesInfo(
                modality="video",
                item_idx=video_idx,
                start_idx=video_placeholder.start_idx,
                tokens=video_placeholder.tokens,
                is_embed=~audio_is_embed,
            )
            video_placeholders.append(video_placeholder_with_mask)

        result_placeholders["audio"] = audio_placeholders
        result_placeholders["video"] = video_placeholders
        return result_placeholders

    def _maybe_apply_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        prompt_ids: list[int],
        mm_kwargs: MultiModalKwargsItems,
        mm_prompt_updates: MultiModalPromptUpdates,
        is_update_applied: bool,
    ) -> tuple[list[int], Mapping[str, list[PlaceholderFeaturesInfo]]]:
        mm_item_counts = mm_items.get_all_counts()
        self._validate_mm_kwargs(mm_kwargs, mm_item_counts)
        self._validate_mm_updates(mm_prompt_updates, mm_item_counts)

        video_use_audio_in_video = self._get_video_use_audio_in_video(mm_kwargs, mm_prompt_updates)
        use_audio_in_video = any(video_use_audio_in_video)

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
            if use_audio_in_video and "audio" in mm_prompt_updates:
                filtered_updates = {k: v for k, v in mm_prompt_updates.items() if k != "audio"}
                prompt_ids, mm_placeholders = self._apply_prompt_updates(
                    prompt_ids,
                    filtered_updates,
                )
                mm_placeholders = self._derive_audio_from_video_placeholders(
                    mm_placeholders,
                    mm_prompt_updates,
                    video_use_audio_in_video,
                )
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


class Qwen2_5OmniConditionalGenerationMixin(Qwen2_5OmniConditionalGenerationMixinBase):
    def _parse_and_validate_audio_input(self, **kwargs: object) -> Qwen2_5OmniAudioFeatureInputs | None:
        input_audio_features = kwargs.pop("input_audio_features", None)
        audio_feature_lengths = kwargs.pop("audio_feature_lengths", None)
        feature_attention_mask = kwargs.pop("feature_attention_mask", None)
        if input_audio_features is None:
            return None
        if (
            input_audio_features is not None
            and isinstance(input_audio_features, torch.Tensor)
            and input_audio_features.ndim == 3
        ):
            input_audio_features = input_audio_features.reshape(-1, input_audio_features.shape[-1])
        elif input_audio_features is not None and isinstance(input_audio_features, list):
            input_audio_features = torch.cat(input_audio_features, dim=-1)
        if (
            audio_feature_lengths is not None
            and isinstance(audio_feature_lengths, torch.Tensor)
            and audio_feature_lengths.ndim == 2
        ):
            audio_feature_lengths = audio_feature_lengths.reshape(-1)
        elif audio_feature_lengths is not None and isinstance(audio_feature_lengths, list):
            audio_feature_lengths = torch.cat(audio_feature_lengths, dim=-1)
        if (
            feature_attention_mask is not None
            and isinstance(feature_attention_mask, torch.Tensor)
            and feature_attention_mask.ndim == 3
        ):
            feature_attention_mask = feature_attention_mask.reshape(-1, feature_attention_mask.shape[-1])
        elif feature_attention_mask is not None and isinstance(feature_attention_mask, list):
            for i in range(len(feature_attention_mask)):
                feature_attention_mask[i] = feature_attention_mask[i].reshape(-1)
        return Qwen2_5OmniAudioFeatureInputs(
            type="audio_features",
            input_features=input_audio_features,
            audio_feature_lengths=audio_feature_lengths,
            feature_attention_mask=feature_attention_mask,
        )

    def _parse_and_validate_image_input(
        self,
        **kwargs: dict[str, Any],
    ) -> Qwen2_5_VLImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None and image_embeds is None:
            return None
        if pixel_values is not None and isinstance(pixel_values, torch.Tensor) and pixel_values.ndim == 3:
            pixel_values = pixel_values.reshape(-1, pixel_values.shape[-1])
        if image_embeds is not None and isinstance(image_embeds, torch.Tensor) and image_embeds.ndim == 3:
            image_embeds = image_embeds.reshape(-1, image_embeds.shape[-1])
        if image_grid_thw is not None and isinstance(image_grid_thw, torch.Tensor) and image_grid_thw.ndim == 3:
            image_grid_thw = image_grid_thw.reshape(-1, image_grid_thw.shape[-1])
        if pixel_values is not None:
            return Qwen2_5_VLImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

        if image_embeds is not None:
            return Qwen2_5_VLImageEmbeddingInputs(
                type="image_embeds",
                image_embeds=image_embeds,
                image_grid_thw=image_grid_thw,
            )

    def _parse_and_validate_video_input(
        self,
        **kwargs: dict[str, Any],
    ) -> Qwen2_5_VLVideoInputs | None:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_embeds = kwargs.pop("video_embeds", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)

        if pixel_values_videos is None and video_embeds is None:
            return None

        if (
            pixel_values_videos is not None
            and isinstance(pixel_values_videos, torch.Tensor)
            and pixel_values_videos.ndim == 3
        ):
            pixel_values_videos = pixel_values_videos.reshape(-1, pixel_values_videos.shape[-1])
        if video_grid_thw is not None and isinstance(video_grid_thw, torch.Tensor) and video_grid_thw.ndim == 3:
            video_grid_thw = video_grid_thw.reshape(-1, video_grid_thw.shape[-1])
        if video_embeds is not None and isinstance(video_embeds, torch.Tensor) and video_embeds.ndim == 3:
            video_embeds = video_embeds.reshape(-1, video_embeds.shape[-1])
        if pixel_values_videos is not None:
            return Qwen2_5_VLVideoPixelInputs(
                type="pixel_values_videos",
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
            )

        if video_embeds is not None:
            if not isinstance(video_embeds, torch.Tensor):
                raise ValueError(f"Incorrect type of video embeddings. Got type: {type(video_embeds)}")
            return Qwen2_5_VLVideoEmbeddingInputs(
                type="video_embeds",
                video_embeds=video_embeds,
                video_grid_thw=video_grid_thw,
            )

    def _process_image_input(self, image_input: Qwen2_5_VLImageInputs) -> tuple[torch.Tensor, ...]:
        if image_input["type"] == "image_embeds":
            return image_input["image_embeds"].type(self.visual.dtype)

        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2

        pixel_values = image_input["pixel_values"].type(self.visual.dtype)
        with set_forward_context(None, self.vllm_config):
            image_embeds = self.visual(pixel_values, grid_thw=grid_thw)
        # Split concatenated embeddings for each image item.
        merge_size = self.visual.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size

        return image_embeds.split(sizes.tolist())

    def _process_video_input(
        self,
        video_input: Qwen2_5_VLVideoInputs,
    ) -> torch.Tensor:
        if video_input["type"] == "video_embeds":
            return video_input["video_embeds"].type(self.visual.dtype)

        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2

        pixel_values_videos = video_input["pixel_values_videos"].type(self.visual.dtype)
        with set_forward_context(None, self.vllm_config):
            video_embeds = self.visual(pixel_values_videos, grid_thw=grid_thw)
        # Split concatenated embeddings for each video item.
        merge_size = self.visual.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size

        return video_embeds.split(sizes.tolist())


@MULTIMODAL_REGISTRY.register_processor(
    Qwen2_5OmniThinkerMultiModalProcessor,
    info=Qwen2_5OmniThinkerProcessingInfo,
    dummy_inputs=Qwen2_5OmniThinkerDummyInputsBuilder,
)
class Qwen2_5OmniThinkerForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsLoRA,
    SupportsMRoPE,
    Qwen2_5OmniConditionalGenerationMixin,
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
        "attn.qkv": [
            "attn.q",
            "attn.k",
            "attn.v",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
        "attn_qkv_proj": [
            "attn_q_proj",
            "attn_k_proj",
            "attn_v_proj",
        ],
        "qkv": [
            "q",
            "k",
            "v",
        ],
    }

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|vision_start|><|IMAGE|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|VIDEO|><|vision_end|>"
        if modality.startswith("audio"):
            return f"Audio {i}: <|audio_bos|><|AUDIO|><|audio_eos|>"

        raise ValueError("Only image, video or audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        thinker_config: Qwen2_5OmniThinkerConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = thinker_config
        self.multimodal_config = multimodal_config

        # force "use_flash_attention_2=True" to audio tower to align
        # the results.
        if flash_attn is not None:
            audio_config = thinker_config.audio_config
            audio_config._attn_implementation_autoset = True
            audio_config._attn_implementation = "flash_attention_2"
        else:
            logger.warning(
                "flash_attn is not available, the model may not yield the "
                "exactly same result as the transformers implementation "
                "in the audio tower part."
            )

        self.quant_config = quant_config

        # Pre-quantized checkpoints (modelopt NVFP4/FP8/MXFP8) only quantize
        # the Thinker LM. Vision encoder weights remain in BF16 with no FP8
        # scale tensors; passing quant_config causes FP8 kernels to run on
        # BF16 weights, producing garbage embeddings. Keep None for encoders.
        visual_quant_config = resolve_encoder_quant_config(quant_config)

        with self._mark_tower_model(vllm_config, "audio"):
            if multimodal_config.get_limit_per_prompt("audio"):
                self.audio_tower = Qwen2_5OmniAudioEncoder(thinker_config.audio_config)
            else:
                self.audio_tower = None

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            if multimodal_config.get_limit_per_prompt("image") or multimodal_config.get_limit_per_prompt("video"):
                self.visual = Qwen2_5_VisionTransformer(
                    vision_config=thinker_config.vision_config,
                    norm_eps=getattr(thinker_config.text_config, "rms_norm_eps", 1e-6),
                    quant_config=visual_quant_config,
                    prefix=maybe_prefix(prefix, "visual"),
                )
            else:
                self.visual = None

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model"),
                hf_config=thinker_config.text_config,
                architectures=["Qwen2ForCausalLM"],
            )

        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

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

    def _get_audio_for_video_mapping(self, mm_features: list[MultiModalFeatureSpec]) -> tuple[dict[int, int], set[int]]:
        """
        Map video offset -> paired audio_feature_length for use_audio_in_video.

        When use_audio_in_video=True, audio is interleaved within video chunks.
        The pairing is based on feature order in mm_features.
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

    def _compute_audio_token_count(self, audio_feature_length: int) -> int:
        return ((audio_feature_length - 1) // 2 + 1 - 2) // 2 + 1

    def iter_mm_features(self, mm_features: list[MultiModalFeatureSpec]) -> Iterator[tuple[int, str, dict[str, Any]]]:
        thinker_config = self.config
        spatial_merge_size = thinker_config.vision_config.spatial_merge_size
        tokens_per_second = getattr(thinker_config.vision_config, "tokens_per_second", 25)

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
                        "t_factor": 1.0 * tokens_per_second,
                    },
                )
            elif modality == "video":
                t, h, w = mm_feature.data["video_grid_thw"].data.tolist()
                second_per_grid_ts = 1.0
                if mm_feature.data.get("second_per_grid_ts"):
                    second_per_grid_ts = mm_feature.data["second_per_grid_ts"].data.item()
                use_audio_in_video = False
                if mm_feature.data.get("use_audio_in_video"):
                    use_audio_in_video = bool(mm_feature.data["use_audio_in_video"].data.item())

                yield (
                    offset,
                    "video",
                    {
                        "grid_t": t,
                        "grid_h": h // spatial_merge_size,
                        "grid_w": w // spatial_merge_size,
                        "t_factor": second_per_grid_ts * tokens_per_second,
                        "use_audio_in_video": use_audio_in_video,
                        "audio_feature_length": audio_for_video.get(offset),
                    },
                )
            elif modality == "audio":
                if offset not in paired_audio_offsets:
                    audio_len = mm_feature.data["audio_feature_lengths"].data.item()
                    yield offset, "audio", {"audio_feature_length": audio_len}

    def _compute_interleaved_positions(self, start_idx: int, data: dict[str, Any]) -> tuple[np.ndarray, int]:
        grid_t = data["grid_t"]
        grid_h = data["grid_h"]
        grid_w = data["grid_w"]
        t_factor = data["t_factor"]
        audio_len = data["audio_feature_length"]

        thinker_config = self.config
        tokens_per_second = getattr(thinker_config.vision_config, "tokens_per_second", 25)
        seconds_per_chunk = thinker_config.seconds_per_chunk
        t_ntoken_per_chunk = int(tokens_per_second * seconds_per_chunk)

        t_index = (np.arange(grid_t) * t_factor).astype(np.int64)
        t_index_split_chunk: list[list[int]] = [[] for _ in range((int(t_index.max()) // t_ntoken_per_chunk) + 1)]
        for t_val in t_index:
            idx = int(t_val) // t_ntoken_per_chunk
            t_index_split_chunk[idx].append(int(t_val))

        pure_audio_len = self._compute_audio_token_count(audio_len)
        added_audio_len = 0
        pos_ids_list: list[np.ndarray] = []
        audio_start_idx = start_idx

        for t_chunk in t_index_split_chunk:
            if not t_chunk:
                continue

            chunk_t = len(t_chunk)

            h_indices = np.tile(np.arange(grid_h).reshape(1, -1, 1), (chunk_t, 1, grid_w)).flatten()
            w_indices = np.tile(np.arange(grid_w).reshape(1, 1, -1), (chunk_t, grid_h, 1)).flatten()
            t_indices = np.repeat(np.array(t_chunk), grid_h * grid_w)

            vision_pos = np.stack([t_indices, h_indices, w_indices]) + start_idx
            pos_ids_list.append(vision_pos)

            audio_chunk_size = min(t_ntoken_per_chunk, pure_audio_len - added_audio_len)
            if audio_chunk_size > 0:
                audio_pos = np.broadcast_to(np.arange(audio_chunk_size), (3, audio_chunk_size)) + audio_start_idx
                pos_ids_list.append(audio_pos)
                audio_start_idx = audio_start_idx + audio_chunk_size
                added_audio_len += audio_chunk_size

        if added_audio_len < pure_audio_len:
            remaining = pure_audio_len - added_audio_len
            remaining_audio_pos = np.broadcast_to(np.arange(remaining), (3, remaining)) + audio_start_idx
            pos_ids_list.append(remaining_audio_pos)

        vision_tokens = grid_t * grid_h * grid_w
        total_tokens = vision_tokens + pure_audio_len

        return np.concatenate(pos_ids_list, axis=1), total_tokens

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        llm_pos_ids_list: list[np.ndarray] = []
        st = 0

        for offset, modality, data in self.iter_mm_features(mm_features):
            text_len = offset - st
            st_idx = int(llm_pos_ids_list[-1].max()) + 1 if llm_pos_ids_list else 0
            if text_len > 0:
                llm_pos_ids_list.append(np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx)
                st_idx += text_len

            if modality == "audio":
                audio_tokens = self._compute_audio_token_count(data["audio_feature_length"])
                llm_pos_ids_list.append(np.broadcast_to(np.arange(audio_tokens), (3, audio_tokens)) + st_idx)
                st = offset + audio_tokens

            elif modality == "image":
                grid_t = data["grid_t"]
                grid_h = data["grid_h"]
                grid_w = data["grid_w"]
                t_factor = data["t_factor"]

                grid_indices = np.indices((grid_t, grid_h, grid_w))
                if t_factor != 1.0:
                    grid_indices[0] = (grid_indices[0] * t_factor).astype(np.int64)
                llm_pos_ids_list.append(grid_indices.reshape(3, -1) + st_idx)
                st = offset + grid_t * grid_h * grid_w

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
                    st = offset + grid_t * grid_h * grid_w
                else:
                    pos_ids, token_count = self._compute_interleaved_positions(st_idx, data)
                    llm_pos_ids_list.append(pos_ids)
                    st = offset + token_count

        if st < len(input_tokens):
            st_idx = int(llm_pos_ids_list[-1].max()) + 1 if llm_pos_ids_list else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx)

        llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
        mrope_position_delta = int(llm_positions.max()) + 1 - len(input_tokens)

        return torch.from_numpy(llm_positions), mrope_position_delta

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            return []

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor corresponding to a multimodal data item (image or video).
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
    ) -> torch.Tensor:
        if multimodal_embeddings is None or is_multimodal is None:
            return super().embed_input_ids(input_ids)

        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.get_language_model().embed_input_ids,
            is_multimodal=is_multimodal,
        )

        if len(multimodal_embeddings) == 0:
            return inputs_embeds

        # Check for audio-in-video: interleaved video and audio tokens
        # in the multimodal region. Only use the interleaved path when
        # needed; otherwise fall back to the default parent implementation.
        video_token_id = self.config.video_token_index
        audio_token_id = self.config.audio_token_index

        input_ids_cpu = input_ids.cpu()
        is_video = is_multimodal & (input_ids_cpu == video_token_id)
        is_audio = is_multimodal & (input_ids_cpu == audio_token_id)

        num_video = is_video.sum().item()
        num_audio = is_audio.sum().item()

        if check_interleaved_audio_video(is_video, is_audio, num_video, num_audio):
            inputs_embeds = self._embed_text_input_ids(
                input_ids,
                self.get_language_model().embed_input_ids,
                is_multimodal=is_multimodal,
            )
            return merge_interleaved_embeddings(
                inputs_embeds,
                multimodal_embeddings,
                is_video,
                is_audio,
                is_multimodal,
                num_video,
                num_audio,
            )

        # Default: standard merge (no interleaving), same as parent class
        return super().embed_input_ids(
            input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        skip_prefixes = ["talker.", "token2wav."]
        if self.audio_tower is None:
            skip_prefixes.extend(["audio_tower."])
        if self.visual is None:
            skip_prefixes.extend(["visual."])

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=skip_prefixes,
        )
        loaded_weights = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

        return loaded_weights

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="merger.",
            tower_model=["visual.", "audio_tower."],
        )
