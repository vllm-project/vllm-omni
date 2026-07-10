"""Thin Omni wrapper: reuse upstream Qwen2.5-Omni thinker with minimal overrides."""

from collections.abc import Iterable, Iterator, Mapping
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
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import MultiModalDataItems, VideoProcessorItems
from vllm.multimodal.processing.processor import (
    MultiModalPromptUpdates,
    PlaceholderFeaturesInfo,
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


def _presampled_videos_hf_kwargs(
    mm_data: Mapping[str, object],
    mm_kwargs: Mapping[str, object],
) -> Mapping[str, object]:
    """Adjust HF video kwargs for videos pre-sampled by vLLM's video loader.

    When ``video_metadata`` is present (emitted by
    ``Qwen2_5OmniVideoProcessorItems``), the frames were already sampled
    according to ``media_io_kwargs``, so the HF processor must not re-sample
    them, and it needs the sampled fps (instead of its default) to compute
    ``video_second_per_grid`` correctly. Otherwise the audio/video temporal
    alignment breaks under ``use_audio_in_video=True``.
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

    # An explicit user-provided fps takes precedence.
    if "fps" not in videos_kwargs:
        fps_values = [_compute_sampled_video_fps(m) for m in video_metadata]
        # HF accepts a single fps per call and uses it for every video's
        # video_second_per_grid.
        known_fps = [fps for fps in fps_values if fps is not None]
        unique_fps = set(known_fps)
        if len(unique_fps) == 1:
            videos_kwargs["fps"] = known_fps[0]
        elif len(unique_fps) > 1:
            logger.warning(
                f"Mixed sampled FPS {sorted(unique_fps)} in one request; HF accepts a single fps, using {known_fps[0]}."
            )
            videos_kwargs["fps"] = known_fps[0]

    mm_kwargs["videos_kwargs"] = videos_kwargs
    return mm_kwargs


class Qwen2_5OmniVideoProcessorItems(VideoProcessorItems):
    """Video items that carry the loader's ``(frames, metadata)`` tuples.

    The tuples are kept in ``data`` so that mm hashing and cache-miss
    re-parsing retain the metadata; ``get_processor_data`` unpacks them into
    the ``videos`` + ``video_metadata`` arguments of the HF processor.
    """

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

        # Normalize to a list of per-video items (same as the base parser).
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

        # Metadata is optional: videos fetched by vLLM's media connector
        # arrive as (frames, metadata) tuples, while plain arrays (e.g.
        # offline `multi_modal_data` or dummy profiling inputs) carry no
        # metadata and parse as before.
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

    def _cached_apply_hf_processor(self, inputs, timing_ctx):
        # If use_audio_in_video, process video and audio together; otherwise,
        # skip cache to avoid errors.
        # Prevents partial cache hits from causing processing failures.
        if inputs.hf_processor_mm_kwargs.get("use_audio_in_video"):
            return self._apply_hf_processor(inputs, timing_ctx)
        return super()._cached_apply_hf_processor(inputs, timing_ctx)

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ):
        mm_kwargs = _presampled_videos_hf_kwargs(mm_data, mm_kwargs)
        return super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

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

        # Detect use_audio_in_video from mm_kwargs
        use_audio_in_video = False
        if "video" in mm_kwargs:
            for item in mm_kwargs["video"]:
                if item and item.get("use_audio_in_video"):
                    use_audio_in_video_tensor = item["use_audio_in_video"].data
                    if use_audio_in_video_tensor.numel() > 0:
                        use_audio_in_video = bool(use_audio_in_video_tensor.item())
                        break
            # for mutilmodality cache
            if any(item is None for item in mm_kwargs["video"]):
                video_token_id = self.info.get_hf_config().video_token_id
                audio_token_id = self.info.get_hf_config().audio_token_id
                video_audio_item_num = sum(id in (video_token_id, audio_token_id) for id in prompt_ids)
                audio_updates_num = len(mm_prompt_updates.get("audio", []))
                video_updates_num = len(mm_prompt_updates.get("video", []))
                if video_audio_item_num != video_updates_num + audio_updates_num:
                    use_audio_in_video = True

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
                mm_placeholders = self._derive_audio_from_video_placeholders(mm_placeholders, mm_prompt_updates)
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
