# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
# Copyright 2024 ANT Group and the HuggingFace Inc. team.
# Adapted from Ming repository modeling_bailingmm2.py and processing_bailingmm2.py
# https://github.com/inclusionAI/Ming

"""Ming-flash-omni-2.0 Thinker stage implementation (multimodal understanding)."""

import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Annotated, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VLImageInputs,
    Qwen2_5_VLImagePixelInputs,
    Qwen2_5_VLVideoInputs,
    Qwen2_5_VLVideoPixelInputs,
)
from vllm.model_executor.models.qwen2_vl import (
    Qwen2VLProcessingInfo,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    _merge_multimodal_embeddings,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    AudioProcessorItems,
    ImageProcessorItems,
    MultiModalDataItems,
    MultiModalDataParser,
    VideoProcessorItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from vllm_omni.model_executor.custom_process_mixin import CustomProcessMixin
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.transformers_utils.configs.ming_flash_omni import BailingMM2Config
from vllm_omni.transformers_utils.processors.ming import (
    PLACEHOLDER_AUDIO_TOKEN_IN_TEXT,
    PLACEHOLDER_IMAGE_TOKEN_IN_TEXT,
    PLACEHOLDER_VIDEO_TOKEN_IN_TEXT,
    MingFlashOmniProcessor,
    MingWhisperFeatureExtractor,
    raise_missing_video_processor,
)

from .audio_encoder import WhisperAudioEncoder
from .modeling_bailing_moe_v2 import BailingMoeV2ForCausalLM
from .projectors import AudioProjector, VisionProjector
from .vision_encoder import MingVisionEncoder

logger = init_logger(__name__)


class MingAudioInput(TensorSchema):
    """
    Dimensions:
        - b:  Batch size
        - l:  Total audio frames (clips concatenated along the time axis)
        - nm: Number of mel bins
        - N:  Max number of audio clips per batch item
    """

    audio_feats: Annotated[
        torch.Tensor,
        TensorShape("b", "l", "nm"),
    ]

    audio_feats_lengths: Annotated[
        torch.Tensor,
        TensorShape("b", "N"),
    ]


class MingFlashOmniThinkerProcessingInfo(Qwen2VLProcessingInfo):
    def get_hf_config(self) -> BailingMM2Config:
        return self.ctx.get_hf_config(BailingMM2Config)

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(MingFlashOmniProcessor, **kwargs)

    def get_target_channels(self) -> int:
        # See `_normalize_audio_tensor` in vllm_omni/transformers_utils/processors/ming.py
        return 1

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # ``img2img`` is the serving-layer alias for a reference image in the
        # image-gen path; treated as a regular image by ``_MingMultiModalDataParser``.
        return {"image": None, "video": None, "audio": None, "img2img": 1}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        mm_counts = mm_counts or {}
        requested_modalities = {m for m, c in mm_counts.items() if c > 0}
        mm_max_tokens: dict[str, int] = {}

        if requested_modalities & {"image", "video"}:
            vl_tokens = super().get_mm_max_tokens_per_item(
                seq_len=seq_len,
                mm_counts=mm_counts,
            )
            mm_max_tokens.update({m: vl_tokens[m] for m in ["image", "video"] if m in requested_modalities})

        if "audio" in requested_modalities:
            # TODO: consider computing from audio config
            mm_max_tokens["audio"] = 3000

        return mm_max_tokens

    def get_feature_extractor(self, **kwargs: object):
        """Return the audio feature extractor from the processor.

        The processor may be loaded via trust_remote_code paths that return
        a stock transformers ``WhisperFeatureExtractor`` rather than
        vllm-omni's subclass ``MingWhisperFeatureExtractor``. We accept both
        as long as the caller-needed attributes (``sampling_rate``) exist.
        """
        hf_processor = self.get_hf_processor(**kwargs)
        feature_extractor = hf_processor.audio_processor
        assert isinstance(feature_extractor, MingWhisperFeatureExtractor)
        return feature_extractor

    def get_data_parser(self):
        feature_extractor = self.get_feature_extractor()
        return _MingMultiModalDataParser(
            target_sr=feature_extractor.sampling_rate,
            target_channels=self.get_target_channels(),
            expected_hidden_size=self._get_expected_hidden_size(),
        )


class _MingMultiModalDataParser(MultiModalDataParser):
    """Rewrite ``img2img`` → ``image`` in-place so the thinker embeds the ref.

    Matches Ming's upstream (``processing_bailingmm2.py::__call__``): when
    ``images`` is provided the image_processor runs → pixel_values + image_grid_thw,
    and ``_expand_image_tokens`` expands ``<IMAGE>`` placeholders in the text
    to ``<imagePatch>*N``. The same PIL then ALSO goes to the diffusion stage
    as ``image_gen_pixel_values_reference``.

    On vllm-omni's serving path serving_chat emits
    ``multi_modal_data={"img2img": PIL}`` (Bagel convention). We mutate the
    caller's mm_data dict here — moving the PIL from the ``img2img`` key into
    the ``image`` key — so that:

      * vLLM's renderer validator sees consistent keys in
        ``mm_data`` and the parser output (``mm_data_items["image"]``)
      * The thinker-side image path (``_call_hf_processor`` reads
        ``mm_data["images"]``, ``_get_prompt_updates`` replaces ``<IMAGE>``
        placeholders, ``_get_mm_fields_config`` declares image field layout)
        all run normally with the ref PIL acting as a regular image input
      * ``thinker2imagegen`` still finds the ref PIL — but now under
        ``multi_modal_data["image"]`` rather than ``["img2img"]``.

    A companion prepend in ``_apply_hf_processor_main`` ensures the prompt
    text carries a matching ``<IMAGE>`` placeholder when needed.
    """

    def parse_mm_data(self, mm_data):
        if "img2img" in mm_data:
            img2img = mm_data.pop("img2img")
            if not isinstance(img2img, list):
                img2img = [img2img]
            existing = mm_data.get("image")
            if existing is None:
                existing_list: list[Any] = []
            elif isinstance(existing, list):
                existing_list = list(existing)
            else:
                existing_list = [existing]
            mm_data["image"] = existing_list + img2img
        return super().parse_mm_data(mm_data)


class MingFlashOmniThinkerDummyInputsBuilder(BaseDummyInputsBuilder[MingFlashOmniThinkerProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)
        num_audios = mm_counts.get("audio", 0)

        hf_processor = self.info.get_hf_processor()

        audio_token: str = hf_processor.audio_token
        image_token: str = hf_processor.image_token
        video_token: str = hf_processor.video_token

        return image_token * num_images + video_token * num_videos + audio_token * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)
        num_audios = mm_counts.get("audio", 0)

        # Default dimensions for dummy data
        image_width, image_height = 448, 448
        video_width, video_height = 448, 448
        num_frames = 8
        audio_duration = 3.0  # seconds
        sample_rate = 16000

        audio_length = int(audio_duration * sample_rate)

        num_img2img = mm_counts.get("img2img", 0)

        mm_data: MultiModalDataDict = {
            "image": self._get_dummy_images(
                width=image_width,
                height=image_height,
                num_images=num_images,
            ),
            "video": self._get_dummy_videos(
                width=video_width,
                height=video_height,
                num_frames=num_frames,
                num_videos=num_videos,
            ),
            "audio": [(np.random.randn(audio_length).astype(np.float32), sample_rate) for _ in range(num_audios)],
        }

        if num_img2img > 0:
            mm_data["img2img"] = self._get_dummy_images(
                width=image_width,
                height=image_height,
                num_images=num_img2img,
            )

        return mm_data


class MingFlashOmniThinkerMultiModalProcessor(BaseMultiModalProcessor[MingFlashOmniThinkerProcessingInfo]):
    """Multimodal processor for Ming-flash-omni Thinker stage.

    Handles preprocessing of 1) image, 2) video, and 3) audio inputs,
    and expands placeholder tokens to the correct number of patch tokens.
    """

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        tokenizer = self.info.get_tokenizer()
        # might want to add a fallback to resolve token ids
        # vocab = tokenizer.get_vocab()
        thinker_config = self.info.get_hf_config()

        # patch/delimiter token IDs (used in replacement sequences)
        image_start_token_id = thinker_config.llm_config.image_start_token
        image_patch_token_id = thinker_config.llm_config.image_patch_token
        image_end_token_id = thinker_config.llm_config.image_end_token

        video_start_token_id = thinker_config.llm_config.video_start_token
        frame_patch_token_id = thinker_config.llm_config.video_patch_token
        video_end_token_id = thinker_config.llm_config.video_end_token

        audio_start_token_id = thinker_config.llm_config.audio_start_token
        audio_patch_token_id = thinker_config.llm_config.audio_patch_token
        audio_end_token_id = thinker_config.llm_config.audio_end_token

        vision_config = thinker_config.vision_config
        spatial_merge_size = vision_config.spatial_merge_size if vision_config else 2

        newline_token_ids: list[int] = tokenizer.encode("\n", add_special_tokens=False)

        out_mm_data = out_mm_kwargs.get_data()

        def get_replacement_image(item_idx: int) -> PromptUpdateDetails:
            """Generate token sequence for an image."""
            grid_thw = out_mm_data.get("image_grid_thw")
            if grid_thw is None:
                raise ValueError(
                    "image_grid_thw missing from processor output; "
                    "cannot determine image patch count for prompt replacement."
                )
            if isinstance(grid_thw, torch.Tensor):
                thw = grid_thw[item_idx]
                num_patches = int(thw.prod().item()) // (spatial_merge_size**2)
            else:
                thw = grid_thw[item_idx]
                num_patches = (thw[0] * thw[1] * thw[2]) // (spatial_merge_size**2)

            # Build token sequence: <image> <imagePatch>*N </image> \n
            # the newline token is added in purpose from original model processing
            tokens: list[int] = []
            tokens.append(image_start_token_id)
            tokens.extend([image_patch_token_id] * num_patches)
            tokens.append(image_end_token_id)
            # Refer to Ming's BailingMM2Processor._expand_image_tokens
            # https://github.com/inclusionAI/Ming/blob/3954fcb880ff5e61ff128bcf7f1ec344d46a6fe3/processing_bailingmm2.py
            tokens.extend(newline_token_ids)

            # Only <imagePatch> tokens receive multimodal embeddings
            return PromptUpdateDetails.select_token_id(tokens, image_patch_token_id)

        def get_replacement_video(item_idx: int) -> PromptUpdateDetails:
            """Generate token sequence for a video."""
            grid_thw = out_mm_data.get("video_grid_thw", None)
            if grid_thw is None:
                raise ValueError(
                    "video_grid_thw missing from processor output; "
                    "cannot determine video patch count for prompt replacement."
                )
            if isinstance(grid_thw, torch.Tensor):
                thw = grid_thw[item_idx]
                num_patches = int(thw.prod().item()) // (spatial_merge_size**2)
            else:
                thw = grid_thw[item_idx]
                num_patches = (thw[0] * thw[1] * thw[2]) // (spatial_merge_size**2)

            # Build token sequence: <video> <framePatch>*N </video> \n
            # the newline token is added in purpose from original model processing
            tokens: list[int] = []
            tokens.append(video_start_token_id)
            tokens.extend([frame_patch_token_id] * num_patches)
            tokens.append(video_end_token_id)
            tokens.extend(newline_token_ids)

            # Only <framePatch> tokens receive multimodal embeddings
            return PromptUpdateDetails.select_token_id(tokens, frame_patch_token_id)

        def get_replacement_audio(item_idx: int) -> PromptUpdateDetails:
            """Generate token sequence for an audio."""
            encoder_feats_lengths = out_mm_data.get("encoder_feats_lengths", None)
            if encoder_feats_lengths is None:
                raise ValueError(
                    "encoder_feats_lengths missing from processor output; "
                    "cannot determine audio patch count for prompt replacement."
                )
            if isinstance(encoder_feats_lengths, torch.Tensor):
                num_patches = int(encoder_feats_lengths[item_idx].item())
            else:
                num_patches = encoder_feats_lengths[item_idx]

            # Build token sequence: <audio> <audioPatch>*N </audio>
            tokens: list[int] = []
            tokens.append(audio_start_token_id)
            tokens.extend([audio_patch_token_id] * num_patches)
            tokens.append(audio_end_token_id)

            # Only <audioPatch> tokens receive multimodal embeddings
            return PromptUpdateDetails.select_token_id(tokens, audio_patch_token_id)

        # Build prompt updates and process replacement
        updates: list[PromptUpdate] = []

        if "image" in mm_items and mm_items.get_items("image", ImageProcessorItems):
            updates.append(
                PromptReplacement(
                    modality="image",
                    target=PLACEHOLDER_IMAGE_TOKEN_IN_TEXT,
                    replacement=get_replacement_image,
                )
            )
        if "video" in mm_items and mm_items.get_items("video", VideoProcessorItems):
            updates.append(
                PromptReplacement(
                    modality="video",
                    target=PLACEHOLDER_VIDEO_TOKEN_IN_TEXT,
                    replacement=get_replacement_video,
                )
            )
        if "audio" in mm_items and mm_items.get_items("audio", AudioProcessorItems):
            updates.append(
                PromptReplacement(
                    modality="audio",
                    target=PLACEHOLDER_AUDIO_TOKEN_IN_TEXT,
                    replacement=get_replacement_audio,
                )
            )
        return updates

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        config: dict[str, MultiModalFieldConfig] = {}

        # Image fields, pixel_values is flat (concatenated patches from all images)
        image_grid_thw = hf_inputs.get("image_grid_thw", torch.empty((0, 3)))
        if "pixel_values" in hf_inputs:
            image_sizes = image_grid_thw.prod(-1)
            config["pixel_values"] = MultiModalFieldConfig.flat_from_sizes(
                "image",
                image_sizes,
            )
        if "image_grid_thw" in hf_inputs:
            config["image_grid_thw"] = MultiModalFieldConfig.batched("image")

        # Video fields, same flat layout as images
        video_grid_thw = hf_inputs.get("video_grid_thw", torch.empty((0, 3)))
        if "pixel_values_videos" in hf_inputs:
            video_sizes = video_grid_thw.prod(-1)
            config["pixel_values_videos"] = MultiModalFieldConfig.flat_from_sizes(
                "video",
                video_sizes,
            )
        if "video_grid_thw" in hf_inputs:
            config["video_grid_thw"] = MultiModalFieldConfig.batched("video")

        # Audio fields
        if "audio_feats" in hf_inputs:
            config["audio_feats"] = MultiModalFieldConfig.batched("audio")
        if "audio_feats_lengths" in hf_inputs:
            config["audio_feats_lengths"] = MultiModalFieldConfig.batched("audio")
        if "encoder_feats_lengths" in hf_inputs:
            config["encoder_feats_lengths"] = MultiModalFieldConfig.batched("audio")
        if "placeholder_audio_loc_lens" in hf_inputs:
            config["placeholder_audio_loc_lens"] = MultiModalFieldConfig.batched("audio")

        return config

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return False

    def _apply_hf_processor_main(
        self,
        prompt: str | list[int],
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
        *,
        enable_hf_prompt_update: bool,
    ):
        """Apply Ming-specific prompt rewrites before delegating to vllm's
        default text / multimodal processing pipeline.

        This is the single point in the base MM-processor call graph where we
        still have BOTH the full prompt and the request-level
        ``hf_processor_mm_kwargs`` together. Downstream, vllm's MM caching
        design splits the work into
        :meth:`_apply_hf_processor_text_only` (prompt, empty kwargs) and
        :meth:`_apply_hf_processor_mm_only` (dummy text, real kwargs), so a
        per-request flag like ``modalities`` cannot reach the real
        tokenization from inside :meth:`_call_hf_processor`.

        For Ming-flash-omni-2.0, image-generation requests carry
        ``mm_processor_kwargs["modalities"]`` set to ``["image"]`` (t2i) or
        ``["img2img"]`` (image edit). We expand the prompt here with the
        ``<image><imagePatch>*N</image>`` query-token block so the expanded
        form flows through both the tokenization and (eventual) MM paths.
        For img2img, we also prepend ``<IMAGE>`` so the thinker-side prompt
        replacement can locate the reference-image placeholder.
        """
        modalities = hf_processor_mm_kwargs.get("modalities") or []
        is_image_gen = "image" in modalities or "img2img" in modalities
        if isinstance(prompt, str) and is_image_gen:
            from vllm_omni.model_executor.models.ming_flash_omni.prompt_utils import (
                DEFAULT_NUM_QUERY_TOKENS,
                maybe_expand_image_gen_prompt,
            )

            # img2img: prepend ``<IMAGE>`` so the ref-image placeholder survives
            # MM caching (when the ref image is cache-warm, the thinker sees 0
            # missing items and cannot inject the placeholder itself).
            if "img2img" in modalities:
                prompt = "<IMAGE>" + prompt

            ig = getattr(self.info.ctx.model_config.hf_config, "image_gen_config", None)
            num_query_tokens = getattr(ig, "num_query_tokens", DEFAULT_NUM_QUERY_TOKENS)
            prompt = maybe_expand_image_gen_prompt(prompt, num_query_tokens=int(num_query_tokens))

        return super()._apply_hf_processor_main(
            prompt=prompt,
            mm_items=mm_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            tokenization_kwargs=tokenization_kwargs,
            enable_hf_prompt_update=enable_hf_prompt_update,
        )

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        """Call sub-processors for multimodal inputs and tokenize.

        We call the image/audio sub-processors directly (instead of going
        through `MingFlashOmniProcessor.__call__`) so that the high-level
        placeholder tokens remain **unexpanded** in the tokenized output.

        NOTE: Image-gen prompt expansion is handled in
        :meth:`_apply_hf_processor_main` — by the time we get here, vllm's
        MM caching has already split text/mm processing and ``mm_kwargs``
        is empty on the text-only branch.
        """
        hf_processor = self.info.get_hf_processor()
        tokenizer = self.info.get_tokenizer()

        data: dict[str, object] = {}

        images = mm_data.get("images", None)
        if images is not None:
            image_outputs = hf_processor.image_processor(
                images=images,
                return_tensors="pt",
            )
            data.update(image_outputs)

        videos = mm_data.get("videos", None)
        if videos is not None:
            video_processor = getattr(hf_processor, "video_processor", None)
            if video_processor is not None:
                video_outputs = video_processor(
                    videos=videos,
                    return_tensors="pt",
                )
            else:
                raise_missing_video_processor()
            # Rename keys to distinguish from images
            if "pixel_values" in video_outputs:
                video_outputs["pixel_values_videos"] = video_outputs.pop("pixel_values")
            if "image_grid_thw" in video_outputs:
                video_outputs["video_grid_thw"] = video_outputs.pop("image_grid_thw")
            data.update(video_outputs)

        audios = mm_data.get("audios", None)
        if audios is not None:
            # vLLM's AudioProcessorItems provides raw numpy arrays (already resampled).
            # MingWhisperAudioProcessor expects (waveform, sr) tuples,
            # so wrap them with the target sample rate.
            target_sr = hf_processor.audio_processor.sampling_rate
            audio_tuples = [(a, target_sr) if not isinstance(a, tuple) else a for a in audios]

            audio_outputs = hf_processor.audio_processor(
                audio_tuples,
                return_tensors="pt",
            )
            data.update(audio_outputs)

        # Tokenize text with placeholders still intact
        text_outputs = tokenizer(prompt, return_tensors="pt", **tok_kwargs)
        data.update(text_outputs)

        return BatchFeature(data=data)


@MULTIMODAL_REGISTRY.register_processor(
    MingFlashOmniThinkerMultiModalProcessor,
    info=MingFlashOmniThinkerProcessingInfo,
    dummy_inputs=MingFlashOmniThinkerDummyInputsBuilder,
)
class MingFlashOmniThinkerForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsMRoPE,
    CustomProcessMixin,
):
    """Ming Thinker stage: multimodal understanding
    (text + image + video + audio) -> text generation.
    """

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"model.": "language_model."},
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        # vllm_omni/transformers_utils/processors/ming.py
        if modality.startswith("image"):
            return "<IMAGE>"
        elif modality.startswith("video"):
            return "<VIDEO>"
        elif modality.startswith("audio"):
            return "<AUDIO>"
        elif modality == "img2img":
            # Ming's img2img ref image is consumed only by the diffusion stage
            # (``thinker2imagegen`` reads the raw PIL). No text-side placeholder
            # is emitted for the thinker.
            return None

        raise ValueError("Only image, video, audio, or img2img modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config

        thinker_config: BailingMM2Config = config
        if (
            thinker_config.llm_config is None
            or thinker_config.vision_config is None
            or thinker_config.audio_config is None
        ):
            raise ValueError(
                "MingFlashOmniThinker requires `llm_config`, `vision_config`, and `audio_config` in `thinker_config`."
            )

        llm_config = thinker_config.llm_config

        self.config = llm_config
        self.thinker_config = thinker_config
        self.have_multimodal_outputs = True

        # Initialize LLM as a component
        with self._mark_language_model(vllm_config):
            llm_vllm_config = vllm_config.with_hf_config(llm_config)
            self.language_model = BailingMoeV2ForCausalLM(
                vllm_config=llm_vllm_config, prefix=maybe_prefix(prefix, "llm")
            )

        # Ming thinker is inherently multimodal; initialize both towers eagerly.
        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.vision = MingVisionEncoder(
                vision_config=thinker_config.vision_config,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "vision"),
            )
            self.linear_proj = VisionProjector(
                vision_dim=self.vision.image_emb_dim,
                llm_dim=llm_config.hidden_size,
                mlp_depth=getattr(thinker_config, "mlp_depth", 2),
            )
        logger.info("Initialized MingVisionEncoder and VisionProjector")

        audio_cfg = thinker_config.audio_config
        whisper_cfg = getattr(audio_cfg, "whisper_encoder_config", {}) or {}
        with self._mark_tower_model(vllm_config, "audio"):
            self.audio = WhisperAudioEncoder(
                **whisper_cfg,
                use_flash_attn=True,
            )
            self.linear_proj_audio = AudioProjector(
                audio_dim=self.audio.audio_emb_dim,
                llm_dim=llm_config.hidden_size,
                ds_kernel_size=getattr(audio_cfg, "ds_kernel_size", 3),
                ds_stride=getattr(audio_cfg, "ds_stride", 2),
                mlp_depth=getattr(thinker_config, "mlp_depth", 1),
            )
        logger.info("Initialized WhisperAudioEncoder and AudioProjector")

        # Image-generation learnable query tokens. Stored as a ParameterDict
        # keyed by scale name (e.g. "16x16" -> [256, hidden_size]) to mirror
        # the upstream Ming layout in ``mlp/query_tokens_dict.<scale>``. These
        # are substituted in at ``<imagePatch>`` token positions during
        # text-to-image requests; image-comprehension requests continue to
        # fill those positions with vision features as before.
        self.query_tokens_dict = nn.ParameterDict()
        self._load_image_gen_query_tokens(vllm_config.model_config.model)

        # Expose interfaces
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

        logger.info(
            f"MingFlashOmniThinker initialized with: "
            f"vision={'yes' if self.vision else 'no'}, "
            f"audio={'yes' if self.audio else 'no'}, "
            f"image_gen_scales={list(self.query_tokens_dict.keys())}"
        )

    def _load_image_gen_query_tokens(self, model_path: str) -> None:
        """Load learnable query tokens used for image generation.

        Expected layout (observed on inclusionAI/Ming-flash-omni-2.0):
            <model_path>/mlp/model.safetensors
                key ``query_tokens_dict.<scale>`` -> [num_tokens, hidden_size]
                e.g. ``query_tokens_dict.16x16`` -> [256, 4096]

        Missing files are logged as a warning and image-gen is disabled
        (text-only inference still works).
        """
        from pathlib import Path

        p = Path(model_path) / "mlp" / "model.safetensors"
        if not p.exists() and not os.path.exists(model_path):
            from vllm_omni.model_executor.model_loader.weight_utils import (
                download_weights_from_hf_specific,
            )

            try:
                local_path = download_weights_from_hf_specific(model_path, None, ["mlp/*"])
                p = Path(local_path) / "mlp" / "model.safetensors"
            except Exception:
                logger.exception("[MingFlashOmniThinker] failed to download mlp/ from %s", model_path)
        if not p.exists():
            logger.warning(
                "[MingFlashOmniThinker] mlp/model.safetensors not found at %s; "
                "text-to-image will be disabled (text-only inference still works).",
                p,
            )
            return
        try:
            from safetensors.torch import load_file  # type: ignore

            state = load_file(str(p))
        except Exception:
            logger.exception("[MingFlashOmniThinker] failed to read %s", p)
            return
        loaded = 0
        for key, tensor in state.items():
            if not key.startswith("query_tokens_dict."):
                continue
            scale_name = key.split(".", 1)[1]
            param = nn.Parameter(tensor.clone(), requires_grad=False)
            self.query_tokens_dict[scale_name] = param
            loaded += 1
            logger.info(
                "[MingFlashOmniThinker] loaded query_tokens_dict[%s] shape=%s dtype=%s",
                scale_name,
                tuple(tensor.shape),
                tensor.dtype,
            )
        if loaded == 0:
            logger.warning("[MingFlashOmniThinker] no query_tokens_dict.* keys in mlp/model.safetensors")

    def extract_image_feature(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """Extract and project image features.

        Args:
            pixel_values: Flattened pixel values from vision processor.
            grid_thw: [num_images, 3] tensor of (t, h, w) grid dimensions.

        Returns:
            [seq_len, hidden_size] L2-normalized image embeddings.
        """
        if self.vision is None:
            raise ValueError("Vision encoder not initialized")

        with torch.amp.autocast(pixel_values.device.type, dtype=torch.bfloat16):
            image_embeds = self.vision(pixel_values, grid_thw=grid_thw)

        if self.vision.use_deepstack:
            image_embeds = image_embeds[:, : self.vision.image_emb_dim]

        image_embeds = self.linear_proj(image_embeds)
        image_embeds = F.normalize(image_embeds, dim=-1)
        return image_embeds

    def extract_audio_feature(
        self, audio_feats: torch.Tensor, audio_feats_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Extract and project audio features.

        Args:
            audio_feats: [B, L_total, n_mels] wrapped mel features — multiple audio
                clips per batch item are concatenated along the time dimension
                (time-first, as produced by MingWhisperFeatureExtractor).
            audio_feats_lengths: [B, N] lengths of each audio clip per batch item.
                N is the max number of clips per item; zero-padded entries are skipped.

        Returns:
            Tuple of per-clip [T'_i, hidden_size] projected audio embeddings.
        """
        if self.audio is None:
            raise ValueError("Audio encoder not initialized")

        # Unwrap packed [B, L_total, n_mels] into a list of [n_mels, T'_i] tensors,
        # one per audio clip, as expected by WhisperAudioEncoder.
        x_list: list[torch.Tensor] = []
        audio_lens: list[int] = []
        for i in range(audio_feats_lengths.shape[0]):
            feat_index = 0
            for j in range(audio_feats_lengths.shape[1]):
                feat_len = int(audio_feats_lengths[i, j].item())
                if feat_len == 0:
                    break
                mel_seg = audio_feats[i, feat_index : feat_index + feat_len].transpose(0, 1)
                x_list.append(mel_seg)
                audio_lens.append(feat_len)
                feat_index += feat_len

        audio_packed = self.audio(x_list, audio_lens)

        # Compute per-clip lengths after Whisper Conv1d (kernel=3, stride=2, pad=1)
        encoded_lens = [(audio_len - 3 + 2) // 2 + 1 for audio_len in audio_lens]

        # Project packed
        proj_packed, proj_lens = self.linear_proj_audio.forward_packed(audio_packed, encoded_lens)

        normalize = getattr(self.thinker_config.audio_config, "norm_query_embeds", False)
        if normalize:
            proj_packed = F.normalize(proj_packed, dim=-1)

        proj_packed = proj_packed.to(audio_feats.dtype)

        # Split into per-clip tensors
        return proj_packed.split(proj_lens)

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        """Parse and validate multimodal kwargs into per-modality dicts."""
        mm_input_by_modality: dict[str, Qwen2_5_VLImageInputs | Qwen2_5_VLVideoInputs | MingAudioInput] = {}

        for key in kwargs:
            if key == "pixel_values" and "image" not in mm_input_by_modality:
                pixel_values = kwargs.get("pixel_values")
                image_grid_thw = kwargs.get("image_grid_thw")
                if pixel_values is not None and image_grid_thw is not None:
                    mm_input_by_modality["image"] = Qwen2_5_VLImagePixelInputs(
                        type="pixel_values",
                        pixel_values=pixel_values,  # type: ignore[arg-type]
                        image_grid_thw=image_grid_thw,  # type: ignore[arg-type]
                    )
            elif key == "pixel_values_videos" and "video" not in mm_input_by_modality:
                pixel_values_videos = kwargs.get("pixel_values_videos")
                video_grid_thw = kwargs.get("video_grid_thw")
                second_per_grid_ts = kwargs.get("second_per_grid_ts")
                if pixel_values_videos is not None and video_grid_thw is not None:
                    mm_input_by_modality["video"] = Qwen2_5_VLVideoPixelInputs(
                        type="pixel_values_videos",
                        pixel_values_videos=pixel_values_videos,  # type: ignore[arg-type]
                        video_grid_thw=video_grid_thw,  # type: ignore[arg-type]
                        second_per_grid_ts=second_per_grid_ts,  # type: ignore[arg-type]
                    )
            elif key == "audio_feats" and "audio" not in mm_input_by_modality:
                audio_feats = kwargs.get("audio_feats")
                audio_feats_lengths = kwargs.get("audio_feats_lengths")
                if audio_feats is not None and audio_feats_lengths is not None:
                    mm_input_by_modality["audio"] = MingAudioInput(
                        audio_feats=audio_feats,  # type: ignore[arg-type]
                        audio_feats_lengths=audio_feats_lengths,  # type: ignore[arg-type]
                    )

        return mm_input_by_modality

    def _process_image_input(self, image_input: Qwen2_5_VLImageInputs) -> list[torch.Tensor]:
        # Splits the flat [total_tokens, D] output of extract_image_feature
        # into one tensor per image.
        pixel_values = image_input["pixel_values"]
        image_grid_thw = image_input["image_grid_thw"]
        image_embeds = self.extract_image_feature(pixel_values, image_grid_thw)
        merge_unit = self.thinker_config.vision_config.spatial_merge_size**2
        sizes = (image_grid_thw.prod(dim=-1) // merge_unit).tolist()
        return list(image_embeds.split([int(s) for s in sizes], dim=0))

    def _process_video_input(self, video_input: Qwen2_5_VLVideoInputs) -> list[torch.Tensor]:
        pixel_values_videos = video_input["pixel_values_videos"]
        video_grid_thw = video_input["video_grid_thw"]
        video_embeds = self.extract_image_feature(pixel_values_videos, video_grid_thw)
        merge_unit = self.thinker_config.vision_config.spatial_merge_size**2
        sizes = (video_grid_thw.prod(dim=-1) // merge_unit).tolist()
        return list(video_embeds.split([int(s) for s in sizes], dim=0))

    def _process_audio_input(self, audio_input: MingAudioInput) -> list[torch.Tensor]:
        return list(self.extract_audio_feature(audio_input["audio_feats"], audio_input["audio_feats_lengths"]))

    def _compute_modality_masks(self, input_ids: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Compute vision and audio MoE-routing masks from input_ids.

        Returns:
            Tuple of (vision_mask, audio_mask), each shape [seq_len] bool.
        """
        llm_config = self.config

        # vision mask
        vision_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        image_token = llm_config.image_patch_token
        video_token = llm_config.video_patch_token
        vision_mask = vision_mask | (input_ids == image_token)
        vision_mask = vision_mask | (input_ids == video_token)

        # audio mask
        audio_token = llm_config.audio_patch_token
        audio_mask = input_ids == audio_token

        return vision_mask, audio_mask

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            return []

        # preserve the order of modalities
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        for modality, mm_input in mm_input_by_modality.items():
            if modality == "image":
                multimodal_embeddings += tuple(self._process_image_input(mm_input))  # type: ignore[arg-type]
            elif modality == "video":
                multimodal_embeddings += tuple(self._process_video_input(mm_input))  # type: ignore[arg-type]
            elif modality == "audio":
                multimodal_embeddings += tuple(self._process_audio_input(mm_input))  # type: ignore[arg-type]

        return multimodal_embeddings

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.model.word_embeddings(input_ids)

        # Step 1: if vision/audio features were extracted, merge them into
        # inputs_embeds at their <patch> positions first. This fills ref-image
        # and other normal vision-input positions with real features.
        if multimodal_embeddings is not None and len(multimodal_embeddings) > 0:
            assert is_multimodal is not None, "`is_multimodal` mask required when `multimodal_embeddings` provided"
            inputs_embeds = _merge_multimodal_embeddings(
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=multimodal_embeddings,
                is_multimodal=is_multimodal,
            )

        # Step 2: image-gen query tokens are appended at the tail of the prompt
        # as a contiguous block of ``<imagePatch>`` tokens. Inject the learnable
        # ``query_tokens_dict`` entries at those positions. Running AFTER step 1
        # is safe because the query-token block is disjoint from ref-image
        # patch positions (ref patches are earlier in the prompt, filled by
        # step 1; query tokens occupy the trailing block and get overwritten
        # here). Injecting always — including when multimodal embeddings are
        # present — matches Ming's img2img-edit flow where BOTH a real image
        # input AND the learnable query tokens need to reach the LLM.
        if len(self.query_tokens_dict) > 0:
            self._maybe_inject_image_gen_query_embeds(input_ids, inputs_embeds)
        return inputs_embeds

    def _maybe_inject_image_gen_query_embeds(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> None:
        """Overwrite the trailing ``<imagePatch>`` block with learnable query tokens.

        The image-gen query-token block is appended at the tail of the prompt
        by ``maybe_expand_image_gen_prompt`` as literal text
        ``<image><imagePatch>*N</image>``. After tokenization:

            ... <image_start_token> <imagePatch>*N <image_end_token>

        is the exact trailing pattern. We verify that signature before
        injecting — without the check, a pure vision-understanding request
        whose final ref-image block ends in N or more ``<imagePatch>`` tokens
        would be misidentified as image-gen and its vision features silently
        overwritten. Decode-phase calls (``input_ids`` is one next-token)
        naturally fail the signature check and return early.
        """
        if input_ids.numel() == 0:
            return
        patch_token_id = self.config.image_patch_token
        image_end_token_id = getattr(self.config, "image_end_token", None)
        if image_end_token_id is None:
            return

        scale_items = list(self.query_tokens_dict.items())
        if not scale_items:
            return
        scale_name, query_param = scale_items[-1]
        num_query_tokens = int(query_param.shape[0])

        flat = input_ids.view(-1)
        L = int(flat.numel())
        # Need at least num_query_tokens patch tokens + 1 <image_end> token.
        if L < num_query_tokens + 1:
            return

        # Signature: the prompt must end with <image_end_token>, and the
        # N positions immediately before it must all be <imagePatch>.
        if int(flat[-1].item()) != image_end_token_id:
            return
        tail_start = L - 1 - num_query_tokens
        tail_end = L - 1  # exclusive — last index is <image_end_token>
        if not (flat[tail_start:tail_end] == patch_token_id).all():
            return

        query_embeds = query_param.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        inputs_embeds[tail_start:tail_end] = query_embeds
        logger.debug(
            "[MingFlashOmniThinker] injected %d image-gen query embeddings (scale=%s) at positions [%d, %d)",
            num_query_tokens,
            scale_name,
            tail_start,
            tail_end,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> OmniOutput:
        # Compute MoE modality masks on every device
        image_mask, audio_mask = self._compute_modality_masks(input_ids)
        hidden_states = self.language_model.forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            image_mask=image_mask,
            audio_mask=audio_mask,
        )

        # Capture embeddings for downstream stages
        multimodal_outputs = {
            "final_hidden_states": hidden_states,
        }

        return OmniOutput(
            text_hidden_states=hidden_states,
            multimodal_outputs=multimodal_outputs,
        )

    def compute_logits(self, hidden_states: torch.Tensor, sampling_metadata) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states, sampling_metadata)

    def sample(self, logits: torch.Tensor, sampling_metadata):
        return self.language_model.sample(logits, sampling_metadata)

    @property
    def sampler(self):
        return self.language_model.sampler

    def iter_mm_features(
        self,
        mm_features: list[MultiModalFeatureSpec],
    ) -> Iterator[tuple[int, str, dict[str, Any]]]:
        """Iterate over image/video features sorted by token position.

        Yields: (offset, modality, feature_data) where feature_data contains:
        - image: {"grid_t", "grid_h", "grid_w", "second_per_grid_t"}
        - video: {"grid_t", "grid_h", "grid_w", "second_per_grid_t"}

        Audio features are not yielded: Ming assigns them sequential
        text positions (same T/H/W value) rather than 3D grid positions.
        """
        spatial_merge_size = self.config.spatial_merge_size

        for mm_feature in sorted(mm_features, key=lambda f: f.mm_position.offset):
            if mm_feature.data is None:
                continue

            offset = mm_feature.mm_position.offset
            modality = mm_feature.modality

            if modality == "image":
                t, h, w = mm_feature.data["image_grid_thw"].data.tolist()
                yield (
                    offset,
                    "image",
                    {
                        "grid_t": int(t),
                        "grid_h": int(h) // spatial_merge_size,
                        "grid_w": int(w) // spatial_merge_size,
                        "second_per_grid_t": 0.0,
                    },
                )
            elif modality == "video":
                t, h, w = mm_feature.data["video_grid_thw"].data.tolist()
                second_per_grid_t = 1.0
                spgt_field = mm_feature.data.get("second_per_grid_ts")
                if spgt_field is not None:
                    second_per_grid_t = float(spgt_field.data.item())
                yield (
                    offset,
                    "video",
                    {
                        "grid_t": int(t),
                        "grid_h": int(h) // spatial_merge_size,
                        "grid_w": int(w) // spatial_merge_size,
                        "second_per_grid_t": second_per_grid_t,
                    },
                )

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec] | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, int]:
        """Compute M-RoPE input positions using mm_features directly."""
        llm_config = self.config
        tokens_per_second: int = getattr(llm_config, "tokens_per_second", 2)
        seq_len = len(input_tokens)

        llm_pos_ids_list: list[np.ndarray] = []
        st = 0  # index of next unprocessed token

        for patch_offset, _modality, data in self.iter_mm_features(mm_features or []):
            text_len = patch_offset - st
            st_idx = int(llm_pos_ids_list[-1].max()) + 1 if llm_pos_ids_list else 0
            if text_len > 0:
                llm_pos_ids_list.append(np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx)
                st_idx += text_len

            # 3-D grid positions for patch tokens
            grid_t: int = data["grid_t"]
            grid_h: int = data["grid_h"]
            grid_w: int = data["grid_w"]
            second_per_grid_t: float = data["second_per_grid_t"]

            t_raw = np.arange(grid_t)
            if second_per_grid_t > 0:
                t_index = (t_raw * second_per_grid_t * tokens_per_second).astype(np.int64)
            else:
                t_index = t_raw.astype(np.int64)
            t_index = np.repeat(t_index, grid_h * grid_w)

            h_index = np.tile(np.arange(grid_h).repeat(grid_w), grid_t)
            w_index = np.tile(np.arange(grid_w), grid_t * grid_h)

            llm_pos_ids_list.append(np.stack([t_index, h_index, w_index]) + st_idx)

            num_patches = grid_t * grid_h * grid_w
            st = patch_offset + num_patches

        if st < seq_len:
            st_idx = int(llm_pos_ids_list[-1].max()) + 1 if llm_pos_ids_list else 0
            tail_len = seq_len - st
            llm_pos_ids_list.append(np.broadcast_to(np.arange(tail_len), (3, tail_len)) + st_idx)

        if llm_pos_ids_list:
            position_ids = torch.from_numpy(np.concatenate(llm_pos_ids_list, axis=1).astype(np.int64))  # (3, seq_len)
        else:
            # text-only, simple sequential positions
            position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(3, -1)

        mrope_position_delta = int(position_ids.max().item()) + 1 - seq_len
        return position_ids, mrope_position_delta

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
