"""Thin Omni wrapper: reuse upstream Qwen2.5-Omni thinker (v0.14) with minimal overrides."""

from collections.abc import Iterable
from typing import Any

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
    Qwen2_5OmniThinkerMultiModalProcessor,
    Qwen2_5OmniThinkerProcessingInfo,
    check_interleaved_audio_video,
    merge_interleaved_embeddings,
)
from vllm.model_executor.models.qwen2_5_omni_thinker import (
    Qwen2_5OmniConditionalGenerationMixin as Qwen2_5OmniConditionalGenerationMixinBase,
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
    _merge_multimodal_embeddings,
    init_vllm_registered_model,
    maybe_prefix,
    split_list_into_ranges,
)
from vllm.model_executor.models.vision import get_llm_pos_ids_for_vision
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
)
from vllm.sequence import IntermediateTensors

try:
    import flash_attn
except (ImportError, ModuleNotFoundError):
    flash_attn = None
logger = init_logger(__name__)


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
        video_hashes: list[str] = None,
        cached_video_embeds: torch.Tensor = None,
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
                    quant_config=quant_config,
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

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        """
        Example:

            (V_i are vision position ids, A_i are audio position ids)

            |V_1 ...    V_n|A_1 ...   A_n|V_n+1 ... V_2n|A_n+1 ... A_2n|...
            |vision chunk 1|audio chunk 1|vision chunk 2|audio chunk 2 |...
        """
        kwargs = MultiModalFeatureSpec.gather_kwargs(
            mm_features,
            {
                "image_grid_thw",
                "video_grid_thw",
                "second_per_grid_ts",
                "audio_feature_lengths",
                "use_audio_in_video",
            },
        )
        image_grid_thw = kwargs.get("image_grid_thw", [])
        video_grid_thw = kwargs.get("video_grid_thw", [])
        second_per_grid_ts = kwargs.get("second_per_grid_ts", [])
        audio_feature_lengths = kwargs.get("audio_feature_lengths", [])
        use_audio_in_video = any(kwargs.get("use_audio_in_video", []))

        image_grid_thw = (torch.stack if image_grid_thw else torch.tensor)(image_grid_thw)
        video_grid_thw = (torch.stack if video_grid_thw else torch.tensor)(video_grid_thw)

        # TODO(fyabc): refactor and share more code with
        #  _vl_get_input_positions_tensor.

        thinker_config = self.config
        audio_token_id = thinker_config.audio_token_index
        image_token_id = thinker_config.image_token_index
        video_token_id = thinker_config.video_token_index
        audio_start_token_id = thinker_config.audio_start_token_id
        audio_end_token_id = thinker_config.audio_end_token_id
        vision_start_token_id = thinker_config.vision_start_token_id
        vision_end_token_id = thinker_config.vision_end_token_id
        seconds_per_chunk = thinker_config.seconds_per_chunk
        spatial_merge_size = thinker_config.vision_config.spatial_merge_size
        tokens_per_second = getattr(thinker_config.vision_config, "tokens_per_second", 25)

        src_item = input_tokens
        audio_seqlens = audio_feature_lengths
        if not second_per_grid_ts:
            second_per_grid_ts = [1] * video_grid_thw.shape[0]
        audio_idx = 0
        video_idx = 0
        image_idx = 0
        new_src_item: list[int] = []
        llm_pos_ids_list: list[torch.Tensor] = []

        idx = 0
        while idx < len(src_item):
            new_src_item_len = len(new_src_item)
            start_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            if src_item[idx] not in [audio_token_id, video_token_id, image_token_id]:
                if use_audio_in_video and idx > 0:
                    if src_item[idx] == vision_end_token_id and src_item[idx - 1] == audio_end_token_id:
                        # processing the <|audio_eos|> before <|vision_eos|>
                        start_idx -= 1
                    elif src_item[idx] == audio_start_token_id and src_item[idx - 1] == vision_start_token_id:
                        # processing the <|audio_bos|> after <|vision_eos|>
                        start_idx -= 1
                new_src_item.append(src_item[idx])
                llm_pos_ids = torch.tensor([start_idx], dtype=torch.long).expand(3, -1)
                llm_pos_ids_list.append(llm_pos_ids)
            elif src_item[idx] == audio_token_id:
                assert audio_seqlens is not None
                audio_seqlen = audio_seqlens[audio_idx]
                place_num = ((audio_seqlen - 1) // 2 + 1 - 2) // 2 + 1
                new_src_item.extend([audio_token_id] * place_num)
                llm_pos_ids = torch.arange(place_num).expand(3, -1) + start_idx
                llm_pos_ids_list.append(llm_pos_ids)
                audio_idx += 1
            elif src_item[idx] == image_token_id:
                grid_t = image_grid_thw[image_idx][0]
                grid_hs = image_grid_thw[:, 1]
                grid_ws = image_grid_thw[:, 2]
                t_index = (torch.arange(grid_t) * 1 * tokens_per_second).long()
                llm_pos_ids = get_llm_pos_ids_for_vision(
                    start_idx, image_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                )
                llm_pos_ids_list.append(llm_pos_ids)
                vision_seqlen = image_grid_thw[image_idx].prod() // (spatial_merge_size**2)
                new_src_item.extend([image_token_id] * vision_seqlen)
                image_idx += 1
            elif src_item[idx] == video_token_id and not use_audio_in_video:
                grid_t = video_grid_thw[video_idx][0]
                grid_hs = video_grid_thw[:, 1]
                grid_ws = video_grid_thw[:, 2]
                t_index = (torch.arange(grid_t) * second_per_grid_ts[video_idx] * tokens_per_second).long()
                llm_pos_ids = get_llm_pos_ids_for_vision(
                    start_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                )
                llm_pos_ids_list.append(llm_pos_ids)
                vision_seqlen = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)
                new_src_item.extend([video_token_id] * vision_seqlen)
                video_idx += 1
            else:
                # read audio from video
                assert audio_seqlens is not None
                audio_seqlen = audio_seqlens[audio_idx]
                vision_seqlen = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)
                grid_t = video_grid_thw[video_idx][0]
                grid_h = video_grid_thw[video_idx][1]
                grid_w = video_grid_thw[video_idx][2]
                grid_hs = video_grid_thw[:, 1]
                grid_ws = video_grid_thw[:, 2]
                t_ntoken_per_chunk = int(tokens_per_second * seconds_per_chunk)
                t_index = (torch.arange(grid_t) * second_per_grid_ts[video_idx] * tokens_per_second).long()
                t_index_split_chunk = split_list_into_ranges(t_index, t_ntoken_per_chunk)
                place_num = (((audio_seqlen - 1) // 2 + 1 - 2) // 2 + 1) + 2
                pure_audio_len = place_num - 2
                added_audio_len = 0
                audio_llm_pos_ids_list: list[torch.Tensor] = []
                for t_chunk in t_index_split_chunk:
                    vision_ntoken_per_chunk = len(t_chunk) * grid_h * grid_w // (spatial_merge_size**2)
                    new_src_item.extend([video_token_id] * vision_ntoken_per_chunk)
                    vision_llm_pos_ids_list = get_llm_pos_ids_for_vision(
                        start_idx,
                        video_idx,
                        spatial_merge_size,
                        t_chunk,
                        grid_hs,
                        grid_ws,
                    ).split(1, dim=1)
                    llm_pos_ids_list.extend(vision_llm_pos_ids_list)
                    new_src_item.extend(min(t_ntoken_per_chunk, pure_audio_len - added_audio_len) * [audio_token_id])
                    audio_start_idx = (
                        start_idx if len(audio_llm_pos_ids_list) == 0 else audio_llm_pos_ids_list[-1][0].item() + 1
                    )
                    if min(t_ntoken_per_chunk, pure_audio_len - added_audio_len) > 0:
                        audio_llm_pos_ids_list = (
                            torch.arange(min(t_ntoken_per_chunk, pure_audio_len - added_audio_len)).expand(3, -1)
                            + audio_start_idx
                        ).split(1, dim=1)
                    else:
                        audio_llm_pos_ids_list = []
                    added_audio_len += min(t_ntoken_per_chunk, pure_audio_len - added_audio_len)
                    llm_pos_ids_list.extend(audio_llm_pos_ids_list)
                if added_audio_len < pure_audio_len:
                    new_src_item.extend((pure_audio_len - added_audio_len) * [audio_token_id])
                    audio_llm_pos_ids_list = (
                        torch.arange(pure_audio_len - added_audio_len).expand(3, -1) + llm_pos_ids_list[-1].max() + 1
                    ).split(1, dim=1)
                    llm_pos_ids_list.extend(audio_llm_pos_ids_list)
                audio_idx += 1
                video_idx += 1
            # move to the next token
            idx += len(new_src_item) - new_src_item_len

        llm_positions = torch.cat(llm_pos_ids_list, dim=1)
        mrope_position_delta = torch.cat(llm_pos_ids_list, dim=1).max() + 1 - len(src_item)

        return llm_positions, mrope_position_delta

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
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        # This is to satisfy the type checker for each overload
        if multimodal_embeddings is None or is_multimodal is None:
            return super().embed_input_ids(input_ids)

        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.get_language_model().embed_input_ids,
            is_multimodal=is_multimodal,
            handle_oov_mm_token=handle_oov_mm_token,
        )

        if len(multimodal_embeddings) == 0:
            return inputs_embeds

        # Check for audio-in-video: interleaved video and audio tokens
        # in the multimodal region.
        video_token_id = self.config.video_token_index
        audio_token_id = self.config.audio_token_index

        is_video = is_multimodal & (input_ids == video_token_id)
        is_audio = is_multimodal & (input_ids == audio_token_id)

        num_video = is_video.sum().item()
        num_audio = is_audio.sum().item()

        if check_interleaved_audio_video(is_video, is_audio, num_video, num_audio):
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
        return _merge_multimodal_embeddings(inputs_embeds, multimodal_embeddings, is_multimodal)

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
