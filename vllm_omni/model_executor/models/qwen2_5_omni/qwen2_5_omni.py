import glob
import os
from collections.abc import Iterable
from functools import cached_property

import numpy as np
import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniConfig,
    Qwen2_5OmniTalkerConfig,
    Qwen2_5OmniThinkerConfig,
)
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsMRoPE, SupportsMultiModal, SupportsPP
from vllm.model_executor.models.qwen2_5_omni_thinker import (
    Qwen2_5OmniConditionalGenerationMixin,
    Qwen2_5OmniThinkerDummyInputsBuilder,
    Qwen2_5OmniThinkerMultiModalProcessor,
    Qwen2_5OmniThinkerProcessingInfo,
)
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix

# from vllm.model_executor.models.qwen2_code2wav_dit import Qwen2Code2wav
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.custom_process_mixin import CustomProcessMixin
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.utils import add_prefix_to_loaded_weights, split_list_into_ranges
from vllm_omni.model_executor.models.vision import get_llm_pos_ids_for_vision
from vllm_omni.platforms import current_omni_platform

TALKER_CODEC_EOS_TOKEN_ID = 8294
TALKER_CODEC_BOS_TOKEN_ID = 8293


logger = init_logger(__name__)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen2_5OmniThinkerMultiModalProcessor,
    info=Qwen2_5OmniThinkerProcessingInfo,
    dummy_inputs=Qwen2_5OmniThinkerDummyInputsBuilder,
)
class Qwen2_5OmniForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, SupportsMRoPE, Qwen2_5OmniConditionalGenerationMixin, CustomProcessMixin
):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.has_preprocess = False
        self.have_multimodal_outputs = True
        config: Qwen2_5OmniConfig = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        # keep vllm_config for later submodule init
        self.vllm_config = vllm_config

        # Initialize thinker components
        thinker_config: Qwen2_5OmniThinkerConfig = config.thinker_config
        self.thinker_config = thinker_config
        self.multimodal_config = multimodal_config

        # Initialize talker components
        talker_config: Qwen2_5OmniTalkerConfig = config.talker_config
        self.talker_config = talker_config

        self.model_stage = vllm_config.model_config.model_stage
        if self.model_stage == "thinker":
            # Initialize thinker model (multimodal processing)
            self.thinker = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "thinker"),
                hf_config=thinker_config,
                # Use registry architecture key
                architectures=["Qwen2_5OmniThinkerModel"],
            )
            self.model = self.thinker
            self.talker = None
            self.token2wav = None

        elif self.model_stage == "talker":
            # register the process function for the talker stage
            self.has_preprocess = True
            self.set_custom_preprocess(self.talker_preprocess)
            self.thinker = None
            # Initialize talker model wrapper (handles projection + LM)
            self.talker = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "talker"),
                hf_config=talker_config,
                # Use registry architecture key
                architectures=["Qwen2_5OmniTalkerModel"],
            )
            self.talker.init_multi_modal(thinker_config)
            self.model = self.talker
            self.token2wav = None
            # set suppress start id according to token2wav
            t2w_token_end_id = getattr(
                getattr(getattr(config, "token2wav_config", None), "dit_config", None), "num_embeds", None
            )
            if t2w_token_end_id:
                self.model.set_suppress_start_id(t2w_token_end_id + 1)
            self.requires_raw_input_tokens = True

        elif self.model_stage == "code2wav":
            self.thinker = None
            self.talker = None
            # Initialize token2wav (code->mel->wav) like thinker/talker
            self.token2wav_config = getattr(config, "token2wav_config", None)
            self.token2wav = None
            if self.token2wav_config is not None:
                self.token2wav = init_vllm_registered_model(
                    vllm_config=vllm_config,
                    prefix=maybe_prefix(prefix, "token2wav"),
                    hf_config=self.token2wav_config,
                    architectures=["Qwen2_5OmniToken2WavModel"],
                )
            # voice resources (loaded on demand)
            self._token2wav_conds: dict[str, torch.Tensor] = {}
            self._token2wav_ref_mels: dict[str, torch.Tensor] = {}
            self.model = self.token2wav
            self.requires_raw_input_tokens = True
        else:
            raise ValueError("Invalid model stage")

        # Set up intermediate tensors
        self.make_empty_intermediate_tensors = (
            (self.thinker.make_empty_intermediate_tensors) if self.model_stage == "thinker" else lambda: None
        )

    # -------------------- Device utilities --------------------
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
        thinker_device: str | torch.device | None = None,
        talker_device: str | torch.device | None = None,
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
        if thinker_device is not None and self.thinker is not None:
            self.thinker.to(thinker_device)
        if talker_device is not None and self.talker is not None:
            self.talker.to(talker_device)
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
        is_multimodal=None,
    ) -> torch.Tensor:
        if self.model_stage == "code2wav":
            return torch.zeros_like(input_ids).reshape(-1, 1).repeat(1, self.vllm_config.model_config.get_hidden_size())
        return self.model.embed_input_ids(
            input_ids=input_ids, multimodal_embeddings=multimodal_embeddings, is_multimodal=is_multimodal
        )

    def embed_multimodal(self, **kwargs):
        # Delegate to thinker model for multimodal processing
        return self.model.embed_multimodal(**kwargs)

    def last_index_of(self, list, value):
        return len(list) - 1 - list[::-1].index(value)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        generate_audio: bool = True,
        voice_type: str = "Chelsie",
        codec: torch.Tensor | None = None,
        sampling_metadata: SamplingMetadata | None = None,
        logits_index: int | None = None,
        sampler=None,
        additional_information: dict[str, object] | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors | OmniOutput:
        """
        Workflow:
        1) Thinker: multimodal understanding → text hidden states.
        2) If audio requested and codec not provided, use talker to derive codec.
        3) If audio requested (or codec provided), use token2wav to synthesize waveform.
        4) Return text hidden states (and audio when applicable).
        """
        if self.model_stage == "thinker":
            # Normalize to batched inputs if caller provides 1D/2D unbatched tensors
            # TODO: Remove this hack when NPU supports batched inputs properly
            added_batch_dim = False
            if input_ids is not None and input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
                added_batch_dim = True
            if positions is not None and positions.ndim == 1:
                positions = positions.unsqueeze(0)
                added_batch_dim = True
            if inputs_embeds is not None and inputs_embeds.ndim == 2:
                inputs_embeds = inputs_embeds.unsqueeze(0)
                added_batch_dim = True
            thinker_dev = self._module_device(self.thinker)

            # if input_ids is None, set it to a zero tensor, in the length of the
            # same as the embedding seq length
            if input_ids is None:
                input_ids = torch.zeros(inputs_embeds.shape[1], dtype=torch.long, device=thinker_dev).unsqueeze(
                    0
                )  # (1, 0)
                added_batch_dim = True

            # 1) Thinker (ensure inputs on thinker's device)
            if input_ids is not None and input_ids.device != thinker_dev:
                input_ids = input_ids.to(thinker_dev)
            if positions is not None and positions.device != thinker_dev:
                positions = positions.to(thinker_dev)
            if inputs_embeds is not None and inputs_embeds.device != thinker_dev:
                inputs_embeds = inputs_embeds.to(thinker_dev)

            if current_omni_platform.is_npu():
                # TODO: remove this hack when NPU supports batched inputs properly
                thinker_input_ids = input_ids[0] if input_ids is not None and added_batch_dim else input_ids
                # For MRoPE, positions shape is [3, num_tokens] (T/H/W), don't slice it
                if positions.ndim == 2 and positions.shape[0] == 3:
                    thinker_positions = positions  # MRoPE positions, keep as is
                else:
                    thinker_positions = positions[0] if positions.ndim > 1 else positions
                thinker_inputs_embeds = (
                    inputs_embeds[0] if inputs_embeds is not None and added_batch_dim else inputs_embeds
                )
            else:
                # Squeeze back if we added batch dim earlier
                thinker_input_ids = input_ids[0] if input_ids is not None and added_batch_dim else input_ids
                # For MRoPE, positions shape is [3, num_tokens] (T/H/W), don't slice it
                if positions.ndim == 2 and positions.shape[0] == 3:
                    thinker_positions = positions  # MRoPE positions, keep as is
                elif added_batch_dim:
                    thinker_positions = positions[0]
                else:
                    thinker_positions = positions
                thinker_inputs_embeds = (
                    inputs_embeds[0] if inputs_embeds is not None and added_batch_dim else inputs_embeds
                )

            # Run thinker
            thinker_output = self.thinker(
                input_ids=thinker_input_ids,
                positions=thinker_positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=thinker_inputs_embeds,
                **kwargs,
            )

            if isinstance(thinker_output, tuple):
                embeds, text_hidden_states = thinker_output
            else:
                text_hidden_states = thinker_output

            # Text-only path
            return OmniOutput(
                text_hidden_states=(text_hidden_states.reshape(-1, text_hidden_states.shape[-1])),
                multimodal_outputs=None,
            )

        # 2) Talker (if codec not provided)
        if self.model_stage == "talker":
            # mock data for profile
            if input_ids is None:
                input_ids = torch.zeros(inputs_embeds.shape[0], dtype=torch.long, device=inputs_embeds.device)
                self.thinker_reply_part = torch.zeros_like(inputs_embeds)

            # TODO(Peiqi): temporal hack here to support voice_type.
            if not hasattr(self, "voice_type"):
                self.voice_type = voice_type

            # For MRoPE, positions shape is [3, num_tokens] (T/H/W), don't slice it
            if positions.ndim == 2 and positions.shape[0] == 3:
                talker_positions = positions  # MRoPE positions, keep as is
            else:
                talker_positions = positions[0]

            with torch.inference_mode():
                talker_hidden = self.talker(
                    input_ids=input_ids,
                    positions=talker_positions,
                    inputs_embeds=inputs_embeds,
                )

            if sampling_metadata is not None:
                # the padding token id is set to text model's pad token id,
                # which do not match with the talker model's word embedding size
                sampling_metadata.prompt_token_ids[sampling_metadata.prompt_token_ids == 152064] = 8448

            return OmniOutput(
                text_hidden_states=talker_hidden,
                multimodal_outputs=None,
            )

        if self.model_stage == "code2wav":
            code = (
                input_ids
                if input_ids is not None
                else torch.zeros(
                    inputs_embeds.shape[0],
                    dtype=torch.long,
                    device=inputs_embeds.device,
                )
            )

            code = code[:-1] if code[-1] == TALKER_CODEC_EOS_TOKEN_ID else code
            code = code[1:] if code[0] == TALKER_CODEC_BOS_TOKEN_ID else code

            audio_tensor = self.generate_audio(code, voice_type)
            return OmniOutput(text_hidden_states=None, multimodal_outputs={"model_outputs": audio_tensor})

        return OmniOutput(
            text_hidden_states=torch.cat(
                [
                    torch.zeros(
                        [inputs_embeds.shape[0], self.talker.config.hidden_size],
                        dtype=torch.bfloat16,
                    ).to(self._module_device(self.model)),
                    self.talker.thinker_to_talker_proj(
                        self.talker.embed_input_ids(
                            torch.tensor([TALKER_CODEC_BOS_TOKEN_ID, TALKER_CODEC_EOS_TOKEN_ID])
                            .to(torch.bfloat16)
                            .to(self._module_device(self.model))
                        )
                    )[0],
                ],
                dim=0,
            ),
            multimodal_outputs=None,
        )

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec] | None = None,
        *,
        hf_config: PretrainedConfig,
        image_grid_thw: list[list[int]] | torch.Tensor,
        video_grid_thw: list[list[int]] | torch.Tensor,
        second_per_grid_ts: list[float] | None = None,
        context_len: int = 0,
        seq_len: int | None = None,
        audio_feature_lengths: torch.Tensor | None = None,
        use_audio_in_video: bool = False,
    ) -> tuple[torch.Tensor, int]:
        """Get mrope input positions and delta value (Qwen2.5-Omni version).

        Differences from MRotaryEmbedding:
            1. Add audio support (and related `audio_feature_lengths`).
            2. Add `use_audio_in_video` option to read audio from video inputs.
                In this case, audio and vision position ids will be split into
                chunks and interleaved.

        Example:

            (V_i are vision position ids, A_i are audio position ids)

            |V_1 ...    V_n|A_1 ...   A_n|V_n+1 ... V_2n|A_n+1 ... A_2n|...
            |vision chunk 1|audio chunk 1|vision chunk 2|audio chunk 2 |...
        """

        # TODO(fyabc): refactor and share more code with
        #  _vl_get_input_positions_tensor.

        thinker_config = hf_config.thinker_config
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

        if isinstance(image_grid_thw, list):
            image_grid_thw = torch.tensor(image_grid_thw)
        if isinstance(video_grid_thw, list):
            video_grid_thw = torch.tensor(video_grid_thw)

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
        llm_positions = llm_positions[:, context_len:seq_len]

        return llm_positions, mrope_position_delta

    def generate_audio(self, code, voice_type):
        token2wav_dev = self._module_device(self.token2wav)
        if isinstance(code, torch.Tensor):
            code_tensor = code.to(dtype=torch.long, device=token2wav_dev)
        else:
            code_tensor = torch.as_tensor(code, dtype=torch.long, device=token2wav_dev)
        if code_tensor.ndim == 2 and code_tensor.shape[0] == 1:
            code_tensor = code_tensor.squeeze(0)

        audio_tensor = self._codec_to_audio(code_tensor, voice_type)

        return audio_tensor

    def _load_talker_embedding(
        self,
    ) -> torch.nn.Embedding:
        return self.talker.language_model.model.embed_tokens

    def _init_special_tokens_embeddings(
        self,
    ):
        # talker embeddings
        self.talker_embedding = self._load_talker_embedding()

        # embed_text_bos_token
        self.tts_text_spk_token_ids = {
            # M02: Male voice with standard Mandarin and a slight northern accent
            "m02": 151870,
            "Ethan": 151870,
            # F030: Your anime-styled virtual girlfriend
            "f030": 151872,
            "Chelsie": 151872,
        }
        self.default_tts_text_spk_type = list(self.tts_text_spk_token_ids.keys())[0]
        self.tts_text_spk_token_ids["prefix_caching"] = 151870

        talker_hf_config = self.talker_config
        if hasattr(talker_hf_config, "talker_config"):
            talker_hf_config = talker_hf_config.talker_config

        self.embed_text_bos_token = self.thinker_embedding(
            torch.tensor(
                [talker_hf_config.tts_text_start_token_id],
                dtype=torch.long,
                device=self._module_device(self.talker),
            )
        )
        self.embed_text_spk_tokens = {
            key: self.thinker_embedding(
                torch.tensor(
                    [value],
                    dtype=torch.long,
                    device=self._module_device(self.talker),
                )
            )
            for key, value in self.tts_text_spk_token_ids.items()
        }
        self.embed_text_eos_token = self.thinker_embedding(
            torch.tensor(
                [talker_hf_config.tts_text_end_token_id],
                dtype=torch.long,
                device=self._module_device(self.talker),
            )
        )
        self.embed_text_pad_token = self.thinker_embedding(
            torch.tensor(
                [talker_hf_config.tts_text_pad_token_id],
                dtype=torch.long,
                device=self._module_device(self.talker),
            )
        )
        self.embed_codec_bos_token = self.talker_embedding(
            torch.tensor(
                [talker_hf_config.tts_codec_start_token_id],
                dtype=torch.long,
                device=self._module_device(self.talker),
            )
        )
        self.embed_codec_eos_token = self.talker_embedding(
            torch.tensor(
                [talker_hf_config.tts_codec_end_token_id],
                dtype=torch.long,
                device=self._module_device(self.talker),
            )
        )
        self.embed_codec_pad_token = self.talker_embedding(
            torch.tensor(
                [talker_hf_config.tts_codec_pad_token_id],
                dtype=torch.long,
                device=self._module_device(self.talker),
            )
        )
        return set(["thinker_embedding.weight", "talker_embedding.weight"])

    def _get_embed_text_spk_token(self, voice_type: str):
        if voice_type not in self.embed_text_spk_tokens:
            return self.embed_text_bos_token
        return self.embed_text_spk_tokens[voice_type]

    def _get_text_spk_token_id(self, voice_type: str):
        talker_hf_config = self.talker_config
        if hasattr(talker_hf_config, "talker_config"):
            talker_hf_config = talker_hf_config.talker_config

        if voice_type not in self.tts_text_spk_token_ids:
            return talker_hf_config.tts_text_start_token_id
        return self.tts_text_spk_token_ids[voice_type]

    def talker_preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        **info_dict: object,
    ):
        # Mixed-mode support: In a single step, both Prefill*n and Decode*n are supported.
        # Rules:
        # - Prefill segments are wrapped with special tokens: [BOS][PAD...][EOS]
        # - Decode segments consist of a single non-special token.
        # - If additional_information is provided (can be a list split by request or a
        #   concatenated tensor plus a list of shapes), then for each request, reconstruct
        #   the thinker→talker input embeddings for the Prefill segments;
        # - For Decode segments, if per-request auxiliary decode embeddings are provided (optional),
        #   add them; otherwise, keep the original embedding.

        # Ensure we have base embeddings when only ids are provided
        if input_embeds is None and input_ids is not None:
            input_embeds = self.talker.embed_input_ids(input_ids)

        span_len = input_ids.shape[0]
        if span_len > 1:
            # prefill
            return self.thinker_to_talker_process(input_ids, input_embeds, **info_dict)
        else:
            # decode
            return self.thinker_to_talker_decode_one_step(input_ids, input_embeds, **info_dict)

    def thinker_to_talker_process(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        **info_dict: object,
    ):
        update_dict = {}

        prompt_embeds = info_dict.get("prompt_embeds")  # Tensor [P,H]
        thinker_result = info_dict.get("thinker_result")  # Tensor [K,H]
        prompt_token_ids = info_dict.get("prompt_token_ids")  # list[int]
        thinker_output_token_ids = info_dict.get("thinker_output_token_ids")  # list[int]

        if not isinstance(prompt_embeds, torch.Tensor):
            prompt_embeds = torch.zeros(
                0, self.talker.config.hidden_size, dtype=input_embeds.dtype, device=self._module_device(self.model)
            )
        if not isinstance(thinker_result, torch.Tensor):
            thinker_result = torch.zeros(
                0, self.talker.config.hidden_size, dtype=input_embeds.dtype, device=self._module_device(self.model)
            )
        if not isinstance(prompt_token_ids, (list, torch.Tensor)):
            prompt_token_ids = []
        if not isinstance(thinker_output_token_ids, (list, torch.Tensor)):
            thinker_output_token_ids = []

        # TODO(Peiqi): add voice_type support
        req_input_ids, req_embeds = self._thinker_to_talker_prefill(
            voice_type=self.voice_type,
            output_prompt_embeds=thinker_result.to(input_embeds.dtype).to(self._module_device(self.model)),
            output_token_ids=thinker_output_token_ids,
            thinker_prompt_embeds=prompt_embeds.to(input_embeds.dtype).to(self._module_device(self.model)),
            prompt_token_ids=prompt_token_ids,
        )

        if thinker_result.ndim == 2 and thinker_result.shape[0] > 0:
            update_dict["thinker_reply_part"] = thinker_result[1:].detach().to("cpu").contiguous()

        return req_input_ids, req_embeds, update_dict

    def _thinker_to_talker_prefill(
        self,
        voice_type: str,
        output_prompt_embeds,
        output_token_ids,
        thinker_prompt_embeds,
        prompt_token_ids,
    ):
        talker_hf_config = self.talker_config
        if hasattr(talker_hf_config, "talker_config"):
            talker_hf_config = talker_hf_config.talker_config

        # if len(output.outputs[0].token_ids) == 2:
        # issue request
        prompt_embeds = torch.cat(
            [
                thinker_prompt_embeds,
                self._get_embed_text_spk_token(voice_type) + self.embed_codec_pad_token,
                output_prompt_embeds[:1] + self.embed_codec_bos_token,
            ],
            dim=0,
        )

        prompt_token_ids_processed = prompt_token_ids + [
            talker_hf_config.tts_codec_pad_token_id,
            output_token_ids[0],
        ]
        input_tokens_len = len(prompt_token_ids_processed)
        # the code below is from model runner in Qwen, may need to further discuss later
        if input_tokens_len > 2:
            prompt_token_ids_processed = [self.talker_config.tts_codec_mask_token_id] * (input_tokens_len - 2) + [
                self.talker_config.tts_codec_pad_token_id,
                self.talker_config.tts_codec_start_token_id,
            ]
        else:
            prompt_token_ids_processed = [
                self.talker_config.tts_codec_pad_token_id,
                self.talker_config.tts_codec_start_token_id,
            ][-input_tokens_len:]
        if isinstance(prompt_token_ids_processed, list):
            prompt_token_ids_processed = (
                torch.Tensor(prompt_token_ids_processed).to(torch.int64).to(self._module_device(self.talker))
            )
        return prompt_token_ids_processed, prompt_embeds

    def thinker_to_talker_decode_one_step(self, input_ids, input_embeds, **info_dict):
        update_dict = {}
        # choose step vector in priority order
        step_vec = None
        q = info_dict.get("thinker_reply_part", None)
        if isinstance(q, torch.Tensor) and q.numel() > 0:
            step_vec = q[0:1]
            new_q = q[1:].detach().to("cpu").contiguous()
            update_dict["thinker_reply_part"] = new_q
        else:
            # B) per-request provided decode vector (optional)
            dv = info_dict.get("decode_output_prompt_embeds") if isinstance(info_dict, dict) else None
            if isinstance(dv, torch.Tensor) and dv.numel() > 0:
                step_vec = dv[0:1] if dv.ndim == 2 else dv.view(1, -1)
            elif (
                hasattr(self, "thinker_reply_part")
                and isinstance(self.thinker_reply_part, torch.Tensor)
                and self.thinker_reply_part.numel() > 0
            ):
                # C) fallback shared pool
                step_vec = self.thinker_reply_part[0:1]
                self.thinker_reply_part = self.thinker_reply_part[1:]

        if isinstance(step_vec, torch.Tensor) and step_vec.numel() > 0:
            one_id = input_ids[0:1]
            _, one_embed = self._thinker_to_talker_decode_one_step(
                output_prompt_embeds=step_vec.to(input_embeds.dtype).to(self._module_device(self.model)),
                output_token_ids=one_id,
            )
            input_embeds[0] = one_embed[0]
        return input_ids[0:1], input_embeds[0:1], update_dict

    def _thinker_to_talker_decode_one_step(
        self,
        output_prompt_embeds,
        output_token_ids,
    ):
        processed_output_token_embeds = output_prompt_embeds + self.talker.embed_input_ids(
            output_token_ids
        )  # for decode
        return output_token_ids, processed_output_token_embeds

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, **kwargs: object) -> torch.Tensor | None:
        # Handle OmniOutput type
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states

        # Use thinker model for logits computation
        return self.model.compute_logits(hidden_states)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        # Use thinker model for sampling
        return self.model.sample(logits, sampling_metadata)

    def generate_speech(self, text_tokens: torch.Tensor, voice_type: str = "default") -> torch.Tensor:
        """
        Generate speech from text tokens using the talker and token2wav models.
        This method is kept for backward compatibility and direct speech generation.

        Args:
            text_tokens: Text tokens from thinker model
            voice_type: Voice type for speech generation

        Returns:
            Audio tensor
        """
        # Generate codec tokens using talker model
        talker_output = self.talker(input_ids=None, positions=None, inputs_embeds=text_tokens)

        # Convert talker output to codec tokens
        codec_tokens = self._convert_to_codec_tokens(talker_output)

        # Generate audio using token2wav model
        return self._codec_to_audio(codec_tokens, voice_type=voice_type)

    def _convert_to_codec_tokens(
        self, talker_output: torch.Tensor, sampling_metadata: SamplingMetadata
    ) -> torch.Tensor:
        """
        Reference (HF): use the talker's codec head to obtain logits, suppress BOS,
        then greedily select the next codec token for the current step.
        """
        with torch.inference_mode():
            logits = self.talker.compute_logits(talker_output, None)
            if logits is None:
                return torch.zeros(
                    (talker_output.size(0), 0),
                    dtype=torch.long,
                    device=talker_output.device,
                )

            # Suppress only codec_bos, consistent with HF generate's
            # suppress_tokens behavior
            bos_id = None
            if hasattr(self, "talker_config") and hasattr(self.talker_config, "tts_codec_start_token_id"):
                bos_id = int(getattr(self.talker_config, "tts_codec_start_token_id"))
            if bos_id is not None:
                logits[..., bos_id] = -1e9

            # Take the distribution at the last step and select greedily
            next_id = self.talker.sample(logits, sampling_metadata).sampled_token_ids
            return next_id.to(dtype=torch.long)

    def _init_token2wav_model(self, hf_model_folder):
        """Initialize speaker resources if provided; model is constructed in
        __init__."""
        if self.token2wav is None or self.token2wav_config is None:
            return
        device = self._module_device(self.token2wav)
        # optional speaker resources
        conds = getattr(self.token2wav_config, "conds", None)
        ref_mels = getattr(self.token2wav_config, "ref_mels", None)
        if isinstance(conds, dict) and isinstance(ref_mels, dict):
            self._token2wav_conds = {k: torch.as_tensor(v, device=device) for k, v in conds.items()}
            self._token2wav_ref_mels = {k: torch.as_tensor(v, device=device) for k, v in ref_mels.items()}
        # legacy: load from directory if provided
        model_path = hf_model_folder
        if isinstance(model_path, str) and os.path.isdir(model_path):
            spk_pt = os.path.join(model_path, "spk_dict.pt")
            if os.path.exists(spk_pt):
                data = torch.load(spk_pt, map_location=device)
                for key, value in data.items():
                    self._token2wav_conds[key] = value["cond"].to(device)
                    self._token2wav_ref_mels[key] = value["ref_mel"].to(device)
            else:
                # legacy npy inputs
                for f in sorted(glob.glob(os.path.join(model_path, "inputs", "*spk_emb.npy"))):
                    key = os.path.basename(f).split("_")[0].lower()
                    self._token2wav_conds[key] = torch.as_tensor(np.load(f), device=device)
                for f in sorted(glob.glob(os.path.join(model_path, "inputs", "*ref_mel.npy"))):
                    key = os.path.basename(f).split("_")[0].lower()
                    self._token2wav_ref_mels[key] = torch.as_tensor(np.load(f), device=device)

    def _codec_to_audio(self, codec_tokens: torch.Tensor, voice_type: str = "default") -> torch.Tensor | None:
        if self.token2wav is None:
            self._init_token2wav_model()
        if self.token2wav is None:
            return None
        # Normalize voice type
        voice = voice_type or "default"
        # Resolve cond / ref_mel if provided
        cond = None
        ref_mel = None
        if voice in self._token2wav_conds and voice in self._token2wav_ref_mels:
            cond = self._token2wav_conds[voice]
            ref_mel = self._token2wav_ref_mels[voice]
        # Fallback: create dummy cond/ref_mel if not provided
        token2wav_dev = self._module_device(self.token2wav)
        if cond is None:
            cond = torch.zeros(
                (1, self.token2wav_config.dit_config.enc_emb_dim),
                device=token2wav_dev,
                dtype=torch.float32,
            )
        if ref_mel is None:
            ref_mel = torch.zeros(
                (1, 300, self.token2wav_config.dit_config.mel_dim),
                device=token2wav_dev,
                dtype=torch.float32,
            )

        # Ensure codec is (1, T) long tensor on correct device
        if isinstance(codec_tokens, torch.Tensor):
            codec = codec_tokens.to(dtype=torch.long, device=token2wav_dev)
            if codec.ndim == 1:
                codec = codec.unsqueeze(0)
        else:
            codec = torch.as_tensor(codec_tokens, dtype=torch.long, device=token2wav_dev).unsqueeze(0)

        # Streaming with chunked process and boundary alignment
        # (rely on token2wav.process_chunk)
        factor = getattr(self.token2wav.token2wav.factor, "factor", 2)
        chunk_size = 48
        mel_dim = getattr(
            self.token2wav.token2wav.code2wav_dit_model,
            "mel_dim",
            self.token2wav_config.dit_config.mel_dim,
        )
        total_mel = int(codec.shape[1] * factor)
        steps = 10

        # Prepare initial noise for the whole sequence
        y_all = torch.randn((1, total_mel, mel_dim), dtype=ref_mel.dtype, device=token2wav_dev)

        logger.info(
            "Currently, we do not use the chunked process, we only use the "
            "token2wav.process_chunk for the whole sequence. "
            "The stream mode will be implemented in the future."
        )

        chunk_ends = []
        for i in range(codec.shape[1]):
            chunk_code_length = i * 2 - 24
            finished = i == (codec.shape[1] - 1)
            if (chunk_code_length > 0 and chunk_code_length % chunk_size == 0) or finished:
                chunk_ends.append(i)

        # Number of chunks in mel domain
        prev_generated = None
        wav_chunks: list = []

        with torch.inference_mode():
            for n, i in enumerate([0]):
                finished = i == codec.shape[1] - 1
                _, audio_chunk = self.token2wav.process_chunk(
                    conditioning=cond,
                    reference_mel=ref_mel,
                    codec_all=codec,
                    y_all=y_all,
                    i=n,
                    steps=steps,
                    prev_generated=prev_generated if prev_generated is not None else [],
                    finished=True,
                )
                prev_generated = audio_chunk
                wav_chunks.append(audio_chunk.detach().cpu().numpy())

        if len(wav_chunks) == 0:
            return torch.zeros(0, device=token2wav_dev)

        waveform = np.concatenate(wav_chunks)
        return torch.as_tensor(waveform, device=token2wav_dev)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights for all components of the omni model."""
        loaded_weights = set()
        thinker_weights = []
        talker_weights = []
        token2wav_weights = []
        for k, v in weights:
            if k.startswith("thinker."):
                thinker_weights.append((k, v))
            elif k.startswith("talker."):
                talker_weights.append((k, v))
            elif k.startswith("token2wav."):
                token2wav_weights.append((k, v))
            else:
                raise ValueError(f"Unknown weight prefix: {k}")

        # Load thinker weights
        if self.thinker:
            if thinker_weights:
                thinker_loaded = self.thinker.load_weights(thinker_weights)
            else:
                thinker_loaded = set([k for k, v in thinker_weights])
            thinker_loaded = add_prefix_to_loaded_weights(thinker_loaded, "thinker")
            loaded_weights.update(thinker_loaded)

        # Load talker weights
        if talker_weights and self.talker is not None:
            # Map talker weights to appropriate components
            if self.thinker is None:
                thinker_embedding_weights = [w for n, w in thinker_weights if n == "thinker.model.embed_tokens.weight"]
                if thinker_embedding_weights:
                    self.thinker_embedding = nn.Embedding(
                        thinker_embedding_weights[0].shape[0],
                        thinker_embedding_weights[0].shape[1],
                    )
                    self.thinker_embedding.weight = nn.Parameter(
                        thinker_embedding_weights[0].to(self._module_device(self.talker))
                    )
            talker_loaded = self.talker.load_weights(talker_weights)
            talker_loaded = add_prefix_to_loaded_weights(talker_loaded, "talker")
            loaded_weights.update(talker_loaded)
            loaded_weights.update(self._init_special_tokens_embeddings())

        # Load token2wav weights (if any)
        if token2wav_weights and self.token2wav is not None:
            # download weights from huggingface for spk_dict.pt
            model_path = self.vllm_config.model_config.model
            download_dir = self.vllm_config.load_config.download_dir
            if os.path.exists(model_path):
                hf_model_folder = model_path
            else:
                hf_model_folder = download_weights_from_hf_specific(
                    model_path,
                    download_dir,
                    allow_patterns=["*.pt"],
                )
            self._init_token2wav_model(hf_model_folder)
            t2w_loaded = self.token2wav.load_weights(token2wav_weights, os.path.join(hf_model_folder, "spk_dict.pt"))
            t2w_loaded = add_prefix_to_loaded_weights(t2w_loaded, "token2wav")
            loaded_weights.update(t2w_loaded)

        return loaded_weights
