import glob
import os
from collections.abc import Iterable
from functools import cached_property
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix

# from vllm.model_executor.models.qwen2_code2wav_dit import Qwen2Code2wav
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_omni.data_entry_keys import OmniPayload
from vllm_omni.model_executor.custom_process_mixin import CustomProcessMixin
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific
from vllm_omni.model_executor.models.minimind_o.config import (
    MiniMindOConfig,
    MiniMindOTalkerConfig,
    MiniMindOThinkerConfig,
)
from vllm_omni.model_executor.models.minimind_o.processor import (
    MiniMindOThinkerDummyInputsBuilder,
    MiniMindOThinkerMultiModalProcessor,
    MiniMindOThinkerProcessingInfo,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.utils import add_prefix_to_loaded_weights
from vllm_omni.platforms import current_omni_platform

# MiniMind-O token IDs (to be updated from actual config)
MIMI_CODEC_EOS_TOKEN_ID = 2050  # audio_stop_token
MIMI_CODEC_BOS_TOKEN_ID = 2049  # audio_pad_token
MIMI_CODEC_PAD_TOKEN_ID = 2049
MIMI_CODEC_SPK_TOKEN_ID = 2051  # audio_spk_token


logger = init_logger(__name__)


@MULTIMODAL_REGISTRY.register_processor(
    MiniMindOThinkerMultiModalProcessor,
    info=MiniMindOThinkerProcessingInfo,
    dummy_inputs=MiniMindOThinkerDummyInputsBuilder,
)
class MiniMindOForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP, CustomProcessMixin):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.has_preprocess = False
        self.have_multimodal_outputs = True
        config: MiniMindOConfig = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        # keep vllm_config for later submodule init
        self.vllm_config = vllm_config

        # Initialize thinker components
        thinker_config: MiniMindOThinkerConfig = config.thinker_config
        self.thinker_config = thinker_config
        self.multimodal_config = multimodal_config

        # Initialize talker components
        talker_config: MiniMindOTalkerConfig = config.talker_config
        self.talker_config = talker_config

        self.model_stage = vllm_config.model_config.model_stage
        if self.model_stage == "thinker":
            # Initialize thinker model (multimodal processing)
            self.thinker = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "thinker"),
                hf_config=thinker_config,
                architectures=["MiniMindOThinkerForConditionalGeneration"],
            )
            self.talker = None
            self.code2wav = None

        elif self.model_stage == "talker":
            multimodal_config.skip_mm_profiling = True
            self.has_preprocess = True
            self.has_postprocess = True
            self.set_custom_preprocess(self.talker_preprocess)
            self.set_custom_postprocess(self.talker_postprocess)
            self.thinker = None
            self.talker = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "talker"),
                hf_config=talker_config,
                architectures=["MiniMindOTalkerForConditionalGeneration"],
            )
            self.code2wav = None
            self.talker_mtp = self.talker
            c2w_token_end_id = getattr(getattr(config, "code2wav_config", None), "codebook_size", None)
            if c2w_token_end_id:
                self.talker.set_suppress_start_id(c2w_token_end_id + 1)
            self.requires_raw_input_tokens = True
            self.thinker_embedding = nn.Embedding(
                self.thinker_config.text_config.vocab_size,
                self.thinker_config.text_config.hidden_size,
            )
            self._init_special_tokens_embeddings()

        elif self.model_stage == "code2wav":
            multimodal_config.skip_mm_profiling = True
            self.thinker = None
            self.talker = None
            self.code2wav_config = getattr(config, "code2wav_config", None)
            self.code2wav = None
            if self.code2wav_config is not None:
                self.code2wav = init_vllm_registered_model(
                    vllm_config=vllm_config,
                    prefix=maybe_prefix(prefix, "code2wav"),
                    hf_config=self.code2wav_config,
                    architectures=["MiniMindOCode2Wav"],
                )
            self._code2wav_conds: dict[str, torch.Tensor] = {}
            self._code2wav_ref_mels: dict[str, torch.Tensor] = {}
            self.requires_raw_input_tokens = True
        else:
            raise ValueError("Invalid model stage")

        # Runner hooks (preprocess/MTP/sampler) live on the orchestrator module.
        self.model = self

        self.make_empty_intermediate_tensors = (
            (self.thinker.make_empty_intermediate_tensors) if self.model_stage == "thinker" else lambda: None
        )

    def _stage_module(self) -> nn.Module:
        if self.model_stage == "thinker":
            return self.thinker
        if self.model_stage == "talker":
            return self.talker
        if self.model_stage == "code2wav":
            return self.code2wav
        raise ValueError(f"Invalid model stage: {self.model_stage}")

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
        code2wav_device: str | torch.device | None = None,
    ) -> None:
        """Optionally move thinker/talker/code2wav to different devices.

        Example:
            model.move_submodules_to_devices(
                thinker_device='cuda:0',
                talker_device='cuda:1',
                code2wav_device='cpu',
            )
        """
        if thinker_device is not None and self.thinker is not None:
            self.thinker.to(thinker_device)
        if talker_device is not None and self.talker is not None:
            self.talker.to(talker_device)
        if code2wav_device is not None and self.code2wav is not None:
            self.code2wav.to(code2wav_device)

    @cached_property
    def sampler(self):
        stage = self._stage_module()
        if hasattr(stage, "sampler"):
            return stage.sampler
        return Sampler()

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal=None,
    ) -> torch.Tensor:
        if self.model_stage == "code2wav":
            return torch.zeros_like(input_ids).reshape(-1, 1).repeat(1, self.vllm_config.model_config.get_hidden_size())
        stage = self._stage_module()
        if self.model_stage == "talker":
            return stage.embed_input_ids(input_ids)
        return stage.embed_input_ids(
            input_ids=input_ids, multimodal_embeddings=multimodal_embeddings, is_multimodal=is_multimodal
        )

    def embed_multimodal(self, **kwargs):
        return self._stage_module().embed_multimodal(**kwargs)

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
        3) If audio requested (or codec provided), use code2wav to synthesize waveform.
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

            talker_positions = positions[0] if positions.ndim > 1 else positions

            with torch.inference_mode():
                talker_hidden = self.talker(
                    input_ids=input_ids,
                    positions=talker_positions,
                    inputs_embeds=inputs_embeds,
                )

            multimodal_outputs = None
            info_dicts = kwargs.get("model_intermediate_buffer")
            if info_dicts is None:
                info_dicts = kwargs.get("runtime_additional_information")
            if isinstance(info_dicts, dict):
                code_rows = []
                for info in info_dicts.values():
                    if not isinstance(info, dict):
                        continue
                    audio = info.get("codes", {}).get("audio")
                    if isinstance(audio, torch.Tensor) and audio.numel() > 0:
                        code_rows.append(audio.reshape(-1, 8) if audio.dim() == 1 else audio)
                if code_rows:
                    multimodal_outputs = {"codes": {"audio": torch.cat(code_rows, dim=0)}}

            return OmniOutput(
                text_hidden_states=talker_hidden,
                multimodal_outputs=multimodal_outputs,
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

            if code.numel() and code[-1] == MIMI_CODEC_EOS_TOKEN_ID:
                code = code[:-1]
            if code.numel() and code[0] == MIMI_CODEC_BOS_TOKEN_ID:
                code = code[1:]

            audio_tensor = self.generate_audio(code, voice_type) if code.numel() else torch.zeros(0, device=code.device)
            return OmniOutput(text_hidden_states=None, multimodal_outputs={"model_outputs": audio_tensor})

        return OmniOutput(
            text_hidden_states=torch.zeros(
                [inputs_embeds.shape[0], self.talker_config.talker_hidden_size],
                dtype=torch.bfloat16,
                device=self._module_device(self.model),
            ),
            multimodal_outputs=None,
        )

    def generate_audio(self, code, voice_type):
        code2wav_dev = self._module_device(self.code2wav)
        if isinstance(code, torch.Tensor):
            code_tensor = code.to(dtype=torch.long, device=code2wav_dev)
        else:
            code_tensor = torch.as_tensor(code, dtype=torch.long, device=code2wav_dev)
        if code_tensor.ndim == 2 and code_tensor.shape[0] == 1:
            code_tensor = code_tensor.squeeze(0)

        audio_tensor = self._codec_to_audio(code_tensor, voice_type)

        return audio_tensor

    def _load_talker_embedding(
        self,
    ) -> torch.nn.Embedding:
        return self.talker.embed_tokens.base

    def _init_special_tokens_embeddings(
        self,
    ):
        # talker embeddings
        self.talker_embedding = self._load_talker_embedding()

        # embed_text_bos_token
        talker_hf_config = self.talker_config
        if hasattr(talker_hf_config, "talker_config"):
            talker_hf_config = talker_hf_config.talker_config

        spk_id = int(getattr(talker_hf_config, "audio_spk_token", talker_hf_config.tts_text_start_token_id))
        self.tts_text_spk_token_ids = {"default": spk_id}
        self.default_tts_text_spk_type = "default"

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
        if not hasattr(self, "embed_text_spk_tokens") or voice_type not in self.embed_text_spk_tokens:
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

        payload: OmniPayload = info_dict

        # Ensure we have base embeddings when only ids are provided
        if input_embeds is None and input_ids is not None:
            input_embeds = self.talker.embed_input_ids(input_ids)

        span_len = input_ids.shape[0]
        if span_len > 1:
            # prefill
            return self.thinker_to_talker_process(input_ids, input_embeds, payload)
        else:
            # decode
            return self.thinker_to_talker_decode_one_step(input_ids, input_embeds, payload)

    def thinker_to_talker_process(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        payload: OmniPayload,
    ):
        embed = payload.get("embed", {})
        hs = payload.get("hidden_states", {})
        ids = payload.get("ids", {})

        update_dict = {}

        prompt_embeds = embed.get("prefill")  # Tensor [P,H]
        thinker_result = hs.get("output")  # Tensor [K,H]
        prompt_token_ids = ids.get("prompt")  # list[int]
        thinker_output_token_ids = ids.get("output")  # list[int]

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

        audio_ids = embed.get("audio_ids")
        if not isinstance(audio_ids, torch.Tensor):
            seq_len = max(len(prompt_token_ids) + 1, 1)
            audio_ids = torch.full(
                (8, seq_len),
                MIMI_CODEC_PAD_TOKEN_ID,
                dtype=torch.long,
                device=self._module_device(self.model),
            )

        req_input_ids, req_embeds = self._thinker_to_talker_prefill(
            speaker=self.voice_type,
            output_prompt_embeds=thinker_result.to(input_embeds.dtype).to(self._module_device(self.model)),
            output_token_ids=thinker_output_token_ids,
            thinker_prompt_embeds=prompt_embeds.to(input_embeds.dtype).to(self._module_device(self.model)),
            prompt_token_ids=prompt_token_ids,
            audio_ids=audio_ids,
        )

        if thinker_result.ndim == 2 and thinker_result.shape[0] > 0:
            update_dict.setdefault("embed", {})["thinker_reply"] = thinker_result[1:].detach().to("cpu").contiguous()

        update_dict.setdefault("meta", {})["audio_step"] = -1
        update_dict.setdefault("embed", {})["audio_ids"] = audio_ids
        return req_input_ids, req_embeds, update_dict

    def _thinker_to_talker_prefill(
        self,
        speaker: str,
        output_prompt_embeds,
        output_token_ids,
        thinker_prompt_embeds,
        prompt_token_ids,
        audio_ids: torch.Tensor,
    ):
        if thinker_prompt_embeds.numel() == 0 and output_prompt_embeds.numel() == 0:
            bridge = thinker_prompt_embeds.new_zeros(0, self.talker_config.hidden_size)
        else:
            parts = [thinker_prompt_embeds]
            if output_prompt_embeds.numel() > 0:
                parts.append(output_prompt_embeds[:1])
            bridge = torch.cat(parts, dim=0)

        bridge_b = bridge.unsqueeze(0)
        if audio_ids.dim() == 2:
            audio_b = audio_ids.unsqueeze(0)
        else:
            audio_b = audio_ids
        if audio_b.shape[-1] < bridge_b.shape[1]:
            pad_cols = bridge_b.shape[1] - audio_b.shape[-1]
            audio_b = torch.cat(
                [
                    audio_b,
                    audio_b.new_full(
                        (*audio_b.shape[:-1], pad_cols),
                        self.talker_config.audio_pad_token,
                    ),
                ],
                dim=-1,
            )
        elif audio_b.shape[-1] > bridge_b.shape[1]:
            audio_b = audio_b[..., : bridge_b.shape[1]]

        req_embeds = self.talker.build_fused_embeds(bridge_b, audio_b).squeeze(0)
        req_input_ids = torch.full(
            (bridge_b.shape[1],),
            self.talker_config.audio_pad_token,
            dtype=torch.int64,
            device=self._module_device(self.talker),
        )
        return req_input_ids, req_embeds

    def thinker_to_talker_decode_one_step(self, input_ids, input_embeds, payload: OmniPayload):
        embed = payload.get("embed", {})
        hs = payload.get("hidden_states", {})

        update_dict = {}
        step_vec = None
        q = embed.get("thinker_reply", None)
        if isinstance(q, torch.Tensor) and q.numel() > 0:
            step_vec = q[0:1]
            new_q = q[1:].detach().to("cpu").contiguous()
            update_dict.setdefault("embed", {})["thinker_reply"] = new_q
        else:
            # B) per-request provided decode vector (optional)
            dv = embed.get("decode")
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

        audio_ids = embed.get("audio_ids")
        meta = payload.get("meta", {})
        audio_step = int(meta.get("audio_step", -1)) + 1
        update_dict.setdefault("meta", {})["audio_step"] = audio_step

        last_hidden = hs.get("last")
        if not isinstance(last_hidden, torch.Tensor):
            last_hidden = torch.zeros(
                self.talker_config.talker_hidden_size,
                device=input_embeds.device,
                dtype=input_embeds.dtype,
            )

        if isinstance(step_vec, torch.Tensor) and step_vec.numel() > 0:
            one_id = input_ids[0:1]
            if not isinstance(audio_ids, torch.Tensor):
                audio_ids = torch.full(
                    (8, 1),
                    self.talker_config.audio_pad_token,
                    dtype=torch.long,
                    device=step_vec.device,
                )
            else:
                if audio_ids.dim() == 2:
                    audio_ids = audio_ids[:, -1:].contiguous()
                else:
                    audio_ids = audio_ids[..., -1:].contiguous()
            _, one_embed = self._thinker_to_talker_decode_one_step(
                output_prompt_embeds=step_vec.to(input_embeds.dtype).to(self._module_device(self.talker)),
                output_token_ids=one_id,
                audio_ids=audio_ids,
            )
            input_embeds = one_embed.unsqueeze(0) if one_embed.dim() == 1 else one_embed

        update_dict["mtp_inputs"] = (
            last_hidden.detach().to(device=input_embeds.device, dtype=input_embeds.dtype),
            step_vec.detach().to(device=input_embeds.device, dtype=input_embeds.dtype)
            if isinstance(step_vec, torch.Tensor) and step_vec.numel() > 0
            else torch.zeros(self.talker_config.hidden_size, device=input_embeds.device, dtype=input_embeds.dtype),
        )
        update_dict.setdefault("embed", {})["audio_ids"] = audio_ids
        return input_ids[0:1], input_embeds[0:1], update_dict

    def _thinker_to_talker_decode_one_step(
        self,
        output_prompt_embeds,
        output_token_ids,
        audio_ids: torch.Tensor,
    ):
        bridge_b = output_prompt_embeds.unsqueeze(0)
        audio_b = audio_ids.unsqueeze(0) if audio_ids.dim() == 2 else audio_ids
        fused = self.talker.build_fused_embeds(bridge_b, audio_b).squeeze(0)
        return output_token_ids, fused

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, **kwargs: object) -> torch.Tensor | None:
        # Handle OmniOutput type
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states

        # Use thinker model for logits computation
        return self._stage_module().compute_logits(hidden_states)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        return self._stage_module().sample(logits, sampling_metadata)

    def talker_postprocess(self, hidden_states: torch.Tensor, **info_dict: object):
        return self.talker.talker_postprocess(hidden_states, **info_dict)

    def preprocess_decode_batch(
        self,
        *,
        input_ids: torch.Tensor,
        req_infos: list[dict],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
        input_ids_flat = input_ids.reshape(-1)
        out_ids: list[torch.Tensor] = []
        out_embeds: list[torch.Tensor] = []
        last_hidden: list[torch.Tensor] = []
        text_steps: list[torch.Tensor] = []
        updates: list[dict] = []
        audio_steps: list[int] = []
        audio_ids_batch: list[torch.Tensor] = []

        for row, info_dict in enumerate(req_infos):
            start = int(row)
            one_id = input_ids_flat[start : start + 1]
            embed_slice = None
            _, one_embed, upd = self.talker_preprocess(one_id, embed_slice, **info_dict)
            mtp = upd.pop("mtp_inputs")
            last_h, text_s = mtp
            out_ids.append(one_id)
            out_embeds.append(one_embed.reshape(1, -1))
            last_hidden.append(last_h.reshape(1, -1))
            text_steps.append(text_s.reshape(1, -1))
            updates.append(upd)
            audio_steps.append(int(upd.get("meta", {}).get("audio_step", -1)))
            audio_ids_batch.append(upd.get("embed", {}).get("audio_ids"))

        self.talker._pending_audio_steps = torch.tensor(audio_steps, dtype=torch.long, device=out_embeds[0].device)
        self.talker._pending_audio_ids = audio_ids_batch
        return (
            torch.stack(out_ids, dim=0),
            torch.cat(out_embeds, dim=0),
            torch.cat(last_hidden, dim=0),
            torch.cat(text_steps, dim=0),
            updates,
        )

    def talker_mtp(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        last_talker_hidden: torch.Tensor,
        text_step: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pending_steps = getattr(self.talker, "_pending_audio_steps", None)
        pending_audio = getattr(self.talker, "_pending_audio_ids", None)
        bsz = int(input_ids.reshape(-1).shape[0])
        if pending_steps is not None and pending_steps.numel() >= bsz:
            kwargs = dict(kwargs)
            kwargs["audio_steps"] = pending_steps[:bsz]
        if pending_audio is not None:
            kwargs = dict(kwargs)
            kwargs["audio_ids_list"] = pending_audio[:bsz]
        return self.talker.talker_mtp(
            input_ids,
            input_embeds,
            last_talker_hidden,
            text_step,
            **kwargs,
        )

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

        # Generate audio using code2wav model
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

    def _init_code2wav_model(self, hf_model_folder):
        """Initialize speaker resources if provided; model is constructed in
        __init__."""
        if self.code2wav is None or self.code2wav_config is None:
            return
        device = self._module_device(self.code2wav)
        # optional speaker resources
        conds = getattr(self.code2wav_config, "conds", None)
        ref_mels = getattr(self.code2wav_config, "ref_mels", None)
        if isinstance(conds, dict) and isinstance(ref_mels, dict):
            self._code2wav_conds = {k: torch.as_tensor(v, device=device) for k, v in conds.items()}
            self._code2wav_ref_mels = {k: torch.as_tensor(v, device=device) for k, v in ref_mels.items()}
        # legacy: load from directory if provided
        model_path = hf_model_folder
        if isinstance(model_path, str) and os.path.isdir(model_path):
            spk_pt = os.path.join(model_path, "spk_dict.pt")
            if os.path.exists(spk_pt):
                data = torch.load(spk_pt, map_location=device)
                for key, value in data.items():
                    self._code2wav_conds[key] = value["cond"].to(device)
                    self._code2wav_ref_mels[key] = value["ref_mel"].to(device)
            else:
                # legacy npy inputs
                for f in sorted(glob.glob(os.path.join(model_path, "inputs", "*spk_emb.npy"))):
                    key = os.path.basename(f).split("_")[0].lower()
                    self._code2wav_conds[key] = torch.as_tensor(np.load(f), device=device)
                for f in sorted(glob.glob(os.path.join(model_path, "inputs", "*ref_mel.npy"))):
                    key = os.path.basename(f).split("_")[0].lower()
                    self._code2wav_ref_mels[key] = torch.as_tensor(np.load(f), device=device)

    def _codec_to_audio(self, codec_tokens: torch.Tensor, voice_type: str = "default") -> torch.Tensor | None:
        del voice_type  # placeholder decoder has no per-speaker conditioning yet
        if self.code2wav is None:
            return None
        code2wav_dev = self._module_device(self.code2wav)
        if isinstance(codec_tokens, torch.Tensor):
            codec = codec_tokens.to(dtype=torch.long, device=code2wav_dev)
            if codec.ndim == 1:
                codec = codec.unsqueeze(0)
        else:
            codec = torch.as_tensor(codec_tokens, dtype=torch.long, device=code2wav_dev).unsqueeze(0)
        with torch.inference_mode():
            waveform = self.code2wav(input_ids=codec)
        return waveform.squeeze().reshape(-1)

    @staticmethod
    def _partition_omni_weights(
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> tuple[list, list, list, list]:
        """Split flat MiniMind-O HF keys or staged thinker./talker./ prefixes."""
        thinker_weights = []
        talker_weights = []
        code2wav_weights = []
        other = []
        thinker_prefixes = ("model.", "lm_head.", "audio_proj.", "vision_proj.")
        for k, v in weights:
            if k.startswith("thinker."):
                thinker_weights.append((k[len("thinker.") :], v))
            elif k.startswith("talker."):
                talker_weights.append((k, v))
            elif k.startswith("code2wav."):
                code2wav_weights.append((k, v))
            elif k.startswith(thinker_prefixes):
                thinker_weights.append((k, v))
            else:
                other.append((k, v))
        return thinker_weights, talker_weights, code2wav_weights, other

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights for all components of the omni model."""
        loaded_weights = set()
        thinker_weights, talker_weights, code2wav_weights, other = self._partition_omni_weights(weights)
        if other:
            raise ValueError(f"Unknown weight keys: {[k for k, _ in other[:5]]}")

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
                thinker_embedding_weights = [
                    w
                    for n, w in thinker_weights
                    if n in ("model.embed_tokens.weight", "thinker.model.embed_tokens.weight")
                ]
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

        # Load code2wav weights (if any)
        if code2wav_weights and self.code2wav is not None:
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
            self._init_code2wav_model(hf_model_folder)
            c2w_loaded = self.code2wav.load_weights(code2wav_weights, os.path.join(hf_model_folder, "spk_dict.pt"))
            c2w_loaded = add_prefix_to_loaded_weights(c2w_loaded, "code2wav")
            loaded_weights.update(c2w_loaded)

        return loaded_weights


class MiniMindOMoeForConditionalGeneration(MiniMindOForConditionalGeneration):
    """Same staged omni stack; enable MoE via HF config ``use_moe: true``."""
