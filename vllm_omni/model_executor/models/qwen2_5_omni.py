import glob
import os
from functools import cached_property
from typing import Dict, Iterable, NamedTuple, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniConfig,
    Qwen2_5OmniTalkerConfig,
    Qwen2_5OmniThinkerConfig,
)

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
from vllm.model_executor.models.qwen2_5_omni_thinker import (
    Qwen2_5OmniConditionalGenerationMixin,
    Qwen2_5OmniThinkerDummyInputsBuilder,
    Qwen2_5OmniThinkerMultiModalProcessor,
    Qwen2_5OmniThinkerProcessingInfo,
)
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix

# from vllm.model_executor.models.qwen2_code2wav_dit import Qwen2Code2wav
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler
from vllm_omni.model_executor.model_loader.weight_utils import (
    download_weights_from_hf_specific,
)
from vllm_omni.model_executor.models.utils import add_prefix_to_loaded_weights

TALKER_CODEC_EOS_TOKEN_ID = 8294
TALKER_CODEC_BOS_TOKEN_ID = 8293


class OmniOutput(NamedTuple):
    """Output from the merged Omni model containing both text and audio."""

    text_hidden_states: torch.Tensor
    multimodal_outputs: dict = {}
    intermediate_tensors: Optional[IntermediateTensors] = None


logger = init_logger(__name__)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen2_5OmniThinkerMultiModalProcessor,
    info=Qwen2_5OmniThinkerProcessingInfo,
    dummy_inputs=Qwen2_5OmniThinkerDummyInputsBuilder,
)
class Qwen2_5OmniForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, Qwen2_5OmniConditionalGenerationMixin
):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
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
            self._token2wav_conds: Dict[str, torch.Tensor] = {}
            self._token2wav_ref_mels: Dict[str, torch.Tensor] = {}
            self.model = self.token2wav
        else:
            raise ValueError("Invalid model stage")

        # Set up intermediate tensors
        self.make_empty_intermediate_tensors = (
            (self.thinker.make_empty_intermediate_tensors)
            if self.model_stage == "thinker"
            else lambda: None
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
        thinker_device: Optional[Union[str, torch.device]] = None,
        talker_device: Optional[Union[str, torch.device]] = None,
        token2wav_device: Optional[Union[str, torch.device]] = None,
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

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
    ) -> torch.Tensor:
        if self.model_stage == "code2wav":
            return (
                torch.zeros_like(input_ids)
                .reshape(-1, 1)
                .repeat(1, self.vllm_config.model_config.get_hidden_size())
            )
        return self.model.get_input_embeddings(input_ids, multimodal_embeddings)

    def get_multimodal_embeddings(self, **kwargs):
        # Delegate to thinker model for multimodal processing
        return self.model.get_multimodal_embeddings(**kwargs)

    def last_index_of(self, list, value):
        return len(list) - 1 - list[::-1].index(value)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        generate_audio: bool = True,
        voice_type: str = "Chelsie",
        codec: Optional[torch.Tensor] = None,
        sampling_metadata: Optional[SamplingMetadata] = None,
        logits_index: Optional[int] = None,
        sampler=None,
        additional_information: Optional[dict[str, object]] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors, OmniOutput]:
        """
        Workflow:
        1) Thinker: multimodal understanding → text hidden states.
        2) If audio requested and codec not provided, use talker to derive codec.
        3) If audio requested (or codec provided), use token2wav to synthesize waveform.
        4) Return text hidden states (and audio when applicable).
        """
        if self.model_stage == "thinker":
            # Normalize to batched inputs if caller provides 1D/2D unbatched tensors
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
                input_ids = torch.zeros(
                    inputs_embeds.shape[1], dtype=torch.long, device=thinker_dev
                ).unsqueeze(
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
            # Run thinker
            thinker_output = self.thinker(
                input_ids=input_ids,
                positions=positions[0],
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )

            if isinstance(thinker_output, tuple):
                embeds, text_hidden_states = thinker_output
            else:
                text_hidden_states = thinker_output

            # Text-only path
            return OmniOutput(
                text_hidden_states=(
                    text_hidden_states.squeeze(0)
                    if added_batch_dim
                    else text_hidden_states
                ),
                multimodal_outputs=None,
            )

        # 2) Talker (if codec not provided)
        if self.model_stage == "talker":
            # Mixed-mode support: In a single step, both Prefill*n and Decode*n are supported.
            # Rules:
            # - Prefill segments are wrapped with special tokens: [BOS][PAD...][EOS]
            # - Decode segments consist of a single non-special token.
            # - If additional_information is provided (can be a list split by request or a concatenated tensor plus a list of shapes),
            #   then for each request, reconstruct the thinker→talker input embeddings for the Prefill segments;
            # - For Decode segments, if per-request auxiliary decode embeddings are provided (optional), add them; otherwise, keep the original embedding.

            if input_ids is None and additional_information is None:
                input_ids = torch.zeros(
                    inputs_embeds.shape[0], dtype=torch.long, device=inputs_embeds.device
                )
                additional_information = {}
                self.thinker_reply_part = torch.zeros_like(inputs_embeds)
                is_profile = True
            else:
                is_profile = False

            # Ensure we have base embeddings when only ids are provided
            if inputs_embeds is None and input_ids is not None:
                inputs_embeds = self.talker.get_input_embeddings(input_ids)

            # ------- Request-scoped additional information (no cross-request concat) -------
            request_ids: Optional[list[str]] = kwargs.get("request_ids")  # ordered
            request_token_spans: Optional[list[tuple[int,int]]] = kwargs.get("request_token_spans")
            addi_by_req: Optional[dict] = kwargs.get("additional_information_by_req_id")
            runtime_addi = kwargs.get("runtime_additional_information")

            # Normalize runtime_addi into a mapping by request_id for convenience
            runtime_addi_by_req: dict[str, dict] = {}
            if isinstance(request_ids, list) and isinstance(runtime_addi, list) and len(runtime_addi) == len(request_ids):
                for i, rid in enumerate(request_ids):
                    if isinstance(rid, str) and isinstance(runtime_addi[i], dict):
                        runtime_addi_by_req[rid] = runtime_addi[i]
            elif isinstance(request_ids, list) and isinstance(runtime_addi, dict):
                for rid in request_ids:
                    if isinstance(rid, str) and isinstance(runtime_addi.get(rid), dict):
                        runtime_addi_by_req[rid] = runtime_addi[rid]

            # Containers to return per-request updates (e.g., thinker_reply_part_per_request)
            update_by_req_id: dict[str, dict] = {}

            # ------- Prefill: span_len > 1 -------
            if (
                not is_profile
                and isinstance(request_ids, list)
                and isinstance(request_token_spans, list)
                and isinstance(addi_by_req, dict)
            ):
                for idx_req, rid in enumerate(request_ids):
                    s, e = request_token_spans[idx_req]
                    span_len = int(e) - int(s)
                    if span_len <= 1:
                        continue
                    info = addi_by_req.get(rid, {}) if isinstance(rid, str) else {}
                    if not isinstance(info, dict):
                        info = {}
                    pe = info.get("prompt_embeds")  # Tensor [P,H]
                    tr = info.get("thinker_result")  # Tensor [K,H]
                    ptoks = info.get("prompt_token_ids")  # list[int]
                    otoks = info.get("thinker_output_token_ids")  # list[int]

                    if not isinstance(pe, torch.Tensor):
                        pe = torch.zeros(0, self.talker.config.hidden_size, dtype=inputs_embeds.dtype, device=self._module_device(self.model))
                    if not isinstance(tr, torch.Tensor):
                        tr = torch.zeros(0, self.talker.config.hidden_size, dtype=inputs_embeds.dtype, device=self._module_device(self.model))
                    if not isinstance(ptoks, (list, torch.Tensor)):
                        ptoks = []
                    if not isinstance(otoks, (list, torch.Tensor)):
                        otoks = []

                    req_input_ids, req_embeds = self._thinker_to_talker_prefill(
                        voice_type=voice_type,
                        output_prompt_embeds=tr.to(inputs_embeds.dtype).to(self._module_device(self.model)),
                        output_token_ids=otoks,
                        thinker_prompt_embeds=pe.to(inputs_embeds.dtype).to(self._module_device(self.model)),
                        prompt_token_ids=ptoks,
                    )
                    seg_len = min(span_len, req_embeds.shape[0])
                    inputs_embeds[s : s + seg_len] = req_embeds[:seg_len]
                    if isinstance(req_input_ids, torch.Tensor) and req_input_ids.numel() == seg_len:
                        input_ids[s : s + seg_len] = req_input_ids

                    # Prepare per-request reply queue for subsequent decode: drop first row
                    if tr.ndim == 2 and tr.shape[0] > 0:
                        update_by_req_id.setdefault(rid, {})["thinker_reply_part_per_request"] = tr[1:].detach().to("cpu").contiguous()

            # ------- Decode: span_len == 1 -------
            if not is_profile and isinstance(request_ids, list) and isinstance(request_token_spans, list):
                for idx_req, rid in enumerate(request_ids):
                    s, e = request_token_spans[idx_req]
                    if (int(e) - int(s)) != 1:
                        continue
                    # choose step vector in priority order
                    step_vec = None
                    # A) runtime queue
                    q = None
                    if isinstance(rid, str):
                        q = runtime_addi_by_req.get(rid, {}).get("thinker_reply_part_per_request")
                    if isinstance(q, torch.Tensor) and q.numel() > 0:
                        step_vec = q[0:1]
                        new_q = q[1:].detach().to("cpu").contiguous()
                        update_by_req_id.setdefault(rid, {})["thinker_reply_part_per_request"] = new_q
                    else:
                        # B) per-request provided decode vector (optional)
                        info = addi_by_req.get(rid, {}) if isinstance(addi_by_req, dict) else {}
                        dv = info.get("decode_output_prompt_embeds") if isinstance(info, dict) else None
                        if isinstance(dv, torch.Tensor) and dv.numel() > 0:
                            step_vec = dv[0:1] if dv.ndim == 2 else dv.view(1, -1)
                        elif hasattr(self, "thinker_reply_part") and isinstance(self.thinker_reply_part, torch.Tensor) and self.thinker_reply_part.numel() > 0:
                            # C) fallback shared pool
                            step_vec = self.thinker_reply_part[0:1]
                            self.thinker_reply_part = self.thinker_reply_part[1:]

                    if isinstance(step_vec, torch.Tensor) and step_vec.numel() > 0:
                        one_id = input_ids[s : s + 1]
                        _, one_embed = self._thinker_to_talker_decode_one_step(
                            output_prompt_embeds=step_vec.to(inputs_embeds.dtype).to(self._module_device(self.model)),
                            output_token_ids=one_id,
                        )
                        inputs_embeds[s] = one_embed[0]

            with torch.inference_mode():
                talker_hidden = self.talker(
                    input_ids=input_ids,
                    positions=positions[0],
                    inputs_embeds=inputs_embeds,
                )
            multimodal_outputs: dict = None
            # Return updates if any
            if update_by_req_id:
                multimodal_outputs = {"additional_information_update_by_req_id": update_by_req_id}

            if sampling_metadata is not None:
                # the padding token id is set to text model's pad token id, which do not match with the talker model's word embedding size
                sampling_metadata.prompt_token_ids[sampling_metadata.prompt_token_ids == 152064] = 8448

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
            
            code = code[:-1] if code[-1] == TALKER_CODEC_EOS_TOKEN_ID else code
            code = code[1:] if code[0] == TALKER_CODEC_BOS_TOKEN_ID else code
            
            audio_tensor = self.generate_audio(
                code, voice_type
            )
            return OmniOutput(
                text_hidden_states=None, multimodal_outputs={"audio": audio_tensor}
            )

        return OmniOutput(
            text_hidden_states=torch.cat(
                [
                    torch.zeros(
                        [inputs_embeds.shape[0], self.talker.config.hidden_size],
                        dtype=torch.bfloat16,
                    ).to(self._module_device(self.model)),
                    self.talker.thinker_to_talker_proj(
                        self.talker.get_input_embeddings(
                            torch.tensor(
                                [TALKER_CODEC_BOS_TOKEN_ID, TALKER_CODEC_EOS_TOKEN_ID]
                            )
                            .to(torch.bfloat16)
                            .to(self._module_device(self.model))
                        )
                    )[0],
                ],
                dim=0,
            ),
            multimodal_outputs=None,
        )

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
            prompt_token_ids_processed = [
                self.talker_config.tts_codec_mask_token_id
            ] * (input_tokens_len - 2) + [
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
                torch.Tensor(prompt_token_ids_processed)
                .to(torch.int64)
                .to(self._module_device(self.talker))
            )
        return prompt_token_ids_processed, prompt_embeds

    def _thinker_to_talker_decode_one_step(
        self,
        output_prompt_embeds,
        output_token_ids,
    ):
        processed_output_token_embeds = (
            output_prompt_embeds + self.talker.get_input_embeddings(output_token_ids)
        )  # for decode
        return output_token_ids, processed_output_token_embeds

    def compute_logits(
        self, hidden_states: Union[torch.Tensor, OmniOutput]
    ) -> Optional[torch.Tensor]:
        # Handle OmniOutput type
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states

        # Use thinker model for logits computation
        return self.model.compute_logits(hidden_states)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        # Use thinker model for sampling
        return self.model.sample(logits, sampling_metadata)

    def generate_speech(self, text_tokens: torch.Tensor, voice_type: str = "default"):
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
        talker_output = self.talker(
            input_ids=None, positions=None, inputs_embeds=text_tokens
        )

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
            if hasattr(self, "talker_config") and hasattr(
                self.talker_config, "tts_codec_start_token_id"
            ):
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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # optional speaker resources
        conds = getattr(self.token2wav_config, "conds", None)
        ref_mels = getattr(self.token2wav_config, "ref_mels", None)
        if isinstance(conds, dict) and isinstance(ref_mels, dict):
            self._token2wav_conds = {
                k: torch.as_tensor(v, device=device) for k, v in conds.items()
            }
            self._token2wav_ref_mels = {
                k: torch.as_tensor(v, device=device) for k, v in ref_mels.items()
            }
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
                for f in sorted(
                    glob.glob(os.path.join(model_path, "inputs", "*spk_emb.npy"))
                ):
                    key = os.path.basename(f).split("_")[0].lower()
                    self._token2wav_conds[key] = torch.as_tensor(
                        np.load(f), device=device
                    )
                for f in sorted(
                    glob.glob(os.path.join(model_path, "inputs", "*ref_mel.npy"))
                ):
                    key = os.path.basename(f).split("_")[0].lower()
                    self._token2wav_ref_mels[key] = torch.as_tensor(
                        np.load(f), device=device
                    )

    def _codec_to_audio(
        self, codec_tokens: torch.Tensor, voice_type: str = "default"
    ) -> Optional[torch.Tensor]:
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
            codec = torch.as_tensor(
                codec_tokens, dtype=torch.long, device=token2wav_dev
            ).unsqueeze(0)

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
        y_all = torch.randn(
            (1, total_mel, mel_dim), dtype=ref_mel.dtype, device=token2wav_dev
        )

        logger.info(
            "Currently, we do not use the chunked process, we only use the "
            "token2wav.process_chunk for the whole sequence. "
            "The stream mode will be implemented in the future."
        )

        chunk_ends = []
        for i in range(codec.shape[1]):
            chunk_code_length = i * 2 - 24
            finished = i == (codec.shape[1] - 1)
            if (
                chunk_code_length > 0 and chunk_code_length % chunk_size == 0
            ) or finished:
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

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
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
                thinker_embedding_weights = [
                    w
                    for n, w in thinker_weights
                    if n == "thinker.model.embed_tokens.weight"
                ]
                if thinker_embedding_weights:
                    self.thinker_embedding = nn.Embedding(
                        thinker_embedding_weights[0].shape[0],
                        thinker_embedding_weights[0].shape[1],
                    )
                    self.thinker_embedding.weight = nn.Parameter(
                        thinker_embedding_weights[0].to(
                            self._module_device(self.talker)
                        )
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
            t2w_loaded = self.token2wav.load_weights(
                token2wav_weights, os.path.join(hf_model_folder, "spk_dict.pt")
            )
            t2w_loaded = add_prefix_to_loaded_weights(t2w_loaded, "token2wav")
            loaded_weights.update(t2w_loaded)

        return loaded_weights
