# Copyright 2026 Tencent.
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn
from transformers import BatchFeature, WhisperFeatureExtractor
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.models import SupportsPP
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.utils import init_vllm_registered_model
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    ProcessorInputs,
    PromptReplacement,
    PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_omni.model_executor.custom_process_mixin import CustomProcessMixin
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.utils import add_prefix_to_loaded_weights

# Max tokens per audio: Whisper 30s → 3000 mel frames → encoder 2x + adapter 8x = 16x → 188
MAX_AUDIO_TOKENS = 188


def _calc_audio_num_tokens(num_samples: int, sample_rate: int) -> int:
    """Calculate the number of audio tokens for a given audio.

    Matches the original Covo-Audio calc_seq_len: 4 rounds of stride-2
    downsampling (Whisper encoder 2x + AudioAdapter 3-layer 8x = 16x total).
    """
    centiseconds = num_samples * 100 // sample_rate
    seq_len = centiseconds
    for _ in range(4):
        seq_len = (seq_len + 1) // 2
    return seq_len


class CovoAudioProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_hf_processor(self, **kwargs):
        return None

    def get_feature_extractor(self, **kwargs):
        return WhisperFeatureExtractor(
            feature_size=128,
            sampling_rate=16000,
        )

    def get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(
            target_sr=16000,
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}

    def get_mm_max_tokens_per_item(self, seq_len, mm_counts):
        return {"audio": MAX_AUDIO_TOKENS}


class CovoAudioDummyInputsBuilder(BaseDummyInputsBuilder[CovoAudioProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        return (" <|begofcAUDIO|><|cAUDIO|><|endofcAUDIO|>" * num_audios).strip()

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        if num_audios == 0:
            return {}
        # 30s at 16kHz = Whisper's max input length, ensures profile_run
        # allocates memory for the worst-case MAX_AUDIO_TOKENS (188) tokens.
        dummy_audio = np.zeros((16000 * 30,), dtype=np.float32)
        return {"audio": [(dummy_audio, 16000)] * num_audios}

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> ProcessorInputs:
        dummy_text = self.get_dummy_text(mm_counts)
        dummy_mm_data = self.get_dummy_mm_data(seq_len, mm_counts, mm_options)
        dummy_mm_items = self.info.parse_mm_data(dummy_mm_data)
        return ProcessorInputs(
            prompt=dummy_text,
            mm_data_items=dummy_mm_items,
        )


class CovoAudioMultiModalProcessor(BaseMultiModalProcessor[CovoAudioProcessingInfo]):
    def _hf_processor_applies_updates(self, prompt_text, mm_items, hf_processor_mm_kwargs, tokenization_kwargs) -> bool:
        return False

    def _call_hf_processor(
        self, prompt: str, mm_data: Mapping[str, object], mm_kwargs: Mapping[str, Any], tok_kwargs: Mapping[str, object]
    ) -> BatchFeature:
        tokenizer = self.info.get_tokenizer()
        feature_extractor = self.info.get_feature_extractor()

        if isinstance(prompt, str):
            prompt_ids = tokenizer.encode(prompt)
        else:
            prompt_ids = prompt

        audios = mm_data.get("audios", []) or mm_data.get("audio", [])
        if not audios:
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        raw_audios = []
        num_tokens_list = []
        for item in audios:
            if isinstance(item, tuple):
                audio_array, _sr = item
            else:
                audio_array = item
            raw_audios.append(audio_array)
            # MultiModalDataParser resamples to target_sr=16000, so always
            # compute token count at 16 kHz regardless of the original sr.
            num_tokens_list.append(_calc_audio_num_tokens(len(audio_array), 16000))

        features = feature_extractor(
            raw_audios,
            sampling_rate=16000,
            return_tensors="pt",
        )

        return BatchFeature(
            data={
                "input_ids": torch.tensor([prompt_ids], dtype=torch.int64),
                "audio_features": features["input_features"],
                "audio_num_tokens": torch.tensor(num_tokens_list, dtype=torch.int64),
            },
            tensor_type=None,
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return {
            "audio_features": MultiModalFieldConfig.batched("audio"),
            "audio_num_tokens": MultiModalFieldConfig.batched("audio"),
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs,
    ) -> Sequence[PromptReplacement]:
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()
        audio_token_id = vocab["<|cAUDIO|>"]

        out_mm_data = out_mm_kwargs.get_data()

        def get_replacement(item_idx: int):
            num_tokens_tensor = out_mm_data.get("audio_num_tokens")
            if num_tokens_tensor is not None:
                num_tokens = int(num_tokens_tensor[item_idx])
            else:
                num_tokens = MAX_AUDIO_TOKENS
            return PromptUpdateDetails.select_token_id(
                [audio_token_id] * num_tokens,
                embed_token_id=audio_token_id,
            )

        return [
            PromptReplacement("audio", "<|cAUDIO|>", get_replacement),
        ]


@MULTIMODAL_REGISTRY.register_processor(
    CovoAudioMultiModalProcessor,
    info=CovoAudioProcessingInfo,
    dummy_inputs=CovoAudioDummyInputsBuilder,
)
class CovoAudioForConditionalGeneration(
    nn.Module,
    SupportsPP,
    SupportsMultiModal,
    CustomProcessMixin,
):
    requires_raw_input_tokens: bool = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return "<|begofcAUDIO|><|cAUDIO|><|endofcAUDIO|>"
        return None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.has_preprocess = False
        self.have_multimodal_outputs = True
        self.vllm_config = vllm_config

        self.model_stage = vllm_config.model_config.model_stage
        if self.model_stage == "fused_thinker_talker":
            self.has_preprocess = True
            self.set_custom_preprocess(self.fused_thinker_talker_preprocess)
            self.fused_thinker_talker = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=vllm_config.model_config.hf_config,
                architectures=["CovoAudioLLMModel"],
            )
            self.code2wav = None
            self.model = self.fused_thinker_talker

        elif self.model_stage == "code2wav":
            self.code2wav = init_vllm_registered_model(
                vllm_config=vllm_config,
                architectures=["CovoAudioCode2WavModel"],
            )
            self.fused_thinker_talker = None
            self.model = self.code2wav
        else:
            raise ValueError("Invalid model stage")

        self.make_empty_intermediate_tensors = (
            self.fused_thinker_talker.make_empty_intermediate_tensors
            if self.model_stage == "fused_thinker_talker"
            else lambda *args, **kwargs: None
        )

    def fused_thinker_talker_preprocess(
        self,
        input_ids: torch.Tensor | None,
        input_embeds: torch.Tensor | None,
        **info_dict: dict,
    ):
        if input_embeds is None and input_ids is not None:
            input_embeds = self.fused_thinker_talker.embed_input_ids(input_ids)
        return input_ids, input_embeds, info_dict

    def embed_multimodal(self, **kwargs: object):
        if self.model_stage == "fused_thinker_talker":
            return self.fused_thinker_talker.embed_multimodal(**kwargs)
        return None

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal: bool = False,
    ) -> torch.Tensor:
        if self.model_stage == "code2wav":
            hidden_size = self.vllm_config.model_config.get_hidden_size()
            return torch.zeros(
                input_ids.numel(),
                hidden_size,
                dtype=self.vllm_config.model_config.dtype,
                device=input_ids.device,
            )
        return self.model.embed_input_ids(input_ids, multimodal_embeddings, is_multimodal=is_multimodal)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata | None = None,
    ) -> torch.Tensor | None:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        return self.model.compute_logits(hidden_states)

    @staticmethod
    def _module_device(module: nn.Module) -> torch.device:
        try:
            return next(module.parameters()).device
        except StopIteration:
            # No parameters; fall back to buffers or cpu
            for _, buf in module.named_buffers(recurse=True):
                return buf.device
            return torch.device("cpu")

    def generate_codes(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        **kwargs: object,
    ):
        llm_dev = self._module_device(self.fused_thinker_talker)

        # Fallback: dummy input_ids matching embedding length
        if input_ids is None:
            input_ids = torch.zeros(inputs_embeds.shape[1], dtype=torch.long, device=llm_dev).unsqueeze(
                0
            )  # (1, seq_len)

        # Thinker (ensure inputs on thinker's device)
        if input_ids is not None and input_ids.device != llm_dev:
            input_ids = input_ids.to(llm_dev)
        if positions is not None and positions.device != llm_dev:
            positions = positions.to(llm_dev)
        if inputs_embeds is not None and inputs_embeds.device != llm_dev:
            inputs_embeds = inputs_embeds.to(llm_dev)

        # Run llm
        llm_output = self.fused_thinker_talker(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        # CovoAudio interleaves audio tokens directly in the LLM output
        # token stream; the stage input processor extracts them from
        # token_ids, so there is no separate codec output here.
        return None, llm_output

    def generate_audio(self, code: torch.Tensor):
        code2wav_dev = self._module_device(self.code2wav)

        if isinstance(code, torch.Tensor):
            code_tensor = code.to(dtype=torch.long, device=code2wav_dev)
        else:
            code_tensor = torch.as_tensor(code, dtype=torch.long, device=code2wav_dev)

        if code_tensor.ndim == 2 and code_tensor.shape[0] == 1:
            code_tensor = code_tensor.squeeze(0)

        with torch.inference_mode():
            audio_tensor = self.code2wav(input_ids=code_tensor)

        return audio_tensor

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights for all components of the omni model."""
        loaded_weights: set[str] = set()

        # code2wav weights are loaded in its __init__ via torch.load;
        # skip iterating safetensors entirely for this stage.
        if self.code2wav:
            c2w_loaded = self.code2wav.load_weights(iter([]))
            c2w_loaded = add_prefix_to_loaded_weights(c2w_loaded, "code2wav")
            loaded_weights.update(c2w_loaded)
            return loaded_weights

        llm_weights = ((k, v) for k, v in weights if k.startswith(("llm", "encoder", "audio_adapter")))
        if self.fused_thinker_talker:
            llm_loaded = self.fused_thinker_talker.load_weights(llm_weights)
            llm_loaded = add_prefix_to_loaded_weights(llm_loaded, "fused_thinker_talker")
            loaded_weights.update(llm_loaded)

        return loaded_weights

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        sampling_metadata: SamplingMetadata | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors | OmniOutput:
        """Route forward to the active stage.

        - fused_thinker_talker: run Whisper + LLM → text hidden states
          (audio tokens are interleaved in the output token stream).
        - code2wav: run BigVGAN vocoder on audio codes → waveform.
        """
        if self.model_stage == "fused_thinker_talker":
            _, text_hidden_states = self.generate_codes(
                input_ids=input_ids,
                positions=positions,
                inputs_embeds=inputs_embeds,
                intermediate_tensors=intermediate_tensors,
                **kwargs,
            )

            return OmniOutput(
                text_hidden_states=text_hidden_states.reshape(-1, text_hidden_states.shape[-1]),
                multimodal_outputs={},
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

            # Text-only response: stage processor sends a single sentinel [-1]
            # token.  Skip BigVGAN inference and return a short silence so the
            # serving layer produces a valid WAV instead of noise.
            if code.numel() == 1 and int(code.flatten()[0]) == -1:
                # 0.1s of silence at 24 kHz
                return OmniOutput(
                    text_hidden_states=None,
                    multimodal_outputs={"model_outputs": torch.zeros(1, 2400, device=code.device)},
                )

            audio_tensor = self.generate_audio(code)
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "model_outputs": audio_tensor.reshape(1, -1) if audio_tensor is not None else audio_tensor
                },
            )
