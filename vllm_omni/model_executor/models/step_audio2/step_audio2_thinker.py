from collections.abc import Iterable, Mapping, Sequence
from functools import cached_property
from typing import Optional, List, Tuple, TypedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
from vllm.model_executor.models.utils import (
    init_vllm_registered_model,
    maybe_prefix,
    merge_multimodal_embeddings,
    flatten_bn,
)
from vllm.sequence import IntermediateTensors
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargs,
    NestedTensors,
)
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder

from vllm_omni.model_executor.models.output_templates import OmniOutput
from .step_audio2_encoder import AudioEncoder, Adaptor

logger = init_logger(__name__)


# Step-Audio2 token ranges
STEP_AUDIO2_TEXT_MAX = 151688
STEP_AUDIO2_AUDIO_START = 151696
STEP_AUDIO2_AUDIO_VOCAB_SIZE = 6562
STEP_AUDIO2_AUDIO_EOS = 6561
AUDIO_PATCH_TOKEN_ID = 151690  # <audio_patch> token


class Step1fAudioInputs(TypedDict):
    """Audio inputs for Step-Audio2"""
    audio_mels: torch.Tensor
    """Shape: (num_audios * num_frames, num_mel_bins)"""

    audio_lens: list[int]
    """Shape: (num_audios,)"""


class StepAudio2ProcessingInfo(BaseProcessingInfo):
    """Processing info for Step-Audio2"""

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"audio": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        # Audio encoder output length calculation
        # After conv2 (stride=2): T//2
        # After avg_pool (stride=2): T//4
        # After adaptor conv (stride=2): T//8
        # Approximately: max_feature_len // 8
        max_audio_tokens = 250  # Conservative estimate for 25s audio
        return {"audio": max_audio_tokens}


class StepAudio2DummyInputsBuilder(BaseDummyInputsBuilder[StepAudio2ProcessingInfo]):
    """Dummy inputs builder for Step-Audio2"""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        return "<audio_patch>" * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        # 25s audio at 16kHz
        audio_len = 16000 * 25
        num_audios = mm_counts.get("audio", 0)
        return {
            "audio": self._get_dummy_audios(length=audio_len, num_audios=num_audios)
        }


class StepAudio2MultiModalProcessor(BaseMultiModalProcessor[StepAudio2ProcessingInfo]):
    """Multi-modal processor for Step-Audio2"""

    def _get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(target_sr=16000)

    def _get_mm_fields_config(
        self,
        hf_inputs,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        audio_lens = hf_inputs.get("audio_lens", torch.empty(0))

        return dict(
            audio_mels=MultiModalFieldConfig.flat_from_sizes("audio", audio_lens),
            audio_lens=MultiModalFieldConfig.batched("audio"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        # Calculate actual audio feature lengths from audio_lens
        audio_lens = out_mm_kwargs.get("audio_lens", [])
        if audio_lens is not None and hasattr(audio_lens, '__iter__'):
            # Convert to list if tensor
            if isinstance(audio_lens, torch.Tensor):
                audio_lens = flatten_bn(audio_lens, concat=True).tolist()
            # Calculate feature lengths: (audio_len - 1) // 8 + 1 (after encoder + adapter)
            feature_lens = [max(1, (length - 1) // 8 + 1) for length in audio_lens]
        else:
            # Fallback to conservative estimate
            feature_lens = [250]  # max_audio_tokens from get_mm_max_tokens_per_item

        return [
            PromptReplacement(
                modality="audio",
                target=[AUDIO_PATCH_TOKEN_ID],
                replacement=lambda item_idx: PromptUpdateDetails.select_token_id(
                    seq=[AUDIO_PATCH_TOKEN_ID] * feature_lens[item_idx],
                    embed_token_id=AUDIO_PATCH_TOKEN_ID,
                ),
            )
        ]


class StepAudio2ThinkerForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):
    """
    Step-Audio2 Thinker - Stage 1 LLM

    Architecture:
        AudioEncoder (6 layers, 512 hidden) →
        Adaptor (512 → LLM dim) →
        Qwen2 LLM (vocab=64012)
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        # Get encoder config
        if hasattr(config, 'audio_encoder_config'):
            encoder_config = config.audio_encoder_config
        else:
            # Default config
            class DefaultEncoderConfig:
                n_mels = 128
                n_audio_ctx = 1500
                n_audio_state = 512
                n_audio_head = 8
                n_audio_layer = 6
                llm_dim = config.hidden_size if hasattr(config, 'hidden_size') else 4096
                kernel_size = 3
                adapter_stride = 2
            encoder_config = DefaultEncoderConfig()

        # Initialize audio encoder
        self.encoder = AudioEncoder(
            n_mels=encoder_config.n_mels,
            n_ctx=encoder_config.n_audio_ctx,
            n_state=encoder_config.n_audio_state,
            n_head=encoder_config.n_audio_head,
            n_layer=encoder_config.n_audio_layer,
        )

        # Initialize adapter
        self.adapter = Adaptor(
            n_state=encoder_config.n_audio_state,
            n_hidden=encoder_config.llm_dim,
            kernel_size=encoder_config.kernel_size,
            stride=encoder_config.adapter_stride,
        )

        # Initialize language model (Qwen2)
        # Use text_config if available, otherwise use main config
        text_config = getattr(config, 'text_config', config)
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=text_config,
            prefix=maybe_prefix(prefix, "language_model")
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        logger.info(
            f"Initialized Step-Audio2 Thinker with encoder: "
            f"{encoder_config.n_audio_layer} layers, "
            f"{encoder_config.n_audio_state} hidden, "
            f"{encoder_config.llm_dim} LLM dim"
        )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("audio"):
            return "<audio_patch>"
        raise ValueError("Only audio modality is supported")

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def _parse_and_validate_audio_input(
        self, **kwargs: object
    ) -> Optional[Step1fAudioInputs]:
        """Parse audio inputs from kwargs"""
        audio_mels = kwargs.get("audio_mels", None)
        audio_lens = kwargs.get("audio_lens", None)

        if audio_mels is None:
            return None

        # Flatten batch dimensions
        audio_mels = flatten_bn(audio_mels, concat=True)
        audio_lens = flatten_bn(audio_lens, concat=True).tolist()

        # Split into list based on lengths
        audio_mels_lst = []
        cur_idx = 0
        for audio_len in audio_lens:
            audio_mels_lst.append(audio_mels[cur_idx:cur_idx + audio_len])
            cur_idx += audio_len

        # Pad to same length
        max_len = max(x.size(0) for x in audio_mels_lst)
        audio_mels = torch.stack(
            [F.pad(x, (0, 0, 0, max_len - x.size(0))) for x in audio_mels_lst],
            dim=0
        )

        return Step1fAudioInputs(
            audio_mels=audio_mels.to(self.dtype).to(self.device),
            audio_lens=audio_lens,
        )

    def _process_audio_input(
        self, audio_input: Step1fAudioInputs
    ) -> tuple[torch.Tensor, ...]:
        """Process audio mels through encoder and adapter"""
        audio_mels = audio_input["audio_mels"]
        audio_lens = torch.tensor(audio_input["audio_lens"], device=self.device)

        # Permute to (B, n_mels, T)
        audio_mels = audio_mels.permute(0, 2, 1)

        # Encode audio
        audio_features, audio_lens = self.encoder(audio_mels, audio_lens)

        # Adapt to LLM dimension
        audio_features = self.adapter(audio_features)

        # Calculate feature lengths after adapter conv
        audio_feature_lens = (audio_lens - 1) // 2 + 1

        # Split into list
        audio_feature_list = [
            audio_features[i, :audio_feature_lens[i]]
            for i in range(audio_features.size(0))
        ]

        return audio_feature_list

    def get_multimodal_embeddings(self, **kwargs) -> Optional[NestedTensors]:
        """Get multimodal embeddings"""
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return None
        else:
            audio_embeddings = self._process_audio_input(audio_input)
            return audio_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[NestedTensors] = None,
    ) -> torch.Tensor:
        """Get input embeddings with multimodal fusion"""
        inputs_embeds = self.language_model.model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
                AUDIO_PATCH_TOKEN_ID
            )
        return inputs_embeds

    def get_language_model(self) -> nn.Module:
        return self.language_model

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ):
        """Forward pass - returns OmniOutput for multi-stage pipeline"""
        if intermediate_tensors is not None:
            inputs_embeds = None
        elif inputs_embeds is None:
            audio_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids, audio_embeddings)
            input_ids = None

        hidden_states = self.language_model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        # Return OmniOutput for multi-stage compatibility
        # Audio tokens will be extracted from generated token_ids by the input processor
        return OmniOutput(
            text_hidden_states=hidden_states,
            multimodal_outputs={},  # Empty for now, audio tokens extracted later
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights with name mapping from HuggingFace to vLLM structure"""
        from vllm.model_executor.models.utils import AutoWeightsLoader

        # Map HuggingFace weight names to our model structure
        # HF: model.*, lm_head.* → Our: language_model.model.*, language_model.lm_head.*
        # HF: encoder.*, adapter.* → Our: encoder.*, adapter.* (no change)
        def weight_mapper(name: str) -> str:
            if name.startswith("lm_head."):
                return name.replace("lm_head.", "language_model.lm_head.")
            elif name.startswith("model."):
                return name.replace("model.", "language_model.model.")
            else:
                # encoder.* and adapter.* remain unchanged
                return name

        # Apply mapping to weights
        mapped_weights = [(weight_mapper(name), tensor) for name, tensor in weights]

        loader = AutoWeightsLoader(self)
        return loader.load_weights(mapped_weights)

    @staticmethod
    def separate_tokens(
        token_ids: List[int],
        text_max: int = STEP_AUDIO2_TEXT_MAX,
        audio_start: int = STEP_AUDIO2_AUDIO_START
    ) -> Tuple[List[int], List[int]]:
        """Separate generated tokens into text and audio tokens"""
        text_tokens = [tid for tid in token_ids if tid < text_max]
        audio_tokens = [tid - audio_start for tid in token_ids if tid >= audio_start]
        return text_tokens, audio_tokens

    @staticmethod
    def has_audio_output(
        token_ids: List[int],
        audio_start: int = STEP_AUDIO2_AUDIO_START
    ) -> bool:
        """Check if generated tokens contain audio tokens"""
        return any(tid >= audio_start for tid in token_ids)


class StepAudio2OutputProcessor:
    """Helper class to process Step-Audio2 outputs"""

    @staticmethod
    def process_output(
        output_ids: torch.Tensor,
        tokenizer,
        remove_audio_padding: bool = True
    ) -> dict:
        """Process model output and separate text/audio tokens"""
        if isinstance(output_ids, torch.Tensor):
            output_ids = output_ids.squeeze().tolist()

        # Separate tokens
        text_tokens, audio_tokens = StepAudio2ThinkerForConditionalGeneration.separate_tokens(
            output_ids
        )

        # Remove audio padding if requested
        if remove_audio_padding and audio_tokens:
            audio_tokens = [t for t in audio_tokens if t < STEP_AUDIO2_AUDIO_EOS]

        # Decode text
        text = tokenizer.decode(text_tokens, skip_special_tokens=False)

        return {
            'text': text,
            'text_tokens': text_tokens,
            'audio_tokens': audio_tokens,
            'has_audio': len(audio_tokens) > 0,
            'all_tokens': output_ids
        }
