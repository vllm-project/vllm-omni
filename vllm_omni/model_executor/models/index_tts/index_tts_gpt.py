"""
IndexTTS GPT Stage (Stage 0): Text → Semantic Codes

This stage takes text input + speaker/emotion conditioning and generates
discrete semantic codes autoregressively using a GPT-2 based model.
"""

from collections.abc import Iterable, Mapping
from typing import Any

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.parse import (
    ModalityDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import BaseMultiModalProcessor, BaseProcessingInfo
from vllm.multimodal.profiling import BaseDummyInputsBuilder

from vllm_omni.model_executor.models.index_tts.gpt.model_v2 import UnifiedVoice
from vllm_omni.model_executor.models.index_tts.index_tts_config import IndexTTSConfig

from .utils.checkpoint import load_checkpoint


class IndexTTSEmbeddingItems(ModalityDataItems):
    """
    Data items for IndexTTS custom embeddings (spk_cond_emb, etc.)
    """

    pass


class IndexTTSGPTDataParser(MultiModalDataParser):
    def _parse_index_tts_embedding(self, data) -> IndexTTSEmbeddingItems:
        if isinstance(data, torch.Tensor):
            return IndexTTSEmbeddingItems([data])
        if isinstance(data, list):
            return IndexTTSEmbeddingItems(data)
        return IndexTTSEmbeddingItems([data])

    def _get_subparsers(self):
        parsers = super()._get_subparsers()
        for key in ["input_ids", "spk_cond_emb", "emo_cond_emb", "emovec_mat", "weight_vector", "emo_vector"]:
            parsers[key] = self._parse_index_tts_embedding
        return parsers


logger = init_logger(__name__)


def index_tts_gpt_field_config(hf_inputs: Mapping[str, torch.Tensor]):
    return dict(
        spk_cond_emb=MultiModalFieldConfig.batched("spk_cond_emb"),
        emo_cond_emb=MultiModalFieldConfig.batched("emo_cond_emb"),
        emovec_mat=MultiModalFieldConfig.batched("emovec_mat"),
        weight_vector=MultiModalFieldConfig.batched("weight_vector"),
        emo_vector=MultiModalFieldConfig.batched("emo_vector"),
    )


class IndexTTSGPTProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(IndexTTSConfig)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"spk_cond_emb": None, "emo_cond_emb": None}


class IndexTTSGPTDummyInputsBuilder(BaseDummyInputsBuilder[IndexTTSGPTProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return "Hello world"

    def get_dummy_mm_data(self, seq_len: int, mm_counts: Mapping[str, int]) -> dict[str, Any]:
        # TODO: Get hidden size from config
        hidden_size = 1024  # Standard default
        config = self.info.get_hf_config()
        # Access hidden size if available in gpt config or semantic codec
        hidden_size = config.gpt.get("hidden_size", 1024)

        return {
            "spk_cond_emb": torch.randn(1, 10, hidden_size),
            "emo_cond_emb": torch.randn(1, 10, hidden_size),
            "emovec_mat": torch.randn(1, 8, hidden_size),
            "weight_vector": torch.randn(1, 8),
            "emo_vector": torch.randn(1, 8),
        }

    def get_dummy_inputs(self, *args, **kwargs):
        hidden_size = 1024
        config = self.info.get_hf_config()
        hidden_size = config.gpt.get("hidden_size", 1024)

        return {
            "input_ids": torch.randint(0, 1000, (1, 12), dtype=torch.long),
            "spk_cond_emb": torch.randn(1, 10, hidden_size),
            "emo_cond_emb": torch.randn(1, 10, hidden_size),
            "emovec_mat": torch.randn(1, 8, hidden_size),
            "weight_vector": torch.randn(1, 8),
            "emo_vector": torch.randn(1, 8),
            "positions": torch.arange(0, 12, dtype=torch.long).unsqueeze(0),
        }


class IndexTTSGPTMultiModalProcessor(BaseMultiModalProcessor[IndexTTSGPTProcessingInfo]):
    def _call_support(self, prompt, mm_data, mm_kwargs):
        # Pass through multimodal data
        return mm_data

    def _get_data_parser(self) -> MultiModalDataParser:
        return IndexTTSGPTDataParser()

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        return index_tts_gpt_field_config(hf_inputs)

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        """
        IndexTTS doesn't insert special tokens into prompts like vision models.
        Audio conditioning is passed separately, not embedded in text.
        """
        return []  # No prompt updates needed


@MULTIMODAL_REGISTRY.register_processor(
    IndexTTSGPTMultiModalProcessor,
    info=IndexTTSGPTProcessingInfo,
    dummy_inputs=IndexTTSGPTDummyInputsBuilder,
)
class IndexTTSGPTForConditionalGeneration(nn.Module, SupportsMultiModal):
    """
    Stage 0: GPT model for generating semantic codes from text.
    Input: text_tokens, spk_cond_emb, emo_cond_emb
    Output: codes (discrete), latent (continuous), code_lens
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config: IndexTTSConfig = vllm_config.model_config.hf_config
        self.prefix = prefix

        self.gpt = UnifiedVoice(**self.config.gpt)
        self.stop_mel_token = self.config.gpt["stop_mel_token"]

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,  # [B, T_text]
        positions: torch.Tensor,
        spk_cond_emb: torch.Tensor,  # [B, T_spk, D]
        emo_cond_emb: torch.Tensor,  # [B, T_emo, D]
        emovec_mat: torch.Tensor,
        weight_vector: torch.Tensor,
        emo_vector: torch.Tensor,
        emo_alpha: float = 1.0,
        **kwargs,
    ) -> dict:
        """
        Forward pass for GPT-based semantic code generation.
        Returns dict with:
            - "gpt_codes": Discrete codes [B, T_codes]
            - "gpt_latent": Continuous latent [B, T_codes, D]
            - "gpt_code_lens": Sequence lengths [B]
        """
        device = next(self.parameters()).device

        ## Combining speaker and emotion conditioning
        emovec = self.gpt.merge_emovec(
            spk_cond_emb,
            emo_cond_emb,
            torch.tensor([spk_cond_emb.shape[-1]], device=device),
            torch.tensor([emo_cond_emb.shape[-1]], device=device),
            alpha=emo_alpha,
        )
        if emo_vector is not None:
            emovec = emovec_mat + (1 - torch.sum(weight_vector)) * emovec

        top_p = kwargs.pop("top_p", 0.8)
        top_k = kwargs.pop("top_k", 30)
        temperature = kwargs.pop("temperature", 0.8)
        autoregressive_batch_size = 1
        length_penalty = kwargs.pop("length_penalty", 0.0)
        num_beams = kwargs.pop("num_beams", 3)
        repetition_penalty = kwargs.pop("repetition_penalty", 10.0)
        max_mel_tokens = kwargs.pop("max_mel_tokens", 1500)

        # GPT inference - generate discrete codes
        codes, speech_conditioning_latent = self.gpt.inference_speech(
            spk_cond_emb,
            input_ids,
            emo_cond_emb,
            cond_lengths=torch.tensor([spk_cond_emb.shape[-1]], device=device),
            emo_cond_lengths=torch.tensor([emo_cond_emb.shape[-1]], device=device),
            emo_vec=emovec,
            do_sample=True,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            num_return_sequences=autoregressive_batch_size,
            length_penalty=length_penalty,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            max_generate_length=max_mel_tokens,
            **kwargs,
        )

        code_lens = []
        max_code_len = 0
        for code in codes:
            if self.stop_mel_token not in code:
                code_len = len(code)
            else:
                # Find stop token position
                len_ = (code == self.stop_mel_token).nonzero(as_tuple=False)[0]
                code_len = len_[0].item() if len_.numel() > 0 else len(code)
            code_lens.append(code_len)
        code_lens = torch.LongTensor(code_lens).to(device)
        max_code_len = max(max_code_len, code_len)
        codes = codes[:, :max_code_len]

        # GPT forward pass - refine codes to latent
        use_speed = torch.zeros(spk_cond_emb.size(0)).to(spk_cond_emb.device).long()
        latent = self.gpt(
            speech_conditioning_latent,
            input_ids,
            torch.tensor([input_ids.shape[-1]], device=device),
            codes,
            torch.tensor([codes.shape[-1]], device=device),
            emo_cond_emb,
            cond_mel_lengths=torch.tensor([spk_cond_emb.shape[-1]], device=device),
            emo_cond_mel_lengths=torch.tensor([emo_cond_emb.shape[-1]], device=device),
            emo_vec=emovec,
            use_speed=use_speed,
        )

        return {
            "gpt_codes": codes,
            "gpt_latent": latent,
            "gpt_code_lens": code_lens,
        }

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        mapper = WeightsMapper(
            orig_to_new_prefix={
                "gpt.": "",
            }
        )
        gpt_ckpt = getattr(self.config, "gpt_checkpoint")

        repo_id = getattr(self.config, "repo_id")

        if not gpt_ckpt:
            raise RuntimeError("IndexTTS: Missing config.gpt_checkpoint")
        gpt_path = hf_hub_download(repo_id, filename=gpt_ckpt)
        load_checkpoint(self.gpt, gpt_path)
        self.gpt.post_init_gpt2_config(use_deepspeed=False)

        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=mapper)
