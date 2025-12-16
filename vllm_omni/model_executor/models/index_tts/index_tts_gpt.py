"""
IndexTTS GPT Stage (Stage 0): Text → Semantic Codes

This stage takes text input + speaker/emotion conditioning and generates
discrete semantic codes autoregressively using a GPT-2 based model.
"""

from collections.abc import Iterable

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper

from vllm_omni.model_executor.models.index_tts.gpt.model_v2 import UnifiedVoice
from vllm_omni.model_executor.models.index_tts.index_tts_config import IndexTTSConfig

from .utils.checkpoint import load_checkpoint

logger = init_logger(__name__)


class IndexTTSGPTForConditionalGeneration(nn.Module):
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

    def forward(
        self,
        text_tokens: torch.Tensor,  # [B, T_text]
        spk_cond_emb: torch.Tensor,  # [B, T_spk, D]
        emo_cond_emb: torch.Tensor,  # [B, T_emo, D]
        emo_alpha: float = 1.0,
        emo_vector=None,
        **generation_kwargs,
    ) -> dict:
        """
        Forward pass for GPT-based semantic code generation.
        Returns dict with:
            - "gpt_codes": Discrete codes [B, T_codes]
            - "gpt_latent": Continuous latent [B, T_codes, D]
            - "gpt_code_lens": Sequence lengths [B]
        """
        device = text_tokens.device

        ## Combining speaker and emotion conditioning
        emovec = self.gpt.merge_emovec(
            spk_cond_emb,
            emo_cond_emb,
            torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device),
            torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens.device),
            alpha=emo_alpha,
        )

        top_p = generation_kwargs.pop("top_p", 0.8)
        top_k = generation_kwargs.pop("top_k", 30)
        temperature = generation_kwargs.pop("temperature", 0.8)
        autoregressive_batch_size = 1
        length_penalty = generation_kwargs.pop("length_penalty", 0.0)
        num_beams = generation_kwargs.pop("num_beams", 3)
        repetition_penalty = generation_kwargs.pop("repetition_penalty", 10.0)
        max_mel_tokens = generation_kwargs.pop("max_mel_tokens", 1500)

        # GPT inference - generate discrete codes
        codes, speech_conditioning_latent = self.gpt.inference_speech(
            spk_cond_emb,
            text_tokens,
            emo_cond_emb,
            cond_lengths=torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device),
            emo_cond_lengths=torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens.device),
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
            **generation_kwargs,
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
            text_tokens,
            torch.tensor([text_tokens.shape[-1]], device=text_tokens.device),
            codes,
            torch.tensor([codes.shape[-1]], device=text_tokens.device),
            emo_cond_emb,
            cond_mel_lengths=torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device),
            emo_cond_mel_lengths=torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens.device),
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
