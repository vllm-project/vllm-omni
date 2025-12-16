"""
Index TTS S2Mel Stage (Stage 1): Converts semantic codes to mel-spectrograms.

This stage takes input semantic codes along with speaker and emotion conditioning
to generate mel-spectrograms using a MaskGCT-based model.
"""

from collections.abc import Iterable

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper

from vllm_omni.model_executor.models.index_tts.index_tts_config import IndexTTSConfig
from vllm_omni.model_executor.models.index_tts.s2mel.modules.commons import MyModel, load_checkpoint2
from vllm_omni.model_executor.models.index_tts.utils.maskgct_utils import JsonHParams, build_semantic_codec

logger = init_logger(__name__)


class IndexTTSS2MelForConditionalGeneration(nn.Module):
    """
    Stage 1: S2Mel model for generating mel-spectrograms from semantic codes.
    Input: semantic_codes, spk_cond_emb, emo_cond_emb
    Output: mel_spectrograms
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config: IndexTTSConfig = VllmConfig.model_config.hf_config
        self.prefix = prefix
        self.semantic_codec = build_semantic_codec(self.config.semantic_codec)
        self.s2mel = MyModel(self.s2mel_cfg, use_gpt_latent=True)
        self.s2mel_cfg = JsonHParams(**self.config.s2mel)

    def forward(
        self,
        codes,  # [B, T_codes]
        latent,  # [B, T_codes, D]
        code_lens,  # [B]
        prompt_condition: torch.Tensor,  # [B, D, T_prompt]
        ref_mel: torch.Tensor,  # [B, D, T_ref]
        style: torch.Tensor,  # [B, D_style]
        **generation_kwargs,
    ):
        """
        Forward pass for S2Mel-based mel-spectrogram generation.
        """
        # S2Mel synthesis - codes to mel spectrogram
        # dtype = None
        diffusion_steps = 25
        inference_cfg_rate = 0.7
        latent = self.s2mel.models["gpt_layer"](latent)
        S_infer = self.semantic_codec.quantizer.vq2emb(codes.unsqueeze(1))
        S_infer = S_infer.transpose(1, 2)
        S_infer = S_infer + latent
        target_lengths = (code_lens * 1.72).long()

        cond = self.s2mel.models["length_regulator"](S_infer, ylens=target_lengths, n_quantizers=3, f0=None)[0]
        cat_condition = torch.cat([prompt_condition, cond], dim=1)
        s2mel_mel_spectrogram = self.s2mel.models["cfm"].inference(
            cat_condition,
            torch.LongTensor([cat_condition.size(1)]).to(cond.device),
            ref_mel,
            style,
            None,
            diffusion_steps,
            inference_cfg_rate=inference_cfg_rate,
        )

        s2mel_mel_spectrogram = s2mel_mel_spectrogram[:, :, ref_mel.size(-1) :]

        return s2mel_mel_spectrogram

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        s2mel_ckpt = getattr(self.config, "s2mel_checkpoint", None)
        repo_id = getattr(self.config, "repo_id")
        s2mel_path = hf_hub_download(repo_id, filename=s2mel_ckpt)

        self.s2mel, _, _, _ = load_checkpoint2(
            self.s2mel,
            None,
            s2mel_path,
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        mapper = WeightsMapper(
            orig_to_new_prefix={
                "gpt.": "",
                "s2mel.": "s2mel.",
            }
        )
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=mapper)
