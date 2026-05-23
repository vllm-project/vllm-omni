# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FunCineForge Code2Wav (Flow Matching + Causal HiFiGAN).

Adapted from CosyVoice3's code2wav pattern but using FunCineForge's
DiT-based flow matching and Causal HiFiGAN vocoder.

Weight layout (DeepSpeed):
  flow/mp_rank_00_model_states.pt    — DiT + codec_embedder + lookahead + xvec_proj
  vocoder/mp_rank_00_model_states.pt — CausalHiFTGenerator + F0Predictor
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger

from vllm_omni.model_executor.models.funcineforge.config import FunCineForgeConfig
from vllm_omni.model_executor.models.funcineforge.utils import load_deepspeed_checkpoint

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Lookahead convolution block (from FunCineForge's flow_matching_model.py)
# ---------------------------------------------------------------------------


class LookaheadBlock(nn.Module):
    """Causal lookahead convolution that peeks ``pre_lookahead_len`` frames ahead.

    Used by the flow matching model to give the DiT right-context before
    the causal mask cuts it off.
    """

    def __init__(self, in_channels: int, channels: int, pre_lookahead_len: int = 1):
        super().__init__()
        self.channels = channels
        self.pre_lookahead_len = pre_lookahead_len
        self.conv1 = nn.Conv1d(
            in_channels,
            channels,
            kernel_size=pre_lookahead_len + 1,
            stride=1,
            padding=0,
        )
        self.conv2 = nn.Conv1d(
            channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=0,
        )

    def forward(self, inputs: torch.Tensor, ilens: torch.Tensor, context: torch.Tensor = torch.zeros(0, 0, 0)):
        """Forward.

        Args:
            inputs: (B, T, C)
            ilens: (B,) lengths
            context: (B, C, lookahead_len) right-context from previous chunk
                     (used in streaming mode)
        Returns:
            outputs: (B, T, C)
            ilens: (B,) unchanged
        """
        outputs = inputs.transpose(1, 2).contiguous()
        context = context.transpose(1, 2).contiguous() if context.numel() > 0 else context

        # Look ahead: pad right with zeros or use cached context
        if context.numel() == 0 or context.size(2) == 0:
            outputs = F.pad(outputs, (0, self.pre_lookahead_len), mode="constant", value=0)
        else:
            assert context.size(2) == self.pre_lookahead_len
            outputs = torch.concat([outputs, context], dim=2)
        outputs = F.leaky_relu(self.conv1(outputs))

        # Causal pad left for conv2
        outputs = F.pad(outputs, (2, 0), mode="constant", value=0)
        outputs = self.conv2(outputs)
        outputs = outputs.transpose(1, 2).contiguous()

        # Build mask and residual
        max_len = inputs.shape[1]
        mask = torch.arange(max_len, device=inputs.device).unsqueeze(0) < ilens.unsqueeze(1)
        mask = mask.unsqueeze(-1).to(inputs.dtype)
        outputs = (outputs + inputs) * mask

        return outputs, ilens


class FunCineForgeCode2Wav(nn.Module):
    """FunCineForge code2wav: Flow Matching DiT → Causal HiFiGAN."""

    def __init__(self, config: FunCineForgeConfig):
        super().__init__()
        self.config = config
        flow_cfg = config.flow
        voc_cfg = config.vocoder

        # --- Flow Matching components ---
        from vllm_omni.model_executor.models.funcineforge.vendor.dit_model import DiT

        codebook_size = int(flow_cfg.get("codebook_size", config.codec_unit))
        num_mels = int(flow_cfg["mel_feat_conf"]["n_mel_channels"])
        model_size = int(flow_cfg.get("model_size", 512))
        xvec_size = int(flow_cfg.get("xvec_size", 198))
        lookahead_length = int(flow_cfg.get("lookahead_length", 4))

        self.input_embedding = nn.Embedding(codebook_size, num_mels)
        self.lookahead_conv1d = LookaheadBlock(num_mels, model_size, lookahead_length)
        self.xvec_proj = nn.Linear(xvec_size, num_mels) if xvec_size else None
        # spk_dim for DiT is the *projected* speaker dim (num_mels), not the
        # raw xvec_size, because xvec_proj maps (xvec_size → num_mels) before
        # the speaker embedding reaches DiT.  The config's dit_conf.spk_dim
        # stores this value (default 80).
        dit_spk_dim = int(flow_cfg["dit_conf"].get("spk_dim", num_mels))
        self.flow_model = DiT(
            dim=int(flow_cfg["dit_conf"]["dim"]),
            depth=int(flow_cfg["dit_conf"]["depth"]),
            heads=int(flow_cfg["dit_conf"]["heads"]),
            dim_head=int(flow_cfg["dit_conf"]["dim_head"]),
            dropout=float(flow_cfg["dit_conf"].get("dropout", 0.1)),
            ff_mult=int(flow_cfg["dit_conf"].get("ff_mult", 4)),
            mel_dim=num_mels,
            spk_dim=dit_spk_dim if dit_spk_dim else None,
            causal_mask_type=flow_cfg["dit_conf"].get("causal_mask_type"),
        )

        # --- Vocoder components ---
        from vllm_omni.model_executor.models.funcineforge.vendor.causal_hifigan import (
            CausalConvRNNF0Predictor,
            CausalHiFTGenerator,
        )

        self.vocoder = nn.Module()
        self.vocoder.generator = CausalHiFTGenerator(**voc_cfg["CausalHiFTGenerator_conf"])
        self.vocoder.generator.f0_predictor = CausalConvRNNF0Predictor(**voc_cfg["CausalConvRNNF0Predictor_conf"])
        self.vocoder.generator.remove_weight_norm()
        self.vocoder.sample_rate = int(voc_cfg.get("sample_rate", 24000))

        # Flow matching parameters
        self.feat_token_ratio = int(flow_cfg.get("feat_token_ratio", 2))
        self.context_size = self.lookahead_conv1d.pre_lookahead_len
        self.rand_noise_max_len = int(flow_cfg.get("rand_noise_max_len", 15000))
        self.rand_noise: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Forward (non-streaming)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def forward(
        self,
        token: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
        n_timesteps: int = 10,
    ) -> torch.Tensor:
        """Generate audio waveform from speech tokens (non-streaming)."""
        feat = self._forward_mel(
            token=token,
            prompt_token=prompt_token,
            prompt_feat=prompt_feat,
            embedding=embedding,
            n_timesteps=n_timesteps,
            token_offset_tokens=0,
            streaming=False,
            finalize=True,
        )

        # Run vocoder
        voc_device = next(self.vocoder.parameters()).device
        tts_mel = feat.to(device=voc_device, dtype=torch.float32)
        if tts_mel.shape[-1] == 0:
            return torch.zeros((tts_mel.shape[0], 1, 0), device=voc_device, dtype=torch.float32)
        tts_speech, _ = self.vocoder.generator.inference(speech_feat=tts_mel, finalize=True)
        return tts_speech

    # ------------------------------------------------------------------
    # Forward (streaming)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def forward_streaming(
        self,
        token: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
        cache_state: dict[str, Any] | None = None,
        n_timesteps: int = 10,
        token_offset_tokens: int = 0,
        finalize: bool = True,
    ) -> tuple[torch.Tensor, dict[str, Any] | None]:
        """Generate audio waveform from speech tokens (streaming)."""
        feat = self._forward_mel(
            token=token,
            prompt_token=prompt_token,
            prompt_feat=prompt_feat,
            embedding=embedding,
            n_timesteps=n_timesteps,
            token_offset_tokens=token_offset_tokens,
            streaming=True,
            finalize=finalize,
        )

        voc_device = next(self.vocoder.parameters()).device
        chunk_mel = feat.to(device=voc_device, dtype=torch.float32)

        cached_mel = None if not cache_state else cache_state.get("mel")
        speech_offset_obj = None if not cache_state else cache_state.get("speech_offset")
        try:
            speech_offset = int(speech_offset_obj) if speech_offset_obj is not None else 0
        except (TypeError, ValueError):
            speech_offset = 0

        if isinstance(cached_mel, torch.Tensor) and cached_mel.numel() > 0:
            cached_mel = cached_mel.to(device=chunk_mel.device, dtype=chunk_mel.dtype)
            tts_mel = torch.cat([cached_mel, chunk_mel], dim=-1) if chunk_mel.numel() > 0 else cached_mel
        else:
            tts_mel = chunk_mel

        if tts_mel.shape[-1] == 0:
            tts_speech = torch.zeros((chunk_mel.shape[0], 1, 0), device=chunk_mel.device, dtype=chunk_mel.dtype)
        else:
            with nullcontext():
                tts_speech, _ = self.vocoder.generator.inference(speech_feat=tts_mel, finalize=finalize)

        tts_speech = tts_speech.reshape(tts_speech.shape[0], -1)
        speech_offset = max(0, min(speech_offset, int(tts_speech.shape[-1])))
        emitted_speech = tts_speech[:, speech_offset:]

        if finalize:
            return emitted_speech.reshape(emitted_speech.shape[0], 1, -1), None

        new_state = {
            "mel": tts_mel.detach().cpu().contiguous(),
            "speech_offset": int(tts_speech.shape[-1]),
        }
        return emitted_speech.reshape(emitted_speech.shape[0], 1, -1), new_state

    # ------------------------------------------------------------------
    # Mel generation via flow matching
    # ------------------------------------------------------------------

    def _forward_mel(
        self,
        token: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
        n_timesteps: int = 10,
        token_offset_tokens: int = 0,
        streaming: bool = False,
        finalize: bool = True,
    ) -> torch.Tensor:
        """Run flow matching ODE to generate mel features from codec tokens.

        Implements the full CFM solver adapted from FunCineForge's
        ``CosyVoiceFlowMatching._inference`` and ``solve_euler``.

        Pipeline:
          1. Embed codec tokens via ``input_embedding`` (codec_embedder).
          2. Run lookahead convolution for right-context.
          3. Upsample by ``feat_token_ratio`` (2× for 50Hz mel / 25Hz codec).
          4. Build conditions from prompt mel features.
          5. Project speaker embedding via ``xvec_proj``.
          6. Solve ODE with Euler method + classifier-free guidance.
        """
        cfg = self.config
        flow_cfg = cfg.flow
        num_mels = int(flow_cfg["mel_feat_conf"]["n_mel_channels"])
        inference_cfg_rate = float(flow_cfg.get("inference_cfg_rate", 0.7))
        temperature = float(flow_cfg.get("temperature", 1.0))
        infer_t_scheduler = flow_cfg.get("infer_t_scheduler", "cosine")
        codebook_size = int(flow_cfg.get("codebook_size", cfg.codec_unit))

        B = token.shape[0]
        device = token.device

        # Ensure all inputs are on the same device (prompt tensors may arrive on CPU
        # from additional_information / stage_input_processors)
        if prompt_token is not None and prompt_token.device != device:
            prompt_token = prompt_token.to(device)
        if prompt_feat is not None and prompt_feat.device != device:
            prompt_feat = prompt_feat.to(device)
        if embedding is not None and embedding.device != device:
            embedding = embedding.to(device)

        # 1. Filter out-of-range tokens and embed
        if (token >= codebook_size).any():
            token = token[token < codebook_size].unsqueeze(0).expand(B, -1)
        codec_lengths = torch.tensor([token.shape[1]], dtype=torch.int64, device=device).expand(B)

        # Official FunCineForge does not pass the reference speech tokens as
        # prompt_codec by default; the reference audio is used for xvec only.
        use_prompt_codec = bool(flow_cfg.get("use_prompt_codec", False))
        if use_prompt_codec and prompt_token is not None and prompt_token.numel() > 0:
            prompt_codec = prompt_token.clone()
            if (prompt_codec >= codebook_size).any():
                prompt_codec = prompt_codec.clamp(max=codebook_size - 1)
            token = torch.cat([prompt_codec, token], dim=1)
            codec_lengths = torch.tensor([token.shape[1]], dtype=torch.int64, device=device).expand(B)

        mask = (token != -1).float().unsqueeze(-1)  # (B, T, 1)
        codec_emb = self.input_embedding(token.clamp(min=0)) * mask

        # 2. Lookahead convolution
        fm_dtype = torch.float32
        self.lookahead_conv1d.to(fm_dtype)
        if finalize:
            context = torch.zeros(B, 0, num_mels, device=device, dtype=fm_dtype)
        else:
            # Streaming: reserve right-context for next chunk
            context = codec_emb[:, -self.context_size :].to(fm_dtype)
            codec_emb = codec_emb[:, : -self.context_size].to(fm_dtype)
            codec_lengths = codec_lengths - self.context_size

        mu, _ = self.lookahead_conv1d(codec_emb.to(fm_dtype), codec_lengths, context)

        # 3. Upsample by feat_token_ratio
        mu = mu.repeat_interleave(self.feat_token_ratio, dim=1)

        # 4. Build conditions from prompt mel features
        conditions = torch.zeros(B, mu.shape[1], num_mels, device=device, dtype=fm_dtype)
        use_prompt_feat = bool(flow_cfg.get("use_prompt_feat", False))
        if use_prompt_feat and prompt_feat is not None and prompt_feat.numel() > 0:
            prompt_feat = prompt_feat.to(device=device, dtype=fm_dtype)
            # prompt_feat shape: (1, n_mels, T) → (1, T, n_mels)
            if prompt_feat.dim() == 3 and prompt_feat.shape[1] == num_mels:
                prompt_feat_t = prompt_feat.transpose(1, 2)
            else:
                prompt_feat_t = prompt_feat
            cond_len = min(prompt_feat_t.shape[1], conditions.shape[1])
            conditions[:, :cond_len] = prompt_feat_t[:, :cond_len]

        # 5. Speaker embedding projection
        rand_xvec = None
        if embedding is not None and embedding.numel() > 0 and self.xvec_proj is not None:
            xvec = embedding.to(device=device, dtype=fm_dtype)
            if xvec.dim() == 2:
                xvec = xvec.unsqueeze(1)  # (B, 1, xvec_dim)
            # Normalize speaker embedding
            xvec_mask = (~xvec.norm(dim=-1).isnan()) & (~xvec.norm(dim=-1).isinf())
            xvec = xvec * xvec_mask.unsqueeze(-1).float()
            xvec = xvec.mean(dim=1)
            xvec = F.normalize(xvec, dim=1)
            self.xvec_proj.to(fm_dtype)
            rand_xvec = self.xvec_proj(xvec.to(fm_dtype))
            rand_xvec = rand_xvec.unsqueeze(1)  # (B, 1, num_mels)

        # 6. Build attention mask for DiT
        # DiT.forward expects mask as (B, 1, T) — it does mask.unsqueeze(1)
        # internally to make it (B, 1, 1, T) for attention.
        feat_lengths = codec_lengths * self.feat_token_ratio
        max_feat_len = mu.shape[1]
        attn_mask = torch.arange(max_feat_len, device=device).unsqueeze(0) < feat_lengths.unsqueeze(1)
        attn_mask = attn_mask.unsqueeze(1)  # (B, 1, T)

        # 7. Solve ODE
        feat = self._solve_ode(
            mu=mu.to(fm_dtype),
            spks=rand_xvec,
            conditions=conditions,
            mask=attn_mask,
            n_timesteps=n_timesteps,
            temperature=temperature,
            inference_cfg_rate=inference_cfg_rate,
            infer_t_scheduler=infer_t_scheduler,
            fm_dtype=fm_dtype,
            infer_causal_mask_type=int(flow_cfg.get("infer_causal_mask_type", 0)),
        )

        # Remove prompt portion if prompt_codec was concatenated
        if use_prompt_codec and prompt_token is not None and prompt_token.numel() > 0 and prompt_feat is not None:
            prompt_len = prompt_token.shape[1] * self.feat_token_ratio
            feat = feat[:, prompt_len:]

        trim_mel = max(0, int(token_offset_tokens)) * int(self.feat_token_ratio)
        if trim_mel > 0:
            feat = feat[:, trim_mel:]

        # Transpose to (B, n_mels, T) for vocoder
        feat = feat.transpose(1, 2)
        return feat

    # ------------------------------------------------------------------
    # ODE solver
    # ------------------------------------------------------------------

    def _solve_ode(
        self,
        mu: torch.Tensor,
        spks: torch.Tensor | None,
        conditions: torch.Tensor,
        mask: torch.Tensor,
        n_timesteps: int = 10,
        temperature: float = 1.0,
        inference_cfg_rate: float = 0.7,
        infer_t_scheduler: str = "cosine",
        fm_dtype: torch.dtype = torch.float32,
        infer_causal_mask_type: int = 0,
    ) -> torch.Tensor:
        """Solve the CFM ODE using Euler method with classifier-free guidance."""
        z = self._get_rand_noise(mu) * temperature

        # Time schedule
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=fm_dtype)
        if infer_t_scheduler == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

        # Euler integration
        t = t_span[0]
        dt = t_span[1] - t_span[0]
        x = z
        bz = x.shape[0]

        for step in range(1, len(t_span)):
            # Classifier-free guidance: double batch
            if inference_cfg_rate > 0:
                x_in = torch.cat([x, x], dim=0)
                spks_in = torch.cat([spks, torch.zeros_like(spks)], dim=0) if spks is not None else None
                mask_in = torch.cat([mask, mask], dim=0)
                mu_in = torch.cat([mu, torch.zeros_like(mu)], dim=0)
                t_in = t.unsqueeze(0).expand(x_in.shape[0]).to(fm_dtype)
                cond_in = torch.cat([conditions, torch.zeros_like(conditions)], dim=0)
            else:
                x_in, mask_in, mu_in = x, mask, mu
                spks_in = spks
                t_in = t
                cond_in = conditions

            # DiT forward
            self.flow_model.to(fm_dtype)
            mask_rand = None
            causal_mask_type = getattr(self.flow_model, "causal_mask_type", None)
            if causal_mask_type is not None:
                if len(causal_mask_type) == 0:
                    raise ValueError("flow_model.causal_mask_type is empty")
                mask_idx = max(0, min(int(infer_causal_mask_type), len(causal_mask_type) - 1))
                chunk_mask_value = causal_mask_type[mask_idx]["prob_min"]
                mask_rand = torch.ones_like(t_in).reshape(-1, 1, 1) * chunk_mask_value
            dphi_dt = self.flow_model(
                x_in,
                cond_in,
                mu_in,
                spks_in.squeeze(1) if spks_in is not None else None,
                t_in,
                mask=mask_in,
                mask_rand=mask_rand,
            )

            # Apply CFG
            if inference_cfg_rate > 0:
                dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [bz, bz], dim=0)
                dphi_dt = (1.0 + inference_cfg_rate) * dphi_dt - inference_cfg_rate * cfg_dphi_dt

            x = x + dt * dphi_dt
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return x

    def _get_rand_noise(self, mu: torch.Tensor) -> torch.Tensor:
        """Return the fixed inference noise buffer used by upstream FunCineForge."""
        if self.rand_noise is None or self.rand_noise.shape[1] < mu.shape[1] or self.rand_noise.shape[2] != mu.shape[2]:
            self.rand_noise = torch.randn(
                (1, self.rand_noise_max_len, mu.shape[2]),
                device=mu.device,
                dtype=mu.dtype,
            )
            logger.info("Initialized fixed random noise for FunCineForge flow")
        noise = self.rand_noise.to(device=mu.device, dtype=mu.dtype)[:, : mu.shape[1], :]
        return torch.cat([noise for _ in range(mu.shape[0])], dim=0)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, model_dir: str, device: torch.device) -> None:
        """Load flow and vocoder weights from DeepSpeed checkpoints.

        The original FunCineForge ``CosyVoiceFlowMatching`` class stores all
        flow-matching components in a single checkpoint with these prefixes:
          - ``codec_embedder.*``   → self.input_embedding
          - ``lookahead_conv1d.*`` → self.lookahead_conv1d
          - ``xvec_proj.*``        → self.xvec_proj
          - ``dit_model.*``        → self.flow_model (DiT)
        """
        flow_path = os.path.join(model_dir, self.config.flow_ckpt)
        if os.path.exists(flow_path):
            flow_state = load_deepspeed_checkpoint(flow_path, device)

            # 1. codec_embedder → input_embedding
            emb_state = {
                k.replace("codec_embedder.", ""): v for k, v in flow_state.items() if k.startswith("codec_embedder.")
            }
            if emb_state and hasattr(self, "input_embedding"):
                self.input_embedding.load_state_dict(emb_state)
                logger.info("Loaded codec_embedder → input_embedding (%d keys)", len(emb_state))

            # 2. lookahead_conv1d → lookahead_conv1d
            la_state = {
                k.replace("lookahead_conv1d.", ""): v
                for k, v in flow_state.items()
                if k.startswith("lookahead_conv1d.")
            }
            if la_state:
                self.lookahead_conv1d.load_state_dict(la_state)
                logger.info("Loaded lookahead_conv1d (%d keys)", len(la_state))

            # 3. xvec_proj → xvec_proj
            if self.xvec_proj is not None:
                xvec_state = {
                    k.replace("xvec_proj.", ""): v for k, v in flow_state.items() if k.startswith("xvec_proj.")
                }
                if xvec_state:
                    self.xvec_proj.load_state_dict(xvec_state)
                    logger.info("Loaded xvec_proj (%d keys)", len(xvec_state))

            # 4. dit_model → flow_model (DiT)
            dit_state = {k.replace("dit_model.", ""): v for k, v in flow_state.items() if k.startswith("dit_model.")}
            if dit_state:
                self.flow_model.load_state_dict(dit_state, strict=False)
                logger.info("Loaded dit_model → flow_model (%d keys)", len(dit_state))
            else:
                self.flow_model.load_state_dict(flow_state, strict=False)
                logger.warning("No dit_model.* keys found; attempted direct load into DiT")

            self.flow_model.to(device).eval()
            self.input_embedding.to(device)
            self.lookahead_conv1d.to(device)
            if self.xvec_proj is not None:
                self.xvec_proj.to(device)
            logger.info("Loaded flow weights from %s", flow_path)
        else:
            logger.warning("Flow checkpoint not found: %s", flow_path)

        # Load vocoder weights
        voc_path = os.path.join(model_dir, self.config.vocoder_ckpt)
        if os.path.exists(voc_path):
            voc_state = load_deepspeed_checkpoint(voc_path, device)
            gen_state = {k.replace("generator.", ""): v for k, v in voc_state.items() if k.startswith("generator.")}
            if gen_state:
                self.vocoder.generator.load_state_dict(gen_state, strict=False)
                logger.info("Loaded vocoder generator (%d keys)", len(gen_state))
            else:
                self.vocoder.generator.load_state_dict(voc_state, strict=False)
                logger.warning("No generator.* prefix in vocoder ckpt; attempted direct load")
            self.vocoder.to(device).eval()
            logger.info("Loaded vocoder weights from %s", voc_path)
        else:
            logger.warning("Vocoder checkpoint not found: %s", voc_path)
