# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""
CosyVoice3 Code2Wav Stage - Converts speech tokens to audio waveforms.

This module contains the code2wav (token-to-waveform) stage which uses:
1. DiT (Diffusion Transformer) with optimized attention backends
2. CFM (Conditional Flow Matching) for mel spectrogram generation
3. HiFiGAN vocoder for waveform synthesis
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from vllm.logger import init_logger

from vllm_omni.diffusion.models.cosyvoice3_audio.cosyvoice3_dit import DiT
from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.cfm import (
    CausalConditionalCFM,
    CausalMaskedDiffWithDiT,
)
from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.hifigan import (
    CausalConvRNNF0Predictor,
    CausalHiFTGenerator,
)
from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.layers import PreLookaheadLayer
from vllm_omni.transformers_utils.configs.cosyvoice3 import CosyVoice3Config

logger = init_logger(__name__)


@dataclass
class CosyVoice3FlowState:
    """Per-request state retained between Flow scheduling ticks."""

    x: torch.Tensor
    mu: torch.Tensor
    mask: torch.Tensor
    spks: torch.Tensor
    cond: torch.Tensor
    t_span: torch.Tensor
    step_index: int
    prompt_mel_len: int
    generated_mel_len: int
    trim_mel: int


class CosyVoice3Code2Wav(nn.Module):
    """CosyVoice3 Code2Wav stage for token-to-waveform conversion.

    This class encapsulates:
    - Flow matching decoder with DiT backbone (using diffusion attention)
    - HiFiGAN vocoder for mel-to-waveform conversion
    """

    def __init__(self, config: CosyVoice3Config):
        super().__init__()
        self.config = config

        # Build flow matching components
        pre_lookahead_layer = PreLookaheadLayer(**config.flow["pre_lookahead_layer"])

        decoder_cfg = config.flow["decoder"]
        cfm_params = DictConfig(decoder_cfg["cfm_params"])

        # DiT estimator using diffusion attention (Flash/Sage/SDPA backends)
        estimator = DiT(**decoder_cfg["estimator"])

        decoder = CausalConditionalCFM(
            in_channels=decoder_cfg["in_channels"],
            estimator=estimator,
            cfm_params=cfm_params,
            n_spks=decoder_cfg["n_spks"],
            spk_emb_dim=decoder_cfg["spk_emb_dim"],
        )

        self.flow_model = CausalMaskedDiffWithDiT(
            input_size=config.flow["input_size"],
            output_size=config.flow["output_size"],
            spk_embed_dim=config.flow["spk_embed_dim"],
            output_type=config.flow["output_type"],
            vocab_size=config.flow["vocab_size"],
            input_frame_rate=config.flow["input_frame_rate"],
            only_mask_loss=config.flow["only_mask_loss"],
            token_mel_ratio=config.flow["token_mel_ratio"],
            pre_lookahead_len=config.flow["pre_lookahead_len"],
            pre_lookahead_layer=pre_lookahead_layer,
            decoder=decoder,
        )

        # Build HiFiGAN vocoder
        f0_predictor = CausalConvRNNF0Predictor(
            num_class=config.hift["f0_predictor"]["num_class"],
            in_channels=config.hift["f0_predictor"]["in_channels"],
            cond_channels=config.hift["f0_predictor"]["cond_channels"],
        )

        self.hift = CausalHiFTGenerator(
            in_channels=config.hift["in_channels"],
            base_channels=config.hift["base_channels"],
            nb_harmonics=config.hift["nb_harmonics"],
            sampling_rate=config.hift["sampling_rate"],
            nsf_alpha=config.hift["nsf_alpha"],
            nsf_sigma=config.hift["nsf_sigma"],
            nsf_voiced_threshold=config.hift["nsf_voiced_threshold"],
            upsample_rates=config.hift["upsample_rates"],
            upsample_kernel_sizes=config.hift["upsample_kernel_sizes"],
            istft_params=config.hift["istft_params"],
            resblock_kernel_sizes=config.hift["resblock_kernel_sizes"],
            resblock_dilation_sizes=config.hift["resblock_dilation_sizes"],
            source_resblock_kernel_sizes=config.hift["source_resblock_kernel_sizes"],
            source_resblock_dilation_sizes=config.hift["source_resblock_dilation_sizes"],
            lrelu_slope=config.hift["lrelu_slope"],
            audio_limit=config.hift["audio_limit"],
            conv_pre_look_right=config.hift["conv_pre_look_right"],
            f0_predictor=f0_predictor,
        )
        # Run hift in float32 to avoid dtype mismatches in internal ops
        self.hift = self.hift.float()

        # Streaming/chunking parameters
        self.token_overlap_len = 20
        self.mel_overlap_len = int(self.token_overlap_len / self.flow_model.input_frame_rate * 22050 / 256)
        self.mel_window = np.hamming(2 * self.mel_overlap_len)
        self.mel_cache_len = 20
        samples_per_mel = int(np.prod(config.hift["upsample_rates"]) * config.hift["istft_params"]["hop_len"])
        self.source_cache_len = self.mel_cache_len * samples_per_mel
        # Non-final causal HiFT output excludes the F0 and conv-pre right
        # context, plus one ISTFT frame. Hold back only the waveform that is
        # actually regenerated by ``mel_cache_len`` frames on the next call.
        f0_lookahead = int(self.hift.f0_predictor.condnet[0].causal_padding)
        waveform_lookahead = f0_lookahead + int(self.hift.conv_pre_look_right) + 1
        self.speech_cache_len = max(0, self.mel_cache_len - waveform_lookahead) * samples_per_mel
        self.register_buffer(
            "speech_window",
            torch.from_numpy(np.hamming(2 * self.speech_cache_len).astype(np.float32)),
            persistent=False,
        )

    @property
    def input_frame_rate(self) -> int:
        """Input frame rate from flow model."""
        return self.flow_model.input_frame_rate

    @property
    def token_mel_ratio(self) -> int:
        """Token to mel ratio."""
        return self.flow_model.token_mel_ratio

    @property
    def output_size(self) -> int:
        """Output mel dimension."""
        return self.flow_model.output_size

    @property
    def input_embedding(self) -> nn.Embedding:
        """Token embedding layer."""
        return self.flow_model.input_embedding

    @property
    def pre_lookahead_layer(self) -> nn.Module:
        """Pre-lookahead layer."""
        return self.flow_model.pre_lookahead_layer

    @property
    def decoder(self) -> nn.Module:
        """Flow matching decoder."""
        return self.flow_model.decoder

    @property
    def spk_embed_affine_layer(self) -> nn.Linear:
        """Speaker embedding affine layer."""
        return self.flow_model.spk_embed_affine_layer

    @torch.inference_mode()
    def _forward_mel(
        self,
        token: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
        n_timesteps: int = 10,
        token_offset_tokens: int = 0,
        streaming: bool = True,
        finalize: bool = False,
    ) -> torch.Tensor:
        """Generate mel features via the upstream flow-model inference path."""
        flow_weight = next(self.flow_model.parameters())
        device = flow_weight.device
        dtype = flow_weight.dtype

        token = token.to(device=device, dtype=torch.int32)
        prompt_token = prompt_token.to(device=device, dtype=torch.int32)
        prompt_feat = prompt_feat.to(device=device, dtype=dtype)
        embedding = embedding.to(device=device, dtype=dtype)
        token_len = torch.tensor([token.shape[1]], device=device, dtype=torch.int32)
        prompt_token_len = torch.tensor([prompt_token.shape[1]], device=device, dtype=torch.int32)
        prompt_feat_len = torch.tensor([prompt_feat.shape[1]], device=device, dtype=torch.int32)

        feat, _ = self.flow_model.inference(
            token=token,
            token_len=token_len,
            prompt_token=prompt_token,
            prompt_token_len=prompt_token_len,
            prompt_feat=prompt_feat,
            prompt_feat_len=prompt_feat_len,
            embedding=embedding,
            streaming=streaming,
            finalize=finalize,
            n_timesteps=n_timesteps,
        )

        trim_mel = max(0, int(token_offset_tokens)) * int(self.token_mel_ratio)
        if trim_mel > 0:
            feat = feat[:, :, trim_mel:]

        return feat

    @torch.inference_mode()
    def prepare_flow_state(
        self,
        token: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
        *,
        n_timesteps: int = 10,
        token_offset_tokens: int = 0,
        finalize: bool = False,
    ) -> CosyVoice3FlowState:
        """Prepare one request for step-wise padded Flow execution."""
        flow_weight = next(self.flow_model.parameters())
        device = flow_weight.device
        dtype = flow_weight.dtype

        token = token.to(device=device, dtype=torch.int32)
        prompt_token = prompt_token.to(device=device, dtype=torch.int32)
        prompt_feat = prompt_feat.to(device=device, dtype=dtype)
        embedding = embedding.to(device=device, dtype=dtype)
        token_len = torch.tensor([token.shape[1]], device=device, dtype=torch.int32)
        prompt_token_len = torch.tensor([prompt_token.shape[1]], device=device, dtype=torch.int32)
        prompt_feat_len = torch.tensor([prompt_feat.shape[1]], device=device, dtype=torch.int32)

        decoder_inputs = self.flow_model.prepare_inference_inputs(
            token=token,
            token_len=token_len,
            prompt_token=prompt_token,
            prompt_token_len=prompt_token_len,
            prompt_feat=prompt_feat,
            prompt_feat_len=prompt_feat_len,
            embedding=embedding,
            finalize=finalize,
        )
        mu = decoder_inputs["mu"]
        assert isinstance(mu, torch.Tensor)
        num_steps = max(1, int(n_timesteps))
        t_span = torch.linspace(0, 1, num_steps + 1, device=mu.device, dtype=mu.dtype)
        if self.decoder.t_scheduler == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

        mask = decoder_inputs["mask"]
        spks = decoder_inputs["spks"]
        cond = decoder_inputs["cond"]
        assert isinstance(mask, torch.Tensor)
        assert isinstance(spks, torch.Tensor)
        assert isinstance(cond, torch.Tensor)
        return CosyVoice3FlowState(
            x=torch.randn_like(mu),
            mu=mu,
            mask=mask,
            spks=spks,
            cond=cond,
            t_span=t_span,
            step_index=0,
            prompt_mel_len=int(decoder_inputs["prompt_mel_len"]),
            generated_mel_len=int(decoder_inputs["generated_mel_len"]),
            trim_mel=max(0, int(token_offset_tokens)) * int(self.token_mel_ratio),
        )

    @torch.inference_mode()
    def forward_flow_step(self, states: list[CosyVoice3FlowState]) -> list[bool]:
        """Advance each state once using one padded DiT batch."""
        if not states:
            return []

        channels = states[0].x.shape[1]
        max_len = max(state.x.shape[2] for state in states)
        batch_size = len(states)
        device = states[0].x.device
        dtype = states[0].x.dtype

        def padded(channels_: int) -> torch.Tensor:
            return torch.zeros((batch_size, channels_, max_len), device=device, dtype=dtype)

        x = padded(channels)
        mu = padded(states[0].mu.shape[1])
        cond = padded(states[0].cond.shape[1])
        mask = torch.zeros((batch_size, 1, max_len), device=device, dtype=dtype)
        for row, state in enumerate(states):
            seq_len = state.x.shape[2]
            x[row, :, :seq_len] = state.x[0]
            mu[row, :, :seq_len] = state.mu[0]
            cond[row, :, :seq_len] = state.cond[0]
            mask[row, :, :seq_len] = state.mask[0]

        spks = torch.cat([state.spks for state in states], dim=0)
        t = torch.stack([state.t_span[state.step_index] for state in states])
        next_t = torch.stack([state.t_span[state.step_index + 1] for state in states])
        x = self.decoder.solve_euler_step(
            x=x,
            t=t,
            dt=next_t - t,
            mu=mu,
            mask=mask,
            spks=spks,
            cond=cond,
        )

        completed: list[bool] = []
        for row, state in enumerate(states):
            seq_len = state.x.shape[2]
            state.x = x[row : row + 1, :, :seq_len].contiguous()
            state.step_index += 1
            completed.append(state.step_index == len(state.t_span) - 1)
        return completed

    @staticmethod
    def flow_state_to_mel(state: CosyVoice3FlowState) -> torch.Tensor:
        """Extract the generated (non-prompt) mel from a completed state."""
        start = state.prompt_mel_len
        end = start + state.generated_mel_len
        feat = state.x.float()[:, :, start:end]
        if state.trim_mel > 0:
            feat = feat[:, :, state.trim_mel :]
        return feat

    @torch.inference_mode()
    def decode_streaming_mel(
        self,
        feat: torch.Tensor,
        *,
        cache_state: dict[str, torch.Tensor] | None = None,
        finalize: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        """Decode one Flow mel chunk with bounded, device-resident HiFT state."""
        hift_weight = self.hift.m_source.l_linear.weight
        chunk_mel = feat.to(device=hift_weight.device, dtype=hift_weight.dtype)

        cached_mel = None if not cache_state else cache_state.get("mel")
        cached_source = None if not cache_state else cache_state.get("source")
        cached_speech = None if not cache_state else cache_state.get("speech")

        if isinstance(cached_mel, torch.Tensor) and cached_mel.numel() > 0:
            cached_mel = cached_mel.to(device=chunk_mel.device, dtype=chunk_mel.dtype)
            tts_mel = torch.cat([cached_mel, chunk_mel], dim=-1) if chunk_mel.numel() > 0 else cached_mel
        else:
            tts_mel = chunk_mel

        if tts_mel.shape[-1] == 0:
            tts_speech = torch.zeros((chunk_mel.shape[0], 1, 0), device=chunk_mel.device, dtype=chunk_mel.dtype)
            source = torch.zeros_like(tts_speech)
        else:
            tts_speech, source = self.hift.inference(
                speech_feat=tts_mel,
                finalize=finalize,
                cache_source=cached_source if isinstance(cached_source, torch.Tensor) else None,
            )

        tts_speech = tts_speech.reshape(tts_speech.shape[0], -1)
        if isinstance(cached_speech, torch.Tensor) and cached_speech.numel() > 0 and tts_speech.numel() > 0:
            cached_speech = cached_speech.to(device=tts_speech.device, dtype=tts_speech.dtype).reshape(
                tts_speech.shape[0], -1
            )
            overlap = min(
                self.speech_cache_len,
                int(cached_speech.shape[-1]),
                int(tts_speech.shape[-1]),
            )
            if overlap > 0:
                if overlap == self.speech_cache_len:
                    window = self.speech_window.to(device=tts_speech.device, dtype=tts_speech.dtype)
                else:
                    window = torch.hamming_window(
                        2 * overlap,
                        periodic=False,
                        device=tts_speech.device,
                        dtype=tts_speech.dtype,
                    )
                # Preserve HiFT's returned waveform because only the overlap
                # prefix belongs to this cache-aware cross-fade.
                tts_speech = tts_speech.clone()
                tts_speech[:, :overlap] = (
                    tts_speech[:, :overlap] * window[:overlap] + cached_speech[:, -overlap:] * window[overlap:]
                )

        if finalize:
            return tts_speech.reshape(tts_speech.shape[0], 1, -1), None

        holdback = min(self.speech_cache_len, int(tts_speech.shape[-1]))
        emitted_speech = tts_speech[:, :-holdback] if holdback > 0 else tts_speech

        new_state = {
            "mel": tts_mel[:, :, -self.mel_cache_len :].detach().contiguous(),
            "source": source[:, :, -self.source_cache_len :].detach().contiguous(),
            "speech": tts_speech[:, -holdback:].detach().contiguous() if holdback > 0 else tts_speech.detach(),
        }
        return emitted_speech.reshape(emitted_speech.shape[0], 1, -1), new_state

    @torch.inference_mode()
    def decode_mel(self, feat: torch.Tensor) -> torch.Tensor:
        """Decode a completed non-streaming Flow mel."""
        hift_weight = self.hift.m_source.l_linear.weight
        tts_mel = feat.to(device=hift_weight.device, dtype=hift_weight.dtype)
        if tts_mel.shape[-1] == 0:
            return torch.zeros(
                (tts_mel.shape[0], 1, 0),
                device=tts_mel.device,
                dtype=tts_mel.dtype,
            )
        tts_speech, _ = self.hift.inference(speech_feat=tts_mel, finalize=True)
        return tts_speech

    @torch.inference_mode()
    def forward_streaming(
        self,
        token: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
        *,
        cache_state: dict[str, torch.Tensor] | None = None,
        n_timesteps: int = 10,
        token_offset_tokens: int = 0,
        finalize: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        """Decode streaming audio with bounded HiFT mel/source/speech caches."""
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
        return self.decode_streaming_mel(feat, cache_state=cache_state, finalize=finalize)

    @torch.inference_mode()
    def forward(
        self,
        token: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
        n_timesteps: int = 10,
        token_offset_tokens: int = 0,
    ) -> torch.Tensor:
        """Generate audio waveform from speech tokens."""
        feat = self._forward_mel(
            token=token,
            prompt_token=prompt_token,
            prompt_feat=prompt_feat,
            embedding=embedding,
            n_timesteps=n_timesteps,
            token_offset_tokens=token_offset_tokens,
            streaming=False,
            finalize=True,
        )

        return self.decode_mel(feat)

    def load_weights(self, model_dir: str, device: torch.device) -> None:
        """Load flow.pt and hift.pt weights.

        Args:
            model_dir: Model directory containing flow.pt and hift.pt
            device: Device to load weights to
        """
        import os

        # Load flow weights
        flow_path = os.path.join(model_dir, "flow.pt")
        self.flow_model.load_state_dict(torch.load(flow_path, map_location=device), strict=True)
        self.flow_model.to(device).eval()
        logger.info(f"Loaded flow weights from {flow_path}")

        # Load hift weights
        hift_path = os.path.join(model_dir, "hift.pt")
        hift_state_dict = {
            k.replace("generator.", ""): v for k, v in torch.load(hift_path, map_location=device).items()
        }
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(device).eval()
        logger.info(f"Loaded hift weights from {hift_path}")
