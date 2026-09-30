# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""
CosyVoice3 Code2Wav Stage - Converts speech tokens to audio waveforms.

This module contains the code2wav (token-to-waveform) stage which uses:
1. DiT (Diffusion Transformer) with optimized attention backends
2. CFM (Conditional Flow Matching) for mel spectrogram generation
3. HiFiGAN vocoder for waveform synthesis
"""

import os
from collections import Counter
from collections.abc import Mapping
from typing import Any, Literal, TypedDict, cast, overload

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from vllm.logger import init_logger

from vllm_omni.diffusion.config import get_current_diffusion_config_or_none, set_current_diffusion_config
from vllm_omni.diffusion.data import OmniDiffusionConfig
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
from vllm_omni.model_executor.models.cosyvoice3.runtime import (
    cosyvoice3_batch_flow_debug,
    cosyvoice3_batch_flow_profile,
    cosyvoice3_full_response_enabled,
    cosyvoice3_packed_inference_enabled,
    cosyvoice3_packed_streaming_enabled,
)
from vllm_omni.transformers_utils.configs.cosyvoice3 import CosyVoice3Config

logger = init_logger(__name__)


def _build_dit_estimator(estimator_config: Mapping[str, Any]) -> DiT:
    """Build CosyVoice's embedded DiT with diffusion backend configuration.

    The code2wav model runs inside an LLM-generation worker, so it does not go
    through the diffusion model loader that normally installs an
    ``OmniDiffusionConfig`` during model construction. Install a local default
    config when needed so ``DIFFUSION_ATTENTION_BACKEND`` and the platform
    default are resolved by the same path as standalone diffusion models.
    """
    if get_current_diffusion_config_or_none() is not None:
        return DiT(**estimator_config)

    diffusion_config = OmniDiffusionConfig()
    with set_current_diffusion_config(diffusion_config):
        return DiT(**estimator_config)


class StreamingHiFTState(TypedDict, total=False):
    """Request-owned mel history, its absolute offset, and oscillator phase."""

    mel: torch.Tensor
    mel_offset: int
    phase_acc: torch.Tensor | None


class StreamingFlowItem(TypedDict, total=False):
    """One entry of the batched-streaming item list passed to forward_streaming_batch."""

    index: int
    req_id: str | None
    stream_finished: bool
    token: torch.Tensor
    prompt_token: torch.Tensor
    prompt_feat: torch.Tensor
    embedding: torch.Tensor
    cache_state: StreamingHiFTState | None
    token_offset_tokens: int
    finalize: bool


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
        estimator = _build_dit_estimator(decoder_cfg["estimator"])

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
        self.source_cache_len = int(self.mel_cache_len * 256)
        self.speech_window = np.hamming(2 * self.source_cache_len)
        # Must cover decode()'s own causal receptive field, not just the F0
        # margin; window_len=48 already passes
        # test_incremental_hift_bounded_window_is_close, so 64 has headroom.
        self._hift_window_len = 64

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
        token_lens: torch.Tensor | None = None,
        prompt_token_lens: torch.Tensor | None = None,
        prompt_feat_lens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate mel features via the upstream flow-model inference path."""
        flow_weight = next(self.flow_model.parameters())
        device = flow_weight.device
        dtype = flow_weight.dtype

        token = token.to(device=device, dtype=torch.int32)
        prompt_token = prompt_token.to(device=device, dtype=torch.int32)
        prompt_feat = prompt_feat.to(device=device, dtype=dtype)
        embedding = embedding.to(device=device, dtype=dtype)
        batch_size = int(token.shape[0])
        token_len = (
            token_lens.to(device=device, dtype=torch.int32)
            if token_lens is not None
            else torch.full((batch_size,), token.shape[1], device=device, dtype=torch.int32)
        )
        prompt_token_len = (
            prompt_token_lens.to(device=device, dtype=torch.int32)
            if prompt_token_lens is not None
            else torch.full((batch_size,), prompt_token.shape[1], device=device, dtype=torch.int32)
        )
        prompt_feat_len = (
            prompt_feat_lens.to(device=device, dtype=torch.int32)
            if prompt_feat_lens is not None
            else torch.full((batch_size,), prompt_feat.shape[1], device=device, dtype=torch.int32)
        )

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

    def _stream_hift_from_feat(
        self,
        feat: torch.Tensor,
        *,
        cache_state: StreamingHiFTState | None = None,
        finalize: bool = False,
    ) -> tuple[torch.Tensor, StreamingHiFTState | None]:
        hift_weight = self.hift.m_source.l_linear.weight
        chunk_mel = feat.to(device=hift_weight.device, dtype=hift_weight.dtype)

        cached_mel = None if not cache_state else cache_state.get("mel")
        cached_offset = 0 if not cache_state else int(cache_state.get("mel_offset", 0))
        window_len = self._hift_window_len
        samples_per_mel = int(np.prod(self.hift.upsample_rates) * self.hift.istft_params["hop_len"])
        trim = int(self.hift.f0_predictor.condnet[0].causal_padding)
        f0_margin_frames = int(self.hift.f0_predictor.left_context_frames)
        f0_margin = 0
        if isinstance(cached_mel, torch.Tensor) and cached_mel.numel() > 0:
            cached_mel = cached_mel.to(device=chunk_mel.device, dtype=chunk_mel.dtype)
            total_hist = cached_mel.shape[-1]
            overlap = min(window_len + trim, total_hist)
            window_mel = torch.cat([cached_mel[..., -overlap:], chunk_mel], dim=-1)
            window_offset = cached_offset + total_hist - overlap
            uv_offset = window_offset * samples_per_mel
            f0_margin = min(f0_margin_frames, total_hist - overlap)
            f0_input_mel = (
                torch.cat(
                    [cached_mel[..., total_hist - overlap - f0_margin : total_hist - overlap], window_mel], dim=-1
                )
                if f0_margin > 0
                else window_mel
            )
        else:
            window_mel = chunk_mel
            f0_input_mel = chunk_mel
            window_offset = 0
            uv_offset = 0

        phase_acc = None if not cache_state else cache_state.get("phase_acc")
        if phase_acc is not None:
            phase_acc = phase_acc.to(device=chunk_mel.device, dtype=chunk_mel.dtype)

        if window_mel.shape[-1] == 0:
            tts_speech = torch.zeros((chunk_mel.shape[0], 1, 0), device=chunk_mel.device, dtype=chunk_mel.dtype)
            new_phase_acc = phase_acc
        else:
            next_overlap = min(window_len + trim, window_mel.shape[-1])
            # On finalize the F0 predictor consumes the whole window (it does not
            # hold back `trim` frames), so trim must not inflate the phase-carry
            # index -- it would point past the end of a short window.
            carry_trim = 0 if finalize else trim
            tts_speech, _, new_phase_acc = self.hift.inference(
                speech_feat=f0_input_mel,
                finalize=finalize,
                phase_acc=phase_acc,
                uv_offset=uv_offset,
                next_overlap=next_overlap,
                trim=carry_trim,
                f0_margin=f0_margin,
            )

        tts_speech = tts_speech.reshape(tts_speech.shape[0], -1)
        new_samples = chunk_mel.shape[-1] * samples_per_mel
        if finalize:
            frames_withheld_per_chunk = trim + self.hift.conv_pre_look_right + 1
            released = frames_withheld_per_chunk * samples_per_mel
            emitted_speech = tts_speech[:, -(new_samples + released) :]
        else:
            emitted_speech = tts_speech[:, -new_samples:] if new_samples > 0 else tts_speech[:, :0]

        if finalize:
            return emitted_speech.reshape(emitted_speech.shape[0], 1, -1), None

        # Clone GPU views so each request owns its cache and does not retain
        # the full batch allocation or alias a later inference output.
        new_state = {
            "mel": (
                f0_input_mel.detach().clone()
                if cosyvoice3_packed_streaming_enabled()
                else f0_input_mel.detach().cpu().contiguous()
            ),
            "mel_offset": window_offset - f0_margin,
            "phase_acc": (
                (
                    new_phase_acc.detach().clone()
                    if cosyvoice3_packed_streaming_enabled()
                    else new_phase_acc.detach().cpu().contiguous()
                )
                if new_phase_acc is not None
                else None
            ),
        }
        return emitted_speech.reshape(emitted_speech.shape[0], 1, -1), new_state

    @torch.inference_mode()
    def forward_streaming_batch(
        self,
        items: list[StreamingFlowItem],
        *,
        n_timesteps: int = 10,
    ) -> list[tuple[torch.Tensor, StreamingHiFTState | None]]:
        """Batch the flow-matching mel path, then run HiFT per request.

        Items are grouped by prompt condition shape and finalization state.
        Codec tokens may have different lengths; those are padded within the
        group and passed to the flow as per-row token lengths.
        """
        if items and cosyvoice3_packed_streaming_enabled():
            results = [None] * len(items)
            for finalize in (False, True):
                group = [(i, x) for i, x in enumerate(items) if bool(x.get("finalize", False)) == finalize]
                if group:
                    values = self.forward_batch([x for _, x in group], n_timesteps=n_timesteps, stream_items=True)
                    for (i, _), value in zip(group, values):
                        results[i] = value
            return results
        if (
            items
            and cosyvoice3_packed_inference_enabled()
            and all(
                item.get("finalize", False)
                and item.get("cache_state") is None
                and int(item.get("token_offset_tokens", 0)) == 0
                for item in items
            )
        ):
            return [(speech, None) for speech in self.forward_batch(items, n_timesteps=n_timesteps)]
        results: list[tuple[torch.Tensor, StreamingHiFTState | None] | None] = [None] * len(items)
        groups: dict[tuple[int, int, int, bool], list[tuple[int, StreamingFlowItem]]] = {}
        for index, item in enumerate(items):
            assert isinstance(item["token"], torch.Tensor)
            assert isinstance(item["prompt_token"], torch.Tensor)
            assert isinstance(item["prompt_feat"], torch.Tensor)
            assert isinstance(item["embedding"], torch.Tensor)
            key = (
                int(item["prompt_token"].shape[1]),
                int(item["prompt_feat"].shape[1]),
                int(item["embedding"].shape[1]),
                bool(item.get("finalize", False)),
            )
            groups.setdefault(key, []).append((index, item))

        if cosyvoice3_batch_flow_debug():
            group_summary = {key: len(group) for key, group in groups.items()}
            group_size_distribution = Counter(group_summary.values())
            batchable_items = sum(size for size in group_summary.values() if size > 1)
            logger.info(
                "CosyVoice3 code2wav debug: forward_streaming_batch items=%d "
                "groups=%s group_size_distribution=%s batchable_items=%d",
                len(items),
                group_summary,
                dict(sorted(group_size_distribution.items())),
                batchable_items,
            )

        for _key, group in groups.items():
            if len(group) == 1:
                index, item = group[0]
                result = self.forward_streaming(
                    token=item["token"],
                    prompt_token=item["prompt_token"],
                    prompt_feat=item["prompt_feat"],
                    embedding=item["embedding"],
                    cache_state=item.get("cache_state"),
                    n_timesteps=n_timesteps,
                    token_offset_tokens=int(item.get("token_offset_tokens", 0)),
                    finalize=bool(item.get("finalize", False)),
                )
                results[index] = result
                continue

            token_tensors = [item["token"] for _, item in group]
            token_lens = torch.tensor(
                [int(token.shape[1]) for token in token_tensors],
                dtype=torch.int32,
            )
            max_token_len = int(token_lens.max().item())
            padded_tokens = []
            for token in token_tensors:
                if int(token.shape[1]) == max_token_len:
                    padded_tokens.append(token)
                else:
                    pad = torch.zeros(
                        (token.shape[0], max_token_len - int(token.shape[1])),
                        device=token.device,
                        dtype=token.dtype,
                    )
                    padded_tokens.append(torch.cat([token, pad], dim=1))
            tokens = torch.cat(padded_tokens, dim=0)
            prompt_tokens = torch.cat([item["prompt_token"] for _, item in group], dim=0)
            prompt_feats = torch.cat([item["prompt_feat"] for _, item in group], dim=0)
            embeddings = torch.cat([item["embedding"] for _, item in group], dim=0)
            prompt_token_lens = torch.full((len(group),), prompt_tokens.shape[1], dtype=torch.int32)
            prompt_feat_lens = torch.full((len(group),), prompt_feats.shape[1], dtype=torch.int32)
            finalize = bool(group[0][1].get("finalize", False))

            with cosyvoice3_batch_flow_profile(f"cosyvoice3_flow_batch_b{len(group)}_t{tokens.shape[1]}"):
                feat = self._forward_mel(
                    token=tokens,
                    prompt_token=prompt_tokens,
                    prompt_feat=prompt_feats,
                    embedding=embeddings,
                    n_timesteps=n_timesteps,
                    token_offset_tokens=0,
                    streaming=True,
                    finalize=finalize,
                    token_lens=token_lens,
                    prompt_token_lens=prompt_token_lens,
                    prompt_feat_lens=prompt_feat_lens,
                )

            for row, (index, item) in enumerate(group):
                trim_mel = max(0, int(item.get("token_offset_tokens", 0))) * int(self.token_mel_ratio)
                valid_tokens = int(token_lens[row].item())
                if not finalize:
                    valid_tokens = max(0, valid_tokens - int(self.flow_model.pre_lookahead_len))
                valid_mel = valid_tokens * int(self.token_mel_ratio)
                row_feat = feat[row : row + 1, :, :valid_mel]
                if trim_mel > 0:
                    row_feat = row_feat[:, :, trim_mel:]
                results[index] = self._stream_hift_from_feat(
                    row_feat,
                    cache_state=item.get("cache_state"),
                    finalize=finalize,
                )

        assert all(result is not None for result in results), "every streaming item must produce exactly one result"
        return cast(list[tuple[torch.Tensor, StreamingHiFTState | None]], results)

    @torch.inference_mode()
    def forward_streaming(
        self,
        token: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
        *,
        cache_state: StreamingHiFTState | None = None,
        n_timesteps: int = 10,
        token_offset_tokens: int = 0,
        finalize: bool = False,
    ) -> tuple[torch.Tensor, StreamingHiFTState | None]:
        """Decode streaming audio using cumulative mel + emitted-speech offset.

        This mirrors upstream CosyVoice3 streaming semantics more closely than
        waveform-domain overlap-add: keep a cumulative mel history per request,
        re-run causal HiFT on the history, and emit only the newly grown speech
        suffix. That preserves causal look-right handling without double
        trimming or duplicated overlap at chunk boundaries.
        """
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
        return self._stream_hift_from_feat(feat, cache_state=cache_state, finalize=finalize)

    @torch.inference_mode()
    def _stream_position_noise(self, width: int, like: torch.Tensor) -> torch.Tensor:
        """Extend fixed-position noise without changing previously emitted prefixes.

        Fixed-size CPU blocks preserve the initial experiment's seed-0 noise,
        isolate the global RNG, and make growth independent of request ordering.
        """
        block_frames = 15000
        cached = getattr(self, "_position_noise", None)
        present = 0 if cached is None else cached.shape[-1]
        blocks = [] if cached is None else [cached.to(like)]
        for index in range(present // block_frames, (width + block_frames - 1) // block_frames):
            generator = torch.Generator(device="cpu").manual_seed(index)
            blocks.append(torch.randn((1, 80, block_frames), generator=generator).to(like))
        if not blocks:
            return like.new_empty((1, 80, 0))
        self._position_noise = torch.cat(blocks, dim=-1) if len(blocks) > 1 else blocks[0]
        return self._position_noise[..., :width]

    @overload
    def forward_batch(
        self, items: list[StreamingFlowItem], *, n_timesteps: int = 10, stream_items: Literal[False] = False
    ) -> list[torch.Tensor]: ...

    @overload
    def forward_batch(
        self, items: list[StreamingFlowItem], *, n_timesteps: int = 10, stream_items: Literal[True]
    ) -> list[tuple[torch.Tensor, StreamingHiFTState | None]]: ...

    @torch.inference_mode()
    def forward_batch(
        self, items: list[StreamingFlowItem], *, n_timesteps: int = 10, stream_items: bool = False
    ) -> list[torch.Tensor] | list[tuple[torch.Tensor, StreamingHiFTState | None]]:
        """Batch Flow across different reference lengths and restore request order.

        Build each prompt/continuation contiguously before right padding. Padding
        the prompt and continuation separately would insert a gap in conditioning.
        The default keeps exact-length CPU-F0 HiFT. The opt-in Hopper profile
        uses packed Flow, batched HiFT, and FP64 GPU F0.
        """
        if not items:
            return []
        if stream_items:
            if not cosyvoice3_packed_streaming_enabled():
                raise ValueError("Packed streaming requires COSYVOICE3_PACKED_STREAMING on Hopper")
            if any(bool(item.get("finalize", False)) != bool(items[0].get("finalize", False)) for item in items):
                raise ValueError("Use forward_streaming_batch to group mixed finalization states")
        if len(items) > 8:
            ordered = sorted(
                enumerate(items), key=lambda pair: pair[1]["token"].shape[1] + pair[1]["prompt_token"].shape[1]
            )
            outputs: list[torch.Tensor | tuple[torch.Tensor, StreamingHiFTState | None] | None] = [None] * len(items)
            for start in range(0, len(ordered), 8):
                group = ordered[start : start + 8]
                for (index, _), speech in zip(
                    group,
                    self.forward_batch([item for _, item in group], n_timesteps=n_timesteps, stream_items=stream_items),
                ):
                    outputs[index] = speech
            return cast(list[torch.Tensor] | list[tuple[torch.Tensor, StreamingHiFTState | None]], outputs)
        flow = self.flow_model
        weight = next(flow.parameters())
        mus, conds, speakers, lengths, prompt_lengths = [], [], [], [], []
        for item in items:
            tokens = torch.cat(
                tuple(item[key].to(device=weight.device, dtype=torch.long) for key in ("prompt_token", "token")), dim=1
            )
            embedded = flow.input_embedding(tokens.clamp(min=0))
            if stream_items and not item.get("finalize", False) and int(flow.pre_lookahead_len) > 0:
                lookahead = int(flow.pre_lookahead_len)
                hidden = flow.pre_lookahead_layer(embedded[:, :-lookahead], context=embedded[:, -lookahead:])
            else:
                hidden = flow.pre_lookahead_layer(embedded)
            mu = hidden.repeat_interleave(self.token_mel_ratio, dim=1)
            prompt = item["prompt_feat"].to(device=weight.device, dtype=mu.dtype)
            cond = torch.zeros_like(mu)
            cond[:, : prompt.shape[1]] = prompt
            speaker = torch.nn.functional.normalize(item["embedding"].to(weight), dim=1)
            speakers.append(flow.spk_embed_affine_layer(speaker))
            mus.append(mu.transpose(1, 2))
            conds.append(cond.transpose(1, 2))
            lengths.append(mu.shape[1])
            prompt_lengths.append(prompt.shape[1])
        width = max(lengths)
        mu = torch.cat([torch.nn.functional.pad(x, (0, width - x.shape[-1])) for x in mus])
        cond = torch.cat([torch.nn.functional.pad(x, (0, width - x.shape[-1])) for x in conds])
        mask = (
            (torch.arange(width, device=weight.device)[None, :] < torch.tensor(lengths, device=weight.device)[:, None])
            .unsqueeze(1)
            .to(mu)
        )
        if cosyvoice3_packed_inference_enabled():
            from .code2wav_core.packed_dit import (
                PackedDiT,
                gather_rows,
                pack_rows,
                scatter_rows,
                solve_flow_euler_packed,
            )

            if not hasattr(self, "_packed_estimator"):
                self._packed_estimator = PackedDiT(flow.decoder.estimator)
                self._packed_estimator.compile(mu.dtype)
            rows = pack_rows(lengths, mu.device)
            if stream_items:
                noise = self._stream_position_noise(width, mu).expand(len(items), -1, -1)
            else:
                noise = torch.randn(mu.shape, device=mu.device, dtype=mu.dtype)
            time_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
            if flow.decoder.t_scheduler == "cosine":
                time_span = 1 - torch.cos(time_span * 0.5 * torch.pi)
            with torch.autocast("cuda", dtype=mu.dtype):
                packed = solve_flow_euler_packed(
                    self._packed_estimator,
                    gather_rows(noise.transpose(1, 2), rows),
                    time_span,
                    gather_rows(mu.transpose(1, 2), rows),
                    torch.cat(speakers),
                    gather_rows(cond.transpose(1, 2), rows),
                    rows,
                    cfg_rate=flow.decoder.inference_cfg_rate,
                    streaming=stream_items and not items[0].get("finalize", False),
                    modulation_key=(n_timesteps, flow.decoder.t_scheduler, mu.dtype, mu.device),
                )
            feat = scatter_rows(packed, rows, width).transpose(1, 2)

        else:
            feat, _ = flow.decoder(
                mu=mu, mask=mask, spks=torch.cat(speakers), cond=cond, n_timesteps=n_timesteps, streaming=False
            )
        if stream_items:
            results = []
            for row, item in enumerate(items):
                start = prompt_lengths[row] + max(0, int(item.get("token_offset_tokens", 0))) * self.token_mel_ratio
                mel = feat[row : row + 1, :, start : lengths[row]].float()
                results.append(
                    self._stream_hift_from_feat(
                        mel, cache_state=item.get("cache_state"), finalize=bool(item.get("finalize", False))
                    )
                )
            return results
        hift_weight = self.hift.m_source.l_linear.weight
        if cosyvoice3_full_response_enabled():
            mels = []
            for row, item in enumerate(items):
                start = prompt_lengths[row] + max(0, int(item.get("token_offset_tokens", 0))) * self.token_mel_ratio
                mels.append(feat[row : row + 1, :, start : lengths[row]].float().to(hift_weight))
            width = max(mel.shape[-1] for mel in mels)
            if width == 0:
                return [mel.new_zeros((1, 1, 0)) for mel in mels]
            from .code2wav_core.packed_dit import bucketed_width

            # Recurring shapes let cuDNN/cuFFT reuse plans; rows are already
            # right padded to the batch maximum and trimmed below.
            width = bucketed_width(width)
            padded = torch.cat([torch.nn.functional.pad(mel, (0, width - mel.shape[-1])) for mel in mels])
            speech, _, _ = self.hift.inference(speech_feat=padded, finalize=True)
            stride = int(np.prod(self.hift.upsample_rates) * self.hift.istft_params["hop_len"])
            return [speech[row : row + 1, ..., : mel.shape[-1] * stride] for row, mel in enumerate(mels)]
        results = []
        for row, item in enumerate(items):
            start = prompt_lengths[row] + max(0, int(item.get("token_offset_tokens", 0))) * self.token_mel_ratio
            mel = feat[row : row + 1, :, start : lengths[row]].float().to(hift_weight)
            if mel.shape[-1] == 0:
                speech = mel.new_zeros((1, 1, 0))
            else:
                speech, _, _ = self.hift.inference(speech_feat=mel, finalize=True)
            results.append(speech)
        return results

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

        # Run vocoder
        hift_weight = self.hift.m_source.l_linear.weight
        tts_mel = feat.to(device=hift_weight.device, dtype=hift_weight.dtype)

        if tts_mel.shape[-1] == 0:
            tts_speech = torch.zeros(
                (tts_mel.shape[0], 1, 0),
                device=tts_mel.device,
                dtype=tts_mel.dtype,
            )
        else:
            tts_speech, _, _ = self.hift.inference(speech_feat=tts_mel, finalize=True)

        return tts_speech

    def load_weights(self, model_dir: str, device: torch.device) -> None:
        """Load flow.pt and hift.pt weights.

        Args:
            model_dir: Model directory containing flow.pt and hift.pt
            device: Device to load weights to
        """

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
        self.hift.to(device)
        # Fold after loading and device placement to avoid recomputing weights
        # for every chunk. Folding changes state_dict keys, so reloading the
        # original checkpoint into this instance is unsupported.
        folded = self.hift.remove_weight_norm()
        logger.info("Folded %d weight-norm layers in HiFT generator", folded)
        if folded == 0:
            logger.warning("HiFT generator had no weight-norm layers to fold; check config drift")
        # Fold on the same device as mainline F0 inference. Folding on CPU
        # and moving to GPU later changes the normalized FP32 weights.
        f0_device = device if device.type == "cuda" and os.getenv("COSYVOICE3_F0_ON_CPU", "0") != "1" else "cpu"
        self.hift.f0_predictor.to(device=f0_device, dtype=torch.float32)
        f0_folded = self.hift.f0_predictor.remove_weight_norm()
        logger.info("Folded %d weight-norm layers in HiFT F0 predictor", f0_folded)
        if f0_folded == 0:
            logger.warning("HiFT F0 predictor had no weight-norm layers to fold; check config drift")
        self.hift.eval()
        logger.info(f"Loaded hift weights from {hift_path}")
