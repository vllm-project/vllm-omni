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

import math
from collections import Counter
from typing import cast

import torch
import torch.nn as nn
from omegaconf import DictConfig
from vllm.logger import init_logger

from vllm_omni.diffusion.models.cosyvoice3_audio.cosyvoice3_dit import DiT
from vllm_omni.model_executor.models.common.audio_stream_utils import (
    build_overlap_window,
    fade_in_out,
)
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
)
from vllm_omni.transformers_utils.configs.cosyvoice3 import CosyVoice3Config

logger = init_logger(__name__)


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
        self.mel_cache_len = 20
        upsample_rates = getattr(self.hift, "upsample_rates", [8, 5, 3])
        istft_hop_len = getattr(self.hift, "istft_params", {}).get("hop_len", 4)
        upsample_scale = int(math.prod(upsample_rates) * istft_hop_len)
        self.source_cache_len = int(self.mel_cache_len * upsample_scale)

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

    def _get_speech_window(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        cached = getattr(self, "_torch_speech_window", None)
        if cached is None or cached.device != device or cached.dtype != dtype:
            cached = build_overlap_window(self.source_cache_len, device=device, dtype=dtype)
            self._torch_speech_window = cached
        return cached

    def _stream_hift_from_feat(
        self,
        feat: torch.Tensor,
        *,
        cache_state: dict[str, torch.Tensor] | None = None,
        finalize: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        hift_param = next(self.hift.parameters(), None) if hasattr(self, "hift") else None
        if hift_param is None and hasattr(self, "hift") and hasattr(self.hift, "m_source"):
            l_linear = getattr(self.hift.m_source, "l_linear", None)
            if l_linear is not None:
                hift_param = getattr(l_linear, "weight", None)
        device = hift_param.device if hift_param is not None else feat.device
        dtype = hift_param.dtype if hift_param is not None else feat.dtype
        chunk_mel = feat.to(device=device, dtype=dtype)

        cached_mel = None if not cache_state else cache_state.get("mel")
        cached_speech = None if not cache_state else cache_state.get("speech")

        # Bounded mel cache (Task C1): bound history to self.mel_cache_len frames
        if isinstance(cached_mel, torch.Tensor) and cached_mel.numel() > 0:
            cached_mel = cached_mel.to(device=chunk_mel.device, dtype=chunk_mel.dtype)
            if cached_mel.shape[-1] > self.mel_cache_len:
                cached_mel = cached_mel[..., -self.mel_cache_len :]
            tts_mel = torch.cat([cached_mel, chunk_mel], dim=-1) if chunk_mel.numel() > 0 else cached_mel
        else:
            tts_mel = chunk_mel

        if tts_mel.shape[-1] == 0:
            tts_speech = torch.zeros((chunk_mel.shape[0], 1, 0), device=chunk_mel.device, dtype=chunk_mel.dtype)
        else:
            tts_speech, _ = self.hift.inference(speech_feat=tts_mel, finalize=finalize)

        tts_speech = tts_speech.reshape(tts_speech.shape[0], 1, -1)

        # Overlap-add crossfade with previous chunk's speech (Task C1)
        if isinstance(cached_speech, torch.Tensor) and cached_speech.numel() > 0:
            cached_speech = cached_speech.to(device=tts_speech.device, dtype=tts_speech.dtype)
            window = self._get_speech_window(device=tts_speech.device, dtype=tts_speech.dtype)
            tts_speech = fade_in_out(tts_speech, cached_speech, window)

        if finalize:
            return tts_speech, None

        overlap_len = min(int(self.source_cache_len), int(tts_speech.shape[-1]))
        if overlap_len > 0:
            emitted_speech = tts_speech[..., :-overlap_len]
            tail_speech = tts_speech[..., -overlap_len:]
        else:
            emitted_speech = tts_speech
            tail_speech = tts_speech[..., :0]

        new_state = {
            "mel": tts_mel[..., -self.mel_cache_len :].detach(),
            "speech": tail_speech.detach(),
            "speech_offset": int(tts_speech.shape[-1]),
        }
        return emitted_speech, new_state

    def _stream_hift_from_feat_batch(
        self,
        items: list[tuple[int, torch.Tensor, dict[str, torch.Tensor] | None]],
        *,
        finalize: bool = False,
    ) -> list[tuple[int, tuple[torch.Tensor, dict[str, torch.Tensor] | None]]]:
        """Batch HiFT vocoder inference across streaming requests via equal-length bucketing.

        Items with equal mel length are stacked and inferred in a single batched
        kernel invocation, completely eliminating serial vocoder tail latency
        while avoiding padding artifacts.
        """
        if not items:
            return []

        # Route to patched method seam or fallback if hift is not initialized
        if type(self)._stream_hift_from_feat != self._stream_hift_from_feat or not hasattr(self, "hift"):
            return [(idx, self._stream_hift_from_feat(f, cache_state=cs, finalize=finalize)) for idx, f, cs in items]

        # Prepare per-item bounded mel and retrieve cached speech
        prepared: list[tuple[int, torch.Tensor, torch.Tensor | None]] = []
        hift_param = next(self.hift.parameters(), None) if hasattr(self, "hift") else None
        if hift_param is None and hasattr(self, "hift") and hasattr(self.hift, "m_source"):
            l_linear = getattr(self.hift.m_source, "l_linear", None)
            if l_linear is not None:
                hift_param = getattr(l_linear, "weight", None)
        device = hift_param.device if hift_param is not None else items[0][1].device
        dtype = hift_param.dtype if hift_param is not None else items[0][1].dtype

        for orig_idx, feat, cache_state in items:
            chunk_mel = feat.to(device=device, dtype=dtype)
            cached_mel = None if not cache_state else cache_state.get("mel")
            cached_speech = None if not cache_state else cache_state.get("speech")

            if isinstance(cached_mel, torch.Tensor) and cached_mel.numel() > 0:
                cached_mel = cached_mel.to(device=device, dtype=dtype)
                if cached_mel.shape[-1] > self.mel_cache_len:
                    cached_mel = cached_mel[..., -self.mel_cache_len :]
                tts_mel = torch.cat([cached_mel, chunk_mel], dim=-1) if chunk_mel.numel() > 0 else cached_mel
            else:
                tts_mel = chunk_mel
            prepared.append((orig_idx, tts_mel, cached_speech))

        # Group by tts_mel.shape[-1]
        buckets: dict[int, list[tuple[int, torch.Tensor, torch.Tensor | None]]] = {}
        for orig_idx, tts_mel, cached_speech in prepared:
            mel_len = int(tts_mel.shape[-1])
            buckets.setdefault(mel_len, []).append((orig_idx, tts_mel, cached_speech))

        results: list[tuple[int, tuple[torch.Tensor, dict[str, torch.Tensor] | None]]] = []
        window = self._get_speech_window(device=device, dtype=dtype)

        for mel_len, bucket_items in buckets.items():
            if mel_len == 0:
                for orig_idx, tts_mel, _ in bucket_items:
                    empty_speech = torch.zeros((1, 1, 0), device=device, dtype=dtype)
                    state = (
                        None
                        if finalize
                        else {
                            "mel": tts_mel[..., -self.mel_cache_len :].detach(),
                            "speech": torch.zeros((1, 1, 0), device=device, dtype=dtype),
                            "speech_offset": 0,
                        }
                    )
                    results.append((orig_idx, (empty_speech, state)))
                continue

            if len(bucket_items) == 1:
                orig_idx, tts_mel, cached_speech = bucket_items[0]
                tts_speech, _ = self.hift.inference(speech_feat=tts_mel, finalize=finalize)
                tts_speech = tts_speech.reshape(1, 1, -1)
                if isinstance(cached_speech, torch.Tensor) and cached_speech.numel() > 0:
                    cached_speech = cached_speech.to(device=device, dtype=dtype)
                    tts_speech = fade_in_out(tts_speech, cached_speech, window)

                if finalize:
                    results.append((orig_idx, (tts_speech, None)))
                else:
                    overlap_len = min(int(self.source_cache_len), int(tts_speech.shape[-1]))
                    if overlap_len > 0:
                        emitted = tts_speech[..., :-overlap_len]
                        tail = tts_speech[..., -overlap_len:]
                    else:
                        emitted = tts_speech
                        tail = tts_speech[..., :0]
                    state = {
                        "mel": tts_mel[..., -self.mel_cache_len :].detach(),
                        "speech": tail.detach(),
                        "speech_offset": int(tts_speech.shape[-1]),
                    }
                    results.append((orig_idx, (emitted, state)))
                continue

            # Batched execution for multiple requests with identical mel length
            batch_mel = torch.cat([item[1] for item in bucket_items], dim=0)
            batch_speech, _ = self.hift.inference(speech_feat=batch_mel, finalize=finalize)
            batch_speech = batch_speech.reshape(batch_speech.shape[0], 1, -1)

            for row, (orig_idx, tts_mel, cached_speech) in enumerate(bucket_items):
                tts_speech = batch_speech[row : row + 1]
                if isinstance(cached_speech, torch.Tensor) and cached_speech.numel() > 0:
                    cached_speech = cached_speech.to(device=device, dtype=dtype)
                    tts_speech = fade_in_out(tts_speech, cached_speech, window)

                if finalize:
                    results.append((orig_idx, (tts_speech, None)))
                else:
                    overlap_len = min(int(self.source_cache_len), int(tts_speech.shape[-1]))
                    if overlap_len > 0:
                        emitted = tts_speech[..., :-overlap_len]
                        tail = tts_speech[..., -overlap_len:]
                    else:
                        emitted = tts_speech
                        tail = tts_speech[..., :0]
                    state = {
                        "mel": tts_mel[..., -self.mel_cache_len :].detach(),
                        "speech": tail.detach(),
                        "speech_offset": int(tts_speech.shape[-1]),
                    }
                    results.append((orig_idx, (emitted, state)))

        return results

    @torch.inference_mode()
    def forward_streaming_batch(
        self,
        items: list[dict[str, object]],
        *,
        n_timesteps: int = 10,
    ) -> list[tuple[torch.Tensor, dict[str, torch.Tensor] | None]]:
        """Batch the flow-matching mel path, then run HiFT per request.

        Items are grouped by prompt condition shape and finalization state.
        Codec tokens may have different lengths; those are padded within the
        group and passed to the flow as per-row token lengths.
        """
        results: list[tuple[torch.Tensor, dict[str, torch.Tensor] | None] | None] = [None] * len(items)
        groups: dict[tuple[int, int, int, bool], list[tuple[int, dict[str, object]]]] = {}
        for index, item in enumerate(items):
            token = item["token"]
            prompt_token = item["prompt_token"]
            prompt_feat = item["prompt_feat"]
            embedding = item["embedding"]
            assert isinstance(token, torch.Tensor)
            assert isinstance(prompt_token, torch.Tensor)
            assert isinstance(prompt_feat, torch.Tensor)
            assert isinstance(embedding, torch.Tensor)
            key = (
                int(prompt_token.shape[1]),
                int(prompt_feat.shape[1]),
                int(embedding.shape[1]),
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
                    token=item["token"],  # type: ignore[arg-type]
                    prompt_token=item["prompt_token"],  # type: ignore[arg-type]
                    prompt_feat=item["prompt_feat"],  # type: ignore[arg-type]
                    embedding=item["embedding"],  # type: ignore[arg-type]
                    cache_state=item.get("cache_state"),  # type: ignore[arg-type]
                    n_timesteps=n_timesteps,
                    token_offset_tokens=int(item.get("token_offset_tokens", 0)),
                    finalize=bool(item.get("finalize", False)),
                )
                results[index] = result
                continue

            token_tensors = [item["token"] for _, item in group]
            assert all(isinstance(token, torch.Tensor) for token in token_tensors)
            token_lens = torch.tensor(
                [int(token.shape[1]) for token in token_tensors],  # type: ignore[union-attr]
                dtype=torch.int32,
            )
            max_token_len = int(token_lens.max().item())
            padded_tokens = []
            for token in token_tensors:
                assert isinstance(token, torch.Tensor)
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
            prompt_tokens = torch.cat([item["prompt_token"] for _, item in group], dim=0)  # type: ignore[list-item]
            prompt_feats = torch.cat([item["prompt_feat"] for _, item in group], dim=0)  # type: ignore[list-item]
            embeddings = torch.cat([item["embedding"] for _, item in group], dim=0)  # type: ignore[list-item]
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

            shift_items = []
            for row, (index, item) in enumerate(group):
                trim_mel = max(0, int(item.get("token_offset_tokens", 0))) * int(self.token_mel_ratio)
                valid_tokens = int(token_lens[row].item())
                if not finalize:
                    valid_tokens = max(0, valid_tokens - int(self.flow_model.pre_lookahead_len))
                valid_mel = valid_tokens * int(self.token_mel_ratio)
                row_feat = feat[row : row + 1, :, :valid_mel]
                if trim_mel > 0:
                    row_feat = row_feat[:, :, trim_mel:]
                shift_items.append((index, row_feat, item.get("cache_state")))

            hift_results = self._stream_hift_from_feat_batch(shift_items, finalize=finalize)
            for orig_idx, res in hift_results:
                results[orig_idx] = res

        assert all(result is not None for result in results), "every streaming item must produce exactly one result"
        return cast(list[tuple[torch.Tensor, dict[str, torch.Tensor] | None]], results)

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
        """Decode streaming audio using bounded mel cache and Hamming cross-fade.

        Retains a bounded mel prefix (up to mel_cache_len frames) across chunks
        to avoid cumulative recomputation overhead, blending overlapping audio
        boundaries via smooth cross-fading.
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
        hift_param = next(self.hift.parameters(), None) if hasattr(self, "hift") else None
        if hift_param is None and hasattr(self, "hift") and hasattr(self.hift, "m_source"):
            l_linear = getattr(self.hift.m_source, "l_linear", None)
            if l_linear is not None:
                hift_param = getattr(l_linear, "weight", None)
        device = hift_param.device if hift_param is not None else feat.device
        dtype = hift_param.dtype if hift_param is not None else feat.dtype
        tts_mel = feat.to(device=device, dtype=dtype)

        if tts_mel.shape[-1] == 0:
            tts_speech = torch.zeros(
                (tts_mel.shape[0], 1, 0),
                device=tts_mel.device,
                dtype=tts_mel.dtype,
            )
        else:
            tts_speech, _ = self.hift.inference(speech_feat=tts_mel, finalize=True)

        return tts_speech

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
