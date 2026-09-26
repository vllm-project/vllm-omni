# Copyright 2025 Xiaomi Corporation.
"""
CUDA Graph wrapper for MiMo Audio Tokenizer decoder.

Captures the full decode path (VQ dequantize -> AudioDecoder -> Vocoder -> Waveform)
into CUDA Graphs for fixed bucket sizes, eliminating kernel-launch overhead that
dominates small-chunk streaming decoding.

The vocoder uses non-causal sliding-window attention, so a pre-built attention mask
(static tensor) is updated in-place before each graph replay to properly handle
padding positions.
"""

from __future__ import annotations

import torch
from torch.cuda import CUDAGraph
from vllm.logger import init_logger

from .modeling_audio_tokenizer import StreamingCache, StreamingConfig, TransformerVocos

logger = init_logger(__name__)


class CUDAGraphMiMoDecoderWrapper:
    """CUDA Graph wrapper for MiMoAudioTokenizer.decode_fixed().

    Usage::

        wrapper = CUDAGraphMiMoDecoderWrapper(tokenizer)
        wrapper.warmup(device)
        wav = wrapper.decode(codes)  # auto graph-replay or eager fallback

    When ``code_rows`` lists multiple RVQ depths (e.g. talker emits 8 rows but
    config.num_quantizers is 20), one CUDA Graph is captured per (depth, bucket).
    """

    DEFAULT_CAPTURE_SIZES = [4, 6, 8, 12, 16, 24, 32, 64, 128, 256]

    def __init__(
        self,
        tokenizer,
        capture_sizes: list[int] | None = None,
        enabled: bool = True,
        code_rows: list[int] | None = None,
    ):
        self.tokenizer = tokenizer

        # num_quantizers = RVQ layers in checkpoint; talker may emit fewer rows
        self.audio_channels = tokenizer.config.num_quantizers
        nq = self.audio_channels
        if code_rows is None:
            rows = {nq}
        else:
            rows = {r for r in code_rows if 1 <= r <= nq}
            rows.add(nq)
        self._capture_code_rows: list[int] = sorted(rows)
        if len(self._capture_code_rows) > 1:
            logger.info(
                "CUDAGraphMiMoDecoderWrapper: capturing graphs for code depths %s (num_quantizers=%d)",
                self._capture_code_rows,
                nq,
            )

        self._explicit_sizes = capture_sizes is not None
        self.capture_sizes: list[int] = sorted(capture_sizes) if capture_sizes else []
        self.enabled = enabled

        self.graphs: dict[tuple[int, int], CUDAGraph] = {}
        self.static_code_inputs: dict[tuple[int, int], torch.Tensor] = {}
        self.static_wav_outputs: dict[tuple[int, int], torch.Tensor] = {}
        self.static_vocoder_masks: dict[int, torch.Tensor] = {}

        self._warmed_up = False
        self._device: torch.device | None = None

        cfg = tokenizer.config
        self.frames_per_token = cfg.avg_pooler * cfg.stride_size * cfg.hop_length

    @staticmethod
    def compute_capture_sizes(
        codec_chunk_frames: int = 0,
        codec_left_context_frames: int = 0,
    ) -> list[int]:
        """Derive bucket sizes from streaming chunk config for high hit rate."""
        sizes: set[int] = set()
        if codec_chunk_frames > 0:
            sizes.add(codec_chunk_frames)
            if codec_left_context_frames > 0:
                sizes.add(codec_chunk_frames + codec_left_context_frames)

        for p2 in CUDAGraphMiMoDecoderWrapper.DEFAULT_CAPTURE_SIZES:
            sizes.add(p2)
        return sorted(sizes)

    def _get_padded_size(self, actual_size: int) -> int | None:
        """Round up to the nearest captured bucket size."""
        for size in self.capture_sizes:
            if actual_size <= size:
                return size
        return None

    def _compute_output_wav_len(self, code_len: int) -> int:
        """Estimate the output waveform length for a given code length."""
        return self.tokenizer._compute_vocoder_seq_len(code_len) * self.tokenizer.config.hop_length

    def _compute_vocoder_seq_len(self, code_len: int) -> int:
        return self.tokenizer._compute_vocoder_seq_len(code_len)

    def warmup(
        self,
        device: torch.device,
        codec_chunk_frames: int = 0,
        codec_left_context_frames: int = 0,
    ):
        """Eager warmup then CUDA Graph capture for all bucket sizes."""
        if device.type != "cuda" or not self.enabled or self._warmed_up:
            return

        self._device = device
        self.tokenizer.eval()

        if not self._explicit_sizes:
            self.capture_sizes = self.compute_capture_sizes(
                codec_chunk_frames=codec_chunk_frames,
                codec_left_context_frames=codec_left_context_frames,
            )

        logger.info(
            "CUDAGraphMiMoDecoderWrapper: starting warmup for %d sizes: %s",
            len(self.capture_sizes),
            self.capture_sizes,
        )

        model_dtype = next(self.tokenizer.parameters()).dtype

        for size in self.capture_sizes:
            voc_len = self._compute_vocoder_seq_len(size)
            mask = TransformerVocos.build_vocoder_attn_mask(
                voc_len,
                voc_len,
                tuple(self.tokenizer.config.vocoder_attn_window_size),
                device,
                model_dtype,
            )
            self.static_vocoder_masks[size] = mask

            for n_rows in self._capture_code_rows:
                dummy = torch.zeros(n_rows, size, dtype=torch.long, device=device)
                with torch.no_grad():
                    _ = self.tokenizer.decode_fixed(dummy, size, vocoder_attn_mask=mask)

        torch.accelerator.synchronize(device)

        expected = len(self.capture_sizes) * len(self._capture_code_rows)
        for n_rows in self._capture_code_rows:
            for size in self.capture_sizes:
                try:
                    self._capture(size, device, n_code_rows=n_rows)
                    logger.info(
                        "  Captured CUDA Graph for code_rows=%d code_len=%d",
                        n_rows,
                        size,
                    )
                except Exception:
                    logger.warning(
                        "  Failed to capture graph for code_rows=%d code_len=%d",
                        n_rows,
                        size,
                        exc_info=True,
                    )

        self._warmed_up = True
        logger.info(
            "CUDAGraphMiMoDecoderWrapper warmup complete: %d/%d captured",
            len(self.graphs),
            expected,
        )

    def _capture(self, size: int, device: torch.device, n_code_rows: int):
        key = (n_code_rows, size)
        static_codes = torch.zeros(n_code_rows, size, dtype=torch.long, device=device)
        mask = self.static_vocoder_masks[size]

        with torch.no_grad():
            _ = self.tokenizer.decode_fixed(static_codes, size, vocoder_attn_mask=mask)
        torch.accelerator.synchronize(device)

        graph = CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(graph):
                static_output = self.tokenizer.decode_fixed(
                    static_codes,
                    size,
                    vocoder_attn_mask=mask,
                )

        self.graphs[key] = graph
        self.static_code_inputs[key] = static_codes
        self.static_wav_outputs[key] = static_output

    def _update_vocoder_mask(self, padded_size: int, actual_code_shape: int):
        """Update the static vocoder mask in-place for the given actual code length."""
        mask = self.static_vocoder_masks[padded_size]
        voc_padded = self._compute_vocoder_seq_len(padded_size)
        voc_actual = self._compute_vocoder_seq_len(actual_code_shape)

        new_mask = TransformerVocos.build_vocoder_attn_mask(
            voc_padded,
            voc_actual,
            tuple(self.tokenizer.config.vocoder_attn_window_size),
            mask.device,
            mask.dtype,
        )
        mask.copy_(new_mask)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode codes with CUDA Graph replay when possible.

        Args:
            codes: [n_rvq_rows, T] on CUDA device (n_rvq_rows may be < num_quantizers).
        Returns:
            Waveform tensor (same shape as MiMoAudioTokenizer.decode output).
        """
        if not self.enabled or not self._warmed_up:
            return self.tokenizer.decode(codes)

        actual_code_shape = codes.shape[-1]
        n_rows = codes.shape[0]
        padded_size = self._get_padded_size(actual_code_shape)
        if padded_size is None:
            return self.tokenizer.decode(codes)

        key = (n_rows, padded_size)
        if key not in self.graphs:
            return self.tokenizer.decode(codes)

        self.static_code_inputs[key].zero_()
        self.static_code_inputs[key][:, :actual_code_shape] = codes

        self._update_vocoder_mask(padded_size, actual_code_shape)

        self.graphs[key].replay()

        actual_wav_len = self._compute_output_wav_len(actual_code_shape)
        output = self.static_wav_outputs[key]
        return output[..., :actual_wav_len].clone()

    @torch.no_grad()
    def streaming_decode(
        self,
        codes_chunks: torch.Tensor,
        chunk_input_lengths: list[int],
        history_cache: StreamingCache | None = None,
        streaming_config: StreamingConfig | None = None,
        last_chunk: bool = False,
    ) -> tuple[list[torch.Tensor | None], StreamingCache]:
        """Streaming decode with CUDA Graph acceleration.

        Mirrors MiMoAudioTokenizer.streaming_decode(): identical cache /
        overlap logic, but replaces ``self.decoder(hs, lengths)`` with
        ``decoder.forward_fixed`` + vocoder attention mask so the
        fixed-shape path (and optionally CUDA Graph) can be used.

        Args:
            codes_chunks: [num_quantizers, sum(chunk_input_lengths)] — codes
                for all samples in the batch, concatenated along dim-1.
            chunk_input_lengths: per-sample code lengths in this chunk.
            history_cache: StreamingCache from the previous call.
            streaming_config: overlap / segment config.
            last_chunk: whether this is the final chunk.

        Returns:
            (return_wavs, history_cache) — same contract as the original.
        """
        if history_cache is None:
            history_cache = StreamingCache()
        if streaming_config is None:
            streaming_config = StreamingConfig()
        tokenizer = self.tokenizer
        hidden_states = tokenizer.encoder.decode_vq(codes_chunks)

        input_lengths: list[int] = []
        input_hidden_states: list[torch.Tensor] = []
        cache_hidden_states: list[torch.Tensor] = []
        start_idx = 0

        for i, length in enumerate(chunk_input_lengths):
            sample_hs = hidden_states[start_idx : start_idx + length]
            start_idx += length
            if history_cache.hidden_states is not None:
                sample_hs = torch.cat(
                    [history_cache.hidden_states[i], sample_hs],
                    dim=0,
                )
                length += history_cache.hidden_states[i].size(0)
            input_hidden_states.append(sample_hs)
            cache_hidden_states.append(sample_hs.clone())
            input_lengths.append(length)

        output = self._batch_decode_hidden_states(input_hidden_states, input_lengths)

        return_wavs: list[torch.Tensor | None] = []
        frames_per_token = self.frames_per_token
        processed_lengths: list[int] = []

        for i, wav in enumerate(output):
            wav = wav.float().detach().cpu()
            prev_processed = history_cache.processed_lengths[i] if history_cache.processed_lengths is not None else 0

            if last_chunk:
                return_wavs.append(wav[:, prev_processed * frames_per_token :])
                new_processed_length = input_lengths[i]
            elif input_lengths[i] <= streaming_config.right_overlap:
                return_wavs.append(None)
                new_processed_length = 0
            else:
                end_idx = input_lengths[i] - streaming_config.right_overlap
                wav = wav[:, prev_processed * frames_per_token : end_idx * frames_per_token]
                return_wavs.append(wav)
                new_processed_length = end_idx
                if input_lengths[i] > streaming_config.left_overlap:
                    cache_hidden_states[i] = cache_hidden_states[i][-streaming_config.left_overlap :]
                    new_processed_length -= input_lengths[i] - streaming_config.left_overlap

            processed_lengths.append(new_processed_length)

        history_cache.hidden_states = cache_hidden_states
        history_cache.processed_lengths = processed_lengths

        return return_wavs, history_cache

    def _batch_decode_hidden_states(
        self,
        hidden_states_list: list[torch.Tensor],
        input_lengths: list[int],
    ) -> list[torch.Tensor]:
        """Decode a list of per-sample hidden_states using the fixed-shape path.

        Each sample is decoded independently with padding + vocoder mask.
        Returns a list of wav tensors, each shaped [1, wav_len] (matching the
        per-sample shape from the original batched decoder output).
        """
        results: list[torch.Tensor] = []
        decoder = self.tokenizer.decoder
        cfg = self.tokenizer.config
        window_size = tuple(cfg.vocoder_attn_window_size)

        mask_dtype = decoder.vocoder.embeddings.weight.dtype

        for sample_hs, actual_code_shape in zip(hidden_states_list, input_lengths):
            padded_size = self._get_padded_size(actual_code_shape)

            if padded_size is None:
                wav = decoder(
                    sample_hs,
                    torch.tensor([actual_code_shape], device=sample_hs.device),
                )
                results.append(wav.squeeze(0))
                continue

            if actual_code_shape < padded_size:
                pad = torch.zeros(
                    padded_size - actual_code_shape,
                    sample_hs.size(-1),
                    device=sample_hs.device,
                    dtype=sample_hs.dtype,
                )
                padded_hs = torch.cat([sample_hs, pad], dim=0)
            else:
                padded_hs = sample_hs

            voc_padded = self._compute_vocoder_seq_len(padded_size)
            voc_actual = self._compute_vocoder_seq_len(actual_code_shape)
            vocoder_mask = TransformerVocos.build_vocoder_attn_mask(
                voc_padded,
                voc_actual,
                window_size,
                padded_hs.device,
                mask_dtype,
            )

            wav = decoder.forward_fixed(
                padded_hs.unsqueeze(0),
                vocoder_attn_mask=vocoder_mask,
            )

            actual_wav_len = self._compute_output_wav_len(actual_code_shape)
            wav = wav[..., :actual_wav_len].squeeze(0)
            results.append(wav)

        return results

    @property
    def is_ready(self) -> bool:
        return self.enabled and self._warmed_up and len(self.graphs) > 0
