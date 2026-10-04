# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright 2025 The Qwen team.
"""Inference-only Qwen3-Omni-Moe Code2Wav model."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeCode2WavConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCausalConvNet,
    Qwen3OmniMoeCausalTransConvNet,
    Qwen3OmniMoeCode2WavDecoderBlock,
    Qwen3OmniMoeCode2WavTransformerModel,
    Qwen3OmniMoeConvNeXtBlock,
    Qwen3OmniMoeSnakeBeta,
)
from vllm.config import VllmConfig  # type: ignore
from vllm.logger import init_logger  # type: ignore
from vllm.model_executor.models.utils import (  # type: ignore
    AutoWeightsLoader,
    WeightsMapper,
)

from vllm_omni.model_executor.models.common.snake_activation import SnakeBeta
from vllm_omni.model_executor.models.qwen3_omni.quantization import (
    Qwen3OmniNestedSupportsQuant,
)
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

# Fixed cost of one streaming decode call (graph replay or eager launch), in
# decoded-frame equivalents: a call on a single frame costs about as much GPU
# time as decoding this many extra frames inside a larger call.
_DECODE_CALL_COST_FRAMES = 32


def use_fused_snake(module: nn.Module) -> int:
    """Replace HF ``Qwen3OmniMoeSnakeBeta`` activations under ``module`` with the fused one.

    The HF decoder blocks recompute ``exp(alpha)`` / ``exp(beta)`` and launch
    about seven broadcast kernels per call; the shared ``SnakeBeta`` computes the
    same ``x + 1/b * sin^2(a*x)`` in one kernel from precomputed caches.
    Parameter names and shapes match, so this runs before weight loading.
    """
    count = 0
    for parent in list(module.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, Qwen3OmniMoeSnakeBeta):
                setattr(parent, name, SnakeBeta(child.in_features))
                count += 1
    return count


def plan_decode_groups(
    lengths: Sequence[int],
    bucket_of: Callable[[int], int | None],
    batch_sizes_for: Callable[[int], Sequence[int]],
    call_cost_frames: int = _DECODE_CALL_COST_FRAMES,
) -> list[tuple[list[int], int]]:
    """Split a streaming batch into decode calls of rows with similar lengths.

    Decoding every row at the batch's longest window wastes most of the work
    when short ramp chunks (1-15 frames) share a step with steady 50-frame
    windows. Rows are sorted by length and cut into contiguous groups of
    graph buckets (``bucket_of``) so that the modelled cost, per call
    ``call_cost_frames`` plus padded rows (to the next captured row count of
    that bucket, ``batch_sizes_for``) times the bucket length, is smallest.
    Returns ``(row indices, decode length)`` per call; the decode length is
    the group's longest row.
    """
    count = len(lengths)
    if count == 0:
        return []
    order = sorted(range(count), key=lambda row: lengths[row])
    buckets = [bucket_of(int(lengths[row])) or int(lengths[row]) for row in order]
    distinct = sorted(set(buckets))
    members = [[row for row, bucket in zip(order, buckets) if bucket == size] for size in distinct]

    def sizes_for(size: int) -> list[int]:
        return sorted({int(b) for b in batch_sizes_for(size) if int(b) > 0} | {1})

    def cost(rows: int, size: int) -> int:
        batch_sizes = sizes_for(size)
        calls, rest = divmod(rows, batch_sizes[-1])
        padded = calls * batch_sizes[-1]
        if rest:
            calls += 1
            padded += next(b for b in batch_sizes if b >= rest)
        return calls * call_cost_frames + padded * size

    best = [0] + [None] * len(distinct)
    cut = [0] * (len(distinct) + 1)
    for end in range(1, len(distinct) + 1):
        rows = 0
        for start in range(end - 1, -1, -1):
            rows += len(members[start])
            total = best[start] + cost(rows, distinct[end - 1])
            if best[end] is None or total < best[end]:
                best[end], cut[end] = total, start
    groups: list[tuple[list[int], int]] = []
    end = len(distinct)
    while end > 0:
        start = cut[end]
        groups.append(([row for bucket_rows in members[start:end] for row in bucket_rows], distinct[end - 1]))
        end = start
    calls: list[tuple[list[int], int]] = []
    for rows, size in reversed(groups):
        max_batch = sizes_for(size)[-1]
        for offset in range(0, len(rows), max_batch):
            part = rows[offset : offset + max_batch]
            calls.append((part, max(int(lengths[row]) for row in part)))
    return calls


class Qwen3OmniMoeCode2Wav(nn.Module, Qwen3OmniNestedSupportsQuant):
    """
    Qwen3 Omni MoE Code2Wav - Converts num_quantizers-layer RVQ codec codes to audio waveform.

    Architecture:
    1. Code Embedding: Embed and average num_quantizers RVQ layers
    2. Pre-Transformer: Add temporal context via sliding-window attention
    3. Upsampling: Progressive upsampling with ConvNeXt blocks
    4. Decoder: Multi-stage upsampling + residual units → waveform

    Input: [batch, num_quantizers, seq_len] - num_quantizers-layer RVQ codes
    Output: [batch, 1, waveform_len] - Audio waveform [-1, 1]

    Total upsampling factor: ~1280x
    Example: 100 codec frames → 128,000 audio samples (8 seconds at 16kHz)
    """

    input_modalities = "audio"

    # Weight mapper
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "code2wav.pre_transformer.": "pre_transformer.",
            "code2wav.code_embedding.": "code_embedding.",
            "code2wav.upsample.": "upsample.",
            "code2wav.decoder.": "decoder.",
            "code2wav.": "",
        }
    )

    def __init__(
        self,
        *,
        vllm_config: VllmConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.config: Qwen3OmniMoeCode2WavConfig = vllm_config.model_config.hf_config
        connector = getattr(vllm_config.model_config, "stage_connector_config", None)
        extra = connector.get("extra", {}) if isinstance(connector, dict) else getattr(connector, "extra", {})
        # Both the separate stage and the Talker's first-frame copy receive
        # this connector config. Keep algorithm selection consistent between
        # them, and leave other models' cuDNN settings unchanged.
        self._cudnn_benchmark = (extra or {}).get("codec_cudnn_benchmark", False)
        if not isinstance(self._cudnn_benchmark, bool):
            raise ValueError("codec_cudnn_benchmark must be a boolean")

        # Calculate total upsampling factor
        self.total_upsample = np.prod(self.config.upsample_rates + self.config.upsampling_ratios)

        # Pre-transformer
        self.pre_transformer = Qwen3OmniMoeCode2WavTransformerModel._from_config(self.config)

        # Code embedding: Single embedding table for all RVQ layers
        self.code_embedding = nn.Embedding(
            self.config.codebook_size * self.config.num_quantizers, self.config.hidden_size
        )

        # Offset for each RVQ layer (layer 0: 0-1023, layer 1: 1024-2047, etc.)
        self.register_buffer(
            "code_offset",
            torch.arange(self.config.num_quantizers).view(1, -1, 1) * self.config.codebook_size,
            persistent=False,
        )

        # Upsampling blocks (e.g., 2x, 2x)
        upsample = []
        for factor in self.config.upsampling_ratios:
            upsample.append(
                nn.ModuleList(
                    [
                        Qwen3OmniMoeCausalTransConvNet(
                            self.config.hidden_size, self.config.hidden_size, factor, factor
                        ),
                        Qwen3OmniMoeConvNeXtBlock(self.config.hidden_size),
                    ]
                )
            )
        self.upsample = nn.ModuleList(upsample)

        # Decoder: Initial projection + progressive upsampling blocks
        decoder = [Qwen3OmniMoeCausalConvNet(self.config.hidden_size, self.config.decoder_dim, kernel_size=7)]

        # Add decoder blocks (each upsamples and reduces channels)
        for i in range(len(self.config.upsample_rates)):
            decoder.append(Qwen3OmniMoeCode2WavDecoderBlock(self.config, i))

        # Final projection to waveform
        output_dim = self.config.decoder_dim // 2 ** len(self.config.upsample_rates)
        decoder += [
            SnakeBeta(output_dim),
            Qwen3OmniMoeCausalConvNet(output_dim, 1, kernel_size=7),
        ]
        self.decoder = nn.ModuleList(decoder)
        # Keep existing V1 and non-CUDA decoder blocks unchanged.
        if (
            (extra or {}).get("codec_fused_snake", False)
            and current_omni_platform.is_cuda()
            and torch.device(vllm_config.device_config.device).type == "cuda"
        ):
            use_fused_snake(self.decoder)

        # CUDA Graph support — reuses CUDAGraphDecoderWrapper from Qwen3-TTS
        self._cudagraph_enabled = False
        self._cudagraph_wrapper = None
        # Captured row counts per graph size; empty keeps one padded decode per batch.
        self._streaming_batch_sizes: dict[int, list[int]] = {}

    def precompute_snake_caches(self):
        """Precompute exp(alpha) and 1/(exp(beta)+eps) for all SnakeBeta modules."""
        count = 0
        for module in self.modules():
            if isinstance(module, SnakeBeta):
                module.precompute_exp_cache()
                count += 1
        if count > 0:
            logger.info("Precomputed exp caches for %d SnakeBeta activations", count)

    def enable_cudagraph(
        self,
        device: torch.device | None = None,
        codec_chunk_frames: int = 0,
        codec_left_context_frames: int = 0,
        extra_capture_sizes: Iterable[int] = (),
        streaming_batch_sizes: Iterable[int] = (),
    ):
        """Enable CUDA graph acceleration (same pattern as Qwen3-TTS Code2Wav).

        ``extra_capture_sizes`` adds exact-size graphs (e.g. a chunk ramp's
        decode windows) to the default buckets, so those chunks replay without
        zero padding. ``streaming_batch_sizes`` (e.g. ``[2, 4, 8, 16]``) also
        captures multi-row graphs for every bucket up to one streaming window
        (chunk + left context); a streaming batch is then decoded as a few
        length groups (``plan_decode_groups``) instead of one eager call
        padded to its longest row.
        """
        from vllm_omni.model_executor.models.qwen3_tts.cuda_graph_decoder_wrapper import (
            CUDAGraphDecoderWrapper,
        )

        if device is None:
            device = next(self.parameters()).device
        if device.type != "cuda":
            logger.warning("Cannot enable CUDA Graph: not on CUDA device (got %s)", device)
            return

        extra = {int(size) for size in extra_capture_sizes if int(size) > 0}
        batch_sizes = sorted({int(b) for b in streaming_batch_sizes if int(b) > 1})
        capture_sizes = None
        extra_shapes: list[tuple[int, int]] = []
        if extra or batch_sizes:
            capture_sizes = sorted(
                set(
                    CUDAGraphDecoderWrapper.compute_capture_sizes(
                        codec_chunk_frames=codec_chunk_frames,
                        codec_left_context_frames=codec_left_context_frames,
                    )
                )
                | extra
            )
            window = codec_chunk_frames + codec_left_context_frames
            streaming_sizes = [size for size in capture_sizes if window <= 0 or size <= window]
            # A multi-row graph never holds more frames than the largest
            # single-row one, so the shared graph pool does not grow.
            largest = max(capture_sizes)
            extra_shapes = [(b, size) for b in batch_sizes for size in streaming_sizes if b * size <= largest]
        wrapper = CUDAGraphDecoderWrapper(
            decoder=self,
            capture_sizes=capture_sizes,
            extra_capture_shapes=extra_shapes,
            num_quantizers=self.config.num_quantizers,
            enabled=True,
        )
        try:
            wrapper.warmup(
                device,
                dtype=torch.long,
                codec_chunk_frames=codec_chunk_frames,
                codec_left_context_frames=codec_left_context_frames,
            )
        except Exception:
            self._cudagraph_wrapper = None
            self._cudagraph_enabled = False
            raise
        self._cudagraph_wrapper = wrapper
        self._cudagraph_enabled = True
        if extra_shapes and torch.cuda.is_available():
            # Return the eager warm-up activations: this stage shares the GPU.
            torch.accelerator.empty_cache()
        self._streaming_batch_sizes = {
            size: [1, *sorted(b for b, shape_size in extra_shapes if shape_size == size)]
            for size in {shape_size for _b, shape_size in extra_shapes}
        }
        logger.info(
            "CUDA Graph enabled for Code2Wav: num_quantizers=%d, sizes=%s, multi-row shapes=%s",
            self.config.num_quantizers,
            self._cudagraph_wrapper.capture_sizes,
            extra_shapes,
        )

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Convert num_quantizers-layer RVQ codes to audio waveform.

        Args:
            codes: [batch, num_quantizers, seq_len] - num_quantizers-layer RVQ codec codes

        Returns:
            waveform: [batch, 1, waveform_len] - Audio waveform clipped to [-1, 1]
        """
        if codes.shape[1] != self.config.num_quantizers:
            raise ValueError(f"Expected {self.config.num_quantizers} layers of codes, got {codes.shape[1]}")

        # Stage 1: Code Embedding
        # Add offset to separate layer vocabularies, then embed and average
        embedded_codes = self.code_embedding(codes + self.code_offset)
        # vLLM batch-invariant reductions may return FP32 for low-precision
        # input. Preserve the embedding/transformer dtype at this boundary.
        hidden = embedded_codes.mean(1).to(embedded_codes.dtype)
        del embedded_codes
        # Shape: [batch, seq_len, hidden_size]

        # Stage 2: Pre-Transformer (add temporal context)
        hidden = self.pre_transformer(inputs_embeds=hidden).last_hidden_state
        # Shape: [batch, seq_len, hidden_size]

        # Algorithm search happens during eager warmup before graph capture.
        # Restore process settings afterward, including on a failed decode.
        # Captured replays use the selected kernels without this Python path.
        cudnn = torch.backends.cudnn
        context = (
            cudnn.flags(
                enabled=cudnn.enabled,
                benchmark=True,
                benchmark_limit=10,
                deterministic=cudnn.deterministic,
                allow_tf32=cudnn.allow_tf32,
            )
            if (
                self._cudnn_benchmark
                and hidden.is_cuda
                and torch.version.hip is None
                and not torch.cuda.is_current_stream_capturing()
            )
            else nullcontext()
        )
        with context:
            return self._decode_waveform(hidden)

    def _decode_waveform(self, hidden: torch.Tensor) -> torch.Tensor:
        # Stage 3: Upsampling
        hidden = hidden.permute(0, 2, 1)  # [batch, hidden_size, seq_len]
        for blocks in self.upsample:
            for block in blocks:
                hidden = block(hidden)
        # Shape: [batch, hidden_size, seq_len * upsample_factor]

        # Stage 4: Decoder (progressive upsampling to waveform)
        wav = hidden
        for block in self.decoder:
            wav = block(wav)
        # Shape: [batch, 1, waveform_len]

        # Clamp to valid audio range
        return wav.clamp(min=-1.0, max=1.0)

    def chunked_decode(
        self,
        codes: torch.Tensor,
        chunk_size: int = 300,
        left_context_size: int = 25,
        seq_token_counts: list[int] | None = None,
    ) -> list[torch.Tensor]:
        """
        Decode long sequences in chunks to avoid OOM.

        Uses overlapping chunks with left context to avoid boundary artifacts.
        When CUDA graphs are enabled, delegates chunk-level decoding to the
        CUDAGraphDecoderWrapper for reduced kernel launch overhead.

        Args:
            codes: [batch, num_quantizers, seq_len] - num_quantizers-layer RVQ codes
            chunk_size: Number of codec frames per chunk
            left_context_size: Number of overlapping frames for context
            seq_token_counts: Token count for each request in batch

        Returns:
            list[torch.Tensor]: Complete waveform decoded from the input
                codes. For ``batch_size == 1``, this is a list containing a
                single tensor with shape ``[1, waveform_len]``.
        """
        # Use CUDA graph wrapper for chunk-level decode when available
        if self._cudagraph_enabled and self._cudagraph_wrapper is not None:
            batch_wav = self._cudagraph_wrapper.chunked_decode_with_cudagraph(codes, chunk_size, left_context_size)
        else:
            wavs = []
            start_index = 0

            while start_index < codes.shape[-1]:
                end_index = min(start_index + chunk_size, codes.shape[-1])
                context_size = left_context_size if start_index >= left_context_size else start_index

                # Extract chunk with left context
                codes_chunk = codes[..., start_index - context_size : end_index]

                # Decode chunk
                wav_chunk = self(codes_chunk)

                # Remove context from output (context_size * total_upsample samples)
                wavs.append(wav_chunk[..., context_size * self.total_upsample :])

                start_index = end_index

            batch_wav = torch.cat(wavs, dim=-1)

        if seq_token_counts is not None:
            code_seq_lens = [seq_len // self.config.num_quantizers for seq_len in seq_token_counts]
        else:
            # Fallback: assume all batch elements share the same sequence length.
            code_seq_lens = [codes.shape[-1]] * codes.shape[0]
        result = []
        for idx, code_seq_len in enumerate(code_seq_lens):
            wav_chunk = batch_wav[idx, :, : code_seq_len * self.total_upsample]
            result.append(wav_chunk)
        return result

    def chunked_decode_streaming(
        self,
        codes: torch.Tensor,
        left_context_size: list[int] | None = None,
        seq_token_counts: list[int] | None = None,
    ) -> list[torch.Tensor]:
        """
        Decode long sequences in chunks to avoid OOM.

        Uses overlapping chunks with left context to avoid boundary artifacts.

        No longer need chunk size here, which is different from chunked_decode

        Args:
            codes: [batch, num_quantizers, seq_len] - num_quantizers-layer RVQ codes
            left_context_size: Number of overlapping frames for context
            seq_token_counts: Token count for each request in batch

        Returns:
            list[torch.Tensor]: Complete waveform decoded from the input
                codes. For ``batch_size == 1``, this is a list containing a
                single tensor with shape ``[1, waveform_len]``.
        """
        if not (left_context_size and seq_token_counts and len(left_context_size) == len(seq_token_counts)):
            logger.warning_once(
                "chunked_decode_streaming: missing/invalid left_context_size or seq_token_counts; "
                "defaulting to left_context_size=zeros(len(codes)). This is expected during cudagraph warmup."
            )
            left_context_size = [0] * codes.shape[0]
        if seq_token_counts is not None:
            code_seq_lens = [n // self.config.num_quantizers for n in seq_token_counts]
        else:
            # Fallback: assume all batch elements share the same sequence length.
            code_seq_lens = [codes.shape[-1]] * codes.shape[0]
        batch, width = int(codes.shape[0]), int(codes.shape[-1])
        graphs = self._cudagraph_wrapper if self._cudagraph_enabled else None
        batch_sizes = getattr(self, "_streaming_batch_sizes", None)
        if graphs is not None and batch_sizes and batch > 1:
            groups = plan_decode_groups(
                code_seq_lens, graphs._get_padded_size, lambda size: batch_sizes.get(size, (1,))
            )
        else:
            groups = [(list(range(batch)), width)]
        wavs: list[torch.Tensor | None] = [None] * batch
        for rows, length in groups:
            if len(rows) == batch and length == width:
                group_codes = codes
            else:
                # Basic indexing only: a device index tensor would need a host copy.
                group_codes = torch.stack([codes[row, :, :length] for row in rows])
            batch_wav = graphs.decode(group_codes) if graphs is not None else self(group_codes)
            # The decoder trims the same right-edge tail from every batch row.
            # Infer it from the decoded window, not a shorter request's length:
            # padding that request with codec zeros does not provide valid context.
            tail = max(0, int(length * self.total_upsample) - batch_wav.shape[-1])
            for position, row in enumerate(rows):
                # Refill the previous chunk's tail using the re-decoded left context,
                # and exclude the current row's padded tail just as singleton decode does.
                start = max(0, left_context_size[row] * self.total_upsample - tail)
                end = max(0, code_seq_lens[row] * self.total_upsample - tail)
                wavs[row] = batch_wav[position, :, start:end]
        return wavs

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from HuggingFace checkpoint."""
        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(
            weights,
            mapper=(self.hf_to_vllm_mapper) | WeightsMapper(orig_to_new_prefix={"thinker.": None, "talker.": None}),
        )

        # Log load summary
        try:
            total_bytes = 0
            for name, param in self.named_parameters():
                if param is not None and param.data is not None:
                    total_bytes += param.data.numel() * param.data.element_size()
            device = next(self.parameters()).device
            logger.info(
                "[Model Loaded] name=%s, success=%s, size=%.2f MB, device=%s",
                self.__class__.__name__,
                True,
                total_bytes / (1024**2),
                str(device),
            )
        except Exception:
            logger.error("Error logging model load summary")

        return loaded
