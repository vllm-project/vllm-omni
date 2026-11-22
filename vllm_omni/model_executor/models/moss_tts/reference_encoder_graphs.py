# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA graphs for batched MOSS reference-audio encoding.

Reference clips are encoded by the upstream MOSS audio tokenizer that the
processor loads. Its eager forward is dominated by kernel launches for the
small batches that serving produces. This runner replays graphs of the same
encoder and quantizer modules, captured per ``(batch, length)`` bucket. For a
given padded input the codes are bit-identical to the tokenizer's own
``_encode_frame``. Only the per-clip crop, which needs host values, runs
outside the graph.

The encoder is causal, so padding a clip to its length bucket leaves its valid
frames unchanged up to kernel selection for the padded shape, the same effect
that batching already has when the tokenizer pads a batch to its longest clip.
"""

from __future__ import annotations

import math
import types
from dataclasses import dataclass

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

DEFAULT_BATCH_SIZES = (1, 2, 4, 8, 16, 32)
DEFAULT_BUCKET_SECONDS = (4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0)
# Fine steps keep padding small: a padded row or padded time is encoded in
# full. Worth the extra graphs where several encoders share one GPU.
FINE_BATCH_SIZES = (1, 2, 3, 4, 5, 6, 8)
FINE_BUCKET_SECONDS = (3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0)
# Graphs are captured only while batch * bucket stays within this much audio,
# which bounds the shared graph memory pool by its largest batch.
DEFAULT_MAX_BATCH_SECONDS = 256.0


@torch.compile(dynamic=False)
def _reference_rope_factors(seqlen: int, head_dim: int, max_period: float, device: torch.device):
    # Use the same compiled FP32 arithmetic as the encoder. Eager libdevice
    # sin/cos and compiled sin/cos can round differently near a VQ boundary.
    ds = torch.arange(head_dim // 2, device=device, dtype=torch.float32)
    freqs = torch.exp(ds * (-math.log(max_period) * 2 / head_dim))
    angles = torch.arange(seqlen, device=device, dtype=torch.float32)[:, None] * freqs[None, :]
    return torch.cos(angles), torch.sin(angles)


def _cached_reference_rope(self, q: torch.Tensor, k: torch.Tensor, cache: dict):
    if self.rope is None:
        return q, k
    batch, heads, seqlen, dim = q.shape
    key = (dim, self.rope.max_period, q.device)
    if key not in cache or cache[key][0].shape[0] < seqlen:
        cache[key] = _reference_rope_factors(seqlen, dim, self.rope.max_period, q.device)
    cos, sin = cache[key]
    cos, sin = cos[:seqlen], sin[:seqlen]
    qr, qi = q.view(batch, heads, seqlen, dim // 2, 2).unbind(-1)
    kr, ki = k.view(batch, heads, seqlen, dim // 2, 2).unbind(-1)
    qr, qi, kr, ki = qr.float(), qi.float(), kr.float(), ki.float()
    # Preserve the original arithmetic, casts, and B/H/T/D output layout.
    qo = torch.stack(((qr * cos - qi * sin).to(q.dtype), (qr * sin + qi * cos).to(q.dtype)), dim=-1)
    ko = torch.stack(((kr * cos - ki * sin).to(k.dtype), (kr * sin + ki * cos).to(k.dtype)), dim=-1)
    return qo.view(batch, heads, seqlen, dim), ko.view(batch, heads, seqlen, dim)


def _reference_flash_attn_varlen_func():
    """Resolve the platform's variable-length FlashAttention entrypoint.

    vLLM's bundled extension is NVIDIA-only. ROCm exposes the compatible
    unpacked-QKV API through AITER, with upstream flash-attn as its fallback.
    """
    if torch.version.hip is not None:
        try:
            from aiter import flash_attn_varlen_func

            return flash_attn_varlen_func, True
        except ImportError:
            from flash_attn import flash_attn_varlen_func
    else:
        from vllm.vllm_flash_attn import flash_attn_varlen_func
    return flash_attn_varlen_func, False


def _windowed_attention(
    self,
    x: torch.Tensor,
    input_lengths: torch.Tensor,
    *,
    sdpa,
    fa_version: int = 2,
    rope_cache: dict | None = None,
    skip_padded_query_mask: bool = False,
) -> torch.Tensor:
    """The tokenizer's non-streaming attention with a local-window flash kernel.

    Same projection, RoPE and padded-row zeroing as the tokenizer's SDPA path.
    That path masks each query to ``context`` keys (distance 0 .. context-1)
    inside a dense block; the flash kernel visits only that window.
    """
    batch, seqlen, _ = x.shape
    q, k, v = self._project_qkv(x)
    q, k = self._apply_dense_rope(q, k) if rope_cache is None else _cached_reference_rope(self, q, k, rope_cache)
    if q.dtype not in (torch.bfloat16, torch.float16):
        return sdpa(x, input_lengths)
    flash_attn_varlen_func, is_aiter = _reference_flash_attn_varlen_func()

    heads, dim = q.shape[1], q.shape[-1]

    def packed(t: torch.Tensor) -> torch.Tensor:
        return t.transpose(1, 2).reshape(batch * seqlen, heads, dim)

    starts = torch.arange(0, (batch + 1) * seqlen, seqlen, device=x.device, dtype=torch.int32)
    window = [self.context - 1, 0] if self.causal and self.context is not None else [-1, -1]
    kwargs = {}
    if torch.version.hip is None:
        # Only vLLM's NVIDIA wrapper accepts its backend-selection argument.
        kwargs["fa_version"] = fa_version
    elif is_aiter and torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        # AITER's autograd wrapper requires LSE to be retained for backward.
        # The inference path avoids this allocation, and upstream flash-attn
        # manages its own autograd state without this AITER-specific keyword.
        kwargs["return_lse"] = True
    result = flash_attn_varlen_func(
        packed(q),
        packed(k),
        packed(v),
        max_seqlen_q=seqlen,
        cu_seqlens_q=starts,
        max_seqlen_k=seqlen,
        cu_seqlens_k=starts,
        causal=self.causal,
        window_size=window,
        **kwargs,
    )
    out = result[0] if isinstance(result, tuple) else result
    out = out.view(batch, seqlen, heads, dim)
    if not skip_padded_query_mask:
        valid = (torch.arange(seqlen, device=x.device).view(1, seqlen) < input_lengths.view(-1, 1)).view(
            batch, seqlen, 1, 1
        )
        out = torch.where(valid, out, torch.zeros((), device=out.device, dtype=out.dtype))
    return out.reshape(batch, seqlen, self.embed_dim)


def _validate_unmasked_reference_encoder(tokenizer: torch.nn.Module) -> None:
    """Restrict the optimization to the audited HF encoder's valid-prefix contract.

    Causal attention and pointwise transformer operations cannot read future
    padded rows. Its patch-downsample floors lengths, so every valid output
    group consists entirely of valid input rows. The quantizer is pointwise
    and the caller crops codes by those lengths. Padded *outputs* may differ.
    This argument does not apply to the streaming decoder or to upsampling.
    """
    transformers = 0
    for module in tokenizer.encoder:
        name = type(module).__name__
        if name == "MossAudioTokenizerPatchedPretransform" and module.is_downsample:
            continue
        if name == "MossAudioTokenizerProjectedTransformer":
            layers = module.transformer.layers
            if layers and all(layer.self_attn.causal for layer in layers):
                transformers += 1
                continue
        raise ValueError("Unmasked reference queries require the causal MOSS patch-downsample encoder")
    if not transformers:
        raise ValueError("Unmasked reference queries require at least one causal MOSS transformer")
    if type(tokenizer.quantizer).__name__ not in {
        "MossAudioTokenizerResidualVQ",
        "MossAudioTokenizerResidualLFQ",
    }:
        raise ValueError("Unmasked reference queries require the pointwise MOSS quantizer")
    for module in tokenizer.quantizer.modules():
        if isinstance(module, torch.nn.Conv1d) and module.kernel_size != (1,):
            raise ValueError("Unmasked reference queries do not support temporal quantizer convolutions")


def install_windowed_attention(
    tokenizer: torch.nn.Module,
    *,
    fa_version: int = 2,
    cache_rope: bool = False,
    skip_padded_query_mask: bool = False,
) -> int:
    """Route the encoder's non-streaming attention through a local-window flash kernel."""
    if skip_padded_query_mask:
        _validate_unmasked_reference_encoder(tokenizer)
        logger.info("MOSS reference encoder: skipping padded query masks (valid-prefix outputs only)")
    count = 0
    rope_cache = {} if cache_rope else None
    if cache_rope:
        tokenizer._moss_reference_rope_cache = rope_cache
        tokenizer._moss_reference_rope_prepared_samples = 0
    for module in tokenizer.encoder.modules():
        sdpa = getattr(module, "_forward_non_streaming_sdpa", None)
        if sdpa is None or not hasattr(module, "_project_qkv") or not getattr(module, "causal", False):
            continue

        def attention(self, x, input_lengths, _sdpa=sdpa):
            return _windowed_attention(
                self,
                x,
                input_lengths,
                sdpa=_sdpa,
                fa_version=fa_version,
                rope_cache=rope_cache,
                skip_padded_query_mask=skip_padded_query_mask,
            )

        module._forward_non_streaming_sdpa = types.MethodType(attention, module)
        count += 1
    return count


@dataclass
class _Graph:
    graph: torch.cuda.CUDAGraph
    audio: torch.Tensor  # (B, C, T) float32 input
    lengths: torch.Tensor  # (B,) int64 input, in samples per channel
    codes: torch.Tensor  # (n_vq, B, frames) output
    code_lengths: torch.Tensor  # (B,) output


class MossReferenceEncoderGraphs:
    """Encode prepared reference waveforms with captured tokenizer graphs."""

    def __init__(
        self,
        tokenizer: torch.nn.Module,
        *,
        n_vq: int,
        batch_sizes: tuple[int, ...] = DEFAULT_BATCH_SIZES,
        bucket_seconds: tuple[float, ...] = DEFAULT_BUCKET_SECONDS,
        max_batch_seconds: float = DEFAULT_MAX_BATCH_SECONDS,
        compile_core: bool = False,
        batched_transfer: bool = False,
        singleton_bucket_seconds: tuple[float, ...] | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._compile_core = compile_core
        self._batched_transfer = batched_transfer
        self._n_vq = int(n_vq)
        self._device = next(tokenizer.parameters()).device
        self._channels = int(tokenizer.number_channels)
        frame = int(tokenizer.downsample_rate)
        sampling_rate = int(tokenizer.sampling_rate)
        # Whole frames, so the tokenizer's own frame padding is a no-op.
        self._buckets = sorted({int(math.ceil(s * sampling_rate / frame)) * frame for s in bucket_seconds})
        self._singleton_buckets = sorted(
            set(self._buckets)
            | {int(math.ceil(s * sampling_rate / frame)) * frame for s in (singleton_bucket_seconds or ())}
        )
        self._batch_sizes = sorted({int(b) for b in batch_sizes if int(b) > 0})
        self._max_batch_samples = float(max_batch_seconds) * sampling_rate
        self._graphs: dict[tuple[int, int], _Graph] = {}
        self._pool = None
        self._transfer_buffers = None

    @property
    def max_batch(self) -> int:
        return self._batch_sizes[-1]

    @property
    def captured(self) -> list[tuple[int, int]]:
        return sorted(self._graphs)

    def _core(self, audio: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Mirrors the tokenizer's _encode_frame without its host-side crop.
        tokenizer = self._tokenizer
        values, value_lengths = tokenizer._flatten_channels_for_codec(audio, lengths)
        with tokenizer._codec_inference_autocast():
            hidden, hidden_lengths = values, value_lengths
            for module in tokenizer.encoder:
                hidden, hidden_lengths = module(hidden, hidden_lengths)
        _, codes, code_lengths = tokenizer.quantizer(hidden.float(), hidden_lengths, self._n_vq)
        return codes, code_lengths

    @torch.no_grad()
    def capture(self) -> None:
        if self._device.type != "cuda" or self._graphs:
            return
        self._pool = torch.cuda.graph_pool_handle()
        stream = torch.cuda.Stream(device=self._device)
        prepared = getattr(self._tokenizer, "_moss_reference_rope_prepared_samples", None)
        max_samples = max(self._singleton_buckets)
        if prepared is not None and prepared < max_samples:
            # Prime immutable FP32 factors at the largest length before any
            # capture or compilation. Smaller lengths use a prefix of the
            # same table; the dictionary stays fixed throughout capture and
            # workers share tables without introducing dynamic-shape guards
            # for a separate dictionary key at each sequence length.
            audio = torch.zeros(1, self._channels, max_samples, device=self._device)
            lengths = torch.full((1,), max_samples, dtype=torch.long, device=self._device)
            self._core(audio, lengths)
            self._tokenizer._moss_reference_rope_prepared_samples = max_samples
        core = self._core
        if self._compile_core:
            # Fuses the encoder's many small elementwise kernels (casts, RoPE,
            # residuals). Only capture runs the compiled code; replays do not.
            core = torch.compile(self._core, dynamic=True)
        # Each encoder stage is its own module shape: allow a compile per stage.
        with torch._dynamo.config.patch(recompile_limit=64):
            self._capture_all(core, stream)
        if self._batched_transfer and self._graphs:
            # One reusable slab per worker, rather than one per graph bucket.
            entries = list(self._graphs.values())
            self._transfer_buffers = (
                torch.empty(max(e.audio.numel() for e in entries), dtype=entries[0].audio.dtype, pin_memory=True),
                torch.empty(max(e.lengths.numel() for e in entries), dtype=entries[0].lengths.dtype, pin_memory=True),
                torch.empty(max(e.codes.numel() for e in entries), dtype=entries[0].codes.dtype, pin_memory=True),
                torch.empty(
                    max(e.code_lengths.numel() for e in entries), dtype=entries[0].code_lengths.dtype, pin_memory=True
                ),
                torch.cuda.Event(),
            )
        logger.info(
            "MOSS reference encoder CUDA graphs: batch=%s buckets(s)=%s compiled=%s captured=%d",
            self._batch_sizes,
            [round(s / int(self._tokenizer.sampling_rate), 2) for s in self._buckets],
            self._compile_core,
            len(self._graphs),
        )
        if self._singleton_buckets != self._buckets:
            logger.info(
                "MOSS reference encoder singleton buckets(s): %s",
                [round(s / int(self._tokenizer.sampling_rate), 2) for s in self._singleton_buckets],
            )

    def _capture_all(self, core, stream: torch.cuda.Stream) -> None:
        for batch in reversed(self._batch_sizes):
            buckets = self._singleton_buckets if batch == 1 else self._buckets
            for samples in reversed(buckets):
                if batch > 1 and batch * samples > self._max_batch_samples:
                    continue
                audio = torch.zeros(batch, self._channels, samples, device=self._device)
                lengths = torch.full((batch,), samples, dtype=torch.long, device=self._device)
                try:
                    stream.wait_stream(torch.cuda.current_stream(self._device))
                    with torch.cuda.stream(stream):
                        # Lazy library setup must happen before capture; one run does it.
                        core(audio, lengths)
                    torch.cuda.current_stream(self._device).wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, pool=self._pool, stream=stream):
                        codes, code_lengths = core(audio, lengths)
                except Exception:
                    logger.warning(
                        "MOSS reference encoder graph (B=%d, samples=%d) not captured; using eager",
                        batch,
                        samples,
                        exc_info=True,
                    )
                    continue
                self._graphs[(batch, samples)] = _Graph(graph, audio, lengths, codes, code_lengths)

    @torch.no_grad()
    def encode(self, wavs: list[torch.Tensor]) -> list[torch.Tensor | None]:
        """Codes ``(frames, n_vq)`` int64 on CPU per clip; None where no graph fits.

        ``wavs`` are prepared the way the processor prepares them: ``(C, T)``
        float32 after channel handling, resampling and loudness normalization.
        Clips are grouped longest first, so each replay pads to the bucket of
        similar-length clips.
        """
        out: list[torch.Tensor | None] = [None] * len(wavs)
        order = sorted(range(len(wavs)), key=lambda i: -int(wavs[i].shape[-1]))
        pos = 0
        while pos < len(order):
            longest = int(wavs[order[pos]].shape[-1])
            # Fine singleton buckets reduce padding without fragmenting an
            # already assembled batch into separate graph replays.
            buckets = self._singleton_buckets if len(order) - pos == 1 else self._buckets
            samples = next((s for s in buckets if s >= longest and (1, s) in self._graphs), None)
            if samples is None:
                pos += 1  # longer than every bucket: left to the caller
                continue
            widest = max(b for b in self._batch_sizes if (b, samples) in self._graphs)
            rows = order[pos : pos + widest]
            batch = next(b for b in self._batch_sizes if b >= len(rows) and (b, samples) in self._graphs)
            for row, codes in zip(rows, self._replay(self._graphs[(batch, samples)], [wavs[i] for i in rows])):
                out[row] = codes
            pos += len(rows)
        return out

    def _replay(self, entry: _Graph, wavs: list[torch.Tensor]) -> list[torch.Tensor]:
        if self._transfer_buffers is not None and all(w.device.type == "cpu" for w in wavs):
            return self._replay_batched_transfer(entry, wavs)
        count, batch, samples = len(wavs), entry.audio.shape[0], entry.audio.shape[-1]
        # Unused rows encode full-length silence: a fully masked row would be
        # all-NaN inside attention, and rows never mix.
        entry.audio.zero_()
        lengths = [int(w.shape[-1]) for w in wavs] + [samples] * (batch - count)
        entry.lengths.copy_(torch.tensor(lengths, dtype=torch.long))
        for row, wav in enumerate(wavs):
            entry.audio[row, :, : wav.shape[-1]].copy_(wav)
        entry.graph.replay()
        codes = entry.codes[:, :count].cpu()
        code_lengths = entry.code_lengths[:count].tolist()
        return [codes[:, row, : int(code_lengths[row])].transpose(0, 1).contiguous().long() for row in range(count)]

    def _replay_batched_transfer(self, entry: _Graph, wavs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Stage one batch on pinned host memory, then wait once for its outputs."""
        audio_slab, lengths_slab, codes_slab, code_lengths_slab, copied = self._transfer_buffers
        count, batch, samples = len(wavs), entry.audio.shape[0], entry.audio.shape[-1]
        audio = audio_slab[: entry.audio.numel()].view_as(entry.audio)
        lengths = lengths_slab[:batch]
        audio.zero_()
        lengths.fill_(samples)
        for row, wav in enumerate(wavs):
            audio[row, :, : wav.shape[-1]].copy_(wav)
            lengths[row] = wav.shape[-1]
        entry.audio.copy_(audio, non_blocking=True)
        entry.lengths.copy_(lengths, non_blocking=True)
        entry.graph.replay()
        result = entry.codes[:, :count]
        codes = codes_slab[: result.numel()].view_as(result)
        code_lengths = code_lengths_slab[:count]
        codes.copy_(result, non_blocking=True)
        code_lengths.copy_(entry.code_lengths[:count], non_blocking=True)
        copied.record(torch.cuda.current_stream(self._device))
        copied.synchronize()
        # Returned codes must outlive this worker's next replay, including
        # one-frame clips whose transpose is already contiguous.
        return [
            codes[:, row, : int(code_lengths[row])].transpose(0, 1).to(dtype=torch.long, copy=True).contiguous()
            for row in range(count)
        ]
