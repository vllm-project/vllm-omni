"""CUDA Graph acceleration for the MOSS Audio Tokenizer codec decoder.

Captures the codec decode method (_decode_frame, or _decode on the vendored codec) for a set of fixed frame-count
bucket sizes, then replays the captured graph at inference time to eliminate
kernel-launch overhead.  Inputs that exceed all captured sizes fall back to
eager execution transparently.
"""

from __future__ import annotations

import time

import torch
from torch.cuda import CUDAGraph
from vllm.logger import init_logger
from vllm.platforms import current_platform

from vllm_omni.model_executor.models.moss_tts.audio_tokenizer import (
    MossAudioTokenizerDecoderOutput,
    MossAudioTokenizerModel,
)

logger = init_logger(__name__)


class MossTTSCUDAGraphCodecWrapper:
    """CUDA Graph wrapper for the codec decode method (_decode_frame, or _decode on the vendored codec).

    Graphs are keyed by padded_T (int).  On each call the actual T is
    bucket-matched to the smallest pre-captured size >= T.  The static code
    buffer [NQ, 1, padded_T] is filled left-aligned (right-zero-padded) and
    the graph is replayed.  The output audio is sliced to the correct length
    by scaling from the captured audio shape (actual_T / padded_T * captured_len),
    avoiding any assumption about downsample_rate vs effective decoder upsample.
    The slice is cloned before returning so the static buffer can be reused.

    Usage::

        wrapper = MossTTSCUDAGraphCodecWrapper(codec_model, capture_sizes, nq)
        wrapper.warmup(device)

        # per-request decode:
        out = wrapper.decode(codes_nq_t)   # codes_nq_t: [NQ, T]
    """

    def __init__(
        self,
        model: MossAudioTokenizerModel,
        capture_sizes: list[int],
        num_quantizers: int,
        enabled: bool = True,
    ) -> None:
        self.model = model
        # Upstream remote-code tokenizer exposes _decode_frame; the vendored
        # MossAudioTokenizerModel still exposes _decode. Resolve once here so
        # both codec implementations work.
        decode_fn = getattr(model, "_decode_frame", None) or getattr(model, "_decode", None)
        if decode_fn is None:
            raise AttributeError(
                f"{type(model).__module__}.{type(model).__name__} exposes neither _decode_frame nor _decode"
            )
        self._decode_fn = decode_fn
        self.capture_sizes: list[int] = sorted(capture_sizes)
        self.num_quantizers = num_quantizers
        self.enabled = enabled

        # All dicts keyed by padded_T.
        self.graphs: dict[int, CUDAGraph] = {}
        self.static_codes: dict[int, torch.Tensor] = {}  # [NQ, 1, padded_T]
        # static_lengths is kept alive here — the captured graph holds a
        # reference to the underlying storage and must not be GC'd.
        self.static_lengths: dict[int, torch.Tensor] = {}  # [1]
        self.static_audio: dict[int, torch.Tensor] = {}  # [1, 1, padded_T * effective_upsample]

        self._warmed_up = False

    # ------------------------------------------------------------------
    # Size helpers
    # ------------------------------------------------------------------

    def _get_padded_size(self, actual_t: int) -> int | None:
        """Return the smallest capture size >= actual_t, or None if too large."""
        for s in self.capture_sizes:
            if actual_t <= s:
                return s
        return None

    # ------------------------------------------------------------------
    # Warmup / capture
    # ------------------------------------------------------------------

    def warmup(self, device: torch.device) -> None:
        """Allocate static buffers and capture CUDA Graphs for all sizes."""
        if device.type != "cuda" or not self.enabled or self._warmed_up:
            return

        nq = self.num_quantizers
        logger.info(
            "MOSS-TTS codec CUDA Graph warmup: nq=%d capture_sizes=%s",
            nq,
            self.capture_sizes,
        )
        t0 = time.perf_counter()

        # One eager run per size to let cuDNN / CUDA allocate memory before
        # the capture window (graph capture forbids new CUDA allocs during it).
        for size in self.capture_sizes:
            dummy_codes = torch.zeros(nq, 1, size, dtype=torch.long, device=device)
            dummy_lengths = torch.tensor([size], dtype=torch.long, device=device)
            with torch.no_grad():
                _ = self._decode_fn(dummy_codes, dummy_lengths)

        torch.accelerator.synchronize(device)

        for size in self.capture_sizes:
            try:
                self._capture(size, device)
                logger.info("  Captured CUDA Graph for size=%d", size)
            except Exception:
                logger.warning(
                    "  Failed to capture CUDA Graph for size=%d; this size will fall back to eager decode",
                    size,
                    exc_info=True,
                )

        self._warmed_up = True
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "MOSS-TTS codec CUDA Graph warmup complete: %d/%d captured in %.1f ms",
            len(self.graphs),
            len(self.capture_sizes),
            elapsed_ms,
        )

    def _capture(self, size: int, device: torch.device) -> None:
        nq = self.num_quantizers
        static_codes = torch.zeros(nq, 1, size, dtype=torch.long, device=device)
        # lengths holds the number of valid code frames; set to full size at
        # capture time so the decoder emits a full-size audio buffer.
        static_lengths = torch.tensor([size], dtype=torch.long, device=device)

        # Extra eager warmup inside capture to ensure all kernels are compiled.
        with torch.no_grad():
            _ = self._decode_fn(static_codes, static_lengths)
        torch.accelerator.synchronize(device)

        graph = CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(graph, pool=current_platform.get_global_graph_pool()):
                static_out = self._decode_fn(static_codes, static_lengths)

        self.graphs[size] = graph
        self.static_codes[size] = static_codes
        self.static_lengths[size] = static_lengths
        # static_out.audio is a static buffer reused every replay; hold a
        # reference so it is not garbage-collected.
        self.static_audio[size] = static_out.audio  # [1, 1, size * effective_upsample]

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def decode(self, codes_nq_t: torch.Tensor) -> MossAudioTokenizerDecoderOutput:
        """Decode [NQ, T] codes to waveform using a CUDA Graph when possible.

        Falls back to eager batch_decode when:
          - CUDA Graph is disabled or not yet warmed up
          - an outer CUDA stream capture is active (e.g. vLLM FULL graph mode)
          - actual T exceeds all pre-captured sizes
        """
        if not self.enabled or not self._warmed_up:
            return self.model.batch_decode(codes_list=[codes_nq_t], num_quantizers=self.num_quantizers)

        # Replaying a graph inside an active stream capture would corrupt the
        # outer graph.  Fall back to eager so the caller can complete its capture.
        if torch.cuda.is_current_stream_capturing():
            return self.model.batch_decode(codes_list=[codes_nq_t], num_quantizers=self.num_quantizers)

        actual_t = int(codes_nq_t.shape[-1])
        padded_size = self._get_padded_size(actual_t)

        if padded_size is None or padded_size not in self.graphs:
            return self.model.batch_decode(codes_list=[codes_nq_t], num_quantizers=self.num_quantizers)

        # --- Fill static buffers then replay ---

        static_codes = self.static_codes[padded_size]  # [NQ, 1, padded_size]

        if actual_t == padded_size:
            # Exact fit: copy the whole buffer at once.
            # codes_nq_t is [NQ, T]; unsqueeze(1) → [NQ, 1, T] matching static_codes.
            static_codes.copy_(codes_nq_t.unsqueeze(1))
        else:
            # Smaller input: zero the buffer first (right-zero-pad), then fill
            # left-aligned.  static_codes[:, 0, :actual_t] is [NQ, actual_t]
            # and codes_nq_t is [NQ, actual_t].
            static_codes.zero_()
            static_codes[:, 0, :actual_t].copy_(codes_nq_t)

        self.graphs[padded_size].replay()

        # static_audio[padded_size] is the live output buffer of the graph.
        # Slice to the real audio length and clone before returning; without
        # the clone the next replay would overwrite the caller's tensor.
        # Derive the slice length from the captured audio shape rather than
        # downsample_rate: the decoder's effective upsample (product of all
        # PatchedPretransform patch sizes) can differ from the config attribute.
        captured_len = self.static_audio[padded_size].shape[-1]
        actual_wav_len = captured_len * actual_t // padded_size
        audio = self.static_audio[padded_size][..., :actual_wav_len].clone()
        audio_lengths = torch.tensor([actual_wav_len], dtype=torch.long, device=audio.device)
        return MossAudioTokenizerDecoderOutput(audio=audio, audio_lengths=audio_lengths)


__all__ = ["MossTTSCUDAGraphCodecWrapper"]
