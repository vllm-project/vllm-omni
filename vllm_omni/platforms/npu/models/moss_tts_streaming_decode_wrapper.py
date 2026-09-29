# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MOSS-TTS codec streaming NPUGraph acceleration on Ascend.

Mirrors :class:`CUDAGraphStreamingDecoderWrapper` but uses
:class:`NPUExactGraphRunner` for capture/replay.  Graph capture is gated
by ``enforce_eager: false`` → ``cudagraph_capture_sizes`` → batch sizes;
set ``enforce_eager: true`` on Stage 1 to disable.
"""

from __future__ import annotations

import os
import time

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.platforms.npu.graph_tools import NPUExactGraphRunner

logger = init_logger(__name__)


def _prepare_moss_codec_graph_runtime() -> None:
    """Select graph-capturable ACLNN kernels before MOSS-TTS codec capture.

    Mirrors ``prepare_code2wav_graph_runtime`` from minicpmo_4_5_code2wav:
    ``allow_internal_format=False`` + ``jit_compile=False`` are required for
    NPUGraph capture of the codec's conv/attention kernels, but must only be
    set when graph capture is actually enabled.
    """
    if os.environ.get("ASCEND_LAUNCH_BLOCKING") == "1":
        raise RuntimeError(
            "MOSS-TTS codec NPUGraph capture is incompatible with "
            "ASCEND_LAUNCH_BLOCKING=1; unset it or set it to 0 before startup."
        )
    npu = torch.npu
    npu.config.allow_internal_format = False
    npu.set_compile_mode(jit_compile=False)
    logger.info("Configured MOSS-TTS codec NPUGraph runtime (allow_internal_format=False, jit_compile=False)")


class NPUGraphStreamingDecoderWrapper:
    """Replay streaming decode graphs keyed by ``(B_bucket, exact_T)`` on NPU.

    Delegates capture/replay to :class:`NPUExactGraphRunner`.  Bucket selection
    (batch + frame padding) mirrors the CUDA wrapper so
    ``_MossCodecStreamSession.step`` can use either interchangeably.  Codec
    state is updated in-place during replay via ``state_slot_ids``; scratch
    slots isolate padding rows.

    Fatal NPUGraph failures propagate as ``RuntimeError`` (eager fallback is
    unsafe after corrupt capture state); restart with ``enforce_eager: true``
    on Stage 1 to recover.
    """

    def __init__(
        self,
        codec: nn.Module,
        *,
        state_capacity: int,
        batch_sizes: list[int],
        frame_sizes: list[int],
        num_quantizers: int,
        vllm_config: VllmConfig,
    ) -> None:
        self.codec = codec
        self.state_capacity = int(state_capacity)
        self.batch_sizes = sorted({int(size) for size in batch_sizes if 0 < int(size) <= state_capacity})
        self.frame_sizes = sorted({int(size) for size in frame_sizes if int(size) > 0})
        self.num_quantizers = int(num_quantizers)

        # vllm_config is accepted for interface parity but unused; graph
        # capture is gated by enforce_eager on Stage 1.
        del vllm_config
        _prepare_moss_codec_graph_runtime()
        max_graphs = len(self.batch_sizes) * len(self.frame_sizes)
        self._graph_runner: NPUExactGraphRunner | None = NPUExactGraphRunner(
            max_graphs=max_graphs,
            component_name="MOSS-TTS Codec",
            disable_config_hint="set enforce_eager: true on Stage 1",
        )
        if not self._graph_runner.is_supported():
            logger.warning("MOSS-TTS codec NPUGraph is not supported on this platform; falling back to eager.")
            self._graph_runner = None

    @property
    def is_ready(self) -> bool:
        return self._graph_runner is not None

    @property
    def scratch_capacity(self) -> int:
        return max(self.batch_sizes, default=0)

    @torch.no_grad()
    def warmup(self, device: torch.device) -> None:
        if self._graph_runner is None or not self.batch_sizes or not self.frame_sizes:
            return
        if device.type != "npu":
            return

        capture_keys = sorted(
            ((b, t) for b in self.batch_sizes for t in self.frame_sizes),
            reverse=True,
        )
        logger.info(
            "MOSS-TTS streaming decoder NPUGraph warmup: n_vq=%d (B,T)=%s",
            self.num_quantizers,
            list(reversed(capture_keys)),
        )
        start_s = time.perf_counter()
        for batch_size, frame_size in capture_keys:
            self._warmup_bucket(batch_size, frame_size, device)
            logger.info(
                "  Captured MOSS-TTS streaming decoder NPUGraph for (B,T)=(%d,%d)",
                batch_size,
                frame_size,
            )

        stats = self._graph_runner.stats if self._graph_runner else {}
        captured = stats.get("captures", 0)
        logger.info(
            "MOSS-TTS streaming decoder NPUGraph warmup complete: %d/%d captured in %.1f ms (stats=%s)",
            captured,
            len(capture_keys),
            (time.perf_counter() - start_s) * 1000.0,
            stats,
        )

    @torch.no_grad()
    def _warmup_bucket(self, batch_size: int, frame_size: int, device: torch.device) -> None:
        """Trigger first-run capture for one (B,T) bucket using scratch slots.

        Feeds scratch slot ids + ``valid_rows=False`` so capture-side state
        writes land in scratch slots, then resets them.
        """
        assert self._graph_runner is not None
        codes = torch.zeros(
            self.num_quantizers,
            batch_size,
            frame_size,
            dtype=torch.long,
            device=device,
        )
        lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        scratch_slots = self.state_capacity + torch.arange(batch_size, dtype=torch.long, device=device)
        valid_rows = torch.zeros(batch_size, dtype=torch.bool, device=device)

        self._graph_runner.run(
            "moss_codec_decode",
            (codes, lengths, scratch_slots, valid_rows),
            (batch_size, frame_size),
            lambda c, lengths, s, v: self.codec.decode_streaming_tensors(c, lengths, s, v),
        )
        self.codec.reset_decoder_state_slots(scratch_slots)

    def _select_batch_size(self, actual_batch_size: int) -> int | None:
        return next((size for size in self.batch_sizes if size >= actual_batch_size), None)

    def _select_frame_size(self, actual_frame_size: int, allow_padding: bool) -> int | None:
        if actual_frame_size in self.frame_sizes:
            return actual_frame_size
        if not allow_padding:
            return None
        return next((size for size in self.frame_sizes if size >= actual_frame_size), None)

    @torch.no_grad()
    def decode(
        self,
        codes: torch.Tensor,
        state_slot_ids: torch.Tensor,
        *,
        allow_frame_padding: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, int] | None:
        if self._graph_runner is None:
            return None
        if codes.device.type != "npu":
            return None

        n_vq, actual_batch_size, frame_size = codes.shape
        if int(n_vq) != self.num_quantizers or state_slot_ids.shape != (actual_batch_size,):
            return None

        batch_size = self._select_batch_size(int(actual_batch_size))
        if batch_size is None:
            return None
        graph_frame_size = self._select_frame_size(int(frame_size), allow_frame_padding)
        if graph_frame_size is None:
            return None

        device = codes.device
        # Padding rows use scratch slots + valid_rows=False to isolate state.
        padded_codes = torch.zeros(
            self.num_quantizers,
            batch_size,
            graph_frame_size,
            dtype=torch.long,
            device=device,
        )
        padded_codes[:, :actual_batch_size, :frame_size].copy_(codes)
        lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        lengths[:actual_batch_size].fill_(int(frame_size))
        padded_slots = self.state_capacity + torch.arange(batch_size, dtype=torch.long, device=device)
        padded_slots[:actual_batch_size].copy_(state_slot_ids)
        valid_rows = torch.zeros(batch_size, dtype=torch.bool, device=device)
        valid_rows[:actual_batch_size].fill_(True)

        audio, audio_lengths = self._graph_runner.run(
            "moss_codec_decode",
            (padded_codes, lengths, padded_slots, valid_rows),
            (batch_size, graph_frame_size),
            lambda c, lengths, s, v: self.codec.decode_streaming_tensors(c, lengths, s, v),
        )

        return audio, audio_lengths, int(actual_batch_size)


__all__ = ["NPUGraphStreamingDecoderWrapper"]
