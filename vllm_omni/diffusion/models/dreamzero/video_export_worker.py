# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch


class DreamZeroVideoExportWorkerExtension:
    """DreamZero worker RPCs used by offline example video export."""

    def gpu_mem_stats(self) -> dict:
        """Peak GPU memory (GiB) for this worker's device, for profiling.

        ``max_memory_reserved`` is the caching-allocator high-water mark (the best
        in-process proxy for "peak VRAM"); ``max_memory_allocated`` is the live
        tensor high-water mark. Both are monotonic from process start, so the
        end-of-run values capture the whole rollout's peak.
        """
        dev = torch.accelerator.current_device_index()
        return {
            "device": int(dev),
            "peak_reserved_gib": torch.accelerator.max_memory_reserved(dev) / (1024**3),
            "peak_allocated_gib": torch.accelerator.max_memory_allocated(dev) / (1024**3),
        }

    def ar_diffusion_perf_stats(self, reset: bool = True) -> dict:
        """Per-request server-side forward E2E times (s) recorded by the AR-Diffusion runner.

        The worker-side compute time per ``execute_model`` — the analog of the
        upstream server's per-request E2E — which excludes the engine<->worker IPC
        that the client's ``omni.generate`` wall time includes. Returns ``[]`` when
        the runner is the base (KV-disabled) runner that does not record timings.
        """
        runner = self.model_runner
        times = list(getattr(runner, "_perf_e2e_times", []) or [])
        if reset and hasattr(runner, "_perf_e2e_times"):
            runner._perf_e2e_times = []
        return {"server_e2e_s": times}

    @staticmethod
    def _latents_to_uint8_frames(decoded: torch.Tensor) -> torch.Tensor:
        decoded = decoded.squeeze(0).permute(1, 2, 3, 0).contiguous()
        decoded = decoded.clamp(-1, 1) * 0.5 + 0.5
        return (decoded * 255.0).round().to(torch.uint8).cpu()

    def decode_video_latents_to_uint8(self, video_latents: torch.Tensor) -> torch.Tensor:
        if self.model_runner is None or self.model_runner.pipeline is None:
            raise RuntimeError("DreamZero pipeline is not initialized on this worker.")

        with torch.inference_mode():
            decoded = self.model_runner.pipeline.decode_video_latents(video_latents)
            return self._latents_to_uint8_frames(decoded)

    def decode_accumulated_video_latents_to_uint8(self, session_id: str = "default") -> torch.Tensor:
        """Decode all AR-chunk latents accumulated on the server for one session."""
        if self.model_runner is None or self.model_runner.pipeline is None:
            raise RuntimeError("DreamZero pipeline is not initialized on this worker.")

        with torch.inference_mode():
            decoded = self.model_runner.pipeline.decode_accumulated_video_latents(session_id)
            return self._latents_to_uint8_frames(decoded)

    def clear_accumulated_video_latents(self, session_id: str = "default") -> None:
        if self.model_runner is None or self.model_runner.pipeline is None:
            raise RuntimeError("DreamZero pipeline is not initialized on this worker.")
        self.model_runner.pipeline.clear_accumulated_video_latents(session_id)
