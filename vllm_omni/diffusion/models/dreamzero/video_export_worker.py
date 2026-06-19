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
        dev = torch.cuda.current_device()
        return {
            "device": int(dev),
            "peak_reserved_gib": torch.cuda.max_memory_reserved(dev) / (1024**3),
            "peak_allocated_gib": torch.cuda.max_memory_allocated(dev) / (1024**3),
        }

    def decode_video_latents_to_uint8(self, video_latents: torch.Tensor) -> torch.Tensor:
        if self.model_runner is None or self.model_runner.pipeline is None:
            raise RuntimeError("DreamZero pipeline is not initialized on this worker.")

        with torch.inference_mode():
            decoded = self.model_runner.pipeline.decode_video_latents(video_latents)
            decoded = decoded.squeeze(0).permute(1, 2, 3, 0).contiguous()
            decoded = decoded.clamp(-1, 1) * 0.5 + 0.5
            decoded = (decoded * 255.0).round().to(torch.uint8).cpu()
        return decoded
