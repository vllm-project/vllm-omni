# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch


class DreamZeroVideoExportWorkerExtension:
    """DreamZero worker RPCs used by offline example video export."""

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
