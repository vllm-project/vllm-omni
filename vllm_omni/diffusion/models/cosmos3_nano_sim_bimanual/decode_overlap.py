# SPDX-License-Identifier: Apache-2.0
"""Request-scoped causal VAE decode and asynchronous CPU return."""

from collections.abc import Callable

import torch


class CausalDecodeQueue:
    """Keep Wan's causal feature cache ordered on one auxiliary CUDA stream.

    The producer records readiness before clean KV refresh. Only the decoder
    waits on that dependency; generation never waits for RGB until finish().
    CPU outputs remain owned and unread until the stream has drained, including
    on errors. A queue serves one fresh full-video request, not streaming ticks.
    """

    def __init__(
        self,
        decode: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device,
        *,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        self.decode = decode
        self.stream = stream if stream is not None else torch.cuda.Stream(device=device)
        self.outputs: list[torch.Tensor] = []
        self.closed = False

    def submit(self, latents: torch.Tensor) -> None:
        if self.closed:
            raise RuntimeError("Cannot submit to a closed decode queue")
        self.stream.wait_stream(torch.cuda.current_stream(latents.device))
        latents.record_stream(self.stream)
        with torch.cuda.stream(self.stream):
            video = self.decode(latents).clamp(-1, 1)
            host = torch.empty(video.shape, dtype=video.dtype, device="cpu", pin_memory=True)
            # Hold the destination before copying so an exception cannot free
            # its storage while an already-enqueued transfer is still using it.
            self.outputs.append(host)
            host.copy_(video, non_blocking=True)

    def close(self) -> None:
        if not self.closed:
            self.stream.synchronize()
            self.closed = True

    def finish(self) -> list[torch.Tensor]:
        self.close()
        return self.outputs
