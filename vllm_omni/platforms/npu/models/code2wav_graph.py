# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exact-shape Ascend graph execution for the shared codec chunk decoder."""

from collections.abc import Iterable

import torch

from vllm_omni.model_executor.models.qwen3_tts.cuda_graph_decoder_wrapper import CUDAGraphDecoderWrapper
from vllm_omni.platforms.npu.graph_tools import NPUExactGraphRunner

MAX_CODE2WAV_GRAPHS = 32


def prepare_code2wav_graph_runtime() -> None:
    """Select ACLNN convolution kernels in the Code2Wav stage process."""
    torch.npu.config.allow_internal_format = False
    torch.npu.set_compile_mode(jit_compile=False)


def plan_code2wav_graph_shapes(
    *,
    steady_window: int,
    start_windows: Iterable[int] = (),
    batch_sizes: Iterable[int] = (),
    max_batch_size: int = 0,
    max_graphs: int = MAX_CODE2WAV_GRAPHS,
) -> list[tuple[int, int]]:
    """Return the ``(batch, frames)`` signatures to capture before serving.

    Start windows (initial chunk or ramp) are captured for one row. The
    steady window, which a mixed streaming batch decodes at, gets every row
    count from 1 up to the remaining budget, or the configured batch sizes.
    """
    shapes = [(1, int(w)) for w in dict.fromkeys(start_windows) if 0 < int(w) != steady_window]
    rows = sorted({1, *(int(b) for b in batch_sizes if int(b) > 0)}) if batch_sizes else range(1, max_graphs + 1)
    shapes += [(b, steady_window) for b in rows if max_batch_size <= 0 or b <= max_batch_size]
    return shapes[: max(0, max_graphs)]


class NPUGraphDecoderWrapper(CUDAGraphDecoderWrapper):
    """Reuse chunk trimming; capture planned exact shapes without padding.

    Outer generation execution stays eager. ``warmup`` captures every planned
    inner decoder signature before serving and then freezes the runner, so
    other shapes run eagerly and no capture happens on a live request.
    """

    def __init__(self, decoder: torch.nn.Module, num_quantizers: int) -> None:
        super().__init__(decoder=decoder, num_quantizers=num_quantizers)
        self.runner = NPUExactGraphRunner(
            max_graphs=MAX_CODE2WAV_GRAPHS,
            component_name="Code2Wav",
            disable_config_hint="set enforce_eager: true for the Code2Wav stage",
        )

    def warmup(  # type: ignore[override]
        self,
        device: torch.device,
        shapes: Iterable[tuple[int, int]],
        dtype: torch.dtype = torch.long,
    ) -> None:
        with torch.inference_mode():
            for batch_size, frames in shapes:
                codes = torch.zeros(batch_size, self.num_quantizers, frames, dtype=dtype, device=device)
                self._decode(codes)
        self.runner.freeze()
        self._warmed_up = True

    def _get_padded_size(self, actual_size: int) -> int:
        return actual_size

    def _forward(self, codes: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (self.decoder(codes, graph_safe_mask=True),)

    def _decode(self, codes: torch.Tensor, *, clone_graph_output: bool = True) -> torch.Tensor:
        # The shared runner always returns owned outputs, including when the
        # caller immediately slices them. Never expose replay-owned storage.
        return self.runner.run("decode", (codes,), (), self._forward)[0]
