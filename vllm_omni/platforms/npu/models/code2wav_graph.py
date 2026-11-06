# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exact-shape Ascend graph execution for the shared codec chunk decoder."""

import torch

from vllm_omni.model_executor.models.qwen3_tts.cuda_graph_decoder_wrapper import CUDAGraphDecoderWrapper
from vllm_omni.platforms.npu.graph_tools import NPUExactGraphRunner


def prepare_code2wav_graph_runtime() -> None:
    """Select ACLNN convolution kernels in the Code2Wav stage process."""
    torch.npu.config.allow_internal_format = False
    torch.npu.set_compile_mode(jit_compile=False)


class NPUGraphDecoderWrapper(CUDAGraphDecoderWrapper):
    """Reuse chunk trimming; capture bounded exact shapes without padding.

    Outer generation execution stays eager. Each inner decoder signature is
    primed and captured on first use, then replayed with owning output storage.
    Unseen shapes beyond the graph budget run eagerly; capture failures raise.
    """

    def __init__(self, decoder: torch.nn.Module, num_quantizers: int) -> None:
        super().__init__(decoder=decoder, num_quantizers=num_quantizers)
        self.runner = NPUExactGraphRunner(
            max_graphs=32,
            component_name="Code2Wav",
            disable_config_hint="set enforce_eager: true for the Code2Wav stage",
        )

    def _get_padded_size(self, actual_size: int) -> int:
        return actual_size

    def _forward(self, codes: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (self.decoder(codes, graph_safe_mask=True),)

    def _decode(self, codes: torch.Tensor, *, clone_graph_output: bool = True) -> torch.Tensor:
        # The shared runner always returns owned outputs, including when the
        # caller immediately slices them. Never expose replay-owned storage.
        return self.runner.run("decode", (codes,), (), self._forward)[0]
