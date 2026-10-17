# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fixed-shape session replay using the platform's existing graph wrapper."""

from collections.abc import Callable

import torch
from vllm.compilation.cuda_graph import CUDAGraphOptions
from vllm.config import CUDAGraphMode, get_current_vllm_config
from vllm.forward_context import BatchDescriptor
from vllm.utils.torch_utils import current_stream

from vllm_omni.platforms import current_omni_platform


class MatrixGraph:
    """Own static inputs; the shared wrapper owns capture, replay and its pool.

    The callable may overwrite current KV, but must not advance the history.
    Each entry belongs to one session and is released before its caches.
    """

    def __init__(
        self, run: Callable[..., torch.Tensor], preserve: tuple[torch.Tensor, ...] = (), *, num_tokens: int = 1
    ):
        caller = torch.accelerator.current_stream()
        self.stream = current_stream()
        torch.accelerator.set_stream(caller)
        self.config = get_current_vllm_config()
        wrapper = current_omni_platform.get_graph_wrapper_cls()
        self.wrapper = wrapper(
            run,
            self.config,
            CUDAGraphMode.FULL,
            cudagraph_options=CUDAGraphOptions(weak_ref_output=False),
        )
        self.inputs: tuple[torch.Tensor | None, ...] | None = None
        self.output: torch.Tensor | None = None
        self.preserve = preserve
        self.num_tokens = num_tokens

    def __call__(self, *inputs: torch.Tensor | None) -> torch.Tensor:
        caller = torch.accelerator.current_stream()
        self.stream.wait_stream(caller)
        try:
            with self.stream:
                return self._call(*inputs)
        finally:
            # The in-process engine presents outputs from a different thread.
            # Keep the runner's original producer stream and join replay work
            # before its postprocess/D2H operations; no host synchronization.
            caller.wait_stream(self.stream)

    def _call(self, *inputs: torch.Tensor | None) -> torch.Tensor:
        first = self.inputs is None
        if first:
            self.inputs = tuple(value.clone() if value is not None else None for value in inputs)
        else:
            assert self.inputs is not None
            for target, source in zip(self.inputs, inputs):
                if target is not None:
                    assert source is not None
                    target.copy_(source)
        if first:
            saved = tuple(value.clone() for value in self.preserve)
            self.wrapper.unwrap()(*self.inputs)
            for target, source in zip(self.preserve, saved):
                target.copy_(source)
            self.stream.synchronize()
        with current_omni_platform.set_forward_context(
            None,
            self.config,
            cudagraph_runtime_mode=CUDAGraphMode.FULL,
            batch_descriptor=BatchDescriptor(num_tokens=self.num_tokens),
        ):
            if first:
                # Capture records operations but does not execute them. Retain
                # ownership of the output before the wrapper returns weak refs.
                self.output = self.wrapper(*self.inputs)
            output = self.wrapper(*self.inputs)
        # The next replay overwrites the static output. Give the codec or
        # sampler ownership until its consumer has finished using this chunk.
        return output.clone()

    def close(self) -> None:
        self.stream.synchronize()
        self.wrapper.clear_graphs()
        self.inputs = None
        self.output = None
