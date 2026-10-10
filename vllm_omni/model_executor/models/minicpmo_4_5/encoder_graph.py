# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Device selection for the shared CUDA adapter and optional NPU encoder graphs."""

import torch


def make_encoder_graph(forward, vllm_config):
    from vllm_omni.platforms import current_omni_platform

    if vllm_config.model_config.enforce_eager:
        return None
    if current_omni_platform.is_npu():
        extra = vllm_config.additional_config or {}
        if not extra.get("encoder_enable_npu_graph", False):
            return None
        return EncoderNPUGraph(forward, max_graphs=int(extra.get("encoder_max_npu_graphs", 4)))
    config = vllm_config.model_config.hf_config
    if not getattr(config, "encoder_cuda_graph", True):
        return None
    from .encoder_cuda_graph import EncoderCudaGraph

    return EncoderCudaGraph(
        forward,
        vllm_config,
        max_graphs=getattr(config, "encoder_cuda_graph_max_graphs", 4),
        min_capture_calls=getattr(config, "encoder_cuda_graph_min_capture_calls", 2),
        min_free_bytes=getattr(config, "encoder_cuda_graph_min_free_bytes", 1 << 30),
        share_pools=getattr(config, "encoder_cuda_graph_share_pools", True),
    )


class NPUEncoderGraphRunners:
    """Per-device/stream runners with private pools and a shared capture cap."""

    def __init__(self, *, max_graphs, component_name, disable_config_hint):
        self.max_graphs = max(0, int(max_graphs))
        self.component_name = component_name
        self.disable_config_hint = disable_config_hint
        self.runners = {}

    @property
    def captures(self):
        return sum(r.stats["captures"] for r in self.runners.values())

    def get(self, stream_key):
        from vllm_omni.platforms.npu.graph_tools import NPUExactGraphRunner

        captured = self.captures
        runner = self.runners.get(stream_key)
        if runner is None:
            if captured >= self.max_graphs:
                return None
            runner = NPUExactGraphRunner(
                max_graphs=self.max_graphs - captured,
                component_name=self.component_name,
                disable_config_hint=self.disable_config_hint,
            )
            runner._graph_pool = torch.npu.graph_pool_handle()
            self.runners[stream_key] = runner
        runner.max_graphs = runner.stats["captures"] + self.max_graphs - captured
        return runner


class EncoderNPUGraph:
    """Keep optional masks and stream identity in the capture signature.

    Caller guards exclude training, stateful audio KV and host-dependent paths.
    Each stream owns a runner/pool, with one shared graph-count budget.
    """

    def __init__(self, forward, *, max_graphs=4):
        self.forward = forward
        self.max_graphs = max(0, max_graphs)
        self._graph_runners = NPUEncoderGraphRunners(
            max_graphs=self.max_graphs,
            component_name="MiniCPM encoder",
            disable_config_hint="set encoder_enable_npu_graph=false in stage-0 additional_config",
        )
        self._runners = self._graph_runners.runners

    def __call__(self, *inputs):
        tensors = tuple(x for x in inputs if x is not None)
        if (
            not tensors
            or self.max_graphs == 0
            or any(x.device.type != "npu" for x in tensors)
            or len({x.device for x in tensors}) != 1
            or torch.is_grad_enabled()
            or torch.is_autocast_enabled("npu")
        ):
            return self.forward(*inputs)
        from vllm_omni.platforms.npu.graph_tools import NPUExactGraphRunner

        if (
            not NPUExactGraphRunner.is_supported()
            or not hasattr(torch.npu, "graph_pool_handle")
            or torch.npu.is_current_stream_capturing()
        ):
            return self.forward(*inputs)
        stream = torch.npu.current_stream(tensors[0].device)
        key = (tensors[0].device, stream.npu_stream)
        runner = self._graph_runners.get(key)
        if runner is None:
            return self.forward(*inputs)
        present = tuple(x is not None for x in inputs)

        def compute(*values):
            values = iter(values)
            return (self.forward(*(next(values) if exists else None for exists in present)),)

        return runner.run("encoder", tensors, (present,), compute)[0]
