# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5.encoder_graph import EncoderNPUGraph, make_encoder_graph

pytestmark = [pytest.mark.core_model]


@pytest.mark.cpu
def test_npu_opt_in_and_enforce_eager(monkeypatch):
    import vllm_omni.platforms as platforms

    monkeypatch.setattr(platforms, "current_omni_platform", SimpleNamespace(is_npu=lambda: True))
    config = SimpleNamespace(model_config=SimpleNamespace(enforce_eager=False), additional_config={})
    assert make_encoder_graph(torch.sin, config) is None
    config.additional_config = {"encoder_enable_npu_graph": True, "encoder_max_npu_graphs": 2}
    assert make_encoder_graph(torch.sin, config).max_graphs == 2
    config.model_config.enforce_eager = True
    assert make_encoder_graph(torch.sin, config) is None


@pytest.mark.cpu
def test_cuda_factory_preserves_merged_admission_and_pool_options(monkeypatch):
    import vllm_omni.platforms as platforms

    monkeypatch.setattr(platforms, "current_omni_platform", SimpleNamespace(is_npu=lambda: False))
    hf_config = SimpleNamespace(
        encoder_cuda_graph=True,
        encoder_cuda_graph_max_graphs=3,
        encoder_cuda_graph_min_capture_calls=5,
        encoder_cuda_graph_min_free_bytes=1024,
        encoder_cuda_graph_share_pools=False,
    )
    config = SimpleNamespace(model_config=SimpleNamespace(enforce_eager=False, hf_config=hf_config))
    graph = make_encoder_graph(torch.sin, config)
    assert graph.max_graphs == 3
    assert graph.min_capture_calls == 5
    assert graph.min_free_bytes == 1024
    assert graph.share_pools is False
    hf_config.encoder_cuda_graph = False
    assert make_encoder_graph(torch.sin, config) is None
    hf_config.encoder_cuda_graph = True
    config.model_config.enforce_eager = True
    assert make_encoder_graph(torch.sin, config) is None


@pytest.mark.cpu
def test_cpu_preserves_gradients():
    graph = EncoderNPUGraph(lambda x, mask: x.sin())
    x = torch.randn(3, requires_grad=True)
    graph(x, None).sum().backward()
    torch.testing.assert_close(x.grad, x.detach().cos())
    assert not graph._runners


@pytest.mark.cpu
@torch.inference_mode()
def test_optional_masks_stream_isolation_and_global_cap(monkeypatch):
    from vllm_omni.platforms.npu.graph_tools import NPUExactGraphRunner

    stream = SimpleNamespace(npu_stream=1)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            current_stream=lambda device: stream, is_current_stream_capturing=lambda: False, graph_pool_handle=object
        ),
        raising=False,
    )
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda device: False)
    monkeypatch.setattr(NPUExactGraphRunner, "is_supported", staticmethod(lambda: True))
    calls = []

    def run(self, operation, inputs, constants, compute):
        calls.append((self, constants))
        self._graphs[constants] = object()
        return compute(*inputs)

    monkeypatch.setattr(NPUExactGraphRunner, "run", run)

    # Simulate NPU placement without requiring the torch-npu runtime.
    class Device:
        type = "npu"

    x = SimpleNamespace(device=Device())
    graph = EncoderNPUGraph(lambda value, mask: mask, max_graphs=2)
    assert graph(x, None) is None
    stream.npu_stream = 2
    assert graph(x, x) is x
    assert calls[0][0] is not calls[1][0]
    assert calls[0][0]._graph_pool is not calls[1][0]._graph_pool
    assert calls[0][1] != calls[1][1]
    stream.npu_stream = 3
    assert graph(x, None) is None
    assert len(calls) == 2
    assert len(graph._runners) == 2


@pytest.mark.skipif(not hasattr(torch, "npu"), reason="torch-npu required")
@torch.inference_mode()
def test_npu_replay_updates_mask_and_retains_outputs():
    graph = EncoderNPUGraph(lambda x, mask: x.sin() if mask is None else x.sin() + mask, max_graphs=2)
    x = torch.randn(2, 8, device="npu")
    mask = torch.randn_like(x)
    graph(x, mask)
    old = graph(x, mask)
    expected_old = old.clone()
    x.add_(1)
    mask.mul_(2)
    torch.testing.assert_close(graph(x, mask), x.sin() + mask)
    torch.testing.assert_close(old, expected_old)
    torch.testing.assert_close(graph(x, None), x.sin())
    assert sum(r.stats["captures"] for r in graph._runners.values()) == 2
