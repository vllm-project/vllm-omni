# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Mammoth mask-keyed MoE FULL graphs on Model Runner V2."""

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor, CudaGraphManager, ModelCudaGraphManager

import vllm_omni.model_executor.models.mammoth_moda2.moe_cudagraph as moe_graphs
from vllm_omni.model_executor.models.mammoth_moda2.mammoth_moda2 import (
    MammothModa2ForConditionalGeneration,
    _enable_moe_routing_cudagraph_specialize,
)
from vllm_omni.model_executor.models.mammoth_moda2.moe_cudagraph import (
    MoERoutingBatchExecutionDescriptor,
    MoERoutingCudaGraphManager,
    MoERoutingKey,
    get_moe_routing_key,
    install_moe_routing_graphs,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner, _needs_capture_tensor_unwrap

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_enable_moe_routing_specialize_auto_on_for_full_when_unset():
    compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL)
    _enable_moe_routing_cudagraph_specialize(SimpleNamespace(compilation_config=compilation_config))
    assert compilation_config.cudagraph_specialize_moe_routing is True


def test_enable_moe_routing_specialize_respects_explicit_false():
    compilation_config = SimpleNamespace(
        cudagraph_mode=CUDAGraphMode.FULL,
        cudagraph_specialize_moe_routing=False,
    )
    _enable_moe_routing_cudagraph_specialize(SimpleNamespace(compilation_config=compilation_config))
    assert compilation_config.cudagraph_specialize_moe_routing is False


def test_enable_moe_routing_specialize_noop_without_compilation_config():
    # Lightweight AR unit stubs may omit compilation_config on VllmConfig.
    _enable_moe_routing_cudagraph_specialize(SimpleNamespace())


GEN_START = 100


@dataclass
class _ReplayBatch:
    """The two ``InputBatch`` fields replay selection reads."""

    input_ids: torch.Tensor
    num_tokens: int


@dataclass
class _Platform:
    device_name: str
    cuda_alike: bool

    def is_cuda_alike(self) -> bool:
        return self.cuda_alike


@dataclass
class _Backbone:
    gen_vocab_start_index: int


@dataclass
class _ArStage:
    language_model: _Backbone


@dataclass
class _CompilationConfig:
    cudagraph_specialize_moe_routing: bool


@dataclass
class _VllmConfig:
    compilation_config: _CompilationConfig


@dataclass
class _HookModel:
    managers: list[object] = field(default_factory=list)

    def specialize_cudagraph_manager(self, manager: object) -> None:
        self.managers.append(manager)


class _PlainModel:
    pass


def _full(num_tokens: int, num_ubatches: int = 1) -> BatchExecutionDescriptor:
    return BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL, num_tokens=num_tokens, num_reqs=min(num_tokens, 4), num_ubatches=num_ubatches
    )


def _batch(ids: list[int]) -> _ReplayBatch:
    return _ReplayBatch(input_ids=torch.tensor(ids), num_tokens=len(ids))


def _manager(descs, cls=ModelCudaGraphManager):
    manager = object.__new__(cls)
    manager._capture_descs = {}
    for desc in descs:
        manager._capture_descs.setdefault(desc.cg_mode, []).append(desc)
    return manager


@pytest.fixture
def cuda_platform(monkeypatch):
    monkeypatch.setattr(moe_graphs, "current_platform", _Platform(device_name="cuda", cuda_alike=True))


def test_install_adds_single_expert_graphs_for_each_full_shape(cuda_platform):
    piecewise = BatchExecutionDescriptor(cg_mode=CUDAGraphMode.PIECEWISE, num_tokens=8, num_reqs=None)
    manager = _manager([_full(8), _full(4), piecewise])

    install_moe_routing_graphs(manager, GEN_START)

    assert isinstance(manager, MoERoutingCudaGraphManager)
    full = manager._capture_descs[CUDAGraphMode.FULL]
    assert [(d.num_tokens, getattr(d, "moe_routing_key", None)) for d in full] == [
        (8, None),
        (8, int(MoERoutingKey.ALL_UND)),
        (8, int(MoERoutingKey.ALL_GEN)),
        (4, None),
        (4, int(MoERoutingKey.ALL_UND)),
        (4, int(MoERoutingKey.ALL_GEN)),
    ]
    # Upstream dispatch tables keep BLEND descriptors, so keyed ones must not
    # compare equal to them.
    assert full[1] != full[0] and hash(full[1]) != hash(full[0])
    assert manager._capture_descs[CUDAGraphMode.PIECEWISE] == [piecewise]


def test_install_skips_microbatched_graphs(cuda_platform):
    manager = _manager([_full(8, num_ubatches=2)])
    install_moe_routing_graphs(manager, GEN_START)
    assert manager._capture_descs[CUDAGraphMode.FULL] == [_full(8, num_ubatches=2)]


def test_install_rejects_other_managers_and_platforms(monkeypatch, cuda_platform):
    class PlatformManager(ModelCudaGraphManager):
        pass

    with pytest.raises(RuntimeError, match="does not implement MoE specialty"):
        install_moe_routing_graphs(_manager([_full(8)], PlatformManager), GEN_START)

    monkeypatch.setattr(moe_graphs, "current_platform", _Platform(device_name="npu", cuda_alike=False))
    with pytest.raises(RuntimeError, match="cudagraph_mode without FULL"):
        install_moe_routing_graphs(_manager([_full(8)]), GEN_START)


def test_capture_runs_each_graph_under_its_specialty(monkeypatch, cuda_platform):
    manager = _manager([_full(4)])
    install_moe_routing_graphs(manager, GEN_START)
    seen = []

    def fake_capture(self, create_forward_fn, progress_bar_desc="", stream=None):
        for desc in self._capture_descs[CUDAGraphMode.FULL]:
            create_forward_fn(desc, warmup=False)(CUDAGraphMode.NONE)

    def create_forward_fn(desc, warmup):
        return lambda cg_mode: seen.append(get_moe_routing_key())

    monkeypatch.setattr(CudaGraphManager, "capture", fake_capture)
    # ModelCudaGraphManager.capture reaches here through super().
    moe_graphs._MoERoutingCaptureScope.capture(manager, create_forward_fn)

    assert seen == [MoERoutingKey.BLEND, MoERoutingKey.ALL_UND, MoERoutingKey.ALL_GEN]
    assert get_moe_routing_key() is MoERoutingKey.BLEND
    assert MoERoutingCudaGraphManager.__mro__.index(moe_graphs._MoERoutingCaptureScope) == (
        MoERoutingCudaGraphManager.__mro__.index(ModelCudaGraphManager) + 1
    )


@pytest.mark.parametrize(
    ("ids", "expected"),
    [
        ([GEN_START, GEN_START + 5], MoERoutingKey.ALL_GEN),
        ([1, 2], MoERoutingKey.ALL_UND),
        ([1, GEN_START], None),
    ],
)
def test_select_replays_graph_matching_unpadded_ids(cuda_platform, ids, expected):
    base = _full(4)
    manager = _manager([base])
    install_moe_routing_graphs(manager, GEN_START)

    selected = manager.select_replay_graph(base, _batch(ids))

    if expected is None:
        assert selected is base
    else:
        assert isinstance(selected, MoERoutingBatchExecutionDescriptor)
        assert selected.moe_routing_key == int(expected)
        assert selected in manager._capture_descs[CUDAGraphMode.FULL]


def test_select_keeps_blend_without_a_specialty_graph(cuda_platform):
    manager = _manager([_full(4)])
    install_moe_routing_graphs(manager, GEN_START)
    uncaptured = _full(2)
    assert manager.select_replay_graph(uncaptured, _batch([GEN_START])) is uncaptured
    eager = BatchExecutionDescriptor(cg_mode=CUDAGraphMode.NONE, num_tokens=4, num_reqs=4)
    assert manager.select_replay_graph(eager, _batch([GEN_START])) is eager


def test_select_ignores_padding_ids(cuda_platform):
    base = _full(4)
    manager = _manager([base])
    install_moe_routing_graphs(manager, GEN_START)
    # Two real all-gen tokens, two pad ids that would read as understanding.
    batch = _ReplayBatch(input_ids=torch.tensor([GEN_START, GEN_START + 1, 0, 0]), num_tokens=2)
    assert manager.select_replay_graph(base, batch).moe_routing_key == int(MoERoutingKey.ALL_GEN)


def test_mammoth_hook_installs_only_when_specialize_is_on(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "vllm_omni.model_executor.models.mammoth_moda2.mammoth_moda2.install_moe_routing_graphs",
        lambda manager, start: calls.append((manager, start)),
    )
    model = object.__new__(MammothModa2ForConditionalGeneration)
    model.ar = _ArStage(language_model=_Backbone(gen_vocab_start_index=GEN_START))
    model.vllm_config = _VllmConfig(compilation_config=_CompilationConfig(cudagraph_specialize_moe_routing=True))

    model.specialize_cudagraph_manager("manager")
    assert calls == [("manager", GEN_START)]

    model.vllm_config.compilation_config.cudagraph_specialize_moe_routing = False
    model.specialize_cudagraph_manager("manager")
    model.ar = None  # DiT stage
    model.specialize_cudagraph_manager("manager")
    assert len(calls) == 1


def test_runner_calls_model_hook_after_kv_cache_init(monkeypatch):
    runner = object.__new__(OmniGPUModelRunner)
    hook_model = _HookModel()
    runner.model = hook_model
    runner.cudagraph_manager = None

    def parent_init(self, *args, **kwargs):
        self.cudagraph_manager = "manager"

    parent = next(cls for cls in OmniGPUModelRunner.__mro__[1:] if "initialize_kv_cache" in vars(cls))
    monkeypatch.setattr(parent, "initialize_kv_cache", parent_init)

    OmniGPUModelRunner.initialize_kv_cache(runner, object(), is_profiling=True)
    assert hook_model.managers == ["manager"]

    # Models without the hook keep the stock manager.
    runner.model = _PlainModel()
    OmniGPUModelRunner.initialize_kv_cache(runner, object())
    assert hook_model.managers == ["manager"]


def test_mammoth_ar_forward_returns_tensor_so_full_graphs_stay_enabled():
    hidden = torch.randn(3, 8)
    model = object.__new__(MammothModa2ForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.have_multimodal_outputs = True
    model.mrv2_forward_returns_tensor = True
    model.model_stage = "ar"
    model.model = lambda *a, **k: OmniOutput(text_hidden_states=hidden, multimodal_outputs={})

    assert model() is hidden
    assert not _needs_capture_tensor_unwrap(model)
