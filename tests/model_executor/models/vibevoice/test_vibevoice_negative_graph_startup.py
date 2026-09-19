# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks of production model hooks; no model weights or CUDA required."""

from types import SimpleNamespace

import pytest
import torch
from pytest_mock import MockerFixture

from vllm_omni.model_executor.models.vibevoice.vibevoice import VibeVoiceForConditionalGeneration as Model
from vllm_omni.worker.named_kv.flash_attention import UnsupportedNamedKVGraphError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_default_negative_graph_follows_eager_gate(mocker: MockerFixture):
    from vllm_omni.model_executor.models.vibevoice.runtime_config import VibeVoiceRuntimeConfig

    assert VibeVoiceRuntimeConfig().negative_cuda_graph is True
    model = _model(mocker)
    assert Model._should_enable_negative_graph(model)
    model.vllm_config.model_config.enforce_eager = True
    assert not Model._should_enable_negative_graph(model)


def _model(mocker: MockerFixture):
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(enforce_eager=False, dtype=torch.bfloat16, max_model_len=256),
            parallel_config=SimpleNamespace(tensor_parallel_size=1),
            scheduler_config=SimpleNamespace(max_num_seqs=4),
        ),
        config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=64)),
        model=SimpleNamespace(language_model=object(), warmup_diffusion_graphs=mocker.Mock()),
        _stateful=SimpleNamespace(default_num_diffusion_steps=10, default_guidance_scale=1.3),
        _diffusion_graph_warmup_batch_sizes=[1],
        _negative_executor=mocker.Mock(),
    )


@pytest.mark.parametrize("eager,tp,eligible", [(True, 1, False), (False, 2, False), (False, 1, True)])
def test_negative_graph_eligibility(mocker: MockerFixture, eager, tp, eligible):
    model = _model(mocker)
    model.vllm_config.model_config.enforce_eager = eager
    model.vllm_config.parallel_config.tensor_parallel_size = tp
    assert Model._should_enable_negative_graph(model) is eligible


@pytest.mark.parametrize("enabled,eligible", [(False, True), (True, False), (True, True)])
def test_bind_requires_opt_in_and_eligibility(mocker: MockerFixture, monkeypatch, enabled, eligible):
    import vllm_omni.model_executor.models.vibevoice.vibevoice as model_module

    model = _model(mocker)
    model._negative_kv_branch = None
    model._runtime_config = SimpleNamespace(negative_cuda_graph=enabled)
    model._should_enable_negative_graph = mocker.Mock(return_value=eligible)
    model._create_optional_negative_executor = mocker.Mock(return_value=object())
    model._stateful.bind_negative_branch = mocker.Mock()
    wrapper = mocker.Mock()
    monkeypatch.setattr(model_module, "VibeVoiceNegativeBranch", wrapper)
    store = object()
    Model.bind_named_kv_branch(model, store)
    if enabled and eligible:
        model._create_optional_negative_executor.assert_called_once_with(store)
        assert wrapper.call_args.kwargs["executor"] is model._negative_executor
    else:
        model._create_optional_negative_executor.assert_not_called()
        assert wrapper.call_args.kwargs["executor"] is None
    assert model._negative_kv_branch is wrapper.return_value


@pytest.mark.parametrize("failure_site", ["wrapper", "publish"])
def test_failed_publication_closes_executor_preserving_error(mocker: MockerFixture, monkeypatch, failure_site):
    import vllm_omni.model_executor.models.vibevoice.vibevoice as model_module

    model = _model(mocker)
    model._negative_kv_branch = None
    model._negative_executor = None
    model._runtime_config = SimpleNamespace(negative_cuda_graph=True)
    model._should_enable_negative_graph = mocker.Mock(return_value=True)
    executor = mocker.Mock()
    executor.close.side_effect = RuntimeError("cleanup failed")
    model._create_optional_negative_executor = mocker.Mock(return_value=executor)
    model._stateful.bind_negative_branch = mocker.Mock()
    wrapper = mocker.Mock()
    error = ValueError("publication failed")
    if failure_site == "wrapper":
        wrapper.side_effect = error
    else:
        model._stateful.bind_negative_branch.side_effect = error
    monkeypatch.setattr(model_module, "VibeVoiceNegativeBranch", wrapper)
    store = mocker.Mock()
    with pytest.raises(ValueError, match="publication failed"):
        Model.bind_named_kv_branch(model, store)
    executor.close.assert_called_once_with()
    store.close.assert_not_called()
    assert model._negative_kv_branch is None
    assert model._negative_executor is None


def test_duplicate_bind_does_not_construct_executor(mocker: MockerFixture):
    model = _model(mocker)
    model._negative_kv_branch = object()
    model._create_optional_negative_executor = mocker.Mock()
    with pytest.raises(RuntimeError, match="bound twice"):
        Model.bind_named_kv_branch(model, mocker.Mock())
    model._create_optional_negative_executor.assert_not_called()


def test_startup_warms_all_sizes_and_propagates_failure(mocker: MockerFixture):
    model = _model(mocker)
    Model.warmup_side_graphs(model)
    model._negative_executor.warmup.assert_called_once_with(batch_sizes=[1, 2, 3, 4])
    model._negative_executor.warmup.side_effect = RuntimeError("capture failed")
    with pytest.raises(RuntimeError, match="capture failed"):
        Model.warmup_side_graphs(model)


@pytest.mark.parametrize("unsupported", [True, False])
def test_unpublished_adapter_is_closed_without_closing_runner_pool(mocker: MockerFixture, monkeypatch, unsupported):
    import vllm_omni.model_executor.models.vibevoice.negative_qwen_adapter as adapter_module
    import vllm_omni.worker.named_kv.executor as executor_module
    import vllm_omni.worker.named_kv.flash_attention as backend_module

    store = mocker.Mock(device=torch.device("cpu"), layer_names=("layer",))
    backend = mocker.Mock()
    backend.get_kv_caches.return_value = ([], [])
    monkeypatch.setattr(backend_module, "FlashAttentionKVBranchAdapter", mocker.Mock(return_value=backend))
    adapter = mocker.Mock()
    error = UnsupportedNamedKVGraphError("unsupported") if unsupported else RuntimeError("binding failed")
    adapter.bind_kv_caches.side_effect = error
    monkeypatch.setattr(adapter_module, "Qwen2KVBranchAdapter", mocker.Mock(return_value=adapter))
    constructor = mocker.Mock()
    monkeypatch.setattr(executor_module, "NamedKVBranchExecutor", constructor)
    if unsupported:
        assert Model._create_optional_negative_executor(_model(mocker), store) is None
    else:
        with pytest.raises(RuntimeError, match="binding failed"):
            Model._create_optional_negative_executor(_model(mocker), store)
    adapter.close.assert_called_once_with()
    store.close.assert_not_called()
    constructor.assert_not_called()
