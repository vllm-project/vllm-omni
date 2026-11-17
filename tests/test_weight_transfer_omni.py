# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Weight transfer integration tests for vllm-omni.

Verifies that the upstream vLLM weight transfer protocol is wired into both
stage types:

* AR stage — ``OmniGPUWorkerBase`` inherits vLLM's ``GPUWorker``, so the
  four-phase lifecycle comes for free. These tests pin that inheritance so a
  future refactor that stops extending ``GPUWorker`` fails loudly.
* Diffusion stage — ``DiffusionWorker`` does not inherit ``GPUWorker``; the
  lifecycle is implemented directly on it and delegates to the same
  ``WeightTransferEngine`` interface.

Also covers the config plumbing: ``weight_transfer_config`` must survive the
deploy-config merge so it reaches every stage's engine args.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = [pytest.mark.cpu, pytest.mark.advanced_model]

# --- Config plumbing ---


def test_weight_transfer_config_is_pipeline_wide_field():
    """weight_transfer_config must propagate to every stage's engine args."""
    from vllm_omni.config.stage_config import PIPELINE_WIDE_ENGINE_FIELDS

    assert "weight_transfer_config" in PIPELINE_WIDE_ENGINE_FIELDS


def test_weight_transfer_config_is_deploy_override_key():
    """It must survive build_stage_runtime_overrides' internal-key filter."""
    from vllm_omni.config.stage_config import deploy_runtime_override_keys

    assert "weight_transfer_config" in deploy_runtime_override_keys()


def test_weight_transfer_config_survives_stage_overrides():
    """A top-level weight_transfer_config reaches a stage's overrides."""
    from vllm_omni.config.stage_config import build_stage_runtime_overrides

    overrides = build_stage_runtime_overrides(
        stage_id=0,
        cli_overrides={"weight_transfer_config": {"backend": "nccl"}},
    )
    assert overrides["weight_transfer_config"] == {"backend": "nccl"}


def test_stage_scoped_weight_transfer_config_targets_one_stage():
    """stage_<id>_weight_transfer_config only lands on the matching stage."""
    from vllm_omni.config.stage_config import build_stage_runtime_overrides

    cli = {"stage_1_weight_transfer_config": {"backend": "ipc"}}

    assert build_stage_runtime_overrides(1, cli) == {"weight_transfer_config": {"backend": "ipc"}}
    assert build_stage_runtime_overrides(0, cli) == {}


# --- AR stage: inherited from upstream GPUWorker ---


_LIFECYCLE_METHODS = (
    "init_weight_transfer_engine",
    "start_weight_update",
    "update_weights",
    "finish_weight_update",
)


@pytest.mark.parametrize("method", _LIFECYCLE_METHODS)
def test_ar_worker_exposes_lifecycle(method):
    from vllm_omni.worker.base import OmniGPUWorkerBase

    assert callable(getattr(OmniGPUWorkerBase, method, None))


def test_ar_worker_still_inherits_upstream_gpu_worker():
    """The AR lifecycle is inherited, not copied — keep it that way."""
    from vllm.v1.worker.gpu_worker import Worker as GPUWorker

    from vllm_omni.worker.base import OmniGPUWorkerBase

    assert issubclass(OmniGPUWorkerBase, GPUWorker)
    for method in _LIFECYCLE_METHODS:
        assert method not in vars(OmniGPUWorkerBase), (
            f"{method} is overridden on OmniGPUWorkerBase; the AR stage should "
            "reuse upstream's implementation so the two stay in sync."
        )


# --- Diffusion stage: implemented on DiffusionWorker ---


@pytest.mark.parametrize("method", _LIFECYCLE_METHODS)
def test_diffusion_worker_exposes_lifecycle(method):
    from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

    assert callable(getattr(DiffusionWorker, method, None))


def _bare_diffusion_worker(engine=None, model=MagicMock()):
    """A DiffusionWorker with __init__ bypassed (no GPU/distributed setup)."""
    from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

    worker = object.__new__(DiffusionWorker)
    worker.weight_transfer_engine = engine
    worker._weight_update_active = False
    worker.model_runner = MagicMock()
    worker.model_runner.pipeline = model
    return worker


def test_diffusion_lifecycle_requires_configured_engine():
    worker = _bare_diffusion_worker(engine=None)

    for method in _LIFECYCLE_METHODS:
        with pytest.raises(RuntimeError, match="not initialized"):
            call = getattr(worker, method)
            call({}) if method in ("init_weight_transfer_engine", "update_weights") else call()


def test_diffusion_happy_path_delegates_to_engine():
    engine = MagicMock()
    worker = _bare_diffusion_worker(engine=engine)

    worker.init_weight_transfer_engine({"backend": "nccl"})
    engine.parse_init_info.assert_called_once_with({"backend": "nccl"})
    engine.init_transfer_engine.assert_called_once_with(engine.parse_init_info.return_value)

    worker.start_weight_update()
    engine.start_weight_update.assert_called_once()
    assert worker._weight_update_active is True

    worker.update_weights({"names": []})
    engine.update_weights.assert_called_once_with({"names": []})

    worker.finish_weight_update()
    engine.finish_weight_update.assert_called_once()
    engine.reset_weight_update_target.assert_called_once()
    assert worker._weight_update_active is False


@pytest.mark.parametrize("has_existing_engine", [False, True])
def test_custom_pipeline_reload_rebinds_weight_transfer(monkeypatch, has_existing_engine):
    """Updates and target resets after reload must use the replacement pipeline."""
    import torch
    from vllm.config.weight_transfer import WeightTransferConfig

    from vllm_omni.diffusion.worker.diffusion_worker import CustomPipelineWorkerExtension

    class Pipeline(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2))
            self.supports_weight_reload = True

        def load_weights(self, weights):
            from vllm.model_executor.model_loader.weight_utils import default_weight_loader

            for name, value in weights:
                param = getattr(self, name)
                getattr(param, "weight_loader", default_weight_loader)(param, value)

    old, replacement = Pipeline(), Pipeline()
    worker = _bare_diffusion_worker(model=old)
    worker.vllm_config = SimpleNamespace(
        weight_transfer_config=WeightTransferConfig(backend="ipc"),
        parallel_config=None,
        model_config=None,
        quant_config=None,
    )
    from vllm_omni.diffusion.model_loader.weight_reload import prepare_model_for_weight_reload

    worker.od_config = _reload_config()
    for pipeline in (old, replacement):
        prepare_model_for_weight_reload(pipeline, worker.od_config, torch.device("cpu"))
    worker.device = torch.device("cpu")
    if has_existing_engine:
        worker._init_weight_transfer_engine()
        previous = worker.weight_transfer_engine
        previous._packed_importer = MagicMock()
    worker.load_model = lambda **kwargs: setattr(worker.model_runner, "pipeline", replacement)
    worker.init_lora_manager = MagicMock()
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)

    CustomPipelineWorkerExtension.re_init_pipeline(worker, {"pipeline_class": "test.Pipeline"})
    if has_existing_engine:
        previous._packed_importer.close.assert_called_once()
    for value in (1.0, 2.0):
        worker.init_weight_transfer_engine({"packed": True})
        worker.start_weight_update()
        assert worker.weight_transfer_engine.model is replacement
        worker.weight_transfer_engine.model.load_weights([("weight", torch.full((2,), value))])
        worker.finish_weight_update()
        assert worker.weight_transfer_engine.model is replacement
        torch.testing.assert_close(replacement.weight, torch.full((2,), value))
        torch.testing.assert_close(old.weight, torch.zeros(2))


def test_custom_pipeline_reload_rejects_active_update():
    from vllm_omni.diffusion.worker.diffusion_worker import CustomPipelineWorkerExtension

    engine = MagicMock()
    worker = _bare_diffusion_worker(engine=engine)
    worker.start_weight_update()
    model = worker.model_runner.pipeline
    with pytest.raises(RuntimeError, match="active weight update"):
        CustomPipelineWorkerExtension.re_init_pipeline(worker, {"pipeline_class": "test.Pipeline"})
    assert worker.model_runner.pipeline is model
    engine.shutdown.assert_not_called()


def test_diffusion_rejects_update_before_start():
    worker = _bare_diffusion_worker(engine=MagicMock())

    with pytest.raises(RuntimeError, match="start_weight_update must be called"):
        worker.update_weights({})


def test_diffusion_rejects_finish_before_start():
    worker = _bare_diffusion_worker(engine=MagicMock())

    with pytest.raises(RuntimeError, match="without a matching start_weight_update"):
        worker.finish_weight_update()


def test_diffusion_rejects_double_start():
    worker = _bare_diffusion_worker(engine=MagicMock())
    worker.start_weight_update()

    with pytest.raises(RuntimeError, match="already"):
        worker.start_weight_update()


def test_diffusion_start_failure_resets_target():
    engine = MagicMock()
    engine.start_weight_update.side_effect = RuntimeError("boom")
    worker = _bare_diffusion_worker(engine=engine)

    with pytest.raises(RuntimeError, match="boom"):
        worker.start_weight_update()

    engine.reset_weight_update_target.assert_called_once()
    assert worker._weight_update_active is False


def test_diffusion_update_failure_ends_session():
    """A failed chunk must not leave a half-open session behind."""
    engine = MagicMock()
    engine.update_weights.side_effect = RuntimeError("nccl abort")
    worker = _bare_diffusion_worker(engine=engine)
    worker.start_weight_update()

    with pytest.raises(RuntimeError, match="nccl abort"):
        worker.update_weights({})

    assert worker._weight_update_active is False
    engine.reset_weight_update_target.assert_called_once()


def test_diffusion_session_is_reusable_after_finish():
    engine = MagicMock()
    worker = _bare_diffusion_worker(engine=engine)

    for _ in range(2):
        worker.start_weight_update()
        worker.update_weights({})
        worker.finish_weight_update()

    assert engine.start_weight_update.call_count == 2
    assert engine.finish_weight_update.call_count == 2


def test_orchestrator_exposes_lifecycle():
    """AsyncOmni must expose the full lifecycle for EngineClient compatibility."""
    from vllm_omni.entrypoints.async_omni import AsyncOmni

    for method in (
        "init_weight_transfer_engine",
        "start_weight_update",
        "update_weights",
        "finish_weight_update",
    ):
        assert callable(getattr(AsyncOmni, method, None))


def _omni_with_stages(stage_configs):
    """A bare AsyncOmni whose engine exposes the given per-stage configs."""
    from vllm_omni.entrypoints.async_omni import AsyncOmni

    omni = object.__new__(AsyncOmni)
    pools = [SimpleNamespace(stage_vllm_config=cfg) for cfg in stage_configs]
    omni.engine = SimpleNamespace(stage_pools=pools)
    return omni


def test_orchestrator_raises_on_stage_error_result():
    """A per-stage error result dict must raise, not silently succeed."""
    import asyncio

    enabled = SimpleNamespace(weight_transfer_config={"backend": "ipc"})

    omni = _omni_with_stages([enabled])
    omni.collective_rpc = AsyncMock(return_value=[{"supported": False, "error": "boom"}])

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(omni.start_weight_update())


def test_orchestrator_routes_only_to_configured_stages():
    """Lifecycle RPCs skip stages without weight_transfer_config."""
    import asyncio

    plain = SimpleNamespace(weight_transfer_config=None)
    diffusion = SimpleNamespace(
        weight_transfer_config=None,
        diffusion_config=SimpleNamespace(weight_transfer_config={"backend": "ipc"}),
    )

    omni = _omni_with_stages([plain, diffusion])
    omni.collective_rpc = AsyncMock(return_value=[True])

    asyncio.run(omni.init_weight_transfer_engine({"packed": False}))
    omni.collective_rpc.assert_awaited_once_with(
        method="init_weight_transfer_engine",
        args=({"packed": False},),
        stage_ids=[1],
    )


def test_orchestrator_rejects_when_no_stage_configured():
    """No configured stage is a hard error, not a silent no-op."""
    import asyncio

    omni = _omni_with_stages([SimpleNamespace(weight_transfer_config=None)])
    omni.collective_rpc = AsyncMock()

    with pytest.raises(RuntimeError, match="No stage has weight transfer enabled"):
        asyncio.run(omni.start_weight_update())
    omni.collective_rpc.assert_not_awaited()


def _reload_config(**overrides):
    return SimpleNamespace(
        **{
            "weight_transfer_config": {"backend": "ipc"},
            "quantization_config": None,
            "parallel_config": SimpleNamespace(use_hsdp=False),
            "cache_backend": "none",
            **overrides,
        }
    )


def test_upstream_reload_preserves_storage_and_frozen_components():
    """Split a layer across chunks and repeat reload without changing frozen state."""
    import torch
    from vllm.config.weight_transfer import WeightTransferConfig
    from vllm.distributed.weight_transfer.ipc_engine import IPCWeightTransferEngine
    from vllm.model_executor.models.utils import AutoWeightsLoader

    from vllm_omni.diffusion.model_loader.weight_reload import prepare_model_for_weight_reload

    class Pipeline(torch.nn.Module):
        supports_weight_reload = True

        def __init__(self):
            super().__init__()
            self.transformer = torch.nn.Module()
            self.transformer.register_parameter("weight", torch.nn.Parameter(torch.zeros(2, 3)))
            self.transformer.register_parameter("gain", torch.nn.Parameter(torch.zeros(2)))
            self.frozen = torch.nn.Linear(2, 2)
            self.frozen.register_buffer("constant", torch.ones(2), persistent=False)

        def load_weights(self, weights):
            return AutoWeightsLoader(self).load_weights(weights)

    pipeline = Pipeline()
    prepare_model_for_weight_reload(pipeline, _reload_config(), torch.device("cpu"))
    tensors = dict(pipeline.named_parameters()) | dict(pipeline.named_buffers())
    pointers = {name: value.data_ptr() for name, value in tensors.items()}
    frozen = {name: value.clone() for name, value in tensors.items() if name.startswith("frozen.")}
    engine = IPCWeightTransferEngine(
        WeightTransferConfig(backend="ipc"),
        SimpleNamespace(parallel_config=None, model_config=None),
        torch.device("cpu"),
        pipeline,
    )
    for value in (2.0, 3.0):
        engine.start_weight_update()
        pipeline.load_weights([("transformer.weight", torch.full((2, 3), value))])
        assert pipeline.transformer.weight.is_meta
        pipeline.load_weights([("transformer.gain", torch.full((2,), value + 1))])
        engine.finish_weight_update()
        torch.testing.assert_close(pipeline.transformer.weight, torch.full((2, 3), value))
        torch.testing.assert_close(pipeline.transformer.gain, torch.full((2,), value + 1))
        current = dict(pipeline.named_parameters()) | dict(pipeline.named_buffers())
        assert {name: tensor.data_ptr() for name, tensor in current.items()} == pointers
        assert all(tensor.device.type == "cpu" for tensor in current.values())
        for name, expected in frozen.items():
            torch.testing.assert_close(current[name], expected)
        assert "constant" in pipeline.frozen._non_persistent_buffers_set


@pytest.mark.parametrize("unsupported", ["pipeline", "quantization", "hsdp", "offload", "cache"])
def test_reload_rejects_unsupported_configuration_before_mutation(unsupported):
    import torch

    from vllm_omni.diffusion.model_loader.weight_reload import prepare_model_for_weight_reload

    model = torch.nn.Linear(2, 2)
    model.supports_weight_reload = unsupported != "pipeline"
    config = _reload_config()
    if unsupported == "quantization":
        config.quantization_config = object()
    elif unsupported == "hsdp":
        config.parallel_config.use_hsdp = True
    elif unsupported == "offload":
        config.enable_cpu_offload = True
    elif unsupported == "cache":
        config.cache_backend = "cache_dit"
    original = model.weight
    with pytest.raises(ValueError):
        prepare_model_for_weight_reload(model, config, torch.device("cpu"))
    assert model.weight is original and not model.weight.is_meta
    assert not getattr(model, "_omni_weight_reload_prepared", False)


def test_reload_preparation_is_opt_in():
    import torch

    from vllm_omni.diffusion.model_loader.weight_reload import prepare_model_for_weight_reload

    model = torch.nn.Linear(2, 2)
    prepare_model_for_weight_reload(model, _reload_config(weight_transfer_config=None), torch.device("cpu"))
    assert not getattr(model, "_omni_weight_reload_prepared", False)


@pytest.mark.parametrize("load_format", ["default", "custom_pipeline"])
def test_loader_prepares_default_and_custom_pipelines(monkeypatch, load_format):
    import torch
    from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info

    from vllm_omni.diffusion.model_loader import diffusers_loader

    class Pipeline(torch.nn.Module):
        supports_weight_reload = True

        def __init__(self, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2))

    monkeypatch.setattr(diffusers_loader, "initialize_model", lambda config: Pipeline())
    monkeypatch.setattr(diffusers_loader, "_resolve_custom_pipeline_cls", lambda name: Pipeline)
    loader = object.__new__(diffusers_loader.DiffusersPipelineLoader)
    loader.od_config = _reload_config()
    loader.quant_config = None
    pipeline = loader._init_from_load_format(load_format, torch.device("cpu"), "test.Pipeline")
    assert pipeline._omni_weight_reload_prepared
    info = get_layerwise_info(pipeline)
    assert info.restore_device == torch.device("cpu")
    assert info.restore_metadata[0]["weight"].is_meta
    assert not pipeline.weight.is_meta


def test_worker_rejects_pipeline_without_reload_metadata():
    import torch
    from vllm.config.weight_transfer import WeightTransferConfig

    worker = _bare_diffusion_worker(model=torch.nn.Linear(2, 2))
    worker.vllm_config = SimpleNamespace(weight_transfer_config=WeightTransferConfig(backend="ipc"))
    with pytest.raises(RuntimeError, match="no weight-reload metadata"):
        worker._init_weight_transfer_engine()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
