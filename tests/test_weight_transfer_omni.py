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

from unittest.mock import MagicMock

import pytest


# --- Config plumbing ---


def test_weight_transfer_config_is_pipeline_wide_field():
    """weight_transfer_config must propagate to every stage's engine args."""
    from vllm_omni.config.stage_config import _PIPELINE_WIDE_ENGINE_FIELDS

    assert "weight_transfer_config" in _PIPELINE_WIDE_ENGINE_FIELDS


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

    assert build_stage_runtime_overrides(1, cli) == {
        "weight_transfer_config": {"backend": "ipc"}
    }
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
    # The pipeline is the load_weights target for the diffusion stage.
    assert engine.model is worker.model_runner.pipeline

    worker.update_weights({"names": []})
    engine.update_weights.assert_called_once_with({"names": []})

    worker.finish_weight_update()
    engine.finish_weight_update.assert_called_once()
    engine.reset_weight_update_target.assert_called_once()
    assert worker._weight_update_active is False


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
