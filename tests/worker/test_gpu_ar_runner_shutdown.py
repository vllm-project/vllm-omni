# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""GPUARModelRunner.shutdown releases the encoder CUDA graph manager. Pure CPU."""

import gc
import weakref

import pytest
import vllm.v1.worker.gpu_model_runner as upstream_v1

from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Model:
    pass


class _Manager:
    def __init__(self, model):
        self.model = model
        self.cleared = False

    def clear(self):
        self.cleared = True


def _runner(monkeypatch, manager):
    runner = object.__new__(GPUARModelRunner)
    runner.talker_mtp = None
    runner.has_talker_mtp = False
    runner.omni_prefix_cache = None
    runner.encoder_cudagraph_manager = manager
    runner.model = manager.model if manager is not None else None
    monkeypatch.setattr(GPUARModelRunner, "_clear_duplex_sampling", lambda self: None)

    def upstream_shutdown(self):
        self.model = None

    monkeypatch.setattr(upstream_v1.GPUModelRunner, "shutdown", upstream_shutdown)
    return runner


def test_shutdown_releases_the_encoder_graph_manager_and_its_model(monkeypatch):
    model = _Model()
    model_ref = weakref.ref(model)
    manager = _Manager(model)
    runner = _runner(monkeypatch, manager)
    del model

    runner.shutdown()
    gc.collect()

    assert manager.cleared
    assert runner.encoder_cudagraph_manager is None
    del manager
    gc.collect()
    assert model_ref() is None


def test_shutdown_without_an_encoder_graph_manager(monkeypatch):
    runner = _runner(monkeypatch, None)

    runner.shutdown()

    assert runner.encoder_cudagraph_manager is None
