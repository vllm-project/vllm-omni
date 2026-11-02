# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Runner shutdown releases the encoder CUDA graph manager before upstream teardown."""

import pytest
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_shutdown_releases_encoder_cudagraph_manager(mocker):
    """Encoder graphs, their pool and the model they reference go before upstream teardown."""
    runner = object.__new__(GPUARModelRunner)
    runner.omni_prefix_cache = None
    runner._sampled_token_ids_cpu_override = None
    manager = mocker.Mock()
    runner.encoder_cudagraph_manager = manager
    mocker.patch.object(runner, "_clear_duplex_sampling")
    events = []
    manager.clear.side_effect = lambda: events.append("clear")
    # Only upstream teardown is stubbed: the release can sit in any omni layer.
    mocker.patch.object(GPUModelRunner, "shutdown", side_effect=lambda self: events.append("upstream"), autospec=True)

    runner.shutdown()

    assert runner.encoder_cudagraph_manager is None
    assert events == ["clear", "upstream"]
