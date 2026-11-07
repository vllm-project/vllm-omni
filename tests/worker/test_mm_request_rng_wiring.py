# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request RNG capability is initialized by the production model-load path."""

import pytest
import torch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_omni.model_executor.mm_request_rng import RandomDrawSpec, RequestRNGProvider
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _RNGModel(torch.nn.Module):
    supports_mm_request_rng = True

    def bind_mm_rng_provider(self, provider: RequestRNGProvider) -> None:
        self.provider = provider

    def advance_mm_rng(self, generator: torch.Generator, spec: RandomDrawSpec) -> None:
        raise AssertionError("Loading a model must not consume a request's random stream")


@pytest.mark.parametrize("enabled", [False, True])
def test_load_model_binds_request_rng_only_for_opted_in_models(mocker, enabled):
    model = _RNGModel() if enabled else torch.nn.Identity()
    runner = object.__new__(OmniGPUModelRunner)

    def load(instance):
        instance.model = model

    mocker.patch.object(GPUModelRunner, "load_model", load)
    for hook in (
        "_maybe_enable_output_token_ids_for_model_sampler",
        "_init_talker_mtp",
        "_prewarm_attention_capture_workspaces",
        "_report_model_local_kv",
        "_warn_unexposed_stage_hooks",
    ):
        mocker.patch.object(runner, hook)

    runner.load_model()

    assert runner._mm_rng_contexts == {}
    if enabled:
        assert isinstance(runner._mm_rng_provider, RequestRNGProvider)
        assert model.provider is runner._mm_rng_provider
        assert runner._mm_rng_provider.streams == {}
    else:
        assert runner._mm_rng_provider is None
