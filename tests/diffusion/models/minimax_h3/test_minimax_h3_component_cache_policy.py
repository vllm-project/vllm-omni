# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""H3 allocator policy covers staged/unstaged offload and bounded owners."""

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.minimax_h3.encoder import MiniMaxH3Qwen3VLEncoder
from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3AudioVAE, MiniMaxH3VideoVAE
from vllm_omni.diffusion.offloader.module_residency import BoundedAllocatorCache, PinnedModuleStager

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


def _component(kind):
    cls = {"video": MiniMaxH3VideoVAE, "audio": MiniMaxH3AudioVAE, "text": MiniMaxH3Qwen3VLEncoder}[kind]
    component = cls.__new__(cls)
    nn.Module.__init__(component)
    if kind == "text":
        component.vision = nn.Linear(1, 1)
        component.text_model = nn.Linear(1, 1)
    else:
        component.remote = nn.Linear(1, 1)
        component._stager = None
        if kind == "video":
            component._encoder_stager = None
            component._decoder_stager = None
    return component


@pytest.mark.parametrize("kind", ["video", "audio", "text"])
@pytest.mark.parametrize("reuse", [False, True])
def test_manual_offload_keeps_cpu_transition_and_only_changes_cache_release(kind, reuse, monkeypatch, mocker):
    if reuse:
        monkeypatch.setenv("H3_COMPONENT_CACHE_REUSE", "1")
    else:
        monkeypatch.delenv("H3_COMPONENT_CACHE_REUSE", raising=False)
    empty_cache = mocker.patch.object(torch.accelerator, "empty_cache")
    component = _component(kind)
    parameters = list(component.parameters())
    expected_values = [p.detach().clone() for p in parameters]
    component.offload_to_cpu()
    assert all(p.device.type == "cpu" for p in parameters)
    assert all(torch.equal(p, expected) for p, expected in zip(parameters, expected_values))
    assert list(component.parameters()) == parameters
    assert empty_cache.call_count == int(not reuse)


@pytest.mark.parametrize("kind", ["video", "audio", "text"])
def test_manual_offload_keeps_bounded_owner_when_reuse_is_enabled(kind, monkeypatch, mocker):
    monkeypatch.setenv("H3_COMPONENT_CACHE_REUSE", "1")
    empty_cache = mocker.patch.object(torch.accelerator, "empty_cache")
    component = _component(kind)
    retention = mocker.Mock(spec=BoundedAllocatorCache)
    component.set_omni_component_cache(retention)
    component.offload_to_cpu()
    retention.release_if_needed.assert_called_once_with()
    empty_cache.assert_not_called()


def test_default_unstaged_text_offload_preserves_its_existing_purge(monkeypatch, mocker):
    monkeypatch.delenv("H3_COMPONENT_CACHE_REUSE", raising=False)
    empty_cache = mocker.patch.object(torch.accelerator, "empty_cache")
    encoder = _component("text")
    retention = mocker.Mock(spec=BoundedAllocatorCache)
    encoder.set_omni_component_cache(retention)
    encoder.offload_to_cpu()
    empty_cache.assert_called_once_with()
    retention.release_if_needed.assert_not_called()


@pytest.mark.parametrize("kind", ["video", "audio", "text"])
def test_staged_offload_receives_reuse_request(kind, monkeypatch, mocker):
    monkeypatch.setenv("H3_COMPONENT_CACHE_REUSE", "1")
    component = _component(kind)
    stager = mocker.create_autospec(PinnedModuleStager, instance=True)
    if kind == "text":
        component._omni_layerwise_enabled = True
        component._omni_layerwise_hooks = []
        component._omni_non_block_stager = stager
    elif kind == "video":
        component._encoder_stager = stager
        component._decoder_stager = mocker.create_autospec(PinnedModuleStager, instance=True)
    else:
        component._stager = stager
    component.offload_to_cpu()
    stager.offload.assert_called_once_with(release_cache=False)
    if kind == "video":
        component._decoder_stager.offload.assert_called_once_with(release_cache=False)
