# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""H3 allocator policy covers staged/unstaged offload and bounded owners."""

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.minimax_h3 import component_cache_policy
from vllm_omni.diffusion.models.minimax_h3.encoder import MiniMaxH3Qwen3VLEncoder
from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3AudioVAE, MiniMaxH3VideoVAE
from vllm_omni.diffusion.offloader.module_residency import BoundedAllocatorCache, PinnedModuleStager

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


@pytest.mark.parametrize("value", [True, False])
def test_component_cache_reuse_accepts_json_booleans(value):
    assert component_cache_policy.resolve_component_cache_reuse({"component_cache_reuse": value}) is value


def test_component_cache_reuse_defaults_off_and_ignores_removed_environment_variable(monkeypatch):
    monkeypatch.setenv("H3_COMPONENT_CACHE_REUSE", "1")
    assert component_cache_policy.resolve_component_cache_reuse(None) is False
    assert component_cache_policy.resolve_component_cache_reuse({}) is False


@pytest.mark.parametrize("value", [1, 0, "true", "false", None])
def test_component_cache_reuse_rejects_non_booleans(value):
    with pytest.raises(ValueError, match="component_cache_reuse"):
        component_cache_policy.resolve_component_cache_reuse({"component_cache_reuse": value})


def _component(kind, reuse=False):
    cls = {"video": MiniMaxH3VideoVAE, "audio": MiniMaxH3AudioVAE, "text": MiniMaxH3Qwen3VLEncoder}[kind]
    component = cls.__new__(cls)
    nn.Module.__init__(component)
    component._component_cache_reuse = reuse
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
def test_manual_offload_keeps_cpu_transition_and_only_changes_cache_release(kind, reuse, mocker):
    empty_cache = mocker.patch.object(torch.accelerator, "empty_cache")
    component = _component(kind, reuse)
    parameters = list(component.parameters())
    expected_values = [p.detach().clone() for p in parameters]
    component.offload_to_cpu()
    assert all(p.device.type == "cpu" for p in parameters)
    assert all(torch.equal(p, expected) for p, expected in zip(parameters, expected_values))
    assert list(component.parameters()) == parameters
    assert empty_cache.call_count == int(not reuse)


@pytest.mark.parametrize("kind", ["video", "audio", "text"])
def test_manual_offload_keeps_bounded_owner_when_reuse_is_enabled(kind, mocker):
    empty_cache = mocker.patch.object(torch.accelerator, "empty_cache")
    component = _component(kind, True)
    retention = mocker.Mock(spec=BoundedAllocatorCache)
    component.set_omni_component_cache(retention)
    component.offload_to_cpu()
    retention.release_if_needed.assert_called_once_with()
    empty_cache.assert_not_called()


def test_default_unstaged_text_offload_preserves_its_existing_purge(mocker):
    empty_cache = mocker.patch.object(torch.accelerator, "empty_cache")
    encoder = _component("text")
    retention = mocker.Mock(spec=BoundedAllocatorCache)
    encoder.set_omni_component_cache(retention)
    encoder.offload_to_cpu()
    empty_cache.assert_called_once_with()
    retention.release_if_needed.assert_not_called()


@pytest.mark.parametrize("kind", ["video", "audio", "text"])
def test_staged_offload_receives_reuse_request(kind, mocker):
    component = _component(kind, True)
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


@pytest.mark.parametrize("part", ["encoder", "decoder"])
@pytest.mark.parametrize("reuse", [False, True])
def test_video_part_offload_receives_reuse_request(part, reuse, mocker):
    component = _component("video", reuse)
    stager = mocker.create_autospec(PinnedModuleStager, instance=True)
    other_stager = mocker.create_autospec(PinnedModuleStager, instance=True)
    component._encoder_stager, component._decoder_stager = (
        (stager, other_stager)
        if part == "encoder"
        else (
            other_stager,
            stager,
        )
    )
    component._offload_part_to_cpu(part)
    stager.offload.assert_called_once_with(release_cache=not reuse)
    other_stager.offload.assert_not_called()
