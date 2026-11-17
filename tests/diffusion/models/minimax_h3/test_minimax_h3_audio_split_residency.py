# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU regression coverage for audio encoder-component staging."""

from contextlib import contextmanager

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as pipeline_module
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3AudioVAE, _AudioVAEPartProxy

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _AudioModel(nn.Module):
    def __init__(self, *, projection: bool = True):
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        if projection:
            self.pre_block = nn.Linear(4, 4)
            self.pre_block.register_buffer("scale", torch.ones(4))
        self.mean_proj = nn.Linear(4, 4)
        self.logs_proj = nn.Linear(4, 4)
        self.dec_in_proj = nn.Linear(4, 4)
        self.decoder = nn.Linear(8, 8)
        self.decoder.register_buffer("filter", torch.ones(8))


class _Remote(nn.Module):
    def __init__(self, *, projection: bool = True):
        super().__init__()
        self.model = _AudioModel(projection=projection)


def _vae(*, projection: bool = True, root_buffer: bool = False, encode_only: bool = False, tied_storage: bool = False):
    vae = object.__new__(MiniMaxH3AudioVAE)
    nn.Module.__init__(vae)
    vae.remote = _Remote(projection=projection)
    vae.model = vae.remote.model
    if root_buffer:
        vae.model.register_buffer("root", torch.ones(1))
    if encode_only:
        del vae.model.decoder
    if tied_storage:
        vae.model.decoder.weight = nn.Parameter(vae.model.encoder.weight.detach())
    vae._device_target = torch.device("cpu")
    vae._stager = vae._encoder_stager = vae._decoder_stager = None
    vae._build_residency_stagers(torch.device("cpu"))
    vae.encoder_component = _AudioVAEPartProxy(vae, "encoder")
    return vae


def _pointer(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


@pytest.mark.parametrize("projection", [False, True])
def test_encoder_load_preserves_all_decoder_storage(projection):
    vae = _vae(projection=projection)
    modules = [vae.model.decoder, vae.model.dec_in_proj, vae.model.logs_proj]
    decoder = [tensor for module in modules for tensor in (*module.parameters(), *module.buffers())]
    pointers = [_pointer(tensor) for tensor in decoder]
    encoder_pointer = _pointer(vae.model.encoder.weight)
    mean_pointer = _pointer(vae.model.mean_proj.weight)
    vae.encoder_component.load_to_device()
    assert vae._encoder_stager.loaded
    assert not vae._decoder_stager.loaded
    assert _pointer(vae.model.encoder.weight) != encoder_pointer
    assert _pointer(vae.model.mean_proj.weight) != mean_pointer
    assert [_pointer(tensor) for tensor in decoder] == pointers
    vae.encoder_component.offload_to_cpu()
    assert _pointer(vae.model.encoder.weight) == encoder_pointer
    assert _pointer(vae.model.mean_proj.weight) == mean_pointer


def test_whole_component_retains_full_load_contract_and_values():
    vae = _vae()
    expected = {name: tensor.clone() for name, tensor in vae.model.state_dict().items()}
    pointers = {name: _pointer(tensor) for name, tensor in vae.model.named_parameters()}
    vae.load_to_device()
    assert vae._encoder_stager.loaded and vae._decoder_stager.loaded
    for name, tensor in vae.model.named_parameters():
        assert _pointer(tensor) != pointers[name]
    vae.offload_to_cpu()
    for name, tensor in vae.model.state_dict().items():
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)


@pytest.mark.parametrize("options", [{"root_buffer": True}, {"encode_only": True}, {"tied_storage": True}])
def test_root_state_and_partial_adapters_keep_whole_staging(options):
    vae = _vae(**options)
    assert vae._stager is not None
    assert vae._encoder_stager is vae._decoder_stager is None
    vae.encoder_component.load_to_device()
    assert vae._stager.loaded
    vae.encoder_component.offload_to_cpu()
    assert not vae._stager.loaded


def _pipeline(vae):
    pipeline = object.__new__(MiniMaxH3Pipeline)
    nn.Module.__init__(pipeline)
    pipeline.audio_vae = vae
    return pipeline


def test_encoding_scope_releases_encoder_after_failure():
    vae = _vae()
    pipeline = _pipeline(vae)
    pipeline._uses_manual_component_offload = lambda component: True
    with pytest.raises(RuntimeError, match="encode failed"):
        with pipeline._audio_encoding_component_on_device(vae):
            assert vae._encoder_stager.loaded
            assert not vae._decoder_stager.loaded
            raise RuntimeError("encode failed")
    assert not vae._encoder_stager.loaded
    assert not vae._decoder_stager.loaded


def test_sequential_encoding_unwraps_to_hooked_audio_module(monkeypatch):
    vae = _vae()
    pipeline = _pipeline(vae)
    pipeline._model_cpu_offload_modules = [vae]
    seen = []

    @contextmanager
    def sequential(component):
        seen.append(component)
        yield

    monkeypatch.setattr(pipeline_module, "sequential_offload_component", sequential)
    with pipeline._audio_encoding_component_on_device(vae):
        pass
    assert seen == [vae]


def test_resident_scope_and_unrelated_components_preserve_placement():
    vae = _vae()
    pipeline = _pipeline(vae)
    pipeline._uses_manual_component_offload = lambda component: False
    with pipeline._audio_encoding_component_on_device(vae):
        assert not vae._encoder_stager.loaded
        assert not vae._decoder_stager.loaded
    other = nn.Linear(2, 2)
    seen = []

    @contextmanager
    def component_scope(component):
        seen.append(component)
        yield

    pipeline._component_on_device = component_scope
    with pipeline._audio_encoding_component_on_device(other):
        pass
    assert seen == [other]


def test_proxy_and_cache_have_one_owner():
    vae = _vae()
    assert list(vae.encoder_component.parameters()) == []
    assert vae.encoder_component.sequential_offload_target is vae
    cache = object()
    vae.encoder_component.set_omni_component_cache(cache)
    assert vae._encoder_stager.cache_retention is cache
    assert vae._decoder_stager.cache_retention is cache


def test_unknown_part_fails_before_loading():
    vae = _vae()
    with pytest.raises(ValueError, match="unknown audio VAE part"):
        vae._load_part_to_device("unknown")
