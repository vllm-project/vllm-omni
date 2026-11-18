# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import copy

import pytest
import torch
from torch import nn

from tests.diffusion.offloader.helpers import patch_offload_runtime
from tests.diffusion.test_mossimage_slg import pipeline, request, sampling
from vllm_omni.diffusion.models.mossimage.mossimage_transformer import MOSSImageDenoiseBody
from vllm_omni.diffusion.models.mossimage.pipeline_mossimage import MOSSImagePipeline
from vllm_omni.diffusion.offloader import distributed_layerwise_backend, plan_resolver
from vllm_omni.diffusion.offloader.base import OffloadConfig, OffloadStrategy
from vllm_omni.diffusion.offloader.distributed_layerwise_backend import DistributedLayerwiseOffloadBackend
from vllm_omni.diffusion.offloader.plan_resolver import resolve_offload_plan
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import StepRequestState

pytestmark = [pytest.mark.cpu, pytest.mark.diffusion, pytest.mark.core_model]


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 4) / 8)
        self.bias = nn.Parameter(torch.randn(4) / 8)

    def forward(self, x, rope=None, modulation=None, key_mask=None):
        return x + x @ self.weight + self.bias


class TextProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.blocks = nn.ModuleList([Block(), Block()])

    def forward(self, x):
        x = self.proj(x)
        for block in self.blocks:
            x = block(x)
        return x


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Block() for _ in range(6)])
        self.text_embedder = TextProjection()
        self.denoise_body = MOSSImageDenoiseBody(self.layers)

    def forward(self, x, skip_layers=()):
        return self.denoise_body(self.text_embedder(x), None, None, skip_layers=skip_layers)


class ResidencyController:
    def __init__(self, events: list[str]):
        self.events = events

    def load_resident_layers(self) -> None:
        self.events.append("load")

    def offload_resident_layers(self) -> None:
        self.events.append("release")


def offload_pipeline():
    model = MOSSImagePipeline.__new__(MOSSImagePipeline)
    nn.Module.__init__(model)
    model.transformer = Transformer()
    model.text_encoder = nn.Module()
    model.text_encoder.language_model = nn.Module()
    model.text_encoder.language_model.layers = nn.ModuleList([Block(), Block()])
    model.vae = nn.Identity()
    return model


def config(resident=0, dp_size=1):
    return OffloadConfig(
        strategy=OffloadStrategy.DISTRIBUTED_LAYER_WISE,
        pin_cpu_memory=False,
        components=frozenset({"dit", "text_encoder"}),
        dlo_use_allgather=False,
        dlo_resident_layers=resident,
        dp_size=dp_size,
    )


def test_declared_plan_resolves_each_block_once_including_large_aliases(monkeypatch):
    model = offload_pipeline()
    original_size = plan_resolver._module_size_mb
    monkeypatch.setattr(
        plan_resolver,
        "_module_size_mb",
        lambda module: 2048 if module is model.transformer.denoise_body else original_size(module),
    )
    resolved = resolve_offload_plan(model, config(resident=2))
    dit = resolved.dits[0]
    assert dit.stacks[0].blocks == tuple(model.transformer.layers)
    assert dit.stacks[0].resident_head == 2
    assert [child.path for child in dit.children] == ["transformer.text_embedder"]
    assert dit.children[0].stacks[0].blocks == tuple(model.transformer.text_embedder.blocks)
    assert resolved.encoders[0].stacks[0].blocks == tuple(model.text_encoder.language_model.layers)


@pytest.mark.parametrize("resident", [0, 2, 6])
@pytest.mark.parametrize("skip_layers", [(2,), (0,), (5,), (0, 1), (2, 3), (0, 2, 4), tuple(range(6))])
@torch.no_grad()
def test_dlo_preserves_skipped_and_normal_forwards_and_restores_weights(monkeypatch, mocker, resident, skip_layers):
    patch_offload_runtime(monkeypatch, distributed_layerwise_backend.current_omni_platform, synchronize=True)
    model = offload_pipeline()
    reference = copy.deepcopy(model)
    original = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    # Moving this alias wholesale defeats streaming and can corrupt a second
    # ring's host masters. Placement must exclude the parent's block tensors.
    mocker.patch.object(
        model.transformer.denoise_body, "to", side_effect=RuntimeError("bulk placement of aliased blocks")
    )
    backend = DistributedLayerwiseOffloadBackend(config(resident), torch.device("cpu"))
    backend.enable(model)
    assert len(backend._all_hook_groups) == (2 if resident == 6 else 3)
    x = torch.randn(2, 3, 4)
    try:
        for skip in ((), skip_layers, (1,), (0,), (2, 3), (0, 1), skip_layers, ()):
            # Exercise the shared device slots changing between encoder,
            # text-refinement and main-DiT groups, including skipped heads.
            encoded = x
            for block in model.text_encoder.language_model.layers:
                encoded = block(encoded)
            expected_encoded = x
            for block in reference.text_encoder.language_model.layers:
                expected_encoded = block(expected_encoded)
            torch.testing.assert_close(encoded, expected_encoded, atol=0, rtol=0)
            with model._dit_residency():
                actual = model.transformer(encoded, skip_layers=skip)
            expected = reference.transformer(expected_encoded, skip_layers=skip)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    finally:
        backend.disable()
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, original[name], atol=0, rtol=0)
    assert not model.text_encoder._omni_layerwise_enabled


@pytest.mark.parametrize("stepwise", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_resident_blocks_are_released_after_denoise_even_on_failure(stepwise, fail):
    model = pipeline()
    events: list[str] = []
    model._dlo_residency_controller = ResidencyController(events)
    original_predict = model.predict_noise_with_slg

    def predict(*args, **kwargs):
        assert events[-1] == "load"
        if fail:
            raise RuntimeError("injected denoise failure")
        return original_predict(*args, **kwargs)

    model.predict_noise_with_slg = predict
    params = sampling()
    if stepwise:
        state = StepRequestState(request_id="req", sampling=params, prompt="positive")
        model.prepare_encode(state)

        def run():
            return model.denoise_step(InputBatch.make_batch([state]))
    else:
        original_decode = model._vae_decode

        def decode(latents):
            assert events[-1] == "release"
            return original_decode(latents)

        model._vae_decode = decode

        def run():
            return model(DiffusionRequestBatch([request(params)]))

    if fail:
        with pytest.raises(RuntimeError, match="injected denoise"):
            run()
    else:
        run()
    assert events == ["load", "release"]
