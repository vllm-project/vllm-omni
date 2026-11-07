# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Image routing and lifecycle regressions for the shared SeaCache hook."""

import pytest
import torch

from vllm_omni.diffusion.cache.seacache import SeaCacheBackend
from vllm_omni.diffusion.cache.seacache import backend as backend_module
from vllm_omni.diffusion.cache.selector import get_cache_backend
from vllm_omni.diffusion.data import DiffusionCacheConfig, DiffusionParallelConfig, OmniDiffusionConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("pipeline_name", "extractor_name"),
    [
        ("FluxPipeline", "extract_flux_seacache_context"),
        ("Flux2Pipeline", "extract_flux2_seacache_context"),
        ("Flux2KleinPipeline", "extract_flux2_seacache_context"),
        ("QwenImagePipeline", "extract_qwen_seacache_context"),
        ("QwenImageEditPipeline", "extract_qwen_seacache_context"),
        ("QwenImageEditPlusPipeline", "extract_qwen_seacache_context"),
    ],
)
def test_image_pipeline_routing_and_zero_threshold(pipeline_name, extractor_name, mocker):
    pipeline = type(pipeline_name, (), {})()
    pipeline.transformer = torch.nn.Identity()
    install = mocker.patch.object(backend_module, "apply_sea_cache_hook")
    get_cache_backend("sea_cache", {"sea_threshold": 0}).enable(pipeline)

    config = install.call_args.args[1]
    assert (config.threshold, config.power_exp, config.max_consecutive_cached) == (0, 2.0, 0)
    assert install.call_args.kwargs["extractor_fn"] is getattr(backend_module, extractor_name)


@pytest.mark.parametrize(("power_exp", "max_cached"), [(None, None), (4.0, None), (None, 1), (None, 0)])
def test_model_defaults_and_overrides_do_not_mutate_shared_config(power_exp, max_cached, mocker):
    config = DiffusionCacheConfig(sea_power_exp=power_exp, sea_max_consecutive_cached=max_cached)
    backend = SeaCacheBackend(config)
    install = mocker.patch.object(backend_module, "apply_sea_cache_hook")
    for pipeline_name, default_power, default_cap in [("Flux2Pipeline", 2.0, 0), ("Cosmos3OmniPipeline", 3.0, 2)]:
        pipeline = type(pipeline_name, (), {})()
        pipeline.transformer = torch.nn.Identity()
        backend.enable(pipeline)
        effective = install.call_args.args[1]
        assert effective.power_exp == (default_power if power_exp is None else power_exp)
        assert effective.max_consecutive_cached == (default_cap if max_cached is None else max_cached)
        assert (config.sea_power_exp, config.sea_max_consecutive_cached) == (power_exp, max_cached)


def test_unqualified_image_variant_is_rejected():
    pipeline = type("QwenImageLayeredPipeline", (), {})()
    with pytest.raises(ValueError, match="does not support pipeline"):
        SeaCacheBackend(DiffusionCacheConfig()).enable(pipeline)


@pytest.mark.parametrize("mode", ["sp", "distributed_offload"])
def test_image_parallel_fallback_uses_original_forward_before_extraction(mode, monkeypatch):
    pipeline = type("Flux2Pipeline", (), {})()
    pipeline.transformer = torch.nn.Identity()
    pipeline.transformer.parallel_config = DiffusionParallelConfig()
    pipeline.od_config = OmniDiffusionConfig()

    def unexpected_extraction(*args, **kwargs):
        pytest.fail("parallel fallback must not run the cache extractor")

    monkeypatch.setitem(backend_module._IMAGE_EXTRACTORS, "Flux2Pipeline", unexpected_extraction)
    backend = SeaCacheBackend(DiffusionCacheConfig())
    backend.enable(pipeline)
    # Runtime hooks/configuration may be applied after enabling the backend.
    if mode == "sp":
        pipeline.transformer.parallel_config.sequence_parallel_size = 2
    else:
        pipeline.od_config.enable_distributed_layerwise_offload = True
    value = torch.ones(1, 4, 8)
    assert pipeline.transformer(value) is value
