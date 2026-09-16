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

    assert install.call_args.args[1].threshold == 0
    assert install.call_args.kwargs["extractor_fn"] is getattr(backend_module, extractor_name)


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
