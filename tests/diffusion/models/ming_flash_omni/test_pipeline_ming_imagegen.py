# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.ming_flash_omni.pipeline_ming_imagegen import (
    MingImagePipeline,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_MODULE = "vllm_omni.diffusion.models.ming_flash_omni.pipeline_ming_imagegen"


@pytest.mark.parametrize(
    "cache_config",
    [
        pytest.param({}, id="default-coefficients"),
        pytest.param(
            {"coefficients": [1.0, -0.5, 0.1, -0.01, 0.001]},
            id="custom-coefficients",
        ),
    ],
)
def test_constructor_rejects_teacache_before_model_download(mocker, cache_config):
    download_weights = mocker.patch(f"{_MODULE}.download_weights_from_hf_specific")
    od_config = OmniDiffusionConfig(
        model="remote-ming-model",
        cache_backend="tea_cache",
        cache_config=cache_config,
    )

    with pytest.raises(
        NotImplementedError,
        match="TeaCache is not supported by MingImagePipeline",
    ):
        MingImagePipeline(od_config=od_config)

    download_weights.assert_not_called()


def test_ming_image_postprocess_defaults_when_vae_config_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Ming's Flux-format VAE config is optional: a missing file must fall
    back to the default scale factor instead of failing fast."""

    class FakeImageProcessor:
        def __init__(self, *, vae_scale_factor: int, **kwargs) -> None:
            FakeImageProcessor.captured = vae_scale_factor

        def postprocess(self, images):
            return images

    import vllm_omni.diffusion.models.ming_flash_omni.pipeline_ming_imagegen as ming

    monkeypatch.setattr(ming, "VaeImageProcessor", FakeImageProcessor)
    postprocess = ming.get_ming_image_post_process_func(OmniDiffusionConfig(model=str(tmp_path), output_type="pil"))
    import torch

    postprocess(torch.zeros(1, 3, 4, 4))
    assert FakeImageProcessor.captured == 16  # default 8, doubled by the factory
