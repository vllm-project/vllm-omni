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
