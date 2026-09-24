import pytest

from vllm_omni.diffusion.models.hunyuan_image3 import layers as selected_layers
from vllm_omni.diffusion.models.hunyuan_image3.layers.native.transformer_blocks import (
    ResBlock as NativeResBlock,
)
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.skipif(
    current_omni_platform.is_cuda(),
    reason="Native ResBlock dispatch is selected only on non-CUDA platforms",
)
def test_non_cuda_dispatch_selects_native_resblock() -> None:
    assert selected_layers.ResBlock is NativeResBlock
