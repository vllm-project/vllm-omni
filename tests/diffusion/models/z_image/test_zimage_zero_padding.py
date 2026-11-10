# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.models.z_image.z_image_transformer import (
    ZImageTransformer2DModel,
    _has_image_alignment_padding,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _bare_transformer() -> ZImageTransformer2DModel:
    transformer = object.__new__(ZImageTransformer2DModel)
    nn.Module.__init__(transformer)
    return transformer


def test_image_padding_detection_uses_host_shape_metadata():
    assert not _has_image_alignment_padding([4096], [(1, 128, 128)], 2, 1)
    assert _has_image_alignment_padding([4128], [(1, 128, 128)], 2, 1)


def test_patchify_does_not_materialize_empty_image_padding():
    transformer = _bare_transformer()
    caption = [torch.arange(12, dtype=torch.float32).reshape(3, 4)]
    image = torch.arange(32, dtype=torch.float32).reshape(1, 1, 4, 8)

    outputs = transformer.patchify_and_embed(
        [image],
        caption,
        patch_size=1,
        f_patch_size=1,
    )

    assert outputs[0][0].shape == (32, 1)
    assert outputs[0][0].data_ptr() == image.data_ptr()


def test_patchify_preserves_required_alignment_padding():
    transformer = _bare_transformer()
    caption = [torch.arange(12, dtype=torch.float32).reshape(3, 4)]
    image = torch.arange(3, dtype=torch.float32).reshape(1, 1, 1, 3)

    outputs = transformer.patchify_and_embed(
        [image],
        caption,
        patch_size=1,
        f_patch_size=1,
    )

    assert outputs[0][0].shape == (32, 1)
    assert torch.equal(outputs[0][0][:3, 0], torch.arange(3, dtype=torch.float32))
    assert torch.equal(outputs[0][0][3:, 0], torch.full((29,), 2.0))
