# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.models.z_image.z_image_transformer import (
    UnifiedPrepare,
    validate_zimage_tp_constraints,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_unified_prepare_applies_padding_masks_before_sharding():
    x = torch.zeros((2, 4, 2))
    cap = torch.zeros((2, 3, 2))
    x_rope = torch.zeros((2, 4, 1))
    cap_rope = torch.zeros((2, 3, 1))
    x_mask = torch.tensor([[1, 0, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    cap_mask = torch.tensor([[1, 1, 0], [1, 0, 1]], dtype=torch.bool)

    *_, unified_mask = UnifiedPrepare()(
        x,
        x_rope,
        x_rope,
        cap,
        cap_rope,
        cap_rope,
        [4, 2],
        [3, 3],
        x_mask,
        cap_mask,
    )

    assert torch.equal(
        unified_mask,
        torch.tensor(
            [[1, 0, 1, 0, 1, 1, 0], [1, 1, 1, 0, 1, 0, 0]],
            dtype=torch.bool,
        ),
    )


def test_validate_zimage_tp_constraints_tp2_ok():
    ffn_hidden_dim, final_out_dims, supported_tp = validate_zimage_tp_constraints(
        dim=3840,
        n_heads=30,
        n_kv_heads=30,
        in_channels=16,
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        tensor_parallel_size=2,
    )
    assert ffn_hidden_dim == 10240
    assert final_out_dims == [64]
    assert supported_tp == [1, 2]


def test_validate_zimage_tp_constraints_tp4_fails_on_heads():
    with pytest.raises(ValueError, match=r"n_heads % tensor_parallel_size"):
        validate_zimage_tp_constraints(
            dim=3840,
            n_heads=30,
            n_kv_heads=30,
            in_channels=16,
            all_patch_size=(2,),
            all_f_patch_size=(1,),
            tensor_parallel_size=4,
        )


def test_validate_zimage_tp_constraints_tp3_fails_on_ffn_hidden_dim():
    with pytest.raises(ValueError, match=r"ffn_hidden_dim % tensor_parallel_size"):
        validate_zimage_tp_constraints(
            dim=3840,
            n_heads=30,
            n_kv_heads=30,
            in_channels=16,
            all_patch_size=(2,),
            all_f_patch_size=(1,),
            tensor_parallel_size=3,
        )
