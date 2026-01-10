import pytest

from vllm_omni.diffusion.models.z_image.z_image_transformer import validate_zimage_tp_constraints


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
