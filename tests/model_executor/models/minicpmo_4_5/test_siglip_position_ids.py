# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import SiglipVisionEmbeddings

pytestmark = [pytest.mark.core_model]


@dataclass(frozen=True)
class _VisionConfig:
    hidden_size: int = 4
    image_size: int = 8
    patch_size: int = 2
    num_channels: int = 3


def _reference_position_ids(
    embeddings: SiglipVisionEmbeddings,
    patch_attention_mask: torch.BoolTensor,
    tgt_sizes: torch.IntTensor | None,
) -> torch.Tensor:
    batch_size = patch_attention_mask.size(0)
    position_ids = torch.zeros((batch_size, patch_attention_mask[0].numel()), dtype=torch.long)
    boundaries = torch.arange(
        1 / embeddings.num_patches_per_side,
        1.0,
        1 / embeddings.num_patches_per_side,
    )

    for batch_idx, patch_mask in enumerate(patch_attention_mask):
        if tgt_sizes is None:
            patch_grid_height = int(patch_mask[:, 0].sum())
            patch_grid_width = int(patch_mask[0].sum())
        else:
            patch_grid_height = int(tgt_sizes[batch_idx, 0])
            patch_grid_width = int(tgt_sizes[batch_idx, 1])

        fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / patch_grid_height)
        fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / patch_grid_width)
        bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
        bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)
        grid_ids = (bucket_coords_h[:, None] * embeddings.num_patches_per_side + bucket_coords_w).flatten()
        position_ids[batch_idx, patch_mask.flatten()] = grid_ids

    return position_ids


def _mixed_patch_masks() -> torch.BoolTensor:
    return torch.tensor(
        [
            [
                [True, True, True, False],
                [True, True, True, False],
                [False, False, False, False],
            ],
            [
                [True, True, True, True],
                [False, False, False, False],
                [False, False, False, False],
            ],
            [
                [True, False, True, True],
                [False, True, False, True],
                [True, False, False, False],
            ],
        ],
        dtype=torch.bool,
    )


@pytest.mark.parametrize("num_patches_per_side", [1, 2, 7, 16, 70])
@pytest.mark.cpu
def test_grid_ids_match_bucketized_reference(num_patches_per_side: int) -> None:
    embeddings = SiglipVisionEmbeddings(_VisionConfig(image_size=num_patches_per_side * 2))
    boundaries = torch.arange(
        1 / num_patches_per_side,
        1.0,
        1 / num_patches_per_side,
    )
    target_shapes = [
        (1, 1),
        (max(1, num_patches_per_side - 1), num_patches_per_side),
        (num_patches_per_side, num_patches_per_side + 1),
        (num_patches_per_side * 7, num_patches_per_side * 7),
        (num_patches_per_side * 2 + 1, 3),
    ]

    for patch_grid_height, patch_grid_width in target_shapes:
        fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / patch_grid_height)
        fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / patch_grid_width)
        bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
        bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)
        expected = (bucket_coords_h[:, None] * num_patches_per_side + bucket_coords_w).flatten()

        actual = embeddings._create_grid_position_ids(
            patch_grid_height,
            patch_grid_width,
            boundaries,
        )

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.cpu
def test_position_ids_match_reference_for_mixed_and_repeated_grids(mocker) -> None:
    embeddings = SiglipVisionEmbeddings(_VisionConfig())
    patch_attention_mask = _mixed_patch_masks()
    tgt_sizes = torch.tensor([[2, 3], [1, 4], [2, 3]], dtype=torch.int32)
    grid_builder = mocker.spy(embeddings, "_create_grid_position_ids")

    actual = embeddings._create_position_ids(patch_attention_mask, tgt_sizes, device=torch.device("cpu"))
    expected = _reference_position_ids(embeddings, patch_attention_mask, tgt_sizes)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert grid_builder.call_count == 2


@pytest.mark.cpu
def test_position_ids_match_reference_without_target_sizes() -> None:
    embeddings = SiglipVisionEmbeddings(_VisionConfig())
    patch_attention_mask = torch.tensor(
        [
            [
                [True, True, True, False],
                [True, True, True, False],
                [False, False, False, False],
            ],
            [
                [True, True, False, False],
                [True, True, False, False],
                [True, True, False, False],
            ],
        ],
        dtype=torch.bool,
    )

    actual = embeddings._create_position_ids(patch_attention_mask, None, device=torch.device("cpu"))
    expected = _reference_position_ids(embeddings, patch_attention_mask, None)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.cpu
def test_forward_matches_reference_position_embedding() -> None:
    torch.manual_seed(0)
    embeddings = SiglipVisionEmbeddings(_VisionConfig())
    pixel_values = torch.randn(3, 3, 6, 8)
    patch_attention_mask = _mixed_patch_masks()
    tgt_sizes = torch.tensor([[2, 3], [1, 4], [2, 3]], dtype=torch.int32)

    with torch.no_grad():
        patch_embeds = embeddings.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        reference_ids = _reference_position_ids(embeddings, patch_attention_mask, tgt_sizes)
        expected = patch_embeds + embeddings.position_embedding(reference_ids)
        actual = embeddings(pixel_values, patch_attention_mask, tgt_sizes)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_position_ids_match_reference_for_processor_shapes(mocker) -> None:
    embeddings = SiglipVisionEmbeddings(_VisionConfig(image_size=140)).to(device="cuda")
    target_shapes = [(40, 25), (32, 32), (40, 25), (28, 37)]
    max_num_patches = max(height * width for height, width in target_shapes)
    patch_attention_mask = torch.zeros((len(target_shapes), 1, max_num_patches), dtype=torch.bool)
    for batch_idx, (height, width) in enumerate(target_shapes):
        patch_attention_mask[batch_idx, 0, : height * width] = True

    # Keep the valid-patch count while exercising non-contiguous padding.
    patch_attention_mask[0, 0, 5] = False
    patch_attention_mask[0, 0, -1] = True
    tgt_sizes = torch.tensor(target_shapes, dtype=torch.int32)
    expected = _reference_position_ids(embeddings, patch_attention_mask, tgt_sizes)
    grid_builder = mocker.spy(embeddings, "_create_grid_position_ids")

    actual = embeddings._create_position_ids(
        patch_attention_mask.to(device="cuda"),
        tgt_sizes.to(device="cuda"),
        device=torch.device("cuda"),
    )

    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    assert grid_builder.call_count == 3
    assert all(call.args[2].device.type == "cpu" for call in grid_builder.call_args_list)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_forward_matches_reference_for_processor_shape() -> None:
    torch.manual_seed(0)
    embeddings = SiglipVisionEmbeddings(_VisionConfig(image_size=140)).to(device="cuda")
    pixel_values = torch.randn(1, 3, 80, 50, device="cuda")
    patch_attention_mask = torch.ones((1, 1, 40 * 25), dtype=torch.bool, device="cuda")
    tgt_sizes = torch.tensor([[40, 25]], dtype=torch.int32, device="cuda")

    with torch.no_grad():
        patch_embeds = embeddings.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        reference_ids = _reference_position_ids(
            embeddings,
            patch_attention_mask.cpu(),
            tgt_sizes.cpu(),
        ).to(device="cuda")
        expected = patch_embeds + embeddings.position_embedding(reference_ids)
        actual = embeddings(pixel_values, patch_attention_mask, tgt_sizes)

    torch.testing.assert_close(actual, expected)
