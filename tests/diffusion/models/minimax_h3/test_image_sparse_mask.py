# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts against matching MindIE-SD source; no media or native kernels.

Set MINDIE_SD_SOURCE to the repository containing mindiesd/. Source extraction
omits NPU imports while executing the actual permutation, pooling, mask and
public parameter validation functions.
"""

import ast
import logging
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Optional

import pytest
import torch
from einops import rearrange

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture
def mindie():
    root = os.environ.get("MINDIE_SD_SOURCE", "")
    if not root:
        pytest.skip("MINDIE_SD_SOURCE must identify the matching MindIE-SD source")
    directory = Path(root) / "mindiesd/layers/flash_attn"
    scope = {
        "torch": torch,
        "math": math,
        "rearrange": rearrange,
        "Mapping": Mapping,
        "Sequence": Sequence,
        "Optional": Optional,
        "ParametersInvalid": ValueError,
        "is_a5_device": lambda: True,
        "logger": logging.getLogger(__name__),
        "MAX_TOKEN": 2147483647,
        "A5_V3_INNER_PRECISE": 4,
    }
    for filename in ("sparse_flash_attn_rf_v2.py", "sparse_flash_attn.py"):
        path = directory / filename
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[future, *functions], type_ignores=[])), str(path), "exec"
            ),
            scope,
        )
    return scope


@pytest.mark.parametrize("layout", ["BSND", "BNSD"])
def test_permutation_and_mask_follow_all_video_queries(mindie, layout):
    images = [{"start": 0, "latent_shape": [16, 24]}]
    videos = [
        {"start": 384, "latent_shape": [3, 8, 16]},
        {"start": 768, "latent_shape": [4, 8, 16]},
    ]
    q = torch.randn(1, 1408, 2, 8, generator=torch.Generator().manual_seed(5))
    if layout == "BNSD":
        q = q.transpose(1, 2)
    prepared = mindie["do_multi_span_tensor_rearrange_pooling"](
        q, q, q, videos, 128, layout, images, return_reference_plan=True
    )
    qr, _, _, pool, inverse, dense, first, plan = prepared
    axis = 1 if layout == "BSND" else 2
    torch.testing.assert_close(qr.index_select(axis, inverse), q)
    groups, targets = plan
    assert groups == ((0, 1, 2),)
    # Reference-video and target-video rows both participate; each first frame is protected.
    assert {4, 5, 7, 8, 9}.issubset(targets)
    eligible = sorted(set(targets) - dense - first)
    assert eligible == [4, 5, 7, 8, 9]
    mask = mindie["get_multi_span_blockwise_mask"](pool, 0.8, 1.0, dense, first, layout, True, plan, 0.9)
    qp, kp, _ = torch.chunk(pool, 3, dim=0)
    if layout == "BSND":
        scores = torch.einsum("blnd,bsnd->bnls", qp, kp)
    else:
        scores = torch.einsum("bnld,bnsd->bnls", qp, kp)
    for head in range(2):
        for row in eligible:
            expected = int(scores[0, head, row, :3].argmax())
            assert mask[0, head, row, :3].sum().item() == 1
            assert mask[0, head, row, expected].item() == 1
    assert mask[:, :, sorted(dense | first), :].all()


@pytest.mark.parametrize("shape", [(1, 3, 5), (1, 8, 16)])
@pytest.mark.parametrize("layout", ["BSND", "BNSD"])
def test_single_frame_rearrangement_roundtrip(mindie, shape, layout):
    q = torch.arange(math.prod(shape) * 2).reshape(1, math.prod(shape), 1, 2)
    if layout == "BNSD":
        q = q.transpose(1, 2)
    arranged = mindie["rearrange_with_remaining"](q, shape, shape, layout)
    restored = mindie["inv_rearrange_with_remaining"](arranged, shape, shape, layout)
    torch.testing.assert_close(restored, q)


@pytest.mark.parametrize("drop,keep", [(0.0, 3), (0.5, 2), (0.9, 1), (1.0, 1)])
def test_image_drop_rounding_and_protected_rows(mindie, drop, keep):
    pool = torch.randn(3, 8, 2, 4, generator=torch.Generator().manual_seed(3))
    mask = mindie["get_multi_span_blockwise_mask"](
        pool, 0.8, 1.0, {0, 1, 2}, {3}, "BSND", True, (((0, 1, 2),), (4, 5, 6, 7)), drop
    )
    assert (mask[:, :, 4:, :3].sum(-1) == keep).all()
    assert mask[:, :, :4, :].all()


@pytest.mark.parametrize(
    "image_spans,image_sparsity",
    [
        ([], None),
        (None, 0.9),
        ([], True),
        ([], float("nan")),
        ([], -0.1),
        ([], 1.1),
    ],
)
def test_public_api_rejects_invalid_image_pairs(mindie, image_spans, image_sparsity):
    q = torch.zeros(1, 128, 1, 8)
    with pytest.raises(ValueError):
        mindie["sparse_attention"](
            q,
            q,
            q,
            input_layout="BSND",
            sparse_type="rf_v2",
            video_spans=[],
            image_spans=image_spans,
            image_sparsity=image_sparsity,
        )


@pytest.mark.parametrize("precision", ["bf16", "mix"])
def test_public_api_forwards_image_parameters_to_v3(mindie, precision):
    observed = []

    def execute(q, k, v, **kwargs):
        observed.append(kwargs)
        return q, None

    mindie["bsa_sparse_attention_v3"] = execute
    q = torch.zeros(1, 256, 1, 128, dtype=torch.bfloat16)
    images = [{"start": 0, "latent_shape": [8, 16]}]
    videos = [{"start": 128, "latent_shape": [1, 8, 16]}]
    result = mindie["sparse_attention"](
        q,
        q,
        q,
        input_layout="BSND",
        sparse_type="rf_v2",
        precision=precision,
        video_spans=videos,
        image_spans=images,
        image_sparsity=0.9,
    )
    assert result is q
    assert observed[0]["image_spans"] == images
    assert observed[0]["image_sparsity"] == 0.9
    assert observed[0]["video_spans"] == videos
    assert observed[0]["precision"] == precision
