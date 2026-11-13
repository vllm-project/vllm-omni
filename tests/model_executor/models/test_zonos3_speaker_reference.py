# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Offline speaker network parity; no claim of release-weight audio parity."""

import ast
import math
import os
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from vllm_omni.model_executor.models.zonos3.speaker_encoder import ASTP, MECT, MECTSpeakerNetwork

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def official_classes():
    root = os.environ.get("ZONOS3_REFERENCE_ROOT")
    if not root:
        pytest.skip("Set ZONOS3_REFERENCE_ROOT for official MECT arithmetic checks")
    directory = Path(root) / "model/src/zonos3/models/third_party/antspeaker/models"
    namespace = {"torch": torch, "nn": nn, "F": F, "math": math, "Tensor": torch.Tensor}
    for path in (directory / "backbones/moe.py", directory / "backbones/mect.py", directory / "poolings.py"):
        tree = ast.parse(path.read_text())
        # Keep original method bodies; remove only package registry/imports.
        tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        for node in tree.body:
            node.decorator_list = []
        exec(compile(tree, str(path), "exec"), namespace)
    return namespace


@torch.inference_mode()
def test_offline_mect_backbone_pooling_and_speaker_vector_match_source():
    official = official_classes()
    # Four actual stages with spatial downsampling, MoE and temporal resampling;
    # reduced channels avoid allocating the full release network in CPU CI.
    stages = {
        "in_channels": 2,
        "feat_dim": 8,
        "block1": [(2, 2, 8, 1, 1, True, 2)],
        "block2": [(2, 4, 8, 1, (2, 1), True, 2), (1, 4, 8, 1, (1, 2), True, 1)],
        "block3": [(2, 8, 8, 1, (2, 1), True, 1)],
        "block4": [(2, 16, 8, 1, (2, 1), True, 1)],
        "moe_type": "token",
        "num_experts_token": 4,
    }
    torch.manual_seed(87)
    reference = official["MECT"](**stages).eval()
    native = MECT(**stages).eval()
    native.load_state_dict(reference.state_dict(), strict=True)
    reference_pool = official["ASTP"](16).eval()
    network = MECTSpeakerNetwork(native).eval()
    network.encoding_layers.load_state_dict(reference_pool.state_dict(), strict=True)
    for batch in (1, 4):
        features = torch.randn(batch, 8, 24)
        centered = features - features.mean(-1, keepdim=True)
        expected_backbone = reference(centered)
        torch.testing.assert_close(native(centered), expected_backbone, rtol=0, atol=0)
        expected = network.linear(network.bn(reference_pool(expected_backbone)))
        torch.testing.assert_close(network(features), expected, rtol=0, atol=0)
    assert isinstance(network.encoding_layers, ASTP)
