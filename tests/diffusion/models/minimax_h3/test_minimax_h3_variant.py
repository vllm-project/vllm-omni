# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The variant decides which conditioning features a weight set serves."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3.fasth3_checkpoint import FastH3CheckpointSpec, FastH3OmniCheckpointSpec
from vllm_omni.diffusion.models.minimax_h3.lora import parse_turbo_filename
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from vllm_omni.diffusion.models.minimax_h3.variant import (
    BASE_H3,
    FASTH3_OMNI,
    FASTH3_PREVIEW,
    FASTH3_V2,
    H3Feature,
    control_task,
)
from vllm_omni.errors import OmniClientError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

REF2V_TURBO = parse_turbo_filename("minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors")


def _pipeline(checkpoint=None, fused=None):
    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline._fasth3_checkpoint = checkpoint
    pipeline._fasth3 = fused
    return pipeline


@pytest.mark.parametrize("variant", [BASE_H3, FASTH3_OMNI])
def test_reference_weights_serve_every_conditioning_feature(variant):
    for feature in H3Feature:
        assert variant.supports(feature)
        variant.require(feature)


@pytest.mark.parametrize("variant", [FASTH3_V2, FASTH3_PREVIEW])
def test_vsa_students_name_why_guides_are_off(variant):
    assert variant.supports(H3Feature.CONTROL)
    with pytest.raises(OmniClientError, match="timeline guides require base H3 weights or FastH3 Omni"):
        variant.require(H3Feature.TIMELINE_GUIDES)


def test_an_adapter_keeps_the_features_of_the_weights_under_it():
    adapted = BASE_H3.with_adapter(REF2V_TURBO.filename)
    assert adapted.name == REF2V_TURBO.filename
    assert all(adapted.supports(feature) for feature in H3Feature)
    assert not FASTH3_V2.with_adapter("x").supports(H3Feature.TIMELINE_GUIDES)


@pytest.mark.parametrize(
    ("partition", "task"),
    [("fl2va", "t2va"), ("ref2va", "ref2va"), ("combined", None)],
)
def test_control_task_is_the_primary_dits_task(partition, task):
    assert control_task(partition) == task


@pytest.mark.parametrize(
    ("checkpoint", "fused", "expected"),
    [
        (None, None, BASE_H3),
        (FastH3CheckpointSpec(), None, FASTH3_V2),
        (FastH3OmniCheckpointSpec(), None, FASTH3_OMNI),
        (None, SimpleNamespace(), FASTH3_PREVIEW),
    ],
)
def test_pipeline_reports_the_variant_it_loaded(checkpoint, fused, expected):
    assert _pipeline(checkpoint, fused)._variant is expected


def test_request_variant_takes_a_few_step_adapter_on_base_weights_only():
    assert _pipeline()._request_variant() is BASE_H3
    assert _pipeline()._request_variant(turbo_spec=REF2V_TURBO).name == REF2V_TURBO.filename
    assert _pipeline()._request_variant(native_spec=object()).supports(H3Feature.TIMELINE_GUIDES)
    with pytest.raises(OmniClientError, match="FastH3 Omni supports ordinary LoRA adapters"):
        _pipeline(FastH3OmniCheckpointSpec())._request_variant(turbo_spec=REF2V_TURBO)
