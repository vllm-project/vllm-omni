# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for runner-side payload-to-scheduler metadata translation."""

from __future__ import annotations

import pytest
import torch

from vllm_omni.outputs import SchedulingMetadataUpdate
from vllm_omni.worker.scheduling_metadata_adapter import (
    DefaultSchedulingMetadataAdapter,
    resolve_scheduling_metadata_adapter,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_default_adapter_extracts_generic_generation_effects() -> None:
    update = DefaultSchedulingMetadataAdapter().extract(
        {"codes": {"audio": [[10, 20], [30]]}, "meta": {"next_stage_prompt_len": 3}},
        model_mode="generation",
    )

    assert update == SchedulingMetadataUpdate(
        prompt_token_ids=(10, 20, 30),
        resize_prompt_to=3,
    )


def test_default_adapter_flattens_tensor_codes() -> None:
    update = DefaultSchedulingMetadataAdapter().extract(
        {"codes": {"audio": torch.tensor([[1, 2, 3]], dtype=torch.long)}},
        model_mode="generation",
    )

    assert update == SchedulingMetadataUpdate(prompt_token_ids=(1, 2, 3))


def test_default_adapter_does_not_leak_codec_metadata_to_scheduler() -> None:
    update = DefaultSchedulingMetadataAdapter().extract(
        {"meta": {"left_context_size": 7}},
        model_mode="generation",
    )

    assert update is None


def test_default_adapter_keeps_ar_prompt_tokens_unchanged() -> None:
    update = DefaultSchedulingMetadataAdapter().extract(
        {"codes": {"audio": [1, 2]}, "meta": {"next_stage_prompt_len": 2}},
        model_mode="ar",
    )

    assert update == SchedulingMetadataUpdate(resize_prompt_to=2)


def test_resolve_accepts_an_adapter_instance() -> None:
    class CustomAdapter:
        def extract(self, payload, *, model_mode):
            del payload, model_mode
            return SchedulingMetadataUpdate(resize_prompt_to=7)

    adapter = resolve_scheduling_metadata_adapter(CustomAdapter())

    assert adapter.extract({}, model_mode="generation") == SchedulingMetadataUpdate(resize_prompt_to=7)
