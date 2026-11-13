# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
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


def test_explicit_generation_prompt_keeps_rvq_payload_out_of_scheduler(monkeypatch) -> None:
    codes = torch.tensor([[17, -1, -1], [19, 2, 3]])
    payload = {"codes": {"audio": codes}, "meta": {"next_stage_prompt_len": 1, "next_stage_prompt_ids": [0]}}
    # The declared prompt must avoid even inspecting/flattening raw codec values.
    monkeypatch.setattr(DefaultSchedulingMetadataAdapter, "_flatten", lambda _: pytest.fail("flattened RVQ payload"))
    update = DefaultSchedulingMetadataAdapter().extract(payload, model_mode="generation")
    assert update == SchedulingMetadataUpdate(prompt_token_ids=(0,), resize_prompt_to=1)
    assert torch.equal(codes, torch.tensor([[17, -1, -1], [19, 2, 3]]))


@pytest.mark.parametrize("ids", [[], [-1], [True], [0.0], "0"])
def test_explicit_generation_prompt_rejects_invalid_ids(ids) -> None:
    with pytest.raises(ValueError, match="non-empty list"):
        DefaultSchedulingMetadataAdapter().extract({"meta": {"next_stage_prompt_ids": ids}}, model_mode="generation")


def test_explicit_generation_prompt_validates_declared_length() -> None:
    with pytest.raises(ValueError, match="must match"):
        DefaultSchedulingMetadataAdapter().extract(
            {"meta": {"next_stage_prompt_ids": [0], "next_stage_prompt_len": 2}}, model_mode="generation"
        )


def test_explicit_generation_prompt_does_not_change_ar_prompt() -> None:
    update = DefaultSchedulingMetadataAdapter().extract(
        {"meta": {"next_stage_prompt_ids": [0], "next_stage_prompt_len": 1}}, model_mode="ar"
    )
    assert update == SchedulingMetadataUpdate(resize_prompt_to=1)


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


def test_resolve_rejects_an_incompatible_extract_signature() -> None:
    class WrongSignatureAdapter:
        def extract(self):
            return None

    with pytest.raises(TypeError, match=r"extract\(payload, \*, model_mode=\.\.\.\)"):
        resolve_scheduling_metadata_adapter(WrongSignatureAdapter())


def test_resolve_rejects_a_non_callable_extract() -> None:
    class NonCallableExtractAdapter:
        extract = None

    with pytest.raises(TypeError, match="scheduling_metadata_adapter.extract must be callable"):
        resolve_scheduling_metadata_adapter(NonCallableExtractAdapter())
