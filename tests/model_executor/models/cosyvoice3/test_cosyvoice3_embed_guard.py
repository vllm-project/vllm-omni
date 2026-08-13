# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the CosyVoice3 talker speech-embedding OOB guard.

See https://github.com/vllm-project/vllm-omni/issues/4721 — a non-multimodal request
(e.g. text-only ``/v1/completions`` without an audio prompt) reaches the talker's ``else``
branch in ``embed_input_ids`` and indexes ``speech_embedding`` with out-of-range text
token ids, triggering a CUDA device-side assert that kills the EngineCore. The serving
layer rejects unsupported generate routes before dispatch; this guard remains
defense-in-depth for any path that reaches the talker prefill, turning OOB ids into a
clear ``ValueError`` before the gather.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
import torch
from pytest_mock import MockerFixture

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _guard():
    # Defer the heavy cosyvoice3 import until a test actually runs.
    from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 import _validate_speech_token_ids

    return _validate_speech_token_ids


def _embed_input_ids(
    input_ids: torch.Tensor,
    prefill_token_mask: bool | torch.Tensor | None,
) -> torch.Tensor:
    from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 import CosyVoice3Model

    model = SimpleNamespace(
        model_stage="cosyvoice3_talker",
        model=SimpleNamespace(speech_embedding=torch.nn.Embedding(10, 4)),
    )
    return CosyVoice3Model.embed_input_ids(
        model,
        input_ids,
        is_multimodal=None,
        prefill_token_mask=prefill_token_mask,
    )


def test_single_token_prefill_rejects_out_of_range_id() -> None:
    with pytest.raises(ValueError, match="out of range"):
        _embed_input_ids(torch.tensor([6562]), prefill_token_mask=True)


def test_mixed_batch_validates_only_prefill_tokens() -> None:
    with pytest.raises(ValueError, match="6562"):
        _embed_input_ids(
            torch.tensor([3, 6562]),
            prefill_token_mask=torch.tensor([False, True]),
        )


def test_decode_skips_prefill_validation(mocker: MockerFixture) -> None:
    import vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 as cosyvoice3

    guard = mocker.patch.object(cosyvoice3, "_validate_speech_token_ids")

    output = _embed_input_ids(torch.tensor([3]), prefill_token_mask=False)

    assert output.shape == (1, 4)
    guard.assert_not_called()


def test_multimodal_prefill_skips_non_multimodal_validation(mocker: MockerFixture) -> None:
    import vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 as cosyvoice3

    guard = mocker.patch.object(cosyvoice3, "_validate_speech_token_ids")
    hidden_size = 4
    model = SimpleNamespace(
        model_stage="cosyvoice3_talker",
        model=SimpleNamespace(
            llm=SimpleNamespace(model=SimpleNamespace(embed_tokens=torch.nn.Embedding(16, hidden_size))),
            speech_embedding=torch.nn.Embedding(10, hidden_size),
            sos=8,
            task_id=9,
        ),
    )

    output = cosyvoice3.CosyVoice3Model.embed_input_ids(
        model,
        torch.tensor([0, 1, 2, 3]),
        multimodal_embeddings=[torch.zeros(1, hidden_size)],
        is_multimodal=torch.tensor([True, True, True, False]),
        prefill_token_mask=True,
    )

    assert output.shape == (4, hidden_size)
    guard.assert_not_called()


def test_missing_prefill_metadata_fails_safe() -> None:
    with pytest.raises(ValueError, match="out of range"):
        _embed_input_ids(torch.tensor([6562]), prefill_token_mask=None)


def test_inclusive_boundaries_pass() -> None:
    # 0 and num_speech_tokens - 1 are valid and must not raise.
    _guard()(torch.tensor([0, 1, 5, 9], dtype=torch.long), num_speech_tokens=10)


def test_empty_ids_pass() -> None:
    _guard()(torch.empty(0, dtype=torch.long), num_speech_tokens=10)


@pytest.mark.parametrize(
    "ids",
    [
        pytest.param([10], id="equal_to_size"),
        pytest.param([0, 1, 6562], id="text_token_id_above_codec_range"),
        pytest.param([-1], id="negative"),
    ],
)
def test_out_of_range_ids_raise(ids: list[int]) -> None:
    with pytest.raises(ValueError, match="out of range"):
        _guard()(torch.tensor(ids, dtype=torch.long), num_speech_tokens=10)


def test_out_of_range_error_message_guides_user() -> None:
    with pytest.raises(ValueError, match="provide the required audio prompt"):
        _guard()(torch.tensor([6562], dtype=torch.long), num_speech_tokens=10)


def test_error_message_caps_listed_ids() -> None:
    # Many bad ids must not make the error message blow up; at most 8 are listed.
    ids = torch.arange(100, 150, dtype=torch.long)  # 50 out-of-range ids
    with pytest.raises(ValueError) as exc:
        _guard()(ids, num_speech_tokens=10)
    match = re.search(r"ids \[(.*?)\]", str(exc.value))
    assert match is not None, f"Expected 'ids [...]' in error message, got: {exc.value}"
    listed = match.group(1)
    assert 0 < len([x for x in listed.split(",") if x.strip()]) <= 8
