# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""UTs for VoxCPM async-chunk stage input processing."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.voxcpm import (
    _VOXCPM_LATENT_MAGIC,
    _coerce_finished_flag,
    latent2vae_async_chunk,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _request(*, finished):
    return SimpleNamespace(is_finished=lambda: finished)


def _decode_serialized_latent(codes: list[int]) -> torch.Tensor:
    assert codes[0] == _VOXCPM_LATENT_MAGIC
    latent_dim = codes[1]
    time_dim = codes[2]
    payload = torch.tensor(codes[3:], dtype=torch.int32).to(torch.uint16)
    return payload.view(torch.bfloat16).to(torch.float32).reshape(1, latent_dim, time_dim)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),
        (False, False),
        (True, True),
        (torch.tensor(False), False),
        (torch.tensor(True), True),
        ([torch.tensor(True)], True),
        (([True],), True),
        ([], False),
    ],
)
def test_coerce_finished_flag(value, expected):
    assert _coerce_finished_flag(value) is expected


def test_latent2vae_async_chunk_serializes_latent_payload():
    latent = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    payload = latent2vae_async_chunk(
        transfer_manager=None,
        pooling_output={"latent_audio_feat": latent},
        request=_request(finished=False),
        is_finished=torch.tensor(False),
    )

    assert payload is not None
    assert torch.equal(payload["finished"], torch.tensor(False, dtype=torch.bool))
    recovered = _decode_serialized_latent(payload["code_predictor_codes"])
    torch.testing.assert_close(recovered, latent.to(torch.bfloat16).to(torch.float32).unsqueeze(0))


def test_latent2vae_async_chunk_returns_terminal_marker_without_latent():
    payload = latent2vae_async_chunk(
        transfer_manager=None,
        pooling_output=None,
        request=_request(finished=[torch.tensor(True)]),
        is_finished=False,
    )

    assert payload == {
        "code_predictor_codes": [],
        "finished": torch.tensor(True, dtype=torch.bool),
    }


def test_latent2vae_async_chunk_returns_none_for_nonterminal_empty_chunk():
    payload = latent2vae_async_chunk(
        transfer_manager=None,
        pooling_output={"latent_audio_feat": torch.zeros((0,), dtype=torch.float32)},
        request=_request(finished=False),
        is_finished=False,
    )

    assert payload is None
