# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.models.lance.pipeline_lance import _edit_replay_condition

pytestmark = [pytest.mark.core_model]


@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cpu", marks=pytest.mark.cpu),
        pytest.param("cuda", marks=hardware_marks(res={"cuda": "L4"})),
    ],
)
@pytest.mark.parametrize("ref_tokens", [2, 8], ids=["image", "video"])
def test_edit_replay_exports_generation_mask(device, ref_tokens):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for the replay mask device regression")
    segments = [
        {"role": role, "ids": torch.tensor([10]), "positions": torch.tensor([0])}
        for role in ("prefix", "gen_instruction", "gen_separator")
    ]
    vit_rows = torch.ones(3, 4, device=device)
    vae_rows = torch.full((ref_tokens + 2, 4), 2.0, device=device)
    segments.extend(
        [
            {"role": "ref_vit", "rows": vit_rows, "positions": torch.zeros(3, device=device)},
            {
                "role": "ref_vae",
                "rows": vae_rows,
                "positions": torch.zeros(3, ref_tokens + 2, device=device),
                "gen_indexes": torch.arange(1, ref_tokens + 1, device=device),
            },
        ]
    )
    latent_input = {
        "packed_position_ids": torch.arange(6, device=device).reshape(3, 2),
        "packed_vae_position_ids": torch.arange(2, device=device),
    }

    result = _edit_replay_condition(segments, latent_input, anchor=3.0, text_cap=4)

    expected_mask = torch.tensor([[False] * 4 + [True] * ref_tokens + [False]])
    assert result["condition_ref_is_gen"].device == vae_rows.device
    torch.testing.assert_close(result["condition_ref_is_gen"].cpu(), expected_mask)
    torch.testing.assert_close(result["condition_ref_rows"][0], torch.cat([vit_rows, vae_rows]))
    torch.testing.assert_close(result["condition_latent_positions"][0], latent_input["packed_position_ids"].cpu())
    assert result["condition_ref_positions"].shape == (1, 3, ref_tokens + 5)
    assert result["condition_gen_tail_mask"].tolist() == [[True, True, False, False]]
