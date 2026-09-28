# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The MiniCPM-o 4.5 Stage-0 duplex runtime is built with the model, not in the first session."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5 import minicpmo_4_5_omni
from vllm_omni.model_executor.models.minicpmo_4_5.duplex import compat
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
    MiniCPMO45OmniForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _StageModel(torch.nn.Module):
    def make_empty_intermediate_tensors(self):
        return None


@pytest.mark.parametrize(
    ("model_stage", "session_mode", "expected_builds"),
    [
        ("llm", "duplex", 1),
        ("llm", "turn", 0),
        ("tts", "duplex", 0),
    ],
)
def test_init_builds_duplex_runtime_only_for_duplex_thinker(
    monkeypatch: pytest.MonkeyPatch,
    model_stage: str,
    session_mode: str,
    expected_builds: int,
) -> None:
    monkeypatch.setattr(minicpmo_4_5_omni, "init_vllm_registered_model", lambda **kwargs: _StageModel())
    monkeypatch.setattr(compat, "patch_minicpmo_remote_config", lambda config: None)
    build_devices: list[torch.device] = []
    monkeypatch.setattr(
        MiniCPMO45OmniForConditionalGeneration,
        "_duplex_data_plane_helper",
        lambda self: build_devices.append(torch.get_default_device()),
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(),
            multimodal_config=None,
            model_stage=model_stage,
            session_mode=session_mode,
        )
    )

    # The loader constructs the model under the target-device context.
    with torch.device("meta"):
        MiniCPMO45OmniForConditionalGeneration(vllm_config=vllm_config)

    assert build_devices == [torch.device("cpu")] * expected_builds
