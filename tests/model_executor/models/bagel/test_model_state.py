# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The MRv2 entry point must use BAGEL's prompt positions and preserve IDs."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.bagel.bagel import OmniBagelForConditionalGeneration
from vllm_omni.model_executor.models.bagel.model_state import BagelModelState
from vllm_omni.worker_v2.model_states import init_omni_model_state
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_runner_selects_bagel_state_with_one_position_channel(mocker):
    model = object.__new__(OmniBagelForConditionalGeneration)
    torch.nn.Module.__init__(model)
    cfg = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=3),
        model_config=SimpleNamespace(max_model_len=8192),
    )

    def initialize(self, config, model, encoder_cache, device):
        self.scheduler_config = config.scheduler_config
        self.model_config = config.model_config
        self.max_num_tokens = 4096

    mocker.patch.object(OmniModelState, "__init__", initialize)
    rope = mocker.patch("vllm_omni.model_executor.models.bagel.model_state.RopeState")
    state = init_omni_model_state(cfg, model, None, torch.device("cpu"))

    assert isinstance(state, BagelModelState)
    assert state.rope_state is rope.return_value
    rope.assert_called_once_with(
        num_dims=1, max_num_reqs=3, max_num_tokens=4096, max_model_len=8192, device=torch.device("cpu")
    )


def test_preparation_keeps_static_positions_and_restores_placeholder_ids(mocker):
    state = object.__new__(BagelModelState)
    positions = torch.tensor([[0, 0, 0, 1, 2]])
    embeds = torch.zeros(5, 2)
    base = mocker.patch.object(
        OmniModelState, "prepare_inputs", return_value={"positions": positions, "inputs_embeds": embeds}
    )
    batch = SimpleNamespace(input_ids=torch.tensor([9, 9, 9, 1, 2]))
    req_states = object()

    inputs = state.prepare_inputs(batch, req_states)

    base.assert_called_once_with(batch, req_states)
    assert inputs["positions"].tolist() == [0, 0, 0, 1, 2]
    assert inputs["positions"].data_ptr() == positions.data_ptr()
    assert inputs["inputs_embeds"] is embeds
    assert inputs["input_ids"] is batch.input_ids


def test_capture_uses_the_same_one_dimensional_position_view(mocker):
    state = object.__new__(BagelModelState)
    positions = torch.zeros(1, 8, dtype=torch.long)
    mocker.patch.object(OmniModelState, "prepare_dummy_inputs", return_value={"positions": positions})
    inputs = state.prepare_dummy_inputs(2, 8)
    assert inputs["positions"].shape == (8,)
    assert inputs["positions"].data_ptr() == positions.data_ptr()
