# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Startup capture of the packed streaming CUDA graphs: which layouts it covers
and which stage records them."""

from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.packed_dit import flow_graph_total, flow_graph_totals
from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 import CosyVoice3Model

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("limit", [1, 128, 129, 1024, 1025, 4096, 4097, 16384])
def test_flow_graph_totals_cover_every_length_up_to_the_limit(limit):
    totals = flow_graph_totals(limit)
    assert totals == sorted(set(totals))
    assert all(flow_graph_total(total) == total for total in totals)
    assert {flow_graph_total(length) for length in range(1, limit + 1)} == set(totals)


@pytest.mark.parametrize("stage", ["cosyvoice3_talker", "cosyvoice3_code2wav"])
def test_only_the_codec_stage_captures_graphs(stage):
    captured = []
    model = SimpleNamespace(model_stage=stage, code2wav=SimpleNamespace(capture_graphs=lambda: captured.append(1)))
    CosyVoice3Model.capture_auxiliary_graphs(model)
    assert captured == ([1] if stage == "cosyvoice3_code2wav" else [])
