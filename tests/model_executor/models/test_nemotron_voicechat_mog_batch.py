# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Pin ready-row sampling, CFG order and independently owned feedback."""

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.nemotron_voicechat.nemotron_voicechat_talker import (
    NemotronVoiceChatTalkerForConditionalGeneration,
)
from vllm_omni.model_executor.models.nemotron_voicechat.talker_native import MoGStepGraph

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Sampler(MoGStepGraph):
    def __init__(self, rows: int, guidance: bool) -> None:
        self.rows = rows
        self.guidance = guidance
        self.inputs: list[torch.Tensor] = []

    def run(self, hidden: torch.Tensor) -> torch.Tensor:
        self.inputs.append(hidden.clone())
        result = hidden[: self.rows, :, :2].to(torch.long)
        if self.guidance:
            result = result + 100 * hidden[self.rows :, :, :2].to(torch.long)
        return result


def _talker(guidance: bool = False) -> NemotronVoiceChatTalkerForConditionalGeneration:
    model = NemotronVoiceChatTalkerForConditionalGeneration.__new__(NemotronVoiceChatTalkerForConditionalGeneration)
    nn.Module.__init__(model)
    model._dtype = torch.float32
    model._native_guidance = guidance
    model._mog_graphs = {size: _Sampler(size, guidance) for size in (1, 2, 4)}
    model._sessions = {}
    return model


def _session(pending: int | None = 1, uncond: float = 0.0) -> dict:
    return {
        "use_native": True,
        "pending_step": pending,
        "step": 1,
        "uncond_hidden": torch.full((1, 1, 3), uncond),
        "codes_rows": [],
        "prompt_len": 1,
        "codec_streaming": True,
        "codec_request_id": "codec",
        "timeline": torch.zeros(8, dtype=torch.long),
        "upstream_finished": False,
    }


@pytest.mark.parametrize("guidance", [False, True])
def test_gathers_last_hidden_rows_and_keeps_feedback_private(guidance) -> None:
    model = _talker(guidance)
    model._sessions = {"first": _session(uncond=3), "second": _session(uncond=7)}
    hidden = torch.tensor([[90.0] * 3, [1.0] * 3, [2.0] * 3])
    infos = [{"request_id": "first"}, {"request_id": "second"}]
    output = model._native_make_omni_output(
        hidden, runtime_additional_information=infos, request_token_spans=[(0, 2), (2, 3)]
    )
    sampler = model._mog_graphs[2]
    expected_hidden = [1.0, 2.0, 3.0, 7.0] if guidance else [1.0, 2.0]
    assert sampler.inputs[0][:, 0, 0].tolist() == expected_hidden
    assert len(sampler.inputs) == 1
    audio = output.multimodal_outputs["codes"]["audio"]
    assert audio[0].tolist() == [[301, 301] if guidance else [1, 1]]
    assert audio[1].tolist() == [[702, 702] if guidance else [2, 2]]
    first, second = model._sessions.values()
    assert first["step"] == second["step"] == 2
    assert first["pending_step"] is second["pending_step"] is None
    assert first["code"].untyped_storage().nbytes() == first["code"].numel() * first["code"].element_size()
    first["code"].fill_(9)
    assert second["code"].flatten()[0].item() == (702 if guidance else 2)


@pytest.mark.parametrize("guidance", [False, True])
def test_padding_keeps_cfg_halves_aligned_and_returns_only_active_rows(guidance) -> None:
    model = _talker(guidance)
    cond = torch.tensor([[[1.0] * 3], [[2.0] * 3], [[3.0] * 3]])
    hidden = torch.cat([cond, cond + 10]) if guidance else cond
    codes = model._sample_native_mog(hidden)
    assert codes.shape == (3, 1, 2)
    assert codes[:, 0, 0].tolist() == ([1101, 1202, 1303] if guidance else [1, 2, 3])
    packed = model._mog_graphs[4].inputs[0]
    expected = [1, 2, 3, 0, 11, 12, 13, 0] if guidance else [1, 2, 3, 0]
    assert packed[:, 0, 0].tolist() == expected


def test_prefill_history_does_not_join_live_sampling_or_advance() -> None:
    model = _talker()
    prefill = _session(pending=None)
    prefill["codes_rows"] = [torch.tensor([[8, 9]])]
    live = _session()
    live["timeline"] = torch.zeros(2, dtype=torch.long)
    model._sessions = {"prefill": prefill, "live": live}
    infos = [{"request_id": "prefill"}, {"request_id": "live"}, {"request_id": "absent"}]
    result = model._native_make_omni_output(
        torch.tensor([[10.0] * 3, [2.0] * 3, [30.0] * 3]),
        runtime_additional_information=infos,
        request_token_spans=[(0, 1), (1, 2), (2, 3)],
    )
    assert model._mog_graphs[1].inputs[0][:, 0, 0].tolist() == [2]
    assert prefill["step"] == 1
    audio = result.multimodal_outputs["codes"]["audio"]
    assert audio[0].tolist() == [[8, 9]]
    assert audio[1].tolist() == [[2, 2]]
    assert audio[2].numel() == 0
    assert result.text_hidden_states[:, 0].tolist() == [0, 1, 0]


def test_empty_dispatch_does_not_sample() -> None:
    model = _talker()
    result = model._native_make_omni_output(torch.zeros((1, 3)), runtime_additional_information=[{}])
    assert result.multimodal_outputs == {}
    assert all(not sampler.inputs for sampler in model._mog_graphs.values())


def test_cfg_requires_unconditional_hidden_for_each_live_request() -> None:
    model = _talker(True)
    session = _session()
    session["uncond_hidden"] = None
    model._sessions = {"request": session}
    with pytest.raises(RuntimeError, match="CFG desynchronized"):
        model._native_make_omni_output(
            torch.zeros((1, 3)),
            runtime_additional_information=[{"request_id": "request"}],
            request_token_spans=[(0, 1)],
        )
    assert session["pending_step"] == 1


@pytest.mark.parametrize("guidance", [False, True])
def test_failed_sampling_preserves_pending_frames_for_retry(monkeypatch, guidance) -> None:
    model = _talker(guidance)
    model._sessions = {"first": _session(uncond=3), "second": _session(uncond=7)}
    hidden = torch.tensor([[1.0] * 3, [2.0] * 3])
    infos = [{"request_id": "first"}, {"request_id": "second"}]
    sampler = model._mog_graphs[2]
    run = sampler.run

    def fail(hidden):
        raise RuntimeError("injected sampling failure")

    monkeypatch.setattr(sampler, "run", fail)
    with pytest.raises(RuntimeError, match="injected sampling failure"):
        model._native_make_omni_output(
            hidden, runtime_additional_information=infos, request_token_spans=[(0, 1), (1, 2)]
        )
    for session in model._sessions.values():
        assert session["pending_step"] == session["step"] == 1
        assert not session["codes_rows"] and "code" not in session
        assert session["uncond_hidden"] is not None

    monkeypatch.setattr(sampler, "run", run)
    model._native_make_omni_output(hidden, runtime_additional_information=infos, request_token_spans=[(0, 1), (1, 2)])
    for session in model._sessions.values():
        assert session["step"] == 2 and session["pending_step"] is None
        assert len(session["codes_rows"]) == 1
