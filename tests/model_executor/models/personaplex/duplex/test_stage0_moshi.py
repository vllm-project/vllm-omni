# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Kyutai Moshi checkpoints on the PersonaPlex Stage 0 duplex runtime (moshi.server framing)."""

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tests.model_executor.models.personaplex.duplex.test_stage0_runtime import (
    _duplex_info,
    _FakeCodec,
    _FakeTalker,
)
from vllm_omni.model_executor.models.personaplex.duplex.policy import INITIAL_AUDIO_TOKEN
from vllm_omni.model_executor.models.personaplex.duplex.stage0 import (
    PersonaPlexStage0DuplexRuntime,
    checkpoint_depformer_steps,
    is_moshi_checkpoint,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

INITIAL = [INITIAL_AUDIO_TOKEN] * 8


class _FakeMoshiTalker(_FakeTalker):
    def __init__(self) -> None:
        super().__init__()
        self.initial_calls = 0

    def _initial_frame_embed(self, device):
        self.initial_calls += 1
        return torch.full((1, 4), -1.0, device=device)


def _moshi_runtime(codec: _FakeCodec, max_sessions: int = 1) -> PersonaPlexStage0DuplexRuntime:
    def no_prompt(*_args):
        raise AssertionError("a Moshi session has no voice or persona prompt")

    return PersonaPlexStage0DuplexRuntime(
        _FakeMoshiTalker(),
        model_path="/unused",
        device="cpu",
        codec=codec,
        max_sessions=max_sessions,
        tokenizer=no_prompt,
        voice_loader=no_prompt,
        moshi=True,
    )


def test_first_append_is_the_initial_frame_and_encodes_nothing() -> None:
    codec = _FakeCodec()
    runtime = _moshi_runtime(codec)

    runtime.encode_appends([_duplex_info(seq=1)])
    first = runtime.prepare_append(_duplex_info(seq=1), prompt_len=1, request_id="req")

    assert first.prefill_applied is True
    assert first.prompt_offset == 0
    assert first.user_frame is None
    assert first.inputs_embeds.tolist() == [[-1.0] * 4]
    assert first.info_update["meta"] == {"pplex_frame": 0, "pplex_prefill_len": 0}
    # moshi.server encodes the first frame and resets the encoder: the stream starts at frame 2.
    assert codec.encode_calls == 0
    assert runtime.stage_model.frame_calls == []


def test_initial_frame_forces_the_delayed_agent_codebooks_to_the_initial_token() -> None:
    runtime = _moshi_runtime(_FakeCodec())

    runtime.prepare_append(_duplex_info(seq=1), prompt_len=1, request_id="req")
    tokens, provided = runtime.depformer_teacher_forcing(["req"])
    assert tokens.tolist() == [INITIAL * 2]
    assert provided.tolist() == [[False] + [True] * 7 + [False] * 8]

    runtime.record_samples(request_ids=["req"], text_tokens=torch.tensor(101), agent_codes=torch.arange(8))
    runtime.prepare_append(_duplex_info(seq=2), prompt_len=2, request_id="req")

    (call,) = runtime.stage_model.frame_calls
    assert call["text_tokens"].tolist() == [101]
    # Agent cb0 comes from the initial step; cb1..7 have delay 1 and still read the initial token.
    assert call["last_agent"].tolist() == [[0] + INITIAL[1:]]


def test_live_frames_read_the_user_codes_of_the_frame_that_brings_them() -> None:
    runtime = _moshi_runtime(_FakeCodec())

    runtime.prepare_append(_duplex_info(seq=1), prompt_len=1, request_id="req")
    for seq in (2, 3):
        runtime.record_samples(request_ids=["req"], text_tokens=torch.tensor(100 + seq), agent_codes=torch.arange(8))
        runtime.prepare_append(_duplex_info(seq=seq), prompt_len=seq, request_id="req")

    second, third = runtime.stage_model.frame_calls
    # The fake codec numbers each row's encoded frames from 1.
    assert second["user_d0"].tolist() == [[1] * 8]
    assert second["user_d1"].tolist() == [INITIAL]
    assert third["user_d0"].tolist() == [[2] * 8]
    assert third["user_d1"].tolist() == [[1] * 8]


def test_a_new_epoch_restarts_from_the_initial_frame() -> None:
    codec = _FakeCodec()
    runtime = _moshi_runtime(codec)
    runtime.prepare_append(_duplex_info(seq=1), prompt_len=1, request_id="old")
    runtime.record_samples(request_ids=["old"], text_tokens=torch.tensor(7), agent_codes=torch.arange(8))
    runtime.prepare_append(_duplex_info(seq=2), prompt_len=2, request_id="old")

    restarted = runtime.prepare_append(_duplex_info(seq=1, epoch=1), prompt_len=1, request_id="new")

    assert restarted.inputs_embeds.tolist() == [[-1.0] * 4]
    assert codec.reset_slots == [0]
    assert runtime.depformer_teacher_forcing(["new"])[0].tolist() == [INITIAL * 2]
    runtime.record_samples(request_ids=["new"], text_tokens=torch.tensor(9), agent_codes=torch.arange(8))
    runtime.prepare_append(_duplex_info(seq=2, epoch=1), prompt_len=2, request_id="new")
    assert runtime.stage_model.frame_calls[-1]["user_d1"].tolist() == [INITIAL]


def test_warm_prefill_builds_the_initial_frame_once() -> None:
    runtime = _moshi_runtime(_FakeCodec())

    runtime.warm_prefill()
    runtime.prepare_append(_duplex_info(seq=1), prompt_len=1, request_id="req")

    assert runtime.stage_model.initial_calls == 1


def _checkpoint(path: Path, steps: int) -> Path:
    path.mkdir()
    save_file({f"linears.{i}.weight": torch.zeros(1) for i in range(steps)}, str(path / "model.safetensors"))
    return path


def test_moshi_detection_counts_the_checkpoint_depformer_steps(tmp_path: Path) -> None:
    moshi = _checkpoint(tmp_path / "moshiko", 8)
    personaplex = _checkpoint(tmp_path / "personaplex", 16)

    assert checkpoint_depformer_steps(str(moshi)) == 8
    assert is_moshi_checkpoint(str(moshi)) is True
    assert is_moshi_checkpoint(str(personaplex)) is False
    assert checkpoint_depformer_steps(str(tmp_path / "missing")) is None
    assert is_moshi_checkpoint(str(tmp_path / "missing")) is False
    assert PersonaPlexStage0DuplexRuntime(_FakeMoshiTalker(), model_path=str(moshi), device="cpu").moshi is True
