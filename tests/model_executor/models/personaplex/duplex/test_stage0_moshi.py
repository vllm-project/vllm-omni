# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Kyutai Moshi checkpoints on the PersonaPlex Stage 0 duplex runtime (moshi.server framing)."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from huggingface_hub import constants as hf_constants
from safetensors.torch import save_file

from tests.model_executor.models.personaplex.duplex.test_stage0_runtime import (
    _duplex_info,
    _FakeCodec,
    _FakeTalker,
)
from vllm_omni.model_executor.models.personaplex.duplex import stage0
from vllm_omni.model_executor.models.personaplex.duplex.stage0 import (
    PersonaPlexStage0DuplexRuntime,
    checkpoint_depformer_steps,
    depformer_steps,
    is_moshi_checkpoint,
)
from vllm_omni.model_executor.models.personaplex.personaplex_mimi import MIMI_CHECKPOINT, model_mimi_checkpoint
from vllm_omni.model_executor.models.personaplex.personaplex_talker import PersonaPlexTalkerForConditionalGeneration
from vllm_omni.transformers_utils import repo_utils

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Moshi's initial audio token is the codec card.
INITIAL = [2048] * 8


class _FakeMoshiTalker(_FakeTalker):
    def __init__(self, card: int = 2048) -> None:
        super().__init__()
        self.config = SimpleNamespace(audio_vocab_size=card)
        self.initial_calls = 0

    def _initial_frame_embed(self, device):
        self.initial_calls += 1
        return torch.full((1, 4), -1.0, device=device)


def _moshi_runtime(codec: _FakeCodec, max_sessions: int = 1, card: int = 2048) -> PersonaPlexStage0DuplexRuntime:
    def no_prompt(*_args):
        raise AssertionError("a Moshi session has no voice or persona prompt")

    return PersonaPlexStage0DuplexRuntime(
        _FakeMoshiTalker(card),
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


def test_the_initial_token_is_the_talker_config_card() -> None:
    runtime = _moshi_runtime(_FakeCodec(), card=1024)

    runtime.prepare_append(_duplex_info(seq=1), prompt_len=1, request_id="req")
    assert runtime.depformer_teacher_forcing(["req"])[0].tolist() == [[1024] * 16]

    runtime.record_samples(request_ids=["req"], text_tokens=torch.tensor(101), agent_codes=torch.arange(8))
    runtime.prepare_append(_duplex_info(seq=2), prompt_len=2, request_id="req")
    (call,) = runtime.stage_model.frame_calls
    assert call["last_agent"].tolist() == [[0] + [1024] * 7]
    assert call["user_d1"].tolist() == [[1024] * 8]


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
    path.mkdir(parents=True)
    save_file({f"linears.{i}.weight": torch.zeros(1) for i in range(steps)}, str(path / "model.safetensors"))
    return path


def _cache_snapshot(cache: Path, repo_id: str, steps: int) -> None:
    """``repo_id`` at ``main`` in a Hub cache laid out like ``snapshot_download``'s."""
    repo = cache / f"models--{repo_id.replace('/', '--')}"
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text("0" * 40)
    _checkpoint(repo / "snapshots" / ("0" * 40), steps)


class _FakeHub:
    def __init__(self, weight_map: dict[str, str] | None) -> None:
        self.weight_map = weight_map
        self.calls: list[tuple[str, str | None]] = []

    def get_safetensors_metadata(self, repo_id: str, *, revision: str | None = None):
        self.calls.append((repo_id, revision))
        if self.weight_map is None:
            raise OSError("offline")
        return SimpleNamespace(weight_map=self.weight_map)


def test_moshi_detection_counts_the_checkpoint_depformer_steps(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(repo_utils, "hf_api", lambda: _FakeHub(None))
    moshi = _checkpoint(tmp_path / "moshiko", 8)
    personaplex = _checkpoint(tmp_path / "personaplex", 16)

    assert checkpoint_depformer_steps(str(moshi)) == 8
    assert is_moshi_checkpoint(str(moshi)) is True
    assert is_moshi_checkpoint(str(personaplex)) is False
    assert checkpoint_depformer_steps(str(tmp_path / "missing")) is None
    assert is_moshi_checkpoint(str(tmp_path / "missing")) is False
    assert PersonaPlexStage0DuplexRuntime(_FakeMoshiTalker(), model_path=str(moshi), device="cpu").moshi is True


def test_moshi_detection_reads_a_repo_id_from_the_hub_cache(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path))
    hub = _FakeHub(None)
    monkeypatch.setattr(repo_utils, "hf_api", lambda: hub)
    _cache_snapshot(tmp_path, "kyutai/moshiko-pytorch-bf16", 8)
    _cache_snapshot(tmp_path, "nvidia/personaplex-7b-v1", 16)

    assert checkpoint_depformer_steps("kyutai/moshiko-pytorch-bf16") == 8
    assert is_moshi_checkpoint("kyutai/moshiko-pytorch-bf16") is True
    assert is_moshi_checkpoint("nvidia/personaplex-7b-v1") is False
    assert hub.calls == []


def test_moshi_detection_reads_the_header_on_the_hub_when_the_repo_is_not_cached(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path))
    hub = _FakeHub(
        {f"linears.{i}.weight": "model.safetensors" for i in range(8)} | {"emb.0.weight": "model.safetensors"}
    )
    monkeypatch.setattr(repo_utils, "hf_api", lambda: hub)

    assert checkpoint_depformer_steps("kyutai/moshiko-pytorch-bf16", "abc123") == 8
    assert hub.calls == [("kyutai/moshiko-pytorch-bf16", "abc123")]


def test_an_unreadable_checkpoint_is_looked_up_again(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(repo_utils, "hf_api", lambda: _FakeHub(None))

    assert checkpoint_depformer_steps("kyutai/moshiko-pytorch-bf16") is None
    _cache_snapshot(tmp_path, "kyutai/moshiko-pytorch-bf16", 8)
    assert checkpoint_depformer_steps("kyutai/moshiko-pytorch-bf16") == 8


def test_the_mimi_checkpoint_of_a_repo_id_comes_from_the_hub_cache(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    _cache_snapshot(tmp_path / "hub", "kyutai/moshiko-pytorch-bf16", 8)
    snapshot = tmp_path / "hub" / "models--kyutai--moshiko-pytorch-bf16" / "snapshots" / ("0" * 40)
    (snapshot / MIMI_CHECKPOINT).write_bytes(b"")
    local = _checkpoint(tmp_path / "local", 8)
    (local / MIMI_CHECKPOINT).write_bytes(b"")

    assert model_mimi_checkpoint("kyutai/moshiko-pytorch-bf16") == str(snapshot / MIMI_CHECKPOINT)
    assert model_mimi_checkpoint(str(local)) == str(local / MIMI_CHECKPOINT)
    assert model_mimi_checkpoint("kyutai/not-cached") is None
    assert model_mimi_checkpoint(str(tmp_path / "missing")) is None


def test_a_download_dir_is_searched_as_the_hub_cache(tmp_path: Path, monkeypatch) -> None:
    # vllm serve <repo id> --download-dir <dir>: the default cache stays empty.
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path / "default"))
    monkeypatch.setattr(repo_utils, "hf_api", lambda: _FakeHub(None))
    download_dir = tmp_path / "download"
    _cache_snapshot(download_dir, "kyutai/moshiko-pytorch-bf16", 8)
    snapshot = download_dir / "models--kyutai--moshiko-pytorch-bf16" / "snapshots" / ("0" * 40)
    (snapshot / MIMI_CHECKPOINT).write_bytes(b"")

    assert checkpoint_depformer_steps("kyutai/moshiko-pytorch-bf16") is None
    assert checkpoint_depformer_steps("kyutai/moshiko-pytorch-bf16", cache_dir=str(download_dir)) == 8
    assert model_mimi_checkpoint("kyutai/moshiko-pytorch-bf16") is None
    assert model_mimi_checkpoint("kyutai/moshiko-pytorch-bf16", cache_dir=str(download_dir)) == str(
        snapshot / MIMI_CHECKPOINT
    )
    runtime = PersonaPlexStage0DuplexRuntime(
        _FakeMoshiTalker(), model_path="kyutai/moshiko-pytorch-bf16", device="cpu", download_dir=str(download_dir)
    )
    assert runtime.moshi is True


def _loaded_talker(steps: int | None) -> SimpleNamespace:
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(model="kyutai/moshiko-pytorch-bf16", revision=None),
            load_config=SimpleNamespace(download_dir=None),
        ),
        config=SimpleNamespace(audio_vocab_size=2048),
        parameters=lambda: iter([torch.zeros(1)]),
        _checkpoint_depformer_steps=steps,
    )


def test_stage0_takes_moshi_from_the_weights_the_talker_loaded(monkeypatch) -> None:
    # The checkpoint lookup misses (a --download-dir copy with no Hub, say); the
    # weights the talker just loaded still decide.
    monkeypatch.setattr(stage0, "checkpoint_depformer_steps", lambda *_args, **_kwargs: None)
    runtime_for = PersonaPlexTalkerForConditionalGeneration._duplex_stage0_runtime

    assert runtime_for(_loaded_talker(depformer_steps(f"linears.{i}.weight" for i in range(8)))).moshi is True
    assert runtime_for(_loaded_talker(depformer_steps(f"linears.{i}.weight" for i in range(16)))).moshi is False
    # No weights seen (e.g. a dummy load): fall back to the checkpoint lookup.
    assert runtime_for(_loaded_talker(None)).moshi is False
