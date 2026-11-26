# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import base64
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import DuplexFence
from vllm_omni.model_executor.models.minicpmo_4_5 import minicpmo_4_5_code2wav as code2wav
from vllm_omni.model_executor.models.minicpmo_4_5.duplex import plugin as module
from vllm_omni.transformers_utils import repo_utils

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _plugin():
    return module.MiniCPMO45DuplexPlugin(lambda *args: None)


def _stub_tokenizer(monkeypatch):
    tokenizer = SimpleNamespace(
        all_special_ids=[90],
        unk_token_id=-1,
        eos_token_id=99,
        encode=lambda text, add_special_tokens=False: [ord(char) for char in text],
        convert_tokens_to_ids=lambda token: {"<|listen|>": 91}.get(token, -1),
    )
    monkeypatch.setattr(module, "_load_tokenizer", lambda config: tokenizer)


@pytest.mark.asyncio
@pytest.mark.parametrize(("extra_body", "mode"), [({}, "basic"), ({"sliding_window_mode": "off"}, "off")])
async def test_window_defaults_to_basic_and_can_be_disabled(monkeypatch, extra_body, mode):
    _stub_tokenizer(monkeypatch)
    config = DuplexSessionConfig(modalities=("text",), extra_body=dict(extra_body))
    runtime = await _plugin().prepare_runtime_config(config, model_config=None)
    window = runtime["duplex_window_config"]
    assert window["sliding_window_mode"] == mode
    assert (window["basic_window_high_tokens"], window["basic_window_low_tokens"]) == (8000, 6000)


@pytest.mark.asyncio
async def test_window_configuration_is_prepared_by_plugin(monkeypatch):
    _stub_tokenizer(monkeypatch)
    config = DuplexSessionConfig(
        modalities=("text",),
        extra_body={
            "sliding_window_mode": "basic",
            "basic_window_high_tokens": 120,
            "basic_window_low_tokens": 80,
        },
    )
    runtime = await _plugin().prepare_runtime_config(config, model_config=None)
    assert runtime["duplex_window_config"]["sliding_window_mode"] == "basic"
    assert "sliding_window_mode" not in config.extra_body
    assert runtime["duplex_window_prefix_tokens"] > 0
    assert runtime["duplex_window_suffix_token_ids"]
    assert runtime["duplex_window_previous_marker_token_ids"]


@pytest.mark.asyncio
async def test_invalid_window_configuration_is_rejected():
    config = DuplexSessionConfig(
        modalities=("text",),
        extra_body={
            "basic_window_high_tokens": 80,
            "basic_window_low_tokens": 80,
        },
    )
    with pytest.raises(module.MiniCPMO45ClientRuntimeConfigError) as exc:
        await _plugin().prepare_runtime_config(config, model_config=None)
    assert exc.value.code == "invalid_sliding_window_config"


def test_window_update_cannot_change_created_configuration():
    config = DuplexSessionConfig(extra_body={"sliding_window_mode": "basic"})
    with pytest.raises(module.MiniCPMO45ClientRuntimeConfigError) as exc:
        _plugin().runtime_config_for_update(config, {"duplex_window_config": {"sliding_window_mode": "off"}})
    assert exc.value.code == "sliding_window_update_unsupported"


def test_window_internal_configuration_is_server_owned():
    with pytest.raises(module.MiniCPMO45ClientRuntimeConfigError):
        _plugin().validate_client_extra_body({"duplex_window_config": {}})


@pytest.mark.asyncio
async def test_audio_session_uses_the_model_bundled_reference_audio(monkeypatch, tmp_path: Path):
    model_dir = tmp_path / "model"
    ref_path = model_dir / "assets" / "HT_ref_audio.wav"
    ref_path.parent.mkdir(parents=True)
    ref_path.touch()

    calls: list[tuple[str, str | None]] = []
    read_calls: list[str] = []

    def snapshot_download(model_ref: str, *, revision: str | None = None, allow_patterns=None) -> str:
        calls.append((model_ref, revision))
        return str(model_dir)

    monkeypatch.setattr(repo_utils, "hf_api", lambda: SimpleNamespace(snapshot_download=snapshot_download))
    monkeypatch.setattr(module, "_load_tokenizer", lambda _model_config: None)

    def read_reference_wav(path: str):
        read_calls.append(path)
        return np.stack((np.ones(1600, dtype=np.float32), np.full(1600, 3.0, dtype=np.float32))), 16_000

    monkeypatch.setattr(code2wav, "_read_reference_wav", read_reference_wav)

    plugin = _plugin()
    model_config = SimpleNamespace(model="openbmb/MiniCPM-o-4_5", revision="main")
    resolver = code2wav._resolve_model_dir
    loader = module._load_default_ref_audio
    resolver.cache_clear()
    loader.cache_clear()
    try:
        runtime = await plugin.prepare_runtime_config(DuplexSessionConfig(), model_config=model_config)
        second_runtime = await plugin.prepare_runtime_config(DuplexSessionConfig(), model_config=model_config)
    finally:
        resolver.cache_clear()
        loader.cache_clear()

    assert calls == [("openbmb/MiniCPM-o-4_5", "main")]
    assert read_calls == [str(ref_path)]
    assert runtime["ref_audio_sample_rate_hz"] == 16_000
    audio = np.frombuffer(base64.b64decode(runtime["ref_audio_data"]), dtype=np.float32)
    assert audio.tolist() == [2.0] * 1600
    assert second_runtime["ref_audio_data"] == runtime["ref_audio_data"]


@pytest.mark.asyncio
async def test_missing_model_reference_audio_is_retried_when_asset_appears(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model_config = SimpleNamespace(model=str(model_dir))
    resolver = code2wav._resolve_model_dir
    loader = module._load_default_ref_audio
    resolver.cache_clear()
    loader.cache_clear()
    try:
        missing = await module.resolve_default_ref_audio(model_config)

        ref_path = Path(code2wav.default_prompt_wav(str(model_dir)))
        ref_path.parent.mkdir(parents=True)
        sf.write(ref_path, np.full(1600, 0.25, dtype=np.float32), 16_000, subtype="PCM_16")
        loaded = await module.resolve_default_ref_audio(model_config)
    finally:
        resolver.cache_clear()
        loader.cache_clear()

    assert missing is None
    assert loaded is not None
    assert loaded[1] == 16_000
    np.testing.assert_allclose(loaded[0], 0.25, atol=1e-4)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [FileNotFoundError("offline and not cached"), OSError("snapshot read failed")],
)
async def test_model_reference_audio_resolution_errors_return_none(monkeypatch, error: OSError):
    def load_default_ref_audio(_model_ref: str, _revision: str | None) -> tuple[np.ndarray, int]:
        raise error

    monkeypatch.setattr(module, "_load_default_ref_audio", load_default_ref_audio)

    result = await module.resolve_default_ref_audio(SimpleNamespace(model="openbmb/MiniCPM-o-4_5"))

    assert result is None


@pytest.mark.asyncio
@pytest.mark.parametrize("sample_count", [1599, 1600, 96256])
async def test_explicit_reference_audio_uses_media_connector_and_skips_model_default(
    monkeypatch, tmp_path: Path, sample_count: int
):
    ref_path = tmp_path / "reference.wav"
    sf.write(ref_path, np.full(sample_count, 0.25, dtype=np.float32), 16_000, subtype="PCM_16")

    async def fail_default(_model_config):
        raise AssertionError("the model default must not replace an explicit ref_audio")

    monkeypatch.setattr(module, "resolve_default_ref_audio", fail_default)
    monkeypatch.setattr(module, "_load_tokenizer", lambda _model_config: None)

    config = DuplexSessionConfig(ref_audio=ref_path.as_uri())
    model_config = SimpleNamespace(
        allowed_local_media_path=str(tmp_path),
        allowed_media_domains=None,
    )
    runtime = await _plugin().prepare_runtime_config(config, model_config=model_config)

    assert runtime["ref_audio_sample_rate_hz"] == 16_000
    audio = np.frombuffer(base64.b64decode(runtime["ref_audio_data"]), dtype=np.float32)
    assert audio.shape == (sample_count,)
    np.testing.assert_allclose(audio, 0.25, atol=1e-4)


@pytest.mark.asyncio
async def test_text_only_session_does_not_resolve_model_default(monkeypatch):
    async def fail_default(_model_config):
        raise AssertionError("text-only sessions must not load the model default reference audio")

    monkeypatch.setattr(module, "resolve_default_ref_audio", fail_default)
    monkeypatch.setattr(module, "_load_tokenizer", lambda _model_config: None)

    runtime = await _plugin().prepare_runtime_config(
        DuplexSessionConfig(modalities=("text",)),
        model_config=SimpleNamespace(model="openbmb/MiniCPM-o-4_5"),
    )

    assert "ref_audio_data" not in runtime


@pytest.mark.asyncio
async def test_resolve_ref_audio_reads_local_file_uri_with_media_connector(tmp_path: Path):
    ref_path = tmp_path / "reference.wav"
    sf.write(ref_path, np.full(1600, 0.25, dtype=np.float32), 16_000, subtype="PCM_16")

    waveform, sample_rate = await module.resolve_ref_audio(
        ref_path.as_uri(),
        model_config=SimpleNamespace(
            allowed_local_media_path=str(tmp_path),
            allowed_media_domains=None,
        ),
    )

    assert sample_rate == 16_000
    assert np.asarray(waveform).size == 1600
    np.testing.assert_allclose(np.asarray(waveform).reshape(-1), 0.25, atol=1e-4)


@pytest.mark.parametrize("final", [False, True])
@pytest.mark.parametrize("seq,samples,expected", [(1, 32000, 16), (1, 48000, 28), (2, 16000, 13)])
def test_first_and_final_append_reserve_exact_window_input(seq, samples, expected, final):
    prompt = module.build_duplex_data_plane_prompt(
        request_id="window-request",
        fence=DuplexFence("sid", turn_id=1),
        session_config={},
        runtime_config={"duplex_first_append_context_tokens": 5},
        seq=seq,
        turn_seq=seq,
        payload={
            "audio": base64.b64encode(bytes(samples * 4)).decode(),
            "format": "pcm_f32le",
            "sample_rate_hz": 16000,
        },
        final=final,
    )
    assert len(prompt["prompt_token_ids"]) == expected


@pytest.mark.parametrize("seq", [1, 2, 3])
def test_final_exact_chunk_append_reserves_one_unit_not_two(seq):
    """A final append whose audio is already an exact number of chunks must
    reserve exactly what Stage0 feeds: one unit per chunk (``<unit>`` plus its
    audio) and the closure pair for every unit after the first. Serving pads
    the final residual itself, so the old extra 12-slot silent-unit
    reservation only put 12 pad positions ahead of the audio in the KV."""

    def _budget(*, seq, final):
        prompt = module.build_duplex_data_plane_prompt(
            request_id="final-request",
            fence=DuplexFence("sid", turn_id=1),
            session_config={},
            runtime_config={"duplex_first_append_context_tokens": 0},
            seq=seq,
            turn_seq=seq,
            payload={
                "audio": base64.b64encode(bytes(16000 * 4)).decode(),
                "format": "pcm_f32le",
                "sample_rate_hz": 16000,
            },
            final=final,
        )
        return len(prompt["prompt_token_ids"])

    assert _budget(seq=seq, final=True) == _budget(seq=seq, final=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("late_commit", [False, True], ids=["commit-before-done", "commit-after-done"])
async def test_cancel_restore_keeps_committed_user_media_and_excludes_cancelled_answer(monkeypatch, late_commit):
    _stub_tokenizer(monkeypatch)
    plugin = _plugin()
    state = plugin.create_session_state()
    runtime = await plugin.prepare_runtime_config(DuplexSessionConfig(modalities=("text",)), model_config=None)

    async def plan(fence, seq, payload, config=None):
        return await plugin.prepare_append_plan(
            request_id="request",
            fence=fence,
            session_config=config or {},
            runtime_config=runtime,
            seq=seq,
            turn_seq=1,
            payload=payload,
            final=False,
            sampling_params=None,
        )

    blue = DuplexFence("session", epoch=0, turn_id=1)
    cancelled = DuplexFence("session", epoch=0, turn_id=2)
    recovery = DuplexFence("session", epoch=1, turn_id=3)
    first = {
        "type": "audio",
        "audio": base64.b64encode(np.zeros(32000, dtype=np.float32).tobytes()).decode(),
        "format": "pcm_f32le",
        "sample_rate_hz": 16000,
        "video_frames": ["blue-frame"],
    }
    old = {"type": "audio", "video_frames": ["old-frame"]}
    new = {"type": "audio", "video_frames": ["new-frame"]}

    first_plan = await plan(blue, 1, first)
    plugin.observe_submitted_append(state=state, fence=blue, prompt=first_plan.prompt, input_commit_seq=1)
    plugin.observe_committed_input(state=state, fence=blue, item_id="user_blue", source_id=1, input_commit_seq=1)
    plugin.observe_response_end(state=state, fence=blue, response_id="blue", completed=True)
    old_plan = await plan(cancelled, 2, old)
    plugin.observe_submitted_append(state=state, fence=cancelled, prompt=old_plan.prompt, input_commit_seq=2)
    if not late_commit:
        plugin.observe_committed_input(
            state=state, fence=cancelled, item_id="user_old", source_id=2, input_commit_seq=2
        )
    # Even if a generated reply was archived before playback, cancel excludes
    # it. An already-visible prefix can still be present in conversation.
    plugin.observe_response_end(state=state, fence=cancelled, response_id="old", completed=True)
    if late_commit:
        plugin.observe_committed_input(
            state=state,
            fence=DuplexFence("session", epoch=0, turn_id=3),
            item_id="user_old",
            source_id=2,
            input_commit_seq=2,
        )
    plugin.observe_response_end(state=state, fence=cancelled, response_id="old", completed=False)
    config = plugin.prepare_prompt_config(
        {
            "conversation": [
                {"role": "user", "content": "What color is this?"},
                {"role": "assistant", "content": "Blue."},
                {"role": "user", "content": "Please count."},
                {"role": "assistant", "content": "One"},
            ],
            "conversation_item_ids": ("user_blue", "item_blue", "user_old", "item_old"),
            "conversation_user_source_ids": {"user_blue": 1, "user_old": 2},
        },
        state=state,
        payload=new,
    )
    baseline = await plan(recovery, 1, new)
    recovered = await plan(recovery, 1, new, config)
    envelope = recovered.prompt["model_intermediate_buffer"]["duplex"]
    turn, interrupted = envelope[module.REPLAY_CONTEXT_KEY]
    assert turn["inputs"] == [{key: first[key] for key in ("audio", "format", "sample_rate_hz", "video_frames")}]
    assert turn["response_token_ids"] == list(map(ord, "Blue."))
    assert interrupted["user_only"] is True
    assert interrupted["inputs"] == [{"video_frames": ["old-frame"]}]
    assert interrupted["response_token_ids"] == []
    assert envelope["payload"] == new
    assert len(recovered.prompt["prompt_token_ids"]) == (
        len(baseline.prompt["prompt_token_ids"]) + turn["token_budget"] + interrupted["token_budget"]
    )
    assert envelope["session_window_replay"] is False
    assert "session_window_replay" not in baseline.prompt["model_intermediate_buffer"]["duplex"]

    deleted_user = {
        **config,
        "conversation": config["conversation"][1:],
        "conversation_item_ids": config["conversation_item_ids"][1:],
        "conversation_user_source_ids": {"user_old": 2},
    }
    reused_public_id = {**config, "conversation_user_source_ids": {"user_blue": 1, "user_old": 9}}
    for edited, expected in ((deleted_user, [interrupted]), (reused_public_id, [turn])):
        result = await plan(recovery, 1, new, edited)
        duplex = result.prompt["model_intermediate_buffer"]["duplex"]
        assert duplex[module.REPLAY_CONTEXT_KEY] == expected
        assert duplex["session_window_replay"] is False
    removed = {**config, "conversation_user_source_ids": {}}
    result = await plan(recovery, 1, new, removed)
    duplex = result.prompt["model_intermediate_buffer"]["duplex"]
    assert module.REPLAY_CONTEXT_KEY not in duplex
    assert duplex["session_window_replay"] is False
    for fence, seq in ((blue, 1), (recovery, 2)):
        continued = await plan(fence, seq, new, config)
        duplex = continued.prompt["model_intermediate_buffer"]["duplex"]
        assert module.REPLAY_CONTEXT_KEY not in duplex
        assert duplex.get("session_window_replay", True) is (fence.epoch == 0)
