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


@pytest.mark.asyncio
async def test_window_configuration_is_prepared_by_plugin(monkeypatch):
    tokenizer = SimpleNamespace(
        all_special_ids=[90],
        unk_token_id=-1,
        eos_token_id=99,
        encode=lambda text, add_special_tokens=False: [ord(char) for char in text],
        convert_tokens_to_ids=lambda token: {"<|listen|>": 91}.get(token, -1),
    )
    monkeypatch.setattr(module, "_load_tokenizer", lambda config: tokenizer)
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
async def test_explicit_reference_audio_skips_the_model_default(monkeypatch):
    async def fail_default(_model_config):
        raise AssertionError("the model default must not replace an explicit ref_audio")

    monkeypatch.setattr(module, "resolve_default_ref_audio", fail_default)

    async def resolve_explicit(ref_audio, *, model_config):
        del model_config
        return _resolved_audio(ref_audio)

    monkeypatch.setattr(module, "_load_tokenizer", lambda _model_config: None)
    monkeypatch.setattr(module, "resolve_ref_audio", resolve_explicit)

    config = DuplexSessionConfig(ref_audio="data:audio/wav;base64,AAAA")
    runtime = await _plugin().prepare_runtime_config(config, model_config=None)

    assert runtime["ref_audio_sample_rate_hz"] == 16_000


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


def _resolved_audio(ref_audio: str):
    del ref_audio
    return np.ones(1600, dtype=np.float32), 16_000


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
