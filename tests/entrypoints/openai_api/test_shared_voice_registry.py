# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Runtime voices registered through one API process are seen by the others.

Every frontend of a multi-API deployment is modelled as its own
``OmniOpenAIServingSpeech`` sharing ``SPEAKER_SAMPLES_DIR``.
"""

import asyncio
import io
import json
import os
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
from fastapi import UploadFile
from starlette.datastructures import Headers

from vllm_omni.config.speech_cache import SpeechCacheConfig
from vllm_omni.entrypoints.openai.errors import InvalidVoiceReferenceError
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_EMBEDDING = json.dumps([0.25] * 8)


@pytest.fixture
def frontend(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEAKER_SAMPLES_DIR", str(tmp_path))

    def make() -> OmniOpenAIServingSpeech:
        server = OmniOpenAIServingSpeech.__new__(OmniOpenAIServingSpeech)
        server.speech_cache_config = SpeechCacheConfig()
        server._adapter = None
        server._init_speaker_storage()
        return server

    return make


def _wav_upload(name: str = "voice.wav") -> UploadFile:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(16000 * 2, dtype=np.float32), 16000, format="WAV")
    buf.seek(0)
    return UploadFile(file=buf, filename=name, headers=Headers({"content-type": "audio/wav"}))


def test_uploads_and_deletions_reach_every_frontend(frontend, tmp_path):
    a, b = frontend(), frontend()
    asyncio.run(a.upload_voice(_wav_upload(), "consent", "Alice", ref_text="hello"))
    asyncio.run(a.upload_voice_embedding(_EMBEDDING, "consent", "Bob"))

    assert {"alice", "bob"} <= b._get_available_speakers()
    assert b.uploaded_speakers["alice"]["file_path"] == a.uploaded_speakers["alice"]["file_path"]
    assert b.uploaded_speakers["alice"]["ref_text"] == "hello"
    assert b._get_uploaded_speaker_embedding("bob") == pytest.approx([0.25] * 8)

    b._ref_audio_data_url_cache["alice"] = "stale"
    asyncio.run(a.delete_voice("alice"))
    assert "alice" not in b._get_available_speakers()
    assert "alice" not in b._ref_audio_data_url_cache
    with pytest.raises(InvalidVoiceReferenceError):
        asyncio.run(b.delete_voice("alice"))
    # No temporary or stale voice files are left behind.
    assert sorted(p.name for p in tmp_path.iterdir() if p.name != ".voices.lock") == [
        a.uploaded_speakers["bob"]["file_path"].rsplit("/", 1)[-1]
    ]


def test_overwrite_from_another_frontend_replaces_the_voice(frontend, tmp_path):
    a, b = frontend(), frontend()
    first = asyncio.run(a.upload_voice_embedding(_EMBEDDING, "c1", "Carol"))
    a._ref_audio_data_url_cache["carol"] = "stale"
    second = asyncio.run(b.upload_voice_embedding(json.dumps([0.5] * 8), "c2", "Carol"))

    # Upload times stay unique across frontends, so cached artifacts are versioned.
    assert second["created_at"] > first["created_at"]
    assert len(list(tmp_path.glob("*.safetensors"))) == 1
    a._refresh_uploaded_speakers()
    assert a.uploaded_speakers["carol"]["created_at"] == second["created_at"]
    assert "carol" not in a._ref_audio_data_url_cache
    assert a._get_uploaded_speaker_embedding("carol") == pytest.approx([0.5] * 8)


def test_upload_cap_and_immutable_policy_count_every_frontend(frontend, monkeypatch):
    monkeypatch.setenv("SPEAKER_MAX_UPLOADED", "1")
    a, b = frontend(), frontend()
    asyncio.run(a.upload_voice_embedding(_EMBEDDING, "c", "One"))
    with pytest.raises(ValueError, match="limit reached"):
        asyncio.run(b.upload_voice_embedding(_EMBEDDING, "c", "Two"))

    monkeypatch.setenv("SPEAKER_MAX_UPLOADED", "10")
    monkeypatch.setenv("VLLM_OMNI_SPEAKER_REGISTRATION_POLICY", "immutable")
    c, d = frontend(), frontend()
    asyncio.run(c.upload_voice_embedding(_EMBEDDING, "c", "Fixed"))
    with pytest.raises(ValueError, match="immutable"):
        asyncio.run(d.upload_voice_embedding(_EMBEDDING, "c", "Fixed"))


def test_registration_waits_for_another_frontends_lock(frontend):
    a, b = frontend(), frontend()

    async def scenario() -> None:
        async with a._voice_registry_lock():
            upload = asyncio.create_task(b.upload_voice_embedding(_EMBEDDING, "c", "Dora"))
            await asyncio.sleep(0.2)
            assert not upload.done()
        await asyncio.wait_for(upload, timeout=5)

    asyncio.run(scenario())
    assert "dora" in a._get_available_speakers()


def test_unchanged_directory_costs_no_file_reads(frontend):
    a, b = frontend(), frontend()
    asyncio.run(a.upload_voice_embedding(_EMBEDDING, "c", "Eve"))
    asyncio.run(a.upload_voice_embedding(_EMBEDDING, "c", "Finn"))
    b._refresh_uploaded_speakers()

    import safetensors

    with patch.object(safetensors, "safe_open", wraps=safetensors.safe_open) as reads:
        b._refresh_uploaded_speakers()
        b._get_available_speakers()
        assert reads.call_count == 0
        asyncio.run(a.upload_voice_embedding(_EMBEDDING, "c", "Gil"))
        b._refresh_uploaded_speakers()
        assert reads.call_count == 1  # only the new file


def test_change_within_one_timestamp_tick_is_not_missed(frontend, tmp_path):
    a, b = frontend(), frontend()
    asyncio.run(a.upload_voice_embedding(_EMBEDDING, "c", "Ivy"))
    b._refresh_uploaded_speakers()
    stamp = b._voice_dir_mtime_ns
    asyncio.run(a.upload_voice_embedding(_EMBEDDING, "c", "Jay"))
    # Coarse filesystem timestamps can give both changes the same directory mtime.
    os.utime(tmp_path, ns=(stamp, stamp))
    b._voice_scan_ns = stamp + 1  # b's last scan started within that tick
    assert "jay" in b._get_available_speakers()

    # Once a scan started well after the mtime, an unchanged directory is not listed.
    b._voice_scan_ns = stamp + 10**9
    with patch("os.scandir", wraps=os.scandir) as listing:
        b._get_available_speakers()
        assert listing.call_count == 0


def test_partial_files_and_manual_entries_are_left_alone(frontend, tmp_path):
    a = frontend()
    a.uploaded_speakers["manual"] = {"name": "manual", "created_at": 1}
    (tmp_path / ".half.safetensors.tmp").write_bytes(b"partial")

    def failing_save(tensors, path, metadata=None):
        open(path, "wb").write(b"partial")
        raise OSError("disk full")

    with patch("safetensors.torch.save_file", side_effect=failing_save), pytest.raises(OSError):
        asyncio.run(a.upload_voice_embedding(_EMBEDDING, "c", "Hal"))
    a._refresh_uploaded_speakers(force=True)
    assert set(a.uploaded_speakers) == {"manual"}
    # The failed write left no voice file and removed its temporary file.
    assert sorted(p.name for p in tmp_path.iterdir()) == [".half.safetensors.tmp", ".voices.lock"]


def test_failed_reupload_keeps_the_previous_voice(frontend, tmp_path):
    a, b = frontend(), frontend()
    asyncio.run(a.upload_voice_embedding(_EMBEDDING, "c", "Kim"))
    previous = a.uploaded_speakers["kim"]["file_path"]

    with patch("safetensors.torch.save_file", side_effect=OSError("disk full")), pytest.raises(OSError):
        asyncio.run(b.upload_voice_embedding(json.dumps([0.5] * 8), "c", "Kim"))
    # The replacement never landed, so the old voice stays registered everywhere.
    assert b.uploaded_speakers["kim"]["file_path"] == previous
    assert "kim" in a._get_available_speakers()
    assert [p.name for p in tmp_path.glob("*.safetensors")] == [previous.rsplit("/", 1)[-1]]
