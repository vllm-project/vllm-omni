# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from io import BytesIO

import numpy as np
import pytest
import soundfile as sf
from vllm.multimodal.media import audio as audio_io

import vllm_omni  # noqa: F401  # Install Omni's runtime compatibility patches.

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def ogg_bytes():
    samples = np.sin(np.arange(8192) * 0.17).astype(np.float32)
    buffer = BytesIO()
    sf.write(buffer, samples, 16000, format="OGG", subtype="VORBIS")
    return buffer.getvalue()


@pytest.mark.parametrize("source", ["bytes", "file"])
def test_default_audio_decode_preserves_soundfile_waveform(ogg_bytes, tmp_path, source):
    expected, expected_rate = audio_io.load_audio_soundfile(BytesIO(ogg_bytes), sr=None)
    media = audio_io.AudioMediaIO()
    if source == "file":
        path = tmp_path / "sample.ogg"
        path.write_bytes(ogg_bytes)
        actual, rate = media.load_file(path)
    else:
        actual, rate = media.load_bytes(ogg_bytes)

    assert rate == expected_rate
    np.testing.assert_array_equal(actual, expected)


def test_auto_audio_decode_prefers_soundfile_and_forwards_limits(monkeypatch):
    expected = np.zeros(8, dtype=np.float32)
    calls = []

    def soundfile_loader(path, **kwargs):
        calls.append((path.tell(), kwargs))
        return expected, 16000

    monkeypatch.setattr(audio_io, "load_audio_soundfile", soundfile_loader)
    monkeypatch.setattr(audio_io, "load_audio_torchcodec", lambda *args, **kwargs: pytest.fail("wrong decoder"))
    buffer = BytesIO(b"encoded-audio")
    buffer.seek(4)
    actual, rate = audio_io.load_audio(
        buffer, sr=8000, mono=False, max_duration_s=10, max_decode_bytes=4096, backend="auto"
    )

    assert actual is expected and rate == 16000
    assert calls == [(0, {"sr": 8000, "mono": False, "max_duration_s": 10, "max_decode_bytes": 4096})]


@pytest.mark.parametrize("failure", [ImportError("missing soundfile"), sf.LibsndfileError(1)])
def test_auto_audio_decode_retains_upstream_format_fallback(monkeypatch, failure):
    expected = np.zeros(8, dtype=np.float32)
    calls = []

    def soundfile_loader(path, **kwargs):
        calls.append("soundfile")
        path.read()
        raise failure

    def torchcodec_loader(path, **kwargs):
        calls.append("torchcodec")
        assert path.tell() == 0
        return expected, 16000

    monkeypatch.setattr(audio_io, "load_audio_soundfile", soundfile_loader)
    monkeypatch.setattr(audio_io, "load_audio_torchcodec", torchcodec_loader)
    actual, rate = audio_io.AudioMediaIO().load_bytes(b"non-soundfile-container")

    assert actual is expected and rate == 16000
    assert calls == ["soundfile", "torchcodec"]


@pytest.mark.parametrize("failure", [ValueError("decode limit exceeded"), sf.LibsndfileError(2)])
def test_auto_audio_decode_propagates_guard_and_recognized_format_errors(monkeypatch, failure):
    def soundfile_loader(*args, **kwargs):
        raise failure

    monkeypatch.setattr(audio_io, "load_audio_soundfile", soundfile_loader)
    monkeypatch.setattr(audio_io, "load_audio_torchcodec", lambda *args, **kwargs: pytest.fail("error was swallowed"))

    with pytest.raises(type(failure)) as caught:
        audio_io.AudioMediaIO().load_bytes(b"invalid")
    assert caught.value is failure


@pytest.mark.parametrize("backend", ["soundfile", "torchcodec", "pyav"])
def test_explicit_audio_backend_is_preserved(monkeypatch, backend):
    expected = np.zeros(8, dtype=np.float32)
    selected = []

    def explicit_loader(*args, **kwargs):
        selected.append(backend)
        return expected, 16000

    monkeypatch.setitem(audio_io._AUDIO_LOADERS, backend, explicit_loader)
    monkeypatch.setattr(
        audio_io, "load_audio_soundfile", lambda *args, **kwargs: pytest.fail("explicit backend changed")
    )
    actual, rate = audio_io.AudioMediaIO(audio_backend=backend).load_bytes(b"explicit")

    assert actual is expected and rate == 16000
    assert selected == [backend]


def test_auto_audio_decode_retains_pyav_when_other_decoders_are_unavailable(monkeypatch):
    expected = np.zeros(8, dtype=np.float32)

    def soundfile_loader(*args, **kwargs):
        raise sf.LibsndfileError(1)

    def torchcodec_loader(*args, **kwargs):
        raise ImportError("missing torchcodec")

    def pyav_loader(path, **kwargs):
        assert path.tell() == 0
        return expected, 16000

    monkeypatch.setattr(audio_io, "load_audio_soundfile", soundfile_loader)
    monkeypatch.setattr(audio_io, "load_audio_torchcodec", torchcodec_loader)
    monkeypatch.setattr(audio_io, "load_audio_pyav", pyav_loader)

    actual, rate = audio_io.AudioMediaIO().load_bytes(b"fallback")
    assert actual is expected and rate == 16000


@pytest.mark.parametrize("limits", [{"max_duration_s": 0.1}, {"max_decode_bytes": 32}])
def test_auto_audio_decode_enforces_real_pcm_limits(ogg_bytes, limits):
    with pytest.raises(ValueError, match="duration|allocate"):
        audio_io.load_audio(BytesIO(ogg_bytes), sr=None, **limits)
