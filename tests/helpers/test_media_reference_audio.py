# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from tests.helpers import media as media_helpers
from tests.helpers.media import resolve_test_audio_file


def test_resolve_test_audio_file_uses_env_path(tmp_path, monkeypatch):
    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"not-empty")
    monkeypatch.setenv("TEST_REF_AUDIO_PATH", str(ref_audio))

    resolved = resolve_test_audio_file(
        env_var="TEST_REF_AUDIO_PATH",
        asset_relative_path="missing/ref.wav",
        cache_name="unused.wav",
        url="https://example.invalid/ref.wav",
    )

    assert resolved == ref_audio.resolve()


def test_resolve_test_audio_file_uses_vendored_asset(tmp_path, monkeypatch):
    asset = tmp_path / "missing" / "ref.wav"
    asset.parent.mkdir()
    asset.write_bytes(b"not-empty")
    monkeypatch.delenv("TEST_REF_AUDIO_PATH", raising=False)
    monkeypatch.setattr(media_helpers, "_TEST_ASSETS_ROOT", tmp_path)

    resolved = resolve_test_audio_file(
        env_var="TEST_REF_AUDIO_PATH",
        asset_relative_path="missing/ref.wav",
        cache_name="unused.wav",
        url="https://example.invalid/ref.wav",
    )

    assert resolved == asset


def test_resolve_test_audio_file_uses_cache(tmp_path, monkeypatch):
    cache_dir = tmp_path / "asset-cache"
    cache_dir.mkdir()
    cached_audio = cache_dir / "cached.wav"
    cached_audio.write_bytes(b"not-empty")
    monkeypatch.delenv("TEST_REF_AUDIO_PATH", raising=False)
    monkeypatch.setenv("VLLM_OMNI_TEST_ASSET_CACHE", str(cache_dir))

    resolved = resolve_test_audio_file(
        env_var="TEST_REF_AUDIO_PATH",
        asset_relative_path="missing/ref.wav",
        cache_name="cached.wav",
        url="https://example.invalid/ref.wav",
    )

    assert resolved == cached_audio


def test_resolve_test_audio_file_downloads_to_cache(tmp_path, monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return b"downloaded"

    def fake_urlopen(url, timeout):
        assert url == "https://example.invalid/ref.wav"
        assert timeout == 30.0
        return Response()

    cache_dir = tmp_path / "empty-cache"
    monkeypatch.delenv("TEST_REF_AUDIO_PATH", raising=False)
    monkeypatch.setenv("VLLM_OMNI_TEST_ASSET_CACHE", str(cache_dir))
    monkeypatch.setattr(media_helpers.urllib.request, "urlopen", fake_urlopen)

    resolved = resolve_test_audio_file(
        env_var="TEST_REF_AUDIO_PATH",
        asset_relative_path="missing/ref.wav",
        cache_name="cached.wav",
        url="https://example.invalid/ref.wav",
    )

    assert resolved == cache_dir / "cached.wav"
    assert resolved.read_bytes() == b"downloaded"
