# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402

# Stub out the server-entry modules that are pulled in by
# vllm_omni/entrypoints/openai/__init__.py before the real import.
# Those modules transitively import vllm.entrypoints.chat_utils symbols
# (e.g. get_history_tool_calls_cnt) that may not exist in all installed
# vllm versions.  The static method under test only uses stdlib, so it
# loads cleanly once the chain is broken.
import sys
from unittest.mock import MagicMock

sys.modules.setdefault("vllm_omni.entrypoints.openai.serving_chat", MagicMock())
sys.modules.setdefault("vllm_omni.entrypoints.openai.api_server", MagicMock())

import hashlib
import logging
import pathlib
import tempfile
import time

import pytest

from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_local_file_cache_key_invalidation():
    """Modified file must produce a different cache key (mtime change)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        test_file = tmp_path / "test_audio.wav"

        # 1. Create a dummy file and get its cache key
        test_file.write_text("dummy audio data 1")
        file_uri = f"file://{test_file}"

        key1 = OmniOpenAIServingSpeech._get_ref_audio_cache_key(file_uri)

        # Wait to ensure mtime changes on the filesystem
        time.sleep(0.01)

        # 2. Modify the file — same byte length so the test exercises mtime,
        #    not size.
        test_file.write_text("dummy audio data 2")

        # 3. Get the new cache key
        key2 = OmniOpenAIServingSpeech._get_ref_audio_cache_key(file_uri)

        assert key1 != key2, "Cache key should change when file is modified!"


def test_remote_url_cache_key():
    """Remote URLs must produce a stable, repeatable cache key."""
    url = "https://example.com/audio.wav"
    key1 = OmniOpenAIServingSpeech._get_ref_audio_cache_key(url)
    key2 = OmniOpenAIServingSpeech._get_ref_audio_cache_key(url)

    assert key1 == key2, "Cache key for URLs should remain constant!"


def test_missing_file_cache_key(caplog):
    """A missing file falls back to the URI-string hash and emits a warning
    so the stale-cache risk is visible in the logs."""
    file_uri = "file:///path/to/nonexistent/file.wav"

    # Attach caplog.handler directly to the module logger because
    # vllm_omni reparents under vllm which sets propagate=False,
    # so records never reach the root logger where caplog installs
    # its handler by default.
    target_logger = logging.getLogger("vllm_omni.entrypoints.openai.serving_speech")
    target_logger.addHandler(caplog.handler)
    prev_level = target_logger.level
    target_logger.setLevel(logging.DEBUG)
    try:
        key = OmniOpenAIServingSpeech._get_ref_audio_cache_key(file_uri)
    finally:
        target_logger.removeHandler(caplog.handler)
        target_logger.setLevel(prev_level)

    expected_key = hashlib.sha1(file_uri.encode("utf-8")).hexdigest()
    assert key == expected_key, "Missing files should fallback to the string hash."
    assert any("stale cache" in r.getMessage() for r in caplog.records), (
        "A warning about stale cache must be emitted when os.stat fails"
    )


def test_percent_encoded_file_uri():
    """Percent-encoded file:// URIs (e.g. %20 for spaces) must be decoded
    so that os.stat hits the real file, not a wrong/nonexistent path."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        # Create a file whose name contains a space
        test_file = tmp_path / "my audio.wav"
        test_file.write_text("dummy audio data")
        # Build a URI with the space percent-encoded
        encoded_uri = f"file://{str(test_file).replace(' ', '%20')}"

        key = OmniOpenAIServingSpeech._get_ref_audio_cache_key(encoded_uri)
        # The key must incorporate mtime/size (not fall back to string-only),
        # so it should differ from a plain SHA-1 of the URI string.
        string_only_key = hashlib.sha1(encoded_uri.encode("utf-8")).hexdigest()
        assert key != string_only_key, (
            "Percent-encoded URI should be decoded and stat'd, not fall back to string-only cache key!"
        )
