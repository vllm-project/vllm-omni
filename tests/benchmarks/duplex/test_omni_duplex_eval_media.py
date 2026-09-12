# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import pytest

from vllm_omni.benchmarks.duplex.omni_duplex_eval_media import materialize_media

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_dict_with_bytes_materializes_a_real_file(tmp_path):
    # A Hugging Face Audio/Video feature value {"bytes": ..., "path": ...} used
    # to short-circuit on ``path`` (the artifact name inside the dataset, which
    # does not exist on disk), so downstream ffmpeg saw a missing file. Inline
    # bytes must win and be written to a real file whose content equals bytes.
    payload = b"\x00\xff\x01RIFFfake-content"
    resolved = materialize_media({"bytes": payload, "path": "question_audio.wav"}, tmp_path, "s1_question", ".wav")
    assert resolved.is_file()
    assert resolved.read_bytes() == payload
    assert resolved.parent == tmp_path
    assert resolved.name == "s1_question.wav"


def test_dict_extension_comes_from_path_safely(tmp_path):
    payload = b"flac-ish-bytes"
    resolved = materialize_media({"bytes": payload, "path": "audio/clip.FLAC"}, tmp_path, "s1", ".wav")
    assert resolved.name == "s1.flac"
    assert resolved.read_bytes() == payload
    # A path with no usable extension falls back to the caller's suffix.
    resolved = materialize_media({"bytes": payload, "path": "audio/clip"}, tmp_path, "s2", ".wav")
    assert resolved.name == "s2.wav"
    assert resolved.read_bytes() == payload


def test_materialization_is_idempotent_and_updated(tmp_path):
    payload = b"abc"
    first = materialize_media({"bytes": payload, "path": "q.wav"}, tmp_path, "s", ".wav")
    second = materialize_media({"bytes": payload, "path": "q.wav"}, tmp_path, "s", ".wav")
    assert first == second
    assert first.read_bytes() == payload
    replacement = b"xyz"
    third = materialize_media({"bytes": replacement, "path": "q.wav"}, tmp_path, "s", ".wav")
    assert third == first
    assert third.read_bytes() == replacement


def test_str_path_and_path_only_mapping_returned_as_is(tmp_path):
    existing = tmp_path / "clip.mp4"
    existing.write_bytes(b"video")
    assert materialize_media(str(existing), tmp_path, "s", ".mp4") == existing
    assert materialize_media({"path": str(existing)}, tmp_path, "s", ".mp4") == existing


def test_raw_bytes_value_is_written(tmp_path):
    payload = b"raw-media"
    resolved = materialize_media(payload, tmp_path, "r", ".bin")
    assert resolved.name == "r.bin"
    assert resolved.read_bytes() == payload


def test_unresolvable_value_raises_clear_error(tmp_path):
    with pytest.raises(ValueError, match="cannot materialize"):
        materialize_media(None, tmp_path, "s", ".wav")
    with pytest.raises(ValueError, match="cannot materialize"):
        materialize_media({"path": None}, tmp_path, "s", ".wav")
