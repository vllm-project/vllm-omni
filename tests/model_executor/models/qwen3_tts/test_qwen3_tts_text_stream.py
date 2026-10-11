# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the Qwen3-TTS text stream: request fields and the ids/end files."""

import numpy as np
import pytest

from vllm_omni.model_executor.models.qwen3_tts import text_stream


@pytest.fixture
def stream_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(text_stream, "STREAM_DIR", str(tmp_path))
    return tmp_path


def _append(stream_dir, key, ids):
    with open(stream_dir / f"{key}.ids", "ab") as f:
        f.write(np.asarray(ids, dtype="<i4").tobytes())


def test_request_fields_name_the_stream_and_the_lead():
    assert text_stream.stream_key({"text_stream": ["take-1"]}) == "take-1"
    assert text_stream.stream_key({}) is None
    assert text_stream.stream_key(None) is None
    assert text_stream.text_lead({"text_lead": [16]}) == 16
    assert text_stream.text_lead({}) == 0


def test_a_stream_reports_appended_ids_then_its_end(stream_dir):
    assert text_stream.count("k") == (0, False)
    assert text_stream.read_ids("k")[0].size == 0

    _append(stream_dir, "k", [5, 6, 7])
    assert text_stream.count("k") == (3, False)
    ids, ended = text_stream.read_ids("k", 1)
    assert ids.tolist() == [6, 7]
    assert not ended

    _append(stream_dir, "k", [8])
    (stream_dir / "k.end").touch()
    assert text_stream.count("k") == (4, True)
    assert text_stream.read_ids("k", 3)[0].tolist() == [8]


def test_a_partly_written_id_is_not_read(stream_dir):
    (stream_dir / "k.ids").write_bytes(np.asarray([9], dtype="<i4").tobytes() + b"\x01\x02")
    assert text_stream.read_ids("k")[0].tolist() == [9]
