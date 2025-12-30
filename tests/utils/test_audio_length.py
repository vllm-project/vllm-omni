# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest


def test_resolve_max_mel_frames_default(monkeypatch):
    from vllm_omni.utils.audio_length import VLLM_OMNI_MAX_MEL_FRAMES_ENV, resolve_max_mel_frames

    monkeypatch.delenv(VLLM_OMNI_MAX_MEL_FRAMES_ENV, raising=False)
    assert resolve_max_mel_frames(None, default=30000) == 30000


def test_resolve_max_mel_frames_env_override(monkeypatch):
    from vllm_omni.utils.audio_length import VLLM_OMNI_MAX_MEL_FRAMES_ENV, resolve_max_mel_frames

    monkeypatch.setenv(VLLM_OMNI_MAX_MEL_FRAMES_ENV, "6000")
    assert resolve_max_mel_frames(None, default=30000) == 6000
    # Explicit argument always wins
    assert resolve_max_mel_frames(123, default=30000) == 123


@pytest.mark.parametrize("repeats", [2, 4])
@pytest.mark.parametrize("code_len", [0, 1, 2, 3, 10, 32768])
@pytest.mark.parametrize("max_mel_frames", [None, -1, 0, 1, 2, 3, 5, 6, 7, 6000, 30000])
def test_cap_and_align_mel_length_no_mismatch(repeats, code_len, max_mel_frames):
    """Guard that any max_mel_frames yields a mel length aligned to repeats, and
    consistent with the truncated code length (prevents concat mismatch).
    """
    from vllm_omni.utils.audio_length import cap_and_align_mel_length

    target_code_len, target_mel_len = cap_and_align_mel_length(
        code_len=code_len,
        repeats=repeats,
        max_mel_frames=max_mel_frames,
    )

    assert isinstance(target_code_len, int)
    assert isinstance(target_mel_len, int)

    if code_len == 0:
        assert target_code_len == 0
        assert target_mel_len == 0
        return

    assert target_code_len >= 1
    assert target_mel_len >= repeats
    assert target_mel_len % repeats == 0
    assert target_mel_len == target_code_len * repeats
    assert target_code_len <= code_len

    if max_mel_frames is not None and int(max_mel_frames) > 0 and int(max_mel_frames) >= repeats:
        assert target_mel_len <= int(max_mel_frames)
