# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_resolve_max_mel_frames_default():
    from vllm_omni.model_executor.models.qwen2_5_omni.audio_length import resolve_max_mel_frames

    assert resolve_max_mel_frames(None, default=30000) == 30000
    assert resolve_max_mel_frames(None, default=6000) == 6000


def test_resolve_max_mel_frames_explicit():
    from vllm_omni.model_executor.models.qwen2_5_omni.audio_length import resolve_max_mel_frames

    # Explicit argument always wins over default
    assert resolve_max_mel_frames(123, default=30000) == 123
    assert resolve_max_mel_frames(6000, default=30000) == 6000
    assert resolve_max_mel_frames(0, default=30000) == 0


@pytest.mark.parametrize("repeats", [2, 4])
@pytest.mark.parametrize("code_len", [0, 1, 32768])
@pytest.mark.parametrize("max_mel_frames", [None, -1, 0, 1, 6000, 30000])
def test_cap_and_align_mel_length_no_mismatch(repeats, code_len, max_mel_frames):
    """Guard that any max_mel_frames yields a mel length aligned to repeats, and
    consistent with the truncated code length (prevents concat mismatch).
    """
    from vllm_omni.model_executor.models.qwen2_5_omni.audio_length import cap_and_align_mel_length

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
