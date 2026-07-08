# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.diffusion import io_support

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_dummy_run_num_frames_uses_explicit_model_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    class JointAudioVideoModel:
        dummy_run_num_frames = 2

    monkeypatch.setattr(
        io_support.DiffusionModelRegistry,
        "_try_load_model_cls",
        lambda model_class_name: JointAudioVideoModel,
    )

    assert io_support.get_dummy_run_num_frames("joint_audio_video", supports_audio_input=False) == 2


def test_dummy_run_num_frames_keeps_audio_output_default(monkeypatch: pytest.MonkeyPatch) -> None:
    class AudioOutputModel:
        support_audio_output = True

    monkeypatch.setattr(
        io_support.DiffusionModelRegistry,
        "_try_load_model_cls",
        lambda model_class_name: AudioOutputModel,
    )

    assert io_support.get_dummy_run_num_frames("audio_output", supports_audio_input=False) == 2


def test_dummy_run_num_frames_defaults_to_single_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    class VideoOnlyModel:
        pass

    monkeypatch.setattr(
        io_support.DiffusionModelRegistry,
        "_try_load_model_cls",
        lambda model_class_name: VideoOnlyModel,
    )

    assert io_support.get_dummy_run_num_frames("video_only", supports_audio_input=False) == 1


def test_dummy_run_num_frames_uses_audio_input_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        io_support.DiffusionModelRegistry,
        "_try_load_model_cls",
        lambda model_class_name: None,
    )

    assert io_support.get_dummy_run_num_frames("unknown", supports_audio_input=True) == 2
