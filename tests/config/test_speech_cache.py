# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.config.omni_config import _get_deploy_config
from vllm_omni.config.speech_cache import SpeechCacheConfig
from vllm_omni.config.stage_config import PipelineConfig, StagePipelineConfig, load_deploy_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("field", ["resolve_max_bytes", "resolve_max_entries", "speaker_max_bytes"])
@pytest.mark.parametrize("value", [-1, True, 1.5, "1024", None])
def test_invalid_limits(field, value):
    with pytest.raises(ValueError, match=field):
        SpeechCacheConfig(**{field: value})


def test_yaml_inheritance(tmp_path):
    (tmp_path / "base.yaml").write_text(
        "speech_cache:\n  resolve_max_bytes: 1234\n  resolve_max_entries: 8\n  speaker_max_bytes: 16\n"
    )
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text("base_config: base.yaml\nspeech_cache:\n  resolve_max_entries: 0\n")
    deploy = load_deploy_config(overlay)
    expected = SpeechCacheConfig(resolve_max_bytes=1234, resolve_max_entries=0, speaker_max_bytes=16)
    assert deploy.speech_cache == expected


@pytest.mark.parametrize("value", ["null", "[]", "false", "12", "{resolve_max_bytes: -1}", "{unknown: 10}"])
@pytest.mark.parametrize("inherited", [False, True])
def test_invalid_yaml(tmp_path, value, inherited):
    path = tmp_path / "deploy.yaml"
    (tmp_path / "base.yaml").write_text("speech_cache:\n  resolve_max_entries: 8\n")
    prefix = "base_config: base.yaml\n" if inherited else ""
    path.write_text(f"{prefix}speech_cache: {value}\n")
    with pytest.raises((ValueError, TypeError)):
        load_deploy_config(path)


@pytest.mark.parametrize("explicit", [True, False])
def test_selected_deploy_path_preserves_speech_cache(tmp_path, monkeypatch, explicit):
    import vllm_omni.config.omni_config as module

    path = tmp_path / "test.yaml"
    path.write_text("speech_cache:\n  resolve_max_entries: 17\n")
    monkeypatch.setattr(module, "_DEPLOY_DIR", tmp_path)
    pipeline = PipelineConfig(
        model_type="test",
        default_deploy_config_name="test.yaml",
        stages=(StagePipelineConfig(stage_id=0, model_stage="test", final_output=True),),
    )
    deploy, selected_path = _get_deploy_config(pipeline, None, str(path) if explicit else None)
    assert selected_path == str(path)
    assert load_deploy_config(selected_path).speech_cache == deploy.speech_cache
    assert deploy.speech_cache.resolve_max_entries == 17
