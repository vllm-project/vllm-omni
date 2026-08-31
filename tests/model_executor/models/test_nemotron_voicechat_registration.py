# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L1 registration + config-shim tests for NemotronVoiceChat (CPU, no weights)."""

import subprocess
from pathlib import Path

import pytest

from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
from vllm_omni.config.stage_config import StageExecutionType
from vllm_omni.model_executor.models.nemotron_voicechat.configuration_nemotron_voicechat import (
    NemotronVoiceChatConfig,
)
from vllm_omni.model_executor.models.registry import _OMNI_MODELS

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_ARCHS = (
    "NemotronVoiceChatThinkerForConditionalGeneration",
    "NemotronVoiceChatTalkerForConditionalGeneration",
    "NemotronVoiceChatCode2Wav",
)


def _minimal_nemo_dict() -> dict:
    return {
        "data": {"frame_length": 0.08, "source_sample_rate": 16000, "target_sample_rate": 22050},
        "model": {
            "inference_speaker_name": "Aria",
            "stt": {
                "model": {
                    "perception": {"encoder": {}, "preprocessor": {}, "modality_adapter": {}},
                    "pretrained_llm": "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
                    "use_function_head": True,
                }
            },
            "speech_generation": {
                "model": {
                    "tts_config": {"backbone_config": {"hidden_size": 1152, "sliding_window": 7500}},
                    "codec_config": {"num_quantizers": 31, "codebook_size": 1024},
                }
            },
        },
    }


def test_pipeline_registered_with_three_stages() -> None:
    pipeline = OMNI_PIPELINES["nemotron_voicechat"]
    assert OMNI_PIPELINES["nemotron_labs_voicechat"] is pipeline
    assert len(pipeline.stages) == 3
    thinker, talker, code2wav = pipeline.stages
    assert thinker.execution_type == StageExecutionType.LLM_AR
    assert thinker.final_output and thinker.final_output_type == "text"
    assert thinker.owns_tokenizer
    assert talker.execution_type == StageExecutionType.LLM_AR
    assert talker.model_arch == "NemotronVoiceChatTalkerForConditionalGeneration"
    assert talker.input_sources == (0,)
    assert code2wav.execution_type == StageExecutionType.LLM_GENERATION
    assert code2wav.final_output and code2wav.final_output_type == "audio"
    assert code2wav.input_sources == (1,)
    assert pipeline.validate() == []


def test_architectures_registered() -> None:
    for arch in _ARCHS:
        assert arch in _OMNI_MODELS, f"{arch} missing from _OMNI_MODELS"


def test_config_parses_nemo_layout_without_hub_access(monkeypatch) -> None:
    # Provide the thinker text config explicitly so the test never touches HF.
    cfg = NemotronVoiceChatConfig(
        thinker_text_config={"model_type": "llama", "hidden_size": 4480, "num_hidden_layers": 56},
        **_minimal_nemo_dict(),
    )
    assert cfg.get_text_config().hidden_size == 4480
    assert cfg.talker_config.hidden_size == 1152
    assert cfg.talker_config.sliding_window == 7500
    assert cfg.code2wav_config.num_quantizers == 31
    assert cfg.inference_speaker_name == "Aria"


@pytest.mark.parametrize(
    "mutation",
    ["drop_stt", "drop_speech_generation", "drop_perception", "drop_codec"],
)
def test_config_rejects_non_voicechat_checkpoints(mutation: str) -> None:
    raw = _minimal_nemo_dict()
    if mutation == "drop_stt":
        del raw["model"]["stt"]
    elif mutation == "drop_speech_generation":
        del raw["model"]["speech_generation"]
    elif mutation == "drop_perception":
        del raw["model"]["stt"]["model"]["perception"]
    else:
        del raw["model"]["speech_generation"]["model"]["codec_config"]
    with pytest.raises(ValueError, match="NemotronVoiceChat|missing"):
        NemotronVoiceChatConfig(
            thinker_text_config={"model_type": "llama", "hidden_size": 4480},
            **raw,
        )


def test_no_nemo_imports_in_nemotron_voicechat_tree() -> None:
    """Dependency guard: the vendored tree must not import the nemo package.

    Scoped to the nemotron_voicechat model tree (other models may have their
    own optional nemo integrations).
    """
    repo_root = Path(__file__).resolve().parents[3]
    targets = [
        str(repo_root / "vllm_omni/model_executor/models/nemotron_voicechat"),
        str(repo_root / "vllm_omni/model_executor/stage_input_processors/nemotron_voicechat.py"),
    ]
    result = subprocess.run(
        ["grep", "-rnE", r"^\s*(import nemo\b|from nemo(\.|\s))", *targets, "--include=*.py"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, f"Found nemo imports in the nemotron_voicechat tree:\n{result.stdout}"
