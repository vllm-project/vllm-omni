# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""GPU L2 smoke test for Spark-TTS's native two-stage offline pipeline.

This uses the 0.5B checkpoint's real weights. Random LLM weights do not
produce the semantic and global token pair required by BiCodec, so the usual
``core_model`` dummy-weight patch cannot exercise this model's output path.
"""

import importlib.util
import os

import numpy as np
import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import concat_audio
from tests.helpers.runtime import get_model_prefix
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni import Omni
from vllm_omni.model_executor.models.spark_tts.prompt import control_prompt

MODEL = os.environ.get("SPARK_TTS_MODEL_PATH") or f"{get_model_prefix()}SparkAudio/Spark-TTS-0.5B"
DEPLOY_CONFIG = get_deploy_config_path("spark_tts.yaml")
SAMPLE_RATE = 16_000

pytestmark = pytest.mark.tts


@pytest.fixture(scope="module")
def spark_omni():
    if importlib.util.find_spec("sparktts") is None:
        pytest.fail("Official Spark-TTS sparktts sources must be importable; see the Spark-TTS CI setup")
    # Use Omni directly so L2 exercises real codec tokens and only shuts down
    # the stage processes belonging to this test's engine.
    engine = Omni(model=MODEL, deploy_config=DEPLOY_CONFIG)
    try:
        yield engine
    finally:
        engine.close()


@pytest.mark.core_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_spark_tts_control_voice_offline(spark_omni: Omni) -> None:
    """Text -> native Qwen2 stage -> BiCodec stage -> finite 16 kHz audio."""
    outputs = spark_omni.generate([control_prompt("Hello, this is a Spark-TTS test.")])

    assert len(outputs) == 1
    assert len(outputs[0].outputs) == 1
    payload = outputs[0].outputs[0].multimodal_output
    audio_value = payload.get("audio")
    if audio_value is None:
        audio_value = payload.get("model_outputs")
    assert audio_value is not None, f"Spark-TTS produced no audio: {list(payload)}"

    audio = concat_audio(audio_value)
    assert audio.size > 0, "Spark-TTS produced empty audio"
    assert np.isfinite(audio).all(), "Spark-TTS produced non-finite audio"

    sample_rate = payload.get("sr")
    if isinstance(sample_rate, list):
        assert sample_rate, "Spark-TTS omitted its sample rate"
        sample_rate = sample_rate[-1]
    if hasattr(sample_rate, "item"):
        sample_rate = sample_rate.item()
    assert sample_rate == SAMPLE_RATE
