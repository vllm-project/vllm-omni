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
import soundfile as sf

from tests.helpers.mark import hardware_test
from tests.helpers.media import concat_audio, convert_audio_file_to_text, preprocess_text
from tests.helpers.runtime import get_model_prefix
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni import Omni
from vllm_omni.model_executor.models.spark_tts.prompt import control_prompt

MODEL = os.environ.get("SPARK_TTS_MODEL_PATH") or f"{get_model_prefix()}SparkAudio/Spark-TTS-0.5B"
DEPLOY_CONFIG = get_deploy_config_path("spark_tts.yaml")
SAMPLE_RATE = 16_000

pytestmark = [pytest.mark.tts, pytest.mark.core_model, pytest.mark.advanced_model]


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


def _synthesize(spark_omni: Omni, text: str) -> np.ndarray:
    outputs = spark_omni.generate([control_prompt(text)])

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
    return audio


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_spark_tts_control_voice_offline(spark_omni: Omni) -> None:
    _synthesize(spark_omni, "Hello, this is a Spark-TTS test.")


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_spark_tts_long_paragraph(spark_omni: Omni, run_level: str, tmp_path) -> None:
    paragraph = (
        "When we reached the village, the rain had finally stopped. "
        "We crossed the bridge and waited by the station while the last train rolled out of sight. "
        "A woman at the ticket window gave us a map and showed us the road that led to the harbor. "
        "We followed it past the bakery and the old stone school, listening to the sound of the waves. "
        "By the time we found the boat, every light along the shore was out. "
        "We stood on the wooden deck until we heard the silver bell ring at midnight."
    )
    audio = _synthesize(spark_omni, paragraph)
    # The old 1024-token cap could emit at most ~20.5 s at 50 codec frames/s.
    assert audio.size > 21 * SAMPLE_RATE, f"Long paragraph cut short at {audio.size / SAMPLE_RATE:.2f} s"
    if run_level == "advanced_model":
        path = tmp_path / "paragraph.wav"
        sf.write(path, audio, SAMPLE_RATE)
        transcript = convert_audio_file_to_text(str(path), language="en")
        tail = preprocess_text(transcript).split()[-12:]
        assert "midnight" in tail and ("silver" in tail or "bell" in tail), transcript
