# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real-checkpoint serving coverage for CosyVoice3 draft verification."""

import io
import json
import os
import tempfile
import wave
from copy import deepcopy
from pathlib import Path

import pytest
from vllm.logger import DEFAULT_LOGGING_CONFIG

from tests.helpers.mark import hardware_test
from tests.helpers.media import get_asset_path
from tests.helpers.runtime import OmniServerParams, get_model_prefix
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config
from vllm_omni.transformers_utils.repo_utils import hf_api

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
pytestmark = [pytest.mark.advanced_model, pytest.mark.tts, pytest.mark.slow]

_MODEL = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
_DRAFT = "yuekai/cosyvoice3_llm_dspark"
_DRAFT_REVISION = "0b1b28dbb751bac3b74eea5dcc11e0570c91563b"
_TEMP_DIR = tempfile.TemporaryDirectory(prefix="cosyvoice3-dspark-")
_DRAFT_DIR = Path(_TEMP_DIR.name) / "draft-speech"
_LOG = Path(_TEMP_DIR.name) / "verification.log"
_LOG_CONFIG_PATH = Path(_TEMP_DIR.name) / "logging.json"
_LOG_CONFIG = deepcopy(DEFAULT_LOGGING_CONFIG)
_LOG_CONFIG["handlers"]["cosyvoice3_verification"] = {
    "class": "logging.FileHandler",
    "filename": str(_LOG),
    "formatter": "vllm",
    "level": "DEBUG",
}
_LOG_CONFIG["loggers"]["vllm_omni.model_executor.models.cosyvoice3.cosyvoice3"] = {
    "handlers": ["cosyvoice3_verification"],
    "level": "DEBUG",
    "propagate": True,
}
_LOG_CONFIG_PATH.write_text(json.dumps(_LOG_CONFIG), encoding="utf-8")


@pytest.fixture(scope="session")
def cosyvoice3_dspark_draft() -> Path:
    from vllm_omni.model_executor.models.cosyvoice3.dspark import convert_checkpoint

    source = Path(get_model_prefix()) / _DRAFT if get_model_prefix() else None
    if source is None or not (source / "model.safetensors").is_file():
        source = Path(
            hf_api().snapshot_download(
                _DRAFT, revision=_DRAFT_REVISION, allow_patterns=["config.json", "model.safetensors"]
            )
        )
    convert_checkpoint(source, _DRAFT_DIR)
    return _DRAFT_DIR


_SERVER_PARAMS = [
    pytest.param(
        OmniServerParams(
            model=_MODEL,
            stage_config_path=modify_stage_config(
                get_deploy_config_path("cosyvoice3.yaml"),
                updates={
                    "stages": {
                        0: {
                            "model_runner": "v2",
                            "max_num_seqs": 2,
                            "max_model_len": 4096,
                            "speculative_config": {
                                "method": "dspark",
                                "model": str(_DRAFT_DIR),
                                "num_speculative_tokens": 7,
                                "draft_sample_method": "probabilistic",
                            },
                        },
                        1: {"max_num_seqs": 2},
                    }
                },
            ),
            server_args=["--trust-remote-code", "--hf-overrides", '{"cosyvoice3_sampling_mode":"standard"}'],
            env_dict={
                "COSYVOICE3_TRT": "0",
                "VLLM_CONFIGURE_LOGGING": "1",
                "VLLM_LOGGING_CONFIG_PATH": str(_LOG_CONFIG_PATH),
            },
        ),
        id="cosyvoice3_dspark",
    )
]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _SERVER_PARAMS, indirect=True)
def test_text_to_audio_talker_dspark(cosyvoice3_dspark_draft, omni_server, online_client) -> None:
    offset = _LOG.stat().st_size if _LOG.exists() else 0
    prompts = (
        "Hello, this is a voice cloning test with English text.",
        "Today we are testing speculative speech generation with a fixed reference voice.",
    )
    for seed, text in enumerate(prompts, 1000):
        response = online_client.client.audio.speech.create(
            model=omni_server.model,
            input=text,
            voice=None,
            response_format="wav",
            extra_body={
                "ref_audio": get_asset_path("cosyvoice3/zero_shot_prompt.wav", as_data_url=True),
                "ref_text": "希望你以后能够做的比我还好呦。",
                "seed": seed,
            },
            timeout=300.0,
        )
        with wave.open(io.BytesIO(response.read())) as audio:
            assert audio.getnchannels() == 1
            assert audio.getframerate() == 24000
            assert audio.getnframes() > 0
            assert audio.readframes(audio.getnframes())
    with _LOG.open("rb") as stream:
        stream.seek(offset)
        verification_log = stream.read().decode("utf-8")
    assert "CosyVoice3 Talker is verifying draft codes" in verification_log, (
        "Audio was returned without evidence that the Talker verified draft codes"
    )
