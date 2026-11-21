# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""
Cross-request voice isolation check for Qwen3-TTS Base voice cloning (#8555).

Same check as ``test_cosyvoice3_voice_isolation.py`` (#4370 was the Qwen3-TTS case):
32 seeded requests round-robin over 3 reference voices, each output scored by
CAM++ and ``microsoft/wavlm-base-plus-sv``, and a leak is an output both embedders
put in the same wrong voice. See ``tests/helpers/voice_isolation.py`` for the rules.

Three ways of naming the voice are tested, each on its own at concurrency 8 and as a
concurrency-1 control, on one async-chunk server:

* registered: uploaded with ``POST /v1/audio/voices`` and requested by name;
* inline: ``ref_audio`` and ``ref_text`` in the request body (in-context cloning);
* inline_xvec: ``ref_audio`` with ``x_vector_only_mode`` (speaker embedding only).

The CAM++ model is the ``campplus.onnx`` that ships with CosyVoice3; only that file
is downloaded. The scorer itself is checked by the swap test in the CosyVoice3 module.

Reproduce::

    pytest -s -v tests/e2e/online_serving/test_qwen3_tts_voice_isolation.py \\
        --run-level full_model
"""

from __future__ import annotations

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.speaker_similarity import WAVLM_SV_MODEL, CampPlusEmbedder, WavLMSVEmbedder
from tests.helpers.stage_config import get_deploy_config_path
from tests.helpers.voice_isolation import SOURCES, VoiceIsolationSuite

pytestmark = [
    pytest.mark.full_model,
    pytest.mark.tts,
]

MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
CAMPPLUS_REPO = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"

SUITE = VoiceIsolationSuite(
    model=MODEL,
    voices={
        "clone_2": (
            "qwen3_tts/clone_2.wav",
            "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you.",
        ),
        "jiayan_zh": ("glm_tts/jiayan_zh.wav", "他当时还跟线下其他的站姐吵架，然后，打架进局子了。"),
        "indextts2": ("indextts2/ref_audio.wav", "翻译翻译，什么叫惊喜。"),
    },
    texts=[
        "Thank you so much for the nice gift you sent me last week.",
        "The weather is lovely today, with warm sunshine and a gentle breeze, so we decided to walk in the park.",
        "Please remember to bring your notebook and a pen to the meeting tomorrow morning.",
        "After the long winter, the farmers were glad to see the first green shoots appear in the fields, "
        "and the children ran outside to play until the sun went down behind the hills.",
    ],
)
CONCURRENCY = SUITE.concurrency

ASYNC_CHUNK = pytest.param(
    OmniServerParams(
        model=MODEL,
        stage_config_path=get_deploy_config_path("qwen3_tts.yaml"),
        server_args=["--trust-remote-code"],
    ),
    id="async_chunk",
)


@pytest.fixture(scope="module")
def scorers():
    """Embedders and reference embeddings. Both embedders run on CPU, after the ``generated`` server has stopped."""
    from huggingface_hub import snapshot_download

    return SUITE.build_scorers(
        [
            CampPlusEmbedder(snapshot_download(CAMPPLUS_REPO, allow_patterns=["campplus.onnx"])),
            WavLMSVEmbedder(WAVLM_SV_MODEL, device="cpu"),
        ]
    )


@pytest.fixture(scope="module", params=[ASYNC_CHUNK])
def generated(request, run_level) -> str:
    """One server: every voice source at concurrency 8, then each one at concurrency 1."""
    plan = [(source, CONCURRENCY) for source in SOURCES] + [(source, 1) for source in SOURCES]
    return SUITE.generate(request, run_level, plan)


@hardware_test(res={"cuda": ["L4", "B200"]}, num_cards=1)
@pytest.mark.parametrize("source", SOURCES)
def test_voice_isolation_concurrent(generated, scorers, source) -> None:
    """
    Concurrent requests must each come back in their own reference voice.
    Deploy Setting: qwen3_tts.yaml, async_chunk on (default)
    Input Modal: text + voice (registered name, ref_audio + ref_text, or ref_audio + x_vector_only_mode)
    Output Modal: audio
    Input Setting: 32 seeded requests, round-robin over 3 voices, concurrency 8
    Datasets: tests/assets reference clips
    """
    run = SUITE.get_run(generated, source, CONCURRENCY, scorers)
    SUITE.assert_no_leak(f"qwen3_tts_concurrent_{source}", run, scorers, gate_first_window_on_c1=True)


@hardware_test(res={"cuda": ["L4", "B200"]}, num_cards=1)
@pytest.mark.parametrize("source", SOURCES)
def test_voice_isolation_control_c1(generated, scorers, source) -> None:
    """
    Control: the same requests one at a time. Nothing is in flight, so the full outputs must be clean.
    Deploy Setting: qwen3_tts.yaml, async_chunk on (default)
    Input Modal: text + voice, as in the concurrent test
    Output Modal: audio
    Input Setting: 32 seeded requests, round-robin over 3 voices, concurrency 1
    Datasets: tests/assets reference clips
    """
    run = SUITE.get_run(generated, source, 1, scorers)
    SUITE.assert_no_leak(f"qwen3_tts_control_c1_{source}", run, scorers, gate_first_window_on_c1=False)


@hardware_test(res={"cuda": ["L4", "B200"]}, num_cards=1)
def test_voice_isolation_asr(monkeypatch) -> None:
    """
    Speech content of every generated output, as its own failure after the speaker tests.
    A failure here is a content problem, not a voice leak.
    Input Setting: the outputs kept by the speaker tests of this module
    """
    SUITE.run_asr_check(monkeypatch)
