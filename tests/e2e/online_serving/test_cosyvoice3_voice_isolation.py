# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""
Cross-request voice isolation check for CosyVoice3 voice cloning (#8555).

Under concurrency a voice-cloning request can come back in the voice of another
request in flight at the same time (#8235, fixed by #8224). This test sends 32
seeded requests round-robin over 3 reference voices and asserts that every output
is closest to its own reference.

* Two independent speaker embedders score each output against all references:
  CosyVoice3's CAM++ (``campplus.onnx`` from the model snapshot) and
  ``microsoft/wavlm-base-plus-sv``.
* An output shorter than 0.5 s or silent is not scored and fails as a content problem.
* A leak is an output where both embedders agree on the same wrong voice. Zero leaks
  are allowed on the full output. The first 2 s is checked too, but the model alone
  sometimes opens a request in the wrong voice (nothing in flight), so at concurrency 8
  a wrong first 2 s counts only if the same request at concurrency 1 was right.
  The concurrency-1 controls assert the full output only and print the first-2 s labels.
* Speech content is checked by the ASR test. The speaker tests do not look at it.

Two ways of naming the voice are tested, each on its own: registered (uploaded with
``POST /v1/audio/voices`` and requested by name) and inline (``ref_audio`` /
``ref_text`` in the request body). Both run at concurrency 8 on one async-chunk server.

Controls run the same requests at concurrency 1, in async-chunk mode and with
``--no-async-chunk``, to show the scorer is clean when nothing is in flight. The swap
test feeds the reference clips back as outputs with two labels swapped, to show the
check can fail. It starts no server.

Voices: clone_2 (qwen3_tts), jiayan_zh (glm_tts) and indextts2. They are far apart under
both embedders; cosyvoice3/zero_shot_prompt.wav is not used because its outputs sit too
close to clone_2 for a reliable first-2 s check.

Reproduce::

    pytest -s -v tests/e2e/online_serving/test_cosyvoice3_voice_isolation.py \\
        --run-level full_model
"""

from __future__ import annotations

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.speaker_similarity import (
    LABEL_CORRECT,
    LABEL_WRONG_BOTH_AGREE,
    WAVLM_SV_MODEL,
    CampPlusEmbedder,
    WavLMSVEmbedder,
    score,
)
from tests.helpers.stage_config import get_deploy_config_path
from tests.helpers.voice_isolation import VoiceIsolationSuite, server_mode

pytestmark = [
    pytest.mark.full_model,
    pytest.mark.tts,
]

MODEL = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"

SUITE = VoiceIsolationSuite(
    model=MODEL,
    # Reference voices: (asset under tests/assets, transcript). Transcripts come from
    # the tests that already use each clip.
    voices={
        "clone_2": (
            "qwen3_tts/clone_2.wav",
            "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you.",
        ),
        "jiayan_zh": ("glm_tts/jiayan_zh.wav", "他当时还跟线下其他的站姐吵架，然后，打架进局子了。"),
        "indextts2": ("indextts2/ref_audio.wav", "翻译翻译，什么叫惊喜。"),
    },
    # English targets of clearly different lengths.
    texts=[
        "Thank you so much for the nice gift you sent me last week.",
        "The weather is lovely today, with warm sunshine and a gentle breeze, so we decided to walk in the park.",
        "Please remember to bring your notebook and a pen to the meeting tomorrow morning.",
        "After the long winter, the farmers were glad to see the first green shoots appear in the fields, "
        "and the children ran outside to play until the sun went down behind the hills.",
    ],
)
CONCURRENCY = SUITE.concurrency
FIRST_WINDOW_S = SUITE.first_window_s


def _server_params(*extra_args: str):
    return OmniServerParams(
        model=MODEL,
        stage_config_path=get_deploy_config_path("cosyvoice3.yaml"),
        server_args=["--trust-remote-code", *extra_args],
    )


ASYNC_CHUNK = pytest.param(_server_params(), id="async_chunk")
NO_ASYNC_CHUNK = pytest.param(_server_params("--no-async-chunk"), id="no_async_chunk")


@pytest.fixture(scope="module")
def scorers():
    """Embedders and reference embeddings. Both embedders run on CPU, after the ``generated`` server has stopped."""
    from huggingface_hub import snapshot_download

    return SUITE.build_scorers(
        [
            CampPlusEmbedder(snapshot_download(MODEL, allow_patterns=["campplus.onnx"])),
            WavLMSVEmbedder(WAVLM_SV_MODEL, device="cpu"),
        ]
    )


@pytest.fixture(
    scope="module",
    params=[ASYNC_CHUNK, NO_ASYNC_CHUNK],
)
def generated(request, run_level) -> str:
    """Start one server for this mode, generate every run of the mode, then shut it down.

    In async-chunk mode the registered-voice batch (concurrency 8) comes first, then the
    inline batch (8), then one request at a time (inline, then registered). The
    ``--no-async-chunk`` mode only runs the inline control. Returns the mode id.
    """
    if server_mode(request.param.server_args) == "no_async_chunk":
        plan = [("inline", 1)]
    else:
        plan = [("registered", CONCURRENCY), ("inline", CONCURRENCY), ("inline", 1), ("registered", 1)]
    return SUITE.generate(request, run_level, plan)


@hardware_test(res={"cuda": ["L4", "B200"]}, num_cards=1)
@pytest.mark.parametrize("generated", [ASYNC_CHUNK], indirect=True)
def test_voice_isolation_concurrent_registered(generated, scorers) -> None:
    """
    Concurrent requests for registered voices must each come back in their own voice.
    Deploy Setting: cosyvoice3.yaml, async_chunk on (default)
    Input Modal: text + voice name (voices uploaded via POST /v1/audio/voices)
    Output Modal: audio
    Input Setting: 32 seeded requests, round-robin over 3 voices, concurrency 8
    Datasets: tests/assets reference clips
    """
    run = SUITE.get_run(generated, "registered", CONCURRENCY, scorers)
    SUITE.assert_no_leak("concurrent_registered", run, scorers, gate_first_window_on_c1=True)


@hardware_test(res={"cuda": ["L4", "B200"]}, num_cards=1)
@pytest.mark.parametrize("generated", [ASYNC_CHUNK], indirect=True)
def test_voice_isolation_concurrent(generated, scorers) -> None:
    """
    Concurrent requests with an inline reference must each come back in their own reference voice.
    Deploy Setting: cosyvoice3.yaml, async_chunk on (default)
    Input Modal: text + ref_audio + ref_text
    Output Modal: audio
    Input Setting: 32 seeded requests, round-robin over 3 voices, concurrency 8
    Datasets: tests/assets reference clips
    """
    run = SUITE.get_run(generated, "inline", CONCURRENCY, scorers)
    SUITE.assert_no_leak("concurrent", run, scorers, gate_first_window_on_c1=True)


@hardware_test(res={"cuda": ["L4", "B200"]}, num_cards=1)
@pytest.mark.parametrize("generated", [ASYNC_CHUNK, NO_ASYNC_CHUNK], indirect=True)
def test_voice_isolation_control_c1(generated, scorers) -> None:
    """
    Control: the same requests one at a time. Nothing is in flight, so the full outputs must be clean.
    Deploy Setting: cosyvoice3.yaml, async_chunk on and ``--no-async-chunk``
    Input Modal: text + ref_audio + ref_text
    Output Modal: audio
    Input Setting: 32 seeded requests, round-robin over 3 voices, concurrency 1
    Datasets: tests/assets reference clips
    """
    run = SUITE.get_run(generated, "inline", 1, scorers)
    SUITE.assert_no_leak(f"control_c1_{generated}", run, scorers, gate_first_window_on_c1=False)


@hardware_test(res={"cuda": ["L4", "B200"]}, num_cards=1)
@pytest.mark.parametrize("generated", [ASYNC_CHUNK], indirect=True)
def test_voice_isolation_control_c1_registered(generated, scorers) -> None:
    """
    Control for registered voices: the same requests one at a time (async-chunk only).
    Deploy Setting: cosyvoice3.yaml, async_chunk on (default)
    Input Modal: text + voice name
    Output Modal: audio
    Input Setting: 32 seeded requests, round-robin over 3 voices, concurrency 1
    Datasets: tests/assets reference clips
    """
    run = SUITE.get_run(generated, "registered", 1, scorers)
    SUITE.assert_no_leak(f"control_c1_registered_{generated}", run, scorers, gate_first_window_on_c1=False)


@hardware_test(res={"cuda": ["L4", "B200"]}, num_cards=1)
def test_swapped_labels_fail_with_real_embedders(scorers) -> None:
    """
    The scorer must flag a swap: reference clips as outputs, two labels swapped.
    Both swaps must be wrong_both_agree and the rest correct, on the full clip and on its first 2 s.
    No server is started.
    Input Setting: the 3 reference clips, labels of two of them swapped
    Datasets: tests/assets reference clips
    """
    refs, embedders = scorers["refs"], scorers["embedders"]
    names = list(refs)
    # request index -> voice label the output of ``names[index]`` is (wrongly) sent under
    swapped = {1: "indextts2", 2: "jiayan_zh"}
    outputs = [(swapped.get(i, n), refs[n]) for i, n in enumerate(names)]
    for window in (None, FIRST_WINDOW_S):
        results = score(outputs, refs, embedders, window=window, reference_embeddings=scorers["ref_emb"])
        labels = {r.idx: r.label for r in results}
        assert labels == {0: LABEL_CORRECT, 1: LABEL_WRONG_BOTH_AGREE, 2: LABEL_WRONG_BOTH_AGREE}, (
            window,
            [r.to_dict() for r in results],
        )
        assert results[1].wrong_voice == "jiayan_zh" and results[2].wrong_voice == "indextts2"


@hardware_test(res={"cuda": ["L4", "B200"]}, num_cards=1)
def test_voice_isolation_asr(monkeypatch) -> None:
    """
    Speech content of every generated output, as its own failure after the speaker tests.
    A failure here is a content problem, not a voice leak.
    The servers have exited by now (the ``generated`` fixture stops them), so Whisper can use the GPU;
    on CPU its threads are capped so it does not oversubscribe a large host.
    Input Setting: the outputs kept by the speaker tests of this module
    """
    SUITE.run_asr_check(monkeypatch)
