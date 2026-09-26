# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""E2E online tests for YuE2-3B text-to-music on ``/v1/audio/speech``.

Scenario: heavy-metal "Twinkle Twinkle Little Star" — Chinese lyrics plus a
metal style caption plus an ABC score of the melody (``extra_params.cot =
"melody"``). The golden score is a SheetSage2 transcription of the Wikimedia
piano rendition, reduced to a single voice (``tests/assets/yue2``).

Two programmatic content checks, both calibrated against a same-seed control
request with ``cot="off"`` (no ABC conditioning):

- Style: CLAP zero-shot probability of "heavy metal" must clear 0.9.
- Melody: the generated song is transcribed back with SheetSage2 and its
  pitch-class sequence is LCS-matched against the golden score (max over
  major/minor variants and 12 transpositions, tolerating pedal-tone
  insertions). SheetSage2's transcription of distorted metal vocals carries
  roughly ±0.05 noise, so the bar sits at 0.80; the ABC-conditioned request
  must also beat the control.

The content checks run in a dedicated verification interpreter because
SheetSage2 wants a dependency set (mir_eval, pretty_midi, numpy<2) that must
not leak into the serving environment. They are enabled by three env vars:

- ``YUE2_E2E_VERIFY_PYTHON``: python with sheetsage2 + CLAP deps installed
- ``YUE2_E2E_CLAP_DIR``: local laion/clap-htsat-unfused snapshot (~0.6 GB)
- ``YUE2_E2E_SHEETSAGE2_DIR``: local m-a-p/SheetSage2 snapshot (~0.25 GB,
  plus its MERT-v2 backbone ~2.5 GB from the HF cache)

Without them the test still asserts the structural floor (non-empty payload
above the byte floor) and passes on that alone, following the MiniMax Music 3
precedent.

Weekly rather than per-PR: each request is a 16 s song and the verification
checkpoints add ~5.5 GB of downloads. Validated on a single RTX 4090
(24 GB); the hardware mark targets an H100 because the CI fleet has no
consumer cards.
"""

import json
import os
import subprocess

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import get_asset_path
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

MODEL = "m-a-p/YuE2-3B"
DEFAULT_AUDIO_SPEECH_TIMEOUT_S = 900.0

# 400 frames = 16 s of song. The byte floor sits far below the observed
# payload (~3.0 MB of 48 kHz stereo 16-bit) to leave room for an early EOS.
_FRAMES = 400
_MIN_BYTES = 1_000_000

LYRICS = "一闪一闪亮晶晶\n满天都是小星星\n挂在天空放光明\n好像千万小眼睛"
CAPTION = (
    "Chinese heavy metal, distorted electric guitars, aggressive double-kick drums, powerful vocals, 140 BPM, key of C"
)
SEED = 831001

_VERIFY_PYTHON = os.environ.get("YUE2_E2E_VERIFY_PYTHON")
_CLAP_DIR = os.environ.get("YUE2_E2E_CLAP_DIR")
_SHEETSAGE2_DIR = os.environ.get("YUE2_E2E_SHEETSAGE2_DIR")
_CONTENT_CHECKS_ENABLED = all([_VERIFY_PYTHON, _CLAP_DIR, _SHEETSAGE2_DIR])
_VERIFY_HELPER = os.path.join(os.path.dirname(__file__), "yue2_metal_verify.py")


tts_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("yue2.yaml"),
            server_args=["--trust-remote-code", "--disable-log-stats"],
        ),
        id="yue2",
    )
]


@pytest.mark.slow
@pytest.mark.tts
@hardware_test(res={"cuda": "H100"}, num_cards={"cuda": 1})
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_yue2_metal_twinkle_001(omni_server, openai_client, tmp_path) -> None:
    """Heavy-metal Twinkle: ABC-conditioned request vs same-seed control.

    Deploy Setting: yue2.yaml
    Input Modal: text (lyrics) + text (style caption) + ABC score (extra_params)
    Output Modal: audio, 48 kHz stereo
    Input Setting: stream=False, seed pinned, 400 frames
    """
    golden_abc = get_asset_path("yue2/twinkle_golden.abc").read_text()

    def _request(extra_params: dict) -> bytes:
        responses = openai_client.send_audio_speech_request(
            {
                "model": omni_server.model,
                "input": LYRICS,
                "instructions": CAPTION,
                "seed": SEED,
                "max_new_tokens": _FRAMES,
                "stream": False,
                "response_format": "wav",
                "timeout": DEFAULT_AUDIO_SPEECH_TIMEOUT_S,
                "min_audio_bytes": _MIN_BYTES,
                # Sung metal vocals are not whisper-transcribable; skip the
                # full_model transcript check (our content checks do the work).
                "transcript_expected_text": "",
                "extra_params": extra_params,
            }
        )
        audio = responses[0].audio_bytes
        assert audio, "empty audio payload"
        return audio

    metal_path = tmp_path / "metal.wav"
    control_path = tmp_path / "control.wav"
    metal_path.write_bytes(_request({"cot": "melody", "abc": golden_abc}))
    control_path.write_bytes(_request({"cot": "off"}))

    # Without the verification assets the structural assertions above
    # (non-empty payload, byte floor) are the whole test — pass on them.
    if not _CONTENT_CHECKS_ENABLED:
        return
    assert _VERIFY_PYTHON is not None
    assert _CLAP_DIR is not None
    assert _SHEETSAGE2_DIR is not None

    proc = subprocess.run(
        [
            _VERIFY_PYTHON,
            _VERIFY_HELPER,
            "--clap-dir",
            _CLAP_DIR,
            "--sheetsage2-dir",
            _SHEETSAGE2_DIR,
            "--work-dir",
            str(tmp_path),
            str(metal_path),
            str(control_path),
        ],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert proc.returncode == 0, f"verification helper failed:\n{proc.stderr}"
    report = json.loads(proc.stdout.strip().splitlines()[-1])

    metal_prob = report["metal_probability"]
    assert metal_prob >= 0.9, f"style check: P(heavy metal)={metal_prob:.3f} < 0.9"

    melody_score = report["melody_score"]
    control_score = report["control_score"]
    assert melody_score >= 0.80, f"melody check: score={melody_score:.3f} < 0.80"
    assert melody_score > control_score, (
        f"calibration: ABC-conditioned {melody_score:.3f} not above control {control_score:.3f}"
    )
