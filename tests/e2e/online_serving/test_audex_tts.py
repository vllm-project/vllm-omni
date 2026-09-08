# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E online tests for Audex (Nemotron-Labs-Audex-2B) TTS.

Covers /v1/audio/speech non-streaming and audio-byte streaming, plus the
Audex request policies: single built-in voice (missing/empty/"default" only),
no reference audio, and cfg_scale reserved at 1.0.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

pytestmark = [pytest.mark.slow, pytest.mark.tts]

MODEL = "nvidia/Nemotron-Labs-Audex-2B"
PROMPT = "The weather is so good, and I want to enjoy the beautiful morning in the park."

audex_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("audex_tts.yaml"),
            server_args=["--trust-remote-code"],
        ),
        id="audex",
    )
]
# 30B-A3B variant: opt-in (pulls ~60 GB); enable with AUDEX_E2E_30B=1.
_AUDEX_30B_MODEL = os.environ.get("VLLM_OMNI_AUDEX_30B_MODEL_DIR") or "nvidia/Nemotron-Labs-Audex-30B-A3B"
if os.environ.get("AUDEX_E2E_30B") == "1":
    audex_server_params.append(
        pytest.param(
            OmniServerParams(
                model=_AUDEX_30B_MODEL,
                stage_config_path=get_deploy_config_path("audex_tts_30b.yaml"),
                server_args=["--trust-remote-code"],
            ),
            id="online_tts_30b",
        )
    )


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", audex_server_params, indirect=True)
def test_audex_tts_nonstream(omni_server, openai_client) -> None:
    """Plain English TTS, non-streaming WAV response."""
    request_config = {
        "model": omni_server.model,
        "input": PROMPT,
        "stream": False,
        "response_format": "wav",
    }
    openai_client.send_audio_speech_request(request_config)


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", audex_server_params, indirect=True)
def test_audex_tts_stream_audio(omni_server, openai_client) -> None:
    """Plain English TTS, raw audio-byte streaming."""
    request_config = {
        "model": omni_server.model,
        "input": PROMPT,
        "stream": True,
        "stream_format": "audio",
        "response_format": "wav",
    }
    openai_client.send_audio_speech_request(request_config)


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", audex_server_params, indirect=True)
def test_audex_tts_max_new_tokens_caps_duration(omni_server, openai_client, run_level) -> None:
    """max_new_tokens must cap stage-0 generation (review P2 regression)."""
    import io
    import wave

    import requests

    if run_level not in {"advanced_model", "full_model"}:
        pytest.skip("duration cap needs real weights")

    url = f"{openai_client.base_url.rstrip('/')}/v1/audio/speech"
    resp = requests.post(
        url,
        json={
            "model": omni_server.model,
            "input": PROMPT,
            "response_format": "wav",
            "max_new_tokens": 25,
        },
    )
    assert resp.status_code == 200, resp.text
    with wave.open(io.BytesIO(resp.content)) as wav:
        duration_s = wav.getnframes() / wav.getframerate()
    # 25 codec frames at 50 fps = 0.5 s (+ small lookahead tail); the full
    # sentence would be ~3 s, so a capped result proves the override applied.
    assert duration_s <= 1.0, f"max_new_tokens ignored: duration={duration_s:.2f}s"


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", audex_server_params, indirect=True)
def test_audex_tts_request_policies(omni_server, openai_client) -> None:
    """Unsupported voice / CFG / empty input must be rejected with 400."""
    import requests

    url = f"{openai_client.base_url.rstrip('/')}/v1/audio/speech"

    ok = requests.post(url, json={"model": omni_server.model, "input": PROMPT, "voice": "default"})
    assert ok.status_code == 200, ok.text

    for bad_payload, needle in (
        ({"input": PROMPT, "voice": "alloy"}, "voice"),
        ({"input": "   "}, "non-empty"),
        ({"input": PROMPT, "extra_params": {"cfg_scale": 0.5}}, "cfg_scale"),
        ({"input": PROMPT, "extra_params": {"cfg_scale": 10.5}}, "cfg_scale"),
        ({"input": PROMPT, "extra_params": {"cfg_scale": "big"}}, "cfg_scale"),
        ({"input": PROMPT, "extra_params": {"cfg_role": "cond"}}, "cfg_role"),
        ({"input": PROMPT, "extra_params": {"cfg_pair_id": "x"}}, "cfg_pair_id"),
        ({"input": PROMPT, "ref_audio": "https://example.com/ref.wav"}, "reference audio"),
    ):
        resp = requests.post(url, json={"model": omni_server.model, **bad_payload})
        assert resp.status_code == 400, f"{bad_payload} -> {resp.status_code}: {resp.text[:200]}"
        assert needle in resp.text, f"{bad_payload} error missing {needle!r}: {resp.text[:200]}"


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", audex_server_params, indirect=True)
def test_audex_tts_cfg_nonstream(omni_server, openai_client) -> None:
    """Guided (cfg_scale=1.5) non-streaming request returns one WAV."""
    request_config = {
        "model": omni_server.model,
        "input": PROMPT,
        "stream": False,
        "response_format": "wav",
        "extra_params": {"cfg_scale": 1.5},
    }
    openai_client.send_audio_speech_request(request_config)


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", audex_server_params, indirect=True)
def test_audex_tts_cfg_stream_audio(omni_server, openai_client) -> None:
    """Guided (cfg_scale=1.5) streaming request yields a single audio stream."""
    request_config = {
        "model": omni_server.model,
        "input": PROMPT,
        "stream": True,
        "stream_format": "audio",
        "response_format": "wav",
        "extra_params": {"cfg_scale": 1.5},
    }
    openai_client.send_audio_speech_request(request_config)


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", audex_server_params, indirect=True)
def test_audex_tts_mixed_cfg_concurrency(omni_server, openai_client) -> None:
    """Mixed guided/unguided requests at concurrency 4 all complete with audio."""
    import concurrent.futures

    import requests

    url = f"{openai_client.base_url.rstrip('/')}/v1/audio/speech"
    payloads = [
        {"model": omni_server.model, "input": f"{PROMPT} Take {i}.", "response_format": "wav"}
        | ({"extra_params": {"cfg_scale": 1.5}} if i % 2 == 0 else {})
        for i in range(4)
    ]

    def _post(payload):
        return requests.post(url, json=payload, timeout=300)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(_post, payloads))

    for payload, resp in zip(payloads, responses):
        assert resp.status_code == 200, f"{payload} -> {resp.status_code}: {resp.text[:200]}"
        assert resp.content[:4] == b"RIFF", f"{payload}: not a WAV response"
        assert len(resp.content) > 44, f"{payload}: empty audio body"


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", audex_server_params, indirect=True)
def test_audex_tts_stream_abort_leaves_server_healthy(omni_server, openai_client) -> None:
    """Aborting a guided streaming request mid-flight must not wedge the server.

    The abort cascades to the CFG companion (cfg_companion_tracker); a
    follow-up guided request on the same server must still succeed.
    """
    import requests

    url = f"{openai_client.base_url.rstrip('/')}/v1/audio/speech"
    long_input = " ".join([PROMPT] * 4)

    with requests.post(
        url,
        json={
            "model": omni_server.model,
            "input": long_input,
            "stream": True,
            "stream_format": "audio",
            "response_format": "wav",
            "extra_params": {"cfg_scale": 1.5},
        },
        stream=True,
        timeout=300,
    ) as resp:
        assert resp.status_code == 200, resp.text
        # Read one chunk, then abort the connection mid-stream.
        next(resp.iter_content(chunk_size=4096))

    follow_up = requests.post(
        url,
        json={
            "model": omni_server.model,
            "input": PROMPT,
            "response_format": "wav",
            "extra_params": {"cfg_scale": 1.5},
        },
        timeout=300,
    )
    assert follow_up.status_code == 200, follow_up.text[:200]
    assert follow_up.content[:4] == b"RIFF" and len(follow_up.content) > 44
