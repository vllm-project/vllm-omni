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
* A leak is an output where both embedders agree on the same wrong voice. It is
  checked on the full output and on its first 2 s. Zero leaks are allowed.
* Speech content is checked separately by the ASR test, so a content failure is
  never reported as a voice leak.

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

import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from tests.helpers.assertions import assert_audio_speech_response
from tests.helpers.client import OmniResponse
from tests.helpers.mark import hardware_test
from tests.helpers.media import get_asset_path
from tests.helpers.runtime import OmniServerParams
from tests.helpers.speaker_similarity import (
    LABEL_CORRECT,
    LABEL_DISAGREE,
    LABEL_WRONG_BOTH_AGREE,
    WAVLM_SV_MODEL,
    CampPlusEmbedder,
    FailureReport,
    OutputScore,
    SpeakerEmbedder,
    WavLMSVEmbedder,
    embed_references,
    format_table,
    load_audio_16k,
    min_margins,
    reference_matrix,
    retain_failed_voice_isolation,
    score,
    wrong_both_agree,
)
from tests.helpers.stage_config import get_deploy_config_path

pytestmark = [
    pytest.mark.full_model,
    pytest.mark.tts,
]

MODEL = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"

# Reference voices: (asset under tests/assets, transcript). Transcripts come from
# the tests that already use each clip.
VOICES: dict[str, tuple[str, str]] = {
    "clone_2": (
        "qwen3_tts/clone_2.wav",
        "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you.",
    ),
    "jiayan_zh": ("glm_tts/jiayan_zh.wav", "他当时还跟线下其他的站姐吵架，然后，打架进局子了。"),
    "indextts2": ("indextts2/ref_audio.wav", "翻译翻译，什么叫惊喜。"),
}

# English targets of clearly different lengths. The text index is decorrelated
# from the voice index so requests finish at different steps and new requests
# are admitted into batches that are already running.
TEXTS = [
    "Thank you so much for the birthday gift you sent me last week.",
    "The weather is lovely today, with warm sunshine and a gentle breeze, so we decided to walk in the park.",
    "Please remember to bring your notebook and a pen to the meeting tomorrow morning.",
    "After the long winter, the farmers were glad to see the first green shoots appear in the fields, "
    "and the children ran outside to play until the sun went down behind the hills.",
]

N_REQUESTS = 32
BASE_SEED = 1
CONCURRENCY = 8
FIRST_WINDOW_S = 2.0
REQUEST_TIMEOUT_S = 600.0


def _server_params(*extra_args: str):
    return OmniServerParams(
        model=MODEL,
        stage_config_path=get_deploy_config_path("cosyvoice3.yaml"),
        server_args=["--trust-remote-code", *extra_args],
    )


ASYNC_CHUNK = pytest.param(_server_params(), id="async_chunk")
NO_ASYNC_CHUNK = pytest.param(_server_params("--no-async-chunk"), id="no_async_chunk")


@dataclass
class RunResult:
    mode: str
    concurrency: int
    requests: list[dict[str, Any]]
    audio: list[bytes]
    wall_s: float
    full: list[OutputScore]
    first: list[OutputScore]
    source: str = "inline"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, text=True
        ).strip()
    except Exception:
        return "unknown"


def plan_requests(n: int = N_REQUESTS, base_seed: int = BASE_SEED) -> list[dict[str, Any]]:
    names = list(VOICES)
    k = len(names)
    return [
        {
            "idx": i,
            "voice": names[i % k],
            "text_idx": (i // k) % len(TEXTS),
            "text": TEXTS[(i // k) % len(TEXTS)],
            "seed": base_seed * 1000 + i,
        }
        for i in range(n)
    ]


@pytest.fixture(scope="module")
def reference_audio_urls() -> dict[str, str]:
    return {name: get_asset_path(rel, as_data_url=True) for name, (rel, _) in VOICES.items()}


@pytest.fixture(scope="module")
def scorers():
    """Embedders and reference embeddings. Both embedders run on CPU, after the ``generated`` server has stopped."""
    from huggingface_hub import snapshot_download

    embedders: list[SpeakerEmbedder] = [
        CampPlusEmbedder(snapshot_download(MODEL, allow_patterns=["campplus.onnx"])),
        WavLMSVEmbedder(WAVLM_SV_MODEL, device="cpu"),
    ]
    refs = {name: load_audio_16k(get_asset_path(rel)) for name, (rel, _) in VOICES.items()}
    return {
        "embedders": embedders,
        "refs": refs,
        "ref_emb": embed_references(refs, embedders),
        "ref_matrix": reference_matrix(refs, embedders),
    }


# Outputs per (server mode, voice source, concurrency), filled by the ``generated``
# fixture while its server is up, then scored by the speaker tests and checked by
# the ASR test. ``source`` is "registered" (voice uploaded via /v1/audio/voices and
# requested by name) or "inline" (ref_audio / ref_text in each request).
_GENERATED: dict[tuple[str, str, int], dict[str, Any]] = {}
# Scored runs, so the speaker tests of a mode score each run once.
_RUNS: dict[tuple[str, str, int], RunResult] = {}

REGISTERED_PREFIX = "voice_isolation_"
CONSENT_ID = "voice-isolation-test"


def _registered_name(voice: str) -> str:
    return f"{REGISTERED_PREFIX}{voice}"


def _register_voices(base_url: str) -> None:
    """Upload the reference voices the way a client registers a voice (audio_sample, name, consent, ref_text)."""
    import httpx

    for voice, (rel, ref_text) in VOICES.items():
        path = get_asset_path(rel)
        resp = httpx.post(
            f"{base_url}/v1/audio/voices",
            files={"audio_sample": (path.name, path.read_bytes(), "audio/wav")},
            data={"name": _registered_name(voice), "consent": CONSENT_ID, "ref_text": ref_text},
            timeout=120.0,
        )
        assert resp.status_code == 200 and resp.json().get("success"), f"voice upload {voice} failed: {resp.text[:300]}"


def _unregister_voices(base_url: str) -> None:
    """Best effort: do not leave uploaded voices behind on the runner."""
    import httpx

    for voice in VOICES:
        try:
            httpx.delete(f"{base_url}/v1/audio/voices/{_registered_name(voice)}", timeout=30.0)
        except Exception as exc:
            print(f"could not delete registered voice {voice}: {exc}")


def _send(client, model: str, req: dict[str, Any], urls: dict[str, str], source: str) -> bytes:
    if source == "registered":
        voice, extra = _registered_name(req["voice"]), {"seed": req["seed"]}
    else:
        voice = None
        extra = {"ref_audio": urls[req["voice"]], "ref_text": VOICES[req["voice"]][1], "seed": req["seed"]}
    resp = client.client.audio.speech.create(
        model=model,
        input=req["text"],
        voice=voice,
        response_format="wav",
        extra_body=extra,
        timeout=REQUEST_TIMEOUT_S,
    )
    return resp.read()


def _generate(client, model: str, concurrency: int, urls: dict[str, str], source: str) -> dict[str, Any]:
    requests = plan_requests()
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_send, client, model, r, urls, source) for r in requests]
        audio = [f.result() for f in futures]
    return {"requests": requests, "audio": audio, "wall_s": time.perf_counter() - t0}


@pytest.fixture(
    scope="module",
    params=[ASYNC_CHUNK, NO_ASYNC_CHUNK],
)
def generated(request, run_level, reference_audio_urls) -> str:
    """Start one server for this mode, generate every run of the mode, then shut it down.

    In async-chunk mode the registered-voice batch (concurrency 8) comes first, then the
    inline batch (8), then one request at a time (inline, then registered). The
    ``--no-async-chunk`` mode only runs the inline control. The server is stopped before
    this fixture returns, so scoring and the ASR check run with the GPU free (Whisper then
    uses the GPU instead of falling back to CPU next to the server). A batch that raises is
    stored as its error and fails only its own test; the later batches still run. Returns
    the mode id.
    """
    from tests.helpers.client import OnlineOmniClient
    from tests.helpers.fixtures.runtime import omni_fixture_lock
    from tests.helpers.runtime import iter_omni_server

    mode = "no_async_chunk" if "--no-async-chunk" in (request.param.server_args or []) else "async_chunk"
    plan = (
        [("registered", CONCURRENCY), ("inline", CONCURRENCY), ("inline", 1), ("registered", 1)]
        if mode == "async_chunk"
        else [("inline", 1)]
    )
    gen = iter_omni_server(request, run_level, omni_fixture_lock)
    server = next(gen)
    base_url = f"http://{server.host}:{server.port}"
    registered = False
    try:
        client = OnlineOmniClient(
            host=server.host, port=server.port, api_key="EMPTY", run_level=run_level, log_stats=server.log_stats
        )
        for source, c in plan:
            try:
                if source == "registered" and not registered:
                    _register_voices(base_url)
                    registered = True
                _GENERATED[(mode, source, c)] = _generate(client, server.model, c, reference_audio_urls, source)
            except Exception as exc:
                _GENERATED[(mode, source, c)] = {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        if registered:
            _unregister_voices(base_url)
        gen.close()  # stops the server and releases its GPU memory
    return mode


def _get_run(mode: str, source: str, concurrency: int, scorers) -> RunResult:
    key = (mode, source, concurrency)
    if key in _RUNS:
        return _RUNS[key]
    gen = _GENERATED[key]
    if "error" in gen:
        pytest.fail(f"generation failed for mode={mode} voices={source} concurrency={concurrency}: {gen['error']}")
    requests, audio, wall = gen["requests"], gen["audio"], gen["wall_s"]

    outputs = [(r["voice"], a) for r, a in zip(requests, audio)]
    kwargs = {"reference_embeddings": scorers["ref_emb"]}
    full = score(outputs, scorers["refs"], scorers["embedders"], window=None, **kwargs)
    first = score(outputs, scorers["refs"], scorers["embedders"], window=FIRST_WINDOW_S, **kwargs)

    print(
        f"\n=== voice isolation: mode={mode} voices={source} concurrency={concurrency} "
        f"n={len(requests)} wall={wall:.1f}s ==="
    )
    print(format_table(full, first, [len(r["text"]) for r in requests]))
    for label, results in (("full", full), ("first_2s", first)):
        counts: dict[str, int] = {}
        for r in results:
            counts[r.label] = counts.get(r.label, 0) + 1
        alone = {n: sum(r.argmax[n] != r.voice for r in results) for n in results[0].margin}
        print(
            f"{label}: labels={counts} disagree={counts.get(LABEL_DISAGREE, 0)} "
            f"wrong_both_agree={counts.get(LABEL_WRONG_BOTH_AGREE, 0)} "
            f"wrong_by_one_embedder={alone} min_margin={ {k: round(v, 4) for k, v in min_margins(results).items()} }"
        )
    short = [r.idx for r in first if r.window_short]
    if short:
        print(f"first_2s: outputs shorter than {FIRST_WINDOW_S}s, scored whole: {short}")

    run = RunResult(mode, concurrency, requests, audio, wall, full, first, source)
    _RUNS[key] = run
    return run


def _assert_no_leak(test: str, run: RunResult, scorers) -> None:
    leaks_full = wrong_both_agree(run.full)
    leaks_first = wrong_both_agree(run.first)
    if leaks_full or leaks_first:
        retain_failed_voice_isolation(
            FailureReport(
                test=test,
                voices={n: rel for n, (rel, _) in VOICES.items()},
                requests=run.requests,
                full=run.full,
                first=run.first,
                audio=run.audio,
                reference_matrix=scorers["ref_matrix"],
                extra={"git_sha": _git_sha(), "mode": run.mode, "voices": run.source, "concurrency": run.concurrency},
            )
        )

    def describe(results: list[OutputScore]) -> str:
        return ", ".join(f"req {r.idx} ({r.voice} -> {r.wrong_voice})" for r in results)

    assert not leaks_full, (
        f"voice leak on full output: {len(leaks_full)}/{len(run.full)} outputs have the wrong voice "
        f"under both embedders: {describe(leaks_full)}"
    )
    assert not leaks_first, (
        f"voice leak in the first {FIRST_WINDOW_S}s: {len(leaks_first)}/{len(run.first)} outputs have the "
        f"wrong voice under both embedders: {describe(leaks_first)}"
    )


def _asr_failures(mode: str, source: str, concurrency: int, gen: dict[str, Any]) -> list[str]:
    failures = []
    for req, audio in zip(gen["requests"], gen["audio"]):
        response = OmniResponse(success=True, audio_bytes=audio, audio_format="audio/wav")
        # Every text is English. Left on auto-detect, Whisper labels English spoken in a
        # Chinese reference voice as zh or ko and returns an unrelated transcript.
        config = {"response_format": "wav", "input": req["text"], "transcript_language": "en"}
        try:
            assert_audio_speech_response(response, config, run_level="full_model")
        except AssertionError as exc:
            failures.append(f"req {req['idx']} ({req['voice']}, mode={mode}, voices={source}, c={concurrency}): {exc}")
    return failures


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
    run = _get_run(generated, "registered", CONCURRENCY, scorers)
    _assert_no_leak("concurrent_registered", run, scorers)


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
    run = _get_run(generated, "inline", CONCURRENCY, scorers)
    _assert_no_leak("concurrent", run, scorers)


@hardware_test(res={"cuda": ["L4", "B200"]}, num_cards=1)
@pytest.mark.parametrize("generated", [ASYNC_CHUNK, NO_ASYNC_CHUNK], indirect=True)
def test_voice_isolation_control_c1(generated, scorers) -> None:
    """
    Control: the same requests one at a time. Nothing is in flight, so the scorer must be clean.
    Deploy Setting: cosyvoice3.yaml, async_chunk on and ``--no-async-chunk``
    Input Modal: text + ref_audio + ref_text
    Output Modal: audio
    Input Setting: 32 seeded requests, round-robin over 3 voices, concurrency 1
    Datasets: tests/assets reference clips
    """
    run = _get_run(generated, "inline", 1, scorers)
    _assert_no_leak(f"control_c1_{generated}", run, scorers)


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
    run = _get_run(generated, "registered", 1, scorers)
    _assert_no_leak(f"control_c1_registered_{generated}", run, scorers)


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


def _wait_for_free_vram(min_free_gib: float = 16.0, timeout_s: float = 120.0) -> None:
    """Give the exited servers a moment to hand their GPU memory back before ASR picks a device."""
    from vllm_omni.platforms import current_omni_platform

    if not current_omni_platform.is_available():
        return
    device = current_omni_platform.get_torch_device(0)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if current_omni_platform.get_free_memory(device) / 1024**3 >= min_free_gib:
            return
        time.sleep(2)


@hardware_test(res={"cuda": ["L4", "B200"]}, num_cards=1)
def test_voice_isolation_asr(monkeypatch) -> None:
    """
    Speech content of every generated output, as its own failure after the speaker tests.
    A failure here is a content problem, not a voice leak.
    The servers have exited by now (the ``generated`` fixture stops them), so Whisper can use the GPU;
    on CPU its threads are capped so it does not oversubscribe a large host.
    Input Setting: the outputs kept by the speaker tests of this module
    """
    from tests.helpers.media import release_audio_transcriber

    if not _GENERATED:
        pytest.skip("no generated outputs: run together with the speaker tests")
    release_audio_transcriber()
    threads = max(1, min(16, (os.cpu_count() or 2) // 2))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(var, str(threads))
    _wait_for_free_vram()
    failures: list[str] = []
    t0 = time.perf_counter()
    try:
        for (mode, source, c), gen in sorted(_GENERATED.items()):
            if "error" not in gen:  # already failed by its speaker test
                failures += _asr_failures(mode, source, c, gen)
    finally:
        release_audio_transcriber()
    n = sum(len(g["audio"]) for g in _GENERATED.values() if "error" not in g)
    print(f"ASR check: {n} outputs in {time.perf_counter() - t0:.0f}s ({threads} CPU threads if on CPU)")
    assert not failures, "ASR content check failed (not a voice leak):\n" + "\n".join(failures)
