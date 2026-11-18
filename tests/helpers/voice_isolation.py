# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared driver for the cross-request voice isolation e2e tests (#8555).

A voice-cloning model under concurrency can answer a request in the voice of
another request in flight at the same time. Each model's test module builds a
:class:`VoiceIsolationSuite` with its model, reference voices and target texts;
the suite generates seeded requests against a running server, scores every output
with :mod:`tests.helpers.speaker_similarity`, and asserts that no output is in
another request's voice.

Voices are named in one of three ways (``source``):

* ``registered``: uploaded with ``POST /v1/audio/voices`` and requested by name.
* ``inline``: ``ref_audio`` and ``ref_text`` in each request body.
* ``inline_xvec``: ``ref_audio`` with ``x_vector_only_mode`` (models that
  condition on a speaker embedding alone, such as Qwen3-TTS Base).

A leak is an output where every embedder picks the same wrong voice. The full
output must have none. The first ``first_window_s`` seconds are checked too,
but a model can open a request in the wrong voice with nothing in flight, so at
concurrency > 1 a wrong opening counts only if the same request at concurrency 1
opened in the right voice. Speech content is checked separately by
:meth:`VoiceIsolationSuite.run_asr_check`.
"""

from __future__ import annotations

import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest

from tests.helpers.media import get_asset_path
from tests.helpers.speaker_similarity import (
    LABEL_CORRECT,
    LABEL_DISAGREE,
    LABEL_UNSCORABLE,
    LABEL_WRONG_BOTH_AGREE,
    FailureReport,
    OutputScore,
    SpeakerEmbedder,
    embed_references,
    format_table,
    load_audio_16k,
    min_margins,
    reference_matrix,
    retain_failed_voice_isolation,
    score,
    wrong_both_agree,
)

SOURCES = ("registered", "inline", "inline_xvec")


@dataclass
class RunResult:
    mode: str
    concurrency: int
    requests: list[dict[str, Any]]
    audio: list[bytes]
    wall_s: float
    full: list[OutputScore]
    first: list[OutputScore]
    source: str


def server_mode(server_args: list[str] | None) -> str:
    return "no_async_chunk" if "--no-async-chunk" in (server_args or []) else "async_chunk"


def gate_first_window(
    mismatched: list[OutputScore], baseline_first: list[OutputScore]
) -> tuple[list[OutputScore], list[tuple[OutputScore, OutputScore]]]:
    """Split first-window mismatches into leaks and openings the model gets wrong on its own.

    A mismatch is a leak only if the same request at concurrency 1 was ``correct`` in
    its first window. Returns ``(leaks, excused)``, with each excused output paired
    with its concurrency-1 score.
    """
    leaks, excused = [], []
    for r in mismatched:
        base = baseline_first[r.idx]
        if base.label == LABEL_CORRECT:
            leaks.append(r)
        else:
            excused.append((r, base))
    return leaks, excused


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, text=True
        ).strip()
    except Exception:
        return "unknown"


def _describe(results: list[OutputScore]) -> str:
    return ", ".join(f"req {r.idx} ({r.voice} -> {r.wrong_voice})" for r in results)


@dataclass
class VoiceIsolationSuite:
    """Requests, generation, scoring and assertions for one model's voice isolation test.

    ``voices`` maps a voice name to ``(asset under tests/assets, transcript)``.
    Generated batches are kept per ``(mode, source, concurrency)`` for the life of
    the test module, so the speaker tests and the ASR test share them.
    """

    model: str
    voices: dict[str, tuple[str, str]]
    texts: list[str]
    n_requests: int = 32
    base_seed: int = 1
    concurrency: int = 8
    first_window_s: float = 2.0
    request_timeout_s: float = 600.0
    transcript_language: str = "en"
    registered_prefix: str = "voice_isolation_"
    consent_id: str = "voice-isolation-test"
    generated: dict[tuple[str, str, int], dict[str, Any]] = field(default_factory=dict, init=False, repr=False)
    runs: dict[tuple[str, str, int], RunResult] = field(default_factory=dict, init=False, repr=False)

    @property
    def window(self) -> str:
        return f"first {self.first_window_s:g}s"

    # ---- requests -------------------------------------------------------------

    def plan_requests(self) -> list[dict[str, Any]]:
        """Round-robin over the voices. The text index is decorrelated from the voice
        index so requests finish at different steps and new requests are admitted into
        batches that are already running."""
        names = list(self.voices)
        k = len(names)
        return [
            {
                "idx": i,
                "voice": names[i % k],
                "text_idx": (i // k) % len(self.texts),
                "text": self.texts[(i // k) % len(self.texts)],
                "seed": self.base_seed * 1000 + i,
            }
            for i in range(self.n_requests)
        ]

    def reference_audio_urls(self) -> dict[str, str]:
        return {name: get_asset_path(rel, as_data_url=True) for name, (rel, _) in self.voices.items()}

    def registered_name(self, voice: str) -> str:
        return f"{self.registered_prefix}{voice}"

    def request_kwargs(self, req: dict[str, Any], urls: dict[str, str], source: str) -> tuple[str | None, dict]:
        """``(voice, extra_body)`` for one request named the way ``source`` says."""
        if source == "registered":
            return self.registered_name(req["voice"]), {"seed": req["seed"]}
        extra: dict[str, Any] = {"ref_audio": urls[req["voice"]], "seed": req["seed"]}
        if source == "inline":
            extra["ref_text"] = self.voices[req["voice"]][1]
        elif source == "inline_xvec":
            extra["x_vector_only_mode"] = True
        else:
            raise ValueError(f"unknown voice source {source!r}")
        return None, extra

    def register_voices(self, base_url: str) -> None:
        """Upload the reference voices the way a client registers a voice (audio_sample, name, consent, ref_text)."""
        import httpx

        for voice, (rel, ref_text) in self.voices.items():
            path = get_asset_path(rel)
            resp = httpx.post(
                f"{base_url}/v1/audio/voices",
                files={"audio_sample": (path.name, path.read_bytes(), "audio/wav")},
                data={"name": self.registered_name(voice), "consent": self.consent_id, "ref_text": ref_text},
                timeout=120.0,
            )
            assert resp.status_code == 200 and resp.json().get("success"), (
                f"voice upload {voice} failed: {resp.text[:300]}"
            )

    def _send(self, client, model: str, req: dict[str, Any], urls: dict[str, str], source: str) -> bytes:
        voice, extra = self.request_kwargs(req, urls, source)
        resp = client.client.audio.speech.create(
            model=model,
            input=req["text"],
            voice=voice,
            response_format="wav",
            extra_body=extra,
            timeout=self.request_timeout_s,
        )
        return resp.read()

    def _generate(self, client, model: str, concurrency: int, urls: dict[str, str], source: str) -> dict[str, Any]:
        requests = self.plan_requests()
        t0 = time.perf_counter()
        if concurrency == 1:
            # Stop at the first failed control request without queueing later requests.
            audio = [self._send(client, model, r, urls, source) for r in requests]
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [pool.submit(self._send, client, model, r, urls, source) for r in requests]
                audio = [f.result() for f in futures]
        return {"requests": requests, "audio": audio, "wall_s": time.perf_counter() - t0}

    def generate(self, request, run_level, plan: list[tuple[str, int]]) -> str:
        """Start one server for ``request.param``, generate every ``(source, concurrency)`` batch
        in ``plan`` in order, then shut the server down.

        Scoring and the ASR check then run with the GPU free (Whisper uses the GPU instead
        of falling back to CPU next to the server). A batch that raises is stored as its
        error and fails only its own test; the later batches still run. Returns the mode id.
        """
        from tests.helpers.client import OnlineOmniClient
        from tests.helpers.fixtures.runtime import omni_fixture_lock
        from tests.helpers.runtime import iter_omni_server

        mode = server_mode(request.param.server_args)
        urls = self.reference_audio_urls()
        with TemporaryDirectory(prefix="omni-voice-isolation-") as voice_dir, pytest.MonkeyPatch.context() as patch:
            params = request.param
            env = {**(params.env_dict or {}), "SPEAKER_SAMPLES_DIR": voice_dir}
            patch.setattr(request, "param", params._replace(env_dict=env))
            with closing(iter_omni_server(request, run_level, omni_fixture_lock)) as servers:
                server = next(servers)
                client = OnlineOmniClient(
                    host=server.host, port=server.port, api_key="EMPTY", run_level=run_level, log_stats=server.log_stats
                )
                with closing(client.client):
                    if mode == "no_async_chunk":
                        # The serial control reports the first failed attempt and can tear down its server.
                        client.client.max_retries = 0
                    registration_error = None
                    if any(source == "registered" for source, _ in plan):
                        try:
                            self.register_voices(client.base_url)
                        except Exception as exc:
                            registration_error = f"{type(exc).__name__}: {exc}"

                    for source, c in plan:
                        key = (mode, source, c)
                        if source == "registered" and registration_error is not None:
                            self.generated[key] = {"error": registration_error}
                            continue
                        try:
                            self.generated[key] = self._generate(client, server.model, c, urls, source)
                        except Exception as exc:
                            self.generated[key] = {"error": f"{type(exc).__name__}: {exc}"}
        return mode

    # ---- scoring --------------------------------------------------------------

    def build_scorers(self, embedders: list[SpeakerEmbedder]) -> dict[str, Any]:
        refs = {name: load_audio_16k(get_asset_path(rel)) for name, (rel, _) in self.voices.items()}
        return {
            "embedders": embedders,
            "refs": refs,
            "ref_emb": embed_references(refs, embedders),
            "ref_matrix": reference_matrix(refs, embedders),
        }

    def get_run(self, mode: str, source: str, concurrency: int, scorers) -> RunResult:
        key = (mode, source, concurrency)
        if key in self.runs:
            return self.runs[key]
        gen = self.generated[key]
        if "error" in gen:
            pytest.fail(f"generation failed for mode={mode} voices={source} concurrency={concurrency}: {gen['error']}")
        requests, audio, wall = gen["requests"], gen["audio"], gen["wall_s"]

        outputs = [(r["voice"], a) for r, a in zip(requests, audio)]
        kwargs = {"reference_embeddings": scorers["ref_emb"]}
        full = score(outputs, scorers["refs"], scorers["embedders"], window=None, **kwargs)
        first = score(outputs, scorers["refs"], scorers["embedders"], window=self.first_window_s, **kwargs)

        print(
            f"\n=== voice isolation: mode={mode} voices={source} concurrency={concurrency} "
            f"n={len(requests)} wall={wall:.1f}s ==="
        )
        print(format_table(full, first, [len(r["text"]) for r in requests]))
        for label, results in (("full", full), (self.window, first)):
            counts: dict[str, int] = {}
            for r in results:
                counts[r.label] = counts.get(r.label, 0) + 1
            alone = {n: sum(n in r.argmax and r.argmax[n] != r.voice for r in results) for n in results[0].margin}
            print(
                f"{label}: labels={counts} disagree={counts.get(LABEL_DISAGREE, 0)} "
                f"wrong_both_agree={counts.get(LABEL_WRONG_BOTH_AGREE, 0)} "
                f"wrong_by_one_embedder={alone} min_margin={ {k: round(v, 4) for k, v in min_margins(results).items()} }"
            )
        short = [r.idx for r in first if r.window_short]
        if short:
            print(f"{self.window}: outputs shorter than the window, scored whole: {short}")

        run = RunResult(mode, concurrency, requests, audio, wall, full, first, source)
        self.runs[key] = run
        return run

    def _baseline_run(self, run: RunResult, scorers) -> RunResult:
        """The concurrency-1 run of the same mode and voice source, to gate the first-window check."""
        gen = self.generated.get((run.mode, run.source, 1))
        if gen is None or "error" in gen:
            pytest.fail(
                f"cannot gate the {self.window} check: the concurrency-1 baseline for "
                f"mode={run.mode} voices={run.source} is unavailable "
                f"({'not generated' if gen is None else gen['error']})"
            )
        return self.get_run(run.mode, run.source, 1, scorers)

    def assert_no_leak(self, test: str, run: RunResult, scorers, *, gate_first_window_on_c1: bool) -> None:
        """Assert that no output of ``run`` is unscorable or in the wrong voice.

        The full output is always asserted. With ``gate_first_window_on_c1`` a first-window
        leak counts only if the same request at concurrency 1 had the right voice in its first
        window; otherwise it is printed as excused. Without it (the concurrency-1 controls) the
        first-window labels are only printed.
        """

        def retain() -> None:
            retain_failed_voice_isolation(
                FailureReport(
                    test=test,
                    voices={n: rel for n, (rel, _) in self.voices.items()},
                    requests=run.requests,
                    full=run.full,
                    first=run.first,
                    audio=run.audio,
                    reference_matrix=scorers["ref_matrix"],
                    extra={
                        "git_sha": _git_sha(),
                        "model": self.model,
                        "mode": run.mode,
                        "voice_source": run.source,
                        "concurrency": run.concurrency,
                    },
                )
            )

        window = self.window
        unscorable = [r for r in run.full if r.label == LABEL_UNSCORABLE]
        leaks_full = wrong_both_agree(run.full)
        if unscorable or leaks_full:
            retain()
        assert not unscorable, (
            f"output too short or silent; a content problem, not a voice leak: {len(unscorable)}/{len(run.full)} "
            f"outputs are unscorable: " + ", ".join(f"req {r.idx} ({r.voice}, {r.duration_s:.2f}s)" for r in unscorable)
        )
        assert not leaks_full, (
            f"voice leak on full output: {len(leaks_full)}/{len(run.full)} outputs have the wrong voice "
            f"under both embedders: {_describe(leaks_full)}"
        )

        mismatched = wrong_both_agree(run.first)
        if not gate_first_window_on_c1:
            if mismatched:
                print(f"{window} (not asserted at concurrency 1): wrong under both embedders: {_describe(mismatched)}")
            return
        baseline = self._baseline_run(run, scorers)
        leaks_first, excused = gate_first_window(mismatched, baseline.first)
        for r, base in excused:
            print(
                f"{window} excused: req {r.idx} ({r.voice} -> {r.wrong_voice}); the same request at concurrency 1 "
                f"is already {base.label} in its {window}, so this is the model's opening, not a leak"
            )
        if leaks_first:
            retain()
        assert not leaks_first, (
            f"voice leak in the {window}: {len(leaks_first)}/{len(run.first)} outputs have the "
            f"wrong voice under both embedders and were right at concurrency 1: {_describe(leaks_first)}"
        )

    # ---- speech content -------------------------------------------------------

    def _asr_failures(self, mode: str, source: str, concurrency: int, gen: dict[str, Any]) -> list[str]:
        from tests.helpers.assertions import assert_audio_speech_response
        from tests.helpers.client import OmniResponse

        failures = []
        for req, audio in zip(gen["requests"], gen["audio"]):
            response = OmniResponse(success=True, audio_bytes=audio, audio_format="audio/wav")
            # Left on auto-detect, Whisper labels English spoken in a Chinese reference
            # voice as zh or ko and returns an unrelated transcript.
            config = {
                "response_format": "wav",
                "input": req["text"],
                "transcript_language": self.transcript_language,
                "transcript_escalation_model": "large-v3",
            }
            try:
                assert_audio_speech_response(response, config, run_level="full_model")
            except AssertionError as exc:
                failures.append(
                    f"req {req['idx']} ({req['voice']}, mode={mode}, voices={source}, c={concurrency}): {exc}"
                )
        return failures

    def run_asr_check(self, monkeypatch) -> None:
        """Speech content of every generated output, as its own failure after the speaker tests.

        The servers have exited by now, so Whisper can use the GPU; on CPU its threads are
        capped so it does not oversubscribe a large host.
        """
        from tests.helpers.media import release_audio_transcriber

        if not self.generated:
            pytest.skip("no generated outputs: run together with the speaker tests")
        release_audio_transcriber()
        threads = max(1, min(16, (os.cpu_count() or 2) // 2))
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            monkeypatch.setenv(var, str(threads))
        _wait_for_free_vram()
        failures: list[str] = []
        t0 = time.perf_counter()
        try:
            for (mode, source, c), gen in sorted(self.generated.items()):
                if "error" not in gen:  # already failed by its speaker test
                    failures += self._asr_failures(mode, source, c, gen)
        finally:
            release_audio_transcriber()
        n = sum(len(g["audio"]) for g in self.generated.values() if "error" not in g)
        print(f"ASR check: {n} outputs in {time.perf_counter() - t0:.0f}s ({threads} CPU threads if on CPU)")
        assert not failures, "ASR content check failed (not a voice leak):\n" + "\n".join(failures)


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
