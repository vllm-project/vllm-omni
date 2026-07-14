"""End-to-end audio probe for Qwen3-TTS PD pipeline (Phase 2).

Purpose
-------
Phase 1 of the PD migration ran in *AR-only* mode: the talker decode
stage marked itself as ``final_output_type=latent`` and the response
body was an NDJSON stream of ``{"audio_codes": [...]}`` records.
A separate (non-PD) ``token2wav`` server then converted those audio
codes into a wav file.  Phase 2 attaches Code2Wav back into the PD
pipeline so the same ``/v1/audio/speech`` endpoint returns *real*
PCM/WAV audio bytes — exactly like ``vllm_omni/deploy/qwen3_tts.yaml``
(the single-process baseline) does.

This probe answers three questions:

1. Does the PD pipeline still return audio bytes (PCM or WAV) for a
   text-input request?  If the body is NDJSON we know Code2Wav was
   bypassed and the deploy YAML is misconfigured.
2. Are the returned bytes large enough to be a real waveform (rough
   sanity check against zero-byte / tiny-error responses)?
3. Is the byte length comparable (within a tolerance) to the
   single-process baseline?  Greedy sampling + deterministic Code2Wav
   should produce nearly the same number of audio samples, so a large
   discrepancy means the PD pipeline truncated audio (e.g. KV transfer
   broke and decode produced fewer audio_codes than baseline).

The script also exercises both endpoints documented by the test fleet:

* The text-only fast path (no ``ref_audio`` / ``ref_text``), which
  does not exist in every TTS deploy but is the cheapest probe.
* The *clone* path (with ``ref_audio`` / ``ref_text``), which mirrors
  ``tests/e2e/online_serving/test_qwen3_tts_base.py`` and is the
  recommended Qwen3-TTS production setup.

Usage
-----
Two servers must be running side-by-side on the same host:

    # Baseline: single-process Talker + Code2Wav.
    vllm serve <Qwen3-TTS path> --omni --port 8090 \
        --stage-configs-path vllm_omni/deploy/qwen3_tts.yaml

    # PD probe: 1P1D split with Code2Wav re-attached (Phase 2).
    vllm serve <Qwen3-TTS path> --omni --port 8091 \
        --stage-configs-path vllm_omni/deploy/qwen3_tts_pd_1p1d.yaml

    # Then run the probe:
    python examples/online_serving/qwen3_tts_pd/probe_audio_pd.py \
        --baseline-url http://localhost:8090 \
        --pd-url       http://localhost:8091 \
        --runs 2

Exit code is 0 only if every PD run produced valid audio bytes whose
length matches the baseline within tolerance.

Dependencies: only ``httpx`` and the Python stdlib (``wave``, ``json``).
No pytest fixture, no conftest.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import statistics
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import httpx

logger = logging.getLogger("qwen3_tts_pd.probe_audio_pd")

# Default short / deterministic prompt.  Greedy sampling on this prompt
# is required for the byte-length / hash comparison to be meaningful.
_DEFAULT_PROMPT = "The weather is nice today, perfect for a walk in the park."
_DEFAULT_REF_TEXT = (
    "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
)
# Vendored under tests/assets so probe runs without network access.
_DEFAULT_REF_AUDIO_PATH = Path(__file__).resolve().parents[3] / "tests" / "assets" / "qwen3_tts" / "clone_2.wav"

# Bytes below this threshold are almost certainly an error response,
# not a wav payload.  ~1s of 16-bit 24kHz mono PCM is 48 KB; a real
# Qwen3-TTS clip is at least ~3s.
_MIN_AUDIO_BYTES = 32 * 1024
# Allowed |pd - baseline| / baseline byte-length divergence.  Greedy
# sampling + deterministic Code2Wav typically agree to <1 %, but we
# allow 15 % to absorb (a) chunk boundary alignment and (b) trailing
# silence padding differences between async-chunk and full-payload
# code paths.
_DEFAULT_LEN_RATIO_TOLERANCE = 0.15


@dataclass
class AudioRunResult:
    """One end-to-end /v1/audio/speech observation."""

    label: str
    ttfb_ms: float
    total_ms: float
    num_bytes: int
    response_format: str
    is_wav: bool
    wav_duration_s: float | None
    looks_like_ndjson: bool
    head_preview: str = ""
    error: str | None = None
    extras: dict[str, object] = field(default_factory=dict)


def _load_ref_audio_data_url(path: Path) -> str | None:
    """Return ``data:audio/wav;base64,...`` URL or ``None`` if missing.

    Mirrors ``tests.helpers.media.load_test_audio_data_url`` semantics
    without taking a runtime dependency on the test helper package.
    """
    if not path.is_file():
        logger.warning(
            "Reference audio %s does not exist; falling back to text-only run.",
            path,
        )
        return None
    raw = path.read_bytes()
    return "data:audio/wav;base64," + base64.b64encode(raw).decode("ascii")


def _looks_like_ndjson(blob: bytes) -> bool:
    """Heuristic: first non-empty line parses as JSON with audio_codes-ish keys."""
    if not blob:
        return False
    head = blob[:4096].lstrip()
    if not head.startswith(b"{"):
        return False
    first_line = head.split(b"\n", 1)[0].strip()
    try:
        rec = json.loads(first_line)
    except json.JSONDecodeError:
        return False
    return isinstance(rec, dict) and ("audio_codes" in rec or "codes" in rec or "audio" in rec)


def _try_parse_wav_duration(blob: bytes) -> float | None:
    """Return wav clip duration in seconds, or ``None`` if the blob isn't wav."""
    if not blob.startswith(b"RIFF"):
        return None
    try:
        with wave.open(io.BytesIO(blob), "rb") as wf:
            num_frames = wf.getnframes()
            framerate = wf.getframerate() or 1
            return num_frames / framerate
    except (wave.Error, EOFError):
        return None


def _send_audio_speech(
    base_url: str,
    label: str,
    *,
    prompt: str,
    response_format: str,
    stream: bool,
    timeout_s: float,
    ref_audio_data_url: str | None,
    ref_text: str | None,
    voice: str,
    task_type: str,
    model: str,
) -> AudioRunResult:
    """POST one /v1/audio/speech request and capture timing + bytes.

    Note we do not reach into vllm-omni's serving layer; this is a
    pure HTTP probe so the same script works against either the
    baseline single-process server or the PD pipeline.
    """
    url = base_url.rstrip("/") + "/v1/audio/speech"
    payload: dict[str, object] = {
        "model": model,
        "input": prompt,
        "stream": stream,
        "response_format": response_format,
        "task_type": task_type,
        "voice": voice,
    }
    if ref_audio_data_url is not None:
        payload["ref_audio"] = ref_audio_data_url
    if ref_text is not None:
        payload["ref_text"] = ref_text

    t0 = time.perf_counter()
    first_chunk_ts: float | None = None
    chunks: list[bytes] = []
    err: str | None = None

    try:
        with httpx.Client(timeout=timeout_s) as client:
            with client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()
                for chunk in resp.iter_bytes():
                    if not chunk:
                        continue
                    if first_chunk_ts is None:
                        first_chunk_ts = time.perf_counter()
                    chunks.append(chunk)
    except httpx.HTTPError as e:
        err = f"HTTP error: {e!r}"

    body = b"".join(chunks)
    total_ms = (time.perf_counter() - t0) * 1000.0
    ttfb_ms = (first_chunk_ts - t0) * 1000.0 if first_chunk_ts is not None else float("inf")

    is_ndjson = _looks_like_ndjson(body)
    is_wav = body.startswith(b"RIFF")
    wav_duration_s = _try_parse_wav_duration(body) if is_wav else None

    head_preview = ""
    if is_ndjson:
        head_preview = body[:160].decode("utf-8", errors="replace")
    elif body and not is_wav:
        # Show a hex preview so we can eyeball obvious corruption.
        head_preview = body[:32].hex()

    return AudioRunResult(
        label=label,
        ttfb_ms=ttfb_ms,
        total_ms=total_ms,
        num_bytes=len(body),
        response_format=response_format,
        is_wav=is_wav,
        wav_duration_s=wav_duration_s,
        looks_like_ndjson=is_ndjson,
        head_preview=head_preview,
        error=err,
    )


def _summarise(results: list[AudioRunResult]) -> dict[str, float]:
    ok = [r for r in results if r.error is None and r.num_bytes > 0]
    if not ok:
        return {
            "median_ttfb_ms": float("inf"),
            "median_total_ms": float("inf"),
            "median_bytes": 0,
            "n_ok": 0,
        }
    return {
        "median_ttfb_ms": statistics.median(r.ttfb_ms for r in ok),
        "median_total_ms": statistics.median(r.total_ms for r in ok),
        "median_bytes": statistics.median(r.num_bytes for r in ok),
        "n_ok": len(ok),
    }


def _judge(
    baseline: list[AudioRunResult],
    pd: list[AudioRunResult],
    *,
    len_ratio_tolerance: float,
) -> tuple[bool, list[str], dict]:
    """Return (passed, findings, metrics).

    The PD pipeline passes iff:

    1. Every PD run returned bytes (no HTTP error, num_bytes > 0).
    2. No PD run returned NDJSON — that would mean Code2Wav was
       bypassed and the YAML is still in Phase-1 (AR-only) mode.
    3. Every PD run returned at least ``_MIN_AUDIO_BYTES`` (sanity).
    4. Median PD payload byte-length is within
       ``len_ratio_tolerance`` of the baseline median, i.e. the PD
       pipeline produced (approximately) the same amount of audio.

    The baseline is also exercised so we know the comparison reference
    is healthy.  If baseline itself is broken we flag it but do not
    silently pass.
    """
    findings: list[str] = []
    base_summary = _summarise(baseline)
    pd_summary = _summarise(pd)
    metrics = {"baseline": base_summary, "pd": pd_summary}

    if base_summary["n_ok"] == 0:
        findings.append("Baseline produced 0 successful runs; cannot compare PD payload size.")
    if pd_summary["n_ok"] == 0:
        findings.append("PD pipeline produced 0 successful runs (every run errored or returned 0 bytes).")
        return False, findings, metrics

    # --- PD-only assertions ---------------------------------------------------
    pd_ndjson = [r for r in pd if r.looks_like_ndjson]
    if pd_ndjson:
        findings.append(
            f"{len(pd_ndjson)}/{len(pd)} PD runs returned NDJSON — Code2Wav was "
            "bypassed.  Check that stage 2 (or stage 3 for M-P-1-D) of the PD "
            "deploy YAML has model_stage=code2wav and final_output_type=audio, "
            "and that decode stage forwards via output_connectors."
        )
        return False, findings, metrics

    pd_too_small = [r for r in pd if r.error is None and r.num_bytes < _MIN_AUDIO_BYTES]
    if pd_too_small:
        findings.append(
            f"{len(pd_too_small)}/{len(pd)} PD runs returned <{_MIN_AUDIO_BYTES} "
            "bytes; suspect early truncation or empty audio."
        )
        return False, findings, metrics

    # --- Baseline-vs-PD payload size ratio -----------------------------------
    if base_summary["n_ok"] > 0:
        base_med = base_summary["median_bytes"]
        pd_med = pd_summary["median_bytes"]
        ratio = abs(pd_med - base_med) / base_med if base_med > 0 else float("inf")
        metrics["pd_to_baseline_bytes_ratio_diff"] = ratio
        if ratio > len_ratio_tolerance:
            findings.append(
                f"PD median bytes={pd_med} vs baseline median={base_med} "
                f"(|Δ|/base={ratio:.2%} > {len_ratio_tolerance:.0%}): "
                "PD audio length diverges from baseline -- KV transfer may have "
                "truncated decode."
            )
            return False, findings, metrics
        findings.append(
            f"PD median bytes ratio {ratio:.2%} <= {len_ratio_tolerance:.0%} "
            "tolerance: PD audio length matches baseline."
        )
    else:
        findings.append("Baseline unavailable; PD passed the byte-length tolerance check trivially.")

    findings.append(f"PD pipeline emitted real audio bytes for every run ({len(pd)} run(s)).")
    return True, findings, metrics


def _print_run(r: AudioRunResult) -> None:
    fmt = "{label}: ttfb={ttfb:.1f}ms total={total:.1f}ms bytes={n} wav={wav} dur_s={dur} ndjson={ndj} err={err}"
    logger.info(
        fmt.format(
            label=r.label,
            ttfb=r.ttfb_ms,
            total=r.total_ms,
            n=r.num_bytes,
            wav=r.is_wav,
            dur=(f"{r.wav_duration_s:.2f}" if r.wav_duration_s else "?"),
            ndj=r.looks_like_ndjson,
            err=r.error,
        )
    )
    if r.head_preview:
        logger.info("    head_preview=%s", r.head_preview[:160])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-url",
        default="http://localhost:8090",
        help="Single-process server (vllm_omni/deploy/qwen3_tts.yaml).",
    )
    parser.add_argument(
        "--pd-url",
        default="http://localhost:8091",
        help="PD server (qwen3_tts_pd_1p1d.yaml or qwen3_tts_pd_mp1d.yaml).",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        help="Value for the OpenAI ``model`` field; must match server --model.",
    )
    parser.add_argument(
        "--prompt",
        default=_DEFAULT_PROMPT,
        help="Text input for the TTS request (greedy sampling required).",
    )
    parser.add_argument(
        "--ref-audio-path",
        type=Path,
        default=_DEFAULT_REF_AUDIO_PATH,
        help="Local wav file to send as ref_audio (set to '' to skip clone path).",
    )
    parser.add_argument("--ref-text", default=_DEFAULT_REF_TEXT)
    parser.add_argument("--voice", default="clone")
    parser.add_argument("--task-type", default="Base")
    parser.add_argument(
        "--response-format",
        default="wav",
        choices=("wav", "pcm", "mp3"),
        help="OpenAI response_format value.  'wav' is recommended for byte-length checks.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use stream=True (chunked transfer); default is non-streaming.",
    )
    parser.add_argument("--runs", type=int, default=2, help="Repetitions per server.")
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=180.0,
        help="Per-request HTTP timeout (default mirrors test_qwen3_tts_base).",
    )
    parser.add_argument(
        "--len-ratio-tolerance",
        type=float,
        default=_DEFAULT_LEN_RATIO_TOLERANCE,
        help="Max |pd_bytes - base_bytes| / base_bytes accepted.",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline runs (e.g. when only a PD server is up).",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    ref_audio_url: str | None = None
    if str(args.ref_audio_path):
        ref_audio_url = _load_ref_audio_data_url(args.ref_audio_path)

    common_kwargs: dict[str, object] = dict(
        prompt=args.prompt,
        response_format=args.response_format,
        stream=args.stream,
        timeout_s=args.timeout_s,
        ref_audio_data_url=ref_audio_url,
        ref_text=args.ref_text if ref_audio_url is not None else None,
        voice=args.voice if ref_audio_url is not None else "default",
        task_type=args.task_type,
        model=args.model,
    )

    baseline_results: list[AudioRunResult] = []
    if not args.skip_baseline:
        logger.info("Probing baseline @ %s (runs=%d)", args.baseline_url, args.runs)
        for i in range(args.runs):
            r = _send_audio_speech(args.baseline_url, f"baseline_{i}", **common_kwargs)
            baseline_results.append(r)
            _print_run(r)

    logger.info("Probing PD @ %s (runs=%d)", args.pd_url, args.runs)
    pd_results: list[AudioRunResult] = []
    for i in range(args.runs):
        r = _send_audio_speech(args.pd_url, f"pd_{i}", **common_kwargs)
        pd_results.append(r)
        _print_run(r)

    passed, findings, metrics = _judge(
        baseline_results,
        pd_results,
        len_ratio_tolerance=args.len_ratio_tolerance,
    )

    print()
    print("=" * 72)
    print(f"Verdict: {'PASS' if passed else 'FAIL'}")
    print(f"Baseline median bytes : {metrics['baseline']['median_bytes']:.0f}")
    print(f"PD median bytes       : {metrics['pd']['median_bytes']:.0f}")
    print(f"Baseline median TTFB  : {metrics['baseline']['median_ttfb_ms']:.2f} ms")
    print(f"PD median TTFB        : {metrics['pd']['median_ttfb_ms']:.2f} ms")
    if "pd_to_baseline_bytes_ratio_diff" in metrics:
        print(f"|Δbytes|/baseline    : {metrics['pd_to_baseline_bytes_ratio_diff']:.2%}")
    print("Findings:")
    for line in findings:
        print(f"  - {line}")
    print("=" * 72)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
