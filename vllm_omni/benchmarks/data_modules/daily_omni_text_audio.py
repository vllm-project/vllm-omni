"""Daily-Omni: optional consistency check between text stream and generated speech.

The benchmark MCQ accuracy uses ``generated_text`` only. When the omni server also
streams ``modality=audio`` (TTS), this module can transcribe the concatenated WAV
with Whisper and compare the inferred option letter to the one parsed from text.

Requires ``openai-whisper`` (``pip install openai-whisper``). Enable via env
``DAILY_OMNI_TEXT_AUDIO_CONSISTENCY=1`` or CLI ``--daily-omni-text-audio-consistency``.

Whisper model name defaults to ``tiny`` (override with ``DAILY_OMNI_WHISPER_MODEL``).
"""

from __future__ import annotations

import logging
import os
import re
import threading
from typing import Any

from vllm_omni.benchmarks.data_modules.daily_omni_dataset import DailyOmniSampleRequest
from vllm_omni.benchmarks.data_modules.daily_omni_eval import extract_predicted_choice

logger = logging.getLogger(__name__)

_whisper_model = None
_whisper_model_name: str | None = None
_whisper_lock = threading.Lock()


def env_text_audio_check_enabled() -> bool:
    return os.environ.get("DAILY_OMNI_TEXT_AUDIO_CONSISTENCY", "").lower() in (
        "1",
        "true",
        "yes",
    )


def extract_choice_from_asr_transcript(transcript: str) -> str | None:
    """Parse A–D from ASR text; extends :func:`extract_predicted_choice` with spoken Chinese phrases."""
    c = extract_predicted_choice(transcript)
    if c:
        return c
    t = transcript or ""
    for pat in (
        r"(?i)选项\s*([ABCD])\b",
        r"(?i)选\s*([ABCD])\b",
        r"(?i)答案\s*是\s*([ABCD])\b",
        r"(?i)答案\s*([ABCD])\b",
    ):
        m = re.search(pat, t)
        if m:
            return m.group(1).upper()
    return None


def _get_whisper_model(model_name: str):
    global _whisper_model, _whisper_model_name
    with _whisper_lock:
        if _whisper_model is None or _whisper_model_name != model_name:
            import whisper

            logger.warning(
                "Loading Whisper model %r for Daily-Omni text/audio consistency (one-time)...",
                model_name,
            )
            _whisper_model = whisper.load_model(model_name)
            _whisper_model_name = model_name
        return _whisper_model


def transcribe_wav_bytes(
    wav_bytes: bytes,
    *,
    language: str | None = None,
    model_name: str | None = None,
) -> tuple[str | None, str | None]:
    """Transcribe WAV bytes. Returns ``(transcript, error)`` — one of them is set.

    Args:
        wav_bytes: RIFF WAV file bytes.
        language: Optional Whisper language code (e.g. ``en``, ``zh``); improves accuracy/latency.
        model_name: Override model id; else ``DAILY_OMNI_WHISPER_MODEL`` or ``tiny``.
    """
    if not wav_bytes:
        return None, "empty_wav"
    if model_name is None or not str(model_name).strip():
        model_name = os.environ.get("DAILY_OMNI_WHISPER_MODEL") or "tiny"
    model_name = str(model_name).strip() or "tiny"
    path: str | None = None
    try:
        import tempfile

        model = _get_whisper_model(model_name)
        fd, path = tempfile.mkstemp(suffix=".wav")
        with os.fdopen(fd, "wb") as fp:
            fp.write(wav_bytes)
        kwargs: dict = {}
        if language:
            kwargs["language"] = language
        result = model.transcribe(path, **kwargs)
        text = (result.get("text") or "").strip()
        return (text if text else None), None
    except ImportError:
        return None, "openai-whisper is not installed (pip install openai-whisper)"
    except Exception as e:
        return None, str(e)[:500]
    finally:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass


def compute_daily_omni_text_audio_consistency_metrics(
    input_requests: list[Any],
    outputs: list[Any],
    *,
    include_per_item: bool = False,
) -> dict[str, Any] | None:
    """Compare option letter from ``generated_text`` vs Whisper transcript of output audio.

    Only considers requests where ``outputs[i]`` has ``generated_audio_wav_bytes`` set
    (populated by the omni benchmark when TA check is enabled).
    """
    if not input_requests or len(input_requests) != len(outputs):
        return None
    if not all(isinstance(r, DailyOmniSampleRequest) for r in input_requests):
        return None

    ta_no_wav = 0
    ta_asr_failed = 0
    ta_text_unparsed = 0
    ta_audio_unparsed = 0
    ta_consistent = 0
    ta_mismatch = 0
    ta_both_parsed = 0
    items: list[dict[str, Any]] = []

    for req, out in zip(input_requests, outputs, strict=True):
        assert isinstance(req, DailyOmniSampleRequest)
        rid = req.request_id
        if not getattr(out, "success", False):
            if include_per_item:
                items.append(
                    {
                        "request_id": rid,
                        "skipped": True,
                        "reason": "request_not_success",
                    }
                )
            continue

        wav = getattr(out, "generated_audio_wav_bytes", None)
        if not wav:
            ta_no_wav += 1
            if include_per_item:
                items.append(
                    {
                        "request_id": rid,
                        "skipped": False,
                        "reason": "no_output_audio",
                        "text_choice": extract_predicted_choice(getattr(out, "generated_text", "") or ""),
                    }
                )
            continue

        transcript, asr_err = transcribe_wav_bytes(wav)
        if asr_err:
            ta_asr_failed += 1
            if include_per_item:
                items.append(
                    {
                        "request_id": rid,
                        "asr_error": asr_err,
                        "text_choice": extract_predicted_choice(getattr(out, "generated_text", "") or ""),
                    }
                )
            continue

        text_choice = extract_predicted_choice(getattr(out, "generated_text", "") or "")
        audio_choice = extract_choice_from_asr_transcript(transcript or "")

        if text_choice is None:
            ta_text_unparsed += 1
        if audio_choice is None:
            ta_audio_unparsed += 1

        if text_choice is not None and audio_choice is not None:
            ta_both_parsed += 1
            if text_choice == audio_choice:
                ta_consistent += 1
            else:
                ta_mismatch += 1

        if include_per_item:
            consistent: bool | None
            if text_choice is None or audio_choice is None:
                consistent = None
            else:
                consistent = text_choice == audio_choice
            items.append(
                {
                    "request_id": rid,
                    "text_choice": text_choice,
                    "audio_choice": audio_choice,
                    "asr_transcript": (transcript or "")[:500],
                    "text_audio_consistent": consistent,
                }
            )

    comparable = ta_consistent + ta_mismatch
    rate = (ta_consistent / comparable) if comparable else None

    out: dict[str, Any] = {
        "daily_omni_ta_enabled": True,
        "daily_omni_ta_no_output_audio": ta_no_wav,
        "daily_omni_ta_asr_failed": ta_asr_failed,
        "daily_omni_ta_text_unparsed": ta_text_unparsed,
        "daily_omni_ta_audio_unparsed": ta_audio_unparsed,
        "daily_omni_ta_both_parsed": ta_both_parsed,
        "daily_omni_ta_consistent": ta_consistent,
        "daily_omni_ta_mismatch": ta_mismatch,
        "daily_omni_ta_consistency_rate": rate,
    }
    if include_per_item:
        out["daily_omni_ta_items"] = items
    return out


def print_daily_omni_text_audio_summary(metrics: dict[str, Any]) -> None:
    if not metrics.get("daily_omni_ta_enabled"):
        return
    print("{s:{c}^{n}}".format(s=" Daily-Omni text vs audio (ASR) ", n=50, c="="))
    print("{:<40} {:<10}".format("No output audio captured:", metrics.get("daily_omni_ta_no_output_audio", 0)))
    print("{:<40} {:<10}".format("ASR failed:", metrics.get("daily_omni_ta_asr_failed", 0)))
    print("{:<40} {:<10}".format("Both text+audio letter parsed:", metrics.get("daily_omni_ta_both_parsed", 0)))
    print("{:<40} {:<10}".format("Consistent (same letter):", metrics.get("daily_omni_ta_consistent", 0)))
    print("{:<40} {:<10}".format("Mismatch:", metrics.get("daily_omni_ta_mismatch", 0)))
    r = metrics.get("daily_omni_ta_consistency_rate")
    if r is not None:
        print("{:<40} {:<10.4f}".format("Consistency rate (of both parsed):", r))
    print(
        "{:<40} {:<10}".format(
            "Text unparsed (among w/ audio):",
            metrics.get("daily_omni_ta_text_unparsed", 0),
        )
    )
    print(
        "{:<40} {:<10}".format(
            "Audio unparsed (among w/ audio):",
            metrics.get("daily_omni_ta_audio_unparsed", 0),
        )
    )
