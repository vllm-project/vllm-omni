# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Guard against new hardcoded audio sample-rate defaults on generic paths.

Fails when a NEW rate literal appears under ``vllm_omni/entrypoints/`` or
``vllm_omni/engine/``: derive the rate from the model output / config, or
add it to the allowlist below consciously. The allowlist only shrinks.
"""

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Rate literals (Hz) that have no business hardcoded on generic paths.
_RATE_LITERALS = frozenset({8000, 16000, 22050, 24000, 32000, 44100, 48000})

# Integer literal with optional underscore separators (24_000).
_NUM_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:_\d{3})+|\d{4,6})(?![\w.])")

# Identifiers that mark a line as sample-rate-related (``asr``/``usr``
# excluded: the char before ``sr`` must be a word boundary or underscore).
_RATE_ID_RE = re.compile(r"sample[_-]?rate|sampling[_-]?rate|SAMPLE_RATE|rate_hz|RATE_HZ|['\"]sr['\"]|_sr\b|\bsr\b")

_SCAN_ROOTS = ("vllm_omni/entrypoints", "vllm_omni/engine")

# (repo-relative path, whitespace-normalized line) pairs known to carry a
# hardcoded rate. Grouped by reason; see each section.
_ALLOWLIST = {
    # --- Generic-path fallbacks: removed by the standardization PRs ---
    ("vllm_omni/entrypoints/openai/serving_chat.py", "sample_rate = 24000"),
    ("vllm_omni/entrypoints/openai/serving_chat.py", "sample_rate = 48000"),
    ("vllm_omni/entrypoints/openai/serving_speech.py", "sample_rate_val = 24000"),
    ("vllm_omni/entrypoints/openai/serving_speech.py", 'sr_raw = audio_output.get("sr", 24000)'),
    ("vllm_omni/entrypoints/openai/serving_speech_stream.py", "_PCM_SAMPLE_RATE = 24000"),
    (
        "vllm_omni/entrypoints/openai/serving_audio_generate.py",
        "default_sr = 44100 # Default sample rate for Stable Audio",
    ),
    ("vllm_omni/entrypoints/openai/video_api_utils.py", "DEFAULT_AUDIO_SAMPLE_RATE = 24_000"),
    (
        "vllm_omni/entrypoints/openai/video_stream_base.py",
        "def _pcm_to_wav_b64(pcm_data: bytes, sample_rate: int = 16000) -> str:",
    ),
    ("vllm_omni/entrypoints/openai/video_stream_base.py", "sample_rate=24000,"),
    ("vllm_omni/engine/orchestrator.py", "audio_sample_rate: int = 24000,"),
    (
        "vllm_omni/engine/duplex/session/runner.py",
        "sample_rate_hz = sr_raw if isinstance(sr_raw, int | float) else 16000",
    ),
    ("vllm_omni/entrypoints/openai/protocol/audio.py", "sample_rate: int = 24000"),
    # --- Protocol-native constants (duplex/realtime wire contract) ---
    ("vllm_omni/entrypoints/duplex/warmup.py", "_WARMUP_SAMPLE_RATE_HZ = 16000"),
    ("vllm_omni/entrypoints/duplex/warmup.py", "_DEFAULT_SILENCE_SAMPLE_RATE_HZ = 16000"),
    (
        "vllm_omni/entrypoints/duplex_omni.py",
        'await session.submit(AppendAudio(audio=pcm_bytes, format="pcm16", sample_rate_hz=16000))',
    ),
    ("vllm_omni/entrypoints/openai/realtime/connection.py", "SAMPLE_RATE_HZ = 24000"),
    ("vllm_omni/engine/duplex/plugin.py", "silence_continuation_sample_rate_hz: int = 16000"),
    (
        "vllm_omni/engine/duplex/session/control.py",
        'payload = dict(event, audio=segment.audio, format="pcm_f32le", sample_rate_hz=16000)',
    ),
    ("vllm_omni/engine/duplex/smart_turn.py", "SMART_TURN_SAMPLE_RATE_HZ = 16_000"),
    ("vllm_omni/engine/duplex/vad.py", "_SAMPLE_RATE_HZ = 16_000"),
    # --- Model-specific adapters (per-model capability contracts) ---
    (
        "vllm_omni/entrypoints/openai/tts_adapters/breeze_tts_2.py",
        "supported_output_sample_rates = frozenset({24000})",
    ),
    (
        "vllm_omni/entrypoints/openai/tts_adapters/qwen3_tts.py",
        "supported_output_sample_rates = frozenset({8000, 24000})",
    ),
    (
        "vllm_omni/entrypoints/openai/tts_adapters/ming3_tts.py",
        "extractor = SpeakerEmbeddingExtractor(model, target_sr=16000)",
    ),
    # --- Docstrings (documentation, not defaults) ---
    (
        "vllm_omni/entrypoints/openai/serving_speech.py",
        "- sample_rate: Target output sample rate (8000 or 24000 Hz)",
    ),
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_no_new_hardcoded_sample_rate_defaults() -> None:
    offenders = []
    for root in _SCAN_ROOTS:
        for path in sorted((_repo_root() / root).rglob("*.py")):
            rel = path.relative_to(_repo_root()).as_posix()
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                literals = {int(m.group(1).replace("_", "")) for m in _NUM_RE.finditer(stripped)}
                if not (literals & _RATE_LITERALS) or not _RATE_ID_RE.search(stripped):
                    continue
                normalized = " ".join(stripped.split())
                if (rel, normalized) not in _ALLOWLIST:
                    offenders.append(f"  {rel}:{lineno}: {stripped}")
    assert not offenders, (
        "New hardcoded audio sample-rate default(s) on a generic path:\n"
        + "\n".join(offenders)
        + "\n\nDerive the rate from the model output (vllm_omni.sample_rate.get_sampling_rate) "
        "or the model config instead. If the constant is genuinely protocol-native or "
        "model-specific, add it to _ALLOWLIST in tests/outputs/test_no_silent_sample_rate_defaults.py "
        "with a comment explaining why."
    )


def test_allowlist_has_no_stale_entries() -> None:
    """Every allowlisted line must still exist verbatim, minus whitespace."""
    root = _repo_root()
    stale = []
    for rel, normalized in sorted(_ALLOWLIST):
        path = root / rel
        if not path.exists():
            stale.append(f"  {rel}: file removed")
            continue
        lines = {" ".join(line.strip().split()) for line in path.read_text(encoding="utf-8").splitlines()}
        if normalized not in lines:
            stale.append(f"  {rel}: {normalized}")
    assert not stale, (
        "Stale allowlist entries (lines changed or removed — update the allowlist "
        "and re-check whether the constant is still needed):\n" + "\n".join(stale)
    )
