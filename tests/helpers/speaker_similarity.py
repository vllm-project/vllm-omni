# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Speaker-similarity scoring for cross-request voice isolation checks.

Voice-cloning TTS models can answer a request in the voice of another request
that is in flight at the same time (#8235 for CosyVoice3, #4370 for Qwen3-TTS).
This module scores generated audio against the reference voices that were sent
with the requests, so a test can assert that every output is closest to its own
reference.

The scoring is model independent. An embedder is anything with a ``name`` and an
``embed(wav16)`` method that maps a mono 16 kHz float32 waveform to an
L2-normalised vector. ``WavLMSVEmbedder`` and ``CampPlusEmbedder`` are provided;
another model can plug in its own embedder without touching ``score``.

An output is labelled:

* ``correct``: every embedder picks the requested voice.
* ``wrong_both_agree``: every embedder picks the same wrong voice. This is the
  only failing class. One embedder alone can be fooled by a mostly non-speech
  output, so a leak needs agreement.
* ``disagree``: anything else. Logged, not failing.

``score(..., window=2.0)`` repeats the check on the first two seconds only, which
is where a leak that affects only the start of an utterance shows up.
"""

from __future__ import annotations

import io
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

SAMPLE_RATE = 16000
LABEL_CORRECT = "correct"
LABEL_WRONG_BOTH_AGREE = "wrong_both_agree"
LABEL_DISAGREE = "disagree"

WAVLM_SV_MODEL = "microsoft/wavlm-base-plus-sv"


class SpeakerEmbedder(Protocol):
    """Maps a mono 16 kHz float32 waveform to an L2-normalised embedding."""

    name: str

    def embed(self, wav16: np.ndarray) -> np.ndarray: ...


def load_audio_16k(
    src: bytes | bytearray | str | os.PathLike | np.ndarray, sample_rate: int | None = None
) -> np.ndarray:
    """Decode wav bytes or a file, or take a raw array, and return mono 16 kHz float32.

    Raw arrays need ``sample_rate``; decoded audio carries its own.
    """
    import soundfile as sf
    from scipy.signal import resample_poly

    if isinstance(src, np.ndarray):
        if sample_rate is None:
            raise ValueError("sample_rate is required for raw arrays")
        wav = src.astype(np.float32, copy=False)
        sr = int(sample_rate)
        if wav.ndim == 2:
            wav = wav.mean(axis=-1 if wav.shape[-1] <= 8 else 0)
    else:
        handle = io.BytesIO(src) if isinstance(src, bytes | bytearray) else str(src)
        wav, sr = sf.read(handle, dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        g = math.gcd(sr, SAMPLE_RATE)
        wav = resample_poly(wav, SAMPLE_RATE // g, sr // g).astype(np.float32)
    return np.ascontiguousarray(wav, dtype=np.float32)


def _unit(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64).reshape(-1)
    return (vec / max(np.linalg.norm(vec), 1e-12)).astype(np.float32)


class WavLMSVEmbedder:
    """x-vector embeddings from ``microsoft/wavlm-base-plus-sv`` (CPU by default)."""

    name = "wavlm"

    def __init__(self, model_name: str = WAVLM_SV_MODEL, device: str = "cpu") -> None:
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector

        self._torch = torch
        self.device = device
        self.model_name = model_name
        self._feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self._model = WavLMForXVector.from_pretrained(model_name).to(device).eval()

    def embed(self, wav16: np.ndarray) -> np.ndarray:
        inputs = self._feature_extractor(wav16, sampling_rate=SAMPLE_RATE, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            emb = self._model(**inputs).embeddings[0].float().cpu().numpy()
        return _unit(emb)


class CampPlusEmbedder:
    """CAM++ speaker embeddings from the ``campplus.onnx`` that ships with CosyVoice3.

    The ONNX session options and the Kaldi fbank front end are the ones the model
    itself uses for its speaker conditioning (``extract_spk_embedding``), so the
    scorer sees the audio the way the model's own speaker encoder does.
    """

    name = "campplus"

    def __init__(self, model_dir: str | os.PathLike, onnx_name: str = "campplus.onnx") -> None:
        import onnxruntime

        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self.onnx_path = str(Path(model_dir) / onnx_name)
        self._session = onnxruntime.InferenceSession(
            self.onnx_path, sess_options=option, providers=["CPUExecutionProvider"]
        )

    def embed(self, wav16: np.ndarray) -> np.ndarray:
        from vllm_omni.model_executor.models.cosyvoice3.utils import extract_spk_embedding

        emb = extract_spk_embedding((wav16, SAMPLE_RATE), self._session, "cpu")
        return _unit(emb.numpy())


@dataclass
class OutputScore:
    """Scores for one generated output against all K references."""

    idx: int
    voice: str
    duration_s: float
    scored_s: float
    window_short: bool
    voices: list[str]
    sims: dict[str, list[float]]  # embedder -> cosine to each reference, in ``voices`` order
    argmax: dict[str, str]  # embedder -> best voice
    margin: dict[str, float]  # embedder -> own score minus best other score
    label: str = LABEL_CORRECT
    wrong_voice: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "idx": self.idx,
            "voice": self.voice,
            "duration_s": round(self.duration_s, 3),
            "scored_s": round(self.scored_s, 3),
            "window_short": self.window_short,
            "voices": self.voices,
            "sims": {k: [round(x, 4) for x in v] for k, v in self.sims.items()},
            "argmax": self.argmax,
            "margin": {k: round(v, 4) for k, v in self.margin.items()},
            "label": self.label,
            "wrong_voice": self.wrong_voice,
        }


def classify(voice: str, argmax: Mapping[str, str]) -> tuple[str, str | None]:
    """Label one output from each embedder's argmax voice."""
    picks = set(argmax.values())
    if picks == {voice}:
        return LABEL_CORRECT, None
    if len(picks) == 1:
        return LABEL_WRONG_BOTH_AGREE, next(iter(picks))
    return LABEL_DISAGREE, None


def crop_window(wav16: np.ndarray, window: float | None) -> tuple[np.ndarray, bool]:
    """Return the first ``window`` seconds, and whether the clip was too short for it."""
    if window is None:
        return wav16, False
    n = int(round(window * SAMPLE_RATE))
    if len(wav16) < n:
        return wav16, True
    return wav16[:n], False


def score(
    outputs: Sequence[tuple[str, Any]],
    references: Mapping[str, Any],
    embedders: Sequence[SpeakerEmbedder],
    window: float | None = None,
    *,
    reference_embeddings: Mapping[str, Mapping[str, np.ndarray]] | None = None,
) -> list[OutputScore]:
    """Score each output against all references.

    ``outputs`` is a list of ``(requested_voice, audio)`` and ``references`` maps
    voice name to audio. Audio may be wav bytes, a path, or a mono 16 kHz float32
    array. ``window`` crops each output to its first ``window`` seconds (references
    are always used whole). ``reference_embeddings`` (embedder name -> voice ->
    vector) skips re-embedding the references on repeated calls.
    """
    voices = list(references)
    if len(set(e.name for e in embedders)) != len(embedders):
        raise ValueError("embedder names must be unique")
    if reference_embeddings is None:
        reference_embeddings = embed_references(references, embedders)

    results: list[OutputScore] = []
    for idx, (voice, audio) in enumerate(outputs):
        if voice not in references:
            raise KeyError(f"output {idx} requests unknown voice {voice!r}")
        wav = audio if isinstance(audio, np.ndarray) else load_audio_16k(audio)
        clip, short = crop_window(wav, window)
        sims: dict[str, list[float]] = {}
        argmax: dict[str, str] = {}
        margin: dict[str, float] = {}
        for emb in embedders:
            vec = emb.embed(clip)
            cos = [float(vec @ reference_embeddings[emb.name][v]) for v in voices]
            sims[emb.name] = cos
            argmax[emb.name] = voices[int(np.argmax(cos))]
            own = cos[voices.index(voice)]
            others = [c for v, c in zip(voices, cos) if v != voice]
            margin[emb.name] = own - max(others) if others else float("nan")
        label, wrong = classify(voice, argmax)
        results.append(
            OutputScore(
                idx=idx,
                voice=voice,
                duration_s=len(wav) / SAMPLE_RATE,
                scored_s=len(clip) / SAMPLE_RATE,
                window_short=short,
                voices=voices,
                sims=sims,
                argmax=argmax,
                margin=margin,
                label=label,
                wrong_voice=wrong,
            )
        )
    return results


def embed_references(
    references: Mapping[str, Any], embedders: Sequence[SpeakerEmbedder]
) -> dict[str, dict[str, np.ndarray]]:
    """Embed every reference once per embedder."""
    wavs = {v: (a if isinstance(a, np.ndarray) else load_audio_16k(a)) for v, a in references.items()}
    return {e.name: {v: e.embed(w) for v, w in wavs.items()} for e in embedders}


def reference_matrix(
    references: Mapping[str, Any], embedders: Sequence[SpeakerEmbedder]
) -> dict[str, list[list[float]]]:
    """Reference-versus-reference cosine matrix per embedder (rows and columns in ``references`` order)."""
    emb = embed_references(references, embedders)
    out = {}
    for name, per_voice in emb.items():
        mat = np.stack([per_voice[v] for v in references])
        out[name] = np.round(mat @ mat.T, 4).tolist()
    return out


def wrong_both_agree(results: Sequence[OutputScore]) -> list[OutputScore]:
    return [r for r in results if r.label == LABEL_WRONG_BOTH_AGREE]


def min_margins(results: Sequence[OutputScore]) -> dict[str, float]:
    """Smallest own-minus-best-other margin per embedder (negative means a wrong argmax)."""
    names = results[0].margin.keys() if results else []
    return {n: min(r.margin[n] for r in results) for n in names}


def format_table(
    full: Sequence[OutputScore], first: Sequence[OutputScore], texts_len: Sequence[int] | None = None
) -> str:
    """Compact per-output table for test logs: argmaxes, margins and labels, full and first window."""
    names = list(full[0].margin) if full else []
    short = {n: n[:4] for n in names}
    head = ["idx", "voice", "chars", "dur_s"]
    head += [f"{short[n]}_pick" for n in names] + [f"{short[n]}_mrg" for n in names] + ["label"]
    head += [f"w_{short[n]}_pick" for n in names] + [f"w_{short[n]}_mrg" for n in names] + ["w_label"]
    rows = [head]
    for i, (a, b) in enumerate(zip(full, first)):
        row = [str(a.idx), a.voice, str(texts_len[i]) if texts_len else "-", f"{a.duration_s:.2f}"]
        row += [a.argmax[n] for n in names] + [f"{a.margin[n]:+.3f}" for n in names] + [a.label]
        row += [b.argmax[n] for n in names] + [f"{b.margin[n]:+.3f}" for n in names]
        row += [b.label + ("*" if b.window_short else "")]
        rows.append(row)
    widths = [max(len(r[c]) for r in rows) for c in range(len(head))]
    return "\n".join("  ".join(cell.ljust(w) for cell, w in zip(r, widths)) for r in rows)


@dataclass
class FailureReport:
    """Everything needed to inspect a failed voice-isolation run offline."""

    test: str
    voices: dict[str, str]
    requests: list[dict[str, Any]]
    full: list[OutputScore]
    first: list[OutputScore]
    audio: Sequence[bytes]
    reference_matrix: dict[str, list[list[float]]] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


def retain_failed_voice_isolation(report: FailureReport, directory: str | os.PathLike | None = None) -> list[Path]:
    """Write failing wavs and the similarity matrices for CI artifacts.

    Writes ``failed_voice_isolation_<test>_<idx>.wav`` for every output that is
    not ``correct`` in either window, and one ``failed_voice_isolation_<test>.json``
    with all similarity rows. Does nothing when ``directory`` (default
    ``$VLLM_OMNI_FAILED_SPEECH_AUDIO_DIR``) is unset. I/O errors are printed, never raised.
    """
    directory = directory or os.environ.get("VLLM_OMNI_FAILED_SPEECH_AUDIO_DIR")
    if not directory:
        return []
    root = Path(directory)
    stem = f"failed_voice_isolation_{report.test}"
    written: list[Path] = []
    try:
        root.mkdir(parents=True, exist_ok=True)
        for a, b in zip(report.full, report.first):
            if a.label == LABEL_CORRECT and b.label == LABEL_CORRECT:
                continue
            path = root / f"{stem}_{a.idx}.wav"
            path.write_bytes(report.audio[a.idx])
            written.append(path)
        path = root / f"{stem}.json"
        path.write_text(
            json.dumps(
                {
                    "test": report.test,
                    "voices": report.voices,
                    "requests": report.requests,
                    "reference_matrix": report.reference_matrix,
                    "full": [r.to_dict() for r in report.full],
                    "first_window": [r.to_dict() for r in report.first],
                    **report.extra,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        written.append(path)
        print(f"Retained failed voice isolation artifacts: {[str(p) for p in written]}")
    except OSError as exc:
        print(f"Could not retain failed voice isolation artifacts: {exc}")
    return written
