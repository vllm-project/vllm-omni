# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Content-gate helper for the YuE2 heavy-metal Twinkle e2e test.

Run as ``__main__`` by a dedicated verification interpreter (see
``YUE2_E2E_VERIFY_PYTHON`` in test_yue2.py), NOT by the serving environment:
SheetSage2 wants its own dependency set (mir_eval, pretty_midi, numpy<2),
which must not leak into the serving venv.

Usage:
    python yue2_metal_verify.py --clap-dir DIR --sheetsage2-dir DIR \
        metal.wav control.wav

Prints one JSON object on stdout:
    {"metal_probability": float, "melody_score": float, "control_score": float}

Gates (applied by the caller):
- metal_probability >= 0.9  (CLAP zero-shot heavy-metal style)
- melody_score >= 0.85 and melody_score > control_score
  (SheetSage2 re-transcription, pitch-class LCS vs the golden score, max over
  major/minor variants and 12 transpositions, pedal-tone tolerant)
"""

import argparse
import json
import re
import wave
from typing import Any

import numpy as np

_NOTE_BASE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_GOLDEN_MAJOR = "C4C4G4G4|A4A4G8|F4F4E4E4|D4D4C8|G4G4F4F4|E4E4D8|G4G4F4F4|E4E4D8|C4C4G4G4|A4A4G8|F4F4E4E4|D4D4C8|"
# Metal arrangements commonly minor-ize the melody; accept both modes.
_GOLDEN_MINOR = (
    "C4C4G4G4|_A4_A4G8|F4F4_E4_E4|D4D4C8|G4G4F4F4|_E4_E4D8|G4G4F4F4|_E4_E4D8|C4C4G4G4|_A4_A4G8|F4F4_E4_E4|D4D4C8|"
)

_STYLE_CANDIDATES = [
    "heavy metal music with distorted guitars and aggressive drums",
    "a soft children's lullaby",
    "classical piano music",
    "pop music",
]


def _abc_note_to_midi(token):
    m = re.match(r"([\^_=]*)([A-Ga-g])([,']*)(\d*)", token)
    if not m:
        return None
    acc, letter, oct_marks, _ = m.groups()
    pc = _NOTE_BASE[letter.upper()] + acc.count("^") - acc.count("_")
    octave = 5 if letter.isupper() else 6
    octave += oct_marks.count("'") - oct_marks.count(",")
    return 12 * octave + pc


def _collapse(seq):
    out: list[Any] = []
    for x in seq:
        if not out or x != out[-1]:
            out.append(x)
    return out


def _golden_events(body):
    return _collapse([_abc_note_to_midi(t) for t in re.findall(r"[\^_=]*[A-G]\d*", body)])


def _melody_events(score_abc):
    """Note events of the voice carrying the most notes in a SheetSage2 score."""
    voices: dict[str | None, list[int]] = {}
    current = None
    for line in score_abc.splitlines():
        line = line.strip()
        if line.startswith("V:"):
            parts = line.split()
            current = parts[1] if len(parts) > 1 else None
            continue
        if current is None or not line or line.startswith(("%", "X:", "T:", "M:", "L:", "Q:", "K:")):
            continue
        for tok in re.findall(r"[\^_=]*[A-Ga-g][,']*\d*", line.replace("|", " ")):
            midi = _abc_note_to_midi(tok)
            if midi is not None:
                voices.setdefault(current, []).append(midi)
    return _collapse(max(voices.values(), key=len, default=[]))


def _lcs(a, b):
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            tmp = dp[j]
            dp[j] = prev + 1 if x == y else max(dp[j], dp[j - 1])
            prev = tmp
    return dp[-1]


def melody_score(score_abc):
    gen = [m % 12 for m in _melody_events(score_abc)]
    if len(gen) < 4:
        return 0.0
    best = 0
    for body in (_GOLDEN_MAJOR, _GOLDEN_MINOR):
        gold = [m % 12 for m in _golden_events(body)]
        best = max(best, max(_lcs(gen, [(p + r) % 12 for p in gold]) for r in range(12)))
    return best / len(_golden_events(_GOLDEN_MAJOR))


def _wav_to_mono(path):
    with wave.open(path) as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())
    data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, sr


def metal_probability(wav_path, clap_dir):
    import torch
    from transformers import ClapModel, ClapProcessor

    model = ClapModel.from_pretrained(clap_dir)
    proc = ClapProcessor.from_pretrained(clap_dir)
    audio, sr = _wav_to_mono(wav_path)
    if sr != 48000:
        n = int(len(audio) * 48000 / sr)
        audio = np.interp(np.linspace(0, len(audio), n), np.arange(len(audio)), audio).astype(np.float32)
    audio = audio[: 48000 * 10]
    inputs = proc(
        text=_STYLE_CANDIDATES,
        audios=[audio] * len(_STYLE_CANDIDATES),
        sampling_rate=48000,
        return_tensors="pt",
        padding=True,
    )
    with torch.no_grad():
        probs = model(**inputs).logits_per_audio[0].softmax(dim=-1)
    return float(probs[0])


def transcribe_melody_abc(wav_path, sheetsage2_dir, output_dir):
    import torch
    from transformers import AutoModel

    audio, sr = _wav_to_mono(wav_path)
    model = AutoModel.from_pretrained(sheetsage2_dir, trust_remote_code=True)
    result = model.transcribe(
        torch.from_numpy(audio[None, :]),
        sampling_rate=sr,
        output_dir=str(output_dir),
        melody_only=True,
        dtype="fp32",
    )
    abc = result.get("abc")
    if not abc:
        raise RuntimeError(f"SheetSage2 produced no melody ABC: {result.get('abc_error')}")
    return abc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clap-dir", required=True)
    parser.add_argument("--sheetsage2-dir", required=True)
    parser.add_argument("metal_wav")
    parser.add_argument("control_wav")
    parser.add_argument("--work-dir", default=None)
    args = parser.parse_args()

    work = args.work_dir or "."
    report = {"metal_probability": metal_probability(args.metal_wav, args.clap_dir)}
    report["melody_score"] = melody_score(
        transcribe_melody_abc(args.metal_wav, args.sheetsage2_dir, f"{work}/metal_score")
    )
    report["control_score"] = melody_score(
        transcribe_melody_abc(args.control_wav, args.sheetsage2_dir, f"{work}/control_score")
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
