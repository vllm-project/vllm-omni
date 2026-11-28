# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Voice prompts for the Tag build (CosyVoice3 flow conditioning).

``voices/<name>.pt``
    precomputed conditioning: ``speech_token`` ``[1, n]``, ``speech_feat``
    ``[1, 2n, 80]`` and ``embedding`` ``[1, 192]``.
``voices/<name>.wav`` / ``.flac``
    zero-shot cloning from a 2-12 s clip of one speaker; the conditioning is
    computed with the same recipe as CosyVoice3's zero-shot frontend.

Clips are prepared by a background thread so that no request waits for it.
Lookup order: ``<voice>``, ``<voice>_<lang>``, then the language's default voice.
A custom voice that is missing or unreadable falls back to the default.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

import numpy as np
import regex as re
import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

HOP = 10  # prompt tokens are cut to a multiple of the first flow chunk
MAX_REFERENCE_S = 12.0
MIN_REFERENCE_S = 2.0
VOICE_NAME = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}")  # names come from clients: no paths
EXTENSIONS = (".pt", ".wav", ".flac")
CLIP_EXTENSIONS = (".wav", ".flac")
SCAN_INTERVAL_S = 5.0


def candidates(name: str, lang: str) -> list[str]:
    name = (name or "").lower().strip()
    if name and not VOICE_NAME.fullmatch(name):
        name = ""
    named = [name, f"{name}_{lang}"] if name else []
    return named + [f"chelsie_{lang}", "vivian_zh" if lang == "zh" else "ryan_en", "vivian_zh"]


class VoiceBank:
    """Conditioning for the voices of one checkpoint, cached by file mtime."""

    def __init__(self, voice_dir: str, talker_dir: str):
        self.voice_dir = voice_dir
        self.talker_dir = talker_dir
        self._cache: dict[str, tuple[tuple[str, float], dict[str, torch.Tensor]]] = {}
        self._bad: set[tuple[str, float]] = set()
        self._lock = threading.Lock()
        self._frontend: dict[str, Any] = {}
        self._scanner: threading.Thread | None = None

    # ------------------------------------------------------------------ clip -> conditioning
    def _sessions(self) -> dict[str, Any]:
        if not self._frontend:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = 2
            cpu = ["CPUExecutionProvider"]
            self._frontend["tokenizer"] = ort.InferenceSession(
                os.path.join(self.talker_dir, "speech_tokenizer_v3.onnx"), sess_options=opts, providers=cpu
            )
            self._frontend["speaker"] = ort.InferenceSession(
                os.path.join(self.talker_dir, "campplus.onnx"), sess_options=opts, providers=cpu
            )
        return self._frontend

    def from_clip(self, path: str) -> dict[str, torch.Tensor]:
        import soundfile as sf
        import torchaudio
        import torchaudio.compliance.kaldi as kaldi

        from vllm_omni.model_executor.models.cosyvoice3.utils import log_mel_spectrogram, mel_spectrogram

        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        wav = torch.from_numpy(np.ascontiguousarray(audio.mean(1)))[None]
        if sr < 16000:
            raise ValueError(f"sample rate {sr} Hz is below 16 kHz")
        wav16 = torchaudio.functional.resample(wav, sr, 16000) if sr != 16000 else wav
        wav16 = wav16[:, : int(16000 * MAX_REFERENCE_S)]
        if wav16.shape[1] < 16000 * MIN_REFERENCE_S:
            raise ValueError(f"reference clip is shorter than {MIN_REFERENCE_S:g} s")
        fe = self._sessions()
        mel = log_mel_spectrogram(wav16, n_mels=128)
        tok_in = fe["tokenizer"].get_inputs()
        tokens = fe["tokenizer"].run(
            None, {tok_in[0].name: mel.numpy(), tok_in[1].name: np.array([mel.shape[2]], dtype=np.int32)}
        )[0]
        tokens = torch.tensor(tokens.reshape(1, -1), dtype=torch.int32)
        wav24 = torchaudio.functional.resample(wav16, 16000, 24000)
        feat = mel_spectrogram(
            wav24,
            n_fft=1920,
            num_mels=80,
            sampling_rate=24000,
            hop_size=480,
            win_size=1920,
            fmin=0,
            fmax=None,
            center=False,
        )
        feat = feat.squeeze(0).transpose(0, 1)[None]  # [1, T, 80]
        fbank = kaldi.fbank(wav16, num_mel_bins=80, dither=0, sample_frequency=16000)
        fbank = fbank - fbank.mean(0, keepdim=True)
        spk_in = fe["speaker"].get_inputs()[0].name
        emb = torch.tensor(fe["speaker"].run(None, {spk_in: fbank[None].numpy()})[0].reshape(1, -1))
        n = min(feat.shape[1] // 2, tokens.shape[1]) // HOP * HOP
        if n < HOP:
            raise ValueError("reference clip too short after tokenization")
        return {"speech_token": tokens[:, :n], "speech_feat": feat[:, : 2 * n].float(), "embedding": emb.float()}

    # ------------------------------------------------------------------ lookup
    def _get(self, name: str, path: str, ext: str, key: tuple[str, float]) -> dict[str, torch.Tensor] | None:
        hit = self._cache.get(name)
        if hit is not None and hit[0] == key:
            return hit[1]
        if key in self._bad:
            return None
        with self._lock:
            hit = self._cache.get(name)
            if hit is not None and hit[0] == key:
                return hit[1]
            try:
                if ext == ".pt":
                    cond = torch.load(path, map_location="cpu", weights_only=True)
                else:
                    cond = self.from_clip(path)
            except Exception as e:  # a bad voice file must not take the stage down
                self._bad.add(key)
                logger.error("Sophon-Realtime: cannot load voice %s: %s", path, e)
                return None
            self._cache[name] = (key, cond)
        if ext != ".pt":
            logger.info("Sophon-Realtime: cloned voice %r from %s", name, path)
        return cond

    def load(self, name: str, lang: str, *, clips: bool = True) -> tuple[str, dict[str, torch.Tensor]]:
        """``(resolved name, conditioning)``. With ``clips=False`` a clip that
        is not prepared yet is skipped instead of being computed here."""
        for cand in candidates(name, lang):
            for ext in EXTENSIONS:
                path = os.path.join(self.voice_dir, cand + ext)
                try:
                    key = (path, os.path.getmtime(path))
                except OSError:
                    continue
                if ext in CLIP_EXTENSIONS and not clips:
                    hit = self._cache.get(cand)
                    if hit is None or hit[0] != key:
                        continue
                cond = self._get(cand, path, ext, key)
                if cond is not None:
                    return cand, cond
        raise ValueError(f"Sophon-Realtime: no voice found in {self.voice_dir}")

    # ------------------------------------------------------------------ background preparation
    def start_scanning(self) -> None:
        """Prepare clip voices in the background: every new or changed clip
        without a ``.pt`` of the same name."""
        if self._scanner is not None:
            return

        def run() -> None:
            try:
                self._sessions()
            except Exception as e:
                logger.warning("Sophon-Realtime: voice cloning unavailable: %s", e)
                return
            while True:
                try:
                    for fname in sorted(os.listdir(self.voice_dir)):
                        name, ext = os.path.splitext(fname)
                        if ext not in CLIP_EXTENSIONS or not VOICE_NAME.fullmatch(name):
                            continue
                        if os.path.exists(os.path.join(self.voice_dir, name + ".pt")):
                            continue
                        path = os.path.join(self.voice_dir, fname)
                        self._get(name, path, ext, (path, os.path.getmtime(path)))
                except Exception as e:
                    logger.warning("Sophon-Realtime: voice scan failed: %s", e)
                time.sleep(SCAN_INTERVAL_S)

        self._scanner = threading.Thread(target=run, name="sophon-realtime-voices", daemon=True)
        self._scanner.start()
