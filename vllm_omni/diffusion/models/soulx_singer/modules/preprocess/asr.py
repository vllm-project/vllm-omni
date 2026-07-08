"""Lyric transcription (Paraformer + Parakeet) for SoulX preprocess."""

import os
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from omegaconf import open_dict
from vllm.logger import init_logger

from .utils import get_audio_duration

logger = init_logger(__name__)


def _import_funasr_automodel():
    """Import FunASR without shelling out to ffmpeg at module import time."""
    import subprocess
    from functools import wraps

    real_check_output = subprocess.check_output

    @wraps(real_check_output)
    def _safe_check_output(cmd, *args, **kwargs):
        if cmd and Path(str(cmd[0])).name == "ffmpeg":
            raise FileNotFoundError("ffmpeg unavailable for FunASR import probe")
        return real_check_output(cmd, *args, **kwargs)

    subprocess.check_output = _safe_check_output
    try:
        from funasr import AutoModel
    finally:
        subprocess.check_output = real_check_output
    return AutoModel


def build_words_with_gaps(raw_words, raw_timestamps, audio_duration: float):
    words, word_durs = [], []
    prev = 0.0
    for w, t in zip(raw_words, raw_timestamps):
        s, e = float(t[0]), float(t[1])
        if s > prev:
            words.append("<SP>")
            word_durs.append(s - prev)
        words.append(w)
        word_durs.append(e - s)
        prev = e

    if audio_duration > prev:
        if not words:
            words.append("<SP>")
            word_durs.append(audio_duration)
        elif words[-1] != "<SP>":
            words.append("<SP>")
            word_durs.append(audio_duration - prev)
        else:
            word_durs[-1] += audio_duration - prev
    return words, word_durs


class ParaformerASR(nn.Module):
    """Mandarin/Cantonese ASR via FunASR Paraformer."""

    def __init__(self, model_path: str, device: str):
        super().__init__()
        from vllm_omni.diffusion.models.soulx_singer.utils import _patch_torchaudio_load

        _patch_torchaudio_load()
        AutoModel = _import_funasr_automodel()
        self.asr = AutoModel(model=model_path, disable_update=True, device=device)
        logger.info("Loaded Paraformer ASR from %s", model_path)

    def _generate(self, source: str) -> tuple[list[str], list[float], float]:
        out = self.asr.generate(source, output_timestamp=True)[0]
        raw_words = out["text"].replace("@", "").split(" ")
        raw_timestamps = [[t[0] / 1000, t[1] / 1000] for t in out["timestamp"]]
        duration = float(raw_timestamps[-1][1]) if raw_timestamps else 0.0
        return raw_words, raw_timestamps, duration

    def _maybe_refine_with_sidecar_f0(
        self,
        wav_path: str,
        words: list[str],
        word_durs: list[float],
        *,
        sample_rate: int,
        hop_size: int,
    ) -> tuple[list[str], list[float]]:
        """Match upstream ``_ASRZhModel.process``: refine using ``{wav}_f0.npy`` if present."""
        f0_path = os.path.splitext(wav_path)[0] + "_f0.npy"
        if not os.path.isfile(f0_path):
            return words, word_durs
        f0 = np.load(f0_path)
        if f0.size == 0:
            return words, word_durs
        return refine_word_durs_with_f0(
            words,
            word_durs,
            f0,
            sample_rate=sample_rate,
            hop_size=hop_size,
        )

    @torch.inference_mode()
    def forward(self, audio: str | np.ndarray, sample_rate: int = 16000) -> tuple[list[str], list[float]]:
        if isinstance(audio, str):
            raw_words, raw_timestamps, _ = self._generate(audio)
            audio_duration = get_audio_duration(audio)
            words, word_durs = build_words_with_gaps(raw_words, raw_timestamps, audio_duration)
            return self._maybe_refine_with_sidecar_f0(
                audio,
                words,
                word_durs,
                sample_rate=24000,
                hop_size=480,
            )

        waveform = np.asarray(audio, dtype=np.float32).squeeze()
        audio_duration = len(waveform) / float(sample_rate)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            sf.write(tmp_path, waveform, sample_rate)
            raw_words, raw_timestamps, _ = self._generate(tmp_path)
            words, word_durs = build_words_with_gaps(raw_words, raw_timestamps, audio_duration)
            return self._maybe_refine_with_sidecar_f0(
                tmp_path,
                words,
                word_durs,
                sample_rate=24000,
                hop_size=480,
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)
            Path(os.path.splitext(tmp_path)[0] + "_f0.npy").unlink(missing_ok=True)


def _disable_nemo_cuda_graphs(model) -> None:
    decoding_cfg = model.cfg.decoding
    with open_dict(decoding_cfg):
        if hasattr(decoding_cfg, "greedy") and decoding_cfg.greedy is not None:
            decoding_cfg.greedy.use_cuda_graph_decoder = False
        if hasattr(decoding_cfg, "beam") and decoding_cfg.beam is not None:
            decoding_cfg.beam.allow_cuda_graphs = False
    model.change_decoding_strategy(decoding_cfg)


class ParakeetASR(nn.Module):
    """English ASR via NeMo Parakeet-TDT."""

    def __init__(self, model_path: str, device: str):
        super().__init__()
        try:
            import nemo.collections.asr as nemo_asr  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "NeMo (nemo_toolkit) is required for ASR English but is not available in this Python env."
            ) from e

        self.asr = nemo_asr.models.ASRModel.restore_from(
            restore_path=model_path,
            map_location=device,
        )
        self.asr.eval()
        _disable_nemo_cuda_graphs(self.asr)
        logger.info("Loaded Parakeet ASR from %s", model_path)

    @staticmethod
    def _clean_word(word: str) -> str:
        return re.sub(r"[\?\.,:]", "", word).strip()

    @staticmethod
    def _extract_word_segments(output: Any) -> list[dict[str, Any]]:
        ts = getattr(output, "timestamp", None)
        if not ts or not isinstance(ts, dict):
            return []
        word_ts = ts.get("word")
        return word_ts if isinstance(word_ts, list) else []

    @torch.inference_mode()
    def forward(self, audio: str | np.ndarray, sample_rate: int = 16000) -> tuple[list[str], list[float]]:
        if isinstance(audio, str):
            sources = [audio]
            duration = 0.0
        else:
            waveform = np.asarray(audio, dtype=np.float32).squeeze()
            duration = len(waveform) / float(sample_rate)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            sf.write(tmp_path, waveform, sample_rate)
            sources = [tmp_path]

        try:
            outputs = self.asr.transcribe(
                sources,
                timestamps=True,
                batch_size=1,
                num_workers=0,
            )
        finally:
            if not isinstance(audio, str):
                Path(sources[0]).unlink(missing_ok=True)

        output = outputs[0] if outputs else None
        raw_words: list[str] = []
        raw_timestamps: list[list[float]] = []
        if output is not None:
            for w in self._extract_word_segments(output):
                s, e = float(w.get("start", 0.0)), float(w.get("end", 0.0))
                word = self._clean_word(str(w.get("word", "")))
                if word:
                    raw_words.append(word)
                    raw_timestamps.append([s, e])

        if raw_timestamps:
            duration = max(duration, float(raw_timestamps[-1][1]))
        return build_words_with_gaps(raw_words, raw_timestamps, duration)


def refine_word_durs_with_f0(
    words: list[str],
    word_durs: list[float],
    f0: np.ndarray,
    *,
    sample_rate: int = 24000,
    hop_size: int = 480,
) -> tuple[list[str], list[float]]:
    boundaries = np.cumsum([0, *[int(dur * sample_rate / hop_size) for dur in word_durs]]).tolist()
    sil_tolerance = 5
    ext_tolerance = 5
    new_words: list[str] = []
    new_word_durs: list[float] = []
    if words:
        new_words.append(words[0])
        new_word_durs.append(word_durs[0])

    for i in range(1, len(words)):
        word = words[i]
        if word == "<SP>":
            start_frame = boundaries[i]
            end_frame = boundaries[i + 1]
            num_frames = end_frame - start_frame
            frame_idx = start_frame
            unvoiced_count = 0
            while frame_idx < end_frame:
                if f0[frame_idx] <= 1:
                    unvoiced_count += 1
                    if unvoiced_count >= sil_tolerance:
                        frame_idx -= sil_tolerance - 1
                        break
                else:
                    unvoiced_count = 0
                frame_idx += 1

            voice_frames = frame_idx - start_frame
            if voice_frames >= int(num_frames * 0.9):
                new_word_durs[-1] += word_durs[i]
            elif voice_frames >= ext_tolerance:
                dur = voice_frames * hop_size / sample_rate
                new_word_durs[-1] += dur
                new_words.append("<SP>")
                new_word_durs.append(word_durs[i] - dur)
            else:
                new_words.append(word)
                new_word_durs.append(word_durs[i])
        else:
            new_words.append(word)
            new_word_durs.append(word_durs[i])
    return new_words, new_word_durs


class LyricModel(nn.Module):
    def __init__(
        self,
        zh_model_path: str,
        en_model_path: str,
        device: str = "cuda",
        *,
        target_sr: int = 24000,
        hop_size: int = 480,
    ):
        super().__init__()
        self.device_str = device
        self.en_model_path = en_model_path
        self.target_sr = target_sr
        self.hop_size = hop_size
        self.zh_asr = ParaformerASR(zh_model_path, device=device)
        self.en_asr: nn.Module | None = None

    @torch.inference_mode()
    def forward(
        self,
        audio: str | np.ndarray,
        language: str = "Mandarin",
        *,
        sample_rate: int = 16000,
        f0: np.ndarray | None = None,
    ) -> tuple[list[str], list[float]]:
        if language not in {"Mandarin", "Cantonese", "English"}:
            raise ValueError(f"Unsupported language: {language}")

        if language.lower() == "english":
            if self.en_asr is None:
                self.en_asr = ParakeetASR(self.en_model_path, device=self.device_str)
                self.add_module("en_asr", self.en_asr)
            words, durs = self.en_asr(audio, sample_rate=sample_rate)
        else:
            # Path-based ASR loads ``{wav}_f0.npy`` inside ParaformerASR (upstream parity).
            if isinstance(audio, str):
                words, durs = self.zh_asr(audio)
            else:
                words, durs = self.zh_asr(audio, sample_rate=sample_rate)
                if f0 is not None and len(f0) > 0:
                    words, durs = refine_word_durs_with_f0(
                        words,
                        durs,
                        f0,
                        sample_rate=self.target_sr,
                        hop_size=self.hop_size,
                    )
        return words, durs
