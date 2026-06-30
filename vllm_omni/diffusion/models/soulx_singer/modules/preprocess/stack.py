"""Lazy-loaded SoulX preprocess model tree."""

import os
import tempfile
from pathlib import Path
from typing import ClassVar

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from vllm.logger import init_logger

from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery

from .asr import LyricModel
from .mel_grid_f0 import extract_f0_file
from .rmvpe import RMVPE
from .rosvot import RosvotModel
from .segmenter import VocalSegmenter
from .utils import resample_mono
from .vocal_sep import MelBandRoformerSeparator

logger = init_logger(__name__)


class SoulXPreprocessStack(nn.Module, SupportsComponentDiscovery):
    """Unified nn.Module tree for SoulX-Singer audio preprocess."""

    _dit_modules: ClassVar[list[str]] = []
    _encoder_modules: ClassVar[list[str]] = ["vocal_sep", "lyric", "rosvot"]
    _vae_modules: ClassVar[list[str]] = []
    _resident_modules: ClassVar[list[str]] = ["rmvpe"]

    def __init__(
        self,
        weights: dict[str, str],
        device: str,
        *,
        target_sr: int = 24000,
        hop_size: int = 480,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.weights = weights
        self.device_str = device
        self.target_sr = target_sr
        self.hop_size = hop_size
        self.verbose = verbose
        self._rmvpe: nn.Module | None = None
        self._vocal_sep: nn.Module | None = None
        self._segmenter: nn.Module | None = None
        self._lyric: nn.Module | None = None
        self._rosvot: nn.Module | None = None

    @property
    def device(self) -> torch.device:
        return torch.device(self.device_str)

    def ensure_rmvpe(self):
        if self._rmvpe is None:
            if self.verbose:
                logger.info("[rmvpe] loading shared pitch model")
            self._rmvpe = RMVPE(self.weights["rmvpe"], device=self.device_str)
            self.add_module("rmvpe", self._rmvpe)
        return self._rmvpe

    def ensure_vocal_sep(self):
        if self._vocal_sep is None:
            if self.verbose:
                logger.info("[vocal extraction] loading model")
            self._vocal_sep = MelBandRoformerSeparator(
                self.weights["sep_config"],
                self.weights["sep_ckpt"],
                use_half=False,
            ).to(self.device)
            self.add_module("vocal_sep", self._vocal_sep)
            if self.verbose:
                logger.info("[vocal extraction] ready on %s, sr=%s", self.device_str, self._vocal_sep.sample_rate)
        return self._vocal_sep

    def ensure_segmenter(self):
        if self._segmenter is None:
            self._segmenter = VocalSegmenter(verbose=self.verbose)
            self.add_module("segmenter", self._segmenter)
        return self._segmenter

    def ensure_lyric(self):
        if self._lyric is None:
            if self.verbose:
                logger.info("[lyric transcription] loading ASR")
            self._lyric = LyricModel(
                self.weights["asr_zh"],
                self.weights["asr_en"],
                device=self.device_str,
                target_sr=self.target_sr,
                hop_size=self.hop_size,
            )
            self.add_module("lyric", self._lyric)
        return self._lyric

    def ensure_rosvot(self):
        if self._rosvot is None:
            if self.verbose:
                logger.info("[note transcription] loading ROSVOT")
            rosvot_src = os.environ.get("ROSVOT_SOURCE_DIR")
            self._rosvot = RosvotModel(
                self.weights["rosvot"],
                pe=self.ensure_rmvpe(),
                verbose=self.verbose,
                rosvot_source_dir=rosvot_src,
            ).to(self.device)
            self.add_module("rosvot", self._rosvot)
        return self._rosvot

    def extract_f0(self, vocal, sample_rate: int, *, f0_path: str | None = None) -> np.ndarray:
        """Extract F0 on the mel grid; ndarray inputs are spooled to a temp wav."""
        rmvpe = self.ensure_rmvpe()
        kwargs = dict(
            target_sr=self.target_sr,
            hop_size=self.hop_size,
            f0_path=f0_path,
            verbose=self.verbose,
        )
        if isinstance(vocal, str):
            return extract_f0_file(rmvpe, vocal, **kwargs)
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            sf.write(tmp_path, np.asarray(vocal, dtype=np.float32).squeeze(), int(sample_rate))
            return extract_f0_file(rmvpe, tmp_path, **kwargs)
        finally:
            if tmp_path is not None:
                Path(tmp_path).unlink(missing_ok=True)

    def extract_vocal(self, audio: str | tuple[np.ndarray, int]) -> tuple[np.ndarray, int]:
        sep = self.ensure_vocal_sep()
        if isinstance(audio, tuple):
            mix, sr = audio
            mix = np.asarray(mix, dtype=np.float32).squeeze()
            if sr != sep.sample_rate:
                mix = resample_mono(mix, orig_sr=sr, target_sr=sep.sample_rate)
            return sep.separate_mono(mix, self.device), sep.sample_rate
        return sep.separate_path(str(audio), self.device)
