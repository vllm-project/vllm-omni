"""BS-RoFormer vocal separation adapter (external pip package + SoulX chunking)."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
import torch.nn as nn
import yaml
from omegaconf import DictConfig, OmegaConf
from vllm.logger import init_logger
from vllm.multimodal.media.audio import load_audio

from vllm_omni.diffusion.models.interface import SupportAudioInput, SupportsComponentDiscovery
from vllm_omni.diffusion.offloader.layerwise_backend import LayerWiseOffloadBackend

logger = init_logger(__name__)


class MelBandRoformerSeparator(nn.Module, SupportAudioInput, SupportsComponentDiscovery):
    support_audio_input: ClassVar[bool] = True
    _dit_modules: ClassVar[list[str]] = []
    _encoder_modules: ClassVar[list[str]] = ["roformer"]
    _vae_modules: ClassVar[list[str]] = []
    _resident_modules: ClassVar[list[str]] = []
    _layerwise_offload_blocks_attrs: ClassVar[list[str]] = ["roformer.layers"]

    def __init__(
        self,
        config_path: str | Path,
        checkpoint_path: str | Path | None = None,
        *,
        chunk_length_sec: float = 5,
        use_half: bool = True,
    ):
        super().__init__()
        try:
            from bs_roformer import MelBandRoformer as ExtModel
        except ImportError as e:
            raise ImportError(
                'Install BS-RoFormer: pip install "vllm-omni[soulx-preprocess]" or pip install BS-RoFormer'
            ) from e

        cfg = self._load_config(Path(config_path))
        model_cfg = self._normalize_cfg(OmegaConf.to_container(cfg.model, resolve=True))
        self.roformer = ExtModel(**model_cfg)
        self.config = cfg
        self.sample_rate = int(cfg.audio.sample_rate)

        LayerWiseOffloadBackend.set_blocks_attr_names(self.roformer, ["layers"])
        dtype = torch.float16 if use_half else torch.float32
        self.roformer = self.roformer.to(dtype=dtype)

        chunk = int(chunk_length_sec * self.sample_rate)
        self.config.inference = getattr(self.config, "inference", {})
        inf = self.config.inference
        inf.chunk_size = chunk
        inf.num_overlap = inf.get("num_overlap", 4)
        inf.use_amp = False
        inf.normalize = inf.get("normalize", True)

        fade = chunk // 10
        w = torch.ones(chunk)
        w[:fade] = torch.linspace(0, 1, fade)
        w[-fade:] = torch.linspace(1, 0, fade)
        self.register_buffer("_window", w.to(dtype=dtype), persistent=False)

        if checkpoint_path:
            self.load_checkpoint(str(checkpoint_path), use_half=use_half)
        else:
            self.roformer.eval()

    @staticmethod
    def _normalize_cfg(cfg: dict) -> dict:
        tuple_keys = {"multi_stft_resolutions_window_sizes", "freqs_per_bands", "multi_stft_window_sizes"}

        def _fix(v):
            if isinstance(v, list):
                return tuple(_fix(x) for x in v)
            if isinstance(v, dict):
                return {k: _fix(val) for k, val in v.items()}
            return v

        for k in tuple_keys:
            if k in cfg:
                cfg[k] = _fix(cfg[k])
        return cfg

    @staticmethod
    def _load_config(path: Path) -> DictConfig:
        class Loader(yaml.SafeLoader):
            pass

        Loader.add_constructor(
            "tag:yaml.org,2002:python/tuple",
            lambda loader, node: tuple(loader.construct_sequence(node)),
        )
        with open(path) as f:
            return OmegaConf.create(yaml.load(f, Loader=Loader))

    def load_checkpoint(self, path: str, *, use_half: bool = True) -> None:
        self.roformer.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
        if use_half:
            self.roformer.half()
        self.roformer.eval()

    @property
    def chunk_size(self) -> int:
        return int(self.config.inference.chunk_size)

    @property
    def num_overlap(self) -> int:
        return int(self.config.inference.num_overlap)

    @property
    def use_amp(self) -> bool:
        return bool(self.config.inference.use_amp)

    @property
    def normalize_input(self) -> bool:
        return bool(self.config.inference.normalize)

    @property
    def num_channels(self) -> int:
        return int(self.config.audio.get("num_channels", 1))

    def _prepare_mix(self, mix: np.ndarray) -> tuple[np.ndarray, tuple[float, float] | None]:
        if mix.ndim == 1:
            mix = mix[np.newaxis, :]
        if self.num_channels == 2 and mix.shape[0] == 1:
            mix = np.repeat(mix, 2, axis=0)
        if not self.normalize_input:
            return mix, None
        mono = mix.mean(0)
        return (mix - mono.mean()) / mono.std(), (float(mono.mean()), float(mono.std()))

    @staticmethod
    def _select_vocal(recon: torch.Tensor) -> torch.Tensor:
        if recon.ndim == 4:
            return recon[0, 0]
        if recon.ndim == 3:
            return recon[0]
        if recon.ndim == 2:
            return recon
        raise ValueError(recon.shape)

    @torch.inference_mode()
    def forward(self, mix: torch.Tensor) -> torch.Tensor:
        if mix.ndim == 1:
            mix = mix.unsqueeze(0)
        if self.num_channels == 2 and mix.shape[0] == 1:
            mix = mix.repeat(2, 1)

        model_dtype = next(self.roformer.parameters()).dtype
        compute_dtype = torch.float32 if model_dtype == torch.float16 else model_dtype
        if mix.dtype != compute_dtype:
            mix = mix.to(dtype=compute_dtype)

        chunk_size = self.chunk_size
        step = chunk_size // self.num_overlap
        fade = chunk_size // 10
        border = chunk_size - step
        n_samples = mix.shape[-1]
        window = self._window.to(mix.device)

        if n_samples > 2 * border and border > 0:
            mix = nn.functional.pad(mix, (border, border), mode="reflect")

        out = torch.zeros_like(mix)
        weight = torch.zeros(mix.shape[-1], device=mix.device, dtype=mix.dtype)
        pos = 0
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            while pos < mix.shape[-1]:
                part = mix[:, pos : pos + chunk_size]
                seg_len = part.shape[-1]
                pad_mode = "reflect" if seg_len > chunk_size // 2 else "constant"
                part = nn.functional.pad(part, (0, chunk_size - seg_len), mode=pad_mode, value=0)
                stem = self._select_vocal(self.roformer(part.unsqueeze(0)))
                win = window.clone()
                if pos == 0:
                    win[:fade] = 1
                elif pos + step >= n_samples:
                    win[-fade:] = 1
                out[:, pos : pos + seg_len] += stem[:, :seg_len] * win[:seg_len]
                weight[pos : pos + seg_len] += win[:seg_len]
                pos += step

        out = out / weight
        if n_samples > 2 * border and border > 0:
            out = out[..., border:-border]
        return out

    @torch.inference_mode()
    def separate_mono(self, mix: np.ndarray, device: torch.device) -> np.ndarray:
        m, norm = self._prepare_mix(mix)
        target_dtype = next(self.roformer.parameters()).dtype
        v = self.forward(torch.as_tensor(m, dtype=target_dtype, device=device))
        if norm:
            v = v * norm[1] + norm[0]
        v = v.mean(0) if v.ndim == 2 else v
        return v.float().cpu().numpy().astype(np.float32)

    def separate_path(self, path: str, device: torch.device) -> tuple[np.ndarray, int]:
        mix, _ = load_audio(path, sr=self.sample_rate, mono=False)
        return self.separate_mono(np.asarray(mix, dtype=np.float32), device), self.sample_rate
