# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniMax H3 remote-code VAE adapters and exact latent contracts."""

from __future__ import annotations

import importlib
import json
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from PIL import Image
from safetensors import safe_open
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from vllm.logger import init_logger
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import (
    DistributedVaeMixin,
)
from vllm_omni.diffusion.distributed.parallel_state import get_world_group
from vllm_omni.diffusion.offloader.module_residency import (
    BoundedAllocatorCache,
    PinnedModuleStager,
)

from .ops import install_h3_vae_optimizations
from .packed_tokens import minimax_h3_patchify_video_latent

MINIMAX_H3_KEYFRAME_ENCODE_SEED = 42
MINIMAX_H3_AUDIO_SAMPLE_RATE = 32000
MINIMAX_H3_AUDIO_CHANNELS = 2


logger = init_logger(__name__)


def _load_component_config(component_path: str) -> dict[str, Any]:
    config_path = Path(component_path) / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    channels = int(config["latent_channels"])
    for key in ("latents_mean", "latents_std"):
        values = config.get(key)
        if not isinstance(values, list) or len(values) != channels:
            raise ValueError(f"{config_path}: {key} must contain {channels} values")
    return config


def _load_remote_component(
    component_path: str,
    config: dict[str, Any],
) -> nn.Module:
    auto_map = config.get("auto_map") or {}
    class_reference = auto_map.get("AutoModel")
    if not isinstance(class_reference, str):
        raise ValueError(f"{component_path}/config.json must define auto_map.AutoModel")
    component_cls = get_class_from_dynamic_module(
        class_reference,
        component_path,
    )
    # Build on the host regardless of the ambient default device. Online
    # quantization wraps pipeline construction in a `with torch.device(<accel>)`
    # block for the DiT's quantized linears, and the checkpoint's own VAE code
    # builds constants with ops that have no accelerator kernel (BigVGAN's
    # anti-aliasing filters call torch.kaiser_window). Callers place the module
    # explicitly right after this returns, so nothing depends on the context.
    with torch.device("cpu"):
        return component_cls.from_pretrained(component_path)


def _load_selected_state_dict(
    model: nn.Module,
    weights_path: Path,
    prefixes: tuple[str, ...],
) -> None:
    with safe_open(str(weights_path), framework="pt", device="cpu") as checkpoint:
        selected = sorted(name for name in checkpoint.keys() if name.startswith(prefixes))
        if not selected:
            raise ValueError(f"{weights_path}: no VAE weights match prefixes {prefixes!r}")
        state_dict = {name: checkpoint.get_tensor(name) for name in selected}
    model.load_state_dict(state_dict, strict=True)


def _remove_modules(model: nn.Module, names: tuple[str, ...]) -> None:
    for name in names:
        if not isinstance(getattr(model, name, None), nn.Module):
            raise ValueError(f"MiniMax H3 VAE model has no module {name!r}")
        delattr(model, name)


def _load_video_vae_encoder(
    component_path: str,
    config: dict[str, Any],
) -> nn.Module:
    class_reference = (config.get("auto_map") or {}).get("AutoModel")
    if not isinstance(class_reference, str):
        raise ValueError(f"{component_path}/config.json must define auto_map.AutoModel")
    component_cls = get_class_from_dynamic_module(class_reference, component_path)
    component_module = importlib.import_module(component_cls.__module__)
    source_classes = getattr(component_module, "_SOURCE_CLASSES", None)
    source_class_name = config.get("source_class_name")
    if not isinstance(source_classes, dict) or source_class_name not in source_classes:
        raise ValueError(f"unsupported MiniMax H3 video VAE source class {source_class_name!r}")
    if bool(config["vae_parallel_tiling"]):
        ensure_parallel_state = getattr(component_module, "_ensure_vae_parallel_state", None)
        if not callable(ensure_parallel_state):
            raise ValueError("MiniMax H3 video VAE remote module does not expose its parallel-state initializer")
        ensure_parallel_state()

    source_path = Path(component_path) / str(config["source_path"])
    weights_path = source_path / str(config["source_safetensors_path"])
    load_kwargs = {
        "clip_length": int(config["vae_clip_length"]),
        "token_drop": int(config["vae_token_drop"]),
        "encoder_tiling": int(config["vae_encoder_tiling"]),
        "decoder_tiling": int(config["vae_decoder_tiling"]),
        "parallel_tiling": int(config["vae_parallel_tiling"]),
        "tile_size": int(config["vae_tile_size"]),
        "tile_overlap_min": int(config["vae_tile_overlap_min"]),
        "encoder_parallel": int(config["vae_encoder_parallel"]),
        "decoder_parallel": int(config["vae_decoder_parallel"]),
        "chunk_dim": int(config["vae_chunk_dim"]),
    }
    source_cls = source_classes[source_class_name]
    source_config = source_cls.load_config(str(source_path))
    with set_default_torch_dtype(torch.float32), torch.device("cpu"):
        model, _unused = source_cls.from_config(source_config, return_unused_kwargs=True, **load_kwargs)
    _remove_modules(model, ("decoder", "post_quant_conv"))
    _load_selected_state_dict(model, weights_path, ("encoder.", "quant_conv."))
    model.eval()
    return component_cls(model)


def _load_audio_vae_encoder(
    component_path: str,
    config: dict[str, Any],
) -> nn.Module:
    class_reference = (config.get("auto_map") or {}).get("AutoModel")
    if not isinstance(class_reference, str):
        raise ValueError(f"{component_path}/config.json must define auto_map.AutoModel")
    component_cls = get_class_from_dynamic_module(class_reference, component_path)
    component_module = importlib.import_module(component_cls.__module__)
    load_yaml = getattr(component_module, "_load_yaml", None)
    model_cls = getattr(component_module, "DacAudioVAE", None)
    if not callable(load_yaml) or not isinstance(model_cls, type):
        raise ValueError("MiniMax H3 audio VAE remote module does not expose its source loader")

    component_dir = Path(component_path)
    audio_config = load_yaml(component_dir / str(config["source_config_path"]))
    metadata_path = component_dir / str(config["source_metadata_path"])
    metadata_doc = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata = metadata_doc["metadata"]["kwargs"]
    with set_default_torch_dtype(torch.float32), torch.device("cpu"):
        model = model_cls(
            encoder_rates=metadata["encoder_rates"],
            decoder_rates=metadata["decoder_rates"],
            attn_proj=metadata["attn_proj"],
            decoder_type=metadata["decoder_type"],
            decoder_dim=audio_config["model_config"]["decoder_dim"],
            vae_latent_channels=audio_config["model_config"]["vae_latent_channels"],
            sample_rate=metadata["sample_rate"],
        )
    _remove_modules(model, ("decoder", "dec_in_proj", "logs_proj"))
    weights_path = component_dir / str(config["source_safetensors_path"])
    _load_selected_state_dict(model, weights_path, ("encoder.", "pre_block.", "mean_proj."))
    model.eval()
    return component_cls(model)


class _AudioVAEDeterminismContext(AbstractContextManager):
    def __enter__(self):
        backends = torch.backends
        self._saved = (
            backends.cuda.matmul.allow_tf32,
            backends.cudnn.allow_tf32,
            backends.cudnn.benchmark,
            backends.cudnn.deterministic,
            backends.cudnn.enabled,
            backends.cuda.flash_sdp_enabled(),
            backends.cuda.mem_efficient_sdp_enabled(),
            backends.cuda.math_sdp_enabled(),
        )
        backends.cuda.matmul.allow_tf32 = False
        backends.cudnn.allow_tf32 = False
        backends.cudnn.benchmark = False
        backends.cudnn.deterministic = True
        backends.cudnn.enabled = False
        backends.cuda.enable_flash_sdp(False)
        backends.cuda.enable_mem_efficient_sdp(False)
        backends.cuda.enable_math_sdp(True)
        return self

    def __exit__(self, exc_type, exc, traceback):
        backends = torch.backends
        (
            backends.cuda.matmul.allow_tf32,
            backends.cudnn.allow_tf32,
            backends.cudnn.benchmark,
            backends.cudnn.deterministic,
            backends.cudnn.enabled,
            flash,
            memory_efficient,
            math_sdp,
        ) = self._saved
        backends.cuda.enable_flash_sdp(flash)
        backends.cuda.enable_mem_efficient_sdp(memory_efficient)
        backends.cuda.enable_math_sdp(math_sdp)
        return False


class MiniMaxH3VideoVAE(nn.Module, DistributedVaeMixin):
    """Adapter around the checkpoint's native parallel-tiled video VAE."""

    def __init__(
        self,
        component_path: str,
        *,
        device: torch.device,
        load_device: torch.device | None = None,
        encode_only: bool = False,
        decode_only: bool = False,
    ) -> None:
        super().__init__()
        self._device_target = device
        self.encode_only = bool(encode_only)
        self.decode_only = bool(decode_only)
        if self.encode_only and self.decode_only:
            raise ValueError("MiniMax H3 video VAE cannot be both encode-only and decode-only")
        self.config_dict = _load_component_config(component_path)
        if self.encode_only:
            self.remote = _load_video_vae_encoder(component_path, self.config_dict)
        else:
            self.remote = _load_remote_component(component_path, self.config_dict)
            if self.decode_only:
                _remove_modules(self.remote.model, ("encoder", "quant_conv"))
        # Match the reference loader contract before installing inference-only
        # decoder fast paths. Keyframe encoding remains FP32; decoder Linear
        # weights may be materialized in FP16 because reference decode casts
        # those same tensors through CUDA autocast on every tile.
        initial_device = load_device or device
        self.remote.eval().to(device=initial_device, dtype=torch.float32)
        decoder = getattr(self.remote.model, "decoder", None)
        if decoder is not None:
            install_h3_vae_optimizations(
                decoder,
                device=device,
            )
        self._stager = None
        if initial_device.type == "cpu" and device.type not in ("cpu", "meta"):
            self._stager = PinnedModuleStager(
                self.remote,
                device,
                pin_memory=True,
            )
        self.model = self.remote.model
        self.use_tiling = True
        self.use_slicing = False
        self.parallel_size = 1
        self.device_module = torch.get_device_module()

    def load_to_device(self) -> None:
        if self._stager is not None:
            self._stager.load()
        else:
            self.remote.to(self._device_target)

    def set_omni_component_cache(self, cache: BoundedAllocatorCache | None) -> None:
        self._omni_component_cache = cache
        if self._stager is not None:
            self._stager.set_cache_retention(cache)

    def offload_to_cpu(self) -> None:
        if self._stager is not None:
            self._stager.offload()
        else:
            self.remote.to("cpu")
            cache = getattr(self, "_omni_component_cache", None)
            if cache is None:
                torch.accelerator.empty_cache()
            else:
                cache.release_if_needed()

    def set_parallel_size(
        self,
        parallel_size: int,
        mode: str = "tile",
        process_group: dist.ProcessGroup | None = None,
    ) -> None:
        if mode != "tile":
            raise ValueError(f"MiniMax H3 VAE supports its native tile parallel mode only, got {mode!r}")
        group = process_group if process_group is not None else get_world_group().device_group
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        parallel_size = int(parallel_size)
        if parallel_size not in (1, world_size):
            raise ValueError(
                "MiniMax H3 native VAE patch parallelism currently requires "
                "vae_patch_parallel_size=1 or the full DiT group size "
                f"({world_size}), got {parallel_size}"
            )
        self.parallel_size = parallel_size
        enabled = parallel_size > 1

        state = self._native_parallel_state()
        state.clear()
        state.update(
            group_size=parallel_size,
            group_rank=rank if enabled else 0,
            local_process_group=group if enabled else None,
            sp_size=parallel_size,
            sp_rank=rank if enabled else 0,
            sp_enabled=enabled,
            sp_process_group=group if enabled else None,
            tp_size=1,
            tp_rank=0,
        )
        self.model.parallel_tiling = enabled

    def _native_parallel_state(self) -> dict[str, Any]:
        """Return the checkpoint's own mutable parallel-state dict."""

        package = self.remote.__class__.__module__.rsplit(".", 1)[0]
        parallel_module = importlib.import_module(f"{package}.parallel")
        return parallel_module.get_parallel_state()

    def _decoder_tile_count(self, latent: torch.Tensor) -> int:
        """Number of decoder tiles the checkpoint will split ``latent`` into.

        Mirrors the checkpoint's ``decode_tiled``: the grid is computed from the
        pixel-space dimensions, so it is a pure function of the latent shape and
        resolves identically on every rank.
        """

        ratio = int(self.model.vae_ratio)
        rows, _, _ = self.model.split_tiles(int(latent.shape[-2]) * ratio, True)
        cols, _, _ = self.model.split_tiles(int(latent.shape[-1]) * ratio, True)
        return len(rows) * len(cols)

    def _encoder_tile_count(self, height: int, width: int) -> int:
        processor = getattr(self.model, "processor", None)
        align = getattr(processor, "_align_to_total_patch_size", None)
        if callable(align):
            height, width = align(int(height), int(width))
        rows, _, _ = self.model.split_tiles(int(height), False)
        cols, _, _ = self.model.split_tiles(int(width), False)
        return len(rows) * len(cols)

    def _encoder_tiling_context(
        self,
        height: int,
        width: int,
    ) -> AbstractContextManager:
        parallel_size = int(getattr(self, "parallel_size", 1))
        if parallel_size <= 1:
            return nullcontext()
        num_tiles = self._encoder_tile_count(height, width)
        if num_tiles >= parallel_size:
            return nullcontext()
        logger.warning_once(
            "MiniMax-H3 VAE encode splits a %dx%d input into %d tile(s) but "
            "the tile group has %d ranks; encoding rank-locally for this "
            "shape instead, which avoids ranks without tiles hanging the "
            "collective.",
            width,
            height,
            num_tiles,
            parallel_size,
        )
        return self._rank_local_tiling()

    @contextmanager
    def _rank_local_tiling(self) -> Iterator[None]:
        state = self._native_parallel_state()
        saved_state = dict(state)
        saved_tiling = self.model.parallel_tiling
        state.update(
            group_size=1,
            group_rank=0,
            local_process_group=None,
            sp_size=1,
            sp_rank=0,
            sp_enabled=False,
            sp_process_group=None,
            tp_size=1,
            tp_rank=0,
        )
        self.model.parallel_tiling = False
        try:
            yield
        finally:
            state.clear()
            state.update(saved_state)
            self.model.parallel_tiling = saved_tiling

    def is_distributed_enabled(self) -> bool:
        return self.parallel_size > 1 and dist.is_initialized()

    @torch.inference_mode()
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        if getattr(self, "decode_only", False):
            raise RuntimeError("MiniMax H3 decode-only video VAE cannot encode images")
        previous_parallel = self.model.parallel_tiling
        if int(getattr(self, "parallel_size", 1)) <= 1:
            self.model.parallel_tiling = False
        parameter = next(self.parameters())
        previous_dtype = parameter.dtype
        if previous_dtype != torch.float32:
            self.to(torch.float32)
        devices = [parameter.device] if parameter.device.type != "cpu" else []
        try:
            with (
                self._encoder_tiling_context(image.height, image.width),
                torch.random.fork_rng(
                    devices=devices,
                    device_type=parameter.device.type,
                ),
            ):
                torch.default_generator.manual_seed(MINIMAX_H3_KEYFRAME_ENCODE_SEED)
                for device in devices:
                    with self.device_module.device(device):
                        self.device_module.manual_seed(MINIMAX_H3_KEYFRAME_ENCODE_SEED)
                latent = self.model.encode_images(
                    image,
                    use_fp16_latent=True,
                )[0]
        finally:
            self.model.parallel_tiling = previous_parallel
            if previous_dtype != torch.float32:
                self.to(previous_dtype)

        # Match the reference contract exactly: normalization and patchify
        # happen on CPU in FP32 after the sampled encode. The condition noise
        # path is sensitive enough that doing these elementwise operations on
        # CUDA can noticeably change the final conditioned video.
        latent = latent.float().cpu()
        if latent.ndim == 4:
            latent = latent[None]
        channels = int(self.config_dict["latent_channels"])
        mean = torch.tensor(
            self.config_dict["latents_mean"],
        ).view(1, channels, 1, 1, 1)
        std = torch.tensor(
            self.config_dict["latents_std"],
        ).view(1, channels, 1, 1, 1)
        return minimax_h3_patchify_video_latent(
            (latent - mean) / std,
            patch_size=(1, 2, 2),
        ).float()

    @torch.inference_mode()
    def encode_video(
        self,
        frames: Any,
    ) -> tuple[torch.Tensor, tuple[int, int, int]]:
        if getattr(self, "decode_only", False):
            raise RuntimeError("MiniMax H3 decode-only video VAE cannot encode videos")
        parameter = next(self.parameters())
        previous_dtype = parameter.dtype
        if previous_dtype != torch.float32:
            self.to(torch.float32)
        devices = [parameter.device] if parameter.device.type != "cpu" else []
        shape = getattr(frames, "shape", None)
        if int(getattr(self, "parallel_size", 1)) > 1:
            if shape is None or len(shape) != 4:
                raise ValueError("parallel MiniMax H3 video encode requires a rank-4 frame array")
            if int(shape[-1]) in {1, 3, 4}:
                frame_height, frame_width = int(shape[-3]), int(shape[-2])
            else:
                frame_height, frame_width = int(shape[-2]), int(shape[-1])
            tiling_context = self._encoder_tiling_context(frame_height, frame_width)
        else:
            tiling_context = nullcontext()
        try:
            with tiling_context, torch.random.fork_rng(devices=devices, device_type=parameter.device.type):
                torch.default_generator.manual_seed(MINIMAX_H3_KEYFRAME_ENCODE_SEED)
                for device in devices:
                    with self.device_module.device(device):
                        self.device_module.manual_seed(MINIMAX_H3_KEYFRAME_ENCODE_SEED)
                latent = self.model.encode_videos(
                    frames,
                    use_fp16_latent=True,
                )[0]
        finally:
            if previous_dtype != torch.float32:
                self.to(previous_dtype)

        latent = latent.float().cpu()
        if latent.ndim == 4:
            latent = latent[None]
        channels = int(self.config_dict["latent_channels"])
        if latent.ndim != 5 or int(latent.shape[1]) != channels:
            raise ValueError(f"unexpected reference video latent shape {tuple(latent.shape)}")
        shape = (
            int(latent.shape[2]),
            int(latent.shape[3]),
            int(latent.shape[4]),
        )
        mean = torch.tensor(
            self.config_dict["latents_mean"],
        ).view(1, channels, 1, 1, 1)
        std = torch.tensor(
            self.config_dict["latents_std"],
        ).view(1, channels, 1, 1, 1)
        rows = minimax_h3_patchify_video_latent(
            (latent - mean) / std,
            patch_size=(1, 2, 2),
        ).float()
        return rows, shape

    @torch.inference_mode()
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if getattr(self, "encode_only", False):
            raise RuntimeError("MiniMax H3 encode-only video VAE cannot decode latents")
        channels = int(self.config_dict["latent_channels"])
        mean = torch.tensor(
            self.config_dict["latents_mean"],
            device=latent.device,
            dtype=latent.dtype,
        ).view(1, channels, 1, 1, 1)
        std = torch.tensor(
            self.config_dict["latents_std"],
            device=latent.device,
            dtype=latent.dtype,
        ).view(1, channels, 1, 1, 1)
        # The checkpoint hands rank r the tiles ``range(r, num_tiles, sp_size)``
        # and then rejects an empty share inside the gather. A rank with no
        # tiles raises and leaves the collective while the others block in it
        # forever, so too few tiles hangs the whole stage rather than failing
        # it. Tile count depends only on the latent shape, so every rank takes
        # this branch together.
        num_tiles = self._decoder_tile_count(latent)
        if self.parallel_size > 1 and num_tiles < self.parallel_size:
            logger.warning_once(
                "MiniMax-H3 VAE decode splits into %d tile(s) but the tile group has "
                "%d ranks; decoding rank-locally for this shape instead, which is "
                "slower but avoids ranks without tiles hanging the collective.",
                num_tiles,
                self.parallel_size,
            )
            tiling_context: AbstractContextManager = self._rank_local_tiling()
        else:
            tiling_context = nullcontext()

        with tiling_context:
            decoded = self.model.decode_base(latent * std + mean)
        frames = self.model.processor.revert_tensor(decoded)
        if frames.ndim == 4:
            frames = frames.unsqueeze(0).transpose(1, 2)
        if frames.ndim != 5:
            raise ValueError(f"unexpected decoded video shape {tuple(frames.shape)}")
        return frames.float()


class MiniMaxH3AudioVAE(nn.Module):
    def __init__(
        self,
        component_path: str,
        *,
        device: torch.device,
        load_device: torch.device | None = None,
        encode_only: bool = False,
        decode_only: bool = False,
    ) -> None:
        super().__init__()
        self._device_target = device
        self.encode_only = bool(encode_only)
        self.decode_only = bool(decode_only)
        if self.encode_only and self.decode_only:
            raise ValueError("MiniMax H3 audio VAE cannot be both encode-only and decode-only")
        self.config_dict = _load_component_config(component_path)
        if self.encode_only:
            self.remote = _load_audio_vae_encoder(component_path, self.config_dict)
        else:
            self.remote = _load_remote_component(component_path, self.config_dict)
            if self.decode_only:
                _remove_modules(self.remote.model, ("encoder", "pre_block", "mean_proj"))
        # The checkpoint's audio VAE contract is FP32 for both reference
        # encoding and waveform decoding.
        initial_device = load_device or device
        self.remote.eval().to(device=initial_device, dtype=torch.float32)
        self._stager = None
        if initial_device.type == "cpu" and device.type not in ("cpu", "meta"):
            self._stager = PinnedModuleStager(
                self.remote,
                device,
                pin_memory=True,
            )
        self.model = self.remote.model
        self.sample_rate = int(self.config_dict["sample_rate"])

    def load_to_device(self) -> None:
        if self._stager is not None:
            self._stager.load()
        else:
            self.remote.to(self._device_target)

    def set_omni_component_cache(self, cache: BoundedAllocatorCache | None) -> None:
        self._omni_component_cache = cache
        if self._stager is not None:
            self._stager.set_cache_retention(cache)

    def offload_to_cpu(self) -> None:
        if self._stager is not None:
            self._stager.offload()
        else:
            self.remote.to("cpu")
            cache = getattr(self, "_omni_component_cache", None)
            if cache is None:
                torch.accelerator.empty_cache()
            else:
                cache.release_if_needed()

    @torch.inference_mode()
    def encode_waveform(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> tuple[torch.Tensor, int]:
        if getattr(self, "decode_only", False):
            raise RuntimeError("MiniMax H3 decode-only audio VAE cannot encode waveforms")
        import torchaudio

        waveform = waveform.float()
        if waveform.ndim == 1:
            waveform = waveform[None]
        if int(sample_rate) != MINIMAX_H3_AUDIO_SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(
                int(sample_rate),
                MINIMAX_H3_AUDIO_SAMPLE_RATE,
            )(waveform)
        if waveform.shape[0] < MINIMAX_H3_AUDIO_CHANNELS:
            waveform = waveform.repeat(
                MINIMAX_H3_AUDIO_CHANNELS,
                1,
            )
        waveform = waveform[:MINIMAX_H3_AUDIO_CHANNELS]
        device = next(self.model.parameters()).device
        waveform = waveform.to(device)

        with _AudioVAEDeterminismContext():
            audio = self.model.preprocess(
                waveform.unsqueeze(1),
                MINIMAX_H3_AUDIO_SAMPLE_RATE,
            )
            latent = self.model.encoder(audio)
            if bool(getattr(self.model, "attn_proj", False)):
                latent = self.model.pre_block(latent.transpose(1, 2)).transpose(1, 2)
            latent = self.model.mean_proj(latent).float().cpu()

        channels = int(self.config_dict["latent_channels"])
        if latent.shape[-1] != channels:
            if latent.shape[1] != channels:
                raise ValueError(f"cannot canonicalize audio latent {tuple(latent.shape)}")
            latent = latent.transpose(1, 2).contiguous()
        mean = torch.tensor(
            self.config_dict["latents_mean"],
        ).view(1, 1, channels)
        std = torch.tensor(
            self.config_dict["latents_std"],
        ).view(1, 1, channels)
        rows = ((latent - mean) / std).reshape(-1, channels)
        return rows.float(), int(latent.shape[1])

    @torch.inference_mode()
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if getattr(self, "encode_only", False):
            raise RuntimeError("MiniMax H3 encode-only audio VAE cannot decode latents")
        channels = int(self.config_dict["latent_channels"])
        mean = torch.tensor(
            self.config_dict["latents_mean"],
            device=latent.device,
            dtype=latent.dtype,
        ).view(1, channels, 1)
        std = torch.tensor(
            self.config_dict["latents_std"],
            device=latent.device,
            dtype=latent.dtype,
        ).view(1, channels, 1)
        waveform = self.remote.decode(latent * std + mean)
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError(f"unexpected decoded audio shape {tuple(waveform.shape)}")
        return waveform.permute(1, 0, 2).contiguous().float()


__all__ = ["MiniMaxH3AudioVAE", "MiniMaxH3VideoVAE"]
