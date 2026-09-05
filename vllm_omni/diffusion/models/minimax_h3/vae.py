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
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from vllm.logger import init_logger

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
    *,
    trust_remote_code: bool,
) -> nn.Module:
    auto_map = config.get("auto_map") or {}
    class_reference = auto_map.get("AutoModel")
    if not isinstance(class_reference, str):
        raise ValueError(f"{component_path}/config.json must define auto_map.AutoModel")
    if not trust_remote_code:
        raise ValueError(
            f"Loading {component_path} executes the modeling code shipped with "
            f"the checkpoint (auto_map.AutoModel = {class_reference}). Pass "
            "--trust-remote-code (or trust_remote_code=True) to allow it."
        )
    # ``trust_remote_code`` is checked here rather than forwarded to
    # ``get_class_from_dynamic_module``: that helper takes no such argument and
    # would silently absorb it into ``**kwargs``, so forwarding it would read
    # like a gate while executing the remote code unconditionally.
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
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__()
        self._device_target = device
        self.config_dict = _load_component_config(component_path)
        self.remote = _load_remote_component(
            component_path,
            self.config_dict,
            trust_remote_code=trust_remote_code,
        )
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
        self._tile_gather_buffers: dict[tuple[Any, ...], torch.Tensor] = {}
        self._tile_gather_stats = {"hits": 0, "allocs": 0, "bytes": 0}
        self._checkpoint_tile_gather = None
        if self._tile_gather_reuse_enabled():
            self._install_persistent_tile_gather()

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
    ) -> None:
        if mode != "tile":
            raise ValueError(f"MiniMax H3 VAE supports its native tile parallel mode only, got {mode!r}")
        group = get_world_group().device_group
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

    def _tile_gather_reuse_enabled(self) -> bool:
        """Whether this device needs the tiled-VAE gather buffer pinned.

        XPU only. The accumulation this guards against is an XCCL-side
        registration that is kept for every distinct receive-buffer address and
        never reclaimed; no other backend in tree does that, so everywhere else
        the checkpoint's own method is left in place and behaviour is unchanged.
        """

        return self._device_target.type == "xpu"

    def _install_persistent_tile_gather(self) -> None:
        """Route the checkpoint's tiled-VAE gather through a pinned buffer.

        The checkpoint's ``_all_gather_tiled_results`` allocates its gather
        output afresh on every call. Model-level offload calls ``empty_cache()``
        once per request, so the next request's output lands on a new device
        address; XCCL registers a non-reclaimable resource per new receive
        address, and the registrations accumulate until the card is full. The
        replacement below keeps one landing buffer per (shape, dtype, device)
        alive on this adapter, so the address the collective writes to is the
        same one every request.

        The override is bound on the checkpoint *instance*, not its class: the
        class is remote code shared by every component loaded from the same
        checkpoint, and only this adapter knows the device it runs on.
        """

        self._checkpoint_tile_gather = self.model._all_gather_tiled_results
        self.model._all_gather_tiled_results = self._persistent_tile_gather
        logger.info(
            "[H3_VAE_GATHER] persistent tile gather installed device=%s",
            self._device_target.type,
        )

    def _persistent_tile_gather(
        self,
        tasks: list[torch.Tensor],
        num_tiles: int,
    ) -> list[torch.Tensor]:
        """Equal-shape replacement for the checkpoint's tiled-result gather.

        Contract kept identical to the checkpoint's method: return a list of
        ``num_tiles`` tensors in global tile order, and raise on an empty local
        share so a rank that owns no tile cannot silently skip the collective.

        Tile ownership is round-robin (``range(sp_rank, num_tiles, sp_size)``),
        so every rank can compute every other rank's task count from
        ``num_tiles`` and ``sp_size`` alone. That makes the per-rank payloads
        equal once the leading task dimension is padded to ``max_tasks``, which
        is what lets a single ``all_gather_into_tensor`` into a pinned buffer
        replace the variable-shape gather.

        Returned tiles are cloned out of the buffer: the buffer is overwritten
        by the next call and callers hold the tiles past that point.
        """

        state = self._native_parallel_state()
        group = state["sp_process_group"]
        sp_size = int(state["sp_size"])
        sp_rank = int(state["sp_rank"])

        if not tasks:
            raise ValueError(f"Found empty tasks on sp rank {sp_rank}")

        max_tasks = -(-num_tiles // sp_size)
        if len(tasks) > max_tasks:
            raise ValueError(
                f"sp rank {sp_rank} holds {len(tasks)} tiles but round-robin "
                f"ownership of {num_tiles} tiles across {sp_size} ranks allows "
                f"at most {max_tasks}"
            )
        if len(tasks) == max_tasks:
            stacked = torch.stack(tasks, dim=0)
        else:
            # Pad the leading (task) dimension only. The padded slots belong to
            # ranks whose share is short by construction, and the unpacking loop
            # below never reads them back.
            stacked = tasks[0].new_empty((max_tasks, *tasks[0].shape))
            torch.stack(tasks, dim=0, out=stacked[: len(tasks)])
            stacked[len(tasks) :].zero_()

        key = (tuple(stacked.shape), stacked.dtype, str(stacked.device))
        buffer = self._tile_gather_buffers.get(key)
        if buffer is None:
            buffer = stacked.new_empty((sp_size, *stacked.shape))
            self._tile_gather_buffers[key] = buffer
            self._tile_gather_stats["allocs"] += 1
            self._tile_gather_stats["bytes"] += buffer.numel() * buffer.element_size()
            reuse = "alloc"
        else:
            self._tile_gather_stats["hits"] += 1
            reuse = "hit"
        # Quantities, not just presence: a line that only says "installed"
        # cannot distinguish a buffer that is being reused from one that is
        # reallocated every request, which is the whole failure being fixed.
        # Debug level, because this fires once per decoder tile batch (12 times
        # per request on the canonical 1344x768 geometry) and the one-shot
        # "installed" line above is what an operator needs at info level.
        logger.debug(
            "[H3_VAE_GATHER] reuse=%s key=%s/%s buf_ptr=0x%x hits=%d allocs=%d resident_mib=%.2f",
            reuse,
            tuple(stacked.shape),
            stacked.dtype,
            buffer.data_ptr(),
            self._tile_gather_stats["hits"],
            self._tile_gather_stats["allocs"],
            self._tile_gather_stats["bytes"] / (1024.0 * 1024.0),
        )
        dist.all_gather_into_tensor(buffer, stacked, group=group)

        results: list[torch.Tensor] = [None] * num_tiles  # type: ignore[list-item]
        for rank in range(sp_size):
            num_rank_tasks = -(-(num_tiles - rank) // sp_size)
            for k in range(num_rank_tasks):
                results[k * sp_size + rank] = buffer[rank][k].clone()
        return results

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

    @contextmanager
    def _rank_local_tiling(self) -> Iterator[None]:
        """Run one decode with tiling kept on this rank, then restore the group.

        Used only when there are fewer tiles than ranks. Every rank then decodes
        every tile, which is slower than sharing the work but is correct and
        involves no collective.
        """

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
        previous_parallel = self.model.parallel_tiling
        self.model.parallel_tiling = False
        parameter = next(self.parameters())
        previous_dtype = parameter.dtype
        if previous_dtype != torch.float32:
            self.to(torch.float32)
        devices = [parameter.device] if parameter.device.type != "cpu" else []
        try:
            with torch.random.fork_rng(devices=devices, device_type=parameter.device.type):
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
        parameter = next(self.parameters())
        previous_dtype = parameter.dtype
        if previous_dtype != torch.float32:
            self.to(torch.float32)
        devices = [parameter.device] if parameter.device.type != "cpu" else []
        try:
            with torch.random.fork_rng(devices=devices, device_type=parameter.device.type):
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
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__()
        self._device_target = device
        self.config_dict = _load_component_config(component_path)
        self.remote = _load_remote_component(
            component_path,
            self.config_dict,
            trust_remote_code=trust_remote_code,
        )
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
