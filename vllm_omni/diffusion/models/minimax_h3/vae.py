# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniMax H3 remote-code VAE adapters and exact latent contracts."""

from __future__ import annotations

import importlib
import json
import os
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path
from typing import Any

import numpy as np
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
from .vae_temporal import install_temporal_stream_patches

MINIMAX_H3_KEYFRAME_ENCODE_SEED = 42
MINIMAX_H3_AUDIO_SAMPLE_RATE = 32000
MINIMAX_H3_AUDIO_CHANNELS = 2

# Escape hatch back to the checkpoint's whole-video numpy preparation path.
_VAE_ENCODE_LEGACY_PREP_ENV = "VLLM_OMNI_VAE_ENCODE_LEGACY_PREP"


logger = init_logger(__name__)


@contextmanager
def _minimax_h3_keyframe_encode_context(
    device: torch.device,
) -> Iterator[None]:
    if device.type != "cuda":
        yield
        return

    # The official keyframe latent uses cuDNN's TF32 convolution path. The
    # default non-deterministic algorithm can select numerically different
    # reductions on H100s, and the difference is amplified by the denoiser.
    # Pin both the algorithm and math mode for this sensitive encode only.
    with torch.backends.cudnn.flags(
        enabled=True,
        benchmark=False,
        deterministic=True,
        allow_tf32=True,
    ):
        yield


def _legacy_encode_prep_enabled() -> bool:
    value = os.environ.get(_VAE_ENCODE_LEGACY_PREP_ENV, "0")
    return value.strip().lower() not in ("", "0", "false", "off")


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


class _VideoVAEPartProxy(nn.Module):
    """Residency proxy for one half of the split video VAE.

    The pipeline's ``_component_on_device`` drives whatever object it is
    given through ``load_to_device``/``offload_to_cpu``. The checkpoint's
    ViT decoder holds ~9GB of FP32 weights while the CNN encoder holds
    ~0.7GB, and each is used by exactly one direction, so routing those
    calls per half keeps the decoder off the device while reference videos
    encode and the encoder off while latents decode. ``object.__setattr__``
    keeps the back-reference out of the module registry (registering the
    adapter as a submodule would recurse through ``parameters()``).
    """

    def __init__(self, vae: MiniMaxH3VideoVAE, part: str) -> None:
        super().__init__()
        object.__setattr__(self, "_vae", vae)
        object.__setattr__(self, "_part", part)

    def load_to_device(self) -> None:
        self._vae._load_part_to_device(self._part)

    def offload_to_cpu(self) -> None:
        self._vae._offload_part_to_cpu(self._part)

    def set_omni_component_cache(self, cache: BoundedAllocatorCache | None) -> None:
        self._vae.set_omni_component_cache(cache)

    @property
    def sequential_offload_target(self) -> MiniMaxH3VideoVAE:
        """The module model-level CPU offload actually hooks.

        ``enable_omni_model_cpu_offload`` registers the sequential hook on the
        real ``video_vae``, and the hook moves the whole module through
        ``parameters()`` — entering the sequential context through this proxy
        would find no ``_hook_registry`` and raise, and whole-module movement
        has no half-residency benefit anyway. ``_component_on_device`` unwraps
        through this property before entering its sequential-offload branch;
        the manual per-half staging above stays proxy-driven, where split
        residency is the whole point.
        """
        return self._vae


class MiniMaxH3VideoVAE(nn.Module, DistributedVaeMixin):
    """Adapter around the checkpoint's native parallel-tiled video VAE."""

    def __init__(
        self,
        component_path: str,
        *,
        device: torch.device,
        load_device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self._device_target = device
        self.config_dict = _load_component_config(component_path)
        self.remote = _load_remote_component(
            component_path,
            self.config_dict,
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
        install_temporal_stream_patches(self.remote.model)
        self.model = self.remote.model
        self._stager = None
        self._encoder_stager = None
        self._decoder_stager = None
        if initial_device.type == "cpu" and device.type not in ("cpu", "meta"):
            self._build_residency_stagers(device)
        self.encoder_component = _VideoVAEPartProxy(self, "encoder")
        self.decoder_component = _VideoVAEPartProxy(self, "decoder")
        self.use_tiling = True
        self.use_slicing = False
        self.parallel_size = 1
        self.device_module = torch.get_device_module()

    def _build_residency_stagers(self, device: torch.device) -> None:
        """Stage the encode and decode halves of the remote separately.

        The encode path touches only ``encoder`` + ``quant_conv`` and the
        decode path only ``post_quant_conv`` + ``decoder`` (verified against
        the checkpoint's ``AutoencoderKLLegacy``: the one cross reference,
        ``tiled_decode``'s ``getattr(self.encoder, "mask_enabled")``, is
        short-circuited by ``self.training`` at inference). No storage is
        shared across the two groups, so independent staging is exact. When
        the checkpoint's structure is not discoverable, fall back to
        whole-module staging (the previous single-stager behavior).
        """
        part_names = ("encoder", "quant_conv", "post_quant_conv", "decoder")
        if all(isinstance(getattr(self.model, name, None), nn.Module) for name in part_names):
            self._encoder_stager = PinnedModuleStager(
                [self.model.encoder, self.model.quant_conv],
                device,
                pin_memory=True,
            )
            self._decoder_stager = PinnedModuleStager(
                [self.model.post_quant_conv, self.model.decoder],
                device,
                pin_memory=True,
            )
            return
        self._stager = PinnedModuleStager(
            self.remote,
            device,
            pin_memory=True,
        )

    def _part_stager(self, part: str) -> PinnedModuleStager | None:
        if part == "encoder":
            return self._encoder_stager
        if part == "decoder":
            return self._decoder_stager
        raise ValueError(f"unknown video VAE part {part!r}")

    def _load_part_to_device(self, part: str) -> None:
        stager = self._part_stager(part)
        if stager is not None:
            stager.load()
        elif self._stager is not None:
            self._stager.load()
        else:
            # No staged residency (the component lives on the device already
            # or is fully CPU-resident): fall back to whole-module placement.
            self.remote.to(self._device_target)

    def _offload_part_to_cpu(self, part: str) -> None:
        stager = self._part_stager(part)
        if stager is not None:
            stager.offload()
            return
        if self._stager is not None:
            self._stager.offload()
            return
        self.remote.to("cpu")
        self._release_component_cache()

    def _release_component_cache(self) -> None:
        cache = getattr(self, "_omni_component_cache", None)
        if cache is None:
            torch.accelerator.empty_cache()
        else:
            cache.release_if_needed()

    def load_to_device(self) -> None:
        if self._encoder_stager is not None:
            self._encoder_stager.load()
            self._decoder_stager.load()
        elif self._stager is not None:
            self._stager.load()
        else:
            self.remote.to(self._device_target)

    def set_omni_component_cache(self, cache: BoundedAllocatorCache | None) -> None:
        self._omni_component_cache = cache
        for stager in (self._encoder_stager, self._decoder_stager, self._stager):
            if stager is not None:
                stager.set_cache_retention(cache)

    def offload_to_cpu(self) -> None:
        if self._encoder_stager is not None:
            self._encoder_stager.offload()
            self._decoder_stager.offload()
        elif self._stager is not None:
            self._stager.offload()
        else:
            self.remote.to("cpu")
            self._release_component_cache()

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
                with _minimax_h3_keyframe_encode_context(parameter.device):
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

    def _stream_prepare_video_tensor(
        self,
        frames: np.ndarray,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Upload a uint8 ``(T, H, W, 3)`` video as one normalized FP32 tensor.

        The checkpoint's numpy path uploads the whole video as FP32, then
        ``transform_tensor`` materializes a second normalized copy, then
        ``encode_temporal`` pads through a full-video ``torch.cat``: three
        resident pixel-scale copies on the device. Here the padding is
        replicated on the host, a single preallocated ``(3, T', H, W)`` tensor
        receives clip-by-clip uploads, and the ``÷255 → (x-mean)/std`` chain
        runs in place on each clip's staging buffer in the same op order as
        ``convert_numpy_to_tensor`` → ``transform_tensor``, so peak device
        memory is the output tensor plus one clip instead of three copies.

        Returns ``None`` whenever the checkpoint contract this mirrors (uint8
        frames, ``clip_length`` alignment, processor ``transform`` constants)
        is not discoverable; callers fall back to the legacy path.
        """
        if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
            return None
        if int(frames.shape[0]) == 0:
            return None
        model = self.model
        clip_length = getattr(model, "clip_length", None)
        transform = getattr(getattr(model, "processor", None), "transform", None)
        mean = getattr(transform, "mean", None)
        std = getattr(transform, "std", None)
        if not isinstance(clip_length, int) or clip_length <= 0:
            return None
        if mean is None or std is None or len(mean) != 3 or len(std) != 3:
            return None
        # Mirror the checkpoint's temporal alignment so the device-side
        # ``get_suitable_video_length`` trim is a no-op and the
        # ``encode_temporal`` padding ``torch.cat`` never triggers. A
        # trim-stable length is ``k * clip_length + tail`` (tail = frame
        # overlap plus the isolated last frame); it also satisfies
        # encode_temporal only when ``tail % clip_length == offset_frame``.
        # With an asymmetric checkpoint (isolated last frame but no isolated
        # first frame) no trim-stable length can satisfy encode_temporal, so
        # the remote re-pads through a whole-video cat regardless and any
        # host-side pad would be trimmed away before that -- keep the legacy
        # frame count there instead of uploading dead frames.
        isolated_first_frame = bool(getattr(model, "isolated_first_frame", False))
        frame_pre_padding = int(getattr(model, "frame_pre_padding", 0) or 0)
        offset = 1 if isolated_first_frame and frame_pre_padding == 0 else 0
        processor = getattr(model, "processor", None)
        tail = int(getattr(processor, "frame_overlap", 0) or 0)
        if bool(getattr(processor, "isolated_last_frame", False)):
            tail += 1
        num_frames = int(frames.shape[0])
        pad = 0
        if tail % clip_length == offset:
            align = getattr(processor, "align_video_length", None)
            if callable(align):
                pad = max(0, int(align(num_frames, mode="pad", granularity="chunk")))
            else:
                chunks = -(-(num_frames - tail) // clip_length)
                pad = max(max(chunks, 1) * clip_length + tail - num_frames, 0)
        if pad:
            frames = np.concatenate([frames, np.repeat(frames[-1:], pad, axis=0)])
            num_frames += pad
        mean_t = torch.as_tensor(mean, dtype=torch.float32, device=device).view(3, 1, 1, 1)
        std_t = torch.as_tensor(std, dtype=torch.float32, device=device).view(3, 1, 1, 1)
        height, width = int(frames.shape[1]), int(frames.shape[2])
        out = torch.empty(
            (3, num_frames, height, width),
            dtype=torch.float32,
            device=device,
        )
        for start in range(0, num_frames, clip_length):
            end = min(start + clip_length, num_frames)
            # Frame slices of a C-contiguous (T, H, W, 3) array stay
            # contiguous, so each clip uploads at uint8 width (4x less host
            # traffic than the legacy FP32 upload) before the in-place
            # normalization chain. permute lands on the checkpoint's
            # (3, T, H, W) layout directly.
            chunk = torch.from_numpy(frames[start:end]).to(device=device)
            chunk = chunk.permute(3, 0, 1, 2).to(torch.float32).div_(255.0)
            chunk = chunk.sub_(mean_t).div_(std_t)
            out[:, start:end].copy_(chunk)
        return out

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
                prepared = None
                if isinstance(frames, np.ndarray) and not _legacy_encode_prep_enabled():
                    prepared = self._stream_prepare_video_tensor(frames, parameter.device)
                if prepared is not None:
                    # Tensor inputs already carry the checkpoint's expected
                    # (3, T, H, W) normalized-FP32 contract, so encode_videos
                    # skips its own convert/transform/pad whole-video copies.
                    frames = [prepared]
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
        if decoded.dtype == torch.uint8:
            # The streaming uint8 write-back already ran the revert and the
            # output quantizer inside write_part; nothing remains but the
            # shape contract.
            frames = decoded
        else:
            frames = self._revert_decoded_inplace(decoded)
        if frames.ndim == 4:
            frames = frames.unsqueeze(0).transpose(1, 2)
        if frames.ndim != 5:
            raise ValueError(f"unexpected decoded video shape {tuple(frames.shape)}")
        return frames if frames.dtype == torch.uint8 else frames.float()

    def _revert_decoded_inplace(self, decoded: torch.Tensor) -> torch.Tensor:
        """In-place counterpart of the processor's ``revert_tensor``.

        The checkpoint version materializes a denormalized copy, a clamped
        copy, and a contiguous copy of the whole decoded video -- three
        pixel-scale tensors resident at the decode peak. Decoding owns the
        tensor here, so the same op order (torchvision ``Normalize`` is
        ``(x - mean) / std``, then ``clamp(0, 1)``) runs in place on the
        original ``(B, C, T, H, W)`` layout, which is bit-identical
        elementwise to normalizing the ``(b t) c h w`` rearrangement and
        returns with zero whole-video copies.

        Falls back to ``processor.revert_tensor`` when the checkpoint's
        denormalization constants are not discoverable.
        """
        processor = getattr(self.model, "processor", None)
        transform_rev = getattr(processor, "transform_rev", None)
        mean = getattr(transform_rev, "mean", None)
        std = getattr(transform_rev, "std", None)
        if mean is None or std is None or len(mean) != 3 or len(std) != 3:
            return processor.revert_tensor(decoded)
        if bool(getattr(processor, "use_3d_conv", True)) and decoded.ndim == 4:
            decoded = decoded.unsqueeze(2)
        mean_t = torch.as_tensor(mean, dtype=decoded.dtype, device=decoded.device).view(1, 3, 1, 1, 1)
        std_t = torch.as_tensor(std, dtype=decoded.dtype, device=decoded.device).view(1, 3, 1, 1, 1)
        return decoded.sub_(mean_t).div_(std_t).clamp_(0.0, 1.0)


class MiniMaxH3AudioVAE(nn.Module):
    def __init__(
        self,
        component_path: str,
        *,
        device: torch.device,
        load_device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self._device_target = device
        self.config_dict = _load_component_config(component_path)
        self.remote = _load_remote_component(
            component_path,
            self.config_dict,
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
