# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniMax H3 remote-code VAE adapters and exact latent contracts."""

from __future__ import annotations

import importlib
import inspect
import json
import os
from collections.abc import Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
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
from vllm_omni.diffusion.models.interface import DecodedChunkConsumer
from vllm_omni.diffusion.offloader.module_residency import (
    BoundedAllocatorCache,
    PinnedModuleStager,
)

from .chunked_decode import decode_h3_chunks
from .ops import (
    H3VAEExactOpStatsSnapshot,
    install_h3_vae_optimizations,
    snapshot_h3_vae_exact_op_stats,
)
from .packed_tokens import minimax_h3_patchify_video_latent
from .vae_collectives import _agree_on_failure

MINIMAX_H3_KEYFRAME_ENCODE_SEED = 42
MINIMAX_H3_AUDIO_SAMPLE_RATE = 32000
MINIMAX_H3_AUDIO_CHANNELS = 2
MINIMAX_H3_VAE_DECODER_TILE_SIZE_ENV = "VLLM_OMNI_MINIMAX_H3_VAE_DECODER_TILE_SIZE"


logger = init_logger(__name__)


def resolve_minimax_h3_vae_decoder_tile_size(raw: str | None = None) -> int:
    """Resolve the experimental decoder tile size without silent fallback.

    Zero preserves the checkpoint configuration. A nonzero override must use
    the 16-pixel alignment required by H3's spatial VAE geometry.
    """

    if raw is None:
        raw = os.environ.get(MINIMAX_H3_VAE_DECODER_TILE_SIZE_ENV, "0")
    normalized = str(raw).strip()
    try:
        value = int(normalized, 10)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{MINIMAX_H3_VAE_DECODER_TILE_SIZE_ENV} must be 0 or a positive multiple of 16, got {raw!r}"
        ) from exc
    if value == 0:
        return 0
    if value < 0 or value % 16 != 0:
        raise ValueError(f"{MINIMAX_H3_VAE_DECODER_TILE_SIZE_ENV} must be 0 or a positive multiple of 16, got {raw!r}")
    return value


def _apply_minimax_h3_vae_decoder_tile_size_override(remote: nn.Module, tile_size: int) -> None:
    """Apply an already validated tile-size override to a loaded component."""

    if tile_size == 0:
        return
    model = getattr(remote, "model", None)
    if model is None:
        raise RuntimeError("MiniMax H3 video VAE tile-size override requires checkpoint attribute model")
    required = ("decoder_tile_size", "decoder_tile_overlap_min")
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise RuntimeError(
            "MiniMax H3 video VAE tile-size override requires checkpoint attribute(s): " + ", ".join(missing)
        )
    try:
        old_tile_size = int(model.decoder_tile_size)
        overlap = int(model.decoder_tile_overlap_min)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("MiniMax H3 checkpoint decoder tile geometry must be integer-valued") from exc
    if old_tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise RuntimeError(
            "MiniMax H3 video VAE tile-size override is incompatible with checkpoint geometry: "
            f"old_tile_size={old_tile_size}, new_tile_size={tile_size}, overlap={overlap}"
        )
    model.decoder_tile_size = tile_size
    logger.warning(
        "MiniMax H3 video VAE decoder tile-size override enabled: old=%d new=%d "
        "decoder_tile_overlap_min=%d (unchanged); output is numerical/non-bit-exact",
        old_tile_size,
        tile_size,
        overlap,
    )


@dataclass(frozen=True)
class _TemporalDecodePlan:
    """Exact released-H3 temporal layout used by the communication shortcut."""

    num_windows: int
    decoded_frames_per_window: int
    gathered_frames_per_window: int
    main_frames: int
    overlap_frames: int
    output_frames: int
    keep_ranges: tuple[tuple[int, int], ...]


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
        _apply_minimax_h3_vae_decoder_tile_size_override(
            self.remote,
            resolve_minimax_h3_vae_decoder_tile_size(),
        )
        # Match the reference loader contract before installing inference-only
        # decoder fast paths. Keyframe encoding remains FP32; decoder Linear
        # weights may be materialized in FP16 because reference decode casts
        # those same tensors through CUDA autocast on every tile.
        initial_device = load_device or device
        self.remote.eval().to(device=initial_device, dtype=torch.float32)
        decoder = getattr(self.remote.model, "decoder", None)
        from . import online_mxfp8

        mxfp8_enabled = online_mxfp8.enabled()
        if mxfp8_enabled and (decoder is None or initial_device.type != "cuda"):
            raise RuntimeError("MXFP8 arm requires the resident CUDA H3 video decoder")
        if decoder is not None:
            exact_ops_installed = install_h3_vae_optimizations(
                decoder,
                device=device,
                persist_fp16_weights=not mxfp8_enabled,
            )
            if mxfp8_enabled:
                if not exact_ops_installed:
                    raise RuntimeError("MXFP8 requires the current-best VAE operators")
                recipe = online_mxfp8.install(decoder)
                logger.info("H3_VAE_MXFP8_READY %s", json.dumps(recipe, sort_keys=True))
            if exact_ops_installed:
                rank = dist.get_rank() if dist.is_initialized() else 0
                blocks = getattr(decoder, "transformer_blocks", ())
                logger.info(
                    "MiniMax H3 VAE exact operators installed: rank=%d device=%s "
                    "blocks=%d linear_dtype=%s counters=enabled",
                    rank,
                    device,
                    len(blocks),
                    "mxfp8" if mxfp8_enabled else "float16",
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
    def decode_latent(self, latent: torch.Tensor, *, return_output: bool = True) -> torch.Tensor | None:
        with self._decode_tiling_context(latent):
            decoded = self.model.decode_base(self._denormalize_latent(latent))
        # All decoder collectives finish before non-output ranks leave.
        if not return_output:
            return None
        return self._normalize_decoded_frames(self.model.processor.revert_tensor(decoded))

    def _decode_tiling_context(self, latent: torch.Tensor) -> AbstractContextManager:
        """Pick the tiling mode a decode of ``latent`` can safely use.

        The checkpoint hands rank r the tiles ``range(r, num_tiles, sp_size)``
        and then rejects an empty share inside the gather. A rank with no
        tiles raises and leaves the collective while the others block in it
        forever, so too few tiles hangs the whole stage rather than failing
        it. Tile count depends only on the latent shape, so every rank takes
        this branch together.
        """
        num_tiles = self._decoder_tile_count(latent)
        if self.parallel_size > 1 and num_tiles < self.parallel_size:
            logger.warning_once(
                "MiniMax-H3 VAE decode splits into %d tile(s) but the tile group has "
                "%d ranks; decoding rank-locally for this shape instead, which is "
                "slower but avoids ranks without tiles hanging the collective.",
                num_tiles,
                self.parallel_size,
            )
            return self._rank_local_tiling()
        return nullcontext()

    def _denormalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        channels = int(self.config_dict["latent_channels"])
        mean = torch.tensor(self.config_dict["latents_mean"], device=latent.device, dtype=latent.dtype)
        std = torch.tensor(self.config_dict["latents_std"], device=latent.device, dtype=latent.dtype)
        shape = (1, channels, 1, 1, 1)
        return latent * std.view(shape) + mean.view(shape)

    @staticmethod
    def _normalize_decoded_frames(decoded: torch.Tensor) -> torch.Tensor:
        """Canonicalize remote H3 decoder output to [B,C,T,H,W]."""
        frames = decoded
        if frames.ndim == 4:
            frames = frames.unsqueeze(0).transpose(1, 2)
        if frames.ndim != 5:
            raise ValueError(f"unexpected decoded video shape {tuple(frames.shape)}")
        return frames.float()

    @torch.inference_mode()
    def decode_with_chunks(self, z: torch.Tensor, *, on_chunk: DecodedChunkConsumer) -> None:
        """Decode temporal clips and synchronously publish frames-only chunks.

        Implements :class:`SupportsChunkedVAEDecode`. Every rank participating
        in distributed VAE execution must invoke this method with a callback so
        the temporal collectives stay in lockstep; ``on_chunk`` is called only
        on the rank that owns output. Chunks arrive as ``[B, C, T, H, W]``
        float frames, normalized through the checkpoint's processor to match
        the complete decode path. After a callback failure, the remaining
        chunks are decoded and discarded before the exception is re-raised.
        """
        if not callable(on_chunk):
            raise TypeError("on_chunk must be callable")
        if not callable(getattr(self.model, "_adaptive_decode", None)):
            raise RuntimeError("Loaded MiniMax-H3 VAE does not expose temporal decode primitives")
        group = None
        if self.is_distributed_enabled():
            group = self._native_parallel_state().get("sp_process_group")
            if group is None or dist.get_world_size(group) != self.parallel_size:
                raise RuntimeError("MiniMax-H3 VAE chunk decode has an invalid spatial-parallel group")
        # Native H3 tiling performs its own collectives for every temporal clip,
        # so this path needs the same too-few-tiles fallback as the complete
        # decode: without it a shape that leaves some ranks tileless hangs the
        # gather instead of decoding rank-locally.
        with self._decode_tiling_context(z):
            decode_h3_chunks(self, z, on_chunk, group=group)

    def supports_chunked_output(self) -> bool:
        """Whether this checkpoint exposes the finalized-chunk callback hook."""

        try:
            parameters = inspect.signature(self.model.decode_base).parameters
        except (TypeError, ValueError):
            return False
        return "output_callback" in parameters

    def validate_chunked_output(self) -> None:
        """Validate the checkpoint and environment before PP8 decode starts."""

        if not self.supports_chunked_output():
            raise RuntimeError("MiniMax H3 checkpoint does not expose chunked VAE decode output")
        raw = (
            os.environ.get(
                "MINIMAX_H3_VAE_DECODER_STREAM_TEMPORAL_CAT",
                "1",
            )
            .strip()
            .lower()
        )
        if raw in {"0", "false", "no", "off", "disable", "disabled"}:
            raise RuntimeError(
                "MiniMax H3 finalized-chunk output requires MINIMAX_H3_VAE_DECODER_STREAM_TEMPORAL_CAT=1"
            )
        self._finalized_chunk_frame_capacity()

    def _finalized_chunk_frame_capacity(self) -> int:
        """Upper bound emitted by the checkpoint's temporal callback.

        ``decode_temporal`` slices each decoded clip at
        ``tokens_chunk_size * vae_ratio_t`` before removing padding. Isolated
        head/tail callbacks contain one frame, so the unsliced size is a safe
        bound for every callback and avoids sizing the ring from a tiny first
        chunk.
        """

        try:
            capacity = int(self.model.tokens_chunk_size) * int(self.model.vae_ratio_t)
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError("MiniMax H3 checkpoint does not expose its temporal chunk geometry") from exc
        if capacity < 1:
            raise RuntimeError(f"invalid MiniMax H3 temporal chunk capacity {capacity}")
        return capacity

    def _log_exact_op_chunked_decode(
        self,
        latent: torch.Tensor,
        *,
        num_tiles: int,
        route: str,
        before: H3VAEExactOpStatsSnapshot | None,
    ) -> None:
        """Emit one fail-closed-friendly dispatch summary per VAE rank.

        This is deliberately a delta around one decode request.  A cumulative
        process counter cannot distinguish the warmup from the measured arm,
        and an installation marker alone cannot prove that every PP8 rank hit
        the optimized kernels without a silent eager fallback.
        """

        decoder = getattr(self.model, "decoder", None)
        after = snapshot_h3_vae_exact_op_stats(decoder) if decoder is not None else None
        if before is None or after is None:
            return
        delta = after.delta(before)
        from . import online_mxfp8

        online_mxfp8.complete(decoder, dist.get_rank() if dist.is_initialized() else 0, logger)

        model = self.model
        latent_tokens = int(latent.shape[2])
        tokens_per_window = int(model.tokens_chunk_size)
        isolated_tokens = int(bool(model.isolated_first_frame) and int(model.frame_pre_padding) == 0) + int(
            bool(model.isolated_last_frame)
        )
        pseudo_tokens = latent_tokens - isolated_tokens + int(model.token_drop)
        remainder = pseudo_tokens % tokens_per_window
        if remainder:
            pseudo_tokens += tokens_per_window - remainder
        temporal_windows = pseudo_tokens // tokens_per_window - int(int(model.token_drop) > 0)

        state = self._native_parallel_state()
        sp_rank = int(state.get("sp_rank", 0))
        sp_size = int(state.get("sp_size", 1))
        if route == "temporal_window_pp":
            owned_windows = len(range(sp_rank, temporal_windows, sp_size))
            local_tiles = num_tiles
        else:
            owned_windows = temporal_windows
            if self.parallel_size > 1 and num_tiles >= self.parallel_size:
                local_tiles = len(range(sp_rank, num_tiles, sp_size))
            else:
                # ``_rank_local_tiling`` makes every rank own the full grid
                # when there are fewer spatial tiles than PP ranks.
                local_tiles = num_tiles

        stack_tiling = bool(getattr(model, "stack_tiling", False))
        decoder_calls_per_window = int(local_tiles > 0) if stack_tiling else local_tiles
        block_count = len(getattr(decoder, "transformer_blocks", ()))
        expected_block_calls = owned_windows * decoder_calls_per_window * block_count
        if route in ("temporal_spatial_pair_pp", "temporal_spatial_mixed_pp"):
            from .paired_vae import jobs

            assert (temporal_windows, num_tiles, sp_size) == (21, 28, 8) and not stack_tiling
            # Per-window ownership varies. -1 is an explicit sentinel, while
            # the exact total comes from the fixed, coverage-checked schedule.
            local_tiles = -1
            expected_block_calls = sum(len(row[sp_rank]) for row in jobs()) * block_count
            if route == "temporal_spatial_mixed_pp":
                from .mixed_vae import batch_plan

                expected_block_calls = len(batch_plan(sp_rank)) * block_count
        fallback_total = (
            delta.transformer_fallback
            + delta.qk_norm_rope_fallback
            + delta.swiglu_fallback
            + delta.scaled_residual_fallback
        )
        counts_ok = (
            delta.transformer_fast == expected_block_calls
            and delta.qk_norm_rope_fast == expected_block_calls
            and delta.swiglu_fast == expected_block_calls
            and delta.scaled_residual_fast == 2 * expected_block_calls
            and fallback_total == 0
        )
        global_rank = dist.get_rank() if dist.is_initialized() else 0
        logger.info(
            "MiniMax H3 VAE exact operators complete: rank=%d sp_rank=%d "
            "route=%s latent_tokens=%d temporal_windows=%d tiles=%d "
            "local_tiles=%d owned_windows=%d blocks=%d expected_block_calls=%d "
            "transformer_fast=%d transformer_fallback=%d "
            "qk_norm_rope_fast=%d qk_norm_rope_fallback=%d "
            "swiglu_fast=%d swiglu_fallback=%d "
            "scaled_residual_fast=%d scaled_residual_fallback=%d status=%s",
            global_rank,
            sp_rank,
            route,
            latent_tokens,
            temporal_windows,
            num_tiles,
            local_tiles,
            owned_windows,
            block_count,
            expected_block_calls,
            delta.transformer_fast,
            delta.transformer_fallback,
            delta.qk_norm_rope_fast,
            delta.qk_norm_rope_fallback,
            delta.swiglu_fast,
            delta.swiglu_fallback,
            delta.scaled_residual_fast,
            delta.scaled_residual_fallback,
            "ok" if counts_ok else "failed",
        )

    @staticmethod
    def _require_remote_signature(function: Any, name: str, parameters: tuple[str, ...]) -> None:
        try:
            actual = tuple(inspect.signature(function).parameters)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"cannot inspect MiniMax H3 checkpoint method {name}") from exc
        if actual != parameters:
            raise RuntimeError(
                f"MiniMax H3 checkpoint method {name} has unsupported signature {actual}; expected {parameters}"
            )

    def _temporal_decode_plan(
        self,
        latent: torch.Tensor,
        *,
        num_tiles: int,
        require_spatial_rank_share: bool = True,
    ) -> _TemporalDecodePlan:
        """Validate and describe the one released temporal layout we can prune.

        The optimization deliberately has no heuristic path. The released H3
        decoder produces 28 frames from every seven-token temporal window, then
        drops frames 0:3 and 20:23 before any cross-window blend. Spatial tile
        blending is frame-local, so gathering only the other 22 frames is exact.
        Any checkpoint/configuration drift is rejected before PP ranks enter the
        first tile collective.
        """

        if latent.ndim != 5 or int(latent.shape[0]) != 1:
            raise RuntimeError(
                "MiniMax H3 temporal pre-gather prune requires one rank-5 video latent, "
                f"got shape={tuple(latent.shape)}"
            )
        if self.parallel_size <= 1:
            raise RuntimeError("MiniMax H3 temporal pre-gather prune requires parallel VAE tiling")
        if require_spatial_rank_share and num_tiles < self.parallel_size:
            raise RuntimeError(
                "MiniMax H3 temporal pre-gather prune requires at least one decoder tile per rank: "
                f"tiles={num_tiles}, ranks={self.parallel_size}"
            )

        model = self.model
        expected_attributes: dict[str, Any] = {
            "decoder_tiling": True,
            "parallel_tiling": True,
            "use_3d_conv": True,
            "tokens_chunk_size": 5,
            "token_overlap": 2,
            "vae_ratio_t": 4,
            "frame_pre_padding": 3,
            "frame_overlap": 5,
            "token_drop": 3,
            "isolated_first_frame": False,
            "isolated_last_frame": False,
        }
        mismatches = []
        for name, expected in expected_attributes.items():
            actual = getattr(model, name, None)
            if actual != expected:
                mismatches.append(f"{name}={actual!r} (expected {expected!r})")
        if bool(getattr(model, "training", False)):
            mismatches.append("training=True (expected inference mode)")
        if mismatches:
            raise RuntimeError(
                "MiniMax H3 temporal pre-gather prune does not match this checkpoint plan: " + ", ".join(mismatches)
            )

        gather = getattr(model, "_all_gather_tiled_results", None)
        adaptive_decode = getattr(model, "_adaptive_decode", None)
        blend = getattr(model, "blend", None)
        output_plan = getattr(model, "_decode_temporal_output_frame_plan", None)
        temporal_dtype = getattr(
            importlib.import_module(model.__class__.__module__), "_resolve_temporal_cat_dtype", None
        )
        for function, name, parameters in (
            (gather, "_all_gather_tiled_results", ("tasks", "num_tiles")),
            (adaptive_decode, "_adaptive_decode", ("z",)),
            (blend, "blend", ("a", "b", "blend_extent", "dim")),
            (
                output_plan,
                "_decode_temporal_output_frame_plan",
                ("z", "z_head", "z_tail", "num_chunks", "pad_tokens"),
            ),
            (temporal_dtype, "_resolve_temporal_cat_dtype", ()),
        ):
            if not callable(function):
                raise RuntimeError(f"MiniMax H3 checkpoint is missing required method {name}")
            self._require_remote_signature(function, name, parameters)

        tokens_chunk_size = int(model.tokens_chunk_size)
        token_overlap = int(model.token_overlap)
        vae_ratio_t = int(model.vae_ratio_t)
        frame_pre_padding = int(model.frame_pre_padding)
        pseudo_total_tokens = int(latent.shape[2]) + int(model.token_drop)
        if pseudo_total_tokens % tokens_chunk_size:
            raise RuntimeError(
                "MiniMax H3 temporal pre-gather prune does not support a padded final token window: "
                f"latent_tokens={int(latent.shape[2])}, token_drop={int(model.token_drop)}"
            )
        num_windows = pseudo_total_tokens // tokens_chunk_size - 1
        if num_windows <= 0:
            raise RuntimeError(f"MiniMax H3 temporal pre-gather prune planned {num_windows} temporal windows")

        decoded_frames = (tokens_chunk_size + token_overlap) * vae_ratio_t
        chunk_dec = tokens_chunk_size * vae_ratio_t
        keep_ranges = tuple(
            (start + frame_pre_padding, min(start + chunk_dec, decoded_frames))
            for start in range(0, decoded_frames, chunk_dec)
        )
        if keep_ranges != ((3, 20), (23, 28)):
            raise RuntimeError(
                "MiniMax H3 temporal pre-gather prune derived an unsupported frame plan: "
                f"decoded_frames={decoded_frames}, keep_ranges={keep_ranges}"
            )
        main_frames = keep_ranges[0][1] - keep_ranges[0][0]
        overlap_frames = keep_ranges[1][1] - keep_ranges[1][0]
        gathered_frames = main_frames + overlap_frames
        output_frames = num_windows * main_frames + overlap_frames

        total, pad, checkpoint_output = output_plan(latent, None, None, num_windows, 0)
        if (int(total), int(pad), int(checkpoint_output)) != (output_frames, 0, output_frames):
            raise RuntimeError(
                "MiniMax H3 checkpoint temporal plan disagrees with pre-gather prune: "
                f"checkpoint={(int(total), int(pad), int(checkpoint_output))}, "
                f"expected={(output_frames, 0, output_frames)}"
            )
        return _TemporalDecodePlan(
            num_windows=num_windows,
            decoded_frames_per_window=decoded_frames,
            gathered_frames_per_window=gathered_frames,
            main_frames=main_frames,
            overlap_frames=overlap_frames,
            output_frames=output_frames,
            keep_ranges=keep_ranges,
        )

    def _paired_decode_plan(
        self,
        latent: torch.Tensor,
        *,
        num_tiles: int,
    ) -> _TemporalDecodePlan:
        """Validate the one canonical 107-token temporal-PP serving shape."""

        plan = self._temporal_decode_plan(
            latent,
            num_tiles=num_tiles,
            require_spatial_rank_share=False,
        )
        observed = (
            int(latent.shape[2]),
            plan.num_windows,
            plan.main_frames,
            plan.overlap_frames,
            plan.output_frames,
        )
        expected = (107, 21, 17, 5, 362)
        if observed != expected:
            raise RuntimeError(
                "MiniMax H3 temporal chunk parallel supports only the canonical "
                "107-token/21-window/17+5 plan: "
                f"observed={observed}, expected={expected}"
            )
        return plan

    @torch.inference_mode()
    def decode_latent_to_chunked_cpu_mp4(
        self,
        latent: torch.Tensor,
        *,
        height: int,
        width: int,
        fps: int,
        audio_sample_rate: int,
        video_codec_options: dict[str, str] | None,
        return_output: bool,
    ) -> Any | None:
        """Stream finalized decoder chunks into rank zero's CPU MP4 sink.

        The default/pruned routes enter the checkpoint-native spatial PP8
        decode on every rank. The explicit temporal route instead assigns
        complete windows round-robin and runs spatial tiles rank-locally.
        Only the designated output rank constructs a sink. Sink-construction
        and route-validation status is reduced before decode so a local error
        cannot strand peers in the first VAE data collective.
        """
        from .chunked_cpu_output import MiniMaxH3ChunkedCpuMp4Output

        sink = None
        paired_plan: _TemporalDecodePlan | None = None
        pair_stats = None
        gather_stream = None
        num_tiles = 0
        world = get_world_group() if dist.is_initialized() else None
        creation_error: BaseException | None = None
        try:
            self.validate_chunked_output()
            num_tiles = self._decoder_tile_count(latent)
            pair_flag = os.environ.get("VLLM_OMNI_H3_VAE_PAIR_PIPELINE", "0")
            if pair_flag not in ("0", "1"):
                raise ValueError("VLLM_OMNI_H3_VAE_PAIR_PIPELINE must be 0 or 1")
            pair_enabled = pair_flag == "1"
            mixed_flag = os.environ.get("VLLM_OMNI_H3_VAE_MIXED_BATCH", "0")
            if mixed_flag not in ("0", "1"):
                raise ValueError("VLLM_OMNI_H3_VAE_MIXED_BATCH must be 0 or 1")
            mixed_enabled = mixed_flag == "1"
            if mixed_enabled and (not pair_enabled):
                raise ValueError("Mixed VAE batching requires paired VAE scheduling")
            from .vae_gather_overlap import prepare_stream

            gather_stream = prepare_stream(latent.device, paired=pair_enabled, mixed=mixed_enabled, owner=self)
            if pair_enabled:
                if self.parallel_size <= 1 or world is None:
                    raise RuntimeError("MiniMax H3 temporal chunk parallel requires distributed VAE parallel_size > 1")
                if int(world.world_size) != int(self.parallel_size):
                    raise RuntimeError(
                        f"MiniMax H3 temporal chunk group size must match VAE parallel_size: "
                        f"group={world.world_size}, VAE={self.parallel_size}"
                    )
                expected_return_output = int(world.rank_in_group) == 0
                if bool(return_output) != expected_return_output:
                    raise RuntimeError(
                        f"MiniMax H3 temporal chunk output rank must be group rank zero: "
                        f"rank={world.rank_in_group}, return_output={return_output}"
                    )
                paired_plan = self._paired_decode_plan(latent, num_tiles=num_tiles)
            if return_output:
                sink = MiniMaxH3ChunkedCpuMp4Output(
                    width=width,
                    height=height,
                    fps=fps,
                    audio_sample_rate=audio_sample_rate,
                    max_chunk_frames=self._finalized_chunk_frame_capacity(),
                    device=latent.device,
                    video_codec_options=video_codec_options,
                )
        except BaseException as exc:
            creation_error = exc
        group = world.device_group if world is not None else None
        if self.parallel_size > 1 and group is not None:
            failed = torch.tensor([creation_error is not None], dtype=torch.int32, device=latent.device)
            dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=group)
            if bool(failed.item()):
                if sink is not None:
                    sink.abort()
                if creation_error is not None:
                    raise RuntimeError("failed to construct chunked CPU MP4 output sink") from creation_error
                raise RuntimeError("chunked CPU MP4 output sink failed on the output rank")
        elif creation_error is not None:
            raise RuntimeError("failed to construct chunked CPU MP4 output sink") from creation_error
        channels = int(self.config_dict["latent_channels"])
        mean = torch.tensor(self.config_dict["latents_mean"], device=latent.device, dtype=latent.dtype).view(
            1, channels, 1, 1, 1
        )
        std = torch.tensor(self.config_dict["latents_std"], device=latent.device, dtype=latent.dtype).view(
            1, channels, 1, 1, 1
        )
        if paired_plan is not None:
            tiling_context = self._rank_local_tiling()
        elif self.parallel_size > 1 and num_tiles < self.parallel_size:
            logger.warning_once(
                "MiniMax-H3 VAE decode splits into %d tile(s) but the tile group has %d ranks; "
                "decoding rank-locally for this shape instead.",
                num_tiles,
                self.parallel_size,
            )
            tiling_context: AbstractContextManager = self._rank_local_tiling()
        else:
            tiling_context = nullcontext()

        def consume(decoded: torch.Tensor, first_frame: int, total_frames: int) -> None:
            if sink is not None:
                sink.submit_decoded(decoded, first_frame, total_frames, self.model.processor)

        decoder = getattr(self.model, "decoder", None)
        exact_stats_before = snapshot_h3_vae_exact_op_stats(decoder) if decoder is not None else None
        exact_route = (
            ("temporal_spatial_mixed_pp" if mixed_enabled else "temporal_spatial_pair_pp")
            if pair_enabled
            else "temporal_window_pp"
            if paired_plan is not None
            else "spatial_pp_chunked"
        )
        try:
            with tiling_context:
                denormalized_latent = latent * std + mean
                if paired_plan is not None:
                    assert world is not None
                    temporal_cat_dtype = importlib.import_module(
                        self.model.__class__.__module__
                    )._resolve_temporal_cat_dtype()
                    from .paired_vae import decode_pairs

                    with ExitStack() as mixed_stack:
                        mixed_stats = None
                        if mixed_enabled:
                            setup_error = None
                            try:
                                from .mixed_vae import mixed_decode

                                mixed_stats = mixed_stack.enter_context(
                                    mixed_decode(self.model, denormalized_latent, world.rank_in_group)
                                )
                            except BaseException as error:
                                setup_error = error
                            if _agree_on_failure(world, denormalized_latent.device, setup_error is not None):
                                raise RuntimeError("Mixed VAE adapter setup failed") from setup_error
                        pair_stats = decode_pairs(
                            self.model,
                            denormalized_latent,
                            world,
                            consume if return_output else None,
                            temporal_cat_dtype=temporal_cat_dtype,
                            gather_stream=gather_stream,
                        )
                    if mixed_stats is not None:
                        pair_stats["mixed"] = mixed_stats
                    decoded = None
                else:
                    decoded = self.model.decode_base(denormalized_latent, output_callback=consume)
        except BaseException:
            if sink is not None:
                sink.abort()
            raise
        if decoded is not None:
            if sink is not None:
                sink.abort()
            raise RuntimeError("chunked MiniMax H3 VAE decode unexpectedly returned a full tensor")
        self._log_exact_op_chunked_decode(latent, num_tiles=num_tiles, route=exact_route, before=exact_stats_before)
        if pair_stats is not None:
            logger.info("H3_VAE_PAIR_COMPLETE %s", json.dumps(pair_stats, sort_keys=True))
        return sink


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
