# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniMax H3 remote-code VAE adapters and exact latent contracts."""

from __future__ import annotations

import importlib
import json
import math
import os
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
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
from vllm_omni.diffusion.models.interface import DecodedChunkConsumer
from vllm_omni.diffusion.offloader.module_residency import (
    BoundedAllocatorCache,
    PinnedModuleStager,
)
from vllm_omni.platforms import current_omni_platform

from .chunked_decode import decode_h3_chunks
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

    # Deterministic cuDNN algorithms can still use reduced-precision TF32.
    # On Hopper, TF32 can shift the keyframe latent enough to push I2VA below
    # the reference accuracy thresholds. Use FP32 deterministic convolutions
    # for this sensitive CUDA encode only. MIOpen has no deterministic
    # algorithm for the MI325 timeline-guide convolution, so applying the
    # cuDNN policy on ROCm raises ``miopenStatusUnknownError``.
    deterministic = current_omni_platform.is_cuda()
    allow_tf32 = True
    if deterministic:
        capability = current_omni_platform.get_device_capability(device.index or 0)
        allow_tf32 = capability is None or capability.major != 9
    with torch.backends.cudnn.flags(
        enabled=True,
        benchmark=False,
        deterministic=deterministic,
        allow_tf32=allow_tf32,
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


def _check_trust_remote_code(
    component_path: str,
    config: dict[str, Any],
    trust_remote_code: bool,
) -> None:
    """Apply the remote-code gate before either full or encoder-only loading."""

    class_reference = (config.get("auto_map") or {}).get("AutoModel")
    if isinstance(class_reference, str) and not trust_remote_code:
        raise ValueError(
            f"Loading {component_path} executes the modeling code shipped with "
            f"the checkpoint (auto_map.AutoModel = {class_reference}). Pass "
            "--trust-remote-code (or trust_remote_code=True) to allow it."
        )


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
        self._jit_executor = torch.jit.optimized_execution(False)
        self._jit_executor.__enter__()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._jit_executor.__exit__(exc_type, exc, traceback)
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

    decode_dtype: torch.dtype = torch.float16
    _pixel_transform: Callable[[torch.Tensor], torch.Tensor] | None = None

    def __init__(
        self,
        component_path: str,
        *,
        device: torch.device,
        load_device: torch.device | None = None,
        encode_only: bool = False,
        decode_only: bool = False,
        trust_remote_code: bool = False,
        decoder_source: str | Mapping[str, str] | None = None,
        lightvae_decoder_path: str | None = None,
    ) -> None:
        super().__init__()
        self._device_target = device
        self.encode_only = bool(encode_only)
        self.decode_only = bool(decode_only)
        if self.encode_only and self.decode_only:
            raise ValueError("MiniMax H3 video VAE cannot be both encode-only and decode-only")
        if lightvae_decoder_path is not None and decoder_source is not None:
            raise ValueError("lightvae_decoder_path cannot be combined with video_vae_decoder")
        from .vae_nvfp4 import (
            install_nvfp4_video_decoder,
            released_decoder_pixels,
            resolve_video_decoder_checkpoint,
        )

        decoder_path = resolve_video_decoder_checkpoint(decoder_source, device)
        self._pixel_transform = released_decoder_pixels if decoder_path is not None else None
        if self.encode_only and decoder_path is not None:
            raise ValueError("An encode-only VAE cannot load a replacement video decoder")
        self.config_dict = _load_component_config(component_path)
        _check_trust_remote_code(component_path, self.config_dict, trust_remote_code)
        if self.encode_only:
            self.remote = _load_video_vae_encoder(component_path, self.config_dict)
        else:
            self.remote = _load_remote_component(component_path, self.config_dict, trust_remote_code=trust_remote_code)
            if self.decode_only:
                _remove_modules(self.remote.model, ("encoder", "quant_conv"))
            if lightvae_decoder_path is not None:
                from .lightvae import install_h3_lite_decoder, read_h3_lite_decoder_checkpoint

                checkpoint = read_h3_lite_decoder_checkpoint(lightvae_decoder_path)
                install_h3_lite_decoder(self.remote.model, checkpoint, native_config=self.config_dict)
                logger.info(
                    "Loaded H3 Lite video decoder: path=%s layers=%d device=%s",
                    lightvae_decoder_path,
                    len(checkpoint.kept_layers),
                    device,
                )
                del checkpoint
        # Match the reference loader contract before installing inference-only
        # decoder fast paths. Keyframe encoding remains FP32; decoder Linear
        # weights may be materialized in FP16 because reference decode casts
        # those same tensors through CUDA autocast on every tile.
        initial_device = load_device or device
        weight_device = torch.device("cpu") if decoder_path is not None else initial_device
        self.remote.eval().to(device=weight_device, dtype=torch.float32)
        if decoder_path is not None:
            install_nvfp4_video_decoder(self.remote.model, decoder_path)
            self.decode_dtype = torch.bfloat16
            self.remote.to(device=initial_device)
        decoder = getattr(self.remote.model, "decoder", None)
        if decoder is not None and decoder_path is None:
            install_h3_vae_optimizations(
                decoder,
                device=device,
            )
        install_temporal_stream_patches(self.remote.model, pixel_transform=self._pixel_transform)
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
        self._tile_gather_workspace: torch.Tensor | None = None
        self._tile_gather_stats = {"hits": 0, "allocs": 0, "workspace_bytes": 0}
        self._checkpoint_tile_gather = None
        if self._tile_gather_reuse_enabled():
            self._install_persistent_tile_gather()

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

    def _tile_gather_reuse_enabled(self) -> bool:
        """Whether this device needs the tiled-VAE gather address kept stable.

        XPU only. The accumulation this guards against is an XCCL-side
        registration that is kept for every distinct receive-buffer address and
        never reclaimed; no other backend in tree does that, so everywhere else
        the checkpoint's own method is left in place and behaviour is unchanged.
        The stable address comes from one grow-only byte workspace, so what the
        adapter holds resident is one buffer for the largest geometry it has
        seen rather than one per geometry.
        """

        return self._device_target.type == "xpu"

    def _install_persistent_tile_gather(self) -> None:
        """Route the checkpoint's tiled-VAE gather through a bounded workspace.

        The checkpoint's ``_all_gather_tiled_results`` allocates its gather
        output afresh on every call. Model-level offload calls ``empty_cache()``
        once per request, so the next request's output lands on a new device
        address; XCCL registers a non-reclaimable resource per new receive
        address, and the registrations accumulate until the card is full. The
        replacement below keeps a single byte workspace alive on this adapter
        and carves every gather out of its front, so the address the collective
        writes to is the same one every request.

        The workspace only ever grows: a geometry that fits in the current
        block reuses it as is, and a larger one replaces the block, which
        drops the previous allocation before allocating its own. Resident
        memory is therefore bounded by one buffer for the largest geometry the
        adapter has seen, not by one buffer per distinct geometry.

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
        is what lets a single ``all_gather_into_tensor`` into a stable buffer
        replace the variable-shape gather.

        The landing buffer is a view onto the front of one byte workspace that
        this adapter keeps alive and only ever grows, so its address stays
        stable across requests while resident memory stays bounded by the
        largest geometry seen rather than growing per geometry. A byte
        workspace is dtype-agnostic, so mixed-precision requests share it too.

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

        need_bytes = sp_size * stacked.numel() * stacked.element_size()
        workspace = self._tile_gather_workspace
        if workspace is None or workspace.device != stacked.device or workspace.numel() < need_bytes:
            # Drop the previous block before allocating its replacement so the
            # two are never resident at once; only one workspace is ever held.
            workspace = None
            self._tile_gather_workspace = None
            workspace = torch.empty(need_bytes, dtype=torch.uint8, device=stacked.device)
            self._tile_gather_workspace = workspace
            self._tile_gather_stats["allocs"] += 1
            self._tile_gather_stats["workspace_bytes"] = need_bytes
            reuse = "alloc"
        else:
            self._tile_gather_stats["hits"] += 1
            reuse = "hit"
        buffer = workspace[:need_bytes].view(stacked.dtype).view(sp_size, *stacked.shape)
        # Quantities, not just presence: a line that only says "installed"
        # cannot distinguish a buffer that is being reused from one that is
        # reallocated every request, which is the whole failure being fixed.
        # Debug level, because this fires once per decoder tile batch (12 times
        # per request on the canonical 1344x768 geometry) and the one-shot
        # "installed" line above is what an operator needs at info level.
        logger.debug(
            "[H3_VAE_GATHER] reuse=%s shape=%s/%s need_mib=%.2f workspace_mib=%.2f buf_ptr=0x%x hits=%d allocs=%d",
            reuse,
            tuple(stacked.shape),
            stacked.dtype,
            need_bytes / (1024.0 * 1024.0),
            self._tile_gather_stats["workspace_bytes"] / (1024.0 * 1024.0),
            workspace.data_ptr(),
            self._tile_gather_stats["hits"],
            self._tile_gather_stats["allocs"],
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

    def _stream_video_prep_metadata(
        self,
        frames: np.ndarray,
    ) -> tuple[int, int, Any, Any] | None:
        """Return the shared stream-prep clip, pad, mean, and std metadata."""
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
        return clip_length, pad, mean, std

    def predict_video_latent_shape(self, frames: np.ndarray) -> tuple[int, int, int]:
        """Predict the checkpoint latent shape for its reference-video input."""
        effective_frames = int(frames.shape[0])
        if not _legacy_encode_prep_enabled():
            prep_metadata = self._stream_video_prep_metadata(frames)
            if prep_metadata is not None:
                effective_frames += prep_metadata[1]

        processor = self.model.processor
        latent_t = int(processor.get_latent_length(effective_frames))
        height, width = processor._align_to_total_patch_size(int(frames.shape[1]), int(frames.shape[2]))
        ratio = int(self.model.vae_ratio)
        return latent_t, int(height // ratio), int(width // ratio)

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
        prep_metadata = self._stream_video_prep_metadata(frames)
        if prep_metadata is None:
            return None
        clip_length, pad, mean, std = prep_metadata
        num_frames = int(frames.shape[0])
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
    def encode_control_latents(self, pixels: torch.Tensor) -> torch.Tensor:
        """Deterministic Fun conditioning: normalized posterior mode, not a sample.

        The native temporal encoder returns the same moments as VideoX-Fun's
        VAE._encode: per-clip spatial tiling, final-frame padding, token_drop.
        Its first 24 channels are the diagonal Gaussian mean.
        Match VideoX-Fun control/inpaint encoding: normalize on-device in FP32,
        then encode under FP16 autocast rather than the keyframe sampling path.
        """
        parameter = next(self.parameters())
        pixels = pixels.to(device=parameter.device, dtype=torch.float32)
        mean = pixels.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1, 1)
        std = pixels.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1, 1)
        previous_parallel = self.model.parallel_tiling
        if int(getattr(self, "parallel_size", 1)) <= 1:
            self.model.parallel_tiling = False
        try:
            with (
                self._encoder_tiling_context(int(pixels.shape[-2]), int(pixels.shape[-1])),
                current_omni_platform.create_autocast_context(
                    device_type=parameter.device.type,
                    dtype=torch.float16,
                    enabled=parameter.device.type != "cpu",
                ),
            ):
                moments = self.model.encode_temporal((pixels - mean) / std)
        finally:
            self.model.parallel_tiling = previous_parallel
        channels = int(self.config_dict["latent_channels"])
        if moments.shape[1] != 2 * channels:
            raise ValueError("H3 control encoder must return mean and log-variance channels")
        latent = moments[:, :channels].float()
        mean = latent.new_tensor(self.config_dict["latents_mean"]).view(1, channels, 1, 1, 1)
        std = latent.new_tensor(self.config_dict["latents_std"]).view(1, channels, 1, 1, 1)
        return (latent - mean) / std

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

    @torch.inference_mode()
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if getattr(self, "encode_only", False):
            raise RuntimeError("MiniMax H3 encode-only video VAE cannot decode latents")
        with self._decode_tiling_context(latent):
            decoded = self.model.decode_base(self._denormalize_latent(latent))
        if decoded.dtype == torch.uint8:
            # The streaming uint8 write-back already ran the revert and the
            # output quantizer inside write_part; nothing remains but the
            # shape contract.
            return self._normalize_decoded_frames(decoded)
        return self._normalize_decoded_frames(self._revert_decoded_inplace(decoded))

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
        return frames if frames.dtype == torch.uint8 else frames.float()

    # Chunks are reverted through the checkpoint's processor, which
    # denormalizes and clamps into the unit interval.
    chunk_value_range: ClassVar[tuple[float, float]] = (0.0, 1.0)

    @property
    def owns_chunk_output(self) -> bool:
        """Whether this rank publishes frames to the streaming MP4 consumer."""
        if not self.is_distributed_enabled():
            return True
        group = self._native_parallel_state().get("sp_process_group")
        if group is None:
            raise RuntimeError("MiniMax-H3 VAE chunk decode has no spatial-parallel group")
        return dist.get_rank(group) == 0

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
        if getattr(self, "encode_only", False):
            raise RuntimeError("MiniMax H3 encode-only video VAE cannot decode latents")
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

    def _revert_decoded(self, decoded: torch.Tensor) -> torch.Tensor:
        """Preserve chunk storage that the temporal decoder may still use."""
        if self._pixel_transform is not None:
            return self._pixel_transform(decoded)
        processor = getattr(self.model, "processor", None)
        return decoded if processor is None else processor.revert_tensor(decoded)

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
        if self._pixel_transform is not None:
            return self._pixel_transform(decoded)
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
        encode_only: bool = False,
        decode_only: bool = False,
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__()
        self._device_target = device
        self.encode_only = bool(encode_only)
        self.decode_only = bool(decode_only)
        if self.encode_only and self.decode_only:
            raise ValueError("MiniMax H3 audio VAE cannot be both encode-only and decode-only")
        self.config_dict = _load_component_config(component_path)
        _check_trust_remote_code(component_path, self.config_dict, trust_remote_code)
        if self.encode_only:
            self.remote = _load_audio_vae_encoder(component_path, self.config_dict)
        else:
            self.remote = _load_remote_component(component_path, self.config_dict, trust_remote_code=trust_remote_code)
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
        # The remote decoder's TorchScript activations round differently on
        # their first (unfused) call; keep every request on the same path.
        with torch.jit.optimized_execution(False):
            waveform = self.remote.decode(latent * std + mean)
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError(f"unexpected decoded audio shape {tuple(waveform.shape)}")
        return waveform.permute(1, 0, 2).contiguous().float()


def _light_vae_opt_enabled() -> bool:
    """Single switch for the Light VAE decode optimizations.

    When enabled (default), the Light VAE keeps its weights resident and uses
    8-rank window balancing. Set ``MINIMAX_H3_LIGHT_VAE_OPT=0`` to run the
    Light VAE without either of them (for A/B comparisons).
    """
    return os.environ.get("MINIMAX_H3_LIGHT_VAE_OPT", "1").strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


class MiniMaxH3LightVideoVAE(nn.Module, DistributedVaeMixin):
    """Lightweight decode-only MiniMax-H3 video VAE.

    Wraps the diffusers :class:`AutoencoderKLMiniMaxH3` release (unchanged H3
    encoder + distilled 26-block video decoder) and exposes the same
    ``decode_latent`` contract as :class:`MiniMaxH3VideoVAE`, so the H3 pipeline
    can swap only the decode path while keeping the official VAE for reference
    encoding. Latents are de-normalized with ``latents_mean``/``latents_std``
    here (the diffusers ``decode`` consumes raw latents), and the decoded
    ``(B, C, T, H, W)`` output is returned directly (the pipeline converts it to
    the channel-last layout itself).

    Decoding is tile-parallel across the SP group: the spatial tile grid is
    distributed over the ranks (:class:`DistributedVaeExecutor`) and stitched
    back on rank 0, so every rank decodes roughly ``1/N`` of the tiles instead of
    the full video.
    """

    def __init__(
        self,
        component_path: str,
        *,
        device: torch.device,
        load_device: torch.device | None = None,
    ) -> None:
        super().__init__()
        from diffusers import AutoencoderKLMiniMaxH3

        self._device_target = device
        self.config_dict = _load_component_config(component_path)
        channels = int(self.config_dict["latent_channels"])
        self._channels = channels
        self.register_buffer(
            "_latents_mean",
            torch.tensor(self.config_dict["latents_mean"], dtype=torch.float32).view(1, channels, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_latents_std",
            torch.tensor(self.config_dict["latents_std"], dtype=torch.float32).view(1, channels, 1, 1, 1),
            persistent=False,
        )
        initial_device = load_device or device
        model = AutoencoderKLMiniMaxH3.from_pretrained(
            component_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        model.eval()
        # Decode is not guaranteed to run under an autocast context (for
        # example in quantized deployments), so normalize every floating
        # submodule to fp16. Otherwise a residual fp32 Conv3d such as
        # ``post_quant_conv`` would mix with the fp16 inputs.
        model.to(torch.float16)
        _attn_backend = os.environ.get("MINIMAX_H3_LIGHT_VAE_ATTN", "off").strip().lower()
        if _attn_backend and _attn_backend not in ("off", "none", "0", "native"):
            try:
                model.set_attention_backend(_attn_backend)
                logger.info("MiniMax-H3 light VAE: attention backend=%s", _attn_backend)
            except Exception as exc:  # pragma: no cover
                logger.warning_once("MiniMax-H3 light VAE: set_attention_backend(%s) failed: %s", _attn_backend, exc)
        try:
            model.enable_tiling()
        except Exception:  # pragma: no cover
            logger.warning_once("MiniMax-H3 light VAE: enable_tiling() unavailable")
        model.to(initial_device)
        self.model = model
        self._stager = None
        if initial_device.type == "cpu" and device.type not in ("cpu", "meta"):
            self._stager = PinnedModuleStager(model, device, pin_memory=True)
        self.use_tiling = True
        try:
            self._light_settings = json.loads((Path(component_path) / "decode_config.json").read_text())
        except Exception:
            self._light_settings = {}
        _tile_env = os.environ.get("MINIMAX_H3_LIGHT_VAE_TILE", "").strip().lower()
        self._tile_off = _tile_env in ("off", "none", "0")
        self._tile_layout = "adaptive"
        self._tile_explicit = None
        if _tile_env and not self._tile_off:
            if _tile_env in ("native", "adaptive"):
                self._tile_layout = _tile_env
            elif "," in _tile_env:
                try:
                    self._tile_explicit = tuple(int(v) for v in _tile_env.split(","))
                except Exception:
                    self._tile_explicit = None
        self.init_distributed()
        self._resident = _light_vae_opt_enabled()
        self._on_device = self._stager is None and initial_device.type == device.type
        if self._resident:
            if self._stager is not None:
                self._stager.load()
            elif device.type != "cpu":
                self.model.to(self._device_target)
            self._on_device = device.type != "cpu"

    def load_to_device(self) -> None:
        if self._on_device:
            return
        if self._stager is not None:
            self._stager.load()
        else:
            self.model.to(self._device_target)
        self._on_device = True

    def offload_to_cpu(self) -> None:
        if self._resident:
            return
        if not self._on_device or self._device_target.type == "cpu":
            return
        if self._stager is not None:
            self._stager.offload()
        else:
            self.model.to("cpu")
        self._on_device = False

    def set_omni_component_cache(self, cache: Any) -> None:
        self._omni_component_cache = cache
        if self._stager is not None:
            self._stager.set_cache_retention(cache)

    @staticmethod
    def _profile_enabled() -> bool:
        return os.environ.get("MINIMAX_H3_VAE_PROFILE", "0").strip().lower() in ("1", "true", "yes")

    @staticmethod
    def _profile_stamp(enabled: bool) -> float:
        if not enabled:
            return 0.0
        torch.accelerator.synchronize()
        return time.perf_counter()

    # -- batched, tile-parallel decode --------------------------------------
    def _tile_geometry(self, frames: int, height: int, width: int):
        m = self.model
        if self._tile_explicit is not None:
            return self._tile_explicit
        if self._tile_layout == "native":
            return (
                m.tile_sample_min_height,
                m.tile_sample_min_width,
                m.tile_sample_min_overlap_height,
                m.tile_sample_min_overlap_width,
            )
        s = self._light_settings
        if frames == 1:
            return (
                s["image_tile_size"],
                s["image_long_axis_tile"],
                s["student_tile_overlap"],
                s["student_tile_overlap"],
            )
        short, long = s["student_tile_size"], s["student_long_axis_tile"]
        ov_short, ov_long = s["student_short_axis_overlap"], s["student_long_axis_overlap"]
        if height < width:
            return short, long, ov_short, ov_long
        if height > width:
            return long, short, ov_long, ov_short
        ov = max(ov_short, ov_long)
        return short, short, ov, ov

    def _spatial_plan(self, latent_t: int, height: int, width: int):
        """Return the spatial tile grid, or one full-frame tile when disabled."""
        if not self.use_tiling or self._tile_off:
            return [0], [height], [], [0], [width], []
        th, tw, oh, ow = self._tile_geometry(latent_t, height, width)
        ys, hs, y_ov = self._split_tiles(height, th, oh)
        xs, ws, x_ov = self._split_tiles(width, tw, ow)
        return ys, hs, y_ov, xs, ws, x_ov

    def _split_tiles(self, length: int, size: int, overlap: int):
        m = self.model
        if self._tile_layout == "native" and self._tile_explicit is None:
            return m._split_tiles(length, size, overlap)
        # adaptive (trained LightX2V) splitter
        if length <= size + overlap:
            return [0], [length], []
        ratio = int(m.spatial_compression_ratio)
        count = math.ceil(length / size)
        required = math.ceil((length + overlap * (count - 1)) / count / ratio) * ratio
        if required > size + overlap:
            required = size
        size = max(size, required)
        while size * count - overlap * (count - 1) < length:
            count += 1
        overlaps = [overlap] * (count - 1)
        remaining = size * count - sum(overlaps) - length
        for i in range(remaining // ratio):
            overlaps[i % (count - 1)] += ratio
        starts = [0]
        for value in overlaps:
            starts.append(starts[-1] + size - value)
        return starts, [size] * count, overlaps

    def _stitch_tiles(self, tiles, height_overlaps, width_overlaps):
        # Match the reference blend order but write directly into one canvas to
        # avoid repeatedly copying complete rows and videos through torch.cat.
        sample = tiles[0][0]
        row_heights = [
            row[0].shape[-2] - (height_overlaps[i] if i < len(tiles) - 1 else 0) for i, row in enumerate(tiles)
        ]
        column_widths = [
            tile.shape[-1] - (width_overlaps[j] if j < len(tiles[0]) - 1 else 0) for j, tile in enumerate(tiles[0])
        ]
        canvas = sample.new_empty((*sample.shape[:-2], sum(row_heights), sum(column_widths)))
        out_y = 0
        for i, row in enumerate(tiles):
            out_x = 0
            for j, tile in enumerate(row):
                if i and height_overlaps[i - 1] > 0:
                    tile = self.model._blend(tiles[i - 1][j], tile, height_overlaps[i - 1], dim=-2)
                if j and width_overlaps[j - 1] > 0:
                    tile = self.model._blend(row[j - 1], tile, width_overlaps[j - 1], dim=-1)
                if i < len(tiles) - 1 and height_overlaps[i] > 0:
                    tile = tile[..., : -height_overlaps[i], :]
                if j < len(row) - 1 and width_overlaps[j] > 0:
                    tile = tile[..., :, : -width_overlaps[j]]
                canvas[..., out_y : out_y + tile.shape[-2], out_x : out_x + tile.shape[-1]].copy_(tile)
                out_x += tile.shape[-1]
            out_y += row_heights[i]
        return canvas

    def _stitch_gathered(self, gathered, ys, xs, y_ov, x_ov):
        rows = [[gathered[i * len(xs) + j] for j in range(len(xs))] for i in range(len(ys))]
        return self._stitch_tiles(rows, y_ov, x_ov)

    def _decode_parallel_state(self):
        """Resolve the configured VAE degree and the group that decodes."""
        world_group = self.distributed_executor.group
        world_size = dist.get_world_size(world_group)
        configured = int(getattr(self.distributed_executor, "parallel_size", 1) or 1)
        size = min(max(configured, 1), world_size)
        if size <= 1 or size >= world_size:
            return world_group, size, world_group, world_size, dist.get_rank(world_group)
        group = getattr(self, "_decode_group", None)
        if group is None:
            group = dist.new_group(ranks=list(range(size)))
            self._decode_group = group
        return group, size, world_group, world_size, dist.get_rank(group)

    def _broadcast_decoded_video(self, output, world_group):
        """Share the stitched video from decode rank 0 with idle ranks."""
        shape = torch.zeros(5, dtype=torch.int64, device=self._device_target)
        if output is not None:
            shape[: output.ndim] = torch.tensor(output.shape, dtype=torch.int64, device=self._device_target)
        dist.broadcast(shape, src=0, group=world_group)
        if output is None:
            dtype = next(self.model.decoder.parameters()).dtype
            output = torch.empty(tuple(shape.tolist()), dtype=dtype, device=self._device_target)
        dist.broadcast(output, src=0, group=world_group)
        return output

    def _gather_tiles(self, dec, local, n, group):
        size = dist.get_world_size(group)
        counts = [len(range(r, n, size)) for r in range(size)]
        maxc = max(counts)
        with self.distributed_executor.sync_errors("light VAE tile gather allocation"):
            if dec.shape[0] != counts[dist.get_rank(group)]:
                raise RuntimeError(
                    f"local Light VAE tile count changed: got {dec.shape[0]}, expected {counts[dist.get_rank(group)]}"
                )
            if dec.shape[0] < maxc:
                pad = dec.new_zeros((maxc - dec.shape[0],) + tuple(dec.shape[1:]))
                dec = torch.cat([dec, pad], dim=0)
            dec = dec.contiguous()
            storage = torch.empty((size, *dec.shape), device=dec.device, dtype=dec.dtype)
            buffers = [storage[r] for r in range(size)]
        dist.all_gather(buffers, dec, group=group)
        out = [None] * n
        for r in range(size):
            for k, i in enumerate(range(r, n, size)):
                out[i] = storage[r, k : k + 1]
        return out

    @staticmethod
    def _window_balance_enabled() -> bool:
        return _light_vae_opt_enabled()

    @staticmethod
    def _window_batch_cap() -> int:
        try:
            return max(1, int(os.environ.get("MINIMAX_H3_LIGHT_VAE_WINDOW_BATCH", "6")))
        except Exception:
            return 6

    def _sync_decode_errors(self, stage: str, distributed: bool):
        if not distributed:
            return nullcontext()
        return self.distributed_executor.sync_errors(stage)

    def _temporal_plan(self, latent_t: int):
        m = self.model
        tcs = int(m.tokens_chunk_size)
        drop = int(m.config.token_drop)
        tratio = int(m.temporal_compression_ratio)
        overlap = int(m.token_overlap)
        num_tokens = latent_t + drop
        pad = (-num_tokens) % tcs
        num_chunks = (num_tokens + pad) // tcs - int(drop > 0)
        if num_chunks <= 0:
            return None
        padded_t = num_tokens + pad
        for w in range(num_chunks):
            if min(tcs + overlap, padded_t - w * tcs) != tcs + overlap:
                return None
        return dict(
            T=latent_t,
            tcs=tcs,
            drop=drop,
            tratio=tratio,
            overlap=overlap,
            chunk_num_frames=tcs * tratio,
            num_chunks=num_chunks,
            pad_tokens=pad,
            frame_pre_padding=int(m.frame_pre_padding),
            frame_overlap=int(m.frame_overlap),
            intra_tail=int(m.config.clip_length % tratio),
        )

    @staticmethod
    def _pad_temporal(z: torch.Tensor, plan) -> torch.Tensor:
        if plan["pad_tokens"] > 0:
            return torch.cat([z, z[:, :, -1:].repeat(1, 1, plan["pad_tokens"], 1, 1)], dim=2)
        return z

    @staticmethod
    def _window_latent(zpad: torch.Tensor, ranges, tile_id: int, window: int, plan) -> torch.Tensor:
        y0, y1, x0, x1 = ranges[tile_id]
        tile = zpad[..., y0:y1, x0:x1]
        start = window * plan["tcs"]
        return tile[:, :, start : start + plan["tcs"] + plan["overlap"]]

    def _gather_window_clips(self, local_clips, tasks, assigned, size, rank, group, n, w_count):
        counts = [len(assigned[r]) for r in range(size)]
        maxc = max(counts)
        with self.distributed_executor.sync_errors("light VAE window gather allocation"):
            local = torch.cat(local_clips, dim=0) if len(local_clips) > 1 else local_clips[0]
            if local.shape[0] < maxc:
                pad = local.new_zeros((maxc - local.shape[0],) + tuple(local.shape[1:]))
                local = torch.cat([local, pad], dim=0)
            local = local.contiguous()
            storage = torch.empty((size, *local.shape), device=local.device, dtype=local.dtype)
            buffers = [storage[r] for r in range(size)]
        dist.all_gather(buffers, local, group=group)
        out = [None] * len(tasks)
        for r in range(size):
            for k, (t, w) in enumerate(assigned[r]):
                out[t * w_count + w] = storage[r, k : k + 1]
        return out

    def _fold_window_clips(self, clips, plan) -> torch.Tensor:
        m = self.model
        just_first = int(plan["drop"] > 0) + 1
        decoded = []
        overlap = None
        pre = plan["frame_pre_padding"]
        fo = plan["frame_overlap"]
        cnf = plan["chunk_num_frames"]
        for clip in clips:
            for j in range(just_first):
                frame_start = j * cnf
                chunk = clip[:, :, frame_start : frame_start + cnf]
                chunk = chunk[:, :, pre:]
                if j == 0:
                    if overlap is not None:
                        chunk = m._blend(overlap, chunk, fo, dim=-3)
                    decoded.append(chunk)
                else:
                    overlap = chunk
        if overlap is not None:
            decoded.append(overlap)
        out = torch.cat(decoded, dim=2)
        if plan["pad_tokens"] > 0:
            tcs = plan["tcs"]
            tratio = plan["tratio"]
            intra = plan["intra_tail"]
            num_tokens_before_pad = plan["T"]
            pad_frames = sum(
                intra if intra and (num_tokens_before_pad + k) % tcs == 0 else tratio for k in range(plan["pad_tokens"])
            )
            out = out[:, :, :-pad_frames]
        return out

    def _decode_video_balanced(self, z, ys, hs, y_ov, xs, ws, x_ov, group, size, rank, profile, started):
        m = self.model
        plan = self._temporal_plan(int(z.shape[2]))
        if plan is None:
            return None
        ratio = int(m.spatial_compression_ratio)
        zpad = self._pad_temporal(z, plan)
        ranges = [
            (y // ratio, y // ratio + h // ratio, x // ratio, x // ratio + w // ratio)
            for y, h in zip(ys, hs)
            for x, w in zip(xs, ws)
        ]
        n = len(ranges)
        w_count = plan["num_chunks"]
        tasks = [(t, w) for t in range(n) for w in range(w_count)]
        assigned = [[] for _ in range(size)]
        for idx, task in enumerate(tasks):
            assigned[idx % size].append(task)
        mine = assigned[rank]
        dtype = next(m.decoder.parameters()).dtype
        cap = self._window_batch_cap()
        prev = m.use_tiling
        m.use_tiling = False
        local_clips = []
        try:
            with self._sync_decode_errors("light VAE window decode", True):
                pos = 0
                for start in range(0, len(mine), cap):
                    batch = mine[start : start + cap]
                    packed = torch.cat([self._window_latent(zpad, ranges, t, w, plan) for (t, w) in batch], dim=0)
                    out = m._decode_clip(packed.to(dtype=dtype))
                    for k in range(out.shape[0]):
                        local_clips.append(out[k : k + 1])
                        pos += 1
        finally:
            m.use_tiling = prev
        decoder_done = self._profile_stamp(profile)
        all_clips = self._gather_window_clips(local_clips, tasks, assigned, size, rank, group, n, w_count)
        gather_done = self._profile_stamp(profile)
        tile_out = []
        for t in range(n):
            tile_out.append(self._fold_window_clips([all_clips[t * w_count + w] for w in range(w_count)], plan))
        rows = [[tile_out[i * len(xs) + j] for j in range(len(xs))] for i in range(len(ys))]
        output = self._stitch_tiles(rows, y_ov, x_ov)
        stitch_done = self._profile_stamp(profile)
        if profile:
            self._last_decode_profile = {
                "rank": rank,
                "world_size": size,
                "latent_shape": list(z.shape),
                "output_shape_fp16": list(output.shape),
                "tiles": n,
                "local_tiles": len(mine),
                "decoder_batch": min(cap, len(mine)),
                "balanced": True,
                "split_pack_ms": 0.0,
                "decoder_ms": (decoder_done - started) * 1000,
                "gather_ms": (gather_done - decoder_done) * 1000,
                "stitch_ms": (stitch_done - gather_done) * 1000,
            }
        return output

    def _decode_video(self, z: torch.Tensor) -> torch.Tensor:
        profile = self._profile_enabled()
        started = self._profile_stamp(profile)
        m = self.model
        # This decode-only VAE is not part of the module discovery used by the
        # distributed layer-wise offload backend, and the pinned stager's
        # device binding can go stale when weights are re-materialized after
        # construction. Trust the real tensor device and stage before decoding.
        if self._device_target.type != "cpu" and next(m.decoder.parameters()).device != self._device_target:
            m.to(self._device_target)
        ratio = int(m.spatial_compression_ratio)
        height = int(z.shape[-2]) * ratio
        width = int(z.shape[-1]) * ratio
        ys, hs, y_ov, xs, ws, x_ov = self._spatial_plan(int(z.shape[2]), height, width)
        tiles = [
            z[..., y // ratio : y // ratio + h // ratio, x // ratio : x // ratio + w // ratio]
            for y, h in zip(ys, hs)
            for x, w in zip(xs, ws)
        ]
        n = len(tiles)
        group, size, world_group, world_size, rank = None, 1, None, 1, 0
        if self.is_distributed_enabled():
            group, size, world_group, world_size, rank = self._decode_parallel_state()
        distributed = group is not None and size > 1 and n >= size
        if (
            distributed
            and size == world_size
            and self._window_balance_enabled()
            and int(z.shape[2]) >= m.tokens_chunk_size + m.token_overlap
        ):
            balanced = self._decode_video_balanced(z, ys, hs, y_ov, xs, ws, x_ov, group, size, rank, profile, started)
            if balanced is not None:
                return balanced
        if not distributed:
            local = list(range(n))
        elif rank >= 0:
            local = list(range(rank, n, size))
        else:
            local = []
        packed = None
        dec = None
        prev = m.use_tiling
        m.use_tiling = False  # tiles are pre-cut; avoid a second spatial split
        try:
            with self._sync_decode_errors("light VAE tile decode", distributed):
                if local:
                    packed = torch.cat([tiles[i] for i in local], dim=0)
                    single = packed.shape[2] == 1
                    if single:
                        packed = packed.expand(
                            -1, -1, int(self._light_settings["image_context_tokens"]), -1, -1
                        ).contiguous()
                    dtype = next(m.decoder.parameters()).dtype
                    packed = packed.to(dtype=dtype)
                    if packed.shape[2] < m.tokens_chunk_size + m.token_overlap:
                        dec = m._decode_clip(packed)
                        start = (
                            int(self._light_settings["image_context_output_phase"])
                            if single
                            else int(m.frame_pre_padding)
                        )
                        dec = dec[:, :, start : start + 1] if single else dec[:, :, start:]
                    else:
                        dec = m._decode(packed)
                packed_done = self._profile_stamp(profile)
        finally:
            m.use_tiling = prev
        decoder_done = self._profile_stamp(profile)
        if distributed:
            if rank >= 0:
                gathered = self._gather_tiles(dec, local, n, group)
            else:
                with self.distributed_executor.sync_errors("light VAE tile gather allocation"):
                    pass
                gathered = None
        else:
            gathered = list(dec.split(1, dim=0))
        gather_done = self._profile_stamp(profile)
        if distributed and size < world_size:
            output = self._stitch_gathered(gathered, ys, xs, y_ov, x_ov) if gathered is not None else None
            output = self._broadcast_decoded_video(output, world_group)
        else:
            output = self._stitch_gathered(gathered, ys, xs, y_ov, x_ov)
        stitch_done = self._profile_stamp(profile)
        if profile:
            self._last_decode_profile = {
                "rank": rank,
                "world_size": size,
                "latent_shape": list(z.shape),
                "output_shape_fp16": list(output.shape),
                "tiles": n,
                "local_tiles": len(local),
                "decoder_batch": int(packed.shape[0]) if packed is not None else 0,
                "split_pack_ms": (packed_done - started) * 1000,
                "decoder_ms": (decoder_done - packed_done) * 1000,
                "gather_ms": (gather_done - decoder_done) * 1000,
                "stitch_ms": (stitch_done - gather_done) * 1000,
            }
        return output

    @torch.inference_mode()
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        _dump = os.environ.get("MINIMAX_H3_LIGHT_VAE_DUMP")
        if _dump and not os.path.exists(_dump):
            try:
                if (not dist.is_initialized()) or dist.get_rank() == 0:
                    torch.save({"latent": latent.detach().cpu()}, _dump)
            except Exception:
                pass
        profile = self._profile_enabled()
        started = self._profile_stamp(profile)
        mean = self._latents_mean.to(device=latent.device, dtype=latent.dtype)
        std = self._latents_std.to(device=latent.device, dtype=latent.dtype)
        z = latent * std + mean
        latent_done = self._profile_stamp(profile)
        # The reference keeps decode/blend/stitch in fp16 and promotes only the
        # assembled result. This also halves tile-parallel communication.
        frames = self._decode_video(z).float()
        cast_done = self._profile_stamp(profile)
        if frames.ndim != 5:
            raise ValueError(f"unexpected decoded video shape {tuple(frames.shape)}")
        _img_std = frames.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
        _img_mean = frames.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
        frames = frames * _img_std + _img_mean
        denorm_done = self._profile_stamp(profile)
        if profile:
            record = dict(getattr(self, "_last_decode_profile", {}))
            record.update(
                latent_denorm_ms=(latent_done - started) * 1000,
                fp16_to_fp32_ms=0.0,
                image_denorm_ms=(denorm_done - cast_done) * 1000,
                internal_total_ms=(denorm_done - started) * 1000,
            )
            # `_decode_video` ends on a synchronized timestamp. Derive the cast
            # cost from its component sum rather than adding another state field.
            inner_ms = sum(record.get(key, 0.0) for key in ("split_pack_ms", "decoder_ms", "gather_ms", "stitch_ms"))
            record["fp16_to_fp32_ms"] = max(
                0.0,
                record["internal_total_ms"] - record["latent_denorm_ms"] - inner_ms - record["image_denorm_ms"],
            )
            logger.info("[LightVAEProfile] %s", json.dumps(record, sort_keys=True))
        if self._resident:
            try:
                rank = dist.get_rank() if dist.is_initialized() else 0
                peak_alloc = torch.accelerator.max_memory_allocated(self._device_target) / (1024**3)
                reserved = torch.accelerator.memory_reserved(self._device_target) / (1024**3)
                logger.info(
                    "[LightVAEHBM] rank=%d peak_alloc_gib=%.2f reserved_gib=%.2f",
                    rank,
                    peak_alloc,
                    reserved,
                )
            except Exception:
                logger.debug("LightVAEHBM logging unavailable", exc_info=True)
        return frames.float()


__all__ = ["MiniMaxH3AudioVAE", "MiniMaxH3LightVideoVAE", "MiniMaxH3VideoVAE"]
