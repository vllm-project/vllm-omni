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
        encode_only: bool = False,
        decode_only: bool = False,
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__()
        self._device_target = device
        self.encode_only = bool(encode_only)
        self.decode_only = bool(decode_only)
        if self.encode_only and self.decode_only:
            raise ValueError("MiniMax H3 video VAE cannot be both encode-only and decode-only")
        self.config_dict = _load_component_config(component_path)
        _check_trust_remote_code(component_path, self.config_dict, trust_remote_code)
        if self.encode_only:
            self.remote = _load_video_vae_encoder(component_path, self.config_dict)
        else:
            self.remote = _load_remote_component(component_path, self.config_dict, trust_remote_code=trust_remote_code)
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
        """
        parameter = next(self.parameters())
        pixels = pixels.to(device=parameter.device, dtype=torch.float32)
        mean = pixels.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1, 1)
        std = pixels.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1, 1)
        with current_omni_platform.create_autocast_context(
            device_type=parameter.device.type,
            dtype=torch.float16,
            enabled=parameter.device.type != "cpu",
        ):
            moments = self.model.encode_temporal((pixels - mean) / std)
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
        waveform = self.remote.decode(latent * std + mean)
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError(f"unexpected decoded audio shape {tuple(waveform.shape)}")
        return waveform.permute(1, 0, 2).contiguous().float()


__all__ = ["MiniMaxH3AudioVAE", "MiniMaxH3VideoVAE"]
