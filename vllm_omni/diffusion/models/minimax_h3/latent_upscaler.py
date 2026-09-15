# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Learned latent super-resolution for MiniMax-H3 video latents.

The released upscaler is a pure-3D convolutional resizer trained on H3's
24-channel VAE latents: it lifts a latent to a target spatial size without a
decode/encode round trip through the ~5B-parameter H3 VAE.  Weights come from
``LBH-123-AI/Minimax_h3_latent_Upscaler`` and are a plain ``state_dict``, so the
module layout below reproduces the checkpoint's parameter names exactly.

The network works in a space one normalization *below* the pipeline latent.
A vLLM-Omni H3 latent is already normalized -- :class:`MiniMaxH3VideoVAE`
denormalizes at decode time -- and the reference ComfyUI node applies the VAE's
``latents_mean`` / ``latents_std`` on top of a latent in that same convention,
so the upscaler was trained on the doubly-normalized tensor.  Feeding it the
pipeline latent directly inflates the output by roughly 5x (measured: output
std 5.4 against an input std of 1.0), which decodes as a magenta grid at one
tile per latent cell.  :meth:`MiniMaxH3LatentUpscaler.upscale` therefore
normalizes in and denormalizes out, exactly as the reference node does.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import regex as re
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from vllm.logger import init_logger

logger = init_logger(__name__)

# H3's video VAE downsamples 16x spatially, so one latent cell is a 16px tile.
MINIMAX_H3_VAE_SPATIAL_DOWNSAMPLE = 16
# The published checkpoints were trained between 1x and 4x.
MINIMAX_H3_LATENT_UPSCALE_MAX_SCALE = 4.0
_EMBED_DIM = 64
_GROUP_NORM_GROUPS = 32


class MiniMaxH3LatentUpscalerError(ValueError):
    """A latent upscaler artifact that cannot be served."""


@dataclass(frozen=True)
class MiniMaxH3LatentUpscalerArch:
    """Shape of a released latent upscaler, recovered from its ``state_dict``."""

    in_channels: int = 24
    channels: int = 512
    in_blocks: int = 12
    out_blocks: int = 12
    temporal_every: int = 2
    temporal_kernel: int = 5


def _normalization(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(_GROUP_NORM_GROUPS, channels)


class _ResBlockEmb3D(nn.Module):
    """3D residual block whose output norm is modulated by the scale embedding."""

    def __init__(self, channels: int, emb_channels: int) -> None:
        super().__init__()
        self.in_layers = nn.Sequential(
            _normalization(channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * channels),
        )
        self.out_norm = _normalization(channels)
        # Index 1 is the training-time dropout. It is a no-op at inference but
        # has to keep its slot so ``out_layers.2`` still names the convolution.
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Identity(),
            nn.Conv3d(channels, channels, 3, padding=1),
        )
        self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        scale, shift = torch.chunk(self.emb_layers(emb).to(h.dtype)[..., None, None, None], 2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        return self.skip(x) + self.out_layers(h)


class _TemporalConv3D(nn.Module):
    """Zero-initialized depthwise temporal residual, for cross-frame coherence."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.norm = _normalization(channels)
        self.dwconv = nn.Conv3d(
            channels,
            channels,
            kernel_size=(kernel_size, 1, 1),
            padding=(kernel_size // 2, 0, 0),
            groups=channels,
        )
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pwconv(self.dwconv(F.silu(self.norm(x))))


class MiniMaxH3LatentResizer3D(nn.Module):
    """The released pure-3D latent resizer.

    ``in_blocks`` run at the source resolution, a trilinear interpolation moves
    the feature volume to the target size, and ``out_blocks`` refine it there.
    The requested scale enters every residual block through a two-layer
    embedding of ``scale - 1``.
    """

    def __init__(self, arch: MiniMaxH3LatentUpscalerArch) -> None:
        super().__init__()
        self.arch = arch
        self.conv_in = nn.Conv3d(arch.in_channels, arch.channels, 3, padding=1)
        self.embed = nn.Sequential(
            nn.Linear(1, _EMBED_DIM),
            nn.SiLU(),
            nn.Linear(_EMBED_DIM, _EMBED_DIM),
        )
        self.in_blocks = self._build_stage(arch, arch.in_blocks)
        self.out_blocks = self._build_stage(arch, arch.out_blocks)
        self.norm_out = _normalization(arch.channels)
        self.conv_out = nn.Conv3d(arch.channels, arch.in_channels, 3, padding=1)

    @staticmethod
    def _build_stage(arch: MiniMaxH3LatentUpscalerArch, num_blocks: int) -> nn.ModuleList:
        # Residual and temporal blocks share one flat ModuleList in the
        # checkpoint, so their interleaving fixes the parameter names.
        stage = nn.ModuleList()
        for index in range(num_blocks):
            stage.append(_ResBlockEmb3D(arch.channels, _EMBED_DIM))
            if arch.temporal_every > 0 and index % arch.temporal_every == 0:
                stage.append(_TemporalConv3D(arch.channels, arch.temporal_kernel))
        return stage

    @property
    def temporal_kernel(self) -> int:
        return self.arch.temporal_kernel if self.arch.temporal_every > 0 else 0

    def forward(self, latent: torch.Tensor, *, scale: float, target_size: tuple[int, int, int]) -> torch.Tensor:
        """Resize ``[B, C, T, H, W]`` to ``target_size`` under a scale hint."""
        emb = self.embed(
            torch.tensor([[scale - 1.0]], dtype=latent.dtype, device=latent.device),
        ).expand(latent.shape[0], -1)

        hidden = self.conv_in(latent)
        for block in self.in_blocks:
            hidden = block(hidden, emb) if isinstance(block, _ResBlockEmb3D) else block(hidden)
        hidden = F.interpolate(hidden, size=target_size, mode="trilinear", align_corners=False)
        for block in self.out_blocks:
            hidden = block(hidden, emb) if isinstance(block, _ResBlockEmb3D) else block(hidden)
        return self.conv_out(F.silu(self.norm_out(hidden)))


def detect_minimax_h3_upscaler_arch(state_dict: Mapping[str, torch.Tensor]) -> MiniMaxH3LatentUpscalerArch:
    """Recover the block layout of a released upscaler from its tensors."""
    conv_in = state_dict.get("conv_in.weight")
    if conv_in is None or conv_in.ndim != 5:
        raise MiniMaxH3LatentUpscalerError("not a MiniMax-H3 3D latent upscaler: expected a rank-5 'conv_in.weight'")
    if any("attn" in key for key in state_dict):
        raise MiniMaxH3LatentUpscalerError(
            "this checkpoint carries attention blocks, which no released MiniMax-H3 upscaler uses"
        )

    residual_blocks: dict[str, set[int]] = {"in_blocks": set(), "out_blocks": set()}
    temporal_blocks: dict[str, set[int]] = {"in_blocks": set(), "out_blocks": set()}
    temporal_kernel = 5
    for key, tensor in state_dict.items():
        match = re.match(r"(in_blocks|out_blocks)\.(\d+)\.(in_layers|dwconv)\b", key)
        if match is None:
            continue
        stage, index, kind = match.group(1), int(match.group(2)), match.group(3)
        if kind == "in_layers":
            residual_blocks[stage].add(index)
        else:
            temporal_blocks[stage].add(index)
            if key.endswith("dwconv.weight"):
                temporal_kernel = int(tensor.shape[2])

    has_temporal = bool(temporal_blocks["in_blocks"] or temporal_blocks["out_blocks"])
    return MiniMaxH3LatentUpscalerArch(
        in_channels=int(conv_in.shape[1]),
        channels=int(conv_in.shape[0]),
        in_blocks=len(residual_blocks["in_blocks"]),
        out_blocks=len(residual_blocks["out_blocks"]),
        temporal_every=2 if has_temporal else 0,
        temporal_kernel=temporal_kernel,
    )


@dataclass(frozen=True)
class MiniMaxH3LatentUpscaleTarget:
    """A resolved upscale target in latent cells, plus the scale hint to feed the network."""

    latent_height: int
    latent_width: int
    scale: float

    @property
    def height(self) -> int:
        return self.latent_height * MINIMAX_H3_VAE_SPATIAL_DOWNSAMPLE

    @property
    def width(self) -> int:
        return self.latent_width * MINIMAX_H3_VAE_SPATIAL_DOWNSAMPLE


def resolve_minimax_h3_latent_upscale_target(
    *,
    latent_height: int,
    latent_width: int,
    scale: float | None = None,
    height: int | None = None,
    width: int | None = None,
    megapixels: float | None = None,
    align: int = 32,
) -> MiniMaxH3LatentUpscaleTarget:
    """Resolve one of the three sizing modes into latent dimensions.

    ``scale`` multiplies the source size; ``height``/``width`` name the target
    in pixels; ``megapixels`` names a total pixel budget and keeps the aspect
    ratio.  For the two size-based modes the network still needs a single scale
    hint, which is the mean of the per-axis ratios -- what the reference node
    feeds the same weights.  Targets snap to ``align`` pixels and then to the
    VAE's 16px cell.
    """
    source_height = latent_height * MINIMAX_H3_VAE_SPATIAL_DOWNSAMPLE
    source_width = latent_width * MINIMAX_H3_VAE_SPATIAL_DOWNSAMPLE
    modes = [
        name
        for name, value in (("scale", scale), ("dimensions", height or width), ("megapixels", megapixels))
        if value is not None
    ]
    if len(modes) != 1:
        raise MiniMaxH3LatentUpscalerError(
            f"latent upscaling takes exactly one of scale, height/width, or megapixels, got {modes or ['nothing']}"
        )
    if align < 1:
        raise MiniMaxH3LatentUpscalerError(f"latent upscale align must be positive, got {align}")

    if scale is not None:
        target_height = source_height * float(scale)
        target_width = source_width * float(scale)
    elif megapixels is not None:
        if megapixels <= 0:
            raise MiniMaxH3LatentUpscalerError(f"latent upscale megapixels must be positive, got {megapixels}")
        aspect = source_width / source_height
        target_height = math.sqrt(float(megapixels) * 1024 * 1024 / aspect)
        target_width = target_height * aspect
    else:
        if height is None or width is None:
            raise MiniMaxH3LatentUpscalerError("latent upscaling by dimensions needs both height and width")
        target_height = float(height)
        target_width = float(width)

    cell = MINIMAX_H3_VAE_SPATIAL_DOWNSAMPLE
    out_height = max(1, int(round(round(target_height / align) * align / cell)))
    out_width = max(1, int(round(round(target_width / align) * align / cell)))
    effective = (out_height / latent_height + out_width / latent_width) / 2.0
    if out_height < latent_height or out_width < latent_width:
        raise MiniMaxH3LatentUpscalerError(
            f"the MiniMax-H3 latent upscaler only upscales: {latent_width}x{latent_height} -> "
            f"{out_width}x{out_height} latent cells"
        )
    if effective > MINIMAX_H3_LATENT_UPSCALE_MAX_SCALE:
        logger.warning(
            "MiniMax-H3 latent upscale factor %.3f exceeds the %.1fx the checkpoint was trained for",
            effective,
            MINIMAX_H3_LATENT_UPSCALE_MAX_SCALE,
        )
    return MiniMaxH3LatentUpscaleTarget(latent_height=out_height, latent_width=out_width, scale=effective)


class MiniMaxH3LatentUpscaler(nn.Module):
    """Serving wrapper around :class:`MiniMaxH3LatentResizer3D`.

    Owns the temporal chunking that keeps long clips inside VRAM and the
    residency policy that keeps the extra ~700MB off the device between
    requests.
    """

    def __init__(
        self,
        resizer: MiniMaxH3LatentResizer3D,
        *,
        device: torch.device,
        dtype: torch.dtype,
        latents_mean: Sequence[float],
        latents_std: Sequence[float],
        chunk_frames: int = 32,
        chunk_overlap: int | None = None,
        resident: bool = False,
    ) -> None:
        super().__init__()
        if chunk_frames < 0:
            raise MiniMaxH3LatentUpscalerError(f"latent upscale chunk_frames must be >= 0, got {chunk_frames}")
        # Plain tuples rather than buffers: this module manages its own device
        # placement, and a buffer would also show up in the pipeline state dict.
        self.latents_mean = tuple(float(value) for value in latents_mean)
        self.latents_std = tuple(float(value) for value in latents_std)
        if len(self.latents_mean) != len(self.latents_std):
            raise MiniMaxH3LatentUpscalerError(
                f"latents_mean/latents_std length mismatch: {len(self.latents_mean)} vs {len(self.latents_std)}"
            )
        if any(value == 0.0 for value in self.latents_std):
            raise MiniMaxH3LatentUpscalerError("latents_std must not contain zeros")
        self.resizer = resizer.to(dtype=dtype).eval().requires_grad_(False)
        self.device = device
        self.dtype = dtype
        self.chunk_frames = chunk_frames
        self.chunk_overlap = resizer.temporal_kernel if chunk_overlap is None else chunk_overlap
        self.resident = resident
        self.resizer.to(device if resident else torch.device("cpu"))

    @torch.inference_mode()
    def upscale(self, latent: torch.Tensor, target: MiniMaxH3LatentUpscaleTarget) -> torch.Tensor:
        """Upscale a normalized ``[B, 24, T, H, W]`` H3 latent to ``target``."""
        if latent.ndim != 5:
            raise MiniMaxH3LatentUpscalerError(f"expected a rank-5 H3 latent, got shape {tuple(latent.shape)}")
        size = (int(latent.shape[2]), target.latent_height, target.latent_width)
        if size[1:] == tuple(latent.shape[-2:]):
            return latent

        if int(latent.shape[1]) != len(self.latents_mean):
            raise MiniMaxH3LatentUpscalerError(
                f"expected {len(self.latents_mean)} latent channels, got {int(latent.shape[1])}"
            )
        source_dtype = latent.dtype
        work = latent.to(device=self.device, dtype=self.dtype)
        mean, std = self._norm_tensors(work.device, work.dtype)
        try:
            if not self.resident:
                self.resizer.to(self.device)
            upscaled = self._run((work - mean) / std, target.scale, size) * std + mean
        finally:
            if not self.resident:
                self.resizer.to(torch.device("cpu"))
        return upscaled.to(dtype=source_dtype)

    def _norm_tensors(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (1, -1, 1, 1, 1)
        return (
            torch.tensor(self.latents_mean, device=device, dtype=dtype).view(shape),
            torch.tensor(self.latents_std, device=device, dtype=dtype).view(shape),
        )

    def _run(self, latent: torch.Tensor, scale: float, size: tuple[int, int, int]) -> torch.Tensor:
        """Resize the clip, in temporal chunks when it does not fit in one pass.

        Chunking is not free: every GroupNorm in the network pools its
        statistics over the whole clip it is given, so a chunked pass is an
        approximation of the single pass everywhere, not only at the seams.
        ``chunk_frames=0`` buys the exact result for the memory a full-length
        activation volume costs.
        """
        frames = int(latent.shape[2])
        overlap = self.chunk_overlap
        if self.chunk_frames == 0 or frames <= self.chunk_frames or overlap == 0:
            return self.resizer(latent, scale=scale, target_size=size)

        logger.debug(
            "MiniMax-H3 latent upscale: %d frames in chunks of %d with %d-frame overlap",
            frames,
            self.chunk_frames,
            overlap,
        )
        # Replicate padding gives the first and last chunk the same amount of
        # temporal context as the interior ones, so the clip does not flicker at
        # its ends. Unlike the reference node the context is symmetric: every
        # chunk sees `overlap` frames on both sides rather than only before.
        padded = F.pad(latent, (0, 0, 0, 0, overlap, overlap), mode="replicate")
        batch, channels = latent.shape[0], latent.shape[1]
        accumulated = torch.zeros(batch, channels, frames, size[1], size[2], device=latent.device, dtype=latent.dtype)
        weights = torch.zeros(1, 1, frames, 1, 1, device=latent.device, dtype=latent.dtype)

        for start in range(0, frames, self.chunk_frames):
            core_start, core_end = start, min(frames, start + self.chunk_frames)
            blend_start, blend_end = max(0, core_start - overlap), min(frames, core_end + overlap)
            # Padded coordinates: original frame `i` sits at `i + overlap`.
            segment = padded[:, :, blend_start : blend_end + 2 * overlap].contiguous()
            decoded = self.resizer(
                segment,
                scale=scale,
                target_size=(int(segment.shape[2]), size[1], size[2]),
            )[:, :, overlap : overlap + (blend_end - blend_start)]

            weight = self._blend_weights(
                blend_start,
                blend_end,
                core_start,
                core_end,
                device=latent.device,
                dtype=latent.dtype,
            )
            accumulated[:, :, blend_start:blend_end] += decoded * weight
            weights[:, :, blend_start:blend_end] += weight
            del segment, decoded

        return accumulated / weights.clamp(min=1e-8)

    @staticmethod
    def _blend_weights(
        blend_start: int,
        blend_end: int,
        core_start: int,
        core_end: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Linear ramps over the frames a chunk shares with its neighbours."""
        weight = torch.ones(blend_end - blend_start, device=device, dtype=dtype)
        lead = core_start - blend_start
        if lead > 0:
            weight[:lead] = torch.arange(1, lead + 1, device=device, dtype=dtype) / (lead + 1)
        trail = blend_end - core_end
        if trail > 0:
            weight[-trail:] = torch.arange(trail, 0, -1, device=device, dtype=dtype) / (trail + 1)
        return weight.view(1, 1, -1, 1, 1)


_UPSCALER_DTYPES = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}


def _resolve_checkpoint_file(path: str | Path) -> Path:
    """Find the single upscaler checkpoint at ``path``."""
    candidate = Path(path).expanduser()
    if candidate.is_file():
        return candidate
    if not candidate.is_dir():
        raise MiniMaxH3LatentUpscalerError(f"MiniMax-H3 latent upscaler not found at {candidate}")
    files = sorted(file for suffix in (".safetensors", ".pth") for file in candidate.glob(f"*{suffix}"))
    if len(files) != 1:
        raise MiniMaxH3LatentUpscalerError(
            f"{candidate} holds {len(files)} latent upscaler checkpoints; point the path at one file"
        )
    return files[0]


def _read_state_dict(file: Path) -> dict[str, torch.Tensor]:
    if file.suffix == ".safetensors":
        with safe_open(file, framework="pt", device="cpu") as checkpoint:
            state_dict = {key: checkpoint.get_tensor(key) for key in checkpoint.keys()}
    elif file.suffix == ".pth":
        state_dict = torch.load(file, map_location="cpu", weights_only=True)
        if isinstance(state_dict, Mapping) and "model" in state_dict:
            state_dict = state_dict["model"]
    else:
        raise MiniMaxH3LatentUpscalerError(f"unsupported latent upscaler checkpoint {file}")
    # Some releases nest the resizer under a training-time wrapper.
    if any(key.startswith("upscaler.") for key in state_dict):
        return {
            key.removeprefix("upscaler."): value for key, value in state_dict.items() if key.startswith("upscaler.")
        }
    return dict(state_dict)


def load_minimax_h3_latent_upscaler(
    path: str | Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
    latents_mean: Sequence[float],
    latents_std: Sequence[float],
    chunk_frames: int = 32,
    chunk_overlap: int | None = None,
    resident: bool = False,
) -> MiniMaxH3LatentUpscaler:
    """Build a serving-ready upscaler from a released checkpoint."""
    file = _resolve_checkpoint_file(path)
    state_dict = _read_state_dict(file)
    arch = detect_minimax_h3_upscaler_arch(state_dict)
    # Built on meta and assigned from the checkpoint: the pipeline is
    # constructed under a CUDA default-device context, and a real
    # initialization would spend ~2.7GB of device memory on values the next
    # line overwrites anyway.
    with torch.device("meta"):
        resizer = MiniMaxH3LatentResizer3D(arch)
    resizer.load_state_dict(state_dict, strict=True, assign=True)
    logger.info(
        "Loaded MiniMax-H3 latent upscaler %s: %d channels, %d+%d blocks, temporal kernel %d, %s",
        file.name,
        arch.channels,
        arch.in_blocks,
        arch.out_blocks,
        arch.temporal_kernel,
        dtype,
    )
    return MiniMaxH3LatentUpscaler(
        resizer,
        device=device,
        dtype=dtype,
        latents_mean=latents_mean,
        latents_std=latents_std,
        chunk_frames=chunk_frames,
        chunk_overlap=chunk_overlap,
        resident=resident,
    )


def resolve_minimax_h3_latent_upscaler(
    od_config,
    *,
    device: torch.device,
    latent_stats: Callable[[], tuple[Sequence[float], Sequence[float]]],
) -> MiniMaxH3LatentUpscaler | None:
    """Build the upscaler declared by ``--additional-config``, if any.

    ``latent_stats`` yields the VAE's per-channel ``latents_mean`` /
    ``latents_std`` and is called only once a checkpoint is configured, so a
    deployment without this stage never reaches for them.

    Recognized keys: ``latent_upscaler_path`` (required to enable the stage),
    ``latent_upscaler_dtype``, ``latent_upscaler_chunk_frames`` (0 disables
    temporal chunking) and ``latent_upscaler_resident`` (keep the weights on
    the device between requests instead of parking them in host memory).
    """
    additional = getattr(od_config, "additional_config", None) or {}
    path = additional.get("latent_upscaler_path")
    if not path:
        return None
    raw_dtype = additional.get("latent_upscaler_dtype")
    if raw_dtype is None:
        dtype = getattr(od_config, "dtype", torch.bfloat16)
    elif str(raw_dtype) not in _UPSCALER_DTYPES:
        raise MiniMaxH3LatentUpscalerError(
            f"latent_upscaler_dtype must be one of {sorted(_UPSCALER_DTYPES)}, got {raw_dtype!r}"
        )
    else:
        dtype = _UPSCALER_DTYPES[str(raw_dtype)]
    latents_mean, latents_std = latent_stats()
    return load_minimax_h3_latent_upscaler(
        path,
        device=device,
        dtype=dtype,
        latents_mean=latents_mean,
        latents_std=latents_std,
        chunk_frames=int(additional.get("latent_upscaler_chunk_frames", 32)),
        resident=bool(additional.get("latent_upscaler_resident", False)),
    )


def parse_minimax_h3_latent_upscale_request(value) -> dict[str, float | int] | None:
    """Normalize ``extra_args['latent_upscale']`` into resolver keywords.

    Accepts a bare multiplier (``2.0``), ``false``/``null`` to opt out of a
    server-side default, or an object naming one sizing mode:
    ``{"scale": 2}``, ``{"width": 2688, "height": 1536}`` or
    ``{"megapixels": 4}``, each optionally with ``align``.
    """
    if value is None or value is False:
        return None
    if isinstance(value, bool):
        raise MiniMaxH3LatentUpscalerError("latent_upscale must be a number or an object, not true")
    if isinstance(value, (int, float)):
        return {"scale": float(value)}
    if not isinstance(value, Mapping):
        raise MiniMaxH3LatentUpscalerError(f"latent_upscale must be a number or an object, got {type(value).__name__}")
    known = {"scale", "width", "height", "megapixels", "align"}
    unknown = set(value) - known
    if unknown:
        raise MiniMaxH3LatentUpscalerError(f"unknown latent_upscale keys {sorted(unknown)}; expected {sorted(known)}")
    resolved: dict[str, float | int] = {}
    for key in ("scale", "megapixels"):
        if value.get(key) is not None:
            resolved[key] = float(value[key])
    for key in ("width", "height", "align"):
        if value.get(key) is not None:
            resolved[key] = int(value[key])
    return resolved or None


@dataclass(frozen=True)
class MiniMaxH3LatentRefineSpec:
    """How much of the sigma schedule a second denoise pass re-runs."""

    strength: float

    def start_index(self, num_sigma_points: int) -> int:
        """The schedule position the refine pass starts from.

        ``strength`` is the fraction of the request's steps to re-run, the
        img2img convention, so the cost of the pass is predictable from the
        request. At least one step always runs.
        """
        total_steps = num_sigma_points - 1
        if total_steps < 1:
            raise MiniMaxH3LatentUpscalerError("a refine pass needs a schedule with at least two sigma points")
        return total_steps - max(1, round(self.strength * total_steps))


def parse_minimax_h3_latent_refine_request(value) -> MiniMaxH3LatentRefineSpec | None:
    """Normalize ``extra_args['latent_refine']`` into a refine spec.

    Accepts a bare strength (``0.4``), ``false``/``null`` to opt out of a
    server-side default, or ``{"strength": 0.4}``.
    """
    if value is None or value is False:
        return None
    if isinstance(value, bool):
        raise MiniMaxH3LatentUpscalerError("latent_refine must be a number or an object, not true")
    if isinstance(value, Mapping):
        unknown = set(value) - {"strength"}
        if unknown:
            raise MiniMaxH3LatentUpscalerError(f"unknown latent_refine keys {sorted(unknown)}; expected ['strength']")
        if value.get("strength") is None:
            return None
        value = value["strength"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MiniMaxH3LatentUpscalerError(f"latent_refine strength must be a number, got {type(value).__name__}")
    strength = float(value)
    if not 0.0 < strength <= 1.0:
        raise MiniMaxH3LatentUpscalerError(f"latent_refine strength must be in (0, 1], got {strength}")
    return MiniMaxH3LatentRefineSpec(strength=strength)


__all__ = [
    "MINIMAX_H3_LATENT_UPSCALE_MAX_SCALE",
    "MINIMAX_H3_VAE_SPATIAL_DOWNSAMPLE",
    "MiniMaxH3LatentRefineSpec",
    "MiniMaxH3LatentResizer3D",
    "MiniMaxH3LatentUpscaleTarget",
    "MiniMaxH3LatentUpscaler",
    "MiniMaxH3LatentUpscalerArch",
    "MiniMaxH3LatentUpscalerError",
    "detect_minimax_h3_upscaler_arch",
    "load_minimax_h3_latent_upscaler",
    "parse_minimax_h3_latent_refine_request",
    "parse_minimax_h3_latent_upscale_request",
    "resolve_minimax_h3_latent_upscale_target",
    "resolve_minimax_h3_latent_upscaler",
]
