# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Open-H3-Regenerate-2K: MiniMax-H3 as a few-step 2K video refiner.

The release (``Efficient-Large-Model/Open-H3-Regenerate-2K``) regenerates a
768P H3 draft at a higher resolution: the draft latent is upscaled with the
3D-conv latent upscaler, put back on the flow at sigma 0.909375 and denoised in
three Euler steps by MiniMax-H3 carrying three adapters, then decoded.

The adapters ship as two artifacts, both edits of the base FL2VA transformer:

* ``lora/open_h3_regenerate_2k_lora.safetensors``: the Open-H3 refiner,
  LightX2V FL2VA Turbo and LMS Sharp LoRAs concatenated along the rank axis
  with their strengths folded into ``B``, so ``B @ A`` is their exact sum. It
  is fused into the checkpoint stream at load time, like FastH3.
* ``lora/sharp_curve_branches.safetensors``: LMS Sharp's per-block AdaLN
  branches, ``up @ (down @ c(t))`` over an 8-D curve of the model timestep.
  Every ``up`` factor in the release is exactly zero, so the branches add
  nothing; the loader verifies that rather than installing them, and refuses a
  bundle whose branches would contribute.

The sampling contract follows the release's reference pipeline: video rows
start at ``(1 - s0) * upscaled + s0 * noise`` and walk a shift-6 grid with CFG
1, while the audio rows hold encoded silence at timestep 1.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import torch
from safetensors import safe_open
from vllm.logger import init_logger

from vllm_omni.diffusion.offloader.config import DIT_COMPONENT, OffloadStrategy, resolve_offload
from vllm_omni.diffusion.sched.sigma_schedule import DMD2SigmaSchedule
from vllm_omni.errors import OmniClientError

from .fasth3 import (
    MiniMaxH3WeightFusion,
    _ParamPatch,
    _resolve_native_target,
    _split_adapter_key,
)

logger = init_logger(__name__)

REGENERATE_LORA_FILE = "lora/open_h3_regenerate_2k_lora.safetensors"
REGENERATE_CURVE_FILE = "lora/sharp_curve_branches.safetensors"
REGENERATE_LORA_CONFIG_FILE = "lora/lora_config.json"
REGENERATE_UPSCALER_FILE = "latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"
REGENERATE_BASE_MODEL = "MiniMaxAI/MiniMax-H3"
_LORA_FORMAT = "lora"
_CURVE_FORMAT = "lms-sharp-curve-adaln"
_CURVE_PREFIX = "transformer.transformer_blocks."
_CURVE_SUFFIXES = {".adaln_proj.linear.curve_A.weight": "down", ".adaln_proj.linear.curve_B.weight": "up"}

# The refiner's grid, as H3 schedule positions (the shift-6 map turns them into
# sigmas). Three steps, the release default, are the refiner's original 4-step
# grid [1, 0.75, 0.5, 0.25, 0] entered at sigma 0.909375 instead of 1; two
# steps divide [0, u(0.7)] evenly, so less of the upscaled draft is re-noised.
REGENERATE_VIDEO_SHIFT = 6.0
REGENERATE_DEFAULT_STEPS = 3


def _position(sigma: float) -> float:
    """Inverse of the shift-6 map ``s * u / (1 + (s - 1) * u)``."""
    return sigma / (REGENERATE_VIDEO_SHIFT - (REGENERATE_VIDEO_SHIFT - 1) * sigma)


_REGENERATE_POSITIONS = {
    3: (_position(0.909375), 0.5, 0.25, 0.0),
    2: (_position(0.7), _position(0.7) / 2, 0.0),
}
# The release's output size, the default upscale target for a 1344x768 draft.
REGENERATE_DEFAULT_UPSCALE = {"width": 2560, "height": 1440, "align": 32}
REGENERATE_SUPPORTED_TASKS = frozenset({"t2va"})


class MiniMaxH3RegenerateError(ValueError):
    """The artifact is an Open-H3-Regenerate bundle, but it cannot be served."""


def regenerate_schedule(num_steps: int) -> DMD2SigmaSchedule:
    """The release's refine grid as positions H3 shifts by :data:`REGENERATE_VIDEO_SHIFT`."""
    positions = _REGENERATE_POSITIONS.get(num_steps)
    if positions is None:
        raise OmniClientError(
            f"Open-H3-Regenerate runs {sorted(_REGENERATE_POSITIONS)} steps, got num_inference_steps={num_steps}"
        )
    return DMD2SigmaSchedule.from_positions(positions)


class _RegenerateLoRAFusion(MiniMaxH3WeightFusion):
    adapter_name: ClassVar[str] = "Open-H3-Regenerate LoRA"
    error_cls: ClassVar[type[ValueError]] = MiniMaxH3RegenerateError


def _fusion_peak_bytes(patches: Mapping[str, Any]) -> tuple[int, int]:
    """LoRA rank and the largest transient accelerator memory one fused parameter needs.

    Fusing a parameter holds its FP32 delta (for grouped QKV, the per-slot
    products and the placed delta at once) next to the BF16 weight and result,
    so twelve bytes per element bound it.
    """
    rank = peak = 0
    for patch in patches.values():
        factors = [(a, b) for a, b in patch.low_rank.values() if a is not None and b is not None]
        if not factors:
            continue
        rank = max(rank, *(a.shape[0] for a, _ in factors))
        elements = sum(b.shape[0] for _, b in factors) * max(a.shape[1] for a, _ in factors)
        peak = max(peak, 12 * elements)
    return rank, peak


@dataclass
class MiniMaxH3RegenerateAdapter:
    """An Open-H3-Regenerate bundle, resolved against one H3 transformer."""

    source: Path
    fusion: _RegenerateLoRAFusion
    fusion_peak_bytes: int = 0

    @property
    def upscaler_path(self) -> Path:
        return self.source / REGENERATE_UPSCALER_FILE

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        head_dim: int,
        num_blocks: int,
        num_refiner_blocks: int,
    ) -> MiniMaxH3RegenerateAdapter | None:
        """Claim a release directory, else return ``None``.

        Only a directory carrying both release artifacts is claimed, so every
        other ``--lora-path`` keeps its existing route.
        """
        root = Path(path)
        lora_path = root / REGENERATE_LORA_FILE
        curve_path = root / REGENERATE_CURVE_FILE
        if not (lora_path.is_file() and curve_path.is_file()):
            return None
        config_path = root / REGENERATE_LORA_CONFIG_FILE
        if config_path.is_file():
            base_model = str(json.loads(config_path.read_text(encoding="utf-8")).get("base_model", ""))
            if not base_model.startswith(REGENERATE_BASE_MODEL):
                raise MiniMaxH3RegenerateError(f"{root} adapts {base_model!r}, not {REGENERATE_BASE_MODEL}")
        patches = _read_lora_patches(lora_path, num_blocks=num_blocks, num_refiner_blocks=num_refiner_blocks)
        _check_curve_branches_are_inert(curve_path, num_blocks=num_blocks)
        if not (root / REGENERATE_UPSCALER_FILE).is_file():
            raise MiniMaxH3RegenerateError(
                f"{root} is missing the release's latent upscaler {REGENERATE_UPSCALER_FILE}"
            )
        rank, peak = _fusion_peak_bytes(patches)
        # The fusion runs on the accelerator while the checkpoint streams in, on
        # top of the weights already loaded; say how much it needs up front.
        logger.info(
            "Open-H3-Regenerate bundle %s: %d fused parameters, LoRA rank %d; fusing needs up to %.2f GiB "
            "of accelerator memory per parameter while the checkpoint loads",
            root,
            len(patches),
            rank,
            peak / 2**30,
        )
        return cls(
            source=root,
            fusion=_RegenerateLoRAFusion(source=lora_path, patches=patches, head_dim=head_dim),
            fusion_peak_bytes=peak,
        )

    def apply(self, weights: Iterable[tuple[str, torch.Tensor]]) -> Iterator[tuple[str, torch.Tensor]]:
        try:
            yield from self.fusion.apply(weights)
        except torch.OutOfMemoryError as exc:
            raise MiniMaxH3RegenerateError(
                f"ran out of accelerator memory fusing {self.source} into the checkpoint at load time; "
                f"each fused parameter needs up to {self.fusion_peak_bytes / 2**30:.2f} GiB beyond the weights "
                "already loaded. Free memory on these devices or serve on devices with more memory."
            ) from exc

    def validate_fully_applied(self, loaded: Iterable[str] | None = None) -> None:
        self.fusion.validate_fully_applied(loaded)

    def check_serving_contract(self, *, partition: str, od_config: Any) -> None:
        if partition != "fl2va":
            raise ValueError(f"Open-H3-Regenerate adapts the FL2VA transformer and cannot serve {partition!r}")
        offload = resolve_offload(od_config)
        if offload.strategy is not OffloadStrategy.NONE and offload.offloads(DIT_COMPONENT):
            # A host-weight plan installs the transformer without passing through
            # load_weights(), where the LoRA is fused. Offloading only the text
            # encoder leaves that path alone.
            raise ValueError(
                "Open-H3-Regenerate is fused while the checkpoint streams in, so the DiT cannot use "
                f"{offload.strategy.value} offload. Offload the text encoder only, or nothing."
            )

    def check_request(self, sampling: Any) -> int:
        """Validate a request against the release's sampler; return its step count."""
        if sampling.lora_request is not None:
            raise OmniClientError(
                f"this server fused {self.source} into the checkpoint at startup, so per-request "
                "lora is unavailable; drop the lora field"
            )
        steps = int(sampling.num_inference_steps or REGENERATE_DEFAULT_STEPS)
        regenerate_schedule(steps)
        extra = sampling.extra_args or {}
        if "flow_shift" in extra:
            try:
                shift = float(extra["flow_shift"])
            except (TypeError, ValueError) as exc:
                raise OmniClientError(f"Open-H3-Regenerate requires flow_shift={REGENERATE_VIDEO_SHIFT:g}") from exc
            if not math.isclose(shift, REGENERATE_VIDEO_SHIFT):
                raise OmniClientError(f"Open-H3-Regenerate requires flow_shift={REGENERATE_VIDEO_SHIFT:g}")
        if extra.get("latent_refine") not in (None, False):
            raise OmniClientError("Open-H3-Regenerate fixes its own refine pass; drop 'latent_refine'")
        if extra.get("sampler", "euler") != "euler":
            raise OmniClientError("Open-H3-Regenerate samples with Euler; drop 'sampler'")
        return steps


def _read_lora_patches(path: Path, *, num_blocks: int, num_refiner_blocks: int) -> dict[str, _ParamPatch]:
    patches: dict[str, _ParamPatch] = {}
    blocks_seen: dict[str, set[int]] = {}
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
        if metadata.get("format") != _LORA_FORMAT:
            raise MiniMaxH3RegenerateError(f"{path} is not a {_LORA_FORMAT!r} artifact: {metadata.get('format')!r}")
        for name in checkpoint.keys():
            split = _split_adapter_key(name.removeprefix("transformer."))
            target = None if split is None or split[1] not in ("lora_a", "lora_b") else _resolve_native_target(split[0])
            if target is None:
                raise MiniMaxH3RegenerateError(f"{path} has a tensor that names no H3 LoRA target: {name}")
            native_module, layout, block = target
            if block is not None:
                blocks_seen.setdefault(block[0], set()).add(block[1])
            patch = patches.setdefault(f"{native_module}.weight", _ParamPatch(layout=layout))
            a, b = patch.low_rank.get(layout, (None, None))
            tensor = checkpoint.get_tensor(name)
            patch.low_rank[layout] = (tensor, b) if split[1] == "lora_a" else (a, tensor)
    for native_param, patch in patches.items():
        for slot, (a, b) in patch.low_rank.items():
            if a is None or b is None or a.shape[0] != b.shape[1]:
                raise MiniMaxH3RegenerateError(f"{path} has an unpaired LoRA factor for {native_param} slot {slot!r}")
    for prefix, count in (("blocks.", num_blocks), ("token_refiner.blocks.", num_refiner_blocks)):
        if blocks_seen.get(prefix, set()) != set(range(count)):
            raise MiniMaxH3RegenerateError(
                f"{path} edits {len(blocks_seen.get(prefix, set()))} of the model's {count} {prefix}* blocks; "
                "it is not the Open-H3-Regenerate LoRA for this checkpoint"
            )
    return patches


def _check_curve_branches_are_inert(path: Path, *, num_blocks: int) -> None:
    """Hold the Sharp curve branches to contributing nothing.

    A branch adds ``up @ (down @ c(t))`` to its block's AdaLN projection, so a
    zero ``up`` factor makes it the identity whatever the curve table holds.
    """
    ups: dict[int, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
        if metadata.get("format") != _CURVE_FORMAT:
            raise MiniMaxH3RegenerateError(f"{path} is not a {_CURVE_FORMAT!r} artifact: {metadata.get('format')!r}")
        for name in checkpoint.keys():
            head, _, tail = name.removeprefix(_CURVE_PREFIX).partition(".")
            role = _CURVE_SUFFIXES.get("." + tail) if name.startswith(_CURVE_PREFIX) and head.isdigit() else None
            if role is None:
                raise MiniMaxH3RegenerateError(f"{path} has a tensor that names no AdaLN curve branch: {name}")
            if role == "up":
                ups[int(head)] = checkpoint.get_tensor(name)
    if set(ups) != set(range(num_blocks)):
        raise MiniMaxH3RegenerateError(f"{path} must carry a curve branch for each of the {num_blocks} blocks")
    active = sorted(index for index, up in ups.items() if bool(up.any()))
    if active:
        raise MiniMaxH3RegenerateError(
            f"{path} has nonzero Sharp curve branches in blocks {active[:5]}; vLLM-Omni only serves releases "
            "whose curve branches are inert (all-zero up factors)"
        )


def resolve_regenerate_adapter(od_config: Any, transformer: Any) -> MiniMaxH3RegenerateAdapter | None:
    """Claim ``--lora-path`` when it points at an Open-H3-Regenerate release."""
    lora_path = getattr(od_config, "lora_path", None)
    if isinstance(lora_path, (list, tuple)):
        if len(lora_path) != 1:
            return None
        lora_path = lora_path[0]
    if not lora_path:
        return None
    arch = transformer.arch
    return MiniMaxH3RegenerateAdapter.from_path(
        lora_path,
        head_dim=arch.attention_head_dim,
        num_blocks=arch.num_layers,
        num_refiner_blocks=arch.token_refiner_num_layers,
    )


__all__ = [
    "REGENERATE_DEFAULT_STEPS",
    "REGENERATE_DEFAULT_UPSCALE",
    "REGENERATE_SUPPORTED_TASKS",
    "REGENERATE_VIDEO_SHIFT",
    "MiniMaxH3RegenerateAdapter",
    "MiniMaxH3RegenerateError",
    "regenerate_schedule",
    "resolve_regenerate_adapter",
]
