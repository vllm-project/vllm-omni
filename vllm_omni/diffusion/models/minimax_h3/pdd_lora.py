# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Load VideoX-Fun Parallel Decoding Distillation (PDD) adapters for MiniMax-H3.

A PDD adapter (``alibaba-pai/MiniMax-H3-Acc-LoRAs``) is two things in one file:

* a rank-``r`` LoRA over the attention, MLP and block AdaLN projections of the
  DiT and the token refiner, in the Diffusers layout with ``lora_down`` /
  ``lora_up`` keys; and
* ``N`` copies of both output heads (``proj_out``, ``audio_proj_out``), one per
  interval of an ``N``-point time grid.

One forward predicts the mean velocity of the next ``L`` intervals, so a clip
takes ``N / L`` forwards. The block boundaries of that grid are the nodes of
H3's own shifted schedule at ``N / L`` steps, and an Euler step is linear in
the velocity, so the ``L`` heads of a block fuse offline into one head weighted
by their interval lengths (``videox_fun/utils/lora_utils_pdd.py``). The adapter
therefore runs on the ordinary sampler: the LoRA binds like a Turbo adapter and
each step selects its fused head.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import regex as re
import torch
from safetensors import safe_open
from vllm.logger import init_logger
from vllm.lora.lora_model import LoRAModel
from vllm.lora.peft_helper import PEFTHelper

from vllm_omni.lora.request import LoRARequest

from .lora import (
    _TURBO_EXPECTED_RAW_TARGETS,
    _TURBO_TARGET_DIMS,
    _TURBO_TARGET_PATTERN,
    _TURBO_TIME_EMBED_DIM,
    _TURBO_WEIGHTS_MAPPER,
    TurboSpec,
    _pack_h3_turbo_fc1,
)

logger = init_logger(__name__)

_DOWN, _UP = ".lora_down", ".lora_up"
# ``MiniMax-H3-<partition>-Acc-<n>Step``: the released name records the base the
# adapter was distilled from. Other exports (``pdd_ema.safetensors``) do not.
_ACC_NAME_RE = re.compile(r"^MiniMax-H3-(?P<partition>FL2VA|Ref2VA)-Acc-(?P<steps>\d+)Step\.safetensors$")
# Diffusers head name -> native projection on ``final_layer``.
_HEADS = {"proj_out": "video_out", "audio_proj_out": "audio_out"}
_BLOCK_ADALN_TARGETS = frozenset(f"transformer_blocks.{index}.adaln_proj.linear" for index in range(50))
# The PDD adapter leaves the output norm's AdaLN alone, so only block AdaLN
# projections are wrapped.
_PDD_TARGET_PATTERN = rf"(?:{_TURBO_TARGET_PATTERN}|^transformer\.blocks\.\d+\.adaln_proj\.linear$)"
# Three modalities of six hidden-width vectors each.
_BLOCK_ADALN_OUT = 18 * 5376


@dataclass(frozen=True)
class PDDSpec(TurboSpec):
    """A PDD adapter's sampling contract, plus the heads its steps project with.

    It is a :class:`TurboSpec` because a request is held to it the same way:
    one artifact, its own step count and the base model's flow shifts.
    ``pdd_heads`` holds ``video_out`` / ``audio_out`` weights and biases with one
    row per denoise step.
    """

    pdd_heads: Mapping[str, torch.Tensor] = field(default_factory=dict, compare=False, repr=False)


def pdd_block_plan(shift: float, grid_steps: int, block_size: int) -> torch.Tensor:
    """Per-step head weights, ``(grid_steps / block_size, grid_steps)``.

    Row ``k`` weights the heads of block ``k`` by their interval lengths on the
    shifted grid ``t_n = 1 - shift * s / (1 + (shift - 1) * s)``, ``s = 1 - n / N``,
    and is zero elsewhere, so it yields the block's mean velocity.
    """
    sigma = torch.linspace(1.0, 0.0, grid_steps + 1, dtype=torch.float64)
    lengths = (1.0 - shift * sigma / (1.0 + (shift - 1.0) * sigma)).diff()
    plan = torch.zeros(grid_steps // block_size, grid_steps, dtype=torch.float64)
    for step in range(plan.shape[0]):
        block = slice(step * block_size, (step + 1) * block_size)
        plan[step, block] = lengths[block] / lengths[block].sum()
    return plan


def _positive_int(metadata: Mapping[str, str], key: str) -> int:
    try:
        value = int(metadata[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"MiniMax-H3 PDD adapter metadata needs an integer {key}, got {metadata.get(key)!r}") from exc
    if value <= 0:
        raise ValueError(f"MiniMax-H3 PDD adapter metadata needs a positive {key}, got {value}")
    return value


def _select_pdd_file(artifact_path: str | Path) -> tuple[Path, Mapping[str, str]] | None:
    """The one safetensors file under ``artifact_path`` that declares a PDD grid, with its metadata."""
    path = Path(artifact_path)
    candidates = [path] if path.is_file() else sorted(path.glob("*.safetensors")) if path.is_dir() else []
    found = []
    for candidate in candidates:
        if candidate.suffix != ".safetensors":
            continue
        try:
            with safe_open(candidate, framework="pt", device="cpu") as checkpoint:
                metadata = checkpoint.metadata() or {}
        except Exception:
            # Probing only: a file this loader cannot read is another loader's to accept or reject.
            continue
        if "pdd_num_steps" in metadata:
            found.append((candidate, metadata))
    if len(found) > 1:
        raise ValueError(
            f"{path} holds {len(found)} MiniMax-H3 PDD adapters ({[c.name for c, _ in found[:4]]}); "
            "point --lora-path at one file."
        )
    return found[0] if found else None


def _fuse_heads(
    checkpoint, *, grid_steps: int, block_size: int, shifts: Mapping[str, float]
) -> dict[str, torch.Tensor]:
    heads: dict[str, torch.Tensor] = {}
    for raw, native in _HEADS.items():
        weight = checkpoint.get_tensor(f"{raw}.weight")
        bias = checkpoint.get_tensor(f"{raw}.bias")
        if weight.ndim != 3 or weight.shape[0] != grid_steps or tuple(bias.shape) != tuple(weight.shape[:2]):
            raise ValueError(
                f"MiniMax-H3 PDD {raw} must hold {grid_steps} heads as (steps, out, in) weights and "
                f"(steps, out) biases, got {tuple(weight.shape)} and {tuple(bias.shape)}"
            )
        plan = pdd_block_plan(shifts[native], grid_steps, block_size)
        heads[f"{native}.weight"] = torch.einsum("sn,noi->soi", plan, weight.double()).float()
        heads[f"{native}.bias"] = torch.einsum("sn,no->so", plan, bias.double()).float()
    return heads


def _convert_lora(checkpoint, *, rank: int) -> dict[str, torch.Tensor]:
    """Validate the adapter's LoRA pairs and rename them for the LoRA manager."""
    tensors: dict[str, torch.Tensor] = {}
    sides: dict[str, set[str]] = {}
    for name in checkpoint.keys():
        if name.endswith(_DOWN):
            target, side = name[: -len(_DOWN)], "A"
        elif name.endswith(_UP):
            target, side = name[: -len(_UP)], "B"
        elif name.rsplit(".", 1)[0] in _HEADS:
            continue
        else:
            raise ValueError(f"Unconsumed MiniMax-H3 PDD tensor: {name!r}")
        if target in _BLOCK_ADALN_TARGETS:
            input_dim, output_dim = _TURBO_TIME_EMBED_DIM, _BLOCK_ADALN_OUT
        else:
            suffix = next((s for s in _TURBO_TARGET_DIMS if target.endswith(s)), None)
            if target not in _TURBO_EXPECTED_RAW_TARGETS or suffix is None:
                raise ValueError(f"Unsupported MiniMax-H3 PDD target: {target!r}")
            input_dim, output_dim = _TURBO_TARGET_DIMS[suffix]
        tensor = checkpoint.get_tensor(name)
        expected = (rank, input_dim) if side == "A" else (output_dim, rank)
        if tuple(tensor.shape) != expected:
            raise ValueError(
                f"MiniMax-H3 PDD tensor has invalid shape: {name}={tuple(tensor.shape)}, expected={expected}"
            )
        if side == "B" and target.endswith("ff.net.0.proj"):
            # Diffusers stores the MLP's value half first; the native fc1 stores its gate first.
            value, gate = tensor.chunk(2, dim=0)
            tensor = torch.cat((gate, value), dim=0).contiguous()
        sides.setdefault(target, set()).add(side)
        tensors[f"{target}.lora_{side}.weight"] = tensor
    incomplete = sorted(target for target, found in sides.items() if found != {"A", "B"})
    if incomplete:
        raise ValueError(f"Incomplete MiniMax-H3 PDD LoRA pairs: {incomplete[:5]}")
    expected_targets = _TURBO_EXPECTED_RAW_TARGETS | _BLOCK_ADALN_TARGETS
    missing, unexpected = sorted(expected_targets - sides.keys()), sorted(sides.keys() - expected_targets)
    if missing or unexpected:
        raise ValueError(
            "MiniMax-H3 PDD target set does not match the released adapter layout: "
            f"missing={len(missing)} {missing[:5]}, unexpected={len(unexpected)} {unexpected[:5]}"
        )
    return tensors


def load_minimax_h3_pdd_lora(
    *,
    partition: str,
    lora_request: LoRARequest,
    lora_path: str | Path,
    dtype: torch.dtype,
    video_shift: float,
    audio_shift: float,
    unsupported_mode: str | None = None,
) -> tuple[LoRAModel, PEFTHelper, PDDSpec] | None:
    """Load a PDD adapter, or return ``None`` when ``lora_path`` is not one.

    ``video_shift`` and ``audio_shift`` are the base checkpoint's flow shifts:
    the heads are fused for that schedule and requests are held to it.
    """
    selected = _select_pdd_file(lora_path)
    if selected is None:
        return None
    lora_file, metadata = selected
    grid_steps = _positive_int(metadata, "pdd_num_steps")
    block_size = _positive_int(metadata, "pdd_block_size")
    rank = _positive_int(metadata, "lora_rank")
    if grid_steps % block_size:
        raise ValueError(f"MiniMax-H3 PDD grid of {grid_steps} steps is not a multiple of block size {block_size}")
    try:
        alpha = float(metadata["lora_alpha"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"MiniMax-H3 PDD adapter metadata needs a numeric lora_alpha, got {metadata.get('lora_alpha')!r}"
        ) from exc
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError(f"MiniMax-H3 PDD lora_alpha must be a positive number, got {alpha!r}")

    named = _ACC_NAME_RE.match(lora_file.name)
    served = "ref2va" if partition == "ref2va" else "fl2va"
    if named is not None and named["partition"].lower() != served:
        raise ValueError(
            f"{lora_file.name} was distilled from the {named['partition']} DiT; start the server with "
            f"--task-type {named['partition'].lower()} (task_type={partition!r} serves a DiT it cannot bind to)"
        )
    if partition not in ("fl2va", "ref2va"):
        # The adapter binds to ``transformer`` and swaps that DiT's heads; a combined server runs two.
        raise ValueError(f"{lora_file.name} needs a single-partition server: --task-type fl2va or --task-type ref2va")
    if unsupported_mode is not None:
        raise ValueError(f"MiniMax-H3 PDD adapters do not support {unsupported_mode}")

    with safe_open(lora_file, framework="pt", device="cpu") as checkpoint:
        tensors = _convert_lora(checkpoint, rank=rank)
        heads = _fuse_heads(
            checkpoint,
            grid_steps=grid_steps,
            block_size=block_size,
            shifts={"video_out": video_shift, "audio_out": audio_shift},
        )
    peft_helper = PEFTHelper.from_dict({"r": rank, "lora_alpha": alpha, "target_modules": _PDD_TARGET_PATTERN})
    lora_model = LoRAModel.from_lora_tensors(
        lora_model_id=lora_request.lora_int_id,
        tensors=tensors,
        peft_helper=peft_helper,
        device="cpu",
        dtype=dtype,
        weights_mapper=_TURBO_WEIGHTS_MAPPER,
    )
    _pack_h3_turbo_fc1(lora_model)
    spec = PDDSpec(
        filename=lora_file.name,
        task_family="ref2v" if served == "ref2va" else "fl2v",
        denoise_steps=grid_steps // block_size,
        video_shift=video_shift,
        audio_shift=audio_shift,
        rank=rank,
        alpha=alpha,
        pdd_heads=heads,
    )
    logger.info(
        "MiniMax-H3 PDD adapter %s: task=%s, %d forwards over a %d-point grid, rank=%d alpha=%g, flow_shift=%g/%g",
        spec.filename,
        spec.task_family,
        spec.denoise_steps,
        grid_steps,
        rank,
        alpha,
        video_shift,
        audio_shift,
    )
    return lora_model, peft_helper, spec
