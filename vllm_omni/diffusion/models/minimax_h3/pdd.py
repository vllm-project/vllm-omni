# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""PDD (Parallel Decoding Distillation, arXiv 2607.26004) loader for MiniMax-H3.

The Alibaba ``MiniMax-H3-{Ref2VA,FL2VA}-Acc-8Step`` artifacts are not PEFT
LoRAs. The backbone (DiT blocks + token refiner) receives a rank-64 LoRA delta,
and the two final heads (``final_layer.video_out``, ``final_layer.audio_out``)
are replaced by a bank of ``num_steps`` heads fused per-step by a
block-conditioned plan. Running ``num_steps/block_size`` (= 8) DiT evaluations
covers the full trajectory, versus 28 in the baseline.

Alibaba released **one artifact per partition**: ``Ref2VA`` distils the Ref2VA
DiT, ``FL2VA`` distils the FL2VA DiT (which also serves T2VA). The two files are
structurally identical -- same 728 tensors, same shapes, same metadata keys, no
``key_format`` tag -- but hold genuinely different weights, so they are *not*
interchangeable and the only discriminator a caller gives us is the file name.
Each one must land on the DiT it was distilled for: in a ``combined``
deployment ``transformer`` holds FL2VA and ``transformers_ref`` holds Ref2VA, so
a component-blind target pattern silently binds the trunk delta to the wrong
DiT while the head bank goes to the right one. ``PDDVariant`` therefore carries
the task set, the acceptable partitions, and the DiT component name, and the
target pattern / weights mapper are built per-variant.

This module provides:

* ``PDDVariant`` -- which release an artifact is, and where it must bind.
* ``PDDConfig`` -- parsed artifact metadata (incl. variant + DiT component).
* ``PDDParallelHead`` -- drop-in replacement for ``ColumnParallelLinear`` on
  the two final projections. Holds a 32-copy fp32 bank, fuses one block per
  forward with ``torch.einsum``, and honours tensor-parallel sharding +
  ``gather_output`` exactly like the layer it replaces.
* ``load_minimax_h3_pdd_lora(...)`` -- validates the artifact, builds the
  trunk ``LoRAModel`` (through the legacy diffusion-LoRA manager, mirroring
  the turbo path), and returns a ``PDDAdapter`` handle the pipeline uses to
  arm the fused heads on every step.

Design notes
------------

* The head bank is loaded as real ``nn.Parameter`` fp32 weights inside
  ``PDDParallelHead`` -- NOT through the generic LoRA wrapper. The generic
  wrapper allocates bf16 buffers, which loses precision on a layer the base
  model deliberately keeps in fp32 (see ``MINIMAX_H3_FP32_PARAM_NAMES`` and
  ``MiniMaxH3FinalLayer.forward`` which upcasts to fp32 before the heads).
* The plan is armed *per request* from the real denoise ``step`` index, not
  from a forward-count hook (which desyncs if any auxiliary forward runs).
* Packed (co-batched) step execution is disabled while PDD is active because
  requests at different step indices need different fused heads; the
  pipeline's existing per-request fallback path handles this case.
* The sigma schedule is the standard time-shifted grid with ``num_steps=9``
  (8 NFE + terminal zero). The production defaults ``video_shift=12.0`` and
  ``audio_shift=3.0`` are already the released PDD shifts, and numerical
  checks (see docs/10) verified vllm-omni's schedule equals the PDD block
  boundaries to ~2.5e-8 -- the schedule itself needs no changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import regex as re
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.lora.lora_model import LoRAModel
from vllm.lora.lora_weights import PackedLoRALayerWeights
from vllm.lora.peft_helper import PEFTHelper
from vllm.model_executor.models.utils import WeightsMapper

from vllm_omni.lora.request import LoRARequest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PDD_RANK = 64
_PDD_ALPHA = 64.0
_PDD_NUM_STEPS = 32
_PDD_BLOCK_SIZE = 4
_PDD_NFE = _PDD_NUM_STEPS // _PDD_BLOCK_SIZE  # 8
_PDD_SIGMA_POINTS = _PDD_NFE + 1  # 9 sigma points -> 8 Euler steps
_PDD_VIDEO_SHIFT = 12.0
_PDD_AUDIO_SHIFT = 3.0
_PDD_HIDDEN_SIZE = 5376
_PDD_VIDEO_OUT_DIM = 96
_PDD_AUDIO_OUT_DIM = 32
_PDD_TIME_EMBED_DIM = 2688
_PDD_ATTENTION_INNER_SIZE = 7168
_PDD_FFN_HIDDEN_SIZE = 14336
_PDD_FINAL_ADALN_OUT = 2 * _PDD_HIDDEN_SIZE  # final_layer.adaln_proj: expand_ratio=2, modality_num=1
_PDD_BLOCK_ADALN_OUT = 18 * _PDD_HIDDEN_SIZE  # DiT block adaln: expand_ratio=6 * modality_num=3
_PDD_LORA_DOWN_SUFFIX = ".lora_down"
_PDD_LORA_UP_SUFFIX = ".lora_up"


# ---------------------------------------------------------------------------
# Released variants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PDDVariant:
    """One released PDD artifact: which tasks it serves and where it binds.

    ``partitions`` are the deployment partitions that actually hold the DiT this
    artifact was distilled for -- a Ref2VA artifact is meaningless on an
    FL2VA-only server and vice versa.
    """

    name: str
    filename: str
    key_format: str
    tasks: frozenset[str]
    partitions: frozenset[str]

    def dit_component(self, partition: str) -> str:
        """Name of the pipeline attribute holding the DiT this artifact belongs to.

        A ``combined`` deployment loads FL2VA into ``transformer`` and Ref2VA
        into ``transformers_ref`` (see the pipeline's ``weights_sources``); a
        single-partition deployment always puts its own DiT at ``transformer``.
        """
        if self.name == "ref2va" and partition == "combined":
            return "transformers_ref"
        return "transformer"


_PDD_VARIANTS: tuple[PDDVariant, ...] = (
    PDDVariant(
        name="ref2va",
        filename="MiniMax-H3-Ref2VA-Acc-8Step.safetensors",
        key_format="minimax-h3-pdd-ref2va",
        tasks=frozenset({"ref2va"}),
        partitions=frozenset({"ref2va", "combined"}),
    ),
    PDDVariant(
        name="fl2va",
        filename="MiniMax-H3-FL2VA-Acc-8Step.safetensors",
        key_format="minimax-h3-pdd-fl2va",
        tasks=frozenset({"fl2va", "t2va"}),
        partitions=frozenset({"fl2va", "combined"}),
    ),
)
_PDD_VARIANTS_BY_NAME = {v.name: v for v in _PDD_VARIANTS}
_PDD_VARIANTS_BY_FILENAME = {v.filename: v for v in _PDD_VARIANTS}
_PDD_VARIANTS_BY_KEY_FORMAT = {v.key_format: v for v in _PDD_VARIANTS}

# Trunk LoRA raw target suffixes (diffusers naming) and their expected shapes
# (input_dim, output_dim) on the *down* (rank, in) and *up* (out, rank) sides.
_PDD_TRUNK_TARGET_SUFFIXES = (
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
    "adaln_proj.linear",
)
_PDD_TRUNK_TARGET_DIMS = {
    # adaln_proj.linear maps time_embed_dim (2688) -> out; the artifact targets
    # DiT-block adaln (out=18*5376=96768) but not final-layer adaln. The input
    # dimension is shared; output differs between block/final.
    "adaln_proj.linear": (_PDD_TIME_EMBED_DIM, _PDD_BLOCK_ADALN_OUT),
    "attn.to_q": (_PDD_HIDDEN_SIZE, _PDD_ATTENTION_INNER_SIZE),
    "attn.to_k": (_PDD_HIDDEN_SIZE, _PDD_ATTENTION_INNER_SIZE),
    "attn.to_v": (_PDD_HIDDEN_SIZE, _PDD_ATTENTION_INNER_SIZE),
    "attn.to_out.0": (_PDD_ATTENTION_INNER_SIZE, _PDD_HIDDEN_SIZE),
    "ff.net.0.proj": (_PDD_HIDDEN_SIZE, 2 * _PDD_FFN_HIDDEN_SIZE),
    "ff.net.2": (_PDD_FFN_HIDDEN_SIZE, _PDD_HIDDEN_SIZE),
}


def _build_trunk_raw_targets() -> frozenset[str]:
    # DiT blocks (50) have all 7 targets (incl. adaln_proj.linear). Token
    # refiner blocks (2) are plain pre-norm blocks (MiniMaxH3TokenRefinerBlock)
    # and have no adaln_proj -- only attn + ff.
    targets: set[str] = set()
    for i in range(50):
        for s in _PDD_TRUNK_TARGET_SUFFIXES:
            targets.add(f"transformer_blocks.{i}.{s}")
    for i in range(2):
        for s in _PDD_TRUNK_TARGET_SUFFIXES:
            if s == "adaln_proj.linear":
                continue
            targets.add(f"token_refiner.refiner_blocks.{i}.{s}")
    return frozenset(targets)


_PDD_TRUNK_RAW_TARGETS = _build_trunk_raw_targets()
# Leaf-name alternation shared by every variant's target pattern.
_PDD_TRUNK_LEAF_PATTERN = (
    r"(?:token_refiner\.blocks|blocks)\.\d+\."
    r"(?:attn\.(?:to_q|to_k|to_v|qkv_proj|out_proj)|mlp\.(?:fc1|fc2)|adaln_proj\.linear)$"
)


def _trunk_target_pattern(dit_component: str) -> str:
    """Anchored regex covering one DiT's trunk, excluding its final layer / heads.

    The manager matches this against ``f"{component}.{module_name}"``, so the
    component anchor is what keeps a Ref2VA delta off the FL2VA DiT (and vice
    versa) in a combined deployment. Layers on the other DiT match nothing and
    are ``reset_lora``'d, which is exactly the desired outcome.
    """
    return rf"^{re.escape(dit_component)}\." + _PDD_TRUNK_LEAF_PATTERN


def _pdd_weights_mapper(dit_component: str) -> WeightsMapper:
    """Map diffusers names to component-qualified vllm-omni LoRA keys.

    Same substring replacements as turbo, plus the lora_down/up -> lora_A/B
    rename (the PDD artifact uses a different suffix convention than the peft
    default), plus a prefix that qualifies every key with the owning DiT.
    Without the prefix the keys are component-relative and the manager's
    fallback lookup happily binds them to whichever DiT it walks first.
    ``orig_to_new_prefix`` is applied after ``orig_to_new_substr``, so it sees
    the already-renamed ``blocks.``/``token_refiner.`` roots.
    """
    return WeightsMapper(
        orig_to_new_substr={
            "token_refiner.refiner_blocks.": "token_refiner.blocks.",
            "transformer_blocks.": "blocks.",
            ".attn.to_out.0.": ".attn.out_proj.",
            ".ff.net.0.proj.": ".mlp.fc1.",
            ".ff.net.2.": ".mlp.fc2.",
            ".lora_down": ".lora_A",
            ".lora_up": ".lora_B",
        },
        orig_to_new_prefix={
            "blocks.": f"{dit_component}.blocks.",
            "token_refiner.": f"{dit_component}.token_refiner.",
        },
    )


# Defaults for the DiT that a single-partition deployment serves from
# ``transformer``. The loader builds the variant's own pattern/mapper; these
# stay for tests and for callers that only need the leaf shape.
_PDD_TRUNK_TARGET_PATTERN = _trunk_target_pattern("transformer")
_PDD_WEIGHTS_MAPPER = _pdd_weights_mapper("transformer")
# After substring remap, the valid leaf target names on the vllm-omni side.
_PDD_VALID_VLLM_LEAVES = frozenset({"to_q", "to_k", "to_v", "out_proj", "fc1", "fc2", "linear"})


# ---------------------------------------------------------------------------
# Time-grid / plan math (matches the reference minimax_h3_pdd.py exactly)
# ---------------------------------------------------------------------------


def _shifted_sigma(shift: float, sigma: torch.Tensor) -> torch.Tensor:
    return shift * sigma / (1 + (shift - 1) * sigma)


def _pdd_time_grid(shift: float, num_steps: int) -> torch.Tensor:
    """Ascending t-grid 0 = t_0 < ... < t_N = 1 for one modality."""
    sigma = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float64)
    return 1.0 - _shifted_sigma(shift, sigma)


def _pdd_sampling_plan(step_sizes: torch.Tensor, start: int, block_size: int) -> torch.Tensor:
    """Mean-velocity weights over one block; rows sum to 1."""
    plan = torch.zeros(1, step_sizes.shape[0], dtype=step_sizes.dtype, device=step_sizes.device)
    span = step_sizes[start : start + block_size].sum()
    plan[0, start : start + block_size] = step_sizes[start : start + block_size] / span
    return plan


def _build_pdd_plans(
    num_steps: int, block_size: int, video_shift: float, audio_shift: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute the ``(nfe, num_steps)`` plan matrices for both modalities.

    Row ``k`` is the convex-combination weight vector used when denoise step
    ``k`` fires. Storing all rows up front keeps the per-step arming a single
    index lookup.
    """
    nfe = num_steps // block_size
    v_steps = _pdd_time_grid(video_shift, num_steps).diff()
    a_steps = _pdd_time_grid(audio_shift, num_steps).diff()
    v_plans = torch.cat([_pdd_sampling_plan(v_steps, k * block_size, block_size) for k in range(nfe)], dim=0)
    a_plans = torch.cat([_pdd_sampling_plan(a_steps, k * block_size, block_size) for k in range(nfe)], dim=0)
    return v_plans.float(), a_plans.float()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PDDConfig:
    num_steps: int = _PDD_NUM_STEPS
    block_size: int = _PDD_BLOCK_SIZE
    rank: int = _PDD_RANK
    alpha: float = _PDD_ALPHA
    video_shift: float = _PDD_VIDEO_SHIFT
    audio_shift: float = _PDD_AUDIO_SHIFT
    filename: str = _PDD_VARIANTS_BY_NAME["ref2va"].filename
    # Which release this is, which tasks it may serve, and which DiT its trunk
    # delta + head bank belong to. The pipeline rejects a task outside ``tasks``
    # rather than quietly accelerating it with the wrong distillation.
    variant: str = "ref2va"
    tasks: frozenset[str] = frozenset({"ref2va"})
    dit_component: str = "transformer"

    @property
    def nfe(self) -> int:
        return self.num_steps // self.block_size

    @property
    def sigma_points(self) -> int:
        return self.nfe + 1

    @property
    def trunk_target_pattern(self) -> str:
        return _trunk_target_pattern(self.dit_component)

    def plans(self) -> tuple[torch.Tensor, torch.Tensor]:
        return _build_pdd_plans(self.num_steps, self.block_size, self.video_shift, self.audio_shift)


def _parse_pdd_metadata(
    raw: dict[str, str], variant: PDDVariant | None = None, partition: str = "combined"
) -> PDDConfig:
    def _get_int(key: str, default: int) -> int:
        v = raw.get(key)
        if v is None:
            return default
        try:
            return int(v)
        except ValueError as exc:
            raise ValueError(f"PDD metadata {key} must be int, got {v!r}") from exc

    def _get_float(key: str, default: float) -> float:
        v = raw.get(key)
        if v is None:
            return default
        try:
            return float(v)
        except ValueError as exc:
            raise ValueError(f"PDD metadata {key} must be numeric, got {v!r}") from exc

    num_steps = _get_int("pdd_num_steps", _PDD_NUM_STEPS)
    block_size = _get_int("pdd_block_size", _PDD_BLOCK_SIZE)
    rank = _get_int("lora_rank", _PDD_RANK)
    alpha = _get_float("lora_alpha", _PDD_ALPHA)
    if block_size < 1 or num_steps % block_size != 0:
        raise ValueError(f"pdd_num_steps={num_steps} must be divisible by pdd_block_size={block_size}")
    if rank <= 0 or alpha <= 0:
        raise ValueError(f"PDD rank/alpha must be positive, got r={rank} alpha={alpha}")
    # Sanity-check the target list mentions adaln (older/different LoRA would be a red flag)
    targets = raw.get("lora_targets", "")
    if "adaln_proj.linear" not in targets:
        raise ValueError(f"PDD artifact must target adaln_proj.linear; got lora_targets={targets!r}")
    if variant is None:
        variant = _PDD_VARIANTS_BY_NAME["ref2va"]
    return PDDConfig(
        num_steps=num_steps,
        block_size=block_size,
        rank=rank,
        alpha=alpha,
        filename=variant.filename,
        variant=variant.name,
        tasks=variant.tasks,
        dit_component=variant.dit_component(partition),
    )


# ---------------------------------------------------------------------------
# PDD parallel head (fp32, TP-sharded, all-gather)
# ---------------------------------------------------------------------------


class PDDParallelHead(nn.Module):
    """Drop-in replacement for ``ColumnParallelLinear`` on the two final heads.

    Holds a ``(num_steps, out_local, in_features)`` fp32 weight bank and a
    ``(num_steps, out_local)`` bias bank, pre-sharded for the local TP rank.
    On each forward, fuses the rows indicated by ``self.plan`` (a
    ``(1, num_steps)`` vector set by :meth:`set_plan`) via einsum and applies
    the resulting fused linear once -- exactly one matmul per forward, same
    cost as the unmodified head plus a 32-element weighting.

    Parameters
    ----------
    source:
        The original ``ColumnParallelLinear`` module being replaced. Its
        existing weight/bias are used to initialize every copy of the bank
        (matching the reference ``MiniMaxH3ParallelHead`` semantics); the
        artifact overwrites them at load time.
    num_steps:
        Number of PDD intervals (32).
    """

    def __init__(self, source: nn.Module, num_steps: int) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.in_features = (
            int(source.input_size_per_partition)
            if hasattr(source, "input_size_per_partition")
            else int(source.in_features)
        )
        # ColumnParallelLinear splits output dim across TP ranks.
        if hasattr(source, "output_size_per_partition"):
            self.out_features_local = int(source.output_size_per_partition)
            self.tp_rank = int(getattr(source, "tp_rank", 0))
            # Note: read tp_size off the source rather than via
            # ``getattr(source, "tp_size", get_tensor_model_parallel_world_size())``
            # -- a getattr default is evaluated eagerly, which would query the
            # (possibly uninitialized) TP group even when the attribute exists.
            tp_size = getattr(source, "tp_size", None)
            if tp_size is None:
                tp_size = get_tensor_model_parallel_world_size()
            self.tp_size = int(tp_size)
            self.gather_output = bool(getattr(source, "gather_output", True))
        else:
            # Fallback for plain nn.Linear (single-process tests).
            self.out_features_local = int(source.out_features)
            self.tp_rank = 0
            self.tp_size = 1
            self.gather_output = False

        # Weight: (num_steps, out_local, in_features) fp32, same convention
        # as the reference implementation so checkpoint loading is 1:1.
        # The source ColumnParallelLinear already holds the per-rank shard
        # (shape [out_local, in]) after TP partitioning; copy it directly.
        # The only time source.weight is full-sized is in single-process CPU
        # tests (nn.Linear path below), where tp_size=1 and narrow is a no-op.
        init_w = source.weight.detach()
        if init_w.ndim == 2 and init_w.shape[0] > self.out_features_local:
            init_w = init_w.narrow(0, self.tp_rank * self.out_features_local, self.out_features_local)
        self.register_buffer("base_weight", init_w.to(torch.float32).clone(), persistent=False)
        self.weight = nn.Parameter(init_w.to(torch.float32)[None].repeat(num_steps, 1, 1).clone())
        if source.bias is not None:
            init_b = source.bias.detach()
            if init_b.shape[0] > self.out_features_local:
                init_b = init_b.narrow(0, self.tp_rank * self.out_features_local, self.out_features_local)
            self.register_buffer("base_bias", init_b.to(torch.float32).clone(), persistent=False)
            self.bias = nn.Parameter(init_b.to(torch.float32)[None].repeat(num_steps, 1).clone())
        else:
            self.register_parameter("bias", None)
            self.register_buffer("base_bias", None, persistent=False)

        # Keep the idle plan initialized, but use the saved base weights until
        # the pipeline explicitly arms a distilled head before forward.
        # The buffer must live on the same device as the bank: heads are
        # installed after the model is already on GPU, so a default-device
        # (CPU) buffer would make the fusing einsum a cross-device bmm.
        self.register_buffer(
            "plan",
            torch.zeros(1, num_steps, dtype=torch.float32, device=self.weight.device),
            persistent=False,
        )
        self.plan[0, 0] = 1.0
        self._use_base_head = True

    def set_plan(self, plan: torch.Tensor) -> None:
        if plan.ndim != 2 or plan.shape[1] != self.num_steps:
            raise ValueError(f"PDD plan must be (num_directions, {self.num_steps}), got {tuple(plan.shape)}")
        self.plan.copy_(plan.to(device=self.weight.device, dtype=torch.float32))
        self._use_base_head = False

    def reset_plan(self) -> None:
        """Use the saved base head; artifact head 0 is also distilled."""
        self._use_base_head = True
        self.plan.zero_()
        self.plan[0, 0] = 1.0

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Returns ``(output, None)`` to mimic ``ColumnParallelLinear(return_bias=True)``."""
        plan = self.plan
        # Fuse weights: (out_local, in_features)
        if self._use_base_head:
            w, b = self.base_weight, self.base_bias
        else:
            w = torch.einsum("pn,noi->oi", plan, self.weight)
            b = None if self.bias is None else torch.einsum("pn,no->o", plan, self.bias)
        # x is fp32 (final_layer.forward upcasts h before calling us).
        out_parallel = F.linear(x, w, b)
        if self.gather_output and self.tp_size > 1:
            out = tensor_model_parallel_all_gather(out_parallel)
        else:
            out = out_parallel
        return out, None


# ---------------------------------------------------------------------------
# Artifact parsing
# ---------------------------------------------------------------------------


def _select_pdd_file(artifact_path: str | Path) -> Path | None:
    path = Path(artifact_path)
    if path.is_file():
        return path if path.suffix == ".safetensors" else None
    if not path.is_dir():
        return None
    present = [path / v.filename for v in _PDD_VARIANTS if (path / v.filename).is_file()]
    if len(present) > 1:
        # The release directory ships both variants. Picking one by iteration
        # order would silently accelerate half the traffic with the wrong
        # distillation, so make the caller name the file.
        raise ValueError(
            f"{path} contains {len(present)} MiniMax-H3 PDD artifacts "
            f"({sorted(p.name for p in present)}); point lora.path at a single file so "
            "the task it was distilled for is unambiguous"
        )
    return present[0] if present else None


def _has_pdd_head_bank(checkpoint) -> bool:
    """Content test for PDD-ness: a 3-D per-step bank on both final heads.

    Deliberately *not* a filename test -- the released artifacts carry no
    ``key_format`` and a caller passing a correctly-shaped bank under any name
    must still take the PDD path. Falling through to the generic PEFT loader
    would drop the head bank while the caller still pins 9 steps, which
    produces garbage rather than an error.
    """
    keys = set(checkpoint.keys())
    if "proj_out.weight" not in keys or "audio_proj_out.weight" not in keys:
        return False
    # get_slice avoids materializing the bank just to read its rank.
    return len(checkpoint.get_slice("proj_out.weight").get_shape()) == 3


def _identify_pdd_variant(lora_file: Path, metadata: dict[str, str]) -> PDDVariant:
    """Decide which release an artifact is, so we know which DiT it binds to."""
    key_format = metadata.get("key_format")
    if key_format:
        variant = _PDD_VARIANTS_BY_KEY_FORMAT.get(key_format)
        if variant is None:
            raise ValueError(
                f"unknown MiniMax-H3 PDD key_format {key_format!r}; expected one of "
                f"{sorted(_PDD_VARIANTS_BY_KEY_FORMAT)}"
            )
        return variant
    variant = _PDD_VARIANTS_BY_FILENAME.get(lora_file.name)
    if variant is None:
        raise ValueError(
            f"{lora_file.name!r} holds a PDD head bank but carries no key_format metadata "
            f"and its name is not a known release ({sorted(_PDD_VARIANTS_BY_FILENAME)}). "
            "The Ref2VA and FL2VA artifacts are structurally identical, so the file name is "
            "the only signal for which DiT they belong to -- rename the file to the released "
            "name or add key_format to its metadata."
        )
    return variant


def _validate_and_convert_tensors(
    checkpoint, cfg: PDDConfig
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Split the artifact into trunk LoRA tensors and head-bank tensors.

    Returns ``(trunk_tensors, head_weights, head_biases)`` where:

    * ``trunk_tensors`` -- dict keyed by the *original* diffusers tensor name,
      ready for ``LoRAModel.from_lora_tensors`` (the weights mapper renames
      them to vllm-omni conventions at load time).
    * ``head_weights`` -- keys ``"video_out"``/``"audio_out"``, values shaped
      ``(num_steps, out_features, in_features)`` fp32.
    * ``head_biases`` -- same keys, values shaped ``(num_steps, out_features)`` fp32.
    """
    trunk: dict[str, torch.Tensor] = {}
    head_weights: dict[str, torch.Tensor] = {}
    head_biases: dict[str, torch.Tensor] = {}
    pairs: dict[str, set[str]] = {}
    raw_targets: set[str] = set()

    for name in checkpoint.keys():
        # Head bank keys are bare (no block prefix).
        if name in ("proj_out.weight", "audio_proj_out.weight"):
            key = "video_out" if name == "proj_out.weight" else "audio_out"
            t = checkpoint.get_tensor(name)
            expected = (
                cfg.num_steps,
                _PDD_VIDEO_OUT_DIM if key == "video_out" else _PDD_AUDIO_OUT_DIM,
                _PDD_HIDDEN_SIZE,
            )
            if tuple(t.shape) != expected:
                raise ValueError(f"PDD head {name} shape mismatch: expected {expected}, got {tuple(t.shape)}")
            head_weights[key] = t.to(torch.float32)
            continue
        if name in ("proj_out.bias", "audio_proj_out.bias"):
            key = "video_out" if name == "proj_out.bias" else "audio_out"
            t = checkpoint.get_tensor(name)
            expected = (cfg.num_steps, _PDD_VIDEO_OUT_DIM if key == "video_out" else _PDD_AUDIO_OUT_DIM)
            if tuple(t.shape) != expected:
                raise ValueError(f"PDD head bias {name} shape mismatch: expected {expected}, got {tuple(t.shape)}")
            head_biases[key] = t.to(torch.float32)
            continue

        if name.endswith(_PDD_LORA_DOWN_SUFFIX):
            raw_target = name[: -len(_PDD_LORA_DOWN_SUFFIX)]
            side = "a"
            # The PDD artifact drops the trailing `.weight` (e.g. `attn.to_q.lora_down`)
            # but vLLM's LoRA key parser requires the qualified weight name ending in
            # `.lora_A.weight` -- append it so the mapper produces canonical keys.
            dict_name = name + ".weight"
        elif name.endswith(_PDD_LORA_UP_SUFFIX):
            raw_target = name[: -len(_PDD_LORA_UP_SUFFIX)]
            side = "b"
            dict_name = name + ".weight"
        else:
            raise ValueError(f"Unconsumed PDD tensor: {name!r}")
        raw_targets.add(raw_target)

        suffix = next((s for s in _PDD_TRUNK_TARGET_SUFFIXES if raw_target.endswith(s)), None)
        if suffix is None:
            raise ValueError(f"PDD LoRA targets an unsupported module: {raw_target!r}")

        # After WeightsMapper the leaf name is the vllm-omni leaf; validate it
        # is one we expect.  Note we pass dict_name (with the trailing
        # `.weight`) so the mapper emits keys parse_fine_tuned_lora_name accepts.
        mapped_name = _PDD_WEIGHTS_MAPPER.apply_list([dict_name])[0]
        mapped_target = mapped_name.rsplit(".lora_", 1)[0]
        leaf = mapped_target.rsplit(".", 1)[-1]
        if leaf not in _PDD_VALID_VLLM_LEAVES:
            raise ValueError(f"Unsupported PDD trunk leaf: {raw_target!r} (mapped to {mapped_target!r})")

        sides = pairs.setdefault(mapped_target, set())
        if side in sides:
            raise ValueError(f"Duplicate PDD tensor for {mapped_target}.{side}")
        sides.add(side)

        tensor = checkpoint.get_tensor(name)
        if tensor.ndim != 2:
            raise ValueError(f"PDD trunk tensors must be matrices, got {name}={tuple(tensor.shape)}")

        input_dim, output_dim = _PDD_TRUNK_TARGET_DIMS[suffix]
        expected_shape = (cfg.rank, input_dim) if side == "a" else (output_dim, cfg.rank)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"PDD tensor {name} shape {tuple(tensor.shape)} != expected {expected_shape}")

        # Mirror the turbo gate/value swap for fc1 (ff.net.0.proj). The base
        # checkpoint chunks weight as (gate, up) and stores them shard-id 0/1,
        # while diffusers orders (value, gate); swap to match.
        if side == "b" and ".ff.net.0.proj." in name:
            value, gate = tensor.chunk(2, dim=0)
            tensor = torch.cat((gate, value), dim=0).contiguous()

        trunk[dict_name] = tensor

    incomplete = sorted(t for t, sides in pairs.items() if sides != {"a", "b"})
    if incomplete:
        raise ValueError(f"Incomplete PDD LoRA pairs: {incomplete}")
    missing = sorted(_PDD_TRUNK_RAW_TARGETS - raw_targets)
    unexpected = sorted(raw_targets - _PDD_TRUNK_RAW_TARGETS)
    if missing:
        raise ValueError(f"PDD trunk is missing {len(missing)} expected targets, e.g. {missing[:5]}")
    if unexpected:
        raise ValueError(f"PDD trunk has {len(unexpected)} unexpected targets, e.g. {unexpected[:5]}")
    for key in ("video_out", "audio_out"):
        if key not in head_weights:
            raise ValueError(f"PDD artifact missing head bank for {key}")
        if key not in head_biases:
            raise ValueError(f"PDD artifact missing head bias for {key}")
    return trunk, head_weights, head_biases


def _pack_pdd_fc1(lora_model: LoRAModel) -> None:
    """Pack fused gate/up LoRA for mlp.fc1, same as turbo."""
    for module_name, weights in tuple(lora_model.loras.items()):
        if not module_name.endswith(".mlp.fc1"):
            continue
        gate_b, up_b = weights.lora_b.chunk(2, dim=0)
        lora_model.loras[module_name] = PackedLoRALayerWeights(
            module_name=module_name,
            rank=weights.rank,
            lora_alphas=[weights.lora_alpha, weights.lora_alpha],
            lora_a=[weights.lora_a, weights.lora_a],
            lora_b=[gate_b.contiguous(), up_b.contiguous()],
            scaling=[weights.scaling, weights.scaling],
        )


# ---------------------------------------------------------------------------
# Handle the pipeline keeps to arm steps / install heads
# ---------------------------------------------------------------------------


@dataclass
class PDDAdapter:
    """Resident handle the pipeline uses while a PDD artifact is active."""

    config: PDDConfig
    lora_id: int
    video_plans: torch.Tensor  # (nfe, num_steps) fp32 on cpu
    audio_plans: torch.Tensor
    _heads_installed: bool = False

    def arm_step(self, transformer: nn.Module, step_index: int) -> None:
        """Set the fused-head plan on a transformer's PDD heads for denoise step ``k``.

        Callers (request-mode loop, step-mode per-request forward) must invoke
        this *immediately before* the forward that consumes step ``k``.
        """
        if not (0 <= step_index < self.config.nfe):
            raise ValueError(f"PDD step index {step_index} out of range [0, {self.config.nfe})")
        fl = getattr(transformer, "final_layer", None)
        if fl is None:
            return
        v_head = getattr(fl, "video_out", None)
        a_head = getattr(fl, "audio_out", None)
        if not isinstance(v_head, PDDParallelHead) or not isinstance(a_head, PDDParallelHead):
            return
        device = v_head.weight.device
        v_head.set_plan(self.video_plans[step_index : step_index + 1].to(device=device))
        a_head.set_plan(self.audio_plans[step_index : step_index + 1].to(device=device))

    def disarm(self, transformer: nn.Module) -> None:
        """Restore a transformer's saved base heads.

        Call on deactivation: heads stay installed as ``PDDParallelHead``
        (never swapped back to plain ``ColumnParallelLinear``), so a later
        request that reuses this transformer without this adapter would
        otherwise keep running through this adapter's last-armed step plan.
        """
        fl = getattr(transformer, "final_layer", None)
        if fl is None:
            return
        v_head = getattr(fl, "video_out", None)
        a_head = getattr(fl, "audio_out", None)
        if isinstance(v_head, PDDParallelHead):
            v_head.reset_plan()
        if isinstance(a_head, PDDParallelHead):
            a_head.reset_plan()

    def install_heads(self, transformer: nn.Module) -> None:
        """Replace final_layer.video_out / audio_out with PDDParallelHead modules.

        Idempotent per-transformer (skips if already replaced). Initializes
        every copy of the bank from the existing (base-checkpoint-initialized)
        weights; the caller follows up by loading the artifact's head tensors
        via :meth:`load_head_bank`.

        Install on **only** the DiT named by ``self.config.dit_component``: in a
        combined deployment the other DiT serves the other task family with its
        own distillation, and a Ref2VA head bank on the FL2VA DiT would corrupt
        every fl2va/t2va request. Guard each call with the ``isinstance`` check
        below rather than a global ``_heads_installed`` flag.
        """
        fl = transformer.final_layer
        if isinstance(fl.video_out, PDDParallelHead) and isinstance(fl.audio_out, PDDParallelHead):
            return
        fl.video_out = PDDParallelHead(fl.video_out, self.config.num_steps)
        fl.audio_out = PDDParallelHead(fl.audio_out, self.config.num_steps)
        # Also flag the transformer so the pipeline can detect PDD state.
        transformer._pdd_adapter = self  # type: ignore[attr-defined]

    def load_head_bank(
        self,
        transformer: nn.Module,
        head_weights: dict[str, torch.Tensor],
        head_biases: dict[str, torch.Tensor],
    ) -> None:
        """Copy the artifact's (num_steps, out, in) fp32 bank into the local TP shard."""
        fl = transformer.final_layer
        for key, attr in (("video_out", "video_out"), ("audio_out", "audio_out")):
            head: PDDParallelHead = getattr(fl, attr)
            bank_w = head_weights[key]
            bank_b = head_biases[key]
            # Slice with the head's own recorded shard geometry rather than
            # re-querying the global TP group: the head already resolved it
            # from the ColumnParallelLinear it replaced, and it is also
            # meaningful in single-process tests where no group exists.
            start = head.tp_rank * head.out_features_local
            end = start + head.out_features_local
            # Use .data.copy_ to stay off autograd.
            head.weight.data.copy_(bank_w[:, start:end, :].contiguous())
            if head.bias is not None:
                head.bias.data.copy_(bank_b[:, start:end].contiguous())

    @staticmethod
    def uninstall_heads(transformer: nn.Module) -> None:
        """Restore originals (for adapter eviction). Out of scope for v1 --
        PDD is loaded once per worker -- but the hook exists if we need it."""
        # We intentionally leave this as a no-op: production serves with a
        # small fixed LoRA set, and replacing ColumnParallelLinear back in
        # requires saving the original references. If eviction is needed
        # later, stash originals in install_heads and restore here.
        raise NotImplementedError("PDD head uninstallation is not implemented")


# ---------------------------------------------------------------------------
# Public loader (mirrors load_minimax_h3_turbo_lora)
# ---------------------------------------------------------------------------


def load_minimax_h3_pdd_lora(
    *,
    partition: str,
    lora_request: LoRARequest,
    lora_path: str | Path,
    dtype: torch.dtype,
    unsupported_offload_mode: str | None = None,
) -> tuple[LoRAModel, PEFTHelper, PDDConfig, dict[str, torch.Tensor], dict[str, torch.Tensor]] | None:
    """Load a released PDD 8-step artifact (Ref2VA or FL2VA).

    Returns ``(lora_model, peft_helper, pdd_config, head_weights, head_biases)``
    on success, or ``None`` if ``lora_path`` does not point at a PDD artifact.
    ``pdd_config.variant`` / ``.tasks`` / ``.dit_component`` tell the caller
    which task the artifact may serve and which DiT it binds to; the trunk
    ``LoRAModel`` keys are already qualified with that component so the manager
    binds them exactly there.  The caller (pipeline) is responsible for calling
    ``PDDAdapter.install_heads`` / ``load_head_bank`` on that DiT module after
    the LoRAModel is bound, since the loader runs before the manager replaces
    trunk layers and does not have a reference to the transformer.

    Raises ``ValueError`` on any shape / metadata / target-set mismatch, on an
    artifact whose variant cannot be determined, and on a partition that does
    not hold the DiT the artifact was distilled for -- these are hard errors
    because a mis-bound or silently truncated PDD artifact produces corrupted
    videos, not load failures.
    """
    lora_file = _select_pdd_file(lora_path)
    if lora_file is None:
        return None
    with safe_open(lora_file, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
        key_format = metadata.get("key_format")
        tagged_as_pdd = key_format in _PDD_VARIANTS_BY_KEY_FORMAT or lora_file.name in _PDD_VARIANTS_BY_FILENAME
        if not _has_pdd_head_bank(checkpoint):
            if tagged_as_pdd:
                raise ValueError(
                    f"{lora_file.name!r} is a MiniMax-H3 PDD release by name/key_format but has "
                    "no per-step head bank (proj_out.weight / audio_proj_out.weight, 3-D). "
                    "Refusing to fall back to the generic LoRA path: the caller still pins the "
                    "distilled 9-step schedule, and 9 steps without the heads produces garbage."
                )
            # Not a PDD artifact -- let the turbo / generic PEFT loader try.
            return None

        variant = _identify_pdd_variant(lora_file, metadata)
        if partition not in variant.partitions:
            raise ValueError(
                f"MiniMax-H3 PDD {variant.name} 8-step artifact needs a partition holding the "
                f"{variant.name} DiT ({sorted(variant.partitions)}), got {partition!r}"
            )
        if unsupported_offload_mode is not None:
            raise ValueError(
                f"MiniMax-H3 PDD does not support {unsupported_offload_mode}; "
                "PDD heads are resident fp32 parameters and do not survive offload."
            )

        cfg = _parse_pdd_metadata(metadata, variant=variant, partition=partition)
        trunk_tensors, head_weights, head_biases = _validate_and_convert_tensors(checkpoint, cfg)

    weights_mapper = _pdd_weights_mapper(cfg.dit_component)
    peft_helper = PEFTHelper.from_dict(
        {
            "r": cfg.rank,
            "lora_alpha": cfg.alpha,
            "target_modules": cfg.trunk_target_pattern,
        }
    )
    lora_model = LoRAModel.from_lora_tensors(
        lora_model_id=lora_request.lora_int_id,
        tensors=trunk_tensors,
        peft_helper=peft_helper,
        device="cpu",
        dtype=dtype,
        weights_mapper=weights_mapper,
    )
    _pack_pdd_fc1(lora_model)
    return lora_model, peft_helper, cfg, head_weights, head_biases


__all__ = [
    "PDDAdapter",
    "PDDConfig",
    "PDDParallelHead",
    "PDDVariant",
    "PDD_NFE",
    "PDD_SIGMA_POINTS",
    "PDD_VIDEO_SHIFT",
    "PDD_AUDIO_SHIFT",
    "load_minimax_h3_pdd_lora",
]

# Public schedule constants for the pipeline's validation paths.
PDD_NFE = _PDD_NFE
PDD_SIGMA_POINTS = _PDD_SIGMA_POINTS
PDD_VIDEO_SHIFT = _PDD_VIDEO_SHIFT
PDD_AUDIO_SHIFT = _PDD_AUDIO_SHIFT
