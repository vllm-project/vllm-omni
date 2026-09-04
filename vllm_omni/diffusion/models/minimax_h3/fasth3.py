# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""FastVideo FastH3: a four-step DMD2 student of MiniMax-H3.

FastH3 replaces H3's 49 denoiser evaluations with four. It ships as an adapter
over the base checkpoint rather than as a full release, so it reuses H3's text
encoder, video VAE, audio VAE, tokenizers and schedulers unchanged.

The artifact is *not* a PEFT LoRA, and it is not request-switchable. Its own
metadata states the reconstruction as::

    W = W_base + lora_B @ lora_A; then .diff/.diff_b added and .set_weight assigned

so besides rank-64 factors it carries full-rank ``.diff``/``.diff_b`` deltas for
RMSNorm weights, biases, patch projections and the final layer - none of which a
LoRA layer can express - and the VSA variants add ``.set_weight`` tensors for
compression gates that do not exist in the base transformer at all. The adapter
is therefore fused into the checkpoint stream at load time, before the weights
are sharded, which is also what the release's model card requires.

The low-rank factors carry no alpha: the reconstruction adds ``lora_B @ lora_A``
directly, i.e. a scale of exactly 1.

Two checkpoint spellings meet here. The adapter is written in the diffusers
namespace (``transformer_blocks.0.attn.to_q``) while vLLM-Omni loads H3's native
one (``blocks.0.attn.qkv_proj``), whose attention and MLP projections are fused.
Every mapping and layout convention below was verified tensor by tensor against
the released full checkpoint (``FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree``):
``W_base + delta`` reproduces it to bf16 rounding.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from safetensors import safe_open
from vllm.logger import init_logger

from vllm_omni.diffusion.sched.sigma_schedule import DMD2SigmaSchedule
from vllm_omni.errors import OmniClientError
from vllm_omni.platforms import current_omni_platform

if TYPE_CHECKING:
    from .minimax_h3_transformer import MiniMaxH3DiTModel

logger = init_logger(__name__)

# FastVideo's generic adapter container, not a FastH3 marker: their tools emit
# it for ordinary H3 adapters too.
FASTH3_FORMAT = "fastvideo-lora-v2"
FASTH3_MANIFEST = "adapter_manifest.json"
# The release identity: the distilled student the adapter came from, over the
# base it edits. The name is held to FastVideo's own namespace as well as the
# student's name, so an unrelated adapter that merely mentions FastH3 is not
# claimed.
FASTH3_BASE_MODEL = "MiniMaxAI/MiniMax-H3"
_FASTH3_IDENTITY_KEY = "finetuned_model"
_FASTH3_IDENTITY_NAMESPACE = "fastvideo/"
_FASTH3_IDENTITY_MARKER = "fasth3"

# The rectified-flow positions the student was distilled at, and the ladder the
# server samples on. The release states them as `dmd_denoising_steps`
# [999, 749, 500, 250] (`sampling.base_timesteps` in the bundle manifest):
# timestep indices out of 1000, i.e. pre-shift positions, closed here with the
# terminal 0.0 every rectified-flow schedule ends on. The opening rung is 0.999
# rather than 1.0 because training capped the noise level there
# (`max_timestep_ratio`).
#
# These are positions, not final sigmas: H3's schedulers apply their own
# per-modality shift on top (12 for video, 3 for audio), which is what
# reproduces the levels the student saw. Nothing here overrides those shifts.
FASTH3_BASE_SCHEDULE = DMD2SigmaSchedule.from_positions((0.999, 0.749, 0.5, 0.25, 0.0))
# Its five points bound four transformer forwards, one per sigma interval. That
# count is the one a request states, the one Cache-DiT is refreshed with and the
# one step execution admits a request on - the same interval contract H3's
# pinned checkpoint schedules and native LoRAs already use.
FASTH3_DENOISE_STEPS = FASTH3_BASE_SCHEDULE.num_inference_steps
# Preview v1 distills the text-to-video-and-audio path only.
FASTH3_SUPPORTED_TASKS = frozenset({"t2va"})

_LORA_A = ".lora_A.weight"
_LORA_B = ".lora_B.weight"
_DIFF = ".diff"
_DIFF_B = ".diff_b"
_SET_WEIGHT = ".set_weight"

# Adapter module prefix -> the native parameter it edits, minus the
# ``.weight``/``.bias`` suffix.
_MODEL_LEVEL_TARGETS = {
    "proj_in": "video_patch_proj",
    "proj_out": "final_layer.video_out",
    "audio_proj_in": "audio_patch_proj",
    "audio_proj_out": "final_layer.audio_out",
    "context_embedder": "condition_proj",
    "time_embedder.linear_1": "time_embedder.proj_in",
    "time_embedder.linear_2": "time_embedder.proj_out",
    "norm_out.linear": "final_layer.adaln_proj.linear",
    "norm_out.norm": "final_layer.norm",
}

# Per-block adapter suffix -> (native suffix, how a delta enters the native
# parameter). H3 stores attention as one grouped QKV matrix and the MLP as one
# fused gate/up matrix, so those deltas need placing rather than adding.
_PLAIN, _QKV_Q, _QKV_K, _QKV_V, _SWAP_HALVES = "plain", "q", "k", "v", "swap_halves"
_QKV_SLOTS = (_QKV_Q, _QKV_K, _QKV_V)
_BLOCK_TARGETS = {
    "attn.to_q": ("attn.qkv_proj", _QKV_Q),
    "attn.to_k": ("attn.qkv_proj", _QKV_K),
    "attn.to_v": ("attn.qkv_proj", _QKV_V),
    "attn.to_out.0": ("attn.out_proj", _PLAIN),
    "attn.to_gate_compress": ("attn.to_gate_compress", _PLAIN),
    "ff.net.0.proj": ("mlp.fc1", _SWAP_HALVES),
    "ff.net.2": ("mlp.fc2", _PLAIN),
    "adaln_proj.linear": ("adaln_proj.linear", _PLAIN),
    "norm1": ("norm1", _PLAIN),
    "norm2": ("norm2", _PLAIN),
}

# The attention role MiniMaxH3Attention gives its 50 DiT blocks. The
# compression gates live on exactly these layers, so this is the role whose
# resolved backend decides whether the artifact runs sparse.
_H3_DIT_ATTENTION_ROLE = "self"

# Adapter block prefix -> native block prefix.
_BLOCK_PREFIXES = (
    ("token_refiner.refiner_blocks.", "token_refiner.blocks."),
    ("transformer_blocks.", "blocks."),
)


def _resolve_dit_attention_backend(od_config: Any) -> str:
    """The backend the 50-block H3 DiT will actually resolve to.

    The DiT's attention layers carry role ``"self"``, so a ``per_role`` entry
    overrides the default for exactly the layers the compression gates live on.
    Reading only the default would accept a config that runs the sparse student
    dense, and reject a per-role-only config that is correct.
    """
    attention_config = getattr(od_config, "diffusion_attention_config", None)
    per_role = getattr(attention_config, "per_role", None) or {}
    spec = per_role.get(_H3_DIT_ATTENTION_ROLE)
    if spec is not None:
        return str(getattr(spec, "backend", "") or "").upper()
    backend = str(getattr(od_config, "diffusion_attention_backend", "") or "").upper()
    if backend:
        return backend
    default_spec = getattr(attention_config, "default", None)
    return str(getattr(default_spec, "backend", "") or "").upper()


class FastH3AdapterError(ValueError):
    """The artifact is a FastH3 adapter, but it cannot be applied as one."""


@dataclass
class _ParamPatch:
    """Everything the adapter contributes to one native parameter."""

    # layout -> (lora_A, lora_B). A grouped QKV parameter collects three.
    low_rank: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = field(default_factory=dict)
    diff: torch.Tensor | None = None
    layout: str = _PLAIN


def _swap_halves(tensor: torch.Tensor) -> torch.Tensor:
    """Exchange the two halves of a fused gate/up matrix.

    The diffusers export stores the feed-forward projection value-first while
    H3's native ``mlp.fc1`` is gate-first, so a delta computed in the diffusers
    layout has to be swapped before it can be added to the native parameter.
    """
    if tensor.shape[0] % 2:
        raise FastH3AdapterError(f"fused gate/up delta must split evenly, got {tuple(tensor.shape)}")
    first, second = tensor.chunk(2, dim=0)
    return torch.cat((second, first), dim=0)


def _place_in_grouped_qkv(deltas: Mapping[str, torch.Tensor], *, head_dim: int) -> torch.Tensor:
    """Interleave per-projection deltas into H3's grouped QKV layout.

    The checkpoint stores one head group at a time as ``[q, k, v]``, which is
    what :func:`_reorder_grouped_qkv_to_qkv` unpacks on the way in. A delta
    built from the separate diffusers projections has to be folded back into
    that order.
    """
    missing = sorted(set(_QKV_SLOTS) - set(deltas))
    if missing:
        raise FastH3AdapterError(f"grouped QKV delta is missing its {missing} projections")
    parts = []
    for slot in _QKV_SLOTS:
        delta = deltas[slot]
        if delta.shape[0] % head_dim:
            raise FastH3AdapterError(
                f"QKV {slot} delta rows {delta.shape[0]} are not a multiple of head_dim {head_dim}"
            )
        parts.append(delta.reshape(delta.shape[0] // head_dim, head_dim, *delta.shape[1:]))
    groups = parts[0].shape[0]
    if any(part.shape[0] != groups for part in parts):
        raise FastH3AdapterError("QKV projections disagree on the number of head groups")
    return torch.cat(parts, dim=1).reshape(groups * 3 * head_dim, *parts[0].shape[2:])


def _resolve_native_target(module: str) -> tuple[str, str, tuple[str, int] | None] | None:
    """Map an adapter module path to ``(native path, layout, block)``.

    ``block`` is the ``(native block prefix, index)`` the module sits in, or
    ``None`` for a model-level one. Coverage is checked per block, so the index
    has to survive the mapping rather than being folded into the path.
    """
    native = _MODEL_LEVEL_TARGETS.get(module)
    if native is not None:
        return native, _PLAIN, None
    for adapter_prefix, native_prefix in _BLOCK_PREFIXES:
        if not module.startswith(adapter_prefix):
            continue
        remainder = module[len(adapter_prefix) :]
        index, _, suffix = remainder.partition(".")
        if not index.isdigit():
            return None
        target = _BLOCK_TARGETS.get(suffix)
        if target is None:
            return None
        native_suffix, layout = target
        return f"{native_prefix}{index}.{native_suffix}", layout, (native_prefix, int(index))
    return None


def _split_adapter_key(name: str) -> tuple[str, str] | None:
    """Split an adapter tensor name into ``(module path, role)``."""
    for marker, role in (
        (_LORA_A, "lora_a"),
        (_LORA_B, "lora_b"),
        (_DIFF_B, "diff_b"),
        (_DIFF, "diff"),
        (_SET_WEIGHT, "set_weight"),
    ):
        if name.endswith(marker):
            return name[: -len(marker)], role
    return None


def _is_fasth3_release(metadata: Mapping[str, str]) -> bool:
    """Whether this is a FastH3 release rather than merely FastVideo's format.

    Claiming on the container alone would fuse an ordinary H3 adapter into the
    checkpoint and put every request on a four-step schedule it was never
    distilled for, and so would a bare ``fasth3`` substring: an adapter someone
    else named after the student is not the student.
    """
    if metadata.get("format") != FASTH3_FORMAT:
        return False
    identity = metadata.get(_FASTH3_IDENTITY_KEY, "").lower()
    if not identity.startswith(_FASTH3_IDENTITY_NAMESPACE) or _FASTH3_IDENTITY_MARKER not in identity:
        return False
    # Only held against a file that states it.
    base_model = metadata.get("base_model")
    return base_model is None or base_model.casefold() == FASTH3_BASE_MODEL.casefold()


def _check_declared_counts(
    metadata: Mapping[str, str],
    counted: Mapping[str, int],
    *,
    weights_path: Path,
) -> None:
    """Hold the artifact to the tensor counts its own metadata declares.

    The writer of the ``fastvideo-lora-v2`` format records how many tensors of
    each kind it emitted, which is the one statement in the file about its own
    completeness. A truncated or partially re-exported artifact is otherwise
    indistinguishable from a small one, so a claimed file has to carry all of
    them rather than opt out by omission.
    """
    for key, seen in counted.items():
        declared = metadata.get(key)
        if declared is None:
            raise FastH3AdapterError(
                f"{weights_path} declares the FastH3 identity but omits {key}; "
                "a claimed adapter has to state what it emitted"
            )
        try:
            expected = int(declared)
        except (TypeError, ValueError) as exc:
            raise FastH3AdapterError(f"{weights_path} declares a non-numeric {key}={declared!r}") from exc
        if expected != seen:
            raise FastH3AdapterError(
                f"{weights_path} declares {key}={expected} but carries {seen}; the adapter is incomplete "
                "and would leave most of the transformer on base H3 weights"
            )


def _check_block_coverage(
    seen: Mapping[str, set[int]],
    *,
    expected: Mapping[str, int],
    weights_path: Path,
) -> None:
    """Every block of the model this adapter is loaded against must be edited.

    The release drops tensors training left unchanged, so per-parameter
    coverage is legitimately sparse - but a distilled student touches every
    block, so a block with no edits at all means the artifact does not match
    this model.
    """
    for prefix, count in expected.items():
        indices = seen.get(prefix, set())
        wanted = set(range(count))
        if indices == wanted:
            continue
        missing = sorted(wanted - indices)
        extra = sorted(indices - wanted)
        raise FastH3AdapterError(
            f"{weights_path} edits {len(indices)} of the model's {count} {prefix}* blocks "
            f"(missing={missing[:5]}, unknown={extra[:5]}); it is not an adapter for this checkpoint"
        )


class FastH3WeightFusion:
    """Fuse a FastH3 adapter into the H3 checkpoint stream as it is loaded."""

    def __init__(
        self,
        *,
        source: Path,
        patches: Mapping[str, _ParamPatch],
        head_dim: int,
        requires_vsa: bool,
        injections: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        self._source = source
        self._patches = dict(patches)
        self._head_dim = head_dim
        self.requires_vsa = requires_vsa
        # Parameters the base checkpoint does not carry, assigned into the
        # stream instead of fused onto an existing weight.
        self._injections = dict(injections or {})
        self._injected: set[str] = set()
        self._applied: set[str] = set()
        self._device: torch.device | None = None

    @property
    def source(self) -> Path:
        return self._source

    @property
    def base_schedule(self) -> tuple[float, ...]:
        """The rectified-flow positions this student samples on.

        The fused checkpoint is a four-step student, so the ladder comes from
        the release rather than from the many-step teacher's metadata or from
        the uniform one ``num_inference_steps`` would otherwise derive.
        """
        return FASTH3_BASE_SCHEDULE.base_schedule

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        head_dim: int,
        num_blocks: int,
        num_refiner_blocks: int,
    ) -> FastH3WeightFusion | None:
        """Build a fusion from an adapter file or directory, else ``None``.

        Returning ``None`` keeps every other ``--lora-path`` artifact on the
        dynamic LoRA route; only a file carrying the FastH3 release identity is
        claimed here.

        The block counts are the model's, and the artifact has to cover them:
        claiming a partial adapter would switch the server onto the four-step
        contract while most of the transformer still held base H3 weights.
        """
        weights_path = _resolve_adapter_file(path)
        if weights_path is None:
            return None

        patches: dict[str, _ParamPatch] = {}
        gate_tensors: list[str] = []
        injections: dict[str, torch.Tensor] = {}
        unmapped: list[str] = []
        blocks_seen: dict[str, set[int]] = {}
        counted = {"low_rank_tensors": 0, "diff_tensors": 0}
        with safe_open(weights_path, framework="pt", device="cpu") as checkpoint:
            metadata = checkpoint.metadata() or {}
            if not _is_fasth3_release(metadata):
                return None
            for name in checkpoint.keys():
                split = _split_adapter_key(name)
                if split is None:
                    unmapped.append(name)
                    continue
                module, role = split
                target = _resolve_native_target(module)
                if role == "set_weight":
                    # A VSA compression gate. The base transformer has no such
                    # parameter, so this is assigned into the stream rather than
                    # fused onto an existing weight.
                    if target is None:
                        unmapped.append(name)
                        continue
                    native_module, _, block = target
                    if block is not None:
                        blocks_seen.setdefault(block[0], set()).add(block[1])
                    gate_tensors.append(name)
                    injections[f"{native_module}.weight"] = checkpoint.get_tensor(name)
                    continue
                if target is None:
                    unmapped.append(name)
                    continue
                native_module, layout, block = target
                if block is not None:
                    blocks_seen.setdefault(block[0], set()).add(block[1])
                counted["diff_tensors" if role in ("diff", "diff_b") else "low_rank_tensors"] += 1
                native_param = f"{native_module}.{'bias' if role == 'diff_b' else 'weight'}"
                patch = patches.setdefault(native_param, _ParamPatch(layout=layout))
                tensor = checkpoint.get_tensor(name)
                if role in ("diff", "diff_b"):
                    if patch.diff is not None:
                        raise FastH3AdapterError(f"duplicate {role} for {native_param}")
                    patch.diff = tensor
                else:
                    a, b = patch.low_rank.get(layout, (None, None))
                    patch.low_rank[layout] = (tensor, b) if role == "lora_a" else (a, tensor)

        if unmapped:
            raise FastH3AdapterError(
                f"FastH3 adapter at {weights_path} has {len(unmapped)} tensors that name no known "
                f"H3 parameter: {sorted(unmapped)[:5]}"
            )
        _check_declared_counts(
            metadata,
            {**counted, "set_weight_tensors": len(gate_tensors)},
            weights_path=weights_path,
        )
        _check_block_coverage(
            blocks_seen,
            expected={"blocks.": num_blocks, "token_refiner.blocks.": num_refiner_blocks},
            weights_path=weights_path,
        )
        for native_param, patch in patches.items():
            for slot, (a, b) in patch.low_rank.items():
                if a is None or b is None:
                    raise FastH3AdapterError(f"FastH3 adapter has an unpaired factor for {native_param} slot {slot!r}")
            # Only the low-rank factors are placed into H3's fused QKV and
            # gate/up layouts; a full-rank delta is added as it comes, so one
            # aimed at a fused parameter would silently land transposed.
            if patch.diff is not None and patch.layout != _PLAIN:
                raise FastH3AdapterError(
                    f"FastH3 adapter carries a full-rank delta for {native_param}, which H3 stores in the "
                    f"{patch.layout!r} fused layout; this loader can only place low-rank factors there"
                )

        fusion = cls(
            source=weights_path,
            patches=patches,
            head_dim=head_dim,
            requires_vsa=bool(gate_tensors),
            injections=injections,
        )
        logger.info(
            "FastH3 adapter %s: rank=%s, parameters patched=%d, low-rank=%s, diff=%s, set_weight=%d",
            weights_path,
            metadata.get("rank", "?"),
            len(patches),
            metadata.get("low_rank_tensors", "?"),
            metadata.get("diff_tensors", "?"),
            len(gate_tensors),
        )
        return fusion

    def _compute_device(self, weight: torch.Tensor) -> torch.device:
        """Where to reconstruct a delta.

        H3's per-block modulation projection is 96768x2688, so rebuilding all
        343 patched parameters is a few TFLOP of rank-64 products. On CPU that
        adds minutes to a load that already has a startup deadline, so the
        accelerator does the arithmetic whenever there is one.
        """
        if weight.device.type != "cpu":
            return weight.device
        if self._device is None:
            # Ask the platform rather than PyTorch's global accelerator
            # registry, so an out-of-tree backend controls its own placement.
            self._device = current_omni_platform.get_torch_device()
        return self._device

    @staticmethod
    def _widen(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Move to ``device``, then widen to float32.

        Asking ``Tensor.to`` for a device and a dtype at once converts on the
        host and ships twice the bytes; splitting it moves bfloat16 and widens
        on the accelerator.
        """
        return tensor.to(device, non_blocking=True).to(torch.float32)

    def fuse(self, name: str, weight: torch.Tensor) -> torch.Tensor:
        """Return ``weight`` with this adapter's contribution added."""
        patch = self._patches.get(name)
        if patch is None:
            return weight
        self._applied.add(name)

        device = self._compute_device(weight)

        delta: torch.Tensor | None = None
        if patch.low_rank:
            if patch.layout in _QKV_SLOTS:
                per_slot = {
                    slot: self._widen(b, device) @ self._widen(a, device) for slot, (a, b) in patch.low_rank.items()
                }
                delta = _place_in_grouped_qkv(per_slot, head_dim=self._head_dim)
            else:
                a, b = patch.low_rank[patch.layout]
                if patch.layout == _SWAP_HALVES:
                    # Permuting the rows of B permutes the rows of the product,
                    # so swap the rank-64 factor instead of the full delta.
                    b = _swap_halves(b)
                delta = self._widen(b, device) @ self._widen(a, device)
        if patch.diff is not None:
            diff = self._widen(patch.diff, device)
            delta = diff if delta is None else delta + diff
        if delta is None:
            return weight
        if delta.shape != weight.shape:
            raise FastH3AdapterError(
                f"FastH3 delta for {name} has shape {tuple(delta.shape)}, parameter is {tuple(weight.shape)}"
            )
        # Leave the result on the compute device. These weights are bound for
        # the accelerator anyway, so returning them to host memory would pay a
        # device-to-host copy of the whole checkpoint only for the loader to
        # send it straight back: measured at 152s against 15s for 60 GiB of
        # patched projections, against 17s for the unavoidable upload alone.
        # Fold the base weight into the freshly built delta in place. Promoting
        # the weight to float32 on its own would allocate two more buffers the
        # size of the parameter, and H3's largest patched projection is 0.5 GiB.
        return delta.add_(weight.to(device, non_blocking=True)).to(weight.dtype)

    def apply(self, weights: Iterable[tuple[str, torch.Tensor]]) -> Iterator[tuple[str, torch.Tensor]]:
        """Fuse every streamed checkpoint tensor on its way into the model."""
        if self._applied:
            # ``validate_fully_applied`` released the deltas, so a second stream
            # would fuse nothing and then pass its own completeness check: the
            # server would serve base H3 weights on the student's ladder.
            raise FastH3AdapterError(f"{self._source} has already been fused into this checkpoint")
        for name, weight in weights:
            if name in self._injections:
                raise FastH3AdapterError(
                    f"the checkpoint already provides {name}, which this adapter assigns; "
                    "assigning it would discard the checkpoint's own weight"
                )
            yield name, self.fuse(name, weight)
        # The VSA gates have no counterpart in the base checkpoint, so they join
        # the stream after it rather than being folded into one of its tensors.
        for name, weight in self._injections.items():
            self._injected.add(name)
            yield name, weight

    def validate_fully_applied(self, loaded: Iterable[str] | None = None) -> None:
        """Close the fusion: every edit must have met its parameter.

        A silently unapplied delta is the failure mode that matters here: the
        model would load and generate, just not as the distilled student. The
        weights are loaded once, so the mapped payloads are dropped afterwards
        rather than held for the life of the process.

        ``loaded`` is the set of parameter names ``load_weights`` actually
        consumed. A gate is assigned rather than fused, so it lands on a module
        the base transformer does not have; if that module was never built,
        ``load_weights`` only logs a skip and the server would serve a
        zero-initialized gate. Yielding a tensor is not evidence it arrived, so
        the injections are closed against that set when it is available.
        """
        missing = sorted(set(self._patches) - self._applied)
        if missing:
            raise FastH3AdapterError(
                f"FastH3 adapter edits {len(missing)} parameters the checkpoint never provided: {missing[:5]}"
            )
        arrived = self._injected if loaded is None else set(loaded)
        uninjected = sorted(set(self._injections) - arrived)
        if uninjected:
            raise FastH3AdapterError(
                f"FastH3 adapter assigns {len(uninjected)} parameters that never reached the model: {uninjected[:5]}"
            )
        for patch in self._patches.values():
            patch.low_rank.clear()
            patch.diff = None
        self._injections.clear()

    def check_serving_contract(
        self,
        *,
        partition: str,
        od_config: Any,
        video_shift: float,
        audio_shift: float,
    ) -> None:
        """Hold a starting server to the ladder this student was trained on."""
        if partition == "ref2va":
            raise ValueError("FastH3 preview v1 distills T2VA only, so it cannot serve a Ref2VA partition")
        offloads = [
            flag
            for flag in ("enable_cpu_offload", "enable_layerwise_offload", "enable_distributed_layerwise_offload")
            if getattr(od_config, flag, False)
        ]
        if offloads:
            # A host-weight plan installs the transformer without going through
            # load_weights(), which is where the fusion and its completeness
            # check live. Serving base H3 weights under a four-step schedule
            # would otherwise degrade output with nothing to signal it.
            raise ValueError(
                f"FastH3 is fused while the checkpoint streams in, so it cannot be combined with "
                f"{sorted(offloads)}. Serve it without offload."
            )
        if self.requires_vsa:
            backend = _resolve_dit_attention_backend(od_config)
            if backend != "FASTVIDEO_VSA":
                raise ValueError(
                    f"{self.source} is a Video Sparse Attention variant of FastH3. Its compression "
                    "gates only mean anything to the VSA kernel, and any other backend would run it "
                    f"as dense attention on a student distilled for 90% sparsity (got {backend or 'default'}). "
                    "Serve it with --diffusion-attention-backend FASTVIDEO_VSA."
                )
            parallel_config = getattr(od_config, "parallel_config", None)
            ring_degree = int(getattr(parallel_config, "ring_degree", 1) or 1)
            allgather_degree = int(getattr(parallel_config, "allgather_degree", 1) or 1)
            if ring_degree != 1 or allgather_degree != 1:
                raise ValueError(
                    "FastH3 VSA supports local attention or pure Ulysses sequence parallelism; "
                    "ring/all-gather SP does not give the block-sparse kernel the complete packed sequence."
                )
        logger.info(
            "FastH3 adapter active: sigma points %s for %d transformer forwards, "
            "flow_shift=%g, audio_flow_shift=%g, tasks=%s",
            list(self.base_schedule),
            FASTH3_DENOISE_STEPS,
            video_shift,
            audio_shift,
            sorted(FASTH3_SUPPORTED_TASKS),
        )

    def check_task(self, task: str) -> None:
        """Refuse a task this preview never distilled."""
        if task not in FASTH3_SUPPORTED_TASKS:
            raise OmniClientError(
                f"FastH3 preview v1 distills {sorted(FASTH3_SUPPORTED_TASKS)} only, got task={task!r}"
            )

    def check_request(self, sampling: Any, *, video_shift: float, audio_shift: float) -> None:
        """Refuse a request that would sample the student off its rungs."""
        if sampling.lora_request is not None:
            # The adapter is already in the weights and the dynamic LoRA manager
            # is skipped, so nothing would apply the requested one. Serving the
            # request anyway would quietly ignore it.
            raise OmniClientError(
                f"this server fused {self.source} into the checkpoint at startup, so per-request "
                "lora is unavailable; drop the lora field"
            )
        if int(sampling.num_inference_steps or 0) != FASTH3_DENOISE_STEPS:
            raise OmniClientError(
                f"FastH3 is a four-step student and requires num_inference_steps={FASTH3_DENOISE_STEPS} "
                "(one transformer forward per sigma interval)"
            )
        # The checkpoint's per-modality shifts turn the release's positions into
        # the noise levels the student was distilled at, so a request that moves
        # them samples where it was never trained.
        extra = sampling.extra_args or {}
        for key, expected in (("flow_shift", video_shift), ("audio_flow_shift", audio_shift)):
            try:
                requested = float(extra.get(key, expected))
            except (TypeError, ValueError) as exc:
                raise OmniClientError(f"FastH3 requires {key}={expected:g}") from exc
            if not math.isclose(requested, expected):
                raise OmniClientError(f"FastH3 requires {key}={expected:g}, got {requested:g}")


def resolve_fasth3_fusion(od_config: Any, transformer: MiniMaxH3DiTModel) -> FastH3WeightFusion | None:
    """Claim ``--lora-path`` when it points at a FastH3 adapter.

    FastH3 rewrites RMSNorm weights and biases, so it cannot be expressed as a
    request-switchable LoRA and is fused into the checkpoint instead. Any other
    artifact returns None here and stays on the dynamic LoRA route.
    """
    lora_path = getattr(od_config, "lora_path", None)
    if isinstance(lora_path, (list, tuple)):
        if len(lora_path) != 1:
            return None
        lora_path = lora_path[0]
    if not lora_path:
        return None
    # Placing a delta into the fused QKV parameter needs the head size, and
    # completeness is judged against the model's depth, so the architecture is
    # only read once an adapter is actually configured.
    arch = transformer.arch
    return FastH3WeightFusion.from_path(
        lora_path,
        head_dim=arch.attention_head_dim,
        num_blocks=arch.num_layers,
        num_refiner_blocks=arch.token_refiner_num_layers,
    )


def _safetensors_metadata(path: Path) -> Mapping[str, str]:
    """The header metadata of a safetensors file, or ``{}`` if unreadable."""
    try:
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            return checkpoint.metadata() or {}
    except Exception:  # noqa: BLE001 - a path that will not open is simply not ours
        return {}


def _resolve_adapter_file(path: str | Path) -> Path | None:
    """Find the single adapter file at ``path``, or ``None``."""
    candidate = Path(path)
    if candidate.is_file():
        return candidate if candidate.suffix == ".safetensors" else None
    if not candidate.is_dir():
        return None
    named = candidate / "adapter_model.safetensors"
    if named.is_file():
        return named
    files = sorted(candidate.glob("*.safetensors"))
    if len(files) == 1:
        return files[0]
    # The published repository bundles four variants under one root, marked by an
    # adapter_manifest.json. Such a bundle is ambiguous rather than loadable - but
    # only hold that against a directory that actually carries FastH3 artifacts;
    # an unrelated multi-shard LoRA stays on the dynamic route via ``None``.
    if (candidate / FASTH3_MANIFEST).is_file() or any(
        _is_fasth3_release(_safetensors_metadata(file)) for file in files
    ):
        raise FastH3AdapterError(
            f"{candidate} holds several FastH3 adapters; point --lora-path at one variant "
            "(for example dense-datafree/adapter_model.safetensors)"
        )
    return None


__all__ = [
    "FASTH3_BASE_MODEL",
    "FASTH3_BASE_SCHEDULE",
    "FASTH3_DENOISE_STEPS",
    "FASTH3_FORMAT",
    "FASTH3_SUPPORTED_TASKS",
    "FastH3AdapterError",
    "FastH3WeightFusion",
    "resolve_fasth3_fusion",
]
