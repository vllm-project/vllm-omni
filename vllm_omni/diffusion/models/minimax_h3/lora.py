# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Loader for the LightX2V Turbo MiniMax-H3 LoRA family.

The published artifacts differ in task family, step count, training resolution
and LoRA alpha, so :class:`TurboSpec` carries the contract of the file actually
loaded and the pipeline validates each request against it. Only the
Diffusers-PEFT exports are served. The native FlashGen contract lives in
``.npu.lora``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import regex as re
import torch
from safetensors import safe_open
from vllm.logger import init_logger
from vllm.lora.lora_model import LoRAModel
from vllm.lora.lora_weights import PackedLoRALayerWeights
from vllm.lora.peft_helper import PEFTHelper
from vllm.model_executor.models.utils import WeightsMapper

from vllm_omni.lora.request import LoRARequest

logger = init_logger(__name__)

_TURBO_RANK = 128
_TURBO_HIDDEN_SIZE = 5376
_TURBO_ATTENTION_INNER_SIZE = 7168
_TURBO_FFN_HIDDEN_SIZE = 14336
# ``minimax_h3_<task>_turbo_<n>step_v<major.minor>[_768p][_bf16]``. The name is
# the only place the sampler contract is recorded.
_TURBO_NAME_RE = re.compile(
    r"^minimax_h3_(?P<task>fl2v|ref2v)_turbo_(?P<steps>\d+)step"
    r"_v\d+\.\d+(?P<res>_768p)?(?:_bf16)?\.safetensors$"
)
# The 768p retrains moved to a shorter video flow shift; the 544p artifacts keep
# the base model's.
_TURBO_VIDEO_SHIFT_768P = 6.0
_TURBO_VIDEO_SHIFT_544P = 12.0
_TURBO_AUDIO_SHIFT = 3.0
# Only ``4step_v0.1`` declares no alpha. LightX2V's reference script never reads
# the metadata: it applies ``scale * alpha / rank`` with ``--lora-alpha``
# defaulting to 8, and its documented v0.1 command does not override that. So an
# artifact that declares nothing is driven at 8, not at its rank.
_TURBO_DEFAULT_ALPHA = 8.0


@dataclass(frozen=True)
class TurboSpec:
    """The sampler contract one Turbo artifact was distilled for."""

    filename: str
    task_family: str
    """``fl2v`` (serves t2va and fl2va) or ``ref2v`` (serves ref2va)."""
    denoise_steps: int
    video_shift: float
    audio_shift: float
    rank: int
    alpha: float

    @property
    def sigma_points(self) -> int:
        """Sigma points the API contract expects: one more than the forwards."""
        return self.denoise_steps + 1

    @property
    def supported_tasks(self) -> frozenset[str]:
        if self.task_family == "ref2v":
            return frozenset({"ref2va"})
        return frozenset({"t2va", "fl2va"})


def parse_turbo_filename(name: str) -> TurboSpec | None:
    """Return the contract a Turbo filename encodes, or ``None``.

    ``alpha`` is the rank-matched default; the loader replaces it with the
    artifact's declared value.
    """

    match = _TURBO_NAME_RE.match(name)
    if match is None:
        return None
    steps = int(match.group("steps"))
    if steps <= 0:
        return None
    return TurboSpec(
        filename=name,
        task_family=match.group("task"),
        denoise_steps=steps,
        video_shift=_TURBO_VIDEO_SHIFT_768P if match.group("res") else _TURBO_VIDEO_SHIFT_544P,
        audio_shift=_TURBO_AUDIO_SHIFT,
        rank=_TURBO_RANK,
        alpha=_TURBO_DEFAULT_ALPHA,
    )


_LORA_A_SUFFIX = ".lora_A.default.weight"
_LORA_B_SUFFIX = ".lora_B.default.weight"
_TURBO_TARGETS = frozenset({"to_q", "to_k", "to_v", "out_proj", "fc1", "fc2"})
_TURBO_RAW_TARGET_SUFFIXES = (
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
)
_TURBO_TARGET_DIMS = {
    "attn.to_q": (_TURBO_HIDDEN_SIZE, _TURBO_ATTENTION_INNER_SIZE),
    "attn.to_k": (_TURBO_HIDDEN_SIZE, _TURBO_ATTENTION_INNER_SIZE),
    "attn.to_v": (_TURBO_HIDDEN_SIZE, _TURBO_ATTENTION_INNER_SIZE),
    "attn.to_out.0": (_TURBO_ATTENTION_INNER_SIZE, _TURBO_HIDDEN_SIZE),
    "ff.net.0.proj": (_TURBO_HIDDEN_SIZE, 2 * _TURBO_FFN_HIDDEN_SIZE),
    "ff.net.2": (_TURBO_FFN_HIDDEN_SIZE, _TURBO_HIDDEN_SIZE),
}
_TURBO_EXPECTED_RAW_TARGETS = frozenset(
    f"{prefix}.{block_index}.{suffix}"
    for prefix, block_count in (
        ("transformer_blocks", 50),
        ("token_refiner.refiner_blocks", 2),
    )
    for block_index in range(block_count)
    for suffix in _TURBO_RAW_TARGET_SUFFIXES
)
_TURBO_TARGET_PATTERN = (
    r"^transformer\.(?:token_refiner\.blocks|blocks)\.\d+\."
    r"(?:attn\.(?:to_q|to_k|to_v|out_proj)|mlp\.(?:fc1|fc2))$"
)

_TURBO_WEIGHTS_MAPPER = WeightsMapper(
    orig_to_new_substr={
        "token_refiner.refiner_blocks.": "token_refiner.blocks.",
        "transformer_blocks.": "blocks.",
        ".attn.to_out.0.": ".attn.out_proj.",
        ".ff.net.0.proj.": ".mlp.fc1.",
        ".ff.net.2.": ".mlp.fc2.",
        ".lora_A.default.": ".lora_A.",
        ".lora_B.default.": ".lora_B.",
    }
)


def _select_turbo_file(artifact_path: str | Path) -> Path | None:
    path = Path(artifact_path)
    if path.is_file():
        return path if path.suffix == ".safetensors" else None
    if not path.is_dir():
        return None

    candidates = sorted(child for child in path.glob("*.safetensors") if parse_turbo_filename(child.name))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(
            f"{path} holds {len(candidates)} MiniMax-H3 Turbo artifacts "
            f"({[c.name for c in candidates[:4]]}); point --lora-path at one file."
        )
    return None


def _validate_and_convert_tensors(checkpoint) -> dict[str, torch.Tensor]:
    """Validate a Diffusers-layout Turbo tensor set and pack it for the manager."""

    tensors: dict[str, torch.Tensor] = {}
    pairs: dict[str, set[str]] = {}
    raw_targets: set[str] = set()
    for name in checkpoint.keys():
        if name.endswith(_LORA_A_SUFFIX):
            raw_target = name[: -len(_LORA_A_SUFFIX)]
            side = "a"
        elif name.endswith(_LORA_B_SUFFIX):
            raw_target = name[: -len(_LORA_B_SUFFIX)]
            side = "b"
        else:
            raise ValueError(f"Unconsumed MiniMax-H3 Turbo tensor: {name!r}")
        raw_targets.add(raw_target)

        mapped_name = _TURBO_WEIGHTS_MAPPER.apply_list([name])[0]
        mapped_target = mapped_name.rsplit(".lora_", 1)[0]
        if mapped_target.rsplit(".", 1)[-1] not in _TURBO_TARGETS:
            raise ValueError(f"Unsupported MiniMax-H3 Turbo target: {raw_target!r}")
        target_sides = pairs.setdefault(mapped_target, set())
        if side in target_sides:
            raise ValueError(f"Duplicate MiniMax-H3 Turbo tensor for {mapped_target}.{side}")
        target_sides.add(side)

        tensor = checkpoint.get_tensor(name)
        if tensor.ndim != 2:
            raise ValueError(f"MiniMax-H3 Turbo LoRA tensors must be matrices, got {name}={tuple(tensor.shape)}")
        suffix = next((suffix for suffix in _TURBO_RAW_TARGET_SUFFIXES if raw_target.endswith(suffix)), None)
        if suffix is None:
            raise ValueError(f"MiniMax-H3 Turbo LoRA contains unsupported target: {raw_target}")
        input_dim, output_dim = _TURBO_TARGET_DIMS[suffix]
        expected_shape = (_TURBO_RANK, input_dim) if side == "a" else (output_dim, _TURBO_RANK)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"MiniMax-H3 Turbo tensor has invalid global shape: {name}={tuple(tensor.shape)}, "
                f"expected={expected_shape}"
            )
        if side == "b" and ".ff.net.0.proj." in name:
            value, gate = tensor.chunk(2, dim=0)
            tensor = torch.cat((gate, value), dim=0).contiguous()
        tensors[name] = tensor

    incomplete = sorted(target for target, sides in pairs.items() if sides != {"a", "b"})
    if incomplete:
        raise ValueError(f"Incomplete MiniMax-H3 Turbo LoRA pairs: {incomplete}")
    missing = sorted(_TURBO_EXPECTED_RAW_TARGETS - raw_targets)
    unexpected = sorted(raw_targets - _TURBO_EXPECTED_RAW_TARGETS)
    if missing or unexpected:
        raise ValueError(
            "MiniMax-H3 Turbo target set does not match the published artifact layout: "
            f"missing={len(missing)} {missing[:5]}, unexpected={len(unexpected)} {unexpected[:5]}"
        )
    return tensors


def _pack_h3_turbo_fc1(lora_model: LoRAModel) -> None:
    """Represent H3's fused gate/up projection without generic layout guesses."""

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


def load_minimax_h3_turbo_lora(
    *,
    partition: str,
    lora_request: LoRARequest,
    lora_path: str | Path,
    dtype: torch.dtype,
    unsupported_offload_mode: str | None = None,
) -> tuple[LoRAModel, PEFTHelper, TurboSpec] | None:
    """Load any published LightX2V Turbo artifact through the legacy manager."""

    lora_file = _select_turbo_file(lora_path)
    if lora_file is None:
        return None
    spec = parse_turbo_filename(lora_file.name)
    if spec is None:
        # Not a Turbo artifact by name; leave it to the next loader.
        return None
    with safe_open(lora_file, framework="pt", device="cpu") as checkpoint:
        # Three published artifacts declare no key_format, so absence is not an
        # error; a declaration naming a different format is.
        metadata = checkpoint.metadata() or {}
        declared_format = metadata.get("key_format")
        if declared_format not in (None, "minimax-h3-diffusers"):
            raise ValueError(
                f"{lora_file.name} declares key_format={declared_format!r}, expected 'minimax-h3-diffusers'"
            )
        raw_alpha = metadata.get("alpha")
        if raw_alpha is None:
            logger.warning(
                "MiniMax-H3 Turbo artifact %s declares no alpha; using the LightX2V reference "
                "default alpha=%g (scale %g). Override with the request-level LoRA scale.",
                spec.filename,
                spec.alpha,
                spec.alpha / spec.rank,
            )
        else:
            try:
                spec = replace(spec, alpha=float(raw_alpha))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"MiniMax-H3 Turbo alpha must be numeric, got {raw_alpha!r}") from exc
        if not math.isfinite(spec.alpha) or spec.alpha <= 0:
            raise ValueError(f"MiniMax-H3 Turbo alpha must be a positive number, got {raw_alpha!r}")

        if partition == "ref2va" and spec.task_family == "fl2v":
            raise ValueError(f"{spec.filename} is an FL2VA/T2VA Turbo artifact; a Ref2VA-only server cannot serve it.")
        # ``combined`` serves ref2va from ``transformers_ref``, but the LoRA
        # target pattern only injects into ``transformer``: the adapter would
        # bind to the stack that never runs.
        if spec.task_family == "ref2v" and partition != "ref2va":
            raise ValueError(
                f"{spec.filename} is a Ref2VA Turbo artifact; start the server with --task-type ref2va "
                f"(task_type={partition!r} serves ref2va from a DiT the adapter cannot bind to)"
            )
        if unsupported_offload_mode is not None:
            raise ValueError(f"MiniMax-H3 Turbo dynamic LoRA does not support {unsupported_offload_mode}")

        tensors = _validate_and_convert_tensors(checkpoint)

    peft_helper = PEFTHelper.from_dict(
        {
            "r": spec.rank,
            "lora_alpha": spec.alpha,
            "target_modules": _TURBO_TARGET_PATTERN,
        }
    )
    lora_model = LoRAModel.from_lora_tensors(
        lora_model_id=lora_request.lora_int_id,
        tensors=tensors,
        peft_helper=peft_helper,
        device="cpu",
        dtype=dtype,
        weights_mapper=_TURBO_WEIGHTS_MAPPER,
    )
    _pack_h3_turbo_fc1(lora_model)
    logger.info(
        "MiniMax-H3 Turbo adapter %s: task=%s, %d denoiser forwards (%d sigma points), "
        "rank=%d alpha=%g, flow_shift=%g/%g",
        spec.filename,
        spec.task_family,
        spec.denoise_steps,
        spec.sigma_points,
        spec.rank,
        spec.alpha,
        spec.video_shift,
        spec.audio_shift,
    )
    return lora_model, peft_helper, spec
