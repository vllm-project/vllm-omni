# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ModelOpt-native (state-dict export) FP8-blockwise checkpoint adapter.

Loads checkpoints produced by quantizing with NVIDIA ModelOpt and saving the
model ``state_dict`` directly to safetensors (as opposed to the HF-style
export the sibling :mod:`.modelopt` adapter handles). On-disk format:

- FP8 (e4m3) weight codes at their original (diffusers) parameter names,
  e.g. ``layers.0.mlp.gate_proj.weight``;
- per-module quantizer tensors ``<module>.weight_quantizer._scale`` (2D block
  grid ``[ceil(rows/block), ceil(cols/block)]``, multiply-to-dequantize) and
  ``<module>.weight_quantizer._amax`` (4D twin; ``_scale == _amax/448``);
- a ``quantization_config.json`` sidecar at the model root declaring recipe
  ``fp8_blockwise_mixed``, the block size, the quantized/kept module patterns
  and the expected tensor counts. The ``modelopt_state.pt`` pickle sidecar is
  intentionally never opened.

The adapter engages purely checkpoint-driven (sidecar present), dequantizes
every FP8 weight to the model dtype with an fp32 intermediate, consumes the
quantizer tensors, and yields a plain full-precision stream. Integrity is
verified fail-fast: a present-but-invalid
sidecar (e.g. a stale per-tensor export) aborts at detection; stream-level
count/exclusion/shape violations abort before serving, aggregated into a
single :class:`CheckpointIntegrityError`.

Run ``python -m vllm_omni.diffusion.model_loader.checkpoint_adapters.\
modelopt_native <model_dir>`` for a GPU-free integrity probe over the
safetensors headers (exit 0 = pass, 1 = fail).

Pure calculations (no I/O): :func:`parse_quant_spec`, :func:`classify_name`,
:func:`scale_name_for`, :func:`weight_name_for`, :func:`matches_any`,
:func:`expand_block_scale`, :func:`dequantize_weight`,
:func:`verify_observations`, :func:`assert_not_fp8`.
Actions: :meth:`ModelOptNativeFp8CheckpointAdapter.detect` (sidecar read),
:meth:`~ModelOptNativeFp8CheckpointAdapter.adapt` (stream orchestration),
:func:`main` (CLI probe).
"""

import enum
import glob
import json
import math
import os
import struct
import sys
from collections.abc import Generator, Iterable, Mapping
from dataclasses import dataclass
from os import PathLike
from typing import NamedTuple

import torch
from vllm.logger import init_logger

from .modelopt import FP8_DTYPES

logger = init_logger(__name__)

# Deploy-side marker for the mount-shadowing probe: its presence (and value)
# distinguishes the patched tree from a stock installation at runtime.
FP8_BLOCKWISE_MARKER = "modelopt-fp8-blockwise"

SIDECAR_FILENAME = "quantization_config.json"
EXPECTED_RECIPE = "fp8_blockwise_mixed"
EXPECTED_GRANULARITY = "blockwise-128x128"
SCALE_SUFFIX = ".weight_quantizer._scale"
AMAX_SUFFIX = ".weight_quantizer._amax"
WEIGHT_SUFFIX = ".weight"
FP8_HEADER_DTYPES = frozenset({"F8_E4M3", "F8_E5M2", "F8_E4M3FNUZ", "F8_E5M2FNUZ"})

# Bound on FP8 weights buffered while waiting for their scale tensor. The
# real checkpoint interleaves weights and scales per module, so more than a
# handful pending means the scales are systematically absent — fail early
# with a partial report instead of buffering gigabytes.
MAX_PENDING_FP8 = 8


class CheckpointIntegrityError(ValueError):
    """Quantized-checkpoint integrity violation. Fail fast."""


class TensorKind(enum.Enum):
    QUANTIZER_SCALE = "quantizer_scale"
    QUANTIZER_AMAX = "quantizer_amax"
    OTHER = "other"


class TensorInfo(NamedTuple):
    """Normalized view of one checkpoint tensor (shared by stream + header)."""

    name: str
    is_fp8: bool
    shape: tuple[int, ...]


@dataclass(frozen=True)
class BlockwiseQuantSpec:
    """Parsed sidecar declaration (Data)."""

    recipe: str
    n_quantized: int
    n_scale: int
    block_rows: int
    block_cols: int
    quantized_patterns: tuple[str, ...]
    kept_patterns: tuple[str, ...]


# --- pure calculations -------------------------------------------------------


def parse_quant_spec(config: Mapping) -> BlockwiseQuantSpec:
    """Validate and parse a ``quantization_config.json`` dict.

    Raises :class:`CheckpointIntegrityError` naming every expected-vs-found
    discrepancy (e.g. for the stale per-tensor export: recipe ``fp8`` and
    granularity ``per-tensor``).
    """
    violations: list[str] = []

    recipe = config.get("recipe")
    if recipe != EXPECTED_RECIPE:
        violations.append(f"recipe: expected {EXPECTED_RECIPE!r}, found {recipe!r}")

    scale_layout = config.get("scale_layout") or {}
    granularity = scale_layout.get("granularity")
    if granularity != EXPECTED_GRANULARITY:
        violations.append(f"scale granularity: expected {EXPECTED_GRANULARITY!r}, found {granularity!r}")
    block_sizes = scale_layout.get("block_sizes") or {}
    block_rows = block_sizes.get("rows")
    block_cols = block_sizes.get("cols")
    if not (isinstance(block_rows, int) and isinstance(block_cols, int)):
        violations.append(f"block_sizes: expected integer rows/cols, found {block_sizes!r}")

    mixed = config.get("mixed_precision") or {}
    quantized_patterns = tuple(mixed.get("quantized") or ())
    kept_patterns = tuple(mixed.get("bf16_kept") or ())
    n_quantized = mixed.get("n_quantized")
    if not quantized_patterns:
        violations.append("mixed_precision.quantized: missing or empty")
    if not isinstance(n_quantized, int) or n_quantized <= 0:
        violations.append(f"mixed_precision.n_quantized: expected positive int, found {n_quantized!r}")

    n_scale = scale_layout.get("n_scale")
    if not isinstance(n_scale, int):
        violations.append(f"scale_layout.n_scale: expected int, found {n_scale!r}")
    elif isinstance(n_quantized, int) and n_scale != 2 * n_quantized:
        violations.append(f"scale_layout.n_scale: expected 2*n_quantized = {2 * n_quantized}, found {n_scale}")

    if violations:
        raise CheckpointIntegrityError(
            "quantization_config.json is not the supported ModelOpt-native "
            "FP8-blockwise deliverable:\n  " + "\n  ".join(violations)
        )
    # Narrow for the type checker: a clean violations list implies these hold.
    assert (
        isinstance(recipe, str)
        and isinstance(n_quantized, int)
        and isinstance(n_scale, int)
        and isinstance(block_rows, int)
        and isinstance(block_cols, int)
    )
    return BlockwiseQuantSpec(
        recipe=recipe,
        n_quantized=n_quantized,
        n_scale=n_scale,
        block_rows=block_rows,
        block_cols=block_cols,
        quantized_patterns=quantized_patterns,
        kept_patterns=kept_patterns,
    )


def classify_name(name: str) -> TensorKind:
    if name.endswith(SCALE_SUFFIX):
        return TensorKind.QUANTIZER_SCALE
    if name.endswith(AMAX_SUFFIX):
        return TensorKind.QUANTIZER_AMAX
    return TensorKind.OTHER


def is_fp8_dtype(dtype: torch.dtype) -> bool:
    return dtype in FP8_DTYPES


def is_fp8_dtype_str(dtype_str: str) -> bool:
    """FP8 test for safetensors-header dtype tags (e.g. ``"F8_E4M3"``)."""
    return dtype_str.upper() in FP8_HEADER_DTYPES


def scale_name_for(weight_name: str) -> str | None:
    """``X.weight`` -> ``X.weight_quantizer._scale``; None for non-weights."""
    if weight_name.endswith(WEIGHT_SUFFIX):
        return weight_name[: -len(WEIGHT_SUFFIX)] + SCALE_SUFFIX
    return None


def weight_name_for(scale_name: str) -> str | None:
    """``X.weight_quantizer._scale`` -> ``X.weight``; None otherwise."""
    if scale_name.endswith(SCALE_SUFFIX):
        return scale_name[: -len(SCALE_SUFFIX)] + WEIGHT_SUFFIX
    return None


def matches_any(module_path: str, patterns: Iterable[str]) -> bool:
    """Sidecar pattern semantics: ``P.*`` matches any module with a path
    component ``P``; a bare pattern matches the exact module path (top-level
    or as a trailing component)."""
    for pattern in patterns:
        if pattern.endswith(".*"):
            component = pattern[: -len(".*")]
            if f".{component}." in f".{module_path}.":
                return True
        elif module_path == pattern or module_path.endswith(f".{pattern}"):
            return True
    return False


def _expected_grid(shape: tuple[int, ...], block: tuple[int, int]) -> tuple[int, int]:
    return (math.ceil(shape[0] / block[0]), math.ceil(shape[1] / block[1]))


def expand_block_scale(
    scale: torch.Tensor,
    weight_shape: tuple[int, ...],
    block: tuple[int, int],
) -> torch.Tensor:
    """Broadcast a 2D per-block scale grid to the full weight shape.

    Pure; returns a new tensor. Raises :class:`CheckpointIntegrityError` when
    the grid does not equal the ceil-div block grid of the weight shape.
    """
    if len(weight_shape) != 2 or scale.ndim != 2:
        raise CheckpointIntegrityError(
            f"blockwise dequant needs 2D weight and 2D scale, got weight "
            f"{tuple(weight_shape)} and scale {tuple(scale.shape)}"
        )
    expected = _expected_grid(weight_shape, block)
    if tuple(scale.shape) != expected:
        raise CheckpointIntegrityError(
            f"scale grid shape mismatch: scale {tuple(scale.shape)} vs expected "
            f"grid {expected} for weight {tuple(weight_shape)} at block {block}"
        )
    rows, cols = weight_shape
    return scale.repeat_interleave(block[0], dim=0)[:rows].repeat_interleave(block[1], dim=1)[:, :cols]


def dequantize_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    target_dtype: torch.dtype,
    block: tuple[int, int],
) -> torch.Tensor:
    """Dequantize FP8 codes with their 2D block-scale grid (fp32 intermediate).

    *block* is the DECLARED block size (from the sidecar), not inferred from the
    tensor shapes. Inferring `ceil(weight/scale)` would place block boundaries
    wrong for any weight dimension not divisible by the block (the last block is
    partial), silently dequantizing with the wrong per-block scale while
    :func:`verify_observations` — which also uses the declared block — still
    passes. `expand_block_scale` rejects a scale grid that disagrees with the
    declared block, so a mismatch raises here (before the weight is yielded)
    rather than corrupting silently. Pure; returns a new tensor.
    """
    full_scale = expand_block_scale(scale.to(torch.float32), tuple(weight.shape), block)
    return (weight.to(torch.float32) * full_scale).to(target_dtype)


def verify_observations(
    infos: Iterable[TensorInfo],
    spec: BlockwiseQuantSpec,
) -> list[str]:
    """Stream/header verification against the sidecar contract (pure).

    Checks fp8 tensors against the declared quantized/kept patterns, pairs
    each fp8 weight with a correctly-shaped ``_scale`` grid, and compares the
    observed counts with the sidecar declaration. Returns human-readable
    violations; ``[]`` when clean.
    """
    infos = list(infos)
    fp8_weights: dict[str, tuple[int, ...]] = {}
    scales: dict[str, tuple[int, ...]] = {}
    n_amax = 0
    violations: list[str] = []

    for info in infos:
        kind = classify_name(info.name)
        if kind is TensorKind.QUANTIZER_SCALE:
            scales[info.name] = info.shape
        elif kind is TensorKind.QUANTIZER_AMAX:
            n_amax += 1
        elif info.is_fp8:
            if not info.name.endswith(WEIGHT_SUFFIX):
                violations.append(f"unexpected fp8 tensor (not a .weight): {info.name}")
                continue
            fp8_weights[info.name] = info.shape

    block = (spec.block_rows, spec.block_cols)
    for name, shape in sorted(fp8_weights.items()):
        module_path = name[: -len(WEIGHT_SUFFIX)]
        if matches_any(module_path, spec.kept_patterns):
            violations.append(
                f"excluded (bf16-kept) module found quantized: {name} (kept patterns: {list(spec.kept_patterns)})"
            )
        elif not matches_any(module_path, spec.quantized_patterns):
            violations.append(
                f"fp8 tensor outside declared quantized patterns: {name} "
                f"(quantized patterns: {list(spec.quantized_patterns)})"
            )
        scale_name = scale_name_for(name)
        if scale_name not in scales:
            violations.append(f"missing {SCALE_SUFFIX} companion for fp8 weight {name}")
        elif len(shape) == 2:
            expected = _expected_grid(shape, block)
            if scales[scale_name] != expected:
                violations.append(
                    f"scale grid shape mismatch for {name}: scale {scales[scale_name]} vs expected grid {expected}"
                )

    for scale_name in sorted(scales):
        weight_name = weight_name_for(scale_name)
        if weight_name not in fp8_weights:
            violations.append(f"orphan quantizer scale without fp8 weight: {scale_name}")

    if len(fp8_weights) != spec.n_quantized:
        violations.append(f"quantized-weight count mismatch: declared {spec.n_quantized}, observed {len(fp8_weights)}")
    n_scale_family = len(scales) + n_amax
    if n_scale_family != spec.n_scale:
        violations.append(
            f"scale-tensor count mismatch: declared {spec.n_scale}, "
            f"observed {n_scale_family} ({len(scales)} _scale + {n_amax} _amax)"
        )
    return violations


def assert_not_fp8(name: str, dtype: torch.dtype) -> None:
    """Last-line guard: no fp8 tensor may reach weight loading unadapted.

    An unadapted fp8 tensor would be silently cast without its dequant scale
    (weights off by the per-block scale factor). Raises
    :class:`CheckpointIntegrityError`.
    """
    if is_fp8_dtype(dtype):
        raise CheckpointIntegrityError(
            f"fp8 tensor {name!r} reached weight loading without dequantization; "
            "quantized checkpoints require the ModelOpt-native checkpoint adapter "
            f"(sidecar {SIDECAR_FILENAME} at the model root)"
        )


def resolved_model_root(source: object) -> str | PathLike | None:
    """Return the resolved local model root for sidecar reads.

    ``model_or_path`` remains the user-facing repo ID/path. The diffusers
    loader sets ``resolved_model_or_path`` after HF download/local resolution
    so checkpoint adapters can inspect sidecars in the actual local snapshot.
    """
    resolved = getattr(source, "resolved_model_or_path", None)
    if resolved is not None:
        return resolved
    return getattr(source, "model_or_path", None)


# --- adapter (Action shell) ----------------------------------------------------


class ModelOptNativeFp8CheckpointAdapter:
    """Streams a ModelOpt-native FP8-blockwise checkpoint as full precision."""

    def __init__(
        self,
        spec: BlockwiseQuantSpec,
        source_prefix: str,
        target_dtype: torch.dtype,
    ) -> None:
        self._spec = spec
        self._prefix = source_prefix
        self._target_dtype = target_dtype

    @staticmethod
    def _is_transformer_source(source: object) -> bool:
        if getattr(source, "subfolder", None) == "transformer":
            return True
        return str(getattr(source, "prefix", "")).startswith("transformer.")

    @classmethod
    def _parse_source_sidecar(cls, source: object) -> BlockwiseQuantSpec | None:
        """Read + validate the source dir's sidecar (Action at the boundary).

        Returns None when the source is not a local transformer dir with a
        sidecar. Raises :class:`CheckpointIntegrityError` when a sidecar
        exists but does not describe the supported deliverable (fail fast —
        never load a mislabeled checkpoint).
        """
        if not cls._is_transformer_source(source):
            return None
        model_dir = resolved_model_root(source)
        if not isinstance(model_dir, (str, os.PathLike)) or not os.path.isdir(model_dir):
            return None
        sidecar_path = os.path.join(model_dir, SIDECAR_FILENAME)
        if not os.path.exists(sidecar_path):
            return None
        try:
            with open(sidecar_path) as f:
                config = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise CheckpointIntegrityError(f"unreadable quantization sidecar {sidecar_path}: {e}") from e
        return parse_quant_spec(config)

    @classmethod
    def validate_source_sidecar(cls, source: object) -> None:
        """Pre-flight check: raise on a present-but-invalid sidecar.

        Called by the loader *before* weight-file discovery so a mislabeled
        quantized checkpoint dies with the integrity report rather than a
        generic file-discovery error.
        """
        cls._parse_source_sidecar(source)

    @classmethod
    def detect(
        cls,
        source: object,
        target_dtype: torch.dtype = torch.bfloat16,
    ) -> "ModelOptNativeFp8CheckpointAdapter | None":
        """Engage iff the source's local model dir carries a supported sidecar.

        Returns None for unquantized checkpoints (no sidecar); raises like
        :meth:`validate_source_sidecar` on an invalid one.
        """
        spec = cls._parse_source_sidecar(source)
        if spec is None:
            return None
        model_dir = resolved_model_root(source)
        logger.info(
            "ModelOpt-native FP8-blockwise checkpoint detected at %s "
            "(declared: %d quantized modules, %d scale tensors, %dx%d blocks)",
            model_dir,
            spec.n_quantized,
            spec.n_scale,
            spec.block_rows,
            spec.block_cols,
        )
        return cls(
            spec=spec,
            source_prefix=str(getattr(source, "prefix", "") or ""),
            target_dtype=target_dtype,
        )

    def _strip_prefix(self, name: str) -> str:
        if self._prefix and name.startswith(self._prefix):
            return name[len(self._prefix) :]
        return name

    def adapt(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Dequantize fp8 weights, consume quantizer tensors, verify at end.

        Yields a full-precision stream. Raises
        :class:`CheckpointIntegrityError` (aggregated) on any integrity
        violation, before the caller can finish loading.
        """
        scales: dict[str, torch.Tensor] = {}
        pending: dict[str, list[tuple[str, torch.Tensor]]] = {}
        n_pending = 0
        infos: list[TensorInfo] = []
        dequantized = 0
        block = (self._spec.block_rows, self._spec.block_cols)  # declared, not inferred

        for full_name, tensor in weights:
            name = self._strip_prefix(full_name)
            kind = classify_name(name)
            infos.append(TensorInfo(name, is_fp8_dtype(tensor.dtype), tuple(tensor.shape)))

            if kind is TensorKind.QUANTIZER_SCALE:
                scales[name] = tensor
                for pending_full_name, pending_weight in pending.pop(name, ()):
                    n_pending -= 1
                    dequantized += 1
                    yield (
                        pending_full_name,
                        dequantize_weight(pending_weight, tensor, self._target_dtype, block),
                    )
                continue
            if kind is TensorKind.QUANTIZER_AMAX:
                continue  # informational twin of _scale; consumed

            if is_fp8_dtype(tensor.dtype):
                scale_name = scale_name_for(name)
                if scale_name is None:
                    continue  # flagged by verify_observations at end of stream
                if scale_name in scales:
                    dequantized += 1
                    yield full_name, dequantize_weight(tensor, scales[scale_name], self._target_dtype, block)
                else:
                    pending.setdefault(scale_name, []).append((full_name, tensor))
                    n_pending += 1
                    if n_pending > MAX_PENDING_FP8:
                        waiting = ", ".join(sorted(pending))
                        raise CheckpointIntegrityError(
                            f"more than {MAX_PENDING_FP8} fp8 weights pending without "
                            f"their quantizer scales — scales appear systematically "
                            f"absent; first missing: {waiting}"
                        )
                continue

            yield full_name, tensor

        violations = verify_observations(infos, self._spec)
        if violations:
            raise CheckpointIntegrityError(
                "quantized checkpoint failed integrity verification "
                f"({len(violations)} violation(s)):\n  " + "\n  ".join(violations)
            )
        logger.info(
            "ModelOpt-native FP8 adapter: dequantized %d/%d quantized weights to %s (marker: %s)",
            dequantized,
            self._spec.n_quantized,
            self._target_dtype,
            FP8_BLOCKWISE_MARKER,
        )


# --- CLI probe (header-only, GPU-free) -------------------------------------------


# safetensors reference caps the JSON header at 100 MB; bound the read so a
# malformed/hostile length field cannot request an arbitrary allocation (the
# probe only runs against the trusted :ro mount, so this is defense-in-depth).
_MAX_SAFETENSORS_HEADER_BYTES = 100 * 1024 * 1024


def _read_safetensors_header(path: str) -> dict:
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        if header_len > _MAX_SAFETENSORS_HEADER_BYTES:
            raise CheckpointIntegrityError(
                f"safetensors header of {path} is {header_len} bytes "
                f"(> {_MAX_SAFETENSORS_HEADER_BYTES}); refusing to read"
            )
        header = json.loads(f.read(header_len))
    header.pop("__metadata__", None)
    return header


def header_tensor_infos(header: Mapping) -> list[TensorInfo]:
    """Pure: safetensors header dict -> normalized TensorInfo list."""
    return [TensorInfo(name, is_fp8_dtype_str(entry["dtype"]), tuple(entry["shape"])) for name, entry in header.items()]


def main(argv: list[str] | None = None) -> int:
    """Integrity probe: exit 0 iff *model_dir* is the supported deliverable."""
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("usage: python -m ...modelopt_native <model_dir>", file=sys.stderr)
        return 2
    model_dir = args[0]

    sidecar_path = os.path.join(model_dir, SIDECAR_FILENAME)
    if not os.path.exists(sidecar_path):
        print(f"INTEGRITY FAIL: no {SIDECAR_FILENAME} sidecar in {model_dir}")
        return 1
    try:
        with open(sidecar_path) as f:
            spec = parse_quant_spec(json.load(f))
    except (OSError, json.JSONDecodeError, CheckpointIntegrityError) as e:
        print(f"INTEGRITY FAIL: {e}")
        return 1

    shard_paths = sorted(glob.glob(os.path.join(model_dir, "transformer", "*.safetensors")))
    if not shard_paths:
        print(f"INTEGRITY FAIL: no transformer/*.safetensors under {model_dir}")
        return 1
    infos: list[TensorInfo] = []
    for shard_path in shard_paths:
        try:
            infos.extend(header_tensor_infos(_read_safetensors_header(shard_path)))
        except (OSError, ValueError, struct.error, json.JSONDecodeError) as e:
            print(f"INTEGRITY FAIL: unreadable safetensors header {shard_path}: {e}")
            return 1

    n_fp8 = sum(1 for i in infos if i.is_fp8)
    n_scale_family = sum(1 for i in infos if classify_name(i.name) is not TensorKind.OTHER)
    violations = verify_observations(infos, spec)
    print(
        f"declared: {spec.n_quantized} quantized / {spec.n_scale} scale tensors; "
        f"observed: {n_fp8} fp8 / {n_scale_family} quantizer tensors "
        f"({len(infos)} total, {len(shard_paths)} shard(s))"
    )
    if violations:
        print(f"INTEGRITY FAIL ({len(violations)} violation(s)):")
        for violation in violations:
            print(f"  {violation}")
        return 1
    print(f"INTEGRITY OK: {model_dir} (marker: {FP8_BLOCKWISE_MARKER})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
