# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ModelOpt-native NVFP4 (weight-only, blockwise) checkpoint adapter.

Loads ``nvfp4_blockwise_mixed_v1`` checkpoints. On-disk format, per quantized target module
(``layers.*.mlp.*`` and ``layers.*.mlp_moe_gen.*`` only):

- ``<module>.weight_packed``       uint8, packed E2M1 FP4 (two codes / byte
                                   along the last dim), shape ``[out, ceil(in/2)]``;
- ``<module>.weight_block_scale``  float8_e4m3fn per-16-block scales, shape
                                   ``[out, ceil(in/16)]`` (blocked along last dim);
- ``<module>.weight_global_scale`` float32 per-tensor scale, shape ``[1]``.

Non-target tensors (attention, embeddings, ``lm_head``, norms, projections,
audio/action modules) stay BF16. A ``transformer/nvfp4_blockwise_mixed_v1.json``
sidecar declares the recipe, block size, scale encoding, the exact quantized
target set (name -> shapes), and the target/forbidden patterns.

Unlike the sibling FP8 native adapter (:mod:`.modelopt_native`), this adapter
**never dequantizes a target weight**. It renames the on-disk names to the
param names vLLM's ``ModelOptNvFp4W4A16LinearMethod`` allocates
(``weight_packed -> weight``, ``weight_block_scale -> weight_scale``,
``weight_global_scale -> weight_scale_2``) and passes the bytes through
unchanged, so target modules remain FP4-resident. Integrity is verified
fail-fast: a present-but-invalid sidecar aborts at detection; count /
pattern / shape / dtype violations and any NaN/Inf scale byte abort before
serving, aggregated into a single :class:`CheckpointIntegrityError`.

Run ``python -m vllm_omni.diffusion.model_loader.checkpoint_adapters.\
modelopt_native_nvfp4 <model_dir>`` for a GPU-free header and sidecar
probe (exit 0 = pass, 1 = fail, 2 = usage).

Pure calculations (no I/O): :func:`parse_nvfp4_spec`, :func:`remap_name`,
:func:`weight_key_for`, :func:`classify_suffix`, :func:`matches_any`,
:func:`expected_packed_width`, :func:`expected_scale_grid`,
:func:`dtype_tag`, :func:`verify_observations`, :func:`assert_scales_finite`.
Actions: :meth:`ModelOptNativeNvfp4CheckpointAdapter.detect`,
:meth:`~ModelOptNativeNvfp4CheckpointAdapter.adapt`, :func:`main`.
"""

import glob
import json
import math
import os
import re
import struct
import sys
from collections.abc import Generator, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

# Deploy-side marker for the mount-shadowing probe (mirrors the FP8 adapter):
# its presence distinguishes the patched tree from a stock install at runtime.
NVFP4_BLOCKWISE_MARKER = "modelopt-nvfp4-blockwise"

SIDECAR_FILENAME = "nvfp4_blockwise_mixed_v1.json"
EXPECTED_RECIPE = "nvfp4_blockwise_mixed_v1"
BLOCK_SIZE = 16

PACKED_SUFFIX = ".weight_packed"
BLOCK_SUFFIX = ".weight_block_scale"
GLOBAL_SUFFIX = ".weight_global_scale"
WEIGHT_SUFFIX = ".weight"
# Target param names allocated by vLLM's ModelOptNvFp4W4A16LinearMethod.
REMAP = {
    PACKED_SUFFIX: ".weight",
    BLOCK_SUFFIX: ".weight_scale",
    GLOBAL_SUFFIX: ".weight_scale_2",
}

# Normalized dtype tags (unify torch dtypes with safetensors header dtype tags).
TAG_U8 = "u8"
TAG_F8E4M3 = "f8e4m3"
TAG_F32 = "f32"
TAG_BF16 = "bf16"


class CheckpointIntegrityError(ValueError):
    """NVFP4 checkpoint integrity violation. Fail fast."""


class TensorInfo(NamedTuple):
    """Normalized view of one checkpoint tensor (shared by stream + header)."""

    name: str
    tag: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class Nvfp4BlockwiseSpec:
    """Parsed ``nvfp4_blockwise_mixed_v1.json`` sidecar (Data)."""

    recipe: str
    block_size: int
    expected_count: int
    target_patterns: tuple[str, ...]
    forbidden_patterns: tuple[str, ...]
    packed_tag: str
    block_scale_tag: str
    global_scale_tag: str
    # weight-name -> {"weight_shape", "packed_shape", "block_scale_shape"}
    tensors: Mapping[str, Mapping[str, Sequence[int]]]


# --- pure calculations -------------------------------------------------------


def dtype_tag(dtype: "torch.dtype | str") -> str:
    """Normalize a torch dtype or safetensors header dtype string to a tag."""
    if isinstance(dtype, str):
        return {
            "U8": TAG_U8,
            "F8_E4M3": TAG_F8E4M3,
            "F32": TAG_F32,
            "BF16": TAG_BF16,
        }.get(dtype.upper(), dtype.upper())
    return {
        torch.uint8: TAG_U8,
        getattr(torch, "float8_e4m3fn", None): TAG_F8E4M3,
        torch.float32: TAG_F32,
        torch.bfloat16: TAG_BF16,
    }.get(dtype, str(dtype))


def parse_nvfp4_spec(config: Mapping) -> Nvfp4BlockwiseSpec:
    """Validate + parse the sidecar, naming every discrepancy (fail fast)."""
    violations: list[str] = []

    recipe = config.get("recipe")
    if recipe != EXPECTED_RECIPE:
        violations.append(f"recipe: expected {EXPECTED_RECIPE!r}, found {recipe!r}")

    block_size = config.get("block_size")
    if block_size != BLOCK_SIZE:
        violations.append(f"block_size: expected {BLOCK_SIZE}, found {block_size!r}")

    enc = config.get("scale_encoding") or {}
    if enc.get("packed_dtype") != "uint8":
        violations.append(f"scale_encoding.packed_dtype: expected 'uint8', found {enc.get('packed_dtype')!r}")
    if enc.get("block_scale_dtype") != "float8_e4m3fn":
        violations.append(
            f"scale_encoding.block_scale_dtype: expected 'float8_e4m3fn', found {enc.get('block_scale_dtype')!r}"
        )
    if enc.get("global_scale_dtype") != "float32":
        violations.append(
            f"scale_encoding.global_scale_dtype: expected 'float32', found {enc.get('global_scale_dtype')!r}"
        )

    target_patterns = tuple(config.get("target_patterns") or ())
    forbidden_patterns = tuple(config.get("forbidden_patterns") or ())
    if not target_patterns:
        violations.append("target_patterns: missing or empty")

    tensors = config.get("tensors")
    if not isinstance(tensors, Mapping) or not tensors:
        violations.append("tensors: missing or empty manifest")
        tensors = {}
    else:
        for tname, decl in tensors.items():
            ws = decl.get("weight_shape") if isinstance(decl, Mapping) else None
            if not (isinstance(ws, (list, tuple)) and len(ws) == 2 and all(isinstance(d, int) for d in ws)):
                violations.append(
                    f"tensors[{tname!r}].weight_shape: expected [rows, cols] ints, found {ws!r}"
                )

    expected_count = config.get("expected_quantized_count")
    if not isinstance(expected_count, int) or expected_count <= 0:
        violations.append(f"expected_quantized_count: expected positive int, found {expected_count!r}")
    elif tensors and len(tensors) != expected_count:
        violations.append(
            f"expected_quantized_count: declared {expected_count} but manifest lists {len(tensors)} tensors"
        )

    if violations:
        raise CheckpointIntegrityError(
            "nvfp4_blockwise_mixed_v1.json is not the supported NVFP4 deliverable:\n  "
            + "\n  ".join(violations)
        )
    assert isinstance(recipe, str) and isinstance(block_size, int) and isinstance(expected_count, int)
    return Nvfp4BlockwiseSpec(
        recipe=recipe,
        block_size=block_size,
        expected_count=expected_count,
        target_patterns=target_patterns,
        forbidden_patterns=forbidden_patterns,
        packed_tag=TAG_U8,
        block_scale_tag=TAG_F8E4M3,
        global_scale_tag=TAG_F32,
        tensors=tensors,
    )


def classify_suffix(name: str) -> str | None:
    """Return which NVFP4 target-tensor family *name* belongs to, or None."""
    for suffix in (PACKED_SUFFIX, BLOCK_SUFFIX, GLOBAL_SUFFIX):
        if name.endswith(suffix):
            return suffix
    return None


def remap_name(name: str) -> str:
    """Rename an on-disk NVFP4 tensor to vLLM's W4A16 param name (pure).

    ``*.weight_packed`` -> ``*.weight``; ``*.weight_block_scale`` ->
    ``*.weight_scale``; ``*.weight_global_scale`` -> ``*.weight_scale_2``.
    Any other name (BF16 passthrough) is returned unchanged.
    """
    for suffix, target in REMAP.items():
        if name.endswith(suffix):
            return name[: -len(suffix)] + target
    return name


def weight_key_for(name: str) -> str | None:
    """Map any target family tensor name to its sidecar weight-name key.

    ``X.weight_packed``/``X.weight_block_scale``/``X.weight_global_scale`` ->
    ``X.weight``; None for non-target names.
    """
    suffix = classify_suffix(name)
    if suffix is None:
        return None
    return name[: -len(suffix)] + WEIGHT_SUFFIX


def module_of(weight_name: str) -> str:
    return weight_name[: -len(WEIGHT_SUFFIX)] if weight_name.endswith(WEIGHT_SUFFIX) else weight_name


def matches_any(module_path: str, patterns: Iterable[str]) -> bool:
    """True if *module_path* matches any pattern (regex; plain substrings are
    valid regexes and match as substrings)."""
    return any(re.search(pattern, module_path) for pattern in patterns)


def expected_packed_width(cols: int) -> int:
    """Packed last-dim width: two FP4 codes per uint8 byte."""
    return (cols + 1) // 2


def expected_scale_grid(shape: tuple[int, ...], block: int) -> tuple[int, int]:
    """Block-scale grid for a 2D weight, blocked along the last dim."""
    rows, cols = shape
    return (rows, math.ceil(cols / block))


def assert_scales_finite(name: str, tensor: torch.Tensor) -> None:
    """Raise on any NaN/Inf scale byte (owner decision: abort, do not clamp)."""
    as_f32 = tensor.to(torch.float32)
    if not torch.isfinite(as_f32).all():
        raise CheckpointIntegrityError(
            f"scale tensor {name!r} contains NaN/Inf bytes; the artifact is corrupt "
            "(the producer guarantees finite scales). Refusing to serve."
        )


def verify_observations(infos: Iterable[TensorInfo], spec: Nvfp4BlockwiseSpec) -> list[str]:
    """Structural verification against the sidecar manifest (pure).

    Uses the sidecar ``tensors`` manifest as the authoritative target set:
    every declared target must have a correctly-shaped/typed packed + block +
    global companion; no undeclared or forbidden module may be packed; counts
    must match. Returns human-readable violations; ``[]`` when clean. (Does not
    inspect values; NaN/Inf is checked in :meth:`adapt`.)
    """
    packed: dict[str, TensorInfo] = {}
    block: dict[str, TensorInfo] = {}
    glob: dict[str, TensorInfo] = {}
    for info in infos:
        wkey = weight_key_for(info.name)
        if wkey is None:
            continue
        if info.name.endswith(PACKED_SUFFIX):
            packed[wkey] = info
        elif info.name.endswith(BLOCK_SUFFIX):
            block[wkey] = info
        elif info.name.endswith(GLOBAL_SUFFIX):
            glob[wkey] = info

    violations: list[str] = []

    if len(packed) != spec.expected_count:
        violations.append(
            f"quantized-target count mismatch: declared {spec.expected_count}, observed {len(packed)}"
        )

    for wname, info in sorted(packed.items()):
        module = module_of(wname)
        if matches_any(module, spec.forbidden_patterns):
            violations.append(f"forbidden module found quantized: {info.name}")
        if not matches_any(module, spec.target_patterns):
            violations.append(f"packed tensor outside target patterns: {info.name}")
        if wname not in spec.tensors:
            violations.append(f"packed tensor not declared in sidecar manifest: {info.name}")
        else:
            decl = spec.tensors[wname]
            wshape = tuple(decl["weight_shape"])
            exp_packed = (wshape[0], expected_packed_width(wshape[-1]))
            if info.shape != exp_packed:
                violations.append(
                    f"packed shape mismatch for {info.name}: found {info.shape}, expected {exp_packed}"
                )
        if info.tag != TAG_U8:
            violations.append(f"packed tensor {info.name} must be uint8, found {info.tag}")

        # companions
        binfo = block.get(wname)
        if binfo is None:
            violations.append(f"missing {BLOCK_SUFFIX} companion for {info.name}")
        else:
            if binfo.tag != TAG_F8E4M3:
                violations.append(f"block scale {binfo.name} must be float8_e4m3fn, found {binfo.tag}")
            if wname in spec.tensors:
                wshape = tuple(spec.tensors[wname]["weight_shape"])
                exp_grid = expected_scale_grid(wshape, spec.block_size)
                if binfo.shape != exp_grid:
                    violations.append(
                        f"block-scale grid mismatch for {binfo.name}: found {binfo.shape}, expected {exp_grid}"
                    )
        ginfo = glob.get(wname)
        if ginfo is None:
            violations.append(f"missing {GLOBAL_SUFFIX} companion for {info.name}")
        else:
            if ginfo.tag != TAG_F32:
                violations.append(f"global scale {ginfo.name} must be float32, found {ginfo.tag}")
            if math.prod(ginfo.shape) != 1:
                violations.append(f"global scale {ginfo.name} must have exactly 1 element, found shape {ginfo.shape}")

    for wname in spec.tensors:
        if wname not in packed:
            violations.append(f"declared target missing its packed weight: {wname}")

    for wname in block:
        if wname not in packed:
            violations.append(f"orphan block scale without packed weight: {wname}")
    for wname in glob:
        if wname not in packed:
            violations.append(f"orphan global scale without packed weight: {wname}")

    return violations


# --- adapter (Action shell) --------------------------------------------------


class ModelOptNativeNvfp4CheckpointAdapter:
    """Streams the NVFP4 blockwise checkpoint FP4-resident (rename, no dequant)."""

    def __init__(self, spec: Nvfp4BlockwiseSpec, source_prefix: str = "") -> None:
        self._spec = spec
        self._prefix = source_prefix

    @staticmethod
    def _is_transformer_source(source: object) -> bool:
        if getattr(source, "subfolder", None) == "transformer":
            return True
        return str(getattr(source, "prefix", "")).startswith("transformer.")

    @classmethod
    def _sidecar_path(cls, model_dir: "str | os.PathLike") -> str:
        return os.path.join(model_dir, "transformer", SIDECAR_FILENAME)

    @classmethod
    def _parse_source_sidecar(cls, source: object) -> Nvfp4BlockwiseSpec | None:
        """Read + validate the source's ``transformer/`` sidecar (Action).

        Returns None when the source is not a local transformer dir carrying the
        NVFP4 sidecar. Raises :class:`CheckpointIntegrityError` when a sidecar is
        present but does not describe the supported deliverable (fail fast).
        """
        if not cls._is_transformer_source(source):
            return None
        model_dir = getattr(source, "model_or_path", None)
        if not isinstance(model_dir, (str, os.PathLike)) or not os.path.isdir(model_dir):
            return None
        sidecar_path = cls._sidecar_path(model_dir)
        if not os.path.exists(sidecar_path):
            return None
        try:
            with open(sidecar_path) as f:
                config = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise CheckpointIntegrityError(f"unreadable NVFP4 sidecar {sidecar_path}: {e}") from e
        return parse_nvfp4_spec(config)

    @classmethod
    def validate_source_sidecar(cls, source: object) -> None:
        """Pre-flight check: raise on a present-but-invalid sidecar."""
        cls._parse_source_sidecar(source)

    @classmethod
    def detect(
        cls,
        source: object,
        target_dtype: torch.dtype = torch.bfloat16,  # accepted for parity; unused (no dequant)
    ) -> "ModelOptNativeNvfp4CheckpointAdapter | None":
        spec = cls._parse_source_sidecar(source)
        if spec is None:
            return None
        model_dir = getattr(source, "model_or_path", None)
        logger.info(
            "ModelOpt-native NVFP4 checkpoint detected at %s (declared: %d quantized targets, block %d)",
            model_dir, spec.expected_count, spec.block_size,
        )
        return cls(spec=spec, source_prefix=str(getattr(source, "prefix", "") or ""))

    def _strip_prefix(self, name: str) -> str:
        if self._prefix and name.startswith(self._prefix):
            return name[len(self._prefix):]
        return name

    def adapt(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Rename target tensors and pass all bytes through unchanged.

        Yields ``(remapped_full_name, tensor)``. Target packed/scale tensors are
        renamed to vLLM's W4A16 param names; scales are checked NaN/Inf-finite;
        no target is ever dequantized. Raises aggregated
        :class:`CheckpointIntegrityError` at end-of-stream on any violation.
        """
        infos: list[TensorInfo] = []
        for full_name, tensor in weights:
            name = self._strip_prefix(full_name)
            suffix = classify_suffix(name)
            infos.append(TensorInfo(name, dtype_tag(tensor.dtype), tuple(tensor.shape)))
            if suffix in (BLOCK_SUFFIX, GLOBAL_SUFFIX):
                assert_scales_finite(name, tensor)
            # rename (target families) or pass through (BF16 non-targets), never dequant
            yield remap_name(full_name), tensor

        violations = verify_observations(infos, self._spec)
        if violations:
            raise CheckpointIntegrityError(
                f"NVFP4 checkpoint failed integrity verification ({len(violations)} violation(s)):\n  "
                + "\n  ".join(violations)
            )
        logger.info(
            "ModelOpt-native NVFP4 adapter: passed %d FP4-resident targets through (marker: %s)",
            self._spec.expected_count, NVFP4_BLOCKWISE_MARKER,
        )


# --- CLI probe (header-only, GPU-free) --------------------------------------

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
    return [
        TensorInfo(name, dtype_tag(entry["dtype"]), tuple(entry["shape"]))
        for name, entry in header.items()
    ]


def main(argv: list[str] | None = None) -> int:
    """Integrity probe: exit 0 iff *model_dir* is the supported NVFP4 deliverable."""
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("usage: python -m ...modelopt_native_nvfp4 <model_dir>", file=sys.stderr)
        return 2
    model_dir = args[0]

    sidecar_path = ModelOptNativeNvfp4CheckpointAdapter._sidecar_path(model_dir)
    if not os.path.exists(sidecar_path):
        print(f"INTEGRITY FAIL: no transformer/{SIDECAR_FILENAME} sidecar in {model_dir}")
        return 1
    try:
        with open(sidecar_path) as f:
            spec = parse_nvfp4_spec(json.load(f))
    except (OSError, json.JSONDecodeError, CheckpointIntegrityError) as e:
        print(f"INTEGRITY FAIL: {e}")
        return 1

    shard_paths = sorted(glob.glob(os.path.join(model_dir, "transformer", "*.safetensors")))
    if not shard_paths:
        print(f"INTEGRITY FAIL: no transformer/*.safetensors under {model_dir}")
        return 1
    infos: list[TensorInfo] = []
    for shard_path in shard_paths:
        infos.extend(header_tensor_infos(_read_safetensors_header(shard_path)))

    n_packed = sum(1 for i in infos if i.name.endswith(PACKED_SUFFIX))
    violations = verify_observations(infos, spec)
    print(
        f"declared: {spec.expected_count} quantized targets (block {spec.block_size}); "
        f"observed: {n_packed} packed weights ({len(infos)} tensors, {len(shard_paths)} shard(s))"
    )
    if violations:
        print(f"INTEGRITY FAIL ({len(violations)} violation(s)):")
        for violation in violations:
            print(f"  {violation}")
        return 1
    print(f"INTEGRITY OK: {model_dir} (marker: {NVFP4_BLOCKWISE_MARKER})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
