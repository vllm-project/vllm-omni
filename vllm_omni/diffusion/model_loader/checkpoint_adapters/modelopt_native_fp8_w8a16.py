# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ModelOpt-native FP8-blockwise W8A16 resident checkpoint adapter.

Loads the SAME on-disk deliverable as the sibling dequant adapter
(:mod:`.modelopt_native`) — recipe ``fp8_blockwise_mixed``, blockwise-128x128, FP8
``<m>.weight`` + 2D ``<m>.weight_quantizer._scale`` (bf16) + ``._amax`` twin, described
by the root ``quantization_config.json`` — but streams it for the **weight-resident**
W8A16 path instead of dequantizing on load:

- the 216 ``mlp.*``/``mlp_moe_gen.*`` targets stay **FP8-resident** (bytes unchanged);
  their ``.weight_quantizer._scale`` is renamed to ``.weight_scale`` (the param name
  :class:`...fp8_blockwise_w8a16.Fp8BlockwiseW8A16LinearMethod` allocates) and passed
  through resident;
- ``lm_head`` and any other non-target FP8 weight are **dequantized to BF16** (reusing
  :func:`.modelopt_native.dequantize_weight`), so only the MLP targets are resident and
  ``lm_head`` stays BF16 at compute;
- BF16 non-targets (attention, norms, projections) pass through unchanged;
- ``._amax`` twins are consumed (informational).

Integrity is fail-fast, reusing the dequant adapter's
:func:`.modelopt_native.verify_observations` on the input stream (same count / pattern
/ grid checks) plus a scale-finiteness check and a routing-count check. The
``transformer/modelopt_state.pt`` pickle is never opened. A GPU-free header probe is
available as ``python -m ...modelopt_native_fp8_w8a16 <model_dir>``.

Selected by the dispatcher only when ``VLLM_OMNI_FP8_BLOCKWISE_W8A16=1`` and
the checkpoint root recipe matches (see ``fp8_w8a16_selected`` in
``checkpoint_adapters/__init__.py``). A checkpoint whose sidecar declares a
quantized target family outside ``mlp.*``/``mlp_moe_gen.*``/``lm_head`` fails fast.
"""

import glob
import json
import os
import struct
import sys
from collections.abc import Generator, Iterable

import torch
from vllm.logger import init_logger

from vllm_omni.quantization.fp8_blockwise_w8a16 import W8A16_MARKER, is_target_prefix

from .modelopt_native import (
    MAX_PENDING_FP8,
    SCALE_SUFFIX,
    SIDECAR_FILENAME,
    WEIGHT_SUFFIX,
    BlockwiseQuantSpec,
    CheckpointIntegrityError,
    ModelOptNativeFp8CheckpointAdapter,
    TensorInfo,
    TensorKind,
    _read_safetensors_header,
    classify_name,
    dequantize_weight,
    header_tensor_infos,
    is_fp8_dtype,
    parse_quant_spec,
    scale_name_for,
    verify_observations,
    weight_name_for,
)

logger = init_logger(__name__)

_RESIDENT_SCALE_SUFFIX = ".weight_scale"  # param name allocated by the W8A16 method


def _module_of_weight(weight_name: str) -> str:
    """``X.weight`` -> ``X`` (module prefix), else the name unchanged (pure)."""
    if weight_name.endswith(WEIGHT_SUFFIX):
        return weight_name[: -len(WEIGHT_SUFFIX)]
    return weight_name


def assert_scale_finite(name: str, tensor: torch.Tensor) -> None:
    """Raise on any NaN/Inf scale byte (owner decision: abort, do not clamp)."""
    if not torch.isfinite(tensor.to(torch.float32)).all():
        raise CheckpointIntegrityError(
            f"scale tensor {name!r} contains NaN/Inf; the artifact is corrupt. Refusing to serve W8A16-resident."
        )


# The families W8A16 may ever quantize: the MLP projections stay FP8-resident and
# ``lm_head`` is dequant-to-BF16. Any other declared quantized family is refused fail-fast.
_ALLOWED_QUANT_FAMILIES = ("mlp", "mlp_moe_gen", "lm_head")


def assert_target_family(spec: BlockwiseQuantSpec) -> None:
    """Fail-fast: every declared quantized pattern is in the W8A16 target family.

    Defends a sidecar whose manifest *declares* a forbidden family (e.g. ``self_attn.*``),
    which the end-of-stream :func:`verify_observations` would otherwise accept as a
    'declared' pattern. Pure calculation over the parsed spec; raises before any load.
    """
    for pattern in spec.quantized_patterns:
        family = pattern.split(".", 1)[0]
        if family not in _ALLOWED_QUANT_FAMILIES:
            raise CheckpointIntegrityError(
                f"quantized pattern {pattern!r} is outside the W8A16 target family "
                f"{_ALLOWED_QUANT_FAMILIES}; refusing to serve W8A16-resident."
            )


class ModelOptNativeFp8W8A16CheckpointAdapter:
    """Streams the FP8-blockwise checkpoint W8A16-resident (MLP resident, lm_head BF16)."""

    def __init__(self, spec: BlockwiseQuantSpec, source_prefix: str, target_dtype: torch.dtype) -> None:
        assert_target_family(spec)  # fail-fast before any weight is routed resident
        self._spec = spec
        self._prefix = source_prefix
        self._target_dtype = target_dtype
        self._block = (spec.block_rows, spec.block_cols)  # declared 128x128

    @classmethod
    def detect(
        cls,
        source: object,
        target_dtype: torch.dtype = torch.bfloat16,
    ) -> "ModelOptNativeFp8W8A16CheckpointAdapter | None":
        """Engage iff the source carries the FP8-blockwise sidecar (same as dequant).

        Reuses the dequant adapter's sidecar parse so detection is identical; the
        dispatcher decides W8A16-vs-dequant via ``fp8_w8a16_selected`` (explicit
        opt-in; dequant is default). Construction asserts the declared target
        family, so an incorrectly declared sidecar fails fast here rather than mid-load.
        """
        spec = ModelOptNativeFp8CheckpointAdapter._parse_source_sidecar(source)
        if spec is None:
            return None
        model_dir = getattr(source, "model_or_path", None)
        logger.info(
            "ModelOpt-native FP8 W8A16-RESIDENT path engaged at %s (declared %d quantized modules, block %dx%d)",
            model_dir,
            spec.n_quantized,
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

    @staticmethod
    def _rename_scale(full_name: str) -> str:
        """``...<m>.weight_quantizer._scale`` -> ``...<m>.weight_scale`` (pure)."""
        return full_name[: -len(SCALE_SUFFIX)] + _RESIDENT_SCALE_SUFFIX

    def adapt(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Yield MLP targets FP8-resident, dequant lm_head/non-targets, verify at end."""
        scales: dict[str, torch.Tensor] = {}  # non-target scale buffer (lm_head)
        pending: dict[str, list[tuple[str, torch.Tensor]]] = {}
        n_pending = 0
        infos: list[TensorInfo] = []
        n_resident = 0
        n_dequant = 0

        for full_name, tensor in weights:
            name = self._strip_prefix(full_name)
            kind = classify_name(name)
            infos.append(TensorInfo(name, is_fp8_dtype(tensor.dtype), tuple(tensor.shape)))

            if kind is TensorKind.QUANTIZER_AMAX:
                continue  # informational twin of _scale; consumed

            if kind is TensorKind.QUANTIZER_SCALE:
                assert_scale_finite(name, tensor)
                module = _module_of_weight(weight_name_for(name) or name)
                if is_target_prefix(module):
                    # resident: rename to the W8A16 method's scale param, pass bf16 through
                    yield self._rename_scale(full_name), tensor
                else:
                    # non-target (lm_head): buffer for dequant + flush any pending weight
                    scales[name] = tensor
                    for pend_full, pend_w in pending.pop(name, ()):
                        n_pending -= 1
                        n_dequant += 1
                        yield pend_full, dequantize_weight(pend_w, tensor, self._target_dtype, self._block)
                continue

            if is_fp8_dtype(tensor.dtype):
                module = _module_of_weight(name)
                if is_target_prefix(module):
                    n_resident += 1
                    yield full_name, tensor  # FP8-resident, bytes unchanged
                    continue
                # non-target fp8 (lm_head): dequant to BF16 (pair with its scale)
                scale_name = scale_name_for(name)
                if scale_name is None:
                    continue  # flagged by verify_observations at end of stream
                if scale_name in scales:
                    n_dequant += 1
                    yield full_name, dequantize_weight(tensor, scales[scale_name], self._target_dtype, self._block)
                else:
                    pending.setdefault(scale_name, []).append((full_name, tensor))
                    n_pending += 1
                    if n_pending > MAX_PENDING_FP8:
                        waiting = ", ".join(sorted(pending))
                        raise CheckpointIntegrityError(
                            f"more than {MAX_PENDING_FP8} non-target fp8 weights pending "
                            f"without their scales; first missing: {waiting}"
                        )
                continue

            yield full_name, tensor  # BF16 passthrough (attention, norms, projections)

        violations = verify_observations(infos, self._spec)
        if violations:
            raise CheckpointIntegrityError(
                "FP8 W8A16 checkpoint failed integrity verification "
                f"({len(violations)} violation(s)):\n  " + "\n  ".join(violations)
            )
        if n_resident + n_dequant != self._spec.n_quantized:
            raise CheckpointIntegrityError(
                f"routing count mismatch: {n_resident} resident + {n_dequant} dequant "
                f"= {n_resident + n_dequant} != declared {self._spec.n_quantized} fp8 "
                "weights (some target/non-target was routed incorrectly or dropped)."
            )
        if n_pending:
            raise CheckpointIntegrityError(f"{n_pending} non-target fp8 weight(s) never received a scale.")
        logger.info(
            "FP8 W8A16 adapter: %d MLP targets FP8-resident, %d non-target(s) dequantized to %s (marker: %s)",
            n_resident,
            n_dequant,
            self._target_dtype,
            W8A16_MARKER,
        )


# --- CLI probe (header-only, GPU-free) --------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Integrity probe: exit 0 iff *model_dir* is the supported FP8 deliverable and
    reports the W8A16 resident/dequant split (216 resident MLP targets + lm_head)."""
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("usage: python -m ...modelopt_native_fp8_w8a16 <model_dir>", file=sys.stderr)
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

    violations = verify_observations(infos, spec)
    n_resident = sum(
        1 for i in infos if i.is_fp8 and i.name.endswith(WEIGHT_SUFFIX) and is_target_prefix(_module_of_weight(i.name))
    )
    n_dequant = sum(
        1
        for i in infos
        if i.is_fp8 and i.name.endswith(WEIGHT_SUFFIX) and not is_target_prefix(_module_of_weight(i.name))
    )
    print(
        f"declared: {spec.n_quantized} fp8 weights; W8A16 split: "
        f"{n_resident} MLP targets resident + {n_dequant} non-target(s) dequant "
        f"({len(infos)} tensors, {len(shard_paths)} shard(s))"
    )
    if violations:
        print(f"INTEGRITY FAIL ({len(violations)} violation(s)):")
        for violation in violations:
            print(f"  {violation}")
        return 1
    print(f"INTEGRITY OK: {model_dir} (marker: {W8A16_MARKER})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
