# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Precomputed AdaLN plan cache for the MiniMax H3 DiT.

Version 2 is the SGLang-compatible base-H3 format. Version 3 is deliberately a
FastH3-only format: its outputs were computed after fusing one exact FastH3
adapter, so the sidecar metadata binds that adapter and its sampling contract.

The expensive adapter digest belongs in the runner/coordinator. Workers receive
the resulting :class:`MiniMaxH3AdalnCacheBinding`, bind it to the cache, and
validate only the small sidecar metadata. That avoids eight workers hashing the
same multi-GiB adapter while still failing closed before its AdaLN weights are
elided.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from safetensors.torch import safe_open

if TYPE_CHECKING:
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3DiTArchConfig,
    )

_BF16_DTYPE = torch.bfloat16
_FP32_DTYPE = torch.float32
_ADALN_MODALITY_NUM = 3
_SUPPORTED_FORMAT_VERSIONS = frozenset({"2", "3"})
_V3_FORMAT_VERSION = "3"
_SHA256_HEX_LENGTH = 64
_FASTH3_FORMAT = "fastvideo-lora-v2"
_FASTH3_BASE_MODEL = "MiniMaxAI/MiniMax-H3"
_FASTH3_ADAPTER_KEYS = {
    "sha256": "fasth3_adapter_sha256",
    "size_bytes": "fasth3_adapter_size_bytes",
    "header_sha256": "fasth3_adapter_header_sha256",
    "header_size_bytes": "fasth3_adapter_header_size_bytes",
    "header_identity": "fasth3_adapter_header_identity",
}
_V3_CONTRACT_KEYS = {
    "base_schedule": "base_schedule",
    "mode": "mode",
    "flow_shift": "flow_shift",
    "audio_flow_shift": "audio_flow_shift",
}


def _validate_sha256(value: str, *, field: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != _SHA256_HEX_LENGTH:
        raise ValueError(f"{field} must be a 64-character SHA256 digest")
    try:
        int(normalized, 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be a hexadecimal SHA256 digest") from exc
    return normalized


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _parse_header_identity(value: str) -> dict[str, str]:
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("fasth3_adapter_header_identity must be canonical JSON") from exc
    if not isinstance(parsed, dict) or any(
        not isinstance(key, str) or not isinstance(item, str) for key, item in parsed.items()
    ):
        raise ValueError("fasth3_adapter_header_identity must encode string metadata")
    canonical = _canonical_json(parsed)
    if canonical != value:
        raise ValueError("fasth3_adapter_header_identity must use canonical JSON encoding")
    return parsed


def _validate_fasth3_header_identity(value: str) -> str:
    metadata = _parse_header_identity(value)
    if metadata.get("format") != _FASTH3_FORMAT:
        raise ValueError(f"FastH3 adapter header identity must declare format={_FASTH3_FORMAT!r}")
    finetuned_model = metadata.get("finetuned_model", "").casefold()
    if not finetuned_model.startswith("fastvideo/") or "fasth3" not in finetuned_model:
        raise ValueError("FastH3 adapter header identity must name a FastVideo FastH3 release")
    base_model = metadata.get("base_model")
    if base_model is not None and base_model.casefold() != _FASTH3_BASE_MODEL.casefold():
        raise ValueError("FastH3 adapter header identity targets an unexpected base model")
    return value


@dataclass(frozen=True)
class MiniMaxH3FastH3AdapterIdentity:
    """Content and header identity of one FastH3 adapter artifact."""

    sha256: str
    size_bytes: int
    header_sha256: str
    header_size_bytes: int
    header_identity: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sha256",
            _validate_sha256(self.sha256, field="fasth3_adapter_sha256"),
        )
        object.__setattr__(
            self,
            "header_sha256",
            _validate_sha256(
                self.header_sha256,
                field="fasth3_adapter_header_sha256",
            ),
        )
        if int(self.size_bytes) <= 8:
            raise ValueError("fasth3_adapter_size_bytes must exceed the header prefix")
        if int(self.header_size_bytes) <= 0:
            raise ValueError("fasth3_adapter_header_size_bytes must be positive")
        if int(self.header_size_bytes) + 8 > int(self.size_bytes):
            raise ValueError("FastH3 adapter header extends beyond the artifact")
        object.__setattr__(self, "size_bytes", int(self.size_bytes))
        object.__setattr__(self, "header_size_bytes", int(self.header_size_bytes))
        object.__setattr__(
            self,
            "header_identity",
            _validate_fasth3_header_identity(self.header_identity),
        )

    def to_dict(self) -> dict[str, str | int]:
        return {
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "header_sha256": self.header_sha256,
            "header_size_bytes": self.header_size_bytes,
            "header_identity": self.header_identity,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> MiniMaxH3FastH3AdapterIdentity:
        expected = set(_FASTH3_ADAPTER_KEYS)
        unknown = set(value) - expected
        missing = expected - set(value)
        if missing or unknown:
            raise ValueError(
                f"invalid FastH3 adapter identity fields: missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        return cls(
            sha256=str(value["sha256"]),
            size_bytes=int(value["size_bytes"]),
            header_sha256=str(value["header_sha256"]),
            header_size_bytes=int(value["header_size_bytes"]),
            header_identity=str(value["header_identity"]),
        )


@dataclass(frozen=True)
class MiniMaxH3AdalnCacheBinding:
    """Serializable identity and sampling contract for one AdaLN sidecar."""

    format_version: str
    model_variant: str | None
    mode: str | None = None
    base_schedule: tuple[float, ...] | None = None
    flow_shift: float | None = None
    audio_flow_shift: float | None = None
    fasth3_adapter: MiniMaxH3FastH3AdapterIdentity | None = None

    def __post_init__(self) -> None:
        version = str(self.format_version)
        object.__setattr__(self, "format_version", version)
        if version not in _SUPPORTED_FORMAT_VERSIONS:
            raise ValueError(f"unsupported MiniMax H3 AdaLN cache format_version {version!r}")
        if self.model_variant is not None and self.model_variant not in {
            "fl2va",
            "ref2va",
        }:
            raise ValueError(f"invalid MiniMax H3 AdaLN cache model_variant {self.model_variant!r}")

        v3_values = (
            self.mode,
            self.base_schedule,
            self.flow_shift,
            self.audio_flow_shift,
            self.fasth3_adapter,
        )
        if version == "2":
            if any(value is not None for value in v3_values):
                raise ValueError("AdaLN cache format v2 cannot carry a FastH3 binding")
            return

        if any(value is None for value in v3_values):
            raise ValueError("AdaLN cache format v3 requires a complete FastH3 binding")
        if self.model_variant != "fl2va":
            raise ValueError("FastH3 AdaLN cache v3 requires model_variant='fl2va'")
        if self.mode != "t2va":
            raise ValueError("FastH3 AdaLN cache v3 requires mode='t2va'")

        assert self.base_schedule is not None
        from vllm_omni.diffusion.sched.sigma_schedule import DMD2SigmaSchedule

        schedule = DMD2SigmaSchedule.from_positions(self.base_schedule)
        object.__setattr__(self, "base_schedule", schedule.base_schedule)
        for field, value in (
            ("flow_shift", self.flow_shift),
            ("audio_flow_shift", self.audio_flow_shift),
        ):
            assert value is not None
            normalized = float(value)
            if not math.isfinite(normalized) or normalized <= 0:
                raise ValueError(f"{field} must be finite and positive")
            object.__setattr__(self, field, normalized)

    @classmethod
    def for_fasth3(
        cls,
        *,
        adapter: MiniMaxH3FastH3AdapterIdentity,
        model_variant: str,
        mode: str,
        base_schedule: Sequence[float],
        flow_shift: float,
        audio_flow_shift: float,
    ) -> MiniMaxH3AdalnCacheBinding:
        return cls(
            format_version=_V3_FORMAT_VERSION,
            model_variant=model_variant,
            mode=mode,
            base_schedule=tuple(float(value) for value in base_schedule),
            flow_shift=float(flow_shift),
            audio_flow_shift=float(audio_flow_shift),
            fasth3_adapter=adapter,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "model_variant": self.model_variant,
            "mode": self.mode,
            "base_schedule": (list(self.base_schedule) if self.base_schedule is not None else None),
            "flow_shift": self.flow_shift,
            "audio_flow_shift": self.audio_flow_shift,
            "fasth3_adapter": (self.fasth3_adapter.to_dict() if self.fasth3_adapter is not None else None),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MiniMaxH3AdalnCacheBinding:
        expected = {
            "format_version",
            "model_variant",
            "mode",
            "base_schedule",
            "flow_shift",
            "audio_flow_shift",
            "fasth3_adapter",
        }
        unknown = set(value) - expected
        missing = expected - set(value)
        if missing or unknown:
            raise ValueError(
                f"invalid MiniMax H3 AdaLN binding fields: missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        raw_adapter = value["fasth3_adapter"]
        adapter = MiniMaxH3FastH3AdapterIdentity.from_dict(raw_adapter) if isinstance(raw_adapter, Mapping) else None
        raw_schedule = value["base_schedule"]
        return cls(
            format_version=str(value["format_version"]),
            model_variant=(str(value["model_variant"]) if value["model_variant"] is not None else None),
            mode=str(value["mode"]) if value["mode"] is not None else None,
            base_schedule=(tuple(float(item) for item in raw_schedule) if raw_schedule is not None else None),
            flow_shift=(float(value["flow_shift"]) if value["flow_shift"] is not None else None),
            audio_flow_shift=(float(value["audio_flow_shift"]) if value["audio_flow_shift"] is not None else None),
            fasth3_adapter=adapter,
        )

    def to_metadata(self) -> dict[str, str]:
        metadata = {
            "format_version": self.format_version,
            "model_variant": self.model_variant or "",
        }
        if self.format_version == "2":
            return metadata
        assert self.base_schedule is not None
        assert self.mode is not None
        assert self.flow_shift is not None
        assert self.audio_flow_shift is not None
        assert self.fasth3_adapter is not None
        metadata.update(
            {
                _V3_CONTRACT_KEYS["mode"]: self.mode,
                _V3_CONTRACT_KEYS["base_schedule"]: _canonical_json(list(self.base_schedule)),
                _V3_CONTRACT_KEYS["flow_shift"]: repr(self.flow_shift),
                _V3_CONTRACT_KEYS["audio_flow_shift"]: repr(self.audio_flow_shift),
                _FASTH3_ADAPTER_KEYS["sha256"]: self.fasth3_adapter.sha256,
                _FASTH3_ADAPTER_KEYS["size_bytes"]: str(self.fasth3_adapter.size_bytes),
                _FASTH3_ADAPTER_KEYS["header_sha256"]: self.fasth3_adapter.header_sha256,
                _FASTH3_ADAPTER_KEYS["header_size_bytes"]: str(self.fasth3_adapter.header_size_bytes),
                _FASTH3_ADAPTER_KEYS["header_identity"]: self.fasth3_adapter.header_identity,
            }
        )
        return metadata

    @classmethod
    def from_metadata(
        cls,
        metadata: Mapping[str, str],
    ) -> MiniMaxH3AdalnCacheBinding:
        version = metadata.get("format_version", "")
        model_variant = metadata.get("model_variant")
        if version == "2":
            return cls(format_version=version, model_variant=model_variant)
        if version != _V3_FORMAT_VERSION:
            raise ValueError("MiniMax H3 AdaLN cache has an unsupported or missing format_version")
        required = set(_V3_CONTRACT_KEYS.values()) | set(_FASTH3_ADAPTER_KEYS.values())
        missing = sorted(key for key in required if key not in metadata)
        if missing:
            raise ValueError(f"MiniMax H3 AdaLN cache v3 metadata is incomplete: {missing}")
        try:
            raw_schedule = json.loads(metadata[_V3_CONTRACT_KEYS["base_schedule"]])
            if not isinstance(raw_schedule, list):
                raise TypeError("base_schedule must be a JSON list")
            adapter = MiniMaxH3FastH3AdapterIdentity(
                sha256=metadata[_FASTH3_ADAPTER_KEYS["sha256"]],
                size_bytes=int(metadata[_FASTH3_ADAPTER_KEYS["size_bytes"]]),
                header_sha256=metadata[_FASTH3_ADAPTER_KEYS["header_sha256"]],
                header_size_bytes=int(metadata[_FASTH3_ADAPTER_KEYS["header_size_bytes"]]),
                header_identity=metadata[_FASTH3_ADAPTER_KEYS["header_identity"]],
            )
            return cls.for_fasth3(
                adapter=adapter,
                model_variant=str(model_variant),
                mode=metadata[_V3_CONTRACT_KEYS["mode"]],
                base_schedule=raw_schedule,
                flow_shift=float(metadata[_V3_CONTRACT_KEYS["flow_shift"]]),
                audio_flow_shift=float(metadata[_V3_CONTRACT_KEYS["audio_flow_shift"]]),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"MiniMax H3 AdaLN cache v3 metadata is malformed: {exc}") from exc


def _resolve_fasth3_adapter_file(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_file():
        if candidate.suffix != ".safetensors":
            raise ValueError(f"FastH3 adapter is not a safetensors file: {candidate}")
        return candidate.resolve()
    if not candidate.is_dir():
        raise ValueError(f"FastH3 adapter does not exist: {candidate}")
    named = candidate / "adapter_model.safetensors"
    if named.is_file():
        return named.resolve()
    files = sorted(candidate.glob("*.safetensors"))
    if len(files) != 1:
        raise ValueError(f"FastH3 adapter directory must contain one safetensors file: {candidate}")
    return files[0].resolve()


def fingerprint_minimax_h3_fasth3_adapter(
    path: str | os.PathLike[str],
) -> MiniMaxH3FastH3AdapterIdentity:
    """Hash one FastH3 adapter once, for runner-side prevalidation.

    This intentionally reads the full payload. Do not call it independently in
    every diffusion worker; serialize the returned identity instead.
    """
    adapter_path = _resolve_fasth3_adapter_file(path)
    digest = hashlib.sha256()
    with adapter_path.open("rb") as adapter_file:
        before = os.fstat(adapter_file.fileno())
        prefix = adapter_file.read(8)
        if len(prefix) != 8:
            raise ValueError(f"FastH3 adapter has a truncated header: {adapter_path}")
        header_size = int.from_bytes(prefix, byteorder="little", signed=False)
        if header_size <= 0 or header_size + 8 > before.st_size:
            raise ValueError(f"FastH3 adapter has an invalid header size: {adapter_path}")
        header = adapter_file.read(header_size)
        if len(header) != header_size:
            raise ValueError(f"FastH3 adapter has a truncated header: {adapter_path}")
        try:
            parsed_header = json.loads(header)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"FastH3 adapter has malformed safetensors JSON: {adapter_path}") from exc
        if not isinstance(parsed_header, dict):
            raise ValueError(f"FastH3 adapter safetensors header is not an object: {adapter_path}")
        metadata = parsed_header.get("__metadata__") or {}
        if not isinstance(metadata, dict) or any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in metadata.items()
        ):
            raise ValueError(f"FastH3 adapter has invalid safetensors metadata: {adapter_path}")
        header_identity = _canonical_json(metadata)
        _validate_fasth3_header_identity(header_identity)

        adapter_file.seek(0)
        while chunk := adapter_file.read(16 * 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(adapter_file.fileno())
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after:
        raise ValueError(f"FastH3 adapter changed while it was hashed: {adapter_path}")
    return MiniMaxH3FastH3AdapterIdentity(
        sha256=digest.hexdigest(),
        size_bytes=before.st_size,
        header_sha256=hashlib.sha256(header).hexdigest(),
        header_size_bytes=header_size,
        header_identity=header_identity,
    )


def read_minimax_h3_adaln_cache_binding(
    path: str | os.PathLike[str],
) -> MiniMaxH3AdalnCacheBinding:
    """Read and validate only the sidecar's metadata binding."""
    cache_path = os.path.abspath(os.path.expanduser(os.fspath(path)))
    if not os.path.isfile(cache_path):
        raise ValueError(f"MiniMax H3 AdaLN cache does not exist: {cache_path}")
    with safe_open(cache_path, framework="pt", device="cpu") as cache_file:
        return MiniMaxH3AdalnCacheBinding.from_metadata(cache_file.metadata() or {})


def _coerce_binding(
    binding: MiniMaxH3AdalnCacheBinding | Mapping[str, Any],
) -> MiniMaxH3AdalnCacheBinding:
    if isinstance(binding, MiniMaxH3AdalnCacheBinding):
        return binding
    if isinstance(binding, Mapping):
        return MiniMaxH3AdalnCacheBinding.from_dict(binding)
    raise TypeError("MiniMax H3 AdaLN runtime binding must be a binding or mapping")


def validate_minimax_h3_adaln_sidecar_binding(
    path: str | os.PathLike[str],
    binding: MiniMaxH3AdalnCacheBinding | Mapping[str, Any],
) -> MiniMaxH3AdalnCacheBinding:
    """Require ``path`` metadata to equal a runner-prevalidated binding."""
    expected = _coerce_binding(binding)
    actual = read_minimax_h3_adaln_cache_binding(path)
    if actual != expected:
        raise ValueError(
            f"MiniMax H3 AdaLN cache binding mismatch: expected={expected.to_dict()!r}, actual={actual.to_dict()!r}"
        )
    return actual


def minimax_h3_adaln_sidecar_satisfies(
    path: str | os.PathLike[str],
    binding: MiniMaxH3AdalnCacheBinding | Mapping[str, Any],
) -> bool:
    """Cheap metadata-only predicate for a pipeline/core sidecar selection."""
    try:
        validate_minimax_h3_adaln_sidecar_binding(path, binding)
    except (OSError, TypeError, ValueError):
        return False
    return True


def prevalidate_minimax_h3_fasth3_adaln_sidecar(
    sidecar_path: str | os.PathLike[str],
    *,
    adapter_path: str | os.PathLike[str],
    model_variant: str,
    mode: str,
    base_schedule: Sequence[float],
    flow_shift: float,
    audio_flow_shift: float,
) -> MiniMaxH3AdalnCacheBinding:
    """Runner-only full adapter verification followed by sidecar validation."""
    adapter = fingerprint_minimax_h3_fasth3_adapter(adapter_path)
    expected = MiniMaxH3AdalnCacheBinding.for_fasth3(
        adapter=adapter,
        model_variant=model_variant,
        mode=mode,
        base_schedule=base_schedule,
        flow_shift=flow_shift,
        audio_flow_shift=audio_flow_shift,
    )
    return validate_minimax_h3_adaln_sidecar_binding(sidecar_path, expected)


class MiniMaxH3AdalnCache(nn.Module):
    """Fixed-schedule AdaLN output sidecar.

    Base-H3 v2 files keep their original behavior. A FastH3 v3 file must be
    bound before :meth:`load`; otherwise the runtime cannot know that the
    fused student, schedule and shifts match the precomputed outputs.
    """

    plan_timesteps: torch.Tensor
    plan_lengths: torch.Tensor
    block_params: torch.Tensor
    final_params: torch.Tensor

    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        *,
        path: str,
        model_variant: str | None = None,
        runtime_binding: MiniMaxH3AdalnCacheBinding | Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if not path:
            raise ValueError("MiniMax H3 AdaLN cache path must not be empty")
        self.path = os.path.abspath(os.path.expanduser(path))
        self.model_variant = model_variant
        self.num_layers = arch.num_layers
        self.hidden_size = arch.hidden_size
        self.block_width = 6 * _ADALN_MODALITY_NUM * arch.hidden_size
        self.final_width = 2 * arch.hidden_size
        self._loaded = False
        self._runtime_binding: MiniMaxH3AdalnCacheBinding | None = None
        self._sidecar_binding: MiniMaxH3AdalnCacheBinding | None = None
        self._binding_validated = False
        if runtime_binding is not None:
            self.bind_runtime_contract(runtime_binding)

    def bind_runtime_contract(
        self,
        binding: MiniMaxH3AdalnCacheBinding | Mapping[str, Any],
    ) -> None:
        """Bind runner-prevalidated identity without rehashing the adapter.

        The runner calls :func:`prevalidate_minimax_h3_fasth3_adaln_sidecar`
        once, serializes ``binding.to_dict()``, and gives the mapping to every
        worker. This method reads only the sidecar header.
        """
        if self._loaded:
            raise RuntimeError("MiniMax H3 AdaLN runtime contract must be bound before load")
        expected = _coerce_binding(binding)
        if self.model_variant is not None and expected.model_variant != self.model_variant:
            raise ValueError(
                "MiniMax H3 AdaLN runtime binding model_variant does not match "
                f"the loaded variant ({expected.model_variant!r} != "
                f"{self.model_variant!r})"
            )
        actual = validate_minimax_h3_adaln_sidecar_binding(self.path, expected)
        if self._runtime_binding is not None and self._runtime_binding != expected:
            raise ValueError("MiniMax H3 AdaLN cache is already bound to another runtime contract")
        self._runtime_binding = expected
        self._sidecar_binding = actual
        self._binding_validated = True

    def sidecar_satisfied_parameter_names(self) -> frozenset[str]:
        """Return FastH3 patches replaced by a verified v3 sidecar.

        These are the parameters a load-time FastH3 fusion may mark satisfied
        instead of reconstructing. ``time_embedder`` is intentionally absent:
        the current transformer still executes it in ``_embed`` and therefore
        still needs its four FastH3 dense patches loaded.
        """
        binding = self._sidecar_binding
        if not self._binding_validated or binding is None or binding.format_version != _V3_FORMAT_VERSION:
            raise RuntimeError("FastH3 AdaLN sidecar parameters require a validated v3 binding")
        names = {
            f"blocks.{index}.adaln_proj.linear.{suffix}"
            for index in range(self.num_layers)
            for suffix in ("weight", "bias")
        }
        names.update(
            {
                "final_layer.adaln_proj.linear.weight",
                "final_layer.adaln_proj.linear.bias",
            }
        )
        return frozenset(names)

    def load(self, device: torch.device) -> None:
        """Validate the sidecar on CPU, then install its tensors on ``device``."""
        if self._loaded:
            if self.block_params.device != device:
                raise ValueError(
                    "MiniMax H3 AdaLN cache is already loaded on "
                    f"{self.block_params.device}, cannot reload it on {device}"
                )
            return
        if not os.path.isfile(self.path):
            raise ValueError(f"MiniMax H3 AdaLN cache does not exist: {self.path}")

        with safe_open(self.path, framework="pt", device="cpu") as cache_file:
            metadata = cache_file.metadata() or {}
            sidecar_binding = MiniMaxH3AdalnCacheBinding.from_metadata(metadata)
            if self.model_variant is not None and sidecar_binding.model_variant != self.model_variant:
                raise ValueError(
                    "MiniMax H3 AdaLN cache model_variant does not match the "
                    "loaded variant "
                    f"({sidecar_binding.model_variant!r} != {self.model_variant!r})"
                )
            if sidecar_binding.format_version == _V3_FORMAT_VERSION:
                if self._runtime_binding is None or not self._binding_validated:
                    raise ValueError("MiniMax H3 AdaLN cache v3 requires bind_runtime_contract() before load")
                if sidecar_binding != self._runtime_binding:
                    raise ValueError("MiniMax H3 AdaLN cache binding changed after runtime prevalidation")
            elif self._runtime_binding is not None:
                if sidecar_binding != self._runtime_binding:
                    raise ValueError("MiniMax H3 AdaLN cache binding does not match the runtime contract")
            plan_timesteps = cache_file.get_tensor("plan_timesteps")
            plan_lengths = cache_file.get_tensor("plan_lengths")
            block_params = cache_file.get_tensor("block_params")
            final_params = cache_file.get_tensor("final_params")

        num_plans = plan_timesteps.shape[0] if plan_timesteps.ndim == 2 else -1
        max_width = plan_timesteps.shape[1] if plan_timesteps.ndim == 2 else -1
        if (
            plan_timesteps.dtype != _FP32_DTYPE
            or plan_timesteps.ndim != 2
            or plan_lengths.dtype != torch.int64
            or plan_lengths.shape != (num_plans,)
            or bool((plan_lengths < 1).any())
            or bool((plan_lengths > max_width).any())
        ):
            raise ValueError("MiniMax H3 AdaLN cache has invalid timestep plans")
        if block_params.dtype != _BF16_DTYPE or block_params.shape != (
            num_plans,
            max_width,
            self.num_layers,
            self.block_width,
        ):
            raise ValueError("MiniMax H3 AdaLN cache has invalid block_params")
        if final_params.dtype != _BF16_DTYPE or final_params.shape != (
            num_plans,
            max_width,
            self.final_width,
        ):
            raise ValueError("MiniMax H3 AdaLN cache has invalid final_params")

        # These tensors are derived data and must not become checkpoint state.
        self.register_buffer("plan_timesteps", plan_timesteps.to(device), persistent=False)
        self.register_buffer("plan_lengths", plan_lengths.to(device), persistent=False)
        self.register_buffer("block_params", block_params.to(device), persistent=False)
        self.register_buffer("final_params", final_params.to(device), persistent=False)
        self._sidecar_binding = sidecar_binding
        self._loaded = True

    def lookup(self, unique_timesteps: torch.Tensor) -> torch.Tensor:
        """Return the exact matching plan slot or fail closed."""
        if not self._loaded:
            raise RuntimeError("MiniMax H3 AdaLN cache has not been loaded")
        unique_timesteps = unique_timesteps.to(
            device=self.plan_timesteps.device,
            dtype=_FP32_DTYPE,
        ).view(-1)
        num_timesteps = unique_timesteps.shape[0]
        if num_timesteps > self.plan_timesteps.shape[1]:
            raise ValueError("MiniMax H3 AdaLN cache does not cover the request timestep plan")
        matches = self.plan_lengths.eq(num_timesteps) & self.plan_timesteps[:, :num_timesteps].eq(unique_timesteps).all(
            dim=-1
        )
        if not bool(matches.any()):
            raise ValueError("MiniMax H3 AdaLN cache does not cover the request timestep plan")
        return matches.to(torch.int64).argmax()

    def block_all(
        self,
        *,
        cache_plan_index: torch.Tensor,
        num_timesteps: int,
    ) -> tuple[tuple[torch.Tensor, ...], ...]:
        """Return all blocks' six AdaLN tensors with one layer-major gather."""
        stacked = self.block_params.permute(2, 0, 1, 3)[:, cache_plan_index, :num_timesteps]
        stacked = stacked.reshape(self.num_layers, -1, 6, self.hidden_size)
        return tuple(tuple(layer.unbind(dim=1)) for layer in stacked)

    def final(
        self,
        cache_plan_index: torch.Tensor,
        num_timesteps: int,
    ) -> tuple[torch.Tensor, ...]:
        params = self.final_params[cache_plan_index, :num_timesteps]
        return tuple(params.reshape(-1, 2, self.hidden_size).unbind(dim=1))


__all__ = [
    "MiniMaxH3AdalnCache",
    "MiniMaxH3AdalnCacheBinding",
    "MiniMaxH3FastH3AdapterIdentity",
    "fingerprint_minimax_h3_fasth3_adapter",
    "minimax_h3_adaln_sidecar_satisfies",
    "prevalidate_minimax_h3_fasth3_adaln_sidecar",
    "read_minimax_h3_adaln_cache_binding",
    "validate_minimax_h3_adaln_sidecar_binding",
]
