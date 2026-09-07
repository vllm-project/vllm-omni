# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Source-gated temporal chunk compatibility for the released MiniMax H3 VAE."""

from __future__ import annotations

import hashlib
import importlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
import torch.nn as nn

MINIMAX_H3_RELEASED_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
# Compatibility pins detect upstream code drift; they are not a trust or
# security boundary for remote code execution.
MINIMAX_H3_RELEASED_BUNDLE_SHA256 = "30e64afbaa940696bf8bfe2c86003a4b1666c84b22fa3aa12403a3e6a03f705d"

_RELEASED_TEMPORAL_CONFIG = {
    "clip_length": 17,
    "tokens_chunk_size": 5,
    "token_overlap": 2,
    "token_drop": 3,
    "vae_ratio_t": 4,
    "frame_pre_padding": 3,
    "frame_overlap": 5,
    "isolated_first_frame": False,
    "isolated_last_frame": False,
}


class MiniMaxH3TemporalChunkCompatibilityError(RuntimeError):
    """Raised when remote H3 code cannot use the local compatibility path."""


class _TemporalChunkCallbackSignature(Protocol):
    """Shared callable shape; each producer defines tensor semantics."""

    def __call__(
        self,
        frames: torch.Tensor,
        /,
        *,
        chunk_index: int,
        total_chunks: int,
        frame_start: int,
        is_final: bool,
    ) -> None: ...


@dataclass(slots=True)
class MiniMaxH3TemporalDecodePlan:
    latent: torch.Tensor
    num_chunks: int
    total_frames: int
    pad_frames: int
    output_frames: int
    chunk_frames: tuple[int, ...]
    temporal_dtype: torch.dtype | None


class _RawChunkAssembly:
    def __init__(
        self,
        *,
        callback: _TemporalChunkCallbackSignature,
        plan: MiniMaxH3TemporalDecodePlan,
    ) -> None:
        self.callback = callback
        self.plan = plan
        self.frames: torch.Tensor | None = None
        self.write_pos = 0
        self.logical_frames = 0
        self.dropped_frames = 0
        self.chunk_index = 0
        self.pending_final: tuple[torch.Tensor, int, int] | None = None
        self.contract_error: RuntimeError | None = None

    def write(self, part: torch.Tensor) -> None:
        part_frames = int(part.shape[2])
        if part_frames <= 0:
            return
        self.logical_frames += part_frames
        remaining = self.plan.output_frames - self.write_pos
        copy_frames = min(part_frames, max(0, remaining))
        if copy_frames <= 0:
            self.dropped_frames += part_frames
            return
        if self.chunk_index >= len(self.plan.chunk_frames):
            self.contract_error = RuntimeError("MiniMax-H3 compatibility decode emitted an extra chunk")
        else:
            expected_frames = self.plan.chunk_frames[self.chunk_index]
            if copy_frames != expected_frames and self.contract_error is None:
                self.contract_error = RuntimeError(
                    "MiniMax-H3 compatibility chunk size mismatch: "
                    f"chunk={self.chunk_index}, frames={copy_frames}/{expected_frames}"
                )

        emitted = part[:, :, :copy_frames]
        if self.frames is None:
            output_shape = list(emitted.shape)
            output_shape[2] = self.plan.output_frames
            self.frames = torch.empty(
                output_shape,
                dtype=emitted.dtype,
                device=emitted.device,
            )
        frame_start = self.write_pos
        frame_stop = frame_start + copy_frames
        self.frames[:, :, frame_start:frame_stop].copy_(emitted)
        if frame_stop == self.plan.output_frames:
            self.pending_final = (emitted, self.chunk_index, frame_start)
        elif self.contract_error is None:
            self.callback(
                emitted,
                chunk_index=self.chunk_index,
                total_chunks=len(self.plan.chunk_frames),
                frame_start=frame_start,
                is_final=False,
            )
        self.chunk_index += 1
        self.write_pos = frame_stop
        self.dropped_frames += part_frames - copy_frames

    def finish(self) -> torch.Tensor:
        if self.contract_error is not None:
            raise self.contract_error
        if self.frames is None or self.pending_final is None:
            raise RuntimeError("MiniMax-H3 compatibility decode produced no terminal chunk")
        if (
            self.logical_frames != self.plan.total_frames
            or self.dropped_frames != self.plan.pad_frames
            or self.write_pos != self.plan.output_frames
            or self.chunk_index != len(self.plan.chunk_frames)
        ):
            raise RuntimeError(
                "MiniMax-H3 compatibility frame plan mismatch: "
                f"logical={self.logical_frames}/{self.plan.total_frames}, "
                f"dropped={self.dropped_frames}/{self.plan.pad_frames}, "
                f"written={self.write_pos}/{self.plan.output_frames}, "
                f"chunks={self.chunk_index}/{len(self.plan.chunk_frames)}"
            )

        final_frames, final_index, final_start = self.pending_final
        self.callback(
            final_frames,
            chunk_index=final_index,
            total_chunks=len(self.plan.chunk_frames),
            frame_start=final_start,
            is_final=True,
        )
        return self.frames


def _source_bundle_sha256(model: nn.Module) -> tuple[str | None, str | None]:
    try:
        source = inspect.getsourcefile(type(model))
    except TypeError:
        return None, None
    if source is None:
        return None, None
    package_dir = Path(source).parent
    files = sorted(path for path in package_dir.glob("*.py") if path.name != "__init__.py")
    digest = hashlib.sha256()
    try:
        for path in files:
            digest.update(path.name.encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    except OSError:
        return str(package_dir), None
    return str(package_dir), digest.hexdigest()


def _validate_released_sources(model: nn.Module) -> None:
    source_dir, actual = _source_bundle_sha256(model)
    if actual != MINIMAX_H3_RELEASED_BUNDLE_SHA256:
        raise MiniMaxH3TemporalChunkCompatibilityError(
            "The vLLM-owned MiniMax-H3 chunk adapter supports only remote code "
            f"from MiniMaxAI/MiniMax-H3@{MINIMAX_H3_RELEASED_REVISION}: "
            f"bundle={actual!r} ({source_dir}). "
            "The normal complete-output decode remains available."
        )


def _validate_released_config(model: nn.Module) -> None:
    actual = {name: getattr(model, name, None) for name in _RELEASED_TEMPORAL_CONFIG}
    if actual != _RELEASED_TEMPORAL_CONFIG:
        raise MiniMaxH3TemporalChunkCompatibilityError(
            "The vLLM-owned MiniMax-H3 chunk adapter requires the released temporal "
            f"configuration {_RELEASED_TEMPORAL_CONFIG}, found {actual}. "
            "The normal complete-output decode remains available."
        )


def _validate_compat_contract(model: nn.Module, latent: torch.Tensor) -> None:
    if latent.ndim != 5 or int(latent.shape[2]) <= 0:
        raise MiniMaxH3TemporalChunkCompatibilityError(
            f"MiniMax-H3 video latent must have positive rank-5 time, got {tuple(latent.shape)}"
        )
    if not bool(getattr(model, "use_3d_conv", False)):
        raise MiniMaxH3TemporalChunkCompatibilityError("MiniMax-H3 temporal chunk decode requires the 3D decoder")
    if bool(getattr(model, "training", False)):
        raise MiniMaxH3TemporalChunkCompatibilityError("MiniMax-H3 temporal chunk decode is inference-only")
    required_methods = (
        "_adaptive_decode",
        "_decode_temporal_output_frame_plan",
        "_decode_temporal_pad_frames",
        "blend",
    )
    missing = [name for name in required_methods if not callable(getattr(model, name, None))]
    if missing:
        raise MiniMaxH3TemporalChunkCompatibilityError(
            "Loaded MiniMax-H3 VAE lacks required released-code methods: " + ", ".join(missing)
        )


def _truncate_chunk_frames(
    logical_chunk_frames: list[int],
    output_frames: int,
) -> tuple[int, ...]:
    remaining = output_frames
    result: list[int] = []
    for logical_frames in logical_chunk_frames:
        emitted_frames = min(logical_frames, max(0, remaining))
        if emitted_frames > 0:
            result.append(emitted_frames)
            remaining -= emitted_frames
    if remaining != 0:
        raise RuntimeError(f"MiniMax-H3 compatibility chunk plan cannot cover output: remaining={remaining}")
    return tuple(result)


def _build_decode_plan(
    model: nn.Module,
    latent: torch.Tensor,
    temporal_dtype: torch.dtype | None,
) -> MiniMaxH3TemporalDecodePlan:
    pseudo_total_tokens = int(latent.shape[2]) + int(model.token_drop)  # type: ignore[attr-defined]
    tokens_chunk_size = int(model.tokens_chunk_size)  # type: ignore[attr-defined]
    remainder = pseudo_total_tokens % tokens_chunk_size
    pad_tokens = 0 if remainder == 0 else tokens_chunk_size - remainder
    pseudo_total_tokens += pad_tokens
    num_chunks = pseudo_total_tokens // tokens_chunk_size - int(model.token_drop > 0)  # type: ignore[attr-defined]
    if num_chunks <= 0:
        raise MiniMaxH3TemporalChunkCompatibilityError(
            f"MiniMax-H3 temporal chunk plan is empty for latent T={latent.shape[2]}"
        )

    temporal_latent = latent
    if pad_tokens:
        pad = temporal_latent[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)
        temporal_latent = torch.cat((temporal_latent, pad), dim=2)
    total_frames, pad_frames, output_frames = model._decode_temporal_output_frame_plan(  # type: ignore[attr-defined]
        temporal_latent,
        None,
        None,
        num_chunks,
        pad_tokens,
    )
    if output_frames <= 0:
        raise MiniMaxH3TemporalChunkCompatibilityError(
            f"MiniMax-H3 temporal chunk plan has {output_frames} output frames"
        )

    chunk_decoded_frames = tokens_chunk_size * int(model.vae_ratio_t)  # type: ignore[attr-defined]
    split_count = int(model.token_drop > 0) + 1  # type: ignore[attr-defined]
    logical_chunk_frames: list[int] = []
    final_overlap_frames = 0
    for temporal_index in range(num_chunks):
        token_start = temporal_index * tokens_chunk_size
        token_stop = token_start + tokens_chunk_size + int(model.token_overlap)  # type: ignore[attr-defined]
        clip_tokens = max(
            0,
            min(token_stop, int(temporal_latent.shape[2])) - min(token_start, int(temporal_latent.shape[2])),
        )
        clip_frames = clip_tokens * int(model.vae_ratio_t)  # type: ignore[attr-defined]
        for split_index in range(split_count):
            frame_start = split_index * chunk_decoded_frames
            frame_stop = min(frame_start + chunk_decoded_frames, clip_frames)
            part_frames = max(
                0,
                frame_stop - frame_start - int(model.frame_pre_padding),  # type: ignore[attr-defined]
            )
            if split_index == 0:
                if part_frames > 0:
                    logical_chunk_frames.append(part_frames)
            else:
                final_overlap_frames = part_frames
    if final_overlap_frames > 0:
        logical_chunk_frames.append(final_overlap_frames)
    if sum(logical_chunk_frames) != int(total_frames):
        raise RuntimeError(
            "MiniMax-H3 compatibility planner disagrees with released remote code: "
            f"frames={sum(logical_chunk_frames)}/{total_frames}"
        )
    chunk_frames = _truncate_chunk_frames(logical_chunk_frames, int(output_frames))
    return MiniMaxH3TemporalDecodePlan(
        latent=temporal_latent,
        num_chunks=num_chunks,
        total_frames=int(total_frames),
        pad_frames=int(pad_frames),
        output_frames=int(output_frames),
        chunk_frames=chunk_frames,
        temporal_dtype=temporal_dtype,
    )


def resolve_minimax_h3_temporal_cat_dtype(model: nn.Module) -> torch.dtype | None:
    module = importlib.import_module(type(model).__module__)
    resolver = getattr(module, "_resolve_temporal_cat_dtype", None)
    if resolver is None:
        return None
    if not callable(resolver):
        raise MiniMaxH3TemporalChunkCompatibilityError(
            "Loaded MiniMax-H3 VAE has a non-callable temporal dtype resolver"
        )
    try:
        dtype = resolver()
    except ValueError as exc:
        raise MiniMaxH3TemporalChunkCompatibilityError(str(exc)) from None
    if dtype is not None and not isinstance(dtype, torch.dtype):
        raise MiniMaxH3TemporalChunkCompatibilityError(
            f"Loaded MiniMax-H3 VAE returned invalid temporal dtype {dtype!r}"
        )
    return dtype


def decode_minimax_h3_temporal_chunks_compat(
    model: nn.Module,
    plan: MiniMaxH3TemporalDecodePlan,
    callback: _TemporalChunkCallbackSignature,
) -> torch.Tensor:
    """Mirror the released Apache-2.0 H3 temporal assembler with callbacks.

    Keep this implementation and the source/config fingerprints above in sync
    with ``MiniMaxAI/MiniMax-H3@42ed227e``. Unknown remote code fails closed
    before this function is called.

    Callback tensors are borrowed decoder-domain views and must not be retained
    or mutated. The high-level adapter owns normalization and storage isolation.
    """

    chunk_decoded_frames = int(model.tokens_chunk_size) * int(model.vae_ratio_t)  # type: ignore[attr-defined]
    split_count = int(model.token_drop > 0) + 1  # type: ignore[attr-defined]
    assembly = _RawChunkAssembly(callback=callback, plan=plan)
    overlap: torch.Tensor | None = None

    for temporal_index in range(plan.num_chunks):
        token_start = temporal_index * int(model.tokens_chunk_size)  # type: ignore[attr-defined]
        token_stop = token_start + int(model.tokens_chunk_size) + int(model.token_overlap)  # type: ignore[attr-defined]
        chunk_latent = plan.latent[:, :, token_start:token_stop]
        decoded = model._adaptive_decode(chunk_latent)  # type: ignore[attr-defined]
        if plan.temporal_dtype is not None and decoded.dtype != plan.temporal_dtype:
            decoded = decoded.to(plan.temporal_dtype)
        if decoded.device != plan.latent.device:
            decoded = decoded.to(plan.latent.device)

        for split_index in range(split_count):
            frame_start = split_index * chunk_decoded_frames
            frame_stop = min(frame_start + chunk_decoded_frames, int(decoded.shape[2]))
            part = decoded[:, :, frame_start:frame_stop]
            part = part[:, :, int(model.frame_pre_padding) :]  # type: ignore[attr-defined]
            if split_index == 0:
                if overlap is not None:
                    part = model.blend(  # type: ignore[attr-defined]
                        overlap,
                        part,
                        int(model.frame_overlap),  # type: ignore[attr-defined]
                        dim=-3,
                    )
                    overlap = None
                assembly.write(part)
            else:
                overlap = part.contiguous()
            del part

        if temporal_index == plan.num_chunks - 1 and overlap is not None:
            assembly.write(overlap)
            overlap = None
        del decoded, chunk_latent

    return assembly.finish()


def prepare_minimax_h3_temporal_chunks_compat(
    model: nn.Module,
    latent: torch.Tensor,
    *,
    validate_sources: bool = True,
) -> MiniMaxH3TemporalDecodePlan:
    """Build a validated plan without entering decoder collectives."""

    if validate_sources:
        _validate_released_sources(model)
    _validate_released_config(model)
    _validate_compat_contract(model, latent)
    temporal_dtype = resolve_minimax_h3_temporal_cat_dtype(model)
    return _build_decode_plan(model, latent, temporal_dtype)


__all__ = [
    "MINIMAX_H3_RELEASED_REVISION",
    "MINIMAX_H3_RELEASED_BUNDLE_SHA256",
    "MiniMaxH3TemporalDecodePlan",
    "MiniMaxH3TemporalChunkCompatibilityError",
    "decode_minimax_h3_temporal_chunks_compat",
    "prepare_minimax_h3_temporal_chunks_compat",
    "resolve_minimax_h3_temporal_cat_dtype",
]
