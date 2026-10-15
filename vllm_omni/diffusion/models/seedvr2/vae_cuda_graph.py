# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exact-profile causal decoder graphs with fixed request history addresses."""

import json
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from vllm.logger import init_logger

from vllm_omni.platforms import current_omni_platform

from .nadit_cuda_graph import _versions

logger = init_logger(__name__)
CONFIG_KEY = "seedvr2_vae_decoder_cuda_graph"


@dataclass(frozen=True)
class VAEDecoderGraphConfig:
    profiles: tuple[tuple[int, int, int], ...]
    max_pool_bytes: int = 4 * 1024**3
    eager_headroom_bytes: int = 8 * 1024**3
    fuse_normalization: bool = False

    @classmethod
    def from_extras(cls, extras: dict[str, Any], *, enforce_eager: bool):
        value = extras.get(CONFIG_KEY)
        if enforce_eager or value is None:
            return None
        if not isinstance(value, dict) or set(value) - {
            "profiles",
            "max_pool_bytes",
            "eager_headroom_bytes",
            "fuse_normalization",
        }:
            raise ValueError(f"{CONFIG_KEY} requires profiles and optional byte budgets")
        shapes = value.get("profiles")
        if not isinstance(shapes, list) or not 1 <= len(shapes) <= 4:
            raise ValueError("SeedVR2 VAE decoder graphs require 1-4 exact latent [T,H,W] profiles")
        parsed = []
        for shape in shapes:
            if not isinstance(shape, (list, tuple)) or len(shape) != 3:
                raise ValueError("VAE graph profile must be a latent [T,H,W] triple")
            if any(type(x) is not int or x <= 0 for x in shape) or shape[0] < 2 or shape[1] % 2 or shape[2] % 2:
                raise ValueError("VAE graph profiles require even positive integer H/W and at least two latent frames")
            parsed.append(tuple(shape))
        if len(set(parsed)) != len(parsed):
            raise ValueError("duplicate VAE decoder graph profile")
        budgets = {key: value.get(key, getattr(cls, key)) for key in ["max_pool_bytes", "eager_headroom_bytes"]}
        if any(type(x) is not int or x <= 0 for x in budgets.values()):
            raise ValueError("VAE decoder graph budgets must be positive integer byte counts")
        fusion = value.get("fuse_normalization", False)
        if type(fusion) is not bool:
            raise ValueError("VAE normalization fusion must be a boolean")
        return cls(tuple(sorted(parsed)), **budgets, fuse_normalization=fusion)


def temporal_phases(frames: int):
    """The unchanged decoder chunk size is two latent frames."""
    return tuple(
        "first" if start == 0 else "tail" if frames - start == 1 else "continuation" for start in range(0, frames, 2)
    )


def _identity(model):
    return (
        model.use_tiling,
        id(model._spatial_group),
        type(model.decoder),
        model.fused_spatial_normalization,
        tuple(
            (
                name,
                type(module),
                getattr(module, "eps", None),
                getattr(module, "kernel_size", None),
                getattr(module, "stride", None),
                getattr(module, "padding", None),
                getattr(module, "temporal_padding", None),
                getattr(module, "temporal_ratio", None),
                type(getattr(module, "processor", None)),
            )
            for name, module in model.decoder.named_modules()
        ),
    )


class VAEDecoderGraphs:
    """One mutable history slot per geometry, leased for the complete decode."""

    def __init__(self, model, config: VAEDecoderGraphConfig):
        self.model, self.config = model, config
        self.group = model._spatial_group
        if self.group is None or not model.use_tiling:
            raise ValueError("SeedVR2 VAE decoder graphs require spatial height sharding")
        self.world, self.rank = dist.get_world_size(self.group), dist.get_rank(self.group)
        if self.world not in (2, 4) or dist.get_world_size() != self.world:
            raise ValueError("SeedVR2 VAE decoder graphs require one SP2/SP4 group per service")
        self.cpu_group = dist.new_group(backend="gloo")
        self.entries, self.manifest = {}, []
        self.pending_graphs = []
        self.pool_bytes = 0
        model.fused_spatial_normalization = config.fuse_normalization
        self.identity = _identity(model)
        self.closed = self.retired = self.failed = self._leased = False
        self.last_event = None
        self.stats = dict(replays=0, eager_misses=0, retired=0, first=0, continuation=0, tail=0)
        model.decoder_graphs = self
        signatures = self._gather(
            (config.profiles, config.max_pool_bytes, config.eager_headroom_bytes, config.fuse_normalization)
        )
        if any(value != signatures[0] for value in signatures):
            raise RuntimeError("SeedVR2 ranks disagree on VAE decoder graph profiles")
        geometries = sorted({shape[1:] for shape in config.profiles})
        with torch.inference_mode():
            for height, width in geometries:
                frames = tuple(shape[0] for shape in config.profiles if shape[1:] == (height, width))
                self._prepare(height, width, frames)
        self.versions = _versions(model.decoder)
        logger.info("SeedVR2 VAE decoder CUDA graph manifest: %s", json.dumps(self.manifest))

    def _gather(self, value):
        rows = [None] * self.world
        dist.all_gather_object(rows, value, group=self.cpu_group)
        return rows

    def _drain(self):
        torch.accelerator.synchronize()
        dist.barrier(group=self.cpu_group)

    def _prepare(self, height, width, frame_counts):
        if height < self.world:
            raise ValueError("VAE graph latent height must cover every spatial rank")
        free, _ = current_omni_platform.get_device_memory()
        if not all(self._gather(free > self.config.eager_headroom_bytes + self.config.max_pool_bytes)):
            self.manifest.append(
                dict(
                    geometry=[height, width],
                    latent_frames=frame_counts,
                    ready=False,
                    reason="headroom_before_capture",
                    remaining_free_bytes=free,
                    pool_budget_bytes=self.config.max_pool_bytes,
                    eager_headroom_bytes=self.config.eager_headroom_bytes,
                )
            )
            return
        model = self.model
        dtype, device = next(model.decoder.parameters()).dtype, next(model.decoder.parameters()).device
        context = model._context(height)
        assert context.spatial is not None
        context.cache_temporal = any(frames > 2 for frames in frame_counts)
        spatial = context.spatial
        local_height = spatial.units[self.rank]
        required = set(phase for frames in frame_counts for phase in temporal_phases(frames))
        phases = [phase for phase in ("first", "continuation", "tail") if phase in required]
        inputs = {
            phase: torch.zeros((1, 16, 1 if phase == "tail" else 2, local_height, width), device=device, dtype=dtype)
            for phase in phases
        }
        # A single first chunk needs no history, matching the eager decoder.
        # Other profiles establish every causal history outside capture.
        model.decoder(inputs["first"], context)
        context.fixed_history = context.cache_temporal
        history_addresses = tuple(tensor.data_ptr() for tensor in context.history.values())
        stream = torch.cuda.Stream()
        pool = torch.cuda.graph_pool_handle()
        captured = {}
        self._drain()
        for phase in phases:
            context.first = phase == "first"
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    model.decoder(inputs[phase], context)
            torch.cuda.current_stream().wait_stream(stream)
            self._drain()
            graph = torch.cuda.CUDAGraph()
            self.pending_graphs.append(graph)
            with torch.cuda.graph(graph, pool=pool, stream=stream):
                output = model.decoder(inputs[phase], context)
            captured[phase] = dict(graph=graph, output=output)
            self._drain()
        entry = dict(
            context=context,
            spatial=spatial,
            inputs=inputs,
            phases=captured,
            geometry=(height, width),
            frame_counts=frame_counts,
            stream=stream,
            dtype=dtype,
            device=device,
        )
        okay, changed = True, False
        for frames in frame_counts:
            count = 16 * frames * height * width
            initial = (torch.arange(count, device=device, dtype=torch.float32) % 127) / 127 - 0.5
            initial = initial.to(dtype).reshape(1, 16, frames, height, width)
            owned = []
            for value in (initial, initial * 0.5 + 0.0625, initial):
                # Compare the captured fused path against original arithmetic.
                fusion = model.fused_spatial_normalization
                try:
                    model.fused_spatial_normalization = False
                    expected = model._decode_eager(value, chunk_size=2)
                finally:
                    model.fused_spatial_normalization = fusion
                self._drain()
                actual = self._execute(entry, value)
                self._drain()
                okay = okay and torch.equal(actual, expected) and bool(torch.isfinite(actual).all())
                owned.append(actual)
            okay = okay and torch.equal(owned[0], owned[2])
            changed = changed or not torch.equal(owned[0], owned[1])
        okay = okay and history_addresses == tuple(tensor.data_ptr() for tensor in context.history.values())
        if not all(self._gather(okay)) or not any(self._gather(changed)):
            for phase in captured.values():
                phase["graph"].reset()
                self.pending_graphs.remove(phase["graph"])
            raise RuntimeError("SeedVR2 VAE temporal graph history or A/B/A parity failed")
        # Phases replay serially in capture order (continuation may repeat).
        # The complete-request lease and owned chunk outputs allow their
        # intermediates to share one pool; pinned history is charged once.
        pool_ids = {tuple(phase["graph"].pool()) for phase in captured.values()}
        snapshot = torch.cuda.memory_snapshot()
        if any("segment_pool_id" not in segment for segment in snapshot):
            raise RuntimeError("CUDA allocator does not expose VAE graph pool residency")
        pool_bytes = sum(segment["total_size"] for segment in snapshot if tuple(segment["segment_pool_id"]) in pool_ids)
        storages = {
            tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
            for tensor in [*inputs.values(), *context.history.values()]
        }
        pinned = sum(storages.values())
        retained = pool_bytes + pinned
        free, _ = current_omni_platform.get_device_memory()
        ready = all(
            self._gather(
                self.pool_bytes + retained <= self.config.max_pool_bytes and free > self.config.eager_headroom_bytes
            )
        )
        self.manifest.append(
            dict(
                geometry=[height, width],
                latent_frames=frame_counts,
                phases=phases,
                ready=ready,
                private_pool_bytes=pool_bytes,
                pinned_history_and_input_bytes=pinned,
                history_required=context.cache_temporal,
                fixed_history=context.fixed_history,
                shared_phase_pool=len(pool_ids) == 1,
                phase_pool_ids=[list(pool_id) for pool_id in sorted(pool_ids)],
                retained_geometry_bytes=retained,
                remaining_free_bytes=free,
                pool_budget_bytes=self.config.max_pool_bytes,
                eager_headroom_bytes=self.config.eager_headroom_bytes,
                history_address_parity=True,
                aba_exact=True,
                compile_enabled=False,
                reason=None if ready else "pool_budget_or_eager_headroom",
            )
        )
        if ready:
            self.pool_bytes += retained
            self.entries[(height, width)] = entry
        else:
            for phase in captured.values():
                phase["graph"].reset()
        for phase in captured.values():
            self.pending_graphs.remove(phase["graph"])

    def begin_request(self, geometry, *, chunk_size=None):
        entry = self.entries.get(tuple(geometry[1:]))
        valid = self.retired or (
            self.identity == _identity(self.model) and self.versions == _versions(self.model.decoder)
        )
        hit = entry is not None and geometry[0] in entry["frame_counts"] and chunk_size in (None, 2)
        rows = self._gather((tuple(geometry), chunk_size, bool(hit and valid and not self.closed), valid, self.failed))
        if any(row[4] for row in rows):
            raise SystemExit("SeedVR2 VAE graph failed; the entire worker group must restart")
        if any(row[:2] != rows[0][:2] for row in rows):
            raise RuntimeError("SeedVR2 ranks disagree on VAE decoder geometry")
        if not all(row[3] for row in rows):
            self.retire()
            entry = None
        elif not all(row[2] for row in rows):
            entry = None
        if entry is None:
            self.stats["eager_misses"] += 1
        return entry

    def _execute(self, entry, value):
        local = entry["spatial"].split(value)
        outputs = []
        for start, phase in zip(range(0, value.shape[2], 2), temporal_phases(value.shape[2])):
            size = entry["inputs"][phase].shape[2]
            entry["inputs"][phase].copy_(local[:, :, start : start + size])
            entry["phases"][phase]["graph"].replay()
            # Own each output before the next chunk or request reuses its slot.
            outputs.append(entry["phases"][phase]["output"].clone())
        return entry["spatial"].gather(torch.cat(outputs, dim=2))

    def replay(self, entry, value):
        if self._leased or self.closed or self.failed or self.retired:
            raise SystemExit("SeedVR2 VAE graph slot unavailable; worker group must restart")
        if self.identity != _identity(self.model) or self.versions != _versions(self.model.decoder):
            raise SystemExit("SeedVR2 VAE graph weights changed during the request; worker group must restart")
        if (
            value.ndim != 5
            or value.shape[0] != 1
            or value.shape[1] != 16
            or tuple(value.shape[-2:]) != entry["geometry"]
            or value.shape[2] not in entry["frame_counts"]
            or value.dtype != entry["dtype"]
            or value.device != entry["device"]
        ):
            raise SystemExit("SeedVR2 VAE graph tensor descriptor changed; worker group must restart")
        self._leased = True
        try:
            current = torch.cuda.current_stream()
            if self.last_event is not None:
                current.wait_event(self.last_event)
            result = self._execute(entry, value)
            self.last_event = torch.cuda.Event()
            self.last_event.record(current)
            self.stats["replays"] += 1
            for phase in temporal_phases(value.shape[2]):
                self.stats[phase] += 1
            return result
        except Exception as error:
            self.failed = True
            raise SystemExit("SeedVR2 VAE temporal graph execution failed; worker group must restart") from error
        finally:
            self._leased = False

    def retire(self):
        if self._leased:
            raise RuntimeError("cannot retire a leased VAE decoder graph")
        if self.retired:
            return
        torch.accelerator.synchronize()
        for entry in self.entries.values():
            for phase in entry["phases"].values():
                phase["graph"].reset()
        for graph in self.pending_graphs:
            graph.reset()
        self.pending_graphs.clear()
        self.entries.clear()
        self.pool_bytes = 0
        self.retired = True
        self.stats["retired"] += 1

    def close(self):
        if self.closed:
            return
        self.retire()
        logger.info("SeedVR2 VAE decoder CUDA graph execution: %s", json.dumps(self.stats))
        if self.cpu_group is not None:
            dist.destroy_process_group(self.cpu_group)
        self.cpu_group = None
        self.closed = True
