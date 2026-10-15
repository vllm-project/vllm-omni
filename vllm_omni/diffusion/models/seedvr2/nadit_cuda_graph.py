# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""A bounded, startup-only collection of exact NaDiT core CUDA graphs.

Each SP group has one mutable execution slot per profile. Inputs are copied on
the consumer stream and every result is cloned before that slot is reusable.
VAE, patchify, text input projection and canonical reconstruction remain eager.
"""

import json
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

import torch
import torch.distributed as dist
from vllm.logger import init_logger

from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)
CONFIG_KEY = "seedvr2_nadit_cuda_graph"


@dataclass(frozen=True)
class NaDiTGraphConfig:
    profiles: tuple[tuple[int, int, int], ...]
    max_pool_bytes: int = 1024**3
    eager_headroom_bytes: int = 8 * 1024**3

    @classmethod
    def from_extras(cls, extras: dict[str, Any], *, enforce_eager: bool):
        value = extras.get(CONFIG_KEY)
        if enforce_eager or value is None:
            return None
        if not isinstance(value, dict) or set(value) - {"profiles", "max_pool_bytes", "eager_headroom_bytes"}:
            raise ValueError(f"{CONFIG_KEY} requires profiles and optional byte budgets")
        profiles = value.get("profiles")
        if not isinstance(profiles, list) or not 1 <= len(profiles) <= 4:
            raise ValueError("SeedVR2 NaDiT CUDA graphs require 1-4 exact latent [T,H,W] profiles")
        parsed = []
        for shape in profiles:
            if not isinstance(shape, (list, tuple)) or len(shape) != 3:
                raise ValueError("NaDiT graph profile must be a latent [T,H,W] triple")
            if any(type(x) is not int or x <= 0 for x in shape) or shape[1] % 2 or shape[2] % 2:
                raise ValueError("NaDiT graph latent dimensions must be positive integers with even H/W")
            parsed.append(tuple(shape))
        if len(set(parsed)) != len(parsed):
            raise ValueError("duplicate NaDiT graph profile")
        budgets = {k: value.get(k, getattr(cls, k)) for k in ("max_pool_bytes", "eager_headroom_bytes")}
        if any(type(x) is not int or x <= 0 for x in budgets.values()):
            raise ValueError("NaDiT graph budgets must be positive integer byte counts")
        # All ranks process profiles in the same order, regardless of JSON order.
        return cls(tuple(sorted(parsed)), **budgets)


def validate_transition_descriptors(all_descriptors):
    """Validate peer sends/receives, allowing different ragged rank vectors."""
    world = len(all_descriptors)
    if not world:
        raise ValueError("empty graph transition descriptors")
    length = len(all_descriptors[0])
    for rank, rows in enumerate(all_descriptors):
        if len(rows) != length:
            raise ValueError("rank transition count mismatch")
        for row in rows:
            if len(row["send"]) != world or len(row["recv"]) != world:
                raise ValueError("transition split vector size mismatch")
            if any(type(x) is not int or x < 0 for x in row["send"] + row["recv"]):
                raise ValueError("invalid transition row count")
            if sum(row["send"]) != row["input_rows"] or sum(row["recv"]) != row["output_rows"]:
                raise ValueError("transition row totals mismatch")
        for peer, other in enumerate(all_descriptors):
            for left, right in zip(rows, other):
                if left["layer"] != right["layer"] or left["send"][peer] != right["recv"][rank]:
                    raise ValueError("peer graph transition splits mismatch")


def _tensor_signature(value):
    return (tuple(value.shape), tuple(value.stride()), value.dtype, value.device)


def _execution_identity(model):
    return (
        model.patch_size,
        model.window,
        model.window_method,
        model.num_layers,
        tuple(
            (
                block.attn.attention_path,
                block.attn.attention_backend_name,
                block.attn.use_varlen_kernel,
                block.attn.heads,
                block.attn.head_dim,
            )
            for block in model.blocks
        ),
    )


def _versions(model):
    # Metadata only: no device scalar reads. Loaders also explicitly retire
    # graphs before copying into inference tensors, which have no version.
    result = []
    for tensor in (*model.parameters(), *model.buffers()):
        try:
            version = tensor._version
        except RuntimeError:
            version = None
        result.append((id(tensor), tensor.data_ptr(), tensor.dtype, tensor.device, version))
    return tuple(result)


def _outside_pool_tensor_bytes(model, runtime, static):
    """Charge all pinned metadata and inputs, with unique CUDA storages."""
    storages = {}

    def visit(value):
        if isinstance(value, torch.Tensor) and value.device.type == "cuda":
            storage = value.untyped_storage()
            storages[(value.device, storage.data_ptr())] = storage.nbytes()
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        elif is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                visit(getattr(value, field.name))

    visit(static)
    visit(runtime._contexts)
    visit(runtime.manager._device_plans)
    for block in model.blocks:
        visit(block.attn.rope._axis_cache)
    return sum(storages.values())


class NaDiTCoreGraphs:
    def __init__(self, model, text, parallel_config, config: NaDiTGraphConfig):
        self.model, self.config = model, config
        self.entries = {}
        self.stats = {"replays": 0, "eager_misses": 0, "retired": 0}
        self.cpu_group = None
        probe = model.build_runtime(
            model.token_grid_for(config.profiles[0]), text_len=text.shape[0], parallel_config=parallel_config
        )
        if probe.world_size not in (1, 2, 4) or (dist.is_initialized() and dist.get_world_size() != probe.world_size):
            raise ValueError("SeedVR2 NaDiT graphs require a single GPU or one SP2/SP4 group per service")
        self.group, self.world, self.rank = probe.group, probe.world_size, probe.rank
        if self.world > 1:
            self.cpu_group = dist.new_group(backend="gloo")
        self.pool_bytes = 0
        self.identity = _execution_identity(model)
        self.closed = False
        self.retired = False
        self.failed = False
        self._leased = False
        self.last_event = None
        self.manifest = []
        model.core_graphs = self  # retain prepared entries even if a later startup profile fails
        signatures = self._gather(
            (
                config.profiles,
                config.max_pool_bytes,
                config.eager_headroom_bytes,
                text.shape[0],
                str(text.dtype),
                self.identity,
            )
        )
        if any(signature != signatures[0] for signature in signatures):
            raise RuntimeError("SeedVR2 SP ranks disagree on the startup graph manifest")
        # Capture failures are startup failures. Do not continue with a possibly
        # damaged communicator or silently turn on generic torch.compile.
        with torch.inference_mode():
            for geometry in config.profiles:
                self._prepare(geometry, text, parallel_config)
        self.versions = _versions(model)
        model.reset_attention_stats()
        logger.info("SeedVR2 NaDiT CUDA graph manifest: %s", json.dumps(self.manifest))

    def _gather(self, value):
        if self.world == 1:
            return [value]
        values = [None] * self.world
        dist.all_gather_object(values, value, group=self.cpu_group)
        return values

    def _drain(self):
        torch.accelerator.synchronize()
        if self.world > 1:
            dist.barrier(group=self.cpu_group)

    def _prepare(self, geometry, text, parallel_config):
        free, _ = current_omni_platform.get_device_memory()
        if not all(self._gather(free > self.config.eager_headroom_bytes + self.config.max_pool_bytes)):
            self.manifest.append(dict(latent_shape=geometry, ready=False, reason="headroom_before_capture"))
            return
        model = self.model
        runtime = model.build_runtime(
            model.token_grid_for(geometry), text_len=text.shape[0], parallel_config=parallel_config
        )
        device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
        runtime.manager.set_device(device)
        descriptors = []
        for layer in range(model.num_layers):
            layout = runtime.layout_for_layer(layer)
            runtime.context(layout, device)
            if layer:
                previous = runtime.layout_for_layer(layer - 1)
                transition = runtime.manager.device_plan(previous, layout)
                descriptors.append(
                    dict(
                        layer=layer,
                        send=list(transition.input_split_sizes),
                        recv=list(transition.output_split_sizes),
                        input_rows=runtime.manager.rank_plan(previous).num_local_tokens,
                        output_rows=runtime.manager.rank_plan(layout).num_local_tokens,
                    )
                )
        validate_transition_descriptors(self._gather(descriptors))
        rows = runtime.manager.rank_plan(runtime.layout_for_layer(0)).num_local_tokens
        static = [
            torch.zeros((rows, model.vid_in.proj.in_features), device=device, dtype=dtype),
            model.txt_in(text.to(dtype)).clone(),
            torch.tensor([1000.0], device=device, dtype=dtype),
        ]
        initial = [x.clone() for x in static]
        changed = [initial[0] + 0.25, initial[1] * 0.95, initial[2].clone()]

        def operation():
            return model.forward_core(*static, runtime)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                operation()
        torch.cuda.current_stream().wait_stream(stream)
        self._drain()
        before_runtime = dict(runtime.stats)
        before_manager = runtime.manager.transitions
        before_attention = [dict(block.attn.attention_stats) for block in model.blocks]
        operation()
        counters = {k: v - before_runtime[k] for k, v in runtime.stats.items()}
        manager_transitions = runtime.manager.transitions - before_manager
        attention = [
            {k: v - old[k] for k, v in block.attn.attention_stats.items()}
            for block, old in zip(model.blocks, before_attention)
        ]
        self._drain()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = operation()
        self._drain()
        owned, okay = [], True
        for values in (initial, changed, initial):
            for destination, source in zip(static, values):
                destination.copy_(source)
            expected = operation()
            self._drain()
            graph.replay()
            actual = output.clone()
            self._drain()
            okay = okay and torch.equal(expected, actual) and bool(torch.isfinite(actual).all())
            owned.append(actual)
        okay = okay and torch.equal(owned[0], owned[2])
        checks = self._gather((bool(okay), not torch.equal(owned[0], owned[1])))
        # Empty-output ranks still take part in all collectives. Content must
        # change the global result, not necessarily each rank's empty tensor.
        if not all(row[0] for row in checks) or not any(row[1] for row in checks):
            graph.reset()
            raise RuntimeError("SeedVR2 NaDiT graph startup A/B/A or owned-output parity failed")
        # Count graph-private allocator segments, rather than a misleading net
        # reserved-memory delta (capture can release the eager allocator cache).
        pool_id = tuple(graph.pool())
        snapshot = torch.cuda.memory_snapshot()
        if any("segment_pool_id" not in segment for segment in snapshot):
            graph.reset()
            raise RuntimeError("CUDA allocator does not expose graph pool residency")
        pool_bytes = sum(segment["total_size"] for segment in snapshot if tuple(segment["segment_pool_id"]) == pool_id)
        tensor_bytes = _outside_pool_tensor_bytes(model, runtime, static)
        retained = pool_bytes + tensor_bytes
        free, _ = current_omni_platform.get_device_memory()
        ready = self.pool_bytes + retained <= self.config.max_pool_bytes and free > self.config.eager_headroom_bytes
        ready = all(self._gather(ready))
        record = dict(
            latent_shape=geometry,
            ready=ready,
            private_pool_bytes=pool_bytes,
            pinned_metadata_and_input_bytes=tensor_bytes,
            rank=self.rank,
            aba_exact=True,
            compile_enabled=False,
            profiles_share_graph_pool=False,
            reason=None if ready else "pool_budget_or_eager_headroom",
        )
        self.manifest.append(record)
        if not ready:
            graph.reset()
            return
        self.pool_bytes += retained
        self.entries[geometry] = dict(
            graph=graph,
            output=output,
            static=static,
            runtime=runtime,
            signatures=tuple(_tensor_signature(x) for x in static),
            counters=counters,
            manager_transitions=manager_transitions,
            attention=attention,
            stream=stream,
        )

    def begin_request(self, geometry, runtime):
        """Agree on graph/eager before video preparation, VAE or NCCL work."""
        entry = self.entries.get(geometry)
        valid_identity = self.retired or (
            self.group is runtime.group
            and self.world == runtime.world_size
            and self.rank == runtime.rank
            and self.versions == _versions(self.model)
            and self.identity == _execution_identity(self.model)
        )
        choice = bool(entry is not None and valid_identity and not self.closed)
        rows = self._gather((geometry, runtime.text_len, runtime.world_size, choice, valid_identity, self.failed))
        if any(row[5] for row in rows):
            raise SystemExit("SeedVR2 NaDiT CUDA graph failed; the entire worker group must restart")
        if any(row[:3] != rows[0][:3] for row in rows):
            raise RuntimeError("SeedVR2 SP ranks disagree on request geometry or conditioning")
        if not all(row[4] for row in rows):
            # All ranks retire together. An untracked tensor/group change never
            # recaptures in the hot path; later requests remain eager.
            self.retire()
            entry = None
        elif not all(row[3] for row in rows):
            entry = None
        runtime._seedvr2_core_graph_entry = entry
        if entry is None:
            self.stats["eager_misses"] += 1

    def replay(self, entry, values, runtime):
        if self._leased:
            raise SystemExit("SeedVR2 NaDiT CUDA graph slot already leased; worker group must restart")
        if tuple(_tensor_signature(x) for x in values) != entry["signatures"]:
            raise SystemExit("SeedVR2 prepared NaDiT graph input descriptor changed; worker group must restart")
        self._leased = True
        try:
            current = torch.cuda.current_stream()
            if self.last_event is not None:
                current.wait_event(self.last_event)
            for destination, source in zip(entry["static"], values):
                destination.copy_(source)
            entry["graph"].replay()
            result = entry["output"].clone()
            self.last_event = torch.cuda.Event()
            self.last_event.record(current)
            for key, delta in entry["counters"].items():
                runtime.stats[key] += delta
            runtime.manager.transitions += entry["manager_transitions"]
            for block, deltas in zip(self.model.blocks, entry["attention"]):
                for key, delta in deltas.items():
                    block.attn.attention_stats[key] += delta
            self.stats["replays"] += 1
            return result
        except Exception as error:
            self.failed = True
            # The executor detects the exited rank and tears down its peers.
            # A partially executed graph is never retried as an eager request.
            raise SystemExit("SeedVR2 NaDiT CUDA graph execution failed; worker group must restart") from error
        finally:
            self._leased = False

    def retire(self):
        if self._leased:
            raise RuntimeError("cannot retire a leased NaDiT CUDA graph")
        if self.retired:
            return
        torch.accelerator.synchronize()
        for entry in self.entries.values():
            entry["graph"].reset()
        self.entries.clear()
        self.pool_bytes = 0
        self.stats["retired"] += 1
        self.retired = True

    def close(self):
        if self.closed:
            return
        self.retire()
        logger.info("SeedVR2 NaDiT CUDA graph execution: %s", json.dumps(self.stats))
        # Reset graphs before communicator teardown; CUDA/NCCL graph resources
        # otherwise outlive the process group whose communication they captured.
        if self.cpu_group is not None:
            dist.destroy_process_group(self.cpu_group)
        self.cpu_group = None
        self.closed = True
