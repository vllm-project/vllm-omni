# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native Omni step pipeline for causal Realization latent blocks.

One public execution segment is one block; persistent pages belong to a fenced
session, not to the transient segment/request. This is the latent stage only:
codec and AR condition connectors must supply the typed layout below.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import torch
from torch import nn

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.quantization import resolve_component_quant_config

from .attention import RealizationPagedAttentionPlan
from .contracts import RealizationSequence
from .graph import RealizationGraphSlot
from .paged_cache import RealizationPagedCachePool, prefix_identity
from .parallel import all_ranks_finite
from .profiling import capture_condition, profile_phase
from .transformer import DuoTransformer
from .weights import load_fixed_adapter


@dataclass(frozen=True)
class RealizationBlockCondition:
    session_id: str
    fence: int
    block: int
    sequence: RealizationSequence
    seed: int
    prefix: RealizationSequence | None = None
    final: bool = False
    tenant: str = "local"
    reset_media_cache: bool = False


@dataclass
class _Session:
    fence: int
    next_block: int
    cache: object
    active: str | None = None
    active_state: object = None
    lookahead_state: object = None
    tenant: str = "local"
    touched: float = 0.0
    history_costs: list[tuple[int, int]] = field(default_factory=list)
    budget_shared_rows: int = 0
    history_blocks: int = 36


class DuoPipeline(nn.Module):
    dummy_run_num_frames = 0
    supports_step_execution = True
    _dit_modules = ["transformer"]
    _encoder_modules = []
    _vae_modules = []

    def __init__(self, od_config, quant_config=None):
        super().__init__()
        if (
            od_config.cache_backend != "none"
            or od_config.enable_cpu_offload
            or od_config.enable_layerwise_offload
            or od_config.enable_distributed_layerwise_offload
            or od_config.parallel_config.use_hsdp
            or od_config.lora_path
        ):
            raise ValueError("Duo requires fixed adapter and resident weights; general quantization hooks disabled")
        if not od_config.step_execution:
            raise ValueError("Duo requires native step_execution=True")
        if od_config.enable_sleep_mode or od_config.host_weight_runtime_mode != "disabled":
            raise ValueError("Duo page/graph lifetimes do not support sleep or HWR")
        self.od_config = od_config
        options = od_config.model_config.get("duo", {})
        self.options = dict(options)
        buckets = self.options.get("text_graph_buckets")
        if buckets and (buckets != sorted(set(buckets)) or buckets[0] < 1 or buckets[-1] < 128):
            raise ValueError("Text graph buckets must increase and cover the 128-token output cap")
        self.options.setdefault("sampling_stage", "causal")
        self.options.setdefault("flow_shift", 1.0)
        self.options.setdefault("compact_padding", True)
        self.options.setdefault("reuse_time_conditioning", True)
        for name in (
            "fused_denoise",
            "reuse_static_text",
            "merge_fixed_adapter",
            "joint_execution",
            "defer_clean_until_codec",
            "joint_early_clean_ack",
            "fp8_gemm",
            "joint_separate_projections",
        ):
            self.options.setdefault(name, False)
            if type(self.options[name]) is not bool:
                raise ValueError("Realization option must be boolean: " + name)
        if self.options["defer_clean_until_codec"] and self.options["joint_execution"]:
            raise ValueError("Deferred clean and joint execution are separate scheduling modes")
        if self.options["joint_execution"] and (
            od_config.max_num_seqs < 2
            or not self.options["fused_denoise"]
            or not self.options["reuse_static_text"]
            or self.options.get("training_token_budget") is not None
        ):
            raise ValueError("Joint Realization requires fused four-step static-text execution and fixed history")
        clean_history = self.options.get("clean_history_chunks")
        if clean_history is not None:
            if type(clean_history) is not int or clean_history < 0:
                raise ValueError("Realization clean_history_chunks must be a nonnegative integer")
            if self.options["joint_execution"]:
                raise ValueError("Realization clean_history_chunks requires separate clean execution")
        if self.options["reuse_static_text"] and not self.options["fused_denoise"]:
            raise ValueError("Realization static text reuse requires fused denoise")
        if type(self.options["reuse_time_conditioning"]) is not bool:
            raise ValueError("reuse_time_conditioning must be boolean")
        if type(self.options.get("compact_padding", False)) is not bool:
            raise ValueError("Realization compact_padding must be explicit bool")
        for name, default in (("history_blocks", 36), ("max_graphs", 8), ("max_sessions", 4)):
            value = self.options.get(name, default)
            if type(value) is not int or value < 1:
                raise ValueError("Invalid Realization resource limit: " + name)
        budget = self.options.get("training_token_budget")
        if budget is not None and (type(budget) is not int or budget < 128):
            raise ValueError("Realization training_token_budget must be an integer >=128")
        kv_budget = self.options.get("kv_token_budget")
        if kv_budget is not None:
            if type(kv_budget) is not int or kv_budget < 128:
                raise ValueError("Realization kv_token_budget must be an integer >=128")
            if budget is not None:
                raise ValueError("Choose kv_token_budget or training_token_budget")
        if self.options.get("session_idle_seconds", 300) <= 0:
            raise ValueError("Realization idle timeout must be positive")
        transformer_quant_config = resolve_component_quant_config(
            quant_config if quant_config is not None else getattr(od_config, "quantization_config", None), "transformer"
        )
        self.transformer = DuoTransformer(od_config, quant_config=transformer_quant_config)
        if self.transformer.quant_config is not None:
            if self.options.get("adapter_path") or self.options["merge_fixed_adapter"] or self.options["fp8_gemm"]:
                raise ValueError(
                    "Merged Realization checkpoint cannot apply an adapter, merge, or online FP8 conversion again"
                )
        root = Path(od_config.model)
        subfolder = None if root.name == "transformer" else "transformer"
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=str(root),
                subfolder=subfolder,
                revision=od_config.revision,
                prefix="transformer.",
                fall_back_to_pt=False,
            )
        ]
        self.pool = None
        self.sessions = {}
        self.retired_fences = {}
        self.graphs = {}
        self.serial_graph_captures_total = 0
        self._text_graph_prewarm_signatures = {}
        self.joint_graphs = {}
        self.weight_identity = None

    def load_weights(self, weights):
        def stream():
            for name, value in weights:
                if not name.startswith("transformer."):
                    raise ValueError("Unexpected Realization pipeline weight: " + name)
                yield name.removeprefix("transformer."), value

        loaded = {"transformer." + name for name in self.transformer.load_weights(stream())}
        if self.transformer.quant_config is not None:
            enabled = self.transformer.legacy_role_embedding
            if enabled is None:
                enabled = self.options.get("role_embedding", False)
            if type(enabled) is not bool:
                raise ValueError("Realization role_embedding must be boolean")
            identity = {"format": "modelopt_fp8", "role_embedding_enabled": enabled}
            self.transformer.role_embedding_enabled = enabled
            self.transformer._realization_adapter_loaded = True
            self.transformer._realization_adapter_merged = True
            merge_identity = {"source": "checkpoint", "merged_lora": True}
        else:
            identity, merge_identity = self._load_fixed_adapter()
        fp8_identity = None
        if self.transformer.quant_config is not None:
            fp8_identity = {
                "source": "checkpoint",
                "quant_method": self.transformer.quant_config.get_name(),
                "quant_algo": self.transformer.quant_config.quant_method,
            }
        elif self.options["fp8_gemm"]:
            from .fp8_linear import install_fp8_body

            fp8_identity = install_fp8_body(self.transformer)
        return self._finish_weight_load(loaded, identity, merge_identity, fp8_identity)

    def _load_fixed_adapter(self):
        adapter = self.options.get("adapter_path")
        if not adapter:
            raise ValueError("Explicit fixed Realization trainables adapter required")
        adapter_format = self.options.get("adapter_format", "native")
        if adapter_format == "torchtitan_split_qkv":
            from .checkpoint import load_training_adapter

            identity = load_training_adapter(
                self.transformer,
                adapter,
                rank=self.options.get("rank", 128),
                alpha=self.options.get("alpha", 128),
                role_embedding=self.options.get("role_embedding", False),
                target_modules=self.options.get("adapter_modules"),
            )
        elif adapter_format == "native":
            identity = load_fixed_adapter(
                self.transformer,
                adapter,
                rank=self.options.get("rank", 128),
                alpha=self.options.get("alpha", 128),
            )
        else:
            raise ValueError("Unknown Realization adapter format: " + str(adapter_format))
        merge_identity = None
        if self.options.get("merge_fixed_adapter", False):
            from .weights import merge_fixed_projections

            merge_identity = merge_fixed_projections(self.transformer)
        return identity, merge_identity

    def _finish_weight_load(self, loaded, identity, merge_identity, fp8_identity):
        from vllm.distributed import get_tp_group

        fp8_scales_identity = None
        if os.environ.get("DUO_FP8_ATTENTION", "0") == "1" and os.environ.get("DUO_FP8_PRECONDITION", "0") in (
            "0",
            "incremental",
        ):
            from vllm.distributed import get_tp_group

            from .weights import file_sha256

            scales_path = os.environ["DUO_FP8_SCALES"].format(rank=get_tp_group().rank_in_group)
            fp8_scales_identity = file_sha256(scales_path)
        fp8_fa3_library_identity = None
        if os.environ.get("DUO_FP8_FA3_LIBRARY"):
            from .weights import file_sha256

            fp8_fa3_library_identity = file_sha256(os.environ["DUO_FP8_FA3_LIBRARY"])
        self.weight_identity = hashlib.sha256(
            json.dumps(
                {
                    "base": self.transformer._realization_base_identity,
                    "adapter": identity,
                    "merge": merge_identity,
                    "fp8_gemm": fp8_identity,
                    # Keep the existing cache identity for supported FA3 paths.
                    "fp8_fa3": None,
                    **({"fp8_fa3_library": fp8_fa3_library_identity} if fp8_fa3_library_identity else {}),
                    "fp8_attention": os.environ.get("DUO_FP8_ATTENTION", "0"),
                    "fp8_scales": fp8_scales_identity,
                    "fp8_precondition": os.environ.get("DUO_FP8_PRECONDITION", "0"),
                    "architecture": asdict(self.transformer.arch),
                    "dtype": str(next(self.transformer.parameters()).dtype),
                    "tp": self.od_config.parallel_config.tensor_parallel_size,
                    "mask": "frame-v5",
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        # The shared loader's expected set includes eagerly loaded fixed branches.
        loaded.update(name for name, _ in self.named_parameters())
        return loaded

    def _reset_media_history(self, condition, session):
        if session.active is not None or session.next_block != condition.block:
            raise RuntimeError("Media reset requires a completed clean boundary")
        pool = self._pool()
        empty = pool.empty()
        try:
            replacement = self._commit_eager(condition.prefix, empty)
        except Exception:
            pool.release(empty)
            raise
        pool.release(session.cache)
        session.cache = replacement
        session.history_costs.clear()
        session.budget_shared_rows = session.cache.length

    def _pool(self):
        if self.pool is None:
            attn = self.transformer.blocks[0].attn
            self.pool = RealizationPagedCachePool(
                layers=len(self.transformer.blocks),
                heads=attn.num_kv_heads,
                head_dim=attn.head_dim,
                num_blocks=self.options.get("num_blocks", 512),
                page_size=self.options.get("page_size", 32),
                device=next(self.transformer.parameters()).device,
                prefix_capacity=self.options.get("prefix_capacity", 2),
            )
            if os.environ.get("DUO_FP8_ATTENTION", "0") == "1" and os.environ.get("DUO_FP8_PRECONDITION", "0") in (
                "0",
                "incremental",
            ):
                from vllm.distributed import get_tp_group

                path = os.environ["DUO_FP8_SCALES"].format(rank=get_tp_group().rank_in_group)
                profile = torch.load(path, map_location=self.pool.storage.device, weights_only=True)
                if os.environ.get("DUO_FP8_PRECONDITION") == "incremental":
                    from .fp8_incremental import install

                    install(self.pool, profile, layout=self.options.get("fp8_cache_layout"))
                else:
                    self.pool._realization_fp8_scales = profile["scales"]
        return self.pool

    def close_session(self, session_id, fence):
        session = self.sessions.get(session_id)
        if session is not None:
            if session.fence != fence:
                raise ValueError("Stale Realization session fence")
            self._pool().release(session.cache)
            self.sessions.pop(session_id)
            self.retired_fences[session_id] = fence

    def _commit_eager(self, sequence, cache):
        pool = self._pool()
        tx = pool.reserve(cache, sequence.groups)
        try:
            plan = RealizationPagedAttentionPlan(
                sequence.groups, capacity=self.options.get("kv_capacity", 32768), device=sequence.device
            )
            plan.update(sequence.groups, cache, pool)
            tx.workspace_offset = plan.native_runtime.past_rows if plan.native_runtime is not None else None
            self.transformer(
                sequence,
                torch.ones(1, device=sequence.device),
                cache=cache,
                commit=True,
                paged_runtime=(plan, pool, tx),
            )
            finite = all(
                torch.isfinite(pool.storage[i, j].flatten(0, 1).index_select(0, tx.slots)).all()
                for i in range(pool.storage.shape[0])
                for j in (0, 1)
            )
            # .item/.all above acknowledges current-stream GPU writes.
            return pool.publish(tx, acknowledged=True, finite=all_ranks_finite(finite, sequence.device))
        except Exception:
            if tx.pending:
                pool.abort(tx)
            raise

    @profile_phase
    def prepare_encode(self, state, **kwargs):
        state.extra["realization_prepare_started_ns"] = time.perf_counter_ns()
        state.extra["realization_captures_before"] = self.serial_graph_captures_total
        c = state.prepared_layout
        validate_condition(c)
        raw_sequence = c.sequence
        buckets = self.options.get("text_graph_buckets")
        if buckets:
            from .assembly import bucket_text_sequence

            c = replace(c, sequence=bucket_text_sequence(c.sequence, buckets))
        steps = state.sampling.num_inference_steps
        if steps is None:
            steps = 4
        if type(steps) is not int or steps < 1 or state.sampling.guidance_scale not in (None, 1.0):
            raise ValueError("Causal Realization needs positive NFE and no CFG")
        from .sampling import sampling_grid

        stage = state.sampling.extra_args.get("realization_sampling_stage", self.options["sampling_stage"])
        shift = state.sampling.extra_args.get("flow_shift", self.options["flow_shift"])
        grid, dts, next_times = sampling_grid(steps, stage, shift)
        device = next(self.transformer.parameters()).device
        c = replace(c, sequence=c.sequence.to(device), prefix=None if c.prefix is None else c.prefix.to(device))
        if self.options.get("compact_padding", False):
            from .contracts import compact_trailing_padding

            c = replace(
                c,
                sequence=compact_trailing_padding(c.sequence),
                prefix=None if c.prefix is None else compact_trailing_padding(c.prefix),
            )
        state.prepared_layout = c
        capture_condition(c)
        now = time.monotonic()
        # Official duplex manager owns admission/expiry and invokes worker
        # lifecycle RPCs. Do not expire clean KV using an independent timer.
        c.sequence.validate()
        live = [g for g in c.sequence.groups if g.kind != "padding"]
        if (
            not live
            or not any(g.kind == "noisy" for g in live)
            or any(g.kind not in ("text", "noisy") or g.block != c.block for g in live)
            or len({g.sample for g in live}) != 1
            or torch.any(c.sequence.time_ids[c.sequence.text_indices] != 0)
            or any(
                torch.any(c.sequence.time_ids[ids] != 1) for ids in (c.sequence.video_indices, c.sequence.audio_indices)
            )
        ):
            raise ValueError("Invalid current Realization block domain/time ownership")
        pool = self._pool()
        session = self.sessions.get(c.session_id)
        # The official manager fences cancellation before admitting a new epoch.
        # Retire the interrupted epoch if its last chunk already completed.
        if session is not None and c.block == 0 and c.fence > session.fence:
            self.close_session(c.session_id, session.fence)
            session = None
        if session is None:
            if c.block != 0 or c.fence <= self.retired_fences.get(c.session_id, -1):
                raise ValueError("New Realization session needs block zero and fresh fence")
            if len(self.sessions) >= self.options.get("max_sessions", 4):
                raise MemoryError("Realization session capacity exhausted")
            # Retired IDs are bounded by rejecting further new IDs, never by
            # forgetting a fence and accidentally accepting stale conditions.
            if c.session_id not in self.retired_fences and len(
                set(self.retired_fences) | set(self.sessions)
            ) >= self.options.get("max_retired_sessions", 4096):
                raise MemoryError("Realization fence registry full; restart with a new engine epoch")
            cache = pool.empty()
            try:
                if c.prefix is not None:
                    c.prefix.validate()
                    if any(g.sample != live[0].sample for g in c.prefix.groups if g.kind != "padding"):
                        raise ValueError("Prefix sample differs from current condition")
                    if self.weight_identity is None:
                        raise RuntimeError("Prefix sharing requires complete effective weight identity")
                    key = prefix_identity(c.prefix, namespace=f"{c.tenant}/{self.weight_identity}/bf16/tp1/frame-v5")
                    hit = pool.lookup_prefix(key)
                    if hit is not None:
                        pool.release(cache)
                        cache = hit
                    else:
                        cache = self._commit_eager(c.prefix, cache)
                        pool.remember_prefix(key, cache)
                window = self.options.get("history_blocks", 36)
                if self.options.get("kv_token_budget") is not None:
                    from .budget import kv_history_blocks

                    media_rows = c.sequence.video.shape[1] + c.sequence.audio.shape[1]
                    window = kv_history_blocks(
                        cache.length,
                        media_rows,
                        self.options["kv_token_budget"],
                        text_rows=max(128, len(c.sequence.text_indices)),
                    )
                session = _Session(
                    c.fence,
                    0,
                    cache,
                    tenant=c.tenant,
                    touched=now,
                    budget_shared_rows=cache.length,
                    history_blocks=window,
                )
                self.sessions[c.session_id] = session
            except Exception:
                pool.release(cache)
                raise
        previous = session.active_state
        lookahead = bool(
            self.options["joint_execution"]
            and previous is not None
            and session.lookahead_state is None
            and not previous.extra["realization_condition"].final
            and previous.extra.get("realization_latent_ready_emitted", False)
            and previous.step_index == previous.extra["realization_denoise_segments"]
            and not previous.extra["realization_ack"]
            and c.block == session.next_block + 1
        )
        if (
            session.fence != c.fence
            or session.tenant != c.tenant
            or (session.next_block != c.block and not lookahead)
            or (session.active is not None and not lookahead)
        ):
            raise ValueError("Duplicate/out-of-order Realization block or stale fence")
        if c.reset_media_cache and c.block:
            self._reset_media_history(c, session)
        if self.options.get("reference_window_chunks") is not None:
            first = max(0, c.block + 1 - self.options["reference_window_chunks"])
            session.cache = pool.evict(session.cache, first)

        budget = self.options.get("training_token_budget")
        state.extra["realization_text_valid_rows"] = raw_sequence.text_indices.numel()
        state.extra["realization_text_capacity"] = c.sequence.text_indices.numel()
        block_cost = sum(
            (2 if g.kind == "noisy" else 1) * (g.end - g.start)
            for g in c.sequence.groups
            if g.kind in ("noisy", "text")
        )
        block_cost -= c.sequence.text_indices.numel() - raw_sequence.text_indices.numel()
        if budget is not None:
            from .budget import budget_drop_count, padded_training_rows

            drop = budget_drop_count(
                session.budget_shared_rows, [cost for _, cost in session.history_costs], block_cost, budget
            )
            if drop:
                first = session.history_costs[drop][0] if drop < len(session.history_costs) else c.block
                session.cache = pool.evict(session.cache, first)
                del session.history_costs[:drop]
            state.extra["realization_training_equivalent_rows"] = padded_training_rows(
                session.budget_shared_rows, [cost for _, cost in session.history_costs] + [block_cost]
            )
        state.extra["realization_block_training_cost"] = block_cost
        rng = torch.Generator(device=c.sequence.device).manual_seed(c.seed % (2**63 - 1))
        video = torch.randn(c.sequence.video.shape, generator=rng, device=c.sequence.device, dtype=torch.float32)
        audio = torch.randn(c.sequence.audio.shape, generator=rng, device=c.sequence.device, dtype=torch.float32)
        noises = (
            tuple(
                (
                    torch.randn(video.shape, generator=rng, device=device, dtype=torch.float32),
                    torch.randn(audio.shape, generator=rng, device=device, dtype=torch.float32),
                )
                for _ in range(steps - 1)
            )
            if stage == "dmd"
            else ()
        )
        state.extra.update(
            realization_sampling_stage=stage,
            realization_flow_shift=shift,
            realization_grid=grid,
            realization_dts=dts,
            realization_next_times=next_times,
            realization_noises=noises,
        )
        state.latents = torch.cat((video.flatten(), audio.flatten())).view(1, -1)
        state.prompt_embeds = c.sequence.refiner_text
        fused = self.options.get("fused_denoise", False)
        if fused and (
            steps != 4 or not self.options["reuse_time_conditioning"] or not self.options.get("use_cuda_graph", True)
        ):
            raise ValueError("Fused Realization denoise requires four NFEs, graphs and time reuse")
        segments = 1 if fused else steps
        state.timesteps = torch.arange(segments + 1, device=c.sequence.device)  # separate clean segment
        state.extra.update(
            realization_condition=c,
            realization_session=session,
            realization_steps=steps,
            realization_denoise_segments=segments,
            realization_ack=False,
            realization_history_rows_before=session.cache.length,
        )
        if buckets and (state.sampling.extra_args or {}).get("realization_prewarm_text_graphs"):
            self._prewarm_text_graphs(state, raw_sequence, buckets)
        elif buckets and self.options.get("prewarm_text_graphs", False):
            self._prewarm_text_graphs(state, raw_sequence, buckets)
        if lookahead:
            session.lookahead_state = state
        else:
            session.active = state.request_id
            session.active_state = state
        session.touched = now
        return state

    def _sequence(self, state, clean=False):
        c = state.extra["realization_condition"]
        seq = c.sequence
        n = seq.video.numel()
        seq = seq.with_media(state.latents[0, :n].view_as(seq.video), state.latents[0, n:].view_as(seq.audio))
        return seq.as_clean() if clean else seq

    def _graph_key(self, sequence, commit):
        # Include exact refiner visibility topology, but not absolute block IDs.
        from .contracts import visible

        return (
            commit,
            tuple((g.kind, g.start, g.end) for g in sequence.groups),
            tuple((g.kind, g.start, g.end) for g in sequence.refiner_groups),
            tuple(visible(q, k, refiner=True) for q in sequence.refiner_groups for k in sequence.refiner_groups),
            tuple(
                (n, tuple(t.shape), t.dtype)
                for n in sequence.__dataclass_fields__
                if isinstance(t := getattr(sequence, n), torch.Tensor)
            ),
        )

    def _get_graph_slot(self, sequence, commit, sampling_stage, flow_shift):
        clean_history = self.options.get("clean_history_chunks") if commit else None
        key = self._graph_key(sequence, commit) + (sampling_stage, flow_shift, clean_history)
        slot = self.graphs.pop(key, None)
        if slot is None:
            if len(self.graphs) >= self.options.get("max_graphs", 8):
                torch.accelerator.synchronize(sequence.device)
                self.graphs.pop(next(iter(self.graphs)))
            slot = RealizationGraphSlot(
                self.transformer,
                self._pool(),
                sequence,
                capacity=self.options.get("kv_capacity", 32768),
                commit=commit,
                reuse_time_conditioning=self.options["reuse_time_conditioning"],
                sampling_stage=sampling_stage,
                flow_shift=flow_shift,
                clean_history_chunks=self.options.get("clean_history_chunks") if commit else None,
            )
        self.graphs[key] = slot
        return slot

    def _prewarm_text_graphs(self, state, raw_sequence, buckets):
        """Capture the declared capacities without publishing synthetic clean KV."""
        from .assembly import bucket_text_sequence
        from .contracts import compact_trailing_padding

        signature = (
            tuple(raw_sequence.video.shape),
            tuple(raw_sequence.audio.shape),
            tuple((g.kind, g.end - g.start) for g in raw_sequence.refiner_groups if g.kind == "shared"),
            state.extra["realization_sampling_stage"],
            state.extra["realization_flow_shift"],
            tuple(buckets),
        )
        keys = self._text_graph_prewarm_signatures.get(signature)
        if keys is not None and keys.issubset(self.graphs):
            return
        if self.options["joint_execution"] or not self.options["fused_denoise"]:
            raise ValueError("Text bucket prewarming requires serial fused four-NFE execution")
        if 2 * len(buckets) > self.options.get("max_graphs", 8):
            raise ValueError("Graph cache must retain chain and clean for every text bucket")
        pool = self._pool()
        cache = state.extra["realization_session"].cache
        stage, shift = state.extra["realization_sampling_stage"], state.extra["realization_flow_shift"]
        device = state.latents.device
        current = self._sequence(state)
        n = raw_sequence.text_indices.numel()
        live = max(g.end for g in raw_sequence.groups if g.kind != "padding")
        source = replace(
            raw_sequence,
            **{
                name: torch.cat((getattr(raw_sequence, name)[:1], getattr(raw_sequence, name)[n:live]))
                for name in ("positions", "tags", "roles", "time_ids")
            },
            video_indices=raw_sequence.video_indices - n + 1,
            audio_indices=raw_sequence.audio_indices - n + 1,
            text_indices=raw_sequence.text_indices[:1],
            text_selection=raw_sequence.text_selection[:1],
            refiner_text=raw_sequence.refiner_text[:, : raw_sequence.refiner_text.shape[1] - n + 1],
            refiner_groups=tuple(
                replace(g, end=g.start + 1) if g.kind == "text" else g for g in raw_sequence.refiner_groups
            ),
            groups=tuple(
                replace(g, end=1) if g.kind == "text" else replace(g, start=g.start - n + 1, end=g.end - n + 1)
                for g in raw_sequence.groups
                if g.kind != "padding"
            ),
        )
        keys = set()
        for capacity in buckets:
            sequence = bucket_text_sequence(source, (capacity,)).to(device).with_media(current.video, current.audio)
            if self.options.get("compact_padding", False):
                sequence = compact_trailing_padding(sequence)
            slot = self._get_graph_slot(sequence, False, stage, shift)
            keys.add(self._graph_key(sequence, False) + (stage, shift, None))
            before = slot.capture_count
            slot.run_chain(
                sequence,
                cache=cache,
                steps=4,
                reuse_static_text=self.options["reuse_static_text"],
                noises=state.extra["realization_noises"],
            )
            self.serial_graph_captures_total += slot.capture_count - before
            clean = sequence.as_clean()
            transaction = pool.reserve(cache, clean.groups)
            try:
                slot = self._get_graph_slot(clean, True, stage, shift)
                keys.add(self._graph_key(clean, True) + (stage, shift, self.options.get("clean_history_chunks")))
                before = slot.capture_count
                slot.run(clean, torch.ones(1, device=device), cache=cache, transaction=transaction)
                self.serial_graph_captures_total += slot.capture_count - before
            finally:
                pool.abort(transaction)
        # Other resident stages need the scratch memory released after capture.
        torch.accelerator.empty_cache()
        self._text_graph_prewarm_signatures = {
            s: k for s, k in self._text_graph_prewarm_signatures.items() if k.issubset(self.graphs)
        }
        self._text_graph_prewarm_signatures[signature] = keys

    @profile_phase
    def denoise_step(self, input_batch, *, states=None, **kwargs):
        if not states:
            raise ValueError("Realization step needs runner states")
        from .joint_execution import execute_joint_pairs

        paired = execute_joint_pairs(self, states) if self.options["joint_execution"] else {}
        outputs = {}
        # Clean must precede denoise on a geometry-fallback batch too.
        work = sorted(states, key=lambda x: x.extra["realization_condition"].block)
        for state in work:
            if state.request_id in paired:
                outputs[state.request_id] = paired[state.request_id]
                continue
            c = state.extra["realization_condition"]
            try:
                pool = self._pool()
                session = state.extra["realization_session"]
                steps = state.extra["realization_steps"]
                segments = state.extra.get("realization_denoise_segments", steps)
                commit = state.step_index == segments
                seq = self._sequence(state, commit)
                times = (
                    torch.ones(1, device=seq.device)
                    if commit
                    else torch.tensor([1.0, state.extra["realization_grid"][state.step_index]], device=seq.device)
                )
                tx = pool.reserve(session.cache, seq.groups) if commit else None
                try:
                    used_graph = self.options.get("use_cuda_graph", True)
                    if used_graph:
                        slot = self._get_graph_slot(
                            seq,
                            commit,
                            state.extra["realization_sampling_stage"],
                            state.extra["realization_flow_shift"],
                        )
                        captures_before = slot.capture_count
                        if (
                            not commit
                            and steps == 4
                            and (segments == 1 or state.extra.get("realization_joint_first_nfe"))
                        ):
                            video, audio = slot.run_chain(
                                seq,
                                cache=session.cache,
                                steps=steps,
                                reuse_static_text=self.options.get("reuse_static_text", False),
                                start_step=1 if state.extra.get("realization_joint_first_nfe") else 0,
                                text_cache=state.extra.get("realization_joint_text_output"),
                                noises=state.extra["realization_noises"],
                            )
                            state.extra.pop("realization_joint_text_output", None)
                        else:
                            video, audio = slot.run(
                                seq,
                                times,
                                cache=session.cache,
                                dt=0 if commit else state.extra["realization_dts"][state.step_index],
                                transaction=tx,
                                noise=state.extra["realization_noises"][state.step_index]
                                if not commit and state.step_index < len(state.extra["realization_noises"])
                                else None,
                                next_time=1.0 if commit else state.extra["realization_next_times"][state.step_index],
                            )
                        self.serial_graph_captures_total += slot.capture_count - captures_before
                    else:
                        plan = RealizationPagedAttentionPlan(
                            seq.groups,
                            capacity=self.options.get("kv_capacity", 32768),
                            device=seq.device,
                            clean_history_chunks=self.options.get("clean_history_chunks") if commit else None,
                        )
                        plan.update(seq.groups, session.cache, pool)
                        (v, a), _ = self.transformer(
                            seq, times, cache=session.cache, commit=commit, paged_runtime=(plan, pool, tx)
                        )
                        if not all_ranks_finite(
                            torch.stack([torch.isfinite(x).all() for x in (v, a)]).all(), seq.device
                        ):
                            raise FloatingPointError("Nonfinite Realization velocity")
                        video, audio = (
                            (seq.video, seq.audio)
                            if commit
                            else (
                                seq.video + v.float() * state.extra["realization_dts"][state.step_index],
                                seq.audio + a.float() * state.extra["realization_dts"][state.step_index],
                            )
                        )
                    if (
                        not used_graph
                        and not commit
                        and state.extra["realization_sampling_stage"] == "dmd"
                        and state.step_index < steps - 1
                    ):
                        from .sampling import renoise

                        renoise(
                            (video, audio),
                            state.extra["realization_noises"][state.step_index],
                            state.extra["realization_next_times"][state.step_index],
                        )
                    # Successful graph.run has synchronously checked latent,
                    # prediction and (on commit) ALL newly written K/V, on all
                    # TP ranks. Reuse that acknowledgement; do not rescan K/V
                    # on the host with 100 individual GPU synchronizations.
                    if not used_graph and not all(torch.isfinite(x).all() for x in (video, audio)):
                        raise FloatingPointError("Nonfinite Realization latent")
                    if commit:
                        if used_graph:
                            finite = True  # graph.run returned its all-rank finite GPU ack
                        else:
                            finite = all(
                                torch.isfinite(pool.storage[i, j].flatten(0, 1).index_select(0, tx.slots)).all()
                                for i in range(pool.storage.shape[0])
                                for j in (0, 1)
                            )
                            finite = all_ranks_finite(finite, seq.device)
                        session.cache = pool.publish(tx, acknowledged=True, finite=finite)
                        if self.options.get("training_token_budget") is not None:
                            session.history_costs.append((c.block, state.extra["realization_block_training_cost"]))
                        else:
                            window = session.history_blocks
                            session.cache = pool.evict(session.cache, c.block - window + 1)
                        state.extra["realization_ack"] = True
                    outputs[state.request_id] = torch.cat((video.flatten(), audio.flatten())).view(1, -1)
                except Exception:
                    if tx is not None and tx.pending:
                        pool.abort(tx)
                    raise
            except Exception:
                self.close_session(c.session_id, c.fence)
                raise
        return torch.cat([outputs[state.request_id] for state in states])

    def step_scheduler(self, state, noise_pred, **kwargs):
        # Runner result already includes the sampling update. Do NOT apply it a second time.
        if noise_pred is None or noise_pred.shape != state.latents.shape:
            raise ValueError("Realization runner output shape mismatch")
        state.latents = noise_pred.clone()
        state.step_index += 1

    def post_step_output(self, state):
        """Owned final latents for internal codec predecode, not a clean ACK.

        Opt-in per request: ordinary Realization consumers still receive one final
        output. The normal session driver gates publication on the later clean
        result. Deferred-clean mode can publish owned AV first, but retains the
        request and cache credit until ACK. This hook advances neither.
        """
        if not state.sampling.extra_args.get("realization_early_codec_dispatch", False):
            return None
        if (
            state.step_index != state.extra.get("realization_denoise_segments", state.extra["realization_steps"])
            or state.extra["realization_ack"]
        ):
            return None
        if state.extra.get("realization_latent_ready_emitted", False):
            return None
        c = state.extra["realization_condition"]
        seq = self._sequence(state)
        result = DiffusionOutput(
            output={
                "payload": {
                    "video_latent": seq.video.detach().cpu().clone(),
                    "audio_latent": seq.audio.detach().cpu().clone(),
                },
                "metadata": {
                    **output_metadata(self, state, c),
                    "commit_nfe": 0,
                    "joint_denoise": state.extra.get("realization_joint_denoise", False),
                    "joint_early_clean_ack": self.options["joint_early_clean_ack"],
                    "graph_text_valid_rows": state.extra["realization_text_valid_rows"],
                    "graph_text_capacity": state.extra["realization_text_capacity"],
                    "serial_graph_captures_total": self.serial_graph_captures_total,
                    "serial_graph_captures_in_block": self.serial_graph_captures_total
                    - state.extra["realization_captures_before"],
                    "phase": "latent_ready",
                    "defer_clean_until_codec": state.sampling.extra_args.get(
                        "realization_defer_clean_until_codec", self.options.get("defer_clean_until_codec", False)
                    ),
                    "clean_committed": False,
                },
            },
            finished=False,
        )
        state.extra["realization_latent_ready_emitted"] = True
        return result

    def post_decode(self, state, **kwargs):
        c = state.extra["realization_condition"]
        session = state.extra["realization_session"]
        if (
            not state.extra["realization_ack"]
            or state.step_index != state.extra.get("realization_denoise_segments", state.extra["realization_steps"]) + 1
        ):
            raise RuntimeError("Realization output before mandatory clean GPU acknowledgement")
        seq = self._sequence(state)
        result = DiffusionOutput(
            output={
                # The early snapshot already owns the codec tensors. Clean only
                # commits KV; returning CUDA clones would serialize them again.
                "payload": (
                    {}
                    if state.extra.get("realization_latent_ready_emitted", False)
                    else {"video_latent": seq.video.clone(), "audio_latent": seq.audio.clone()}
                ),
                "metadata": {
                    **output_metadata(self, state, c),
                    "commit_nfe": 1,
                    "joint_clean": state.extra.get("realization_joint_clean", False),
                    "joint_early_clean_ack": self.options["joint_early_clean_ack"],
                    "graph_text_valid_rows": state.extra["realization_text_valid_rows"],
                    "graph_text_capacity": state.extra["realization_text_capacity"],
                    "serial_graph_captures_total": self.serial_graph_captures_total,
                    "serial_graph_captures_in_block": self.serial_graph_captures_total
                    - state.extra["realization_captures_before"],
                    "phase": "clean_committed",
                    "clean_committed": True,
                    "graph_slots": len(self.graphs),
                    "graph_captures": sum(slot.capture_count for slot in self.graphs.values()),
                    "graph_replays": sum(slot.replay_count for slot in self.graphs.values()),
                },
            }
        )
        session.next_block += 1
        session.active_state = session.lookahead_state
        session.lookahead_state = None
        session.active = None if session.active_state is None else session.active_state.request_id
        session.touched = time.monotonic()
        if c.final:
            self.close_session(c.session_id, c.fence)
        return result

    def release_step_request(self, state):
        # Official runner retirement hook; clean-completed requests have already
        # cleared session.active, so only interrupted work closes its KV.
        self.release_request(state.request_id)

    def release_request(self, request_id):
        for sid, session in tuple(self.sessions.items()):
            if session.active == request_id or getattr(session.lookahead_state, "request_id", None) == request_id:
                self.close_session(sid, session.fence)

    def release_captured_graphs(self):
        if self.graphs:
            torch.accelerator.synchronize(next(self.transformer.parameters()).device)
        self.graphs.clear()
        self.joint_graphs.clear()
        self._text_graph_prewarm_signatures = {}

    def close(self):
        self.release_captured_graphs()
        self.sessions.clear()
        if self.pool is not None:
            self.pool.close()


def validate_condition(c):
    """Cold admission checks, before session allocation or GPU execution."""
    if (
        not isinstance(c, RealizationBlockCondition)
        or not isinstance(c.session_id, str)
        or not c.session_id
        or not isinstance(c.tenant, str)
        or not c.tenant
        or type(c.fence) is not int
        or c.fence < 0
        or type(c.block) is not int
        or c.block < 0
        or type(c.seed) is not int
        or type(c.final) is not bool
        or type(c.reset_media_cache) is not bool
    ):
        raise ValueError("Duo requires a typed fenced block condition")
    c.sequence.validate()
    if c.reset_media_cache and c.prefix is None:
        raise ValueError("Media reset needs the original reference/shared prefix")
    if c.prefix is not None:
        if c.block != 0 and not c.reset_media_cache:
            raise ValueError("Realization prefix only allowed at initialization or an explicit media reset")
        c.prefix.validate()
        if any(g.kind not in ("reference", "shared", "padding") for g in c.prefix.groups):
            raise ValueError("Realization prefix contains mutable domains")
        if torch.any(c.prefix.time_ids != 0):
            raise ValueError("Realization prefix must use clean time zero")


def get_duo_pre_process_func(od_config):
    def preprocess(request):
        if request.prepared_layout is None:
            from .wire import condition_from_wire

            request.prepared_layout = condition_from_wire(request.prompt["additional_information"]["duo_condition"])

        validate_condition(request.prepared_layout)
        c = request.prepared_layout
        if od_config.model_config.get("duo", {}).get("joint_execution", False):
            request.step_lookahead_key = ("realization", c.session_id, c.fence, c.tenant)
            request.step_lookahead_index = c.block
        s = c.sequence
        # Runner InputBatch stacks latents/text. Model-specific geometry must
        # be part of native scheduler admission, not discovered after stacking.
        request.batch_compatibility_key = (
            "duo",
            tuple(
                (name, tuple(t.shape), str(t.dtype))
                for name in s.__dataclass_fields__
                if isinstance(t := getattr(s, name), torch.Tensor)
            ),
        )
        return request

    return preprocess


def output_metadata(self, state, c):
    workspace = getattr(self.pool, "_realization_native_attention", {})
    return {
        "session_id": c.session_id,
        "fence": c.fence,
        "block": c.block,
        "denoise_nfe": state.extra["realization_steps"],
        "media_cache_reset": c.reset_media_cache,
        "sampling_stage": state.extra["realization_sampling_stage"],
        "flow_shift": state.extra["realization_flow_shift"],
        "fp8_gemm": self.options["fp8_gemm"] or self.transformer.quant_config is not None,
        "fp8_attention": os.environ.get("DUO_FP8_ATTENTION", "0") == "1",
        "fp8_fa3_library": Path(os.environ["DUO_FP8_FA3_LIBRARY"]).name
        if os.environ.get("DUO_FP8_FA3_LIBRARY")
        else None,
        "kv_workspace": {
            key: workspace.get(key, 0)
            for key in ("base", "packs", "packed_rows", "incremental_rows", "compactions", "prefix_moves")
        },
        "attention_policy": os.environ.get("DUO_ATTENTION_POLICY", "legacy"),
        "joint_graph_slots": len(self.joint_graphs),
        "joint_graph_captures": sum(g.capture_count for g in self.joint_graphs.values()),
        "joint_graph_replays": sum(g.replay_count for g in self.joint_graphs.values()),
        "fused_denoise": self.options["fused_denoise"],
        "reuse_static_text": self.options["reuse_static_text"],
        "merge_fixed_adapter": self.options["merge_fixed_adapter"],
        "use_vllm_tp_group": self.od_config.parallel_config.use_vllm_tp_group,
        "realization_state_monotonic_ns": time.perf_counter_ns(),
        "realization_prepare_started_ns": state.extra["realization_prepare_started_ns"],
        "history_rows_before": state.extra["realization_history_rows_before"],
        "history_blocks": state.extra["realization_session"].history_blocks,
        "reference_window_chunks": self.options.get("reference_window_chunks"),
        "clean_history_chunks": self.options.get("clean_history_chunks"),
        "kv_token_budget": self.options.get("kv_token_budget"),
        "training_token_budget": self.options.get("training_token_budget"),
        "training_equivalent_rows": state.extra.get("realization_training_equivalent_rows"),
        "tp": self.od_config.parallel_config.tensor_parallel_size,
        "attention_versions": sorted(
            {g.plan.native_runtime.version for g in self.graphs.values() if g.plan.native_runtime is not None}
        ),
    }
