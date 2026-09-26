# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Runner-facing access layer for the omni prefix cache.

One copy of the integration rules shared by the GPU and NPU runners:
one-shot manager construction, the model cache-policy snapshot, and the
per-step lifecycle entry. The manager/controller state machines and the
platform output-building flows stay in their owners.

Not exported from the package ``__init__``; runners import this module
directly. No module-level vLLM import (``get_pp_group`` is resolved at
call time) so ``tests/core`` keeps loading the package without vLLM.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from vllm_omni.core.prefix_cache.adapter import PrefixCacheSchedulerAdapter, PrefixCacheStep
from vllm_omni.core.prefix_cache.group_view import get_prefix_cache_group_view
from vllm_omni.core.prefix_cache.interface import (
    ModelCachePolicy,
    OmniPrefixCacheUnmatchError,
    PrefixCacheConfig,
    StageCacheOutputs,
)
from vllm_omni.core.prefix_cache.manager import OmniPrefixCacheManager
from vllm_omni.data_entry_keys import flatten_payload

if TYPE_CHECKING:
    import torch
    from vllm.v1.core.sched.output import SchedulerOutput


class PrefixCacheRunnerMixin:
    """Prefix-cache integration for a model runner (no state machine here).

    Host-class contract — the runner must provide:
        ``input_batch``        live batch with the group-0 block table
        ``kv_cache_config``    for ``kv_cache_groups`` at construction
        ``omni_prefix_cache``  set here; ``None`` until first-step build
        ``_omni_prefix_cache_cfg``  staged by the runner's kv-cache init
                               (``stage_prefix_cache_config``), consumed here

    No cooperative ``__init__``: the class attributes below are safe
    defaults (``ModelCachePolicy`` is frozen); runners keep initializing
    their instance fields as before.
    """

    # Host-class fields (annotation only; the runner initializes them).
    input_batch: Any
    kv_cache_config: Any
    is_pooling_model: bool
    requests: dict[str, Any]
    _full_payload_token_ends: dict[str, dict[str, int]]
    accumulate_full_payload_output: Callable[..., None]

    omni_prefix_cache: OmniPrefixCacheManager | None = None
    _omni_prefix_cache_cfg: PrefixCacheConfig | None = None
    # Snapshotted once at load_model; the hot path must not re-probe model
    # attributes every step. The same object is registered on the manager.
    _omni_cache_policy: ModelCachePolicy = ModelCachePolicy()
    _prefix_cache_adapter: PrefixCacheSchedulerAdapter | None = None
    _prefix_cache_group_view: Any = None
    _prefix_cache_step: PrefixCacheStep | None = None

    def _snapshot_prefix_cache_model_policy(self, model) -> None:
        """Freeze the model's cache policy at load_model."""
        self._omni_cache_policy = ModelCachePolicy.from_model(model)

    def _model_needs_full_prefix_hidden_states(self) -> bool:
        """Opt-out hook for models whose postprocess only consumes the tail.

        When False, we skip both the per-step hidden-state write into the
        omni prefix cache and the merged-tensor reconstruction on hits;
        postprocess receives the normal scheduled-token slice instead. Models
        that need the full cached_prefix + new_tail span (default) are not
        affected.
        """
        return self._omni_cache_policy.needs_full_hidden_states

    def _ensure_omni_prefix_cache(self) -> None:
        """One-shot construction (caller gates on the staged config).

        Only the last PP rank writes and materializes the cache. Other ranks
        would register hits they can never serve (save_outputs is last-rank
        only), so skip construction there.
        """
        from vllm.distributed.parallel_state import get_pp_group

        cfg = self._omni_prefix_cache_cfg
        self._omni_prefix_cache_cfg = None
        if cfg is None or not get_pp_group().is_last_rank:
            return
        view = get_prefix_cache_group_view(
            self.input_batch,
            cfg.block_size,
            kv_cache_groups=getattr(self.kv_cache_config, "kv_cache_groups", None),
        )
        if view is None:
            raise OmniPrefixCacheUnmatchError(
                "omni prefix caching requires a block table on the input batch; "
                "disable enable_prefix_caching for this model"
            )
        manager = OmniPrefixCacheManager(cfg)
        manager.register_policy(self._omni_cache_policy)
        self.omni_prefix_cache = manager
        self._prefix_cache_adapter = PrefixCacheSchedulerAdapter()
        self._prefix_cache_group_view = view

    def _prefix_cache_step_begin(self, scheduler_output: SchedulerOutput) -> None:
        """Per-step lifecycle entry; must run before ``_update_states``
        removes finished requests (finished/abort escalation happens inside
        ``new_step_starts``). Builds the manager on the first real step."""
        if self.omni_prefix_cache is None and self._omni_prefix_cache_cfg is not None:
            self._ensure_omni_prefix_cache()
        if self.omni_prefix_cache is not None:
            if self._prefix_cache_adapter is None:
                self._prefix_cache_adapter = PrefixCacheSchedulerAdapter()
            step = self._prefix_cache_adapter.translate_step(scheduler_output)
            self._prefix_cache_step = step
            self.omni_prefix_cache.new_step_starts(step)

    def _prefix_cache_save_step(
        self,
        hidden_states: torch.Tensor,
        multimodal_outputs: dict | None,
        *,
        num_tokens_unpadded: int,
        num_tokens_padded: int,
    ) -> int | None:
        """Write this step's outputs into the cache.

        None when this stage/rank does not write (pooling, cache off, or not
        the last PP rank). The returned step id must be consumed exactly once
        — materialize or discard_step — by the output path.
        """
        from vllm.distributed.parallel_state import get_pp_group

        if self.is_pooling_model or self.omni_prefix_cache is None or not get_pp_group().is_last_rank:
            return None
        if self._prefix_cache_adapter is None or self._prefix_cache_group_view is None:
            raise RuntimeError("prefix-cache adapter was not initialized")
        if self._prefix_cache_step is None:
            raise RuntimeError("prefix-cache step snapshot was not initialized")
        layout = self._prefix_cache_adapter.build_write_layout(
            self._prefix_cache_group_view,
            num_scheduled_tokens=dict(self._prefix_cache_step.scheduled_tokens),
        )
        return self.omni_prefix_cache.save_outputs(
            hidden_states,
            flatten_payload(multimodal_outputs) if multimodal_outputs else {},
            num_tokens_unpadded=num_tokens_unpadded,
            num_tokens_padded=num_tokens_padded,
            write_layout=layout,
        )

    def _prefix_cache_materialize(self, step_id: int | None, req_ids: list[str]) -> StageCacheOutputs | None:
        """Per-request merged outputs with delivery metadata for a saved step.

        ``req_ids`` must be the save-time snapshot, never the live
        ``input_batch`` (under async output this runs a step late).
        ``step_id`` None means save did not run — nothing to consume.
        ``mm_outputs`` is all-or-nothing; empty only when the step had no mm.
        """
        if step_id is None or self.omni_prefix_cache is None:
            return None
        return self.omni_prefix_cache.materialize(step_id, list(req_ids))

    def _prefix_cache_accumulate_full_payload(
        self,
        req_ids: list[str],
        payloads: Sequence[dict[str, Any] | None],
        scheduled_tokens: dict[str, int],
        cache_outputs: StageCacheOutputs | None,
    ) -> None:
        for rid, payload in zip(req_ids, payloads):
            request = self.requests.get(rid)
            if request is None or not payload:
                continue
            cached_keys: frozenset[str] = frozenset()
            if cache_outputs is not None:
                token_range = cache_outputs.scheduled_token_ranges[rid]
                keys = set(cache_outputs.cached_mm_keys)
                if cache_outputs.hidden_states is not None:
                    keys.add("hidden")
                cached_keys = frozenset(keys)
            else:
                start = request.num_computed_tokens
                token_range = (start, start + scheduled_tokens[rid])
            self.accumulate_full_payload_output(
                rid, payload, request, token_range=token_range, prefix_cache_keys=cached_keys
            )
            if cache_outputs is not None:
                assert self.omni_prefix_cache is not None
                self.omni_prefix_cache.record_full_payload_delivery(
                    cache_outputs, rid, min(self._full_payload_token_ends[rid].values())
                )
