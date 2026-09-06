# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""DTPS (DiT-priority Type-based Scheduling) for AR+DiT deployments.

Reorders the AR waiting queue by task type + DiT load to prevent DiT
starvation, avoid DiT overload, and guarantee no task starves.

Priority tiers (admit top → bottom):
  L0 — any request waiting > aging_s (starvation guard, FCFS)
  L1 — ar_downstream within DiT budget (sorted by num_prompt_tokens ASC)
  L2 — remaining ar_only (FCFS)
  L3 — ar_downstream beyond budget (sorted by num_prompt_tokens ASC)

Task classification uses the topology-resolved ``omni_final_stage_id``
signal — no model-specific task-type strings.
"""

from __future__ import annotations

import time
from typing import Any, NamedTuple

from vllm.logger import init_logger
from vllm.v1.request import Request

from vllm_omni.core.sched.dit_load_state import DitLoadSnapshot
from vllm_omni.engine.serialization import deserialize_additional_information

logger = init_logger(__name__)

_DEFAULT_AGING_S = 100.0
_DEFAULT_DIT_LOAD_THRESHOLD = 2
_DIT_INFLIGHT_MAX_AGE_S = 1.0


class _InflightEntry(NamedTuple):
    added_mono: float


class DTPSScheduler:
    """One per AR stage replica, invoked once per ``schedule()`` cycle."""

    def __init__(
        self,
        *,
        stage_id: int = 0,
        aging_s: float = _DEFAULT_AGING_S,
        dit_load_threshold: int = _DEFAULT_DIT_LOAD_THRESHOLD,
    ) -> None:
        self._stage_id: int = stage_id
        self.aging_s: float = aging_s
        try:
            self.dit_load_threshold: int = int(dit_load_threshold)
        except (TypeError, ValueError):
            self.dit_load_threshold = _DEFAULT_DIT_LOAD_THRESHOLD
        self._dit_load_snapshot: DitLoadSnapshot | None = None
        self._dit_inflight_ids: dict[str, _InflightEntry] = {}
        self._last_phase_stats: dict[str, int | bool] = {}
        self._ar_only_cache: dict[str, bool] = {}

    def _request_is_ar_only(self, req: Request) -> bool:
        """True when the request terminates at this AR stage (no downstream).

        Result is cached per request_id — deserialization happens at most
        once per request lifetime. Cache is evicted via
        :meth:`evict_request` when the request is freed.
        """
        rid = req.request_id
        cached = self._ar_only_cache.get(rid)
        if cached is not None:
            return cached
        result = self._compute_ar_only(req)
        self._ar_only_cache[rid] = result
        return result

    def evict_request(self, request_id: str) -> None:
        """Remove a finished request's cached classification."""
        self._ar_only_cache.pop(request_id, None)

    def _compute_ar_only(self, req: Request) -> bool:
        info = getattr(req, "additional_information", None)
        if info is None:
            return False
        if isinstance(info, dict):
            pass
        else:
            try:
                info = deserialize_additional_information(info)
            except Exception:
                logger.debug("[OmniDTPS] deserialize additional_information failed", exc_info=True)
                return False
            if not isinstance(info, dict):
                return False
        final_stage_id = info.get("omni_final_stage_id")
        force_kv = bool(info.get("omni_force_kv_transfer", False))
        return final_stage_id == self._stage_id and not force_kv

    def update_dit_load(self, snapshot: DitLoadSnapshot) -> None:
        self._dit_load_snapshot = snapshot

    def register_finished_downstream(self, request_id: str) -> None:
        if self.dit_load_threshold <= 0:
            return
        if not request_id or request_id in self._dit_inflight_ids:
            return
        self._dit_inflight_ids[request_id] = _InflightEntry(added_mono=time.monotonic())

    def _dit_phase(self, inflight: int = 0) -> str:
        """Return ``"idle"`` or ``"busy"`` based on DiT load + in-flight count."""
        inflight_running = max(int(inflight or 0), 0)
        if self.dit_load_threshold <= 0:
            self._last_phase_stats = {}
            return "idle"

        reported_min = 0
        max_waiting = 0
        total_waiting = 0
        total_running = 0
        waiting_ids: frozenset[str] = frozenset()
        running_ids: frozenset[str] = frozenset()
        n_reps = 0
        snap = self._dit_load_snapshot
        if snap is not None:
            try:
                reported_min = int(snap.get("min_waiting", 0))
                max_waiting = int(snap.get("max_waiting", 0))
                total_waiting = int(snap.get("total_waiting", 0))
                total_running = int(snap.get("total_running", 0))
                w_ids = snap.get("waiting_ids")
                r_ids = snap.get("running_ids")
                if isinstance(w_ids, frozenset):
                    waiting_ids = w_ids
                if isinstance(r_ids, frozenset):
                    running_ids = r_ids
                n_reps = int(snap.get("num_replicas", 0))
            except Exception:
                logger.debug(
                    "[OmniDTPS] reading dit_load_snapshot raised; using inflight only",
                    exc_info=True,
                )

        dit_ids = waiting_ids | running_ids
        now_mono = time.monotonic()
        for rid in list(self._dit_inflight_ids):
            entry = self._dit_inflight_ids[rid]
            if rid in dit_ids:
                del self._dit_inflight_ids[rid]
            elif now_mono - entry.added_mono > _DIT_INFLIGHT_MAX_AGE_S:
                del self._dit_inflight_ids[rid]

        inflight_blind = len(self._dit_inflight_ids)
        inflight_total = inflight_running + inflight_blind
        if n_reps <= 1:
            inflight_reduced = inflight_total
        else:
            inflight_reduced = inflight_total // n_reps
        effective_min = reported_min + inflight_reduced
        phase = "busy" if effective_min >= self.dit_load_threshold else "idle"

        self._last_phase_stats = {
            "reported_min": reported_min,
            "max_waiting": max_waiting,
            "total_waiting": total_waiting,
            "total_running": total_running,
            "inflight_running": inflight_running,
            "inflight_blind": inflight_blind,
            "inflight_total": inflight_total,
            "n_reps": n_reps,
            "inflight_reduced": inflight_reduced,
            "effective_min": effective_min,
        }
        return phase

    def maybe_reorder_waiting(
        self,
        waiting: Any,
        running: list[Request] | None = None,
    ) -> None:
        """Reorder the AR ``waiting`` queue by DTPS priority layers.

        L0: any request waiting > aging_s (FCFS)
        L1: ar_downstream within budget (num_prompt_tokens ASC)
        L2: remaining ar_only (FCFS)
        L3: ar_downstream beyond budget (num_prompt_tokens ASC)
        """
        inflight_running = 0
        if running is not None:
            inflight_running = sum(1 for r in running if not self._request_is_ar_only(r))
        self._dit_phase(inflight_running)
        if self.dit_load_threshold <= 0:
            budget_raw: int | None = None
        else:
            stats = self._last_phase_stats
            eff_min = int(stats.get("effective_min", 0))
            n_reps = max(1, int(stats.get("n_reps", 0)))
            budget_raw = max(0, self.dit_load_threshold - eff_min) * n_reps

        ar_only_reqs: list = []
        downstream_reqs: list = []
        starving_reqs: list = []
        aging_threshold = self.aging_s
        now = time.time()

        for req in list(waiting):
            is_ar_only = self._request_is_ar_only(req)
            arrival = getattr(req, "arrival_time", None)
            wait = (now - arrival) if arrival is not None else 0.0
            if wait > aging_threshold:
                starving_reqs.append(req)
            elif is_ar_only:
                ar_only_reqs.append(req)
            else:
                downstream_reqs.append(req)

        downstream_reqs.sort(key=lambda r: getattr(r, "num_prompt_tokens", 0) or 0)

        if budget_raw is None:
            downstream_head = downstream_reqs
            downstream_tail: list = []
        else:
            downstream_head = downstream_reqs[:budget_raw]
            downstream_tail = downstream_reqs[budget_raw:]

        ordered = starving_reqs + downstream_head + ar_only_reqs + downstream_tail
        before_list = list(waiting)

        if [r.request_id for r in ordered] == [r.request_id for r in before_list]:
            return

        def _fmt(reqs: list) -> str:
            parts = []
            for r in reqs:
                w = (now - r.arrival_time) if r.arrival_time else 0.0
                npt = getattr(r, "num_prompt_tokens", 0) or 0
                tag = "AR+DiT" if not self._request_is_ar_only(r) else "AR"
                parts.append(f"{r.request_id}({tag},{w:.1f}s,{npt}t)")
            return "[" + ",".join(parts) + "]"

        logger.debug(
            "[OmniDTPS] reorder: before=%s | L0_starving=%s L1_budget=%s "
            "L2_ar_only=%s L3_overflow=%s | budget=%s eff_min=%s | "
            "after=%s",
            _fmt(before_list),
            _fmt(starving_reqs),
            _fmt(downstream_head),
            _fmt(ar_only_reqs),
            _fmt(downstream_tail),
            budget_raw,
            self._last_phase_stats.get("effective_min"),
            _fmt(ordered),
        )

        if hasattr(waiting, "clear") and hasattr(waiting, "extend"):
            waiting.clear()
            waiting.extend(ordered)
        else:
            waiting.remove_requests(before_list)
            for req in ordered:
                waiting.add_request(req)
