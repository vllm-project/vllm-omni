"""DTPS (DiT-priority Type-based Scheduling) Unified Strategy.

A dynamic, load-aware scheduling strategy tailored for mixed AR+DiT deployments (e.g., serving
both i2t and t2i). Rather than blindly prioritizing downstream tasks, it balances AR/DiT
utilization by actively preventing DiT starvation, avoiding DiT queue overloads, and strictly
guaranteeing that pure AR tasks never starve.

[Dynamic 3-Tier Priority Logic]
The scheduler receives the DiT stage's real-time queue depth via a UTILITY
ZMQ push from the Orchestrator. It then categorizes tasks into three dynamic
priority tiers per batch:

  1. L0 (Highest: Starving AR): `ar_only` tasks that exceed the aging threshold. Anti-starvation
     strictly overrides all load-awareness logic.
  2. L1 (High: Budgeted DiT): When DiT is hungry (`min_waiting < threshold`), a dynamic admission
     budget is granted. `ar_downstream` tasks within this budget are prioritized to feed the downstream.
  3. L2 (Normal: Standard AR): Standard `ar_only` tasks. When DiT is busy or the budget is exhausted,
     these tasks are prioritized. This allows AR to crunch through its own workloads instead of
     piling tasks onto an already overloaded DiT queue.

Over-budget `ar_downstream` tasks are demoted below L2 to prevent downstream pile-ups.

[Intra-Tier Sorting]
- All tiers use FCFS (first-come-first-served) based on the ``waiting`` queue
  order. No model-specific CoT weights or prompt-length proxies are used —
  the scheduler is entirely decoupled from model internals.
"""

from __future__ import annotations

import time
from typing import Any, NamedTuple

from vllm.logger import init_logger
from vllm.v1.request import Request

from vllm_omni.core.sched.dit_load_state import DitLoadSnapshot
from vllm_omni.engine.serialization import deserialize_additional_information

logger = init_logger(__name__)

_DEFAULT_I2T_AGING_S = 100.0
_DEFAULT_DIT_LOAD_THRESHOLD = 2
_DIT_INFLIGHT_MAX_AGE_S = 1.0


class _InflightEntry(NamedTuple):
    """One finished-AR-but-not-yet-in-DiT downstream request."""

    added_mono: float


# i2t / t2t finish at the AR stage -> ar_only; t2i / it2i -> ar_downstream.
_AR_ONLY_TASKS: frozenset[str] = frozenset({"i2t", "t2t"})
_AR_DOWNSTREAM_TASKS: frozenset[str] = frozenset({"t2i", "it2i"})


class DTPSScheduler:
    """A single instance is owned by ``OmniARScheduler`` (one per AR stage replica)
    and invoked once per ``schedule()`` cycle via :meth:`maybe_reorder_waiting`.
    """

    def __init__(
        self,
        *,
        i2t_aging_s: float = _DEFAULT_I2T_AGING_S,
        dit_load_threshold: int = _DEFAULT_DIT_LOAD_THRESHOLD,
    ) -> None:
        self.i2t_aging_s: float = i2t_aging_s
        try:
            self.dit_load_threshold: int = int(dit_load_threshold)
        except (TypeError, ValueError):
            self.dit_load_threshold = _DEFAULT_DIT_LOAD_THRESHOLD
        # DiT-load snapshot pushed by the Orchestrator via UTILITY ZMQ.
        self._dit_load_snapshot: DitLoadSnapshot | None = None

        self._dit_inflight_ids: dict[str, _InflightEntry] = {}
        self._last_phase_stats: dict[str, int | bool] = {}

    @classmethod
    def from_config(cls, dtps_cfg: Any) -> DTPSScheduler:
        # Build a DTPSScheduler from the ``omni_dtps_config`` block.
        if isinstance(dtps_cfg, dict):
            cfg_get = dtps_cfg.get
        else:

            def cfg_get(key: str, default: Any = None) -> Any:
                return getattr(dtps_cfg, key, default)

        if not cfg_get("enabled", False):
            raise ValueError(
                "DTPS config block present but 'enabled' is not True; refusing to construct DTPSScheduler."
            )

        raw_threshold = cfg_get("dit_load_threshold", _DEFAULT_DIT_LOAD_THRESHOLD)
        try:
            dit_load_threshold = int(raw_threshold)
        except (TypeError, ValueError):
            logger.warning(
                "[OmniDTPS] Invalid dit_load_threshold=%r; using default %d.",
                raw_threshold,
                _DEFAULT_DIT_LOAD_THRESHOLD,
            )
            dit_load_threshold = _DEFAULT_DIT_LOAD_THRESHOLD

        return cls(
            dit_load_threshold=dit_load_threshold,
        )

    # ------------------------------------------------------------------ #
    #  Task classification (model-agnostic)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _deserialize_info(req: Request) -> dict[str, Any] | None:
        info = getattr(req, "additional_information", None)
        if info is None:
            return None
        if isinstance(info, dict):
            return info
        try:
            info = deserialize_additional_information(info)
        except Exception:
            logger.debug(
                "[OmniDTPS] deserialize additional_information failed",
                exc_info=True,
            )
            return None
        return info if isinstance(info, dict) else None

    def _classify_task(self, req: Request) -> str:
        """Classify a request by task type: i2t / t2i / it2i / t2t / unknown.

        Primary signal: ``additional_information["omni_task_type"]`` (stamped
        at the API entry).
        """
        info = self._deserialize_info(req)
        if info is None:
            return "unknown"
        tag = info.get("omni_task_type")
        if isinstance(tag, str) and tag:
            return tag
        return "unknown"

    @staticmethod
    def _task_bucket(task: str) -> str:
        """Map a task type to its DTPS bucket: ``ar_only`` or ``ar_downstream``.

        unknown / unrecognized -> ``ar_downstream`` (conservative: never starve
        the downstream stage).
        """
        if task in _AR_ONLY_TASKS:
            return "ar_only"
        if task in _AR_DOWNSTREAM_TASKS:
            return "ar_downstream"
        return "ar_downstream"

    def update_dit_load(self, snapshot: DitLoadSnapshot) -> None:
        """Receive DiT load snapshot from the Orchestrator via UTILITY ZMQ.

        Called by ``OmniARScheduler.update_dit_load``, which is invoked by
        ``StageEngineCoreProc.omni_update_dit_load`` via vLLM's UTILITY
        dispatch (``getattr(self, method_name)``). Runs on the same
        EngineCore busy-loop thread as ``schedule()``, so no lock is needed.
        """
        self._dit_load_snapshot = snapshot

    def register_finished_downstream(self, request_id: str) -> None:
        """Record that a downstream (t2i/it2i) request just finished AR.

        Called from ``OmniARScheduler._free_request`` at the top of the KV-
        transfer block (so only downstream requests register). The id lives
        in ``_dit_inflight_ids`` until a DiT poll reports it (de-duped out)
        or it times out (age cap). Idempotent: re-registering an id already
        tracked does NOT reset its age.
        """
        if self.dit_load_threshold <= 0:
            return
        if not request_id or request_id in self._dit_inflight_ids:
            return
        self._dit_inflight_ids[request_id] = _InflightEntry(added_mono=time.monotonic())

    def _dit_phase(self, inflight: int = 0) -> str:
        """Return the DiT-load phase: ``"idle"`` or ``"busy"``.

        ``inflight`` is the Fix-B feed-forward count of downstream (t2i/it2i)
        requests currently RUNNING on this AR stage — AR knows they will land
        on DiT once they finish here but the polled DiT-load report hasn't
        reflected them yet. ``_dit_inflight_ids`` is the blind-spot set: ids
        that already LEFT AR's running set (t0) but haven't surfaced in a DiT
        poll yet. The two terms are mutually exclusive (a request is in
        AR-running XOR in the blind set), so they sum cleanly.

        De-dup pass: any blind id that appears in DiT's reported waiting OR
        running ids (union across all replicas) has reached DiT — drop it.
        Anything older than ``_DIT_INFLIGHT_MAX_AGE_S`` is also dropped
        (guards a dead DiT).

        Multi-replica: both inflight terms (running + blind) spread uniformly
        across ``n_reps`` DiT replicas, so only ~1/R of them land on the min
        replica and actually raise ``min_waiting``. Fold them together and
        floor-divide by R (R<=1 -> no fold, single-replica Fix-B behavior
        exact). Floor biases toward idle (feed DiT) — safe per DTPS's goal.
        """
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

        Must be called after ``process_pending_chunks`` /
        ``_consume_pending_connector_output`` (so only genuinely schedulable
        WAITING requests remain) and before ``super().schedule()`` (so the
        admission order follows the reorder).

        ``running`` is the AR stage's current running set; its downstream
        (t2i/it2i) members are fed forward as anticipated DiT load (see
        :meth:`_dit_phase`) so the phase decision isn't fooled by the
        poll-lagged DiT-load report. ``None`` -> the running term is 0.

        Priority layers (smaller layer = admitted first):
          L0 — ``ar_only`` requests waiting longer than ``i2t_aging_s``
               (starving; aging boost to prevent starvation). ALWAYS highest.
          L1 — ``ar_downstream`` within the DiT admission budget (FCFS)
          L2 — remaining ``ar_only`` requests (FCFS)
          L3 — ``ar_downstream`` beyond the budget (FCFS)

        Within each layer, FCFS follows the ``waiting`` queue order
        (``arrival_time`` is only read for the L0 starving check).

        ``ar_proxy`` and arrival are read-only; only queue order is mutated.
        FCFSRequestQueue (a deque subclass) is reordered via clear()+extend();
        any other queue type falls back to remove_requests()+add_request().
        """
        inflight_running = 0
        if running is not None:
            inflight_running = sum(1 for r in running if self._task_bucket(self._classify_task(r)) == "ar_downstream")
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
        starving_ar_only: list = []
        aging_threshold = self.i2t_aging_s
        now = time.time()

        for req in list(waiting):
            task = self._classify_task(req)
            bucket = self._task_bucket(task)
            arrival = getattr(req, "arrival_time", None)
            wait = (now - arrival) if arrival is not None else 0.0
            if bucket == "ar_only":
                starving = wait > aging_threshold
                if starving:
                    starving_ar_only.append(req)
                else:
                    ar_only_reqs.append(req)
            else:
                downstream_reqs.append(req)

        if budget_raw is None:
            downstream_head = downstream_reqs
            downstream_tail: list = []
        else:
            downstream_head = downstream_reqs[:budget_raw]
            downstream_tail = downstream_reqs[budget_raw:]

        ordered = starving_ar_only + downstream_head + ar_only_reqs + downstream_tail
        if hasattr(waiting, "clear") and hasattr(waiting, "extend"):
            waiting.clear()
            waiting.extend(ordered)
        else:
            waiting.remove_requests(list(waiting))
            for req in ordered:
                waiting.add_request(req)
