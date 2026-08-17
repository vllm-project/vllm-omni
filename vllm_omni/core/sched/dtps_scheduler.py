"""DTPS (DiT-priority Type-based Scheduling) Unified Strategy.

A dynamic, load-aware scheduling strategy tailored for mixed AR+DiT deployments (e.g., serving
both i2t and t2i). Rather than blindly prioritizing downstream tasks, it balances AR/DiT
utilization by actively preventing DiT starvation, avoiding DiT queue overloads, and strictly
guaranteeing that pure AR tasks never starve.

[Dynamic 4-Tier Priority Logic]
The scheduler reads the DiT stage's real-time queue depth via a lock-free (Seqlock) shared
memory segment. It then categorizes tasks into four dynamic priority tiers per batch:

  1. L0 (Highest: Starving AR): `ar_only` tasks that exceed the aging threshold. Anti-starvation
     strictly overrides all load-awareness logic.
  2. L1 (High: Budgeted DiT): When DiT is hungry (`min_waiting < threshold`), a dynamic admission
     budget is granted. `ar_downstream` tasks within this budget are prioritized to feed the downstream.
  3. L2 (Normal: Standard AR): Standard `ar_only` tasks. When DiT is busy or the budget is exhausted,
     these tasks are prioritized. This allows AR to crunch through its own workloads instead of
     piling tasks onto an already overloaded DiT queue.
  4. L3 (Lowest: Excess DiT): Over-budget `ar_downstream` tasks are demoted here to prevent
     downstream pile-ups.

[Decoupling & Intra-Tier Sorting]
- Intra-tier sorting uses an `ar_proxy` key (combining prefill length, CoT hints, and
  max_tokens budgets), remaining entirely decoupled from specific models.
"""

from __future__ import annotations

import time
from typing import Any, NamedTuple

from vllm.logger import init_logger
from vllm.v1.request import Request

from vllm_omni.core.sched.dit_load_shared import (
    _DIT_LOAD_SHM_NAME_KEY,
    DitLoadSharedState,
)
from vllm_omni.engine.serialization import deserialize_additional_information

logger = init_logger(__name__)

_DEFAULT_I2T_AGING_S = 500.0
_DEFAULT_COT_TAG_KEY = "bot_task"
_DEFAULT_DIT_LOAD_THRESHOLD = 0
_DIT_INFLIGHT_MAX_MISS = 3
_DIT_INFLIGHT_MAX_AGE_S = 1.0


class _InflightEntry(NamedTuple):
    """One finished-AR-but-not-yet-in-DiT downstream request."""

    miss: int
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
        cot_tag_key: str = _DEFAULT_COT_TAG_KEY,
        cot_weight_table: dict[str, int] | None = None,
        dit_load_threshold: int = _DEFAULT_DIT_LOAD_THRESHOLD,
        dit_load_shm_name: str | None = None,
    ) -> None:
        self.i2t_aging_s: float = i2t_aging_s
        self.cot_tag_key: str = cot_tag_key
        try:
            self.dit_load_threshold: int = int(dit_load_threshold)
        except (TypeError, ValueError):
            self.dit_load_threshold = _DEFAULT_DIT_LOAD_THRESHOLD
        # Attach the cross-process DiT-load segment eagerly. The NAME was
        # injected into omni_dtps_config before this subprocess was spawned, so
        # the segment already exists;
        self._dit_load: DitLoadSharedState | None = None
        if dit_load_shm_name and self.dit_load_threshold > 0:
            try:
                self._dit_load = DitLoadSharedState.attach(dit_load_shm_name)
            except Exception:
                logger.debug(
                    "[OmniDTPS] DitLoadSharedState.attach(%r) failed; dit stage will stay idle",
                    dit_load_shm_name,
                    exc_info=True,
                )
        if cot_weight_table is None:
            self.cot_weight_table: dict[str, int] = {}
        elif hasattr(cot_weight_table, "items"):
            self.cot_weight_table = dict(cot_weight_table.items())
        else:
            self.cot_weight_table = dict(cot_weight_table)

        self._dit_inflight_ids: dict[str, _InflightEntry] = {}
        self._last_shm_seq: int = -1
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

        raw_table = cfg_get("cot_weight_table", None)
        if raw_table is None:
            cot_weight_table: dict[str, int] | None = None
        elif hasattr(raw_table, "items"):
            cot_weight_table = {str(k): int(v) for k, v in raw_table.items()}
        else:
            raise ValueError(f"DTPS cot_weight_table must be a mapping; got {type(raw_table).__name__}.")

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

        # The SHM name is injected at runtime by StageRuntime before the AR subprocess is spawned.
        dit_load_shm_name = cfg_get(_DIT_LOAD_SHM_NAME_KEY, None)

        return cls(
            i2t_aging_s=float(cfg_get("i2t_aging_s", _DEFAULT_I2T_AGING_S)),
            cot_tag_key=str(cfg_get("cot_tag_key", _DEFAULT_COT_TAG_KEY)),
            cot_weight_table=cot_weight_table,
            dit_load_threshold=dit_load_threshold,
            dit_load_shm_name=dit_load_shm_name,
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

    def _ar_proxy(self, req: Request) -> int:
        """Compute the ``ar_proxy`` sorting key for a downstream request.

        ``ar_proxy`` combines the prompt length and a CoT-length weight from
        the model-specific tag (e.g. ``bot_task``). Shorter proxy -> admitted
        first within the DiT admission budget.
        """
        num_prompt = getattr(req, "num_prompt_tokens", 0) or 0
        proxy = num_prompt

        # CoT-length weight from the model-specific tag (e.g. bot_task).
        info = self._deserialize_info(req)
        if info is not None and self.cot_weight_table:
            cot_tag = info.get(self.cot_tag_key)
            if cot_tag is not None:
                # Unknown tag value -> 0 extra weight (treated as no-CoT).
                proxy += int(self.cot_weight_table.get(cot_tag, 0))
        return proxy

    def register_finished_downstream(self, request_id: str) -> None:
        """Record that a downstream (t2i/it2i) request just finished AR.

        Called from ``OmniARScheduler._free_request`` at the top of the KV-
        transfer block (so only downstream requests register). The id lives in
        ``_dit_inflight_ids`` until a DiT poll reports it (de-duped out) or it
        times out (3 fresh misses / age cap). Idempotent: re-registering an id
        already tracked does NOT reset its miss/age.
        """
        if self.dit_load_threshold <= 0:
            return
        if not request_id or request_id in self._dit_inflight_ids:
            return
        self._dit_inflight_ids[request_id] = _InflightEntry(miss=0, added_mono=time.monotonic())

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
        running ids (union across all replicas) has reached DiT — drop it. Ids
        that miss across ``_DIT_INFLIGHT_MAX_MISS`` fresh snapshots (DiT empty
        -> straight to running, or aborted) are dropped, as is anything older
        than ``_DIT_INFLIGHT_MAX_AGE_S`` (guards a dead DiT). Miss is counted
        only on a FRESH snapshot (seq advanced + fresh=True); AR ticks are
        µs-scale, so counting per-tick would evict a live id almost instantly.

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
        snap_seq = self._last_shm_seq
        fresh = False
        load = self._dit_load
        if load is not None:
            try:
                read = load.snapshot()
                snap = read.snapshot
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
                snap_seq = int(read.seq)
                fresh = bool(snap.get("fresh", False))
            except Exception:
                logger.debug(
                    "[OmniDTPS] DitLoadSharedState.snapshot() raised; using inflight only",
                    exc_info=True,
                )

        # A snapshot is "fresh for miss-counting" only if it advanced (new seq)
        # AND carries live data. Counting misses on a stale (unchanged) snapshot
        # would evict a live id in µs across AR ticks.
        fresh_poll = fresh and snap_seq != self._last_shm_seq
        self._last_shm_seq = snap_seq

        dit_ids = waiting_ids | running_ids
        now_mono = time.monotonic()
        for rid in list(self._dit_inflight_ids):
            entry = self._dit_inflight_ids[rid]
            if rid in dit_ids:
                del self._dit_inflight_ids[rid]
            elif now_mono - entry.added_mono > _DIT_INFLIGHT_MAX_AGE_S:
                del self._dit_inflight_ids[rid]
            elif fresh_poll:
                miss = entry.miss + 1
                if miss >= _DIT_INFLIGHT_MAX_MISS:
                    del self._dit_inflight_ids[rid]
                else:
                    self._dit_inflight_ids[rid] = _InflightEntry(miss=miss, added_mono=entry.added_mono)

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
            "fresh_poll": fresh_poll,
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
          L1 — ``ar_downstream`` within the DiT admission budget, by ar_proxy
          L2 — remaining ``ar_only`` requests
          L3 — ``ar_downstream`` beyond the budget, by ar_proxy
        Within a layer, FCFS follows the ``waiting`` queue order (``arrival_time``
        is only read for the L0 starving check).

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
        downstream_reqs: list[tuple[int, Any]] = []
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
                proxy = self._ar_proxy(req)
                downstream_reqs.append((proxy, req))

        downstream_reqs.sort(key=lambda t: t[0])

        if budget_raw is None:
            downstream_head = [t[1] for t in downstream_reqs]
            downstream_tail: list = []
        else:
            downstream_head = [t[1] for t in downstream_reqs[:budget_raw]]
            downstream_tail = [t[1] for t in downstream_reqs[budget_raw:]]

        ordered = starving_ar_only + downstream_head + ar_only_reqs + downstream_tail
        if hasattr(waiting, "clear") and hasattr(waiting, "extend"):
            waiting.clear()
            waiting.extend(ordered)
        else:
            waiting.remove_requests(list(waiting))
            for req in ordered:
                waiting.add_request(req)
