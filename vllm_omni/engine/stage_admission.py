"""Pre-launch per-device admission for parallel init and static HBM budgets.

With explicit hbm_limit_gb, sum total per-rank envelopes instead of utilization
claims. Stage reserves are included in those totals. This is budget accounting,
not a guarantee against unprofiled runtime peaks or unrelated allocations.

Legacy parallel initialization uses utilization-based claims plus graph reserves.
Static mode uses explicit total HBM envelopes (including their graph/slack
reserves), with the same external reserve and device safety margin. The existing
SH/EX initialization locks keep profiling phases quiescent; they do not enforce
runtime allocation limits. The evaluator and injectable plan walker run before
any local stage launches.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from vllm.logger import init_logger
from vllm.utils.mem_utils import format_gib

logger = init_logger(__name__)

# Tunable reserves (bytes). Conservative defaults; operators can pass overrides
# to ``check_admission``. Kept as constants (not env vars) per project policy.
_DEFAULT_EXTERNAL_RESERVE_BYTES = 1 * 1024**3  # headroom for unmanaged consumers
_DEFAULT_SAFETY_MARGIN_BYTES = 1 * 1024**3  # fragmentation / allocator slack

# Graph-pool reserve. A single conservative constant used whenever CUDA-graph
# capture is enabled (never 0 in that case); zero when capture is disabled.
# Kept as a constant (not env var / not a measured table) per project policy.
_DEFAULT_GRAPH_RESERVE_BYTES = 2 * 1024**3


class StageAdmissionError(RuntimeError):
    """Raised when the planned stage budgets do not fit the available devices."""


class AdmissionExempt:
    """Type of :data:`ADMISSION_EXEMPT`; not meant to be instantiated elsewhere."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "ADMISSION_EXEMPT"


ADMISSION_EXEMPT = AdmissionExempt()
"""Resolver sentinel: the replica is deliberately outside local admission
(e.g. it runs on a remote node and consumes that node's memory). Only replicas
marked exempt may skip the ledger — a local replica the resolver cannot account
for fails admission instead (see :func:`check_admission`)."""


@dataclass
class StageDemand:
    """One replica's claim on a set of physical devices."""

    stage_id: int
    replica_id: int
    device_ids: list[int]
    utilization: float
    graph_reserve_bytes: int
    is_diffusion: bool = False
    hbm_budget_bytes: int | None = None


@dataclass
class DeviceLedger:
    """Per-device admission accounting (also the integration-run record)."""

    device_id: int
    capacity_bytes: int
    # Historical field name: now sums stage claims (not just KV storage).
    kv_budget_bytes: int = 0
    graph_reserve_bytes: int = 0
    external_reserve_bytes: int = 0
    safety_margin_bytes: int = 0
    contributors: list[str] = field(default_factory=list)

    @property
    def required_bytes(self) -> int:
        return self.kv_budget_bytes + self.graph_reserve_bytes + self.external_reserve_bytes + self.safety_margin_bytes

    @property
    def fits(self) -> bool:
        return self.required_bytes <= self.capacity_bytes


# ---- graph reserve ---------------------------------------------------------


def _cudagraph_disabled(vllm_config: Any) -> bool:
    model_config = getattr(vllm_config, "model_config", None)
    if getattr(model_config, "enforce_eager", False):
        return True
    comp = getattr(vllm_config, "compilation_config", None)
    mode = getattr(comp, "cudagraph_mode", None)
    if mode is not None and str(mode).upper().endswith("NONE"):
        return True
    return False


def graph_reserve_bytes(vllm_config: Any) -> int:
    """Reserve for this stage's CUDA-graph pool (bytes).

    A single conservative constant when capture is enabled (never 0 in that
    case), 0 when capture is disabled. This intentionally over-reserves rather
    than measure per-model — admission stays a safe upper bound.
    """
    if _cudagraph_disabled(vllm_config):
        return 0
    return _DEFAULT_GRAPH_RESERVE_BYTES


# ---- pure evaluation -------------------------------------------------------


def evaluate(
    demands: Sequence[StageDemand],
    capacities: dict[int, int],
    *,
    external_reserve_bytes: int = _DEFAULT_EXTERNAL_RESERVE_BYTES,
    safety_margin_bytes: int = _DEFAULT_SAFETY_MARGIN_BYTES,
) -> dict[int, DeviceLedger]:
    """Build the per-device ledger and raise ``StageAdmissionError`` if any device
    is over-subscribed. Pure: no config objects, no GPU."""
    ledgers: dict[int, DeviceLedger] = {}

    def _ledger(dev: int) -> DeviceLedger:
        if dev not in ledgers:
            if dev not in capacities:
                raise StageAdmissionError(f"No capacity known for physical device {dev}")
            ledgers[dev] = DeviceLedger(
                device_id=dev,
                capacity_bytes=capacities[dev],
                external_reserve_bytes=external_reserve_bytes,
                safety_margin_bytes=safety_margin_bytes,
            )
        return ledgers[dev]

    for d in demands:
        for dev in d.device_ids:
            ledger = _ledger(dev)
            ledger.kv_budget_bytes += (
                d.hbm_budget_bytes
                if d.hbm_budget_bytes is not None
                else int(ledger.capacity_bytes * d.utilization)
            )
            ledger.graph_reserve_bytes += d.graph_reserve_bytes
            ledger.contributors.append(f"stage{d.stage_id}/replica{d.replica_id}")

    over = [led for led in ledgers.values() if not led.fits]
    for led in ledgers.values():
        logger.info(
            "[admission] device %d: capacity=%s required=%s (stage_budget=%s graph=%s ext=%s margin=%s) "
            "headroom=%s contributors=%s%s",
            led.device_id,
            format_gib(led.capacity_bytes),
            format_gib(led.required_bytes),
            format_gib(led.kv_budget_bytes),
            format_gib(led.graph_reserve_bytes),
            format_gib(led.external_reserve_bytes),
            format_gib(led.safety_margin_bytes),
            format_gib(led.capacity_bytes - led.required_bytes),
            led.contributors,
            " <<< OVER" if not led.fits else "",
        )
    if over:
        detail = "; ".join(
            f"device {led.device_id}: need {format_gib(led.required_bytes)} GiB > "
            f"{format_gib(led.capacity_bytes)} GiB (contributors {led.contributors})"
            for led in over
        )
        raise StageAdmissionError(
            "Stage admission failed — per-device budget exceeds capacity. "
            "Lower stage budgets or reduce co-located stages. "
            f"{detail}"
        )
    return ledgers


# ---- plan walking ----------------------------------------------------------


def check_admission(
    stage_plans: Sequence[Any],
    *,
    resolve_physical_devices: Callable[[Any], list[int] | AdmissionExempt | None],
    device_total_memory: Callable[[int], int],
    external_reserve_bytes: int = _DEFAULT_EXTERNAL_RESERVE_BYTES,
    safety_margin_bytes: int = _DEFAULT_SAFETY_MARGIN_BYTES,
    graph_reserve: Callable[[Any], int] = graph_reserve_bytes,
) -> dict[int, DeviceLedger]:
    """Extract demands from the orchestrator's stage plans and admit them.

    ``resolve_physical_devices(replica)`` returns the physical GPU ids a replica
    occupies, or :data:`ADMISSION_EXEMPT` for replicas deliberately outside
    local admission (e.g. running on a remote node). Any *other* replica that
    cannot be accounted — unresolved or non-integer devices, or a diffusion
    stage without an explicit ``gpu_memory_utilization`` — raises
    ``StageAdmissionError``: under parallel init every local replica gets its
    own init group and no parent holds a whole-init exclusive lock, so a
    consumer invisible to the ledger would initialize concurrently with
    unbounded demand, defeating admission (fail-closed).
    ``device_total_memory(id)`` returns a device's total bytes.
    """
    from vllm_omni.config.static_budget import budget_bytes

    static_mode = any(
        replica_hbm_limit(replica) is not None
        for plan in stage_plans for replica in getattr(plan, "replicas", [])
    )
    demands: list[StageDemand] = []
    exempt: list[str] = []
    unaccounted: list[str] = []
    for plan in stage_plans:
        for replica in getattr(plan, "replicas", []):
            metadata = replica.metadata
            label = f"stage{metadata.stage_id}/replica{replica.replica_id}"
            device_ids = resolve_physical_devices(replica)
            if static_mode:
                limit = replica_hbm_limit(replica)
                if limit is None or isinstance(device_ids, AdmissionExempt):
                    raise StageAdmissionError(
                        f"Static HBM requires explicit budgets and local placement for every replica: {label}"
                    )
                if not device_ids:
                    raise StageAdmissionError(f"Unresolved static HBM devices for {label}")
                raw = getattr(getattr(replica, "stage_cfg", None), "engine_args", {}) or {}
                cfg = getattr(getattr(replica, "stage_vllm_config", None), "model_config", None)
                reserve = getattr(cfg, "hbm_reserved_gb", raw.get("hbm_reserved_gb", 2.0))
                if getattr(cfg, "worker_type", raw.get("worker_type")) == "generation":
                    raise StageAdmissionError(f"Static HBM is not supported for generation worker {label}")
                if replica.stage_vllm_config is None:
                    mode = raw.get("diffusion_kv_mode", "dense_legacy")
                    if getattr(mode, "value", mode) != "paged_scheduler":
                        raise StageAdmissionError(f"{label}: static HBM requires profiled paged_scheduler diffusion")
                demands.append(
                    StageDemand(
                        stage_id=metadata.stage_id,
                        replica_id=replica.replica_id,
                        device_ids=list(device_ids),
                        utilization=0.0,
                        graph_reserve_bytes=0,
                        is_diffusion=replica.stage_vllm_config is None,
                        hbm_budget_bytes=budget_bytes(limit, reserve),
                    )
                )
                continue
            if isinstance(device_ids, AdmissionExempt):
                exempt.append(label)
                continue
            if not device_ids:
                unaccounted.append(f"{label} (unresolved devices)")
                continue
            vllm_config = replica.stage_vllm_config
            if vllm_config is None:
                # Diffusion stages have no resolved vllm_config pre-launch and
                # skip profile/capture; admit them with their raw util. A local
                # diffusion stage without one has unknowable demand and must
                # fail admission below, not bypass it.
                util = _diffusion_utilization(replica)
                if util is None:
                    unaccounted.append(f"{label} (diffusion, no gpu_memory_utilization)")
                    continue
                demands.append(
                    StageDemand(
                        stage_id=metadata.stage_id,
                        replica_id=replica.replica_id,
                        device_ids=list(device_ids),
                        utilization=util,
                        graph_reserve_bytes=0,
                        is_diffusion=True,
                    )
                )
                continue
            demands.append(
                StageDemand(
                    stage_id=metadata.stage_id,
                    replica_id=replica.replica_id,
                    device_ids=list(device_ids),
                    utilization=float(vllm_config.cache_config.gpu_memory_utilization),
                    graph_reserve_bytes=int(graph_reserve(vllm_config)),
                )
            )

    if unaccounted:
        raise StageAdmissionError(
            "parallel_stage_init admission cannot account for local replicas: "
            f"{unaccounted}. Every local stage must resolve to integer physical "
            "device ids and declare a memory budget (diffusion stages: set "
            "gpu_memory_utilization in engine_args); otherwise it would "
            "initialize concurrently without bounding its demand. Fix the "
            "stage config or disable parallel_stage_init."
        )
    if exempt:
        logger.info(
            "[admission] exempt from local admission (remote/operator-isolated): %s. "
            "These replicas consume no local device memory.",
            exempt,
        )

    capacities: dict[int, int] = {}
    for d in demands:
        for dev in d.device_ids:
            if dev not in capacities:
                capacities[dev] = int(device_total_memory(dev))

    return evaluate(
        demands,
        capacities,
        external_reserve_bytes=external_reserve_bytes,
        safety_margin_bytes=safety_margin_bytes,
    )


def _diffusion_utilization(replica: Any) -> float | None:
    """Best-effort gpu_memory_utilization for a diffusion replica from raw args."""
    stage_cfg = getattr(replica, "stage_cfg", None)
    engine_args = getattr(stage_cfg, "engine_args", None)
    if isinstance(engine_args, dict):
        util = engine_args.get("gpu_memory_utilization")
        if util is not None:
            return float(util)
    return None


def replica_hbm_limit(replica: Any) -> float | None:
    """Read the resolved AR model config or diffusion pre-launch arguments."""
    config = getattr(getattr(replica, "stage_vllm_config", None), "model_config", None)
    limit = getattr(config, "hbm_limit_gb", None)
    if limit is not None:
        return limit
    raw = getattr(getattr(replica, "stage_cfg", None), "engine_args", None)
    return raw.get("hbm_limit_gb") if isinstance(raw, dict) else None
