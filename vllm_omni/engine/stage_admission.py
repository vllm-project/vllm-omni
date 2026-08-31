"""Pre-launch admission control for parallel stage initialization.

When ``VllmOmniOrchestratorConfig.parallel_stage_init`` is enabled, several stage
engine cores initialize concurrently and allocate KV cache / capture CUDA graphs
independently. The SH/EX phase locks (``stage_phase_lock``) keep each *measurement*
clean, but they cannot bound the *sum* of what all stages will allocate — two
stages can each profile against a mostly-empty GPU, each compute a large KV
budget, and then both allocate → OOM.

Admission is the hard backstop for that: **before any stage launches**, prove for
every physical device *g*::

    Σ_{s on g} capacity(g)·utilization(s) + Σ graph_reserve(s,g)
        + external_reserve + safety_margin  ≤  capacity(g)

If it does not hold, fail fast (``StageAdmissionError``) rather than OOM at
runtime. Because the budgets are proven to fit before anyone allocates, any
profile/allocate interleaving is safe.

The arithmetic (``evaluate``) is pure and unit-testable. Plan-walking
(``check_admission``) takes injectable callables for device resolution and total
memory so it can also be exercised without a GPU.
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


@dataclass
class DeviceLedger:
    """Per-device admission accounting (also the integration-run record)."""

    device_id: int
    capacity_bytes: int
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
            ledger.kv_budget_bytes += int(ledger.capacity_bytes * d.utilization)
            ledger.graph_reserve_bytes += d.graph_reserve_bytes
            ledger.contributors.append(f"stage{d.stage_id}/replica{d.replica_id}")

    over = [led for led in ledgers.values() if not led.fits]
    for led in ledgers.values():
        logger.info(
            "[admission] device %d: capacity=%s required=%s (kv=%s graph=%s ext=%s margin=%s) "
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
            "parallel_stage_init admission failed — per-device budget exceeds capacity. "
            "Lower gpu_memory_utilization, reduce co-located stages, or disable "
            f"parallel_stage_init. {detail}"
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
    demands: list[StageDemand] = []
    exempt: list[str] = []
    unaccounted: list[str] = []
    for plan in stage_plans:
        for replica in getattr(plan, "replicas", []):
            metadata = replica.metadata
            label = f"stage{metadata.stage_id}/replica{replica.replica_id}"
            device_ids = resolve_physical_devices(replica)
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
