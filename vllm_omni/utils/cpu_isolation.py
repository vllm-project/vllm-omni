"""Host-thread CPU isolation for RTF load stability.

Binds hot host threads (orchestrator, stage engine cores) to dedicated CPU
cores and raises their scheduling priority so Python dispatch overhead stays
low even when the machine is loaded. Pure stdlib; silently degrades when
unsupported (non-Linux, cgroup-restricted, <16 cores, or env-disabled).
"""
import logging
import os

logger = logging.getLogger(__name__)

_ENV_DISABLE = "VLLM_OMNI_CPU_ISOLATE"
_ENV_CPUS = "VLLM_OMNI_ISOLATE_CPUS"
_MIN_CORES = 16
_CORES_PER_GROUP = 8


def _cpu_count() -> int:
    try:
        return os.cpu_count() or 1
    except Exception:
        return 1


def _allowed_cores() -> list[int]:
    """Real allowed core ids (respects cgroup affinity restrictions).

    Subprocesses forked from an already-pinned thread inherit that thread's
    narrow mask; honour a full-machine snapshot exported by the parent so
    their group slicing and min-core gate see the real machine.
    """
    raw = os.environ.get(_ENV_CPUS, "")
    if raw:
        try:
            return sorted({int(c) for c in raw.split(",") if c.strip()})
        except ValueError:
            pass
    try:
        return sorted(os.sched_getaffinity(0))
    except Exception:
        return []


def isolate_host_thread(group: int = 0) -> bool:
    """Best-effort: raise priority + pin this thread to a core group.

    Args:
        group: logical group id (0 = orchestrator, 1/2/3 = stage cores).
    Returns:
        True if at least one isolation step succeeded.
    """
    if os.environ.get(_ENV_DISABLE, "0") == "1":
        return False
    allowed = _allowed_cores()
    if len(allowed) < _MIN_CORES:
        return False
    ok = False
    mask: set[int] | None = None
    try:
        os.setpriority(os.PRIO_PROCESS, 0, -10)
        ok = True
    except Exception:
        pass
    try:
        n = len(allowed)
        ngroups = max(1, n // _CORES_PER_GROUP)
        base_idx = (group % ngroups) * _CORES_PER_GROUP
        mask = set(allowed[base_idx : base_idx + _CORES_PER_GROUP])
        os.sched_setaffinity(0, mask)
        ok = True
    except Exception:
        pass
    if ok:
        logger.info(
            "[cpu-isolate] thread group=%d cores=%s affinity+priority applied",
            group,
            sorted(mask) if mask is not None else "?",
        )
    return ok
