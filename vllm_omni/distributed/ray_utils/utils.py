# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import os
from contextlib import contextmanager
from typing import Any

import torch

try:
    import ray
    from ray.util.queue import Queue as RayQueue
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    RAY_AVAILABLE = True
    from ray.util.placement_group import PlacementGroup
except ImportError:
    ray = None
    RayQueue = None
    PlacementGroupSchedulingStrategy = None
    RAY_AVAILABLE = False
    PlacementGroup = Any

logger = logging.getLogger(__name__)


def is_ray_initialized():
    """Check if Ray is initialized without hard dependency on Ray."""
    # 1. Try standard API
    if RAY_AVAILABLE:
        if ray.is_initialized():
            return True
    # 2. Fallback: Check environment variables typical for Ray Workers
    # RAY_RAYLET_PID is always set in Ray workers
    if "RAY_RAYLET_PID" in os.environ:
        return True
    return False


def calculate_total_bytes(size_args, dtype):
    """
    Calculate total bytes for a tensor allocation, handling nested tuples in size args.
    """
    num_elements = 1
    for s in size_args:
        if isinstance(s, (tuple, list)):
            for inner in s:
                num_elements *= inner
        else:
            num_elements *= s

    element_size = torch.tensor([], dtype=dtype).element_size()
    return num_elements * element_size


@contextmanager
def maybe_disable_pin_memory_for_ray(obj, size_bytes, threshold=32 * 1024 * 1024):
    """
    Context manager to temporarily disable pin_memory if running in Ray and
    the allocation size exceeds the threshold.

    This is a workaround for Ray workers often having low ulimit -l (locked memory),
    causing OS call failed errors when allocating large pinned buffers.
    """
    should_disable = False
    old_pin = False

    # Check 1: Are we in a Ray-like environment?
    in_ray = is_ray_initialized()

    # Check 2: Is the size large enough to worry?
    is_large = size_bytes > threshold

    # Check 3: Is pinning currently enabled?
    is_pinned = getattr(obj, "pin_memory", False)

    if in_ray and is_large and is_pinned:
        should_disable = True
        old_pin = obj.pin_memory
        obj.pin_memory = False

    try:
        yield
    finally:
        if should_disable:
            obj.pin_memory = old_pin


# --- Ray specific utilities ---


def get_ray_queue_class():
    if not RAY_AVAILABLE:
        raise ImportError("ray is required for worker_backend='ray'")
    return lambda: RayQueue(maxsize=0)


def initialize_ray_cluster(address: str | None = None):
    if not RAY_AVAILABLE:
        logger.warning("Ray is not available, skipping initialization.")
        return

    if not ray.is_initialized():
        # Pass current PYTHONPATH to workers to ensure they can find vllm_omni
        runtime_env = {"env_vars": {"PYTHONPATH": os.environ.get("PYTHONPATH", "")}}
        ray.init(address=address, ignore_reinit_error=True, runtime_env=runtime_env)


def create_placement_group(number_of_stages: int, address: str | None = None, strategy: str = "PACK") -> PlacementGroup:
    """Create a placement group for the given number of stages.
    Args:
        number_of_stages: The number of stages to create the placement group for.
        strategy: The strategy to use for the placement group.
    Returns:
        The placement group.
    """
    if not RAY_AVAILABLE:
        raise ImportError("ray is required for creating placement group")

    # Initialize Ray if not already initialized (using default args if needed)
    if not ray.is_initialized():
        logger.warning("[Orchestrator] Ray is not initialized. Initializing with default settings.")
        initialize_ray_cluster(address)

    bundles = [{"GPU": 1.0, "CPU": 1.0} for _ in range(number_of_stages)]
    pg = ray.util.placement_group(bundles, strategy=strategy)
    ray.get(pg.ready())
    logger.info("[Orchestrator] Ray Placement Group created")
    return pg


def remove_placement_group(pg):
    if pg and RAY_AVAILABLE:
        try:
            ray.util.remove_placement_group(pg)
        except Exception as e:
            logger.warning(f"Failed to remove placement group: {e}")


def try_close_ray(pg=None):
    """Try to clean up Ray resources including placement group and shutdown."""
    if pg:
        remove_placement_group(pg)
    # Note: We typically don't shutdown ray.init() here as it might be used by other components
    # or the user might want it to persist. If full shutdown is needed, ray.shutdown() can be called.


def kill_ray_actor(actor):
    if actor and RAY_AVAILABLE:
        try:
            ray.kill(actor)
        except Exception as e:
            logger.warning(f"Failed to kill actor: {e}")


def start_ray_actor(
    worker_entry_fn,
    placement_group,
    placement_group_bundle_index: int,
    *args,
    **kwargs,
):
    if not RAY_AVAILABLE:
        raise ImportError("ray is required for starting ray actor")

    @ray.remote(num_gpus=1)
    class OmniStageRayWorker:
        def run(self, func, *args, **kwargs):
            return func(*args, **kwargs)

    worker_actor = OmniStageRayWorker.options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=placement_group, placement_group_bundle_index=placement_group_bundle_index
        ),
        runtime_env={"env_vars": {"PYTHONPATH": os.environ.get("PYTHONPATH", "")}, "CUDA_LAUNCH_BLOCKING": "1"},
    ).remote()

    worker_actor.run.remote(worker_entry_fn, *args, **kwargs)

    return worker_actor
