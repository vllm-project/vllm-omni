import functools
import os
from threading import Lock
import time
from collections.abc import Callable
from typing import Any, Dict

from vllm.logger import init_logger

from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)


def profiler(name: str, func: Callable, instance: Any) -> Callable:
    """Timing a function execution."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        if current_omni_platform.is_available():
            current_omni_platform.synchronize()
        start_time = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            if current_omni_platform.is_available():
                current_omni_platform.synchronize()
            duration = time.perf_counter() - start_time
            logger.info(f"[DiffusionTiming] {name} took {duration:.6f}s")
            # record the profiling data: duration of stages
            with instance._profiler_lock:
                instance._stage_durations[name] = duration

    return wrapper


def _get_attribute_by_path(obj: Any, path: str) -> tuple[Any, str]:
    """Traverse an object by dotted path and return (parent_obj, attribute_name)."""
    parts = path.split(".")
    current = obj

    for part in parts[:-1]:
        current = getattr(current, part, None)
        if current is None:
            return None, None

    return current, parts[-1]


def wrap_methods_by_paths(root_obj: Any, method_paths: list[str]) -> None:
    """Wrap specified methods of an object with profiler."""
    if not hasattr(root_obj, "_profiler_lock"):
        root_obj._profiler_lock = Lock()
        root_obj._stage_durations: Dict[str, float] = {}

    for path in method_paths:
        obj, method_name = _get_attribute_by_path(root_obj, path)
        if not obj or not hasattr(obj, method_name):
            logger.warning(f"[DiffusionTiming] Method path {path} not found")
            continue

        original_method = getattr(obj, method_name)
        if not callable(original_method):
            logger.warning(f"[DiffusionTiming] Attribute {path} is not callable")
            continue

        profiler_name = f"{root_obj.__class__.__name__}.{path}"
        setattr(obj, method_name, profiler(profiler_name, original_method, root_obj))


class DiffusionPipelineProfilerMixin:
    _PROFILER_TARGETS = ["vae.encode", "vae.decode", "diffuse", "text_encoder.forward", "tokenizer.forward"]

    def setup_diffusion_pipeline_profiler(self) -> None:
        if "ENABLE_DIFFUSION_PIPELINE_PROFILER" not in os.environ:
            return
        default_targets = set(self._PROFILER_TARGETS)
        env_targets = {
            t.strip() for t in os.environ.get("DIFFUSION_PIPELINE_PROFILER_TARGETS", "").split(",") if t.strip()
        }

        profiler_targets = list(dict.fromkeys(default_targets | env_targets))
        wrap_methods_by_paths(
            self,
            profiler_targets,
        )
    
    @property
    def stage_durations(self) -> Dict[str, float]:
        with self._profiler_lock:
            return self._stage_durations.copy()
        
    def clear_profiler_records(self) -> None:
        with self._profiler_lock:
            self._stage_durations.clear()
