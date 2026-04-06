import functools
import re
import time
from collections.abc import Callable
from threading import Lock
from typing import Any

from vllm.logger import init_logger

from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)


def profiler(name: str, func: Callable, instance: Any) -> Callable:
    """Timing a function execution."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        if name == f"{instance.__class__.__name__}.forward":
            instance.clear_profiler_records()
        if current_omni_platform.is_available():
            current_omni_platform.synchronize()
        start_time = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            if current_omni_platform.is_available():
                current_omni_platform.synchronize()
            duration = time.perf_counter() - start_time
            logger.info(f"[DiffusionPipelineProfiler] {name} took {duration:.6f}s")
            # record the profiling data: duration of stages
            with instance._profiler_lock:
                instance._stage_durations[name] = instance._stage_durations.get(name, 0.0) + duration

    return wrapper


def _parse_part(part: str) -> tuple[str, int | None]:
    """Parse 'att[num]' into ('att', num)."""
    if m := re.compile(r"(\w+)\[(\d+)\]").fullmatch(part):
        return m.group(1), int(m.group(2))
    return part, None


def _get_attribute_by_path(obj: Any, path: str) -> tuple[Any, str]:
    """Traverse an object by dotted path and return (parent_obj, attribute_name)."""
    parts = path.split(".")
    current = obj

    for part in parts[:-1]:
        attr, idx = _parse_part(part)

        current = getattr(current, attr, None)
        if current is None:
            return None, None
        if idx is not None:
            current = current[idx]

    return current, parts[-1]


def wrap_methods_by_paths(root_obj: Any, method_paths: list[str]) -> None:
    """Wrap specified methods of an object with profiler."""
    for path in method_paths:
        obj, method_name = _get_attribute_by_path(root_obj, path)
        if not obj or not hasattr(obj, method_name):
            logger.warning(f"[DiffusionPipelineProfiler] Method path {path} not found")
            continue

        original_method = getattr(obj, method_name)
        if not callable(original_method):
            logger.warning(f"[DiffusionPipelineProfiler] Attribute {path} is not callable")
            continue

        profiler_name = f"{root_obj.__class__.__name__}.{path}"
        setattr(obj, method_name, profiler(profiler_name, original_method, root_obj))


class DiffusionPipelineProfilerMixin:
    _PROFILER_TARGETS = ["vae.encode", "vae.decode", "diffuse", "text_encoder.forward", "tokenizer.forward"]

    def setup_diffusion_pipeline_profiler(
        self, profiler_targets: list[str] | None = None, enable_diffusion_pipeline_profiler: bool = False
    ) -> None:
        self.enable_diffusion_pipeline_profiler = enable_diffusion_pipeline_profiler
        if not enable_diffusion_pipeline_profiler:
            self.enable_diffusion_pipeline_profiler = enable_diffusion_pipeline_profiler
            return
        self._profiler_lock = Lock()
        self._stage_durations: dict[str, float] = {}
        targets = profiler_targets if profiler_targets is not None else self._PROFILER_TARGETS
        if not targets:
            return

        targets = ["forward"] + [
            t for t in targets if t != "forward"
        ]  # ensure "forward" implement 'clear_profiler_records' at first place

        targets = list(dict.fromkeys(targets))
        wrap_methods_by_paths(
            self,
            targets,
        )

    @property
    def stage_durations(self) -> dict[str, float]:
        with self._profiler_lock:
            return self._stage_durations.copy()

    def clear_profiler_records(self) -> None:
        with self._profiler_lock:
            self._stage_durations.clear()
