import logging
from contextvars import ContextVar

from vllm.logger import init_logger

_stage_context: ContextVar[tuple[int, str] | None] = ContextVar("_stage_context", default=None)


def set_stage_context(stage_id: int, model_stage: str) -> None:
    _stage_context.set((stage_id, model_stage))


def get_stage_context() -> tuple[int, str] | None:
    return _stage_context.get()


def clear_stage_context() -> None:
    _stage_context.set(None)


class StageContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        ctx = _stage_context.get()
        if ctx is not None:
            stage_id, model_stage = ctx
            record.msg = f"[STAGE:{model_stage}] {record.msg}"
            record.stage_id = stage_id
            record.stage_tag = model_stage
        return True


class _StageContextInjector(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        pass


def _configure_vllm_omni_root_logger():
    vllm_root = logging.getLogger("vllm")
    vllm_omni_root = logging.getLogger("vllm_omni")
    vllm_omni_root.handlers = []

    vllm_omni_root.parent = vllm_root

    vllm_omni_root.propagate = True

    vllm_omni_root.setLevel(logging.NOTSET)

    # Install the stage-context filter on both vllm_omni and vllm loggers
    # so that records emitted by vllm.* child loggers (e.g. model-loading,
    # runtime) are also tagged with the stage context in stage subprocesses.
    stage_filter = StageContextFilter()
    for logger_obj in (vllm_omni_root, vllm_root):
        injector = _StageContextInjector()
        injector.addFilter(stage_filter)
        logger_obj.addHandler(injector)


_configure_vllm_omni_root_logger()
init_logger(__name__)
