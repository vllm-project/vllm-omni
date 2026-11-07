import logging

from vllm.logger import init_logger


def _configure_vllm_omni_root_logger():
    """
    Configure the root logger for vllm_omni to propagate to vllm's root logger.
    """
    vllm_root = logging.getLogger("vllm")
    vllm_omni_root = logging.getLogger("vllm_omni")
    vllm_omni_root.handlers = []

    vllm_omni_root.parent = vllm_root

    vllm_omni_root.propagate = True

    vllm_omni_root.setLevel(logging.NOTSET)


def configure_subprocess_logging() -> None:
    """Apply vLLM's logging config in a ``spawn`` child.

    Since vLLM 0.31, importing ``vllm.logger`` no longer configures logging; each
    process must do it. The config is rebuilt from the inherited ``VLLM_*`` env.
    """
    from vllm.config.logging import LoggingConfig
    from vllm.logger import configure_logging_if_needed

    configure_logging_if_needed(LoggingConfig())


_configure_vllm_omni_root_logger()
init_logger(__name__)
