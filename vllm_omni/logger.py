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


_configure_vllm_omni_root_logger()
init_logger(__name__)
