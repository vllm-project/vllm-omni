from vllm_omni.utils.platform_utils import (
    detect_device_type,
    get_device_control_env_var,
    is_npu,
)

__all__ = [
    "detect_device_type",
    "get_device_control_env_var",
    "is_npu",
]
