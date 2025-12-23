from vllm_omni.utils.platform_utils import (
    detect_device_type,
    get_device_control_env_var,
    is_npu,
    is_rocm,
    is_xpu,
    torch_cuda_wrapper_for_xpu,
)

__all__ = [
    "detect_device_type",
    "get_device_control_env_var",
    "is_npu",
    "is_xpu",
    torch_cuda_wrapper_for_xpu,
    "is_rocm",
]
