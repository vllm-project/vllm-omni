import torch
import os

# test if flash_attn is available
try:
    import flash_attn
    from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

try:
    from flash_attn_interface import _flash_attn_forward as flash_attn_forward_hopper
    from flash_attn_interface import _flash_attn_backward as flash_attn_func_hopper_backward
    from flash_attn_interface import flash_attn_func as flash3_attn_func
    HAS_FLASH_ATTN_HOPPER = True
except ImportError:
    HAS_FLASH_ATTN_HOPPER = False

try:
    from flashinfer.prefill import single_prefill_with_kv_cache
    HAS_FLASHINFER = True
    def get_cuda_arch():
        major, minor = torch.cuda.get_device_capability()
        return f"{major}.{minor}"

    # cuda_arch = get_cuda_arch()
    # os.environ['TORCH_CUDA_ARCH_LIST'] = cuda_arch
    # print(f"Set TORCH_CUDA_ARCH_LIST to {cuda_arch}")
except ImportError:
    HAS_FLASHINFER = False

try:
    import aiter
    from aiter import flash_attn_func as flash_attn_func_aiter
    HAS_AITER = True
except ImportError:
    HAS_AITER = False

try:
    import sageattention
    HAS_SAGE_ATTENTION = True
except ImportError:
    HAS_SAGE_ATTENTION = False

try:
    import spas_sage_attn
    HAS_SPARSE_SAGE_ATTENTION = True
except ImportError:
    HAS_SPARSE_SAGE_ATTENTION = False

try:
    import torch_npu
    HAS_NPU = True
except ImportError:
    HAS_NPU = False

