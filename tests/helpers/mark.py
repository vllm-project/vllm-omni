# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pytest marks and decorators for hardware / resource selection (CUDA, ROCm, …)."""

import pytest
from vllm.platforms import current_platform

# Re-exported from tests.helpers.env (GPU wait + DeviceMemoryMonitor).


def cuda_marks(*, res: str, num_cards: int):
    test_platform_detail = pytest.mark.cuda
    if res == "L4":
        test_resource = pytest.mark.L4
    elif res == "H100":
        test_resource = pytest.mark.H100
    else:
        raise ValueError(f"Invalid CUDA resource type: {res}. Supported: L4, H100")
    marks = [test_resource, test_platform_detail]
    if num_cards == 1:
        return marks
    test_distributed = pytest.mark.distributed_cuda(num_cards=num_cards)

    test_skipif = pytest.mark.skipif_cuda(
        not current_platform.is_cuda() or (current_platform.device_count() < num_cards),
        reason=f"Need at least {num_cards} CUDA GPUs to run the test.",
    )
    return marks + [test_distributed, test_skipif]


def rocm_marks(*, res: str, num_cards: int):
    test_platform_detail = pytest.mark.rocm
    if res == "MI325":
        test_resource = pytest.mark.MI325
    else:
        raise ValueError(f"Invalid ROCm resource type: {res}. Supported: MI325")
    marks = [test_resource, test_platform_detail]
    if num_cards == 1:
        return marks
    test_distributed = pytest.mark.distributed_rocm(num_cards=num_cards)
    return marks + [test_distributed]


def xpu_marks(*, res: str, num_cards: int):
    test_platform_detail = pytest.mark.xpu
    if res == "B60":
        test_resource = pytest.mark.B60
    else:
        raise ValueError(f"Invalid XPU resource type: {res}. Supported: B60")
    marks = [test_resource, test_platform_detail]
    if num_cards == 1:
        return marks
    test_distributed = pytest.mark.distributed_rocm(num_cards=num_cards)
    return marks + [test_distributed]


def musa_marks(*, res: str, num_cards: int):
    test_platform_detail = pytest.mark.musa
    if res == "S5000":
        test_resource = pytest.mark.S5000
    else:
        raise ValueError(f"Invalid MUSA resource type: {res}. Supported: S5000")
    marks = [test_resource, test_platform_detail]
    if num_cards == 1:
        return marks
    test_distributed = pytest.mark.distributed_musa(num_cards=num_cards)
    return marks + [test_distributed]


def gpu_marks(*, res: str, num_cards: int):
    test_platform = pytest.mark.gpu
    if res in ("L4", "H100"):
        return [test_platform] + cuda_marks(res=res, num_cards=num_cards)
    if res == "MI325":
        return [test_platform] + rocm_marks(res=res, num_cards=num_cards)
    if res == "B60":
        return [test_platform] + xpu_marks(res=res, num_cards=num_cards)
    if res == "S5000":
        return [test_platform] + musa_marks(res=res, num_cards=num_cards)
    raise ValueError(f"Invalid resource type: {res}. Supported: L4, H100, MI325, B60, S5000")


def npu_marks(*, res: str, num_cards: int):
    test_platform = pytest.mark.npu
    if res == "A2":
        test_resource = pytest.mark.A2
    elif res == "A3":
        test_resource = pytest.mark.A3
    else:
        test_resource = None
    if num_cards == 1:
        return [mark for mark in [test_platform, test_resource] if mark is not None]
    test_distributed = pytest.mark.distributed_npu(num_cards=num_cards)
    return [mark for mark in [test_platform, test_resource, test_distributed] if mark is not None]


def hardware_marks(*, res: dict[str, str], num_cards: int | dict[str, int] = 1):
    for platform, _ in res.items():
        if platform not in ("cuda", "rocm", "xpu", "npu", "musa"):
            raise ValueError(f"Unsupported platform: {platform}")
    if isinstance(num_cards, int):
        num_cards_dict = {platform: num_cards for platform in res.keys()}
    else:
        num_cards_dict = num_cards
        for platform in num_cards_dict.keys():
            if platform not in res:
                raise ValueError(f"Platform '{platform}' in num_cards but not in res.")
        for platform in res.keys():
            if platform not in num_cards_dict:
                num_cards_dict[platform] = 1

    all_marks: list[pytest.MarkDecorator] = []
    for platform, resource in res.items():
        cards = num_cards_dict[platform]
        if platform in ("cuda", "rocm", "xpu"):
            marks = gpu_marks(res=resource, num_cards=cards)
        elif platform == "musa":
            marks = musa_marks(res=resource, num_cards=cards)
        elif platform == "npu":
            marks = npu_marks(res=resource, num_cards=cards)
        else:
            raise ValueError(f"Unsupported platform: {platform}")
        all_marks.extend(marks)
    return all_marks


def hardware_test(*, res: dict[str, str], num_cards: int | dict[str, int] = 1):
    all_marks = hardware_marks(res=res, num_cards=num_cards)

    def wrapper(f):
        func = f
        for mark in reversed(all_marks):
            func = mark(func)
        return func

    return wrapper
