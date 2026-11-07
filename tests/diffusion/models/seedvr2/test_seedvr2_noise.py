# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.seedvr2.pipeline_seedvr2 import sample_noise

pytestmark = [pytest.mark.core_model]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("generator_device", ["cpu", "cuda"])
@pytest.mark.parametrize("strided", [False, True])
def test_sample_noise(generator_device: str, strided: bool) -> None:
    shape = (16, 2, 4, 6) if strided else (2, 4, 6, 16)
    condition = torch.zeros(shape, device="cuda", dtype=torch.float16)
    if strided:
        condition = condition.permute(1, 2, 3, 0)
    if generator_device == "cpu":
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(7723)
            expected = torch.randn_like(condition, device="cpu")
    else:
        expected = torch.randn(
            shape,
            generator=torch.Generator(device=generator_device).manual_seed(7723),
            device=generator_device,
            dtype=condition.dtype,
        )
        if strided:
            expected = expected.permute(1, 2, 3, 0)
    generator = torch.Generator(device=generator_device).manual_seed(7723)
    actual = sample_noise(condition, generator)
    assert actual.device == condition.device
    assert actual.dtype == condition.dtype
    assert actual.stride() == condition.stride()
    torch.testing.assert_close(actual, expected.to(condition.device), rtol=0, atol=0)
    generator.manual_seed(7723)
    torch.testing.assert_close(sample_noise(condition, generator), actual, rtol=0, atol=0)
