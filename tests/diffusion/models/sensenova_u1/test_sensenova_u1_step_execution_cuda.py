# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA-only step-execution sampling test.

Collected by the ready ``Diffusion · Model Test`` job
(``tests/diffusion/models/ -m 'core_model and cuda'``), not by the CPU simple
lane. It lives outside the CPU-marked step-execution module because
``check_test_marks`` rejects combining an inherited ``pytest.mark.cpu`` with a
``hardware_test`` accelerator mark on one item; the helper pipeline is shared
with the CPU module.
"""

import pytest
import torch

from tests.diffusion.models.sensenova_u1.test_sensenova_u1_step_execution import VOCAB, _pipeline
from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_a_generator_on_another_device_is_replaced_by_one_on_the_logits_device():
    # generator_device="cpu" gives the runner's generator a CPU device
    # while the logits it samples from sit on the GPU.
    pipe = _pipeline(flat=True)
    flat = torch.zeros(1, 1, VOCAB, device="cuda")
    cursor = pipe._begin_text(
        flat,
        "kv",
        0,
        max_tokens=4,
        do_sample=True,
        temperature=1.0,
        seed=7,
        request_generator=torch.Generator("cpu").manual_seed(7),
    )
    assert cursor.generator.device == flat.device
    probs = torch.softmax(cursor.logits / cursor.temperature, dim=-1)
    torch.multinomial(probs, num_samples=1, generator=cursor.generator)
