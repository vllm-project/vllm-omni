# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Output metadata must not wait for the preceding model forward on CUDA."""

from types import SimpleNamespace

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
    Qwen3TTSTalkerForConditionalGeneration,
)
from vllm_omni.worker.gpu_ar_model_runner import _snapshot_tensor_payload_to_cpu_async

pytestmark = [pytest.mark.core_model]


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("validity_device", ["cpu", "cuda"])
@torch.inference_mode()
def test_output_validity_preserves_values_without_waiting_for_forward(validity_device):
    talker = Qwen3TTSTalkerForConditionalGeneration.__new__(Qwen3TTSTalkerForConditionalGeneration)
    talker.vllm_config = SimpleNamespace(model_config=SimpleNamespace(async_chunk=True))
    hidden = torch.zeros((3, 8), device="cuda")
    infos = [
        {
            "codes": {"audio": torch.zeros((frames, 16), device="cuda", dtype=torch.long)},
            "meta": {"codec_frame_valid": torch.tensor(valid, device=validity_device)},
        }
        for frames, valid in [(2, True), (1, False)]
    ]
    # Warm allocation paths before checking the stream-ordering contract.
    talker.make_omni_output(hidden, model_intermediate_buffer=infos)
    copy_stream = torch.cuda.Stream()
    torch.accelerator.synchronize()

    preceding_forward = torch.cuda.Event()
    torch.cuda._sleep(500_000_000)
    preceding_forward.record()
    output = talker.make_omni_output(hidden, model_intermediate_buffer=infos)
    waited_for_forward = preceding_forward.query()
    validity = output.multimodal_outputs["meta"]["codec_frame_valid"]
    snapshot = _snapshot_tensor_payload_to_cpu_async(
        output.multimodal_outputs, copy_stream=copy_stream, pin_memory=True
    )
    snapshot.wait()

    assert not waited_for_forward, "output construction synchronized the preceding forward"
    assert validity.device.type == "cuda"
    assert validity.dtype == torch.int8
    cpu_validity = snapshot.payload["meta"]["codec_frame_valid"]
    assert cpu_validity.device.type == "cpu"
    assert cpu_validity.tolist() == [1, 1, 0]
    assert all(info["meta"]["codec_frame_valid"].device.type == validity_device for info in infos)
