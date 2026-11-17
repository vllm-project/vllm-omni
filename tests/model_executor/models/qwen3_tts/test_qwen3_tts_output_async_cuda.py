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
from vllm_omni.worker_v2.omni_ar_model_runner import _async_copy_mm

pytestmark = [pytest.mark.core_model]


@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=1)
@pytest.mark.parametrize(
    ("eager_frames_active", "validity_device"),
    [(False, "cpu"), (False, "cuda"), (True, "cpu")],
    ids=["buffered-cpu", "buffered-cuda", "v2-eager"],
)
@torch.inference_mode()
def test_output_validity_preserves_values_without_waiting_for_forward(eager_frames_active, validity_device):
    talker = Qwen3TTSTalkerForConditionalGeneration.__new__(Qwen3TTSTalkerForConditionalGeneration)
    torch.nn.Module.__init__(talker)
    talker.vllm_config = SimpleNamespace(model_config=SimpleNamespace(async_chunk=True))
    talker.talker_config = SimpleNamespace(num_code_groups=16)
    talker.eager_frames_active = eager_frames_active
    talker.stream_decoder = None
    hidden = torch.zeros((3, 8), device="cuda")
    infos = [
        {
            "codes": {"audio": torch.zeros((frames, 16), device="cuda", dtype=torch.long)},
            "meta": {"codec_frame_valid": torch.tensor(valid, device=validity_device)},
        }
        for frames, valid in [(2, True), (1, False)]
    ]
    output_kwargs = {"model_intermediate_buffer": infos, "request_token_spans": [(0, 2), (2, 3)]}
    # Warm allocation paths before checking the stream-ordering contract.
    talker.make_omni_output(hidden, **output_kwargs)
    copy_stream = torch.cuda.Stream()
    copy_complete = torch.cuda.Event()
    torch.accelerator.synchronize()

    preceding_forward = torch.cuda.Event()
    torch.cuda._sleep(500_000_000)
    preceding_forward.record()
    output = talker.make_omni_output(hidden, **output_kwargs)
    waited_for_forward = preceding_forward.query()
    validity = output.multimodal_outputs["meta"]["codec_frame_valid"]
    # Use the same stream dependency and D2H helper as the V2 async output.
    copy_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(copy_stream):
        snapshot = _async_copy_mm(output.multimodal_outputs, 3, copy_stream=copy_stream, pin_memory=True)
        copy_complete.record()
    copy_complete.synchronize()

    assert not waited_for_forward, "output construction synchronized the preceding forward"
    assert validity.device.type == "cuda"
    assert validity.dtype == torch.int8
    cpu_validity = snapshot["meta"]["codec_frame_valid"]
    assert cpu_validity.device.type == "cpu"
    # Eager V2 publishes placeholders; post_sample fills the sampled rows.
    assert cpu_validity.tolist() == ([0, 0, 0] if eager_frames_active else [1, 1, 0])
    torch.testing.assert_close(snapshot["codes"]["audio"], torch.zeros((3, 16), dtype=torch.long))
    assert all(info["meta"]["codec_frame_valid"].device.type == validity_device for info in infos)
