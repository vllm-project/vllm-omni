# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Device-resident code2wav outputs through the generation runner.

Code2wav models return CUDA tensors and the runner owns the D2H transfer
(``_HostCopyBatch``); the sibling CPU tests stub outputs with host tensors,
so the pinned-copy path these models rely on is exercised here.
"""

import pytest
import torch

from tests.helpers.mark import hardware_test
from tests.worker.test_gpu_generation_model_runner import _make_runner
from vllm_omni.platforms import current_omni_platform
from vllm_omni.worker.gpu_generation_model_runner import GPUGenerationModelRunner, _HostCopyBatch

pytestmark = [pytest.mark.core_model]

_requires_cuda = pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="needs CUDA")


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@_requires_cuda
def test_host_copy_batch_pinned_gpu_tensors():
    """Pinned copies preserve values/dtype and land on the host after wait()."""
    device = torch.device("cuda")
    batch = _HostCopyBatch(pin_memory=True)
    sources = [
        torch.randn(1, 16000, device=device),
        torch.randn(4, 2, device=device).t(),  # non-contiguous
        torch.randn(1, 80, 64, device=device, dtype=torch.float32),
    ]
    hosts = [batch.copy(t) for t in sources]
    batch.wait()
    for host, source in zip(hosts, sources):
        assert host.device.type == "cpu"
        assert host.dtype == source.dtype
        assert host.is_contiguous()
        assert torch.equal(host, source.detach().cpu())


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@_requires_cuda
def test_sample_tokens_moves_gpu_outputs_to_host():
    """The payload shapes the code2wav models emit all reach the host intact."""
    device = torch.device("cuda")
    waveform = torch.randn(1, 16000, device=device)
    sr = torch.tensor(22050, dtype=torch.int32)  # CPU, as the models emit it

    runner = _make_runner({"audio": waveform, "sr": sr})
    runner.device = device
    output = GPUGenerationModelRunner.sample_tokens(runner)

    assert len(output.multimodal_outputs) == 1
    payload = output.multimodal_outputs[0]
    audio = payload["audio"]
    assert audio.device.type == "cpu"
    assert torch.equal(audio, waveform.detach().cpu())
    # CPU-valued fields pass through unchanged.
    assert payload["sr"].device.type == "cpu" and payload["sr"].item() == 22050


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@_requires_cuda
def test_sample_tokens_list_of_gpu_tensors():
    """Per-request waveform lists (higgs/nemotron shape) copy per item."""
    device = torch.device("cuda")
    waves = [torch.randn(2048 * (i + 1), device=device) for i in range(2)]

    runner = _make_runner(waves, num_reqs=2)
    runner.device = device
    output = GPUGenerationModelRunner.sample_tokens(runner)

    assert len(output.multimodal_outputs) == 2
    for payload, wave in zip(output.multimodal_outputs, waves):
        host = payload["model_outputs"]
        assert host.device.type == "cpu"
        assert torch.equal(host, wave.detach().cpu())
