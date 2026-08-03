import pytest
import torch

from vllm_omni.model_executor.models.qwen3_tts.cuda_graph_decoder_wrapper import (
    CUDAGraphDecoderWrapper,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("after", "expected"),
    [
        ((11, 13, 17), (11, 13)),
        (None, None),
    ],
)
def test_warmup_retains_post_warmup_allocator_values(monkeypatch, after, expected):
    wrapper = CUDAGraphDecoderWrapper(
        decoder=torch.nn.Identity(),
        capture_sizes=[1],
        num_quantizers=1,
    )
    memory_stats = iter([(1, 2, 3), after])
    original_empty = torch.empty

    monkeypatch.setattr(wrapper, "_get_cuda_memory_stats", lambda _: next(memory_stats))
    monkeypatch.setattr(wrapper, "_capture", lambda *_: None)
    monkeypatch.setattr(
        torch,
        "zeros",
        lambda *size, dtype, device: original_empty(*size, dtype=dtype).zero_(),
    )
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda _: None)

    wrapper.warmup(torch.device("cuda"))

    assert wrapper.post_warmup_memory_stats == expected
