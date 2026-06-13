import torch

from vllm_omni.diffusion.models.ideogram4.ideogram4_transformer import (
    _pick_nf4_dequant_device,
)


def test_nf4_prefers_cuda_when_weight_is_still_on_cpu():
    weight = torch.zeros((8, 1), dtype=torch.uint8)

    target = _pick_nf4_dequant_device(weight)

    assert target.type == "cuda"
