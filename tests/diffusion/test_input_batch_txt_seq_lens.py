import pytest
import torch

from vllm_omni.diffusion.worker.input_batch import _prepare_request_prompt_field
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_prepare_request_prompt_field_refreshes_txt_seq_lens_after_padding():
    state = StepRequestState(
        request_id="req-0",
        sampling=OmniDiffusionSamplingParams(),
        prompt_embeds=torch.randn(1, 10, 4),
        prompt_embeds_mask=torch.ones(1, 10, dtype=torch.bool),
        txt_seq_lens=[10],
    )

    embeds, mask = _prepare_request_prompt_field(
        state,
        embeds_attr="prompt_embeds",
        mask_attr="prompt_embeds_mask",
        seq_lens_attr="txt_seq_lens",
        target_seq_len=32,
    )

    assert embeds.shape == (1, 32, 4)
    assert mask is not None and mask.shape == (1, 32)
    assert state.txt_seq_lens == [32]
    assert mask.sum(dim=1).tolist() == [10]
