import torch

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
    MiniCPMO45OmniForConditionalGeneration,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.platforms import current_omni_platform


class _RecordingThinker(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.hidden_size = hidden_size
        self.call: dict[str, torch.Tensor | None] = {}

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        self.call = {
            "input_ids": input_ids,
            "positions": positions,
            "inputs_embeds": inputs_embeds,
        }
        assert inputs_embeds is not None
        hidden_states = inputs_embeds.unsqueeze(0)
        return inputs_embeds, hidden_states


def _make_wrapper(hidden_size: int = 4):
    wrapper = object.__new__(MiniCPMO45OmniForConditionalGeneration)
    torch.nn.Module.__init__(wrapper)
    wrapper.model_stage = "llm"
    wrapper.thinker = _RecordingThinker(hidden_size)
    wrapper.talker = None
    return wrapper


def test_thinker_forward_preserves_mrope_positions_for_decode_graph():
    wrapper = _make_wrapper()
    input_ids = torch.tensor([7])
    positions = torch.tensor([[1], [2], [3]])
    inputs_embeds = torch.randn(1, 4)

    output = wrapper(
        input_ids=input_ids,
        positions=positions,
        inputs_embeds=inputs_embeds,
    )

    assert wrapper.thinker.call["input_ids"].shape == input_ids.shape
    assert wrapper.thinker.call["input_ids"].data_ptr() == input_ids.data_ptr()
    assert wrapper.thinker.call["positions"] is positions
    assert wrapper.thinker.call["inputs_embeds"].shape == inputs_embeds.shape
    assert wrapper.thinker.call["inputs_embeds"].data_ptr() == inputs_embeds.data_ptr()
    assert output.text_hidden_states.shape == (1, 4)
    assert output.multimodal_outputs["latent"] is output.text_hidden_states


def test_thinker_make_omni_output_restores_cudagraph_replay_tuple():
    wrapper = _make_wrapper()
    text_hidden_states = torch.randn(1, 4)
    multimodal_outputs = {"latent": text_hidden_states}
    intermediate_tensors = object()
    next_token_id = torch.tensor([7])

    output = wrapper.make_omni_output(
        (text_hidden_states, multimodal_outputs, intermediate_tensors, next_token_id)
    )

    assert isinstance(output, OmniOutput)
    assert output.text_hidden_states is text_hidden_states
    assert output.multimodal_outputs is multimodal_outputs
    assert output.intermediate_tensors is intermediate_tensors
    assert output.next_token_id is next_token_id


def test_thinker_npu_preserves_batched_mrope_positions(monkeypatch):
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: True)
    wrapper = _make_wrapper()
    input_ids = torch.tensor([[7, 8]])
    # A genuine batched M-RoPE tensor: (batch, sequence, rope-axis).
    # It must not be mistaken for an unbatched tensor that needs unwrapping.
    positions = torch.tensor([[[1, 2, 3], [4, 5, 6]]])
    inputs_embeds = torch.randn(1, 2, 4)

    wrapper(
        input_ids=input_ids,
        positions=positions,
        inputs_embeds=inputs_embeds,
    )

    assert wrapper.thinker.call["positions"] is positions
