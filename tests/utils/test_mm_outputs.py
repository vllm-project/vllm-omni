import torch

from vllm_omni.utils.mm_outputs import to_payload_element


def test_to_payload_element_clones_request_invariant_tensors_by_default():
    tensor = torch.tensor([[1.0, 2.0]])

    result = to_payload_element(tensor, idx=0, start=0, end=1, seq_len=2)

    assert torch.equal(result, tensor)
    assert result.data_ptr() != tensor.data_ptr()


def test_to_payload_element_can_reuse_request_invariant_tensors_for_send_only_payloads():
    tensor = torch.tensor([[1.0, 2.0]])

    result = to_payload_element(tensor, idx=0, start=0, end=1, seq_len=2, clone_tensors=False)

    assert result is tensor


def test_to_payload_element_propagates_clone_policy_through_nested_payloads():
    tensor = torch.tensor([[1.0, 2.0]])
    payload = {"embed": {"tts_bos": [tensor]}}

    result = to_payload_element(payload, idx=0, start=0, end=1, clone_tensors=False)

    assert result["embed"]["tts_bos"] is tensor
