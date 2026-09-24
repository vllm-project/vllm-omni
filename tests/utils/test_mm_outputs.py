import pytest
import torch

from vllm_omni.utils import mm_outputs as mm_outputs_mod
from vllm_omni.utils.mm_outputs import build_mm_cpu

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

TOTAL_TOKENS = 5
NUM_TOKEN_ALIGNED_TENSORS = 3  # codes.audio, meta.ref_code_len, meta.codec_streaming


def make_talker_passthrough_payload(num_requests: int, device: str = "cpu") -> dict:
    """The passthrough set qwen3-tts actually hands to build_mm_cpu each decode
    step (qwen3_tts_talker.make_omni_output): a nested codes/meta dict whose
    ``codes.ref`` holds one tensor per request, alongside 1D int token-aligned
    metadata. Under prefix caching only ``codes.audio`` is 2D and token-aligned,
    so it is the sole cache key and this whole structure is passthrough.
    """
    return {
        "codes": {
            "audio": torch.rand(TOTAL_TOKENS, 4, device=device),
            "ref": [torch.arange(idx + 1, dtype=torch.long, device=device) for idx in range(num_requests)],
        },
        "meta": {
            "ref_code_len": torch.full((TOTAL_TOKENS,), 7, dtype=torch.int32, device=device),
            "codec_streaming": torch.ones(TOTAL_TOKENS, dtype=torch.int8, device=device),
        },
    }


def iter_tensors(value):
    """Yield every tensor leaf, recursing the same dict/list nesting _to_cpu does."""
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_tensors(item)
    elif isinstance(value, list):
        for item in value:
            yield from iter_tensors(item)


@pytest.mark.parametrize("num_requests", [1, 3])
def test_build_mm_cpu_d2h_count_grows_per_request(num_requests, mocker):
    """Pins the pre-W2 cost: build_mm_cpu issues one per-tensor copy for each
    token-aligned payload tensor plus one more for every request's ``codes.ref``
    entry, so the transfer count grows linearly with batch size. That growth is
    what RFC #4855 W2 sets out to batch away.

    Counting tensor leaves rather than dict keys is deliberate: a flat dict of
    K tensors makes ``call_count == K`` true by construction, since build_mm_cpu
    is a loop over items. Only the per-request list can move this number.
    """
    spy = mocker.spy(mm_outputs_mod, "_to_cpu")

    result = build_mm_cpu(multimodal_outputs=make_talker_passthrough_payload(num_requests))

    tensor_copies = [call for call in spy.call_args_list if isinstance(call.args[0], torch.Tensor)]
    assert len(tensor_copies) == NUM_TOKEN_ALIGNED_TENSORS + num_requests
    assert len(result["codes"]["ref"]) == num_requests


def test_build_mm_cpu_preserves_int_dtypes_of_talker_metadata():
    """ref_code_len is int32 and codec_streaming int8 at the source; a W2 that
    stages passthrough through a float buffer would corrupt both silently."""
    result = build_mm_cpu(multimodal_outputs=make_talker_passthrough_payload(3))

    assert result["meta"]["ref_code_len"].dtype == torch.int32
    assert result["meta"]["codec_streaming"].dtype == torch.int8
    assert all(ref.dtype == torch.long for ref in result["codes"]["ref"])


def test_build_mm_cpu_returns_cpu_tensors_for_every_nested_leaf():
    """The output contract: every tensor leaf, including those inside the
    ``codes.ref`` list, comes back on CPU.

    On this lane the inputs are already CPU, so this documents the contract
    rather than proving it -- ``.to("cpu")`` is a no-op here. The GPU-gated
    sibling below is the one that can actually fail.
    """
    result = build_mm_cpu(multimodal_outputs=make_talker_passthrough_payload(3))

    leaves = list(iter_tensors(result))
    assert len(leaves) == NUM_TOKEN_ALIGNED_TENSORS + 3
    assert all(leaf.device.type == "cpu" for leaf in leaves)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_build_mm_cpu_moves_a_gpu_payload_off_the_device():
    """D2H is the entire purpose of build_mm_cpu. RFC #4855 W2 lists "skip the
    CPU conversion when the consumer is the same-GPU next stage" as one of its
    three options; that option leaves every tensor on the device and no CPU-lane
    test can observe it, because there ``.to("cpu")`` is a silent no-op.
    """
    payload = make_talker_passthrough_payload(3, device="cuda")
    source_leaves = list(iter_tensors(payload))
    assert all(leaf.device.type == "cuda" for leaf in source_leaves)

    result = build_mm_cpu(multimodal_outputs=payload)

    leaves = list(iter_tensors(result))
    assert len(leaves) == len(source_leaves)
    assert all(leaf.device.type == "cpu" for leaf in leaves)


def test_build_mm_cpu_empty_input_is_a_noop(mocker):
    spy = mocker.spy(mm_outputs_mod, "_to_cpu")

    assert build_mm_cpu(multimodal_outputs={}) == {}
    spy.assert_not_called()


def test_build_mm_cpu_preserves_nested_and_non_tensor_payload_semantics():
    """_to_cpu must recurse into nested dict/list structures, force
    non-contiguous tensors to become contiguous, detach from autograd,
    and leave dtype/shape/values unchanged; non-tensor leaves (scalar,
    string) pass through untouched. Any W2 implementation (cat, deferral,
    or a new materializer) must preserve this exact semantics."""
    non_contig = torch.arange(9, dtype=torch.float32).reshape(3, 3).t()
    assert not non_contig.is_contiguous()
    grad_tensor = torch.tensor([5.0, 6.0], requires_grad=True)

    payload = {
        "nested": {"inner_tensor": non_contig, "inner_scalar": "keep-me"},
        "list_of_tensors": [torch.tensor([1.0, 2.0]), "marker", torch.tensor([3.0, 4.0])],
        "grad_key": grad_tensor,
        "scalar": 42,
    }
    result = build_mm_cpu(multimodal_outputs=payload)

    inner = result["nested"]["inner_tensor"]
    assert inner.is_contiguous()
    assert inner.dtype == non_contig.dtype
    assert inner.shape == non_contig.shape
    assert torch.equal(inner, non_contig)
    assert result["nested"]["inner_scalar"] == "keep-me"

    lst = result["list_of_tensors"]
    assert torch.equal(lst[0], torch.tensor([1.0, 2.0]))
    assert lst[1] == "marker"
    assert torch.equal(lst[2], torch.tensor([3.0, 4.0]))

    assert result["grad_key"].requires_grad is False
    assert torch.equal(result["grad_key"], grad_tensor.detach())

    assert result["scalar"] == 42
