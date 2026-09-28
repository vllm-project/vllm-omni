# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import numpy as np
import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_talker import MossTTSLocalTalkerForGeneration
from vllm_omni.model_executor.models.output_templates import RequestBatchTensor
from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner, _async_copy_mm_value, _slice_pooler_value
from vllm_omni.worker_v2.output_snapshot import pack_output_snapshot


def model():
    m = MossTTSLocalTalkerForGeneration.__new__(MossTTSLocalTalkerForGeneration)
    nn.Module.__init__(m)
    m.n_vq = 2
    m.audio_pad_token_id = 16
    m.audio_assistant_slot_token_id = 7
    m.im_end_token_id = 9
    m.text_vocab_size = 32
    return m


@pytest.mark.parametrize(
    "device", [pytest.param("cpu", marks=pytest.mark.cpu), pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_mixed_prefill_and_reordered_decode_preserve_codes_and_stop_tokens(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    m = model()
    codes = torch.tensor([[16, 16], [1, 3]], device=device)
    m.consume_mtp_batch_mrv2(req_ids=["d", "b"], codes=codes)
    codes.fill_(8)  # producer graph storage may already be reused
    hidden = torch.zeros(8, 4, device=device)
    output = m.make_omni_output(hidden, model_intermediate_buffer=[{"req_id": k} for k in ["a", "b", "c", "d"]])
    logits = m.compute_logits(torch.zeros(4, 4, device=device))
    assert logits.argmax(-1).tolist() == [7, 7, 7, 9]
    assert m._mrv2_mtp_output is None
    snapshot = pack_output_snapshot(output.multimodal_outputs, {}, max_buckets=8)
    cpu = snapshot.copy_to_cpu(lambda tensor: tensor.cpu().clone())
    inter, client = OmniARModelRunner._build_async_chunk_outputs_from_mm(
        cpu, np.array([0, 2, 3, 6, 7]), np.array([2, 1, 3, 1]), 4, 7, 8
    )
    assert client is None
    assert [item["codes.audio"].tolist() for item in inter] == [[[16, 16]], [[1, 3]], [[16, 16]], [[16, 16]]]
    # A following all-prefill batch must not repeat the preceding audio.
    next_output = m.make_omni_output(hidden, model_intermediate_buffer=[{"req_id": "new"}])
    assert not next_output.multimodal_outputs


@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.cpu
def test_explicit_request_axis_ignores_coinciding_token_axis_and_owns_snapshots(keepdim):
    source = torch.arange(12).view(3, 4)
    payload = {"codes": {"audio": RequestBatchTensor(source, keepdim)}}
    slots: dict[int, dict] = {}
    snapshot = pack_output_snapshot(payload, slots, max_buckets=8)
    source.fill_(99)
    cpu = snapshot.copy_to_cpu(lambda tensor: tensor.clone())
    # Reuse the device snapshot slot before consuming the preceding host copy.
    pack_output_snapshot(payload, slots, max_buckets=8)
    wrapped = cpu["codes"]["audio"]
    assert isinstance(wrapped, RequestBatchTensor)
    actual = _slice_pooler_value(wrapped, req_index=2, start=0, end=1, total_tokens=3, padded_total_tokens=4)
    expected = torch.tensor([[8, 9, 10, 11]]) if keepdim else torch.tensor([8, 9, 10, 11])
    torch.testing.assert_close(actual, expected)
    copied = _async_copy_mm_value(wrapped)
    assert isinstance(copied, RequestBatchTensor) and copied.keepdim == keepdim


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_batch_output_survives_real_cuda_graph_replay():
    source = torch.tensor([[2, 4], [16, 16]], device="cuda")
    static = torch.empty_like(source)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static.copy_(source)
    graph.replay()
    m = model()
    m.consume_mtp_batch_mrv2(req_ids=["a", "b"], codes=static)
    source.fill_(8)
    graph.replay()
    output = m.make_omni_output(
        torch.zeros(2, 4, device="cuda"), model_intermediate_buffer=[{"req_id": "a"}, {"req_id": "b"}]
    )
    assert output.multimodal_outputs["codes"]["audio"].tensor.tolist() == [[2, 4], [16, 16]]
    assert m._batch_should_continue.tolist() == [True, False]


pytestmark = [pytest.mark.core_model]
