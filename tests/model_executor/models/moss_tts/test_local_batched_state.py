# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_talker import MossTTSLocalTalkerForGeneration
from vllm_omni.worker_v2.model_states.intermediate_buffer import OmniIntermediateBuffer


def model(device):
    m = MossTTSLocalTalkerForGeneration.__new__(MossTTSLocalTalkerForGeneration)
    nn.Module.__init__(m)
    m.hidden_size = 8
    m.n_vq = 2
    m.audio_pad_token_id = 16
    m.audio_vocab_size = 16
    m.model = nn.Module()
    m.model.embed_tokens = nn.Embedding(32, 8, device=device)
    m._stacked_audio_emb_w = torch.randn(2, 16, 8, device=device)
    return m


@pytest.mark.parametrize(
    "device", [pytest.param("cpu", marks=pytest.mark.cpu), pytest.param("cuda", marks=pytest.mark.cuda)]
)
@pytest.mark.parametrize("active", [[True, True, True], [False, False, False], [True, False, True]])
def test_decode_controls_and_hidden_rows_match_scalar_preprocess(device, active):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    m = model(device)
    ids = torch.tensor([3, 4, 5], device=device)
    infos = [
        {"hidden_states": {"last": torch.randn(8, device=device)}, "audio_state": {"is_stopping": not flag}}
        for flag in active
    ]
    _, embeds, hidden, controls, _ = m.preprocess_decode_batch(input_ids=ids, req_infos=infos)
    for i, info in enumerate(infos):
        _, expected_embeds, updates = m.preprocess(ids[i : i + 1], None, **info)
        expected_hidden, expected_controls = updates["mtp_inputs"]
        torch.testing.assert_close(embeds[i : i + 1], expected_embeds)
        torch.testing.assert_close(hidden[i : i + 1], expected_hidden)
        torch.testing.assert_close(controls[i : i + 1], expected_controls)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_owned_postprocess_survives_graph_replay_and_reordered_request_slots():
    m = model("cuda")
    source = torch.arange(48, device="cuda", dtype=torch.float32).reshape(6, 8)
    static = torch.empty_like(source)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static.copy_(source)
    graph.replay()
    indices = torch.tensor([1, 4, 5], device="cuda")  # mixed spans [2, 3, 1]
    key, owned = m.postprocess_batch_mrv2(hidden_states=static, last_token_indices=indices)
    state = OmniIntermediateBuffer(8)
    state.update_owned_gpu_tensor_rows([6, 1, 3], key, owned.tensor, keepdim=False)
    source.add_(1000)
    graph.replay()
    for request_slot, token_row in zip([6, 1, 3], [1, 4, 5]):
        expected = torch.arange(token_row * 8, token_row * 8 + 8, device="cuda", dtype=torch.float32)
        torch.testing.assert_close(state.buffers[request_slot]["hidden_states"]["last"], expected)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_resident_reference_codes_preserve_chunked_prefill_embeddings():
    m = model("cuda")
    codes = torch.randint(0, 17, (6, 2))
    ids = torch.arange(6, device="cuda")
    actual = []
    for ref in [codes, codes.cuda()]:
        info = {"codes": {"ref": ref}, "_omni_is_prefill": True}
        pieces = []
        for start, end in [(0, 2), (2, 5), (5, 6)]:
            _, embeds, update = m.preprocess(ids[start:end], None, **info)
            pieces.append(embeds)
            info.update(update)
        actual.append(torch.cat(pieces))
    torch.testing.assert_close(actual[0], actual[1], rtol=0, atol=0)


@pytest.mark.cpu
def test_native_decode_reuses_prepared_token_embeddings(mocker):
    m = model("cpu")
    ids = torch.tensor([3, 4])
    infos = [{"hidden_states": {"last": torch.randn(8)}, "audio_state": {"is_stopping": False}} for _ in range(2)]
    expected = m.preprocess_decode_batch(input_ids=ids, req_infos=infos)
    embeddings = m.model.embed_tokens(ids)
    mocker.patch.object(m.model.embed_tokens, "forward", side_effect=AssertionError("duplicate embedding lookup"))
    actual = m.preprocess_decode_batch(input_ids=ids, req_infos=infos, input_embeds=embeddings)
    assert actual[1] is embeddings
    for a, b in zip(actual[:4], expected[:4]):
        torch.testing.assert_close(a, b)


pytestmark = [pytest.mark.core_model]
