# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.helpers.mark import hardware_test
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="requires CUDA")]


class MelOutput(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.m_source = SimpleNamespace(l_linear=SimpleNamespace(weight=self.anchor))

    def inference(self, speech_feat, finalize):
        assert finalize
        return speech_feat, None, None


def _items(shapes, **extra):
    return [
        dict(
            token=torch.randint(0, 64, (1, length), device="cuda"),
            prompt_token=torch.randint(0, 64, (1, prompt)),
            prompt_feat=torch.randn(1, prompt * 2, 80, device="cuda"),
            embedding=torch.randn(1, 192, device="cuda"),
            **extra,
        )
        for prompt, length in shapes
    ]


PACKED_BACKENDS = ["fa2", "fa3", "flashinfer"]


def require_packed_backend(monkeypatch, backend):
    """Pin the packed Flow attention backend, skipping GPUs without it."""
    from vllm_omni.model_executor.models.cosyvoice3.runtime import cosyvoice3_packed_attention_backend

    monkeypatch.setenv("COSYVOICE3_PACKED_ATTENTION_BACKEND", backend)
    if cosyvoice3_packed_attention_backend() != backend:
        pytest.skip(f"{backend} is unavailable on this GPU")


def tiny_flow(monkeypatch, backend=None):
    import vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav as module
    from vllm_omni.transformers_utils.configs.cosyvoice3 import CosyVoice3Config

    monkeypatch.setenv("DIFFUSION_ATTENTION_BACKEND", "TORCH_SDPA")
    monkeypatch.setattr(module, "CausalHiFTGenerator", MelOutput)
    config = CosyVoice3Config()
    config.flow["pre_lookahead_layer"]["channels"] = 32
    # FlashInfer's prefill kernels need a head dim of at least 64.
    heads, dim_head = (2, 64) if backend == "flashinfer" else (4, 8)
    config.flow["decoder"]["estimator"].update(dim=32, depth=2, heads=heads, dim_head=dim_head)
    return module.CosyVoice3Code2Wav(config).cuda().eval()


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("offset", [0, 3])
@pytest.mark.parametrize("count", [3, 11])
@torch.inference_mode()
def test_full_batch_ragged_prompt_matches_independent_flow(monkeypatch, offset, count):
    model = tiny_flow(monkeypatch)
    items = _items(([(7, 17), (13, 9), (3, 31)] * 4)[:count], token_offset_tokens=offset)
    # Fix Flow noise so scheduling changes do not change the comparison input.
    monkeypatch.setattr(torch, "randn", lambda shape, **kw: torch.zeros(shape, **kw))
    expected = [model.forward(**item, n_timesteps=3) for item in items]
    actual = model.forward_batch(items, n_timesteps=3)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=3e-4, atol=3e-5)
    # Reordering unrelated requests must not move their conditioning or crop points.
    reverse = model.forward_batch(items[::-1], n_timesteps=3)
    for result, reference in zip(reverse, expected[::-1]):
        torch.testing.assert_close(result, reference, rtol=3e-4, atol=3e-5)
    assert model.forward_batch([]) == []


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("backend", PACKED_BACKENDS)
@torch.inference_mode()
def test_packed_full_response_preserves_request_isolation(monkeypatch, backend):
    require_packed_backend(monkeypatch, backend)
    monkeypatch.setenv("COSYVOICE3_FULL_RESPONSE_OPTIMIZATIONS", "1")
    model = tiny_flow(monkeypatch, backend).bfloat16()
    model.hift.upsample_rates = [1]
    model.hift.istft_params = {"hop_len": 1}
    items = _items([(7, 17), (13, 9), (3, 31)])
    monkeypatch.setattr(torch, "randn", lambda shape, **kw: torch.zeros(shape, **kw))
    expected = [model.forward(**item, n_timesteps=3) for item in items]
    actual = model.forward_batch(items, n_timesteps=3)
    for output, reference in zip(actual, expected):
        torch.testing.assert_close(output, reference, rtol=0.04, atol=0.04)
    # A different batch order must not leak the other requests' conditioning.
    actual = model.forward_batch(items[::-1], n_timesteps=3)
    for output, reference in zip(actual, expected[::-1]):
        torch.testing.assert_close(output, reference, rtol=0.04, atol=0.04)


@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_packed_stream_noise_growth_preserves_prefix_and_global_rng():
    from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav import CosyVoice3Code2Wav

    model = CosyVoice3Code2Wav.__new__(CosyVoice3Code2Wav)
    nn.Module.__init__(model)
    like = torch.empty(1, 80, 1, dtype=torch.bfloat16)
    rng = torch.random.get_rng_state().clone()
    first = model._stream_position_noise(97, like).clone()
    grown = model._stream_position_noise(30001, like).clone()
    torch.testing.assert_close(grown[..., :97], first, rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    direct = CosyVoice3Code2Wav.__new__(CosyVoice3Code2Wav)
    nn.Module.__init__(direct)
    torch.testing.assert_close(direct._stream_position_noise(30001, like), grown, rtol=0, atol=0)
    assert model._stream_position_noise(0, like).shape == (1, 80, 0)


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("backend", PACKED_BACKENDS)
@pytest.mark.parametrize("chunk_size", [None, 50])
@torch.inference_mode()
def test_packed_attention_matches_row_reference(monkeypatch, backend, chunk_size):
    require_packed_backend(monkeypatch, backend)
    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.packed_dit import (
        attention_uses_pages,
        build_row_attention,
        pack_rows,
        ragged_attention_tensor_geometry,
        row_segments,
    )
    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.packed_flashinfer import FlashInferRowPlan, activate

    # Row starts that are not multiples of 16 exercise FA2's page alignment;
    # full context exercises FA2's unpaged varlen and empty bucket slots.
    lengths = [67, 0, 103]
    rows = pack_rows(lengths, torch.device("cuda"))
    attention = build_row_attention(rows, heads=2, head_dim=64, backend=backend, chunk_size=chunk_size)
    assert attention.paged is attention_uses_pages(backend, chunk_size is not None)
    if backend == "flashinfer":
        plan = FlashInferRowPlan(
            rows.row_ids.device, heads=2, head_dim=64, dtype=torch.bfloat16, chunked=chunk_size is not None
        )
        plan.plan(lengths, *row_segments(lengths, chunk_size))
        activate(plan)
    generator = torch.Generator(device="cuda").manual_seed(7)
    q, k, v = [
        torch.randn(sum(lengths), 2, 64, device="cuda", dtype=torch.bfloat16, generator=generator) for _ in range(3)
    ]
    actual = ragged_attention_tensor_geometry(
        q.reshape(1, -1, 128), k.reshape(1, -1, 128), v.reshape(1, -1, 128), attention, attention.max_seqlen_q
    ).reshape(-1, 2, 64)
    start = 0
    for length in lengths:
        if length == 0:
            continue
        position = torch.arange(length, device="cuda")
        span = chunk_size or length
        mask = position[None, :] < ((position[:, None] // span + 1) * span)
        expected = torch.nn.functional.scaled_dot_product_attention(
            q[start : start + length].transpose(0, 1).float(),
            k[start : start + length].transpose(0, 1).float(),
            v[start : start + length].transpose(0, 1).float(),
            attn_mask=mask,
        ).transpose(0, 1)
        torch.testing.assert_close(actual[start : start + length].float(), expected, rtol=0.03, atol=0.01)
        start += length


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("graph", [False, True])
def test_flashinfer_chunk_plan_hands_kv_indices_over_on_the_device(monkeypatch, graph):
    require_packed_backend(monkeypatch, "flashinfer")
    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.packed_dit import row_segments
    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.packed_flashinfer import (
        FlashInferRowPlan,
        paged_index_capacity,
    )

    lengths = [67, 0, 103]
    segments = row_segments(lengths, 50)
    capacity = (len(segments[1]), paged_index_capacity(sum(lengths), 50)) if graph else None
    plan = FlashInferRowPlan(
        torch.device("cuda"), heads=2, head_dim=64, dtype=torch.bfloat16, chunked=True, graph_capacity=capacity
    )
    planned = []
    wrapper_plan = plan.wrapper.plan

    def record(*args, **kwargs):
        planned.append(args[2])
        return wrapper_plan(*args, **kwargs)

    monkeypatch.setattr(plan.wrapper, "plan", record)
    plan.plan(lengths, *segments)
    # FlashInfer copies host indices into a graph's buffer synchronously.
    assert planned[0].is_cuda and planned[0].dtype == torch.int32


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("backend", PACKED_BACKENDS)
@torch.inference_mode()
def test_packed_stream_mixed_finalization_and_ragged_requests_stay_aligned(monkeypatch, backend):
    require_packed_backend(monkeypatch, backend)
    monkeypatch.setenv("COSYVOICE3_FULL_RESPONSE_OPTIMIZATIONS", "0")
    monkeypatch.setenv("COSYVOICE3_PACKED_STREAMING", "1")
    model = tiny_flow(monkeypatch, backend).bfloat16()
    monkeypatch.setattr(
        model, "_stream_hift_from_feat", lambda mel, cache_state, finalize: (mel, None if finalize else cache_state)
    )
    items = []
    for index, (prompt, generated) in enumerate(([(7, 17), (13, 9), (3, 31)] * 6)[:17]):
        items.append(
            dict(
                token=torch.randint(0, 64, (1, generated), device="cuda"),
                prompt_token=torch.randint(0, 64, (1, prompt)),
                prompt_feat=torch.randn(1, prompt * 2, 80, device="cuda"),
                embedding=torch.randn(1, 192, device="cuda"),
                token_offset_tokens=3,
                finalize=index % 3 == 0,
                cache_state={"request_marker": torch.tensor(index)},
            )
        )
    expected = [model.forward_streaming_batch([item], n_timesteps=3)[0] for item in items]
    for ordered, reference in [(items, expected), (items[::-1], expected[::-1])]:
        actual = model.forward_streaming_batch(ordered, n_timesteps=3)
        for (audio, state), (ref_audio, ref_state), item in zip(actual, reference, ordered):
            torch.testing.assert_close(audio, ref_audio, rtol=0.04, atol=0.04)
            assert state is ref_state
            assert state is (None if item["finalize"] else item["cache_state"])
    assert model.forward_streaming_batch([]) == []
    with pytest.raises(ValueError, match="mixed finalization"):
        model.forward_batch(items, n_timesteps=3, stream_items=True)
    monkeypatch.setenv("COSYVOICE3_PACKED_STREAMING", "0")
    with pytest.raises(ValueError, match="COSYVOICE3_PACKED_STREAMING"):
        model.forward_batch(items[:1], n_timesteps=3, stream_items=True)
