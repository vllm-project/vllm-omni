# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA filtering of the ten RVQ heads, including single-request broadcasts."""

from types import SimpleNamespace

import pytest
import torch

from tests.helpers.mark import hardware_marks
from tests.model_executor.models.test_zonos3_sampler import add_sampling, codes, make_state, run
from vllm_omni.model_executor.models.zonos3.model_state import Zonos3MiniModelState
from vllm_omni.platforms import current_omni_platform

pytestmark = [
    pytest.mark.core_model,
    *hardware_marks(res={"cuda": "L4"}),
    pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="Requires CUDA top-k/top-p kernels"),
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_gpu_determined_tokens_and_owned_snapshots(dtype):
    from vllm import SamplingParams

    state = make_state(4, "cuda", dtype)
    singles = [make_state(1, "cuda", dtype) for _ in range(4)]
    for slot, single in enumerate(singles):
        params = SamplingParams(temperature=0.7, top_k=5, top_p=0.85, seed=100 + slot)
        add_sampling(state, slot, params)
        add_sampling(single, 0, params)
        single.apply_staged_writes()
    state.apply_staged_writes()
    previous, saved = None, None
    for step in range(4):
        hidden = torch.randn(4, 16, device="cuda", dtype=dtype)
        order = [3, 2, 1, 0] if step % 2 else [0, 1, 2, 3]
        output, snapshot = run(state, order, hidden[order])
        assert output.num_sampled.tolist() == [1] * 4
        for row, slot in enumerate(order):
            _, expected = run(singles[slot], [0], hidden[slot : slot + 1])
            assert torch.equal(codes(snapshot)[row], codes(expected)[0])
        if previous is not None:
            assert torch.equal(codes(previous), saved)
        previous, saved = snapshot, codes(snapshot).clone()


@pytest.mark.parametrize("width", [0, 96])
@torch.inference_mode()
def test_fused_rvq_draw_matches_native_filter_and_rng(width):
    """Release vocab size; mixed settings and reordered slots use the same noise stream."""
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

    from vllm_omni.model_executor.models.zonos3.sampling_kernels import sample_rvq
    from vllm_omni.model_executor.models.zonos3.sampling_state import Zonos3ARSamplingState

    torch.manual_seed(104)
    batch, vocab, ring = 32, 4098, 128
    slots = torch.randperm(batch, device="cuda")
    state = Zonos3ARSamplingState(batch, 4096, 4097, "cuda")
    state.position.copy_(torch.arange(batch, device="cuda") * 4)
    history = torch.randint(-1, 4096, (batch, 10, ring), device="cuda")
    values = {
        "temperature": torch.tensor([0.0, 0.7, 1.0, 1.2] * 8, device="cuda"),
        "top_k": torch.tensor([1, 80, 4096, 5] * 8, device="cuda", dtype=torch.int32),
        "top_p": torch.tensor([1.0, 0.69, 0.9, 1.0] * 8, device="cuda"),
        "min_p": torch.tensor([0.0, 0.01, 0.1, 0.0] * 8, device="cuda"),
        "repetition_penalty": torch.tensor([1.0, 1.2, 2.4, 1.5] * 8, device="cuda"),
        "repetition_window": torch.full((batch,), width, device="cuda", dtype=torch.int32),
        "repetition_codebooks": torch.tensor([0, 6, 10, 4] * 8, device="cuda", dtype=torch.int32),
        "min_frames": torch.tensor([0, 20, 100, 0] * 8, device="cuda", dtype=torch.int32),
        "seed": torch.arange(batch, device="cuda", dtype=torch.int64) * 1009 - 20,
    }
    settings = SimpleNamespace(**{k: SimpleNamespace(gpu=v) for k, v in values.items()})
    owner = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(codebook_size=4096, audio_vocab=vocab, eoa_id=4096)),
        sampling_state=SimpleNamespace(position=state.position.cpu(), columns=state.columns.cpu()),
        history=history.cpu(),
        max_model_len=ring,
        _cpu_setting=lambda name, selected: values[name].cpu().index_select(0, selected),
    )
    for _ in range(3):
        logits = torch.randn(batch, 10, vocab, device="cuda") * 4
        owner.sampling_state.position.copy_(state.position.cpu())
        probs = Zonos3MiniModelState._cpu_probabilities(owner, logits.cpu(), slots.cpu(), width).cuda()
        position = state.position[slots, None] * 10 + state.columns
        expected = gumbel_sample(
            probs.log().view(-1, vocab),
            slots.repeat_interleave(10),
            values["temperature"],
            values["seed"],
            position.flatten(),
            apply_temperature=False,
            is_drafting=False,
        ).view(batch, 10)
        actual = sample_rvq(logits, slots, state, settings, history, width, 4096, 4096)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        state.position.add_(1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_fused_stop_state_matches_cpu_through_drain_and_slot_reuse(dtype):
    from vllm_omni.model_executor.models.zonos3.modeling_zonos3_mini import MultiEmbedding
    from vllm_omni.model_executor.models.zonos3.sampling_kernels import commit_feedback
    from vllm_omni.model_executor.models.zonos3.sampling_state import Zonos3ARSamplingState

    cpu = Zonos3ARSamplingState(4, 4096, 4097, "cpu")
    gpu = Zonos3ARSamplingState(4, 4096, 4097, "cuda")
    slots = torch.tensor([3, 0, 2, 1])
    model = SimpleNamespace(
        config=SimpleNamespace(text_vocab=20),
        multi_embedder=MultiEmbedding(SimpleNamespace(n_codebooks=10, audio_vocab=4098, dim=16, text_rows=21))
        .cuda()
        .to(dtype),
    )
    history = torch.full((4, 10, 128), -1, dtype=torch.int64, device="cuda")
    expected_history = history.clone()
    feedback = torch.zeros((4, 16), dtype=dtype, device="cuda")
    expected_feedback = feedback.clone()
    for state in (cpu, gpu):
        device = state.position.device
        state.reset(
            slots.to(device),
            torch.tensor([True, False, True, True], device=device),
            torch.tensor([False, False, True, False], device=device),
            torch.tensor([False, False, False, True], device=device),
        )
    for step in range(17):
        sampled = torch.full((4, 10), 5, dtype=torch.int64)
        if step == 2:
            sampled[:, 4] = 4096
        if step == 4:
            sampled[:, 0] = 4096
        active = torch.tensor([True, step > 0, True, step != 3])
        expected = cpu.accept(slots, sampled, active)
        actual = gpu.accept(slots.cuda(), sampled.cuda(), active.cuda())
        commit_feedback(model, slots.cuda(), actual, gpu, history, feedback)
        embeddings = model.multi_embedder(
            torch.cat((expected.codes.cuda(), torch.full((4, 1), 20, dtype=torch.int64, device="cuda")), -1)
        )
        for row, slot in enumerate(slots.tolist()):
            if expected.active[row]:
                expected_history[slot, :, int(cpu.position[slot] - 1) % 128] = expected.codes[row].cuda()
                expected_feedback[slot] = embeddings[row]
        torch.testing.assert_close(history, expected_history, atol=0, rtol=0)
        torch.testing.assert_close(feedback, expected_feedback, atol=0, rtol=0)
        for field in ("codes", "logical_token_ids", "active", "finished"):
            torch.testing.assert_close(getattr(actual, field).cpu(), getattr(expected, field), atol=0, rtol=0)
        for field in ("position", "eoa_frame", "drain_remaining"):
            torch.testing.assert_close(getattr(gpu, field).cpu(), getattr(cpu, field), atol=0, rtol=0)
    for state in (cpu, gpu):
        state.reset(slots.to(state.position.device), torch.zeros(4, dtype=torch.bool, device=state.position.device))
    torch.testing.assert_close(gpu.position.cpu(), cpu.position, atol=0, rtol=0)
