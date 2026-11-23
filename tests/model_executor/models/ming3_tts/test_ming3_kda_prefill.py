# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.tts, *hardware_marks(res={"cuda": "L4"}, num_cards=1)]
cuda = pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="NVIDIA CUDA required")

HEADS = 4
HEAD_DIM = 128


def _step(lengths: list[int], has_initial_state: list[bool], *, seed: int = 0):
    """Random KDA inputs for one step (decode rows first) and its GDN-style metadata."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    tokens = sum(lengths)

    def randn(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=generator, device="cuda").to(dtype)

    q, k, v = (randn(1, tokens, HEADS, HEAD_DIM) for _ in range(3))
    # Log decays within the safe-gate bound, and sigmoid betas.
    g = -5.0 * torch.rand(1, tokens, HEADS, HEAD_DIM, generator=generator, device="cuda")
    beta = torch.rand(1, tokens, HEADS, generator=generator, device="cuda")
    slots = len(lengths) + 3
    state = torch.randn(slots, HEADS, HEAD_DIM, HEAD_DIM, generator=generator, device="cuda") * 0.1
    # Scattered slots; slot 0 and the rest stay unused.
    state_indices = torch.arange(len(lengths), device="cuda", dtype=torch.int32).flip(0) + 2

    decodes = next((i for i, length in enumerate(lengths) if length > 1), len(lengths))
    query_start_loc = torch.tensor([0, *lengths], device="cuda", dtype=torch.int32).cumsum(0).to(torch.int32)
    has_initial = torch.tensor(has_initial_state, device="cuda")
    metadata = SimpleNamespace(
        num_decodes=decodes,
        non_spec_query_start_loc=query_start_loc,
        non_spec_state_indices_tensor=state_indices,
        prefill_query_start_loc=query_start_loc[decodes:] - decodes,
        prefill_state_indices=state_indices[decodes:],
        prefill_has_initial_state=has_initial[decodes:],
        chunk_indices=None,
    )
    return (q, k, v, g, beta), state, metadata, has_initial


def _fresh(inputs, rows=slice(None)):
    # chunk_kda writes its output into ``v``.
    return [x[:, rows].clone() for x in inputs]


def _chunk_all_rows(inputs, state, metadata, has_initial):
    """The previous path: every row of the step, decodes included, through the chunk kernels."""
    from vllm.third_party.flash_linear_attention.ops.kda import chunk_kda

    indices = metadata.non_spec_state_indices_tensor
    initial = torch.where(has_initial[:, None, None, None], state[indices], 0).contiguous()
    out, last = chunk_kda(
        *_fresh(inputs),
        initial_state=initial,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=metadata.non_spec_query_start_loc,
    )
    state[indices] = last
    return out


@cuda
@pytest.mark.parametrize(
    ("lengths", "has_initial_state"),
    [
        ([1, 1, 1, 130, 64, 7], [True, True, True, False, True, False]),
        ([90, 3], [False, False]),
    ],
)
def test_decode_rows_take_the_recurrent_kernel(lengths, has_initial_state):
    from vllm.third_party.flash_linear_attention.ops.kda import chunk_kda, fused_recurrent_kda

    from vllm_omni.model_executor.models.ming3_tts.kda_prefill import _recurrent_step

    inputs, state, metadata, has_initial = _step(lengths, has_initial_state)
    decodes = metadata.num_decodes
    out = torch.zeros_like(inputs[0])
    split_state = state.clone()
    _recurrent_step(*_fresh(inputs), split_state, metadata, out, object())

    # Decode rows: exactly the decode-only step's kernel, state updated in place.
    if decodes:
        decode_state = state.clone()
        decode_out, _ = fused_recurrent_kda(
            *_fresh(inputs, slice(None, decodes)),
            initial_state=decode_state,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=metadata.non_spec_query_start_loc[: decodes + 1],
            ssm_state_indices=metadata.non_spec_state_indices_tensor[:decodes],
        )
        torch.testing.assert_close(out[:, :decodes], decode_out, rtol=0, atol=0)
        decode_slots = metadata.non_spec_state_indices_tensor[:decodes].long()
        torch.testing.assert_close(split_state[decode_slots], decode_state[decode_slots], rtol=0, atol=0)

    # Prefill rows: exactly the chunk kernels on the prefill rows alone.
    prefill_slots = metadata.prefill_state_indices.long()
    initial = torch.where(metadata.prefill_has_initial_state[:, None, None, None], state[prefill_slots], 0)
    prefill_out, prefill_last = chunk_kda(
        *_fresh(inputs, slice(decodes, None)),
        initial_state=initial.contiguous(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=metadata.prefill_query_start_loc,
    )
    torch.testing.assert_close(out[:, decodes:], prefill_out, rtol=0, atol=0)
    torch.testing.assert_close(split_state[prefill_slots], prefill_last, rtol=0, atol=0)

    # Unused slots are untouched.
    used = set(metadata.non_spec_state_indices_tensor.tolist())
    for slot in range(state.shape[0]):
        if slot not in used:
            assert torch.equal(split_state[slot], state[slot])

    # Agrees with running every row through the chunk kernels, up to bf16
    # rounding: the chunk kernels keep some intermediates in bf16.
    reference_state = state.clone()
    reference_out = _chunk_all_rows(inputs, reference_state, metadata, has_initial)
    torch.testing.assert_close(out, reference_out, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(split_state, reference_state, rtol=2e-2, atol=1e-2)


@cuda
def test_later_layers_replay_the_recorded_chunk_kernels_exactly():
    from vllm_omni.model_executor.models.ming3_tts import kda_prefill

    lengths, has_initial_state = [1, 1, 1, 130, 64, 7], [True, True, True, False, True, False]
    layers = [_step(lengths, has_initial_state, seed=seed) for seed in range(4)]
    metadata = layers[0][2]

    def run_layer(inputs, state, step_metadata, step):
        out = torch.zeros_like(inputs[0])
        kda_prefill._recurrent_step(*_fresh(inputs), state, step_metadata, out, step)
        return out, state

    # One step: the first layer runs eagerly, the second records, the rest
    # replay. Layers of different KV cache groups carry equal metadata objects.
    kda_prefill._step_chunks = None
    kda_prefill._StepChunks._failed = False
    step = object()
    replayed = [
        run_layer(inputs, state.clone(), SimpleNamespace(**vars(metadata)), step) for inputs, state, _, _ in layers
    ]
    assert kda_prefill._step_chunks is not None
    assert kda_prefill._step_chunks.graph is not None
    assert kda_prefill._step_chunks.layers == len(layers)
    # Each layer alone, in a step of its own, runs eagerly.
    for (inputs, state, _, _), (out, new_state) in zip(layers, replayed, strict=True):
        want_out, want_state = run_layer(inputs, state.clone(), metadata, object())
        assert kda_prefill._step_chunks is not None
        assert kda_prefill._step_chunks.graph is None
        torch.testing.assert_close(out, want_out, rtol=0, atol=0)
        torch.testing.assert_close(new_state, want_state, rtol=0, atol=0)


@cuda
def test_step_recordings_reuse_their_memory():
    from vllm_omni.model_executor.models.ming3_tts import kda_prefill

    inputs, state, metadata, _ = _step([1, 1, 130, 64, 7], [True, True, False, True, False])
    out = torch.zeros_like(inputs[0])
    kda_prefill._step_chunks = None
    kda_prefill._StepChunks._failed = False
    reserved: list[int] = []
    for _ in range(12):
        step = object()
        for _ in range(3):
            kda_prefill._recurrent_step(*_fresh(inputs), state, metadata, out, step)
        assert kda_prefill._step_chunks is not None
        assert kda_prefill._step_chunks.graph is not None
        torch.accelerator.synchronize()
        reserved.append(current_omni_platform.memory_reserved())
    # Each step records anew; the memory of the step before is reused.
    assert reserved[-1] == reserved[3]
