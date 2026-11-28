# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Weight-free regression of seeded sampling through the real frame method."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from .test_realtime_batched_gpu_history import DEVICES, GENERATE, load_functions

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _local_model(device):
    rvq, width, vocab = 16, 4, 32

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.codec_embedding = torch.nn.ModuleList([torch.nn.Embedding(vocab, width) for _ in range(rvq - 1)])

        def forward(self, embeds, positions):
            return embeds

    local = SimpleNamespace(config=SimpleNamespace(rvq=rvq, hidden_size=width), model=Decoder().to(device))
    heads = torch.nn.ModuleList([torch.nn.Linear(width, vocab, bias=False) for _ in range(rvq)]).to(device)
    # Equal logits isolate RNG ownership from floating-point batch differences.
    for head in heads:
        head.weight.data.zero_()
    return local, heads, torch.zeros(3, width, device=device)


def _frame(local, heads, hidden, generators):
    return GENERATE(local, hidden, heads, top_k=0, top_p=1.0, generators=generators)


@pytest.mark.parametrize("device", DEVICES)
def test_generate_frame_seeded_rows_survive_reorder_and_removal(device):
    local, heads, hidden = _local_model(device)
    batched_rng = [torch.Generator(device=device).manual_seed(seed) for seed in (7, 11, 23)]
    independent_rng = [torch.Generator(device=device).manual_seed(seed) for seed in (7, 11, 23)]
    initial_states = [gen.get_state().clone() for gen in batched_rng]
    for active in ([0, 1, 2], [2, 0, 1], [1, 2], [2]):
        batch = _frame(local, heads, hidden[active], [batched_rng[i] for i in active])
        independent = torch.cat([_frame(local, heads, hidden[i : i + 1], [independent_rng[i]]) for i in active])
        torch.testing.assert_close(batch, independent)
    for idx, gen in enumerate(batched_rng):
        assert not torch.equal(gen.get_state(), initial_states[idx])
        torch.testing.assert_close(gen.get_state(), independent_rng[idx].get_state())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("do_sample", [True, False])
def test_generate_frame_rejects_generator_count_before_sampling(device, do_sample):
    local, heads, hidden = _local_model(device)
    with pytest.raises(ValueError, match="Expected 3 per-row generators"):
        GENERATE(local, hidden, heads, do_sample=do_sample, generators=[torch.Generator(device=device)])


@pytest.mark.parametrize("device", DEVICES)
def test_generate_frame_mixed_seeded_and_unseeded_rows(device):
    local, heads, hidden = _local_model(device)
    seeded = torch.Generator(device=device).manual_seed(19)
    reference = torch.Generator(device=device).manual_seed(19)
    for _ in range(3):
        batch = _frame(local, heads, hidden[:2], [None, seeded])
        independent = _frame(local, heads, hidden[1:2], [reference])
        torch.testing.assert_close(batch[1:2], independent)
    torch.testing.assert_close(seeded.get_state(), reference.get_state())


def _talker(local, heads):
    return SimpleNamespace(
        n_vq=16,
        local_transformer=SimpleNamespace(generate_frame=lambda *args, **kwargs: GENERATE(local, *args, **kwargs)),
        local_lm_heads=heads,
        AUDIO_EOS=1026,
        AUDIO_BOS=1025,
        audio_pad_token=1024,
    )


@pytest.mark.parametrize("device", DEVICES)
def test_realtime_seeded_request_chain_reorder_cancel_and_new_request(device):
    from .test_realtime_batched_gpu_history import run

    local, heads, hidden = _local_model(device)

    class FixedLogitsHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("logits", -(torch.arange(32, dtype=torch.float32) + 0.5) / 8)

        def forward(self, hidden):
            return self.logits.expand(*hidden.shape[:-1], -1)

    # The talker uses top_k=30/top_p=0.6. Uniform logits put the 18th
    # cumulative probability exactly on the cutoff, where different CUDA
    # scan orders can select different nuclei. Use fixed, tie-free logits
    # away from that boundary to isolate request RNG ownership. The direct
    # frame test above still covers uniform logits without nucleus filtering.
    heads = torch.nn.ModuleList([FixedLogitsHead() for _ in heads]).to(device)
    talker = _talker(local, heads)
    infos: dict[str, dict[str, Any]] = {key: {"seed": [seed]} for key, seed in zip("abcd", (7, 11, 23, 7))}
    singles: dict[str, dict[str, Any]] = {key: {"seed": [seed]} for key, seed in zip("abcd", (7, 11, 23, 7))}
    first_a = None
    # d is a new request reusing a's seed after a was cancelled. It must own a
    # new generator; c/b preserve their RNG positions through changing batches.
    for keys in ("abc", "cab", "bc", "dcb"):
        selected = [infos[key] for key in keys]
        actual = run(talker, hidden[: len(keys)], selected).multimodal_outputs["codes"]["audio"]
        for row, key in enumerate(keys):
            expected = run(talker, hidden[:1], [singles[key]]).multimodal_outputs["codes"]["audio"][0]
            torch.testing.assert_close(actual[row], expected)
            torch.testing.assert_close(
                infos[key]["audio_state"]["local_generator"].get_state(),
                singles[key]["audio_state"]["local_generator"].get_state(),
            )
            if key == "a" and first_a is None:
                first_a = actual[row].clone()
            if key == "d":
                torch.testing.assert_close(actual[row], first_a)
        owned = [info["audio_state"]["local_generator"] for info in selected]
        assert len({id(gen) for gen in owned}) == len(owned)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed_source", ["adapter", "v1_params", "v2_params"])
def test_realtime_seed_transport_sources(device, seed_source):
    from .test_realtime_batched_gpu_history import MAKE, run

    local, heads, hidden = _local_model(device)
    talker = _talker(local, heads)
    params = SimpleNamespace(seed=None, extra_args={"tts_local_seed": 31})
    info: dict[str, Any] = {}
    extra = {}
    if seed_source == "adapter":
        info["seed"] = [31]
    elif seed_source == "v1_params":
        extra["request_sampling_params"] = [params]
    else:
        info["sampling_params"] = params
    result = MAKE(talker, hidden[:1], model_intermediate_buffer=[info], request_token_spans=[(0, 1)], **extra)
    expected = run(talker, hidden[:1], [{"seed": 31}])
    torch.testing.assert_close(
        result.multimodal_outputs["codes"]["audio"][0], expected.multimodal_outputs["codes"]["audio"][0]
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("replacement_id", ["original", "replacement"])
def test_v2_buffer_removal_and_slot_reuse_restart_request_rng(device, replacement_id):
    """Exercise real buffer lifecycle methods, not a full scheduler/engine."""
    from .test_realtime_batched_gpu_history import run

    method_names = {"__init__", "add_request", "remove_request"}
    methods = load_functions(
        "../../../worker_v2/model_states/intermediate_buffer.py",
        method_names,
        "OmniIntermediateBuffer",
    )
    buffer_type = type("BufferLifecycle", (), {name: methods[name] for name in method_names})
    buffer = buffer_type(1)

    def add_request(request_id):
        buffer.add_request(
            0,
            SimpleNamespace(req_id=request_id, mm_features=[], sampling_params=SimpleNamespace(seed=31)),
        )

    local, heads, hidden = _local_model(device)
    talker = _talker(local, heads)
    add_request("original")
    first = run(talker, hidden[:1], [buffer.buffers[0]]).multimodal_outputs["codes"]["audio"][0].clone()
    old_generator = buffer.buffers[0]["audio_state"]["local_generator"]
    first_rng_state = old_generator.get_state().clone()
    run(talker, hidden[:1], [buffer.buffers[0]])
    advanced_rng_state = old_generator.get_state().clone()
    assert not torch.equal(first_rng_state, advanced_rng_state)

    # V2 completion and abort both remove the request-owned buffer before a
    # scheduler slot (and potentially the same request ID) is reused.
    buffer.remove_request(0)
    assert buffer.buffers[0] == {}
    assert buffer.req_id_to_index == {}
    add_request(replacement_id)
    assert buffer.req_id_to_index == {replacement_id: 0}
    assert "audio_state" not in buffer.buffers[0]

    replay = run(talker, hidden[:1], [buffer.buffers[0]]).multimodal_outputs["codes"]["audio"][0]
    new_generator = buffer.buffers[0]["audio_state"]["local_generator"]
    assert new_generator is not old_generator
    torch.testing.assert_close(replay, first)
    torch.testing.assert_close(new_generator.get_state(), first_rng_state)
    torch.testing.assert_close(old_generator.get_state(), advanced_rng_state)
