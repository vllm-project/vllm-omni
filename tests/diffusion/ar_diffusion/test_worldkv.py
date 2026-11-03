# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU numerical oracles for eager WorldKV; no vLLM/CUDA import required."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def load_worldkv():
    name = "_worldkv_under_test"
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[3] / "vllm_omni/experimental/ar_diffusion/worldkv.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
    return module


worldkv = load_worldkv()


def memory(**kwargs):
    return worldkv.WorldKV(
        worldkv.WorldKVConfig(window_frames=6, sink_frames=2, **kwargs),
        num_layers=2,
        tokens_per_frame=2,
    )


def raw(start, frames=2, *, dtype=torch.float64):
    generator = torch.Generator().manual_seed(100 + start)
    return torch.randn(1, frames * 2, 2, 8, generator=generator, dtype=dtype)


def absolute_keys(x, positions, temporal_dim=4):
    # Independent complex-number oracle, unlike production real-pair arithmetic.
    frequency = 1 / 10000 ** (torch.arange(0, temporal_dim, 2, dtype=torch.float64) / temporal_dim)
    pairs = torch.view_as_complex(x[..., :temporal_dim].double().unflatten(-1, (-1, 2)).contiguous())
    phase = torch.exp(1j * positions.double()[:, None] * frequency)
    rotated = torch.view_as_real(pairs * phase[None, :, None]).flatten(-2)
    return torch.cat((rotated.to(x.dtype), x[..., temporal_dim:]), -1)


def append(state, start, frames=2):
    positions = torch.arange(start, start + frames).repeat_interleave(2)
    with state.forward(start, frames, commit=True):
        for layer in range(state.num_layers):
            k = absolute_keys(raw(start) + layer, positions)
            state.assemble(layer, raw(start), k, raw(start) * 2)


def test_long_rollout_matches_explicit_sink_recent_attention():
    state = memory()
    for start in range(0, 40, 2):
        previous = [None if rows is None else rows.key.clone() for rows in state.history]
        outputs = []
        for _ in range(3):
            with state.forward(start, 2, commit=False):
                for layer in range(2):
                    q = raw(start)
                    current = absolute_keys(raw(start) + layer, torch.arange(start, start + 2).repeat_interleave(2))
                    k, v = state.assemble(layer, q, current, raw(start) * 2)
                    ids = list(range(min(start, 2))) + list(range(max(2, start - 2), start))
                    keys, values = [], []
                    for frame in ids:
                        chunk = frame // 2 * 2
                        local = slice((frame % 2) * 2, (frame % 2 + 1) * 2)
                        x = (raw(chunk) + layer)[:, local]
                        target = frame
                        keys.append(absolute_keys(x, torch.full((2,), target)))
                        values.append((raw(chunk) * 2)[:, local])
                    reference_k = torch.cat([*keys, current], 1)
                    reference_v = torch.cat([*values, raw(start) * 2], 1)
                    expected = F.scaled_dot_product_attention(
                        q.transpose(1, 2), reference_k.transpose(1, 2), reference_v.transpose(1, 2)
                    )
                    actual = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
                    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
                    outputs.append(actual)
            for layer, rows in enumerate(state.history):
                if rows is not None:
                    assert torch.equal(rows.key, previous[layer])
        torch.testing.assert_close(outputs[0], outputs[2])
        append(state, start)
        assert all(rows.positions.numel() <= 12 for rows in state.history)


def test_failed_forward_and_failed_publication_leave_history_unchanged(monkeypatch):
    state = memory()
    append(state, 0)
    old = state.history
    with pytest.raises(RuntimeError, match="injected"):
        with state.forward(2, 2, commit=True):
            state.assemble(0, raw(2), raw(2), raw(2))
            raise RuntimeError("injected layer failure")
    assert state.history is old and state.next_frame == 2
    retain = state._retained

    def fail(layer, current):
        if layer == 1:
            raise RuntimeError("injected publication failure")
        return retain(layer, current)

    monkeypatch.setattr(state, "_retained", fail)
    with pytest.raises(RuntimeError, match="publication"):
        append(state, 2)
    assert state.history is old and state.next_frame == 2
    monkeypatch.setattr(state, "_retained", retain)
    append(state, 2)
    assert state.next_frame == 4


def test_incomplete_duplicate_out_of_order_closed_and_session_isolation():
    state, other = memory(), memory()
    with pytest.raises(RuntimeError, match="all layers"):
        with state.forward(0, 2, commit=True):
            state.assemble(0, raw(0), raw(0), raw(0))
    with pytest.raises(RuntimeError, match="exactly once"):
        with state.forward(0, 2, commit=True):
            state.assemble(0, raw(0), raw(0), raw(0))
            state.assemble(0, raw(0), raw(0), raw(0))
    append(state, 0)
    assert other.next_frame == 0 and other.history == [None, None]
    with pytest.raises(ValueError, match="contiguous"):
        with state.forward(0, 2, commit=False):
            pass
    state.close()
    state.close()
    assert state.history == [None, None]
    with pytest.raises(RuntimeError, match="closed"):
        with state.forward(2, 2, commit=False):
            pass


@pytest.mark.parametrize(
    "config",
    [
        {"window_frames": 0},
        {"sink_frames": -1},
        {"sink_frames": True},
        {"window_frames": 9, "sink_frames": 9},
    ],
)
def test_invalid_config(config):
    with pytest.raises(ValueError):
        worldkv.WorldKVConfig(**config)


def test_real_lingbot_attention_uses_memory_view():
    from tests.diffusion.models.lingbot_world.test_lingbot_world_attention import _load_module

    module = _load_module()
    attention = module.LingBotSelfAttention(8, 2)
    state = worldkv.WorldKV(worldkv.WorldKVConfig(window_frames=4, sink_frames=1), num_layers=1, tokens_per_frame=1)
    with torch.inference_mode():
        for start in range(7):
            hidden = torch.randn(1, 1, 8)
            with state.forward(start, 1, commit=True):
                output = attention(hidden, cache=None, current_start=start, sink_tokens=1, worldkv_layer=(state, 0))
            assert output.shape == hidden.shape
            assert torch.isfinite(output).all()
    assert state.next_frame == 7
    assert state.history[0].positions.tolist() == [0, 4, 5, 6]


def test_full_lingbot_transformer_matches_legacy_and_rolls_back_head_failure(monkeypatch):
    from tests.diffusion.models.lingbot_world.test_lingbot_world_attention import _load_module

    module = _load_module()
    model = module.CausalLingBotWorldTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=4,
        in_channels=36,
        out_channels=2,
        text_dim=6,
        freq_dim=4,
        ffn_dim=8,
        num_layers=2,
        sink_size=1,
        num_frames_per_block=1,
        sliding_window_num_frames=3,
    ).eval()
    dense = module.allocate_lingbot_cache(
        batch_size=1,
        num_layers=2,
        max_tokens=12,
        num_local_heads=2,
        head_dim=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    state = worldkv.WorldKV(
        worldkv.WorldKVConfig(window_frames=3, sink_frames=1),
        num_layers=2,
        tokens_per_frame=4,
    )
    policy = module.LingBotTransformerCache([], [None, None], worldkv=state)
    with torch.inference_mode():
        for frame in range(8):
            hidden = torch.randn(1, 36, 1, 4, 4)
            camera = torch.randn(1, 384, 1, 4, 4)
            text = torch.randn(1, 3, 6)
            kwargs = dict(
                hidden_states=hidden,
                camera_hidden_states=camera,
                encoder_hidden_states=text,
                timestep=torch.zeros(1),
                start_frame=frame,
            )
            for commit in (False, False, True):
                actual = model(**kwargs, cache=policy, update_cache=commit)
                reference = model(**kwargs, cache=dense, update_cache=commit)
                assert torch.isfinite(actual).all()
                torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-6)
        old = state.history

        def fail(*args, **kwargs):
            raise RuntimeError("head failed")

        monkeypatch.setattr(model.head, "forward", fail)
        with pytest.raises(RuntimeError, match="head failed"):
            model(**{**kwargs, "start_frame": 8}, cache=policy, update_cache=True)
        assert state.history is old and state.next_frame == 8


@pytest.mark.parametrize("start", [2, 6])
def test_publication_concatenates_only_retained_slices(monkeypatch, start):
    state = memory()
    for offset in range(0, start, 2):
        append(state, offset)
    previous = state.history[0]
    current = worldkv.MemoryRows(
        raw(start),
        raw(start),
        torch.arange(start, start + 2).repeat_interleave(2),
        torch.arange(start * 2, (start + 2) * 2),
    )
    ids = torch.cat((previous.token_ids, current.token_ids))
    keep = torch.cat((torch.arange(min(4, ids.numel())), torch.arange(max(4, ids.numel() - 8), ids.numel())))
    expected_k = torch.cat((previous.key, current.key), 1)[:, keep]

    def forbidden(*args, **kwargs):
        pytest.fail("publication must not copy an assembled window again")

    monkeypatch.setattr(worldkv.MemoryRows, "select", forbidden)
    result = state._retained(0, current)
    torch.testing.assert_close(result.key, expected_k)
    assert torch.equal(result.token_ids, ids[keep])
    assert result.key.untyped_storage().nbytes() == result.key.numel() * result.key.element_size()


NATIVE = {"window_frames": 18, "sink_frames": 9, "chunk_frames": 3}


def test_empty_policy_disables_all_optional_mechanisms_and_keeps_native_sink():
    assert worldkv.WorldKVConfig.parse(None, **NATIVE) is None
    config = worldkv.WorldKVConfig.parse({}, **NATIVE)
    assert config == worldkv.WorldKVConfig(window_frames=18, sink_frames=9)
    assert config.retrieval_frames == config.bank_bytes == 0
    assert config.world_keep_ratio == 1.0
    explicit = worldkv.WorldKVConfig.parse({"sink_frames": 3, "retrieval_frames": 9, "bank_bytes": 1}, **NATIVE)
    assert (explicit.window_frames, explicit.sink_frames) == (18, 3)


@pytest.mark.parametrize(
    "value,match",
    [
        # 9 sink + 8 retrieval leaves no room for the 3-frame chunk.
        ({"retrieval_frames": 8, "bank_bytes": 1}, "complete current chunk"),
        # A chunk at frames 3..5 would straddle a 4-frame sink.
        ({"sink_frames": 4, "retrieval_frames": 6, "bank_bytes": 1}, "multiple of the chunk size"),
        # One chunk compressed at 0.5 needs two frames of budget.
        ({"sink_frames": 3, "retrieval_frames": 1, "bank_bytes": 1, "world_keep_ratio": 0.5}, "one compressed chunk"),
    ],
)
def test_parse_rejects_chunk_geometry_before_execution(value, match):
    with pytest.raises(ValueError, match=match):
        worldkv.WorldKVConfig.parse(value, **NATIVE)


def test_sink_alignment_applies_only_to_retrieval():
    config = worldkv.WorldKVConfig.parse({"sink_frames": 4}, **NATIVE)
    assert config.sink_frames == 4
