# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Dense assembly/publication must preserve sequential sink/window semantics."""

import pytest
import torch

from tests.diffusion.models.cosmos3_nano_sim_bimanual.test_batched_clean_commit import (
    BLOCK,
    HEAD,
    WIDTH,
    layers,
    run_forward,
)
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.dense_attention import CosmosSimDenseAttentionCache
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.state_cosmos3_nano_sim_bimanual import append_dense_kv_history
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.transformer_cosmos3_nano_sim_bimanual import _dense_clean_kv

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.parametrize("history_frames", [0, 1, 3, 6])
@pytest.mark.parametrize("frames", [1, 2, 4])
@pytest.mark.parametrize("window,sinks", [(1, 0), (2, 1), (2, 4), (10, 0), (None, 0)])
@pytest.mark.cpu
def test_direct_clean_assembly_is_bitwise_identical(history_frames, frames, window, sinks):
    torch.manual_seed(42)
    block = 3
    history = torch.randn(1, (history_frames if window is None else min(history_frames, window + sinks)) * block, 2, 8)
    current = torch.randn(1, frames * block, 2, 8)
    text = torch.randn(1, 5, 2, 8)
    for frame in range(frames):
        start, end = frame * block, (frame + 1) * block
        prefix = torch.cat([history, current[:, :start]], dim=1)
        if window is not None and prefix.shape[1] > (window + sinks) * block:
            prefix = torch.cat([prefix[:, : sinks * block], prefix[:, -window * block :]], dim=1)
        expected = torch.cat([text, prefix, current[:, start:end]], dim=1)
        actual = _dense_clean_kv(
            text, history, current, start, end, sinks * block, None if window is None else window * block
        )
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("frames", [1, 2, 4, 8])
@pytest.mark.parametrize("window,sinks", [(1, 0), (2, 1), (2, 4), (10, 0), (None, 0)])
@pytest.mark.cpu
def test_bulk_publication_matches_framewise_across_eviction(frames, window, sinks):
    torch.manual_seed(42)
    settings = dict(tokens_per_frame=3, window_frames=window, sink_frames=sinks)
    sequential = batched = None
    # Include an oversized initial chunk and repeated eviction of old history.
    for _ in range(4):
        current = [(torch.randn(1, frames * 3, 2, 8), torch.randn(1, frames * 3, 2, 8)) for _ in range(2)]
        for start in range(0, frames * 3, 3):
            sequential = append_dense_kv_history(
                sequential, [(k[:, start : start + 3], v[:, start : start + 3]) for k, v in current], **settings
            )
        batched = append_dense_kv_history(batched, current, **settings)
        assert batched is not None and sequential is not None
        for actual, expected in zip(batched, sequential):
            assert all(torch.equal(a, b) for a, b in zip(actual, expected))


@pytest.mark.cpu
def test_joint_buffer_reuse_growth_and_session_reset(monkeypatch):
    class Owner:
        pass

    owner = Owner()
    text = [(torch.ones(1, 3, 2, 8), torch.full((1, 3, 2, 8), 2.0))]
    cache = CosmosSimDenseAttentionCache()
    cache.activate(owner, text, 3, 12)
    key, value = cache.kv[0]
    key[:, 3:7].fill_(7)
    value[:, 3:7].fill_(11)

    def no_copy(*args, **kwargs):
        raise AssertionError("Committing written K/V or reusing an activation must not copy")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "copy_", no_copy)
        history = cache.commit(4)
        cache.activate(owner, text, 3, 12, history)
    assert cache.kv[0][0] is key
    assert history[0][0].untyped_storage().data_ptr() == key.untyped_storage().data_ptr()
    cache.activate(owner, text, 3, 20, history)
    assert cache.kv[0][0] is not key
    assert cache.history_length == 4
    assert torch.equal(cache.kv[0][0][:, 3:7], history[0][0])
    assert torch.equal(cache.kv[0][1][:, 3:7], history[0][1])
    addresses = [tensor.data_ptr() for tensor in cache.kv[0]]
    new_owner = Owner()
    replacement = [(torch.full_like(text[0][0], 13), torch.full_like(text[0][1], 17))]
    cache.activate(new_owner, replacement, 3, 12)
    assert cache.history_length == 0
    assert [tensor.data_ptr() for tensor in cache.kv[0]] == addresses
    assert torch.equal(cache.kv[0][0][:, :3], replacement[0][0])
    assert torch.equal(cache.kv[0][1][:, :3], replacement[0][1])
    with pytest.raises(ValueError, match="capacity"):
        cache.commit(20)


def compare_joint_buffer(device, dtype):
    class Owner:
        pass

    torch.manual_seed(42)
    owner, cache = Owner(), CosmosSimDenseAttentionCache()
    net = layers(device, dtype)
    text = [
        (torch.randn(1, 5, 2, HEAD, device=device, dtype=dtype), torch.randn(1, 5, 2, HEAD, device=device, dtype=dtype))
        for _ in net
    ]
    cache.activate(owner, text, 5, 5 + 8 * BLOCK)
    old_history = new_history = None
    start = 0
    for frames in (1, 2, 4, 1):
        latent = torch.randn(1, frames * BLOCK, WIDTH, device=device, dtype=dtype)
        for commit in (False, True):
            before, old_history = run_forward(
                net, latent, text, start, dense=old_history, window=10, commit=commit, batched=commit
            )
            after, new_history = run_forward(
                net, latent, text, start, dense=new_history, window=10, commit=commit, batched=commit, dense_cache=cache
            )
            assert torch.equal(before, after)
            if commit:
                for original, staged in zip(old_history, new_history):
                    assert all(torch.equal(a, b) for a, b in zip(original, staged))
        start += frames


@pytest.mark.cpu
def test_joint_buffer_preserves_attention_and_history_cpu():
    compare_joint_buffer(torch.device("cpu"), torch.float32)


@pytest.mark.cpu
def test_rope_staging_reuses_addresses_and_refreshes_values():
    cache = CosmosSimDenseAttentionCache()
    cos, sin = torch.randn(1, 4, 1, 8), torch.randn(1, 4, 1, 8)
    first = cache.stage_rope(cos, sin, (2, 2))
    cos.add_(3)
    sin.sub_(5)
    second = cache.stage_rope(cos, sin, (2, 2))
    assert all(a is b for a, b in zip(first, second))
    assert all(torch.equal(a, b) for a, b in zip(second, (cos, sin)))
    third = cache.stage_rope(cos, sin, (1, 4))
    assert all(a is not b for a, b in zip(second, third))
    assert len(cache._rope_buffers) == 1


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_joint_buffer_preserves_attention_and_history_gpu():
    compare_joint_buffer(torch.device("cuda"), torch.bfloat16)


@pytest.mark.cpu
@pytest.mark.parametrize("paged,window", [(False, None), (False, 2), (True, None)])
def test_pipeline_activates_joint_buffer_only_for_full_dense_history(paged, window):
    from types import SimpleNamespace

    from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.pipeline_cosmos3_nano_sim_bimanual import (
        Cosmos3NanoSimBimanualPipeline,
    )
    from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.state_cosmos3_nano_sim_bimanual import (
        Cosmos3NanoSimBimanualSessionState,
    )

    pipe = SimpleNamespace(
        _ar_diffusion_kv_state=object() if paged else None,
        manifest=SimpleNamespace(window_frames=window, conditioning_tokens_per_frame=4),
        _SESSION_CAPACITY=1,
        _MAIN_BRANCH="main",
    )
    state = Cosmos3NanoSimBimanualSessionState("test")
    text = [(torch.ones(1, 3, 2, 8), torch.ones(1, 3, 2, 8))]
    geometry = SimpleNamespace(tokens_per_frame=lambda _: 10)
    Cosmos3NanoSimBimanualPipeline._prepare_dense_attention(pipe, state, text, 3, geometry, 5)
    assert (state.dense_attention is not None) == (not paged and window is None)
    if state.dense_attention is not None:
        assert state.dense_attention.kv[0][0].shape == (1, 53, 2, 8)
