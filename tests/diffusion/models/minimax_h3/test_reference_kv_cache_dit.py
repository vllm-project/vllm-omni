# SPDX-License-Identifier: Apache-2.0
"""Exercise the installed Cache-DiT wrapper with real reference-KV states."""

from types import SimpleNamespace

import cache_dit
import pytest
import torch
from cache_dit import BlockAdapter, DBCacheConfig, ForwardPattern, TaylorSeerCalibratorConfig
from cache_dit.caching.block_adapters import FakeDiffusionPipeline
from torch import nn

from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import _supports_reference_kv_compaction
from vllm_omni.diffusion.models.minimax_h3.reference_kv_cache import (
    MiniMaxH3ReferenceKVObserverState,
    MiniMaxH3ReferenceKVTier1State,
    MiniMaxH3ReferenceKVTier2State,
)
from vllm_omni.diffusion.models.minimax_h3.reference_kv_cache_dit import (
    MiniMaxH3CachedAdapter,
    MiniMaxH3CachedBlocks,
    iter_minimax_h3_blocks,
)


class TinyBlock(nn.Module):
    def __init__(self, index: int):
        super().__init__()
        self.layer_index = index
        self.calls = []
        self.attn = SimpleNamespace(
            attention=SimpleNamespace(use_ring=False, attn_backend=SimpleNamespace(get_name=lambda: "FLASH_ATTN"))
        )

    def forward(self, hidden_states, *, reference_kv_tier1_state=None, reference_kv_compact=False):
        state = reference_kv_tier1_state
        if state is not None:
            self.calls.append(state.current_step)
            k = torch.full(
                (hidden_states.shape[0], 2, 3),
                float(1 + self.layer_index + state.reference_epoch),
                device=hidden_states.device,
            )
            if isinstance(state, MiniMaxH3ReferenceKVTier2State):
                out_k, _ = state.process_post_parallel_layer(
                    self.layer_index,
                    k.unsqueeze(0),
                    k.unsqueeze(0) + 2,
                    compact=reference_kv_compact,
                    parallel_strategy="none",
                )
                torch.testing.assert_close(out_k[0, 0], k[0], atol=0.02, rtol=0.02)
            elif reference_kv_compact:
                cached_k, _ = state.take_cached_reference_layer(self.layer_index, k)
                torch.testing.assert_close(cached_k[0], k[0])
            else:
                state.process_layer(self.layer_index, k, k + 2)
        else:
            self.calls.append(-1)
        return hidden_states + (self.layer_index + 1) * 0.01


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(TinyBlock(i) for i in range(6))
        self.wrapper_observed = False

    def forward(self, hidden_states, **kwargs):
        self.wrapper_observed |= isinstance(self.blocks[0], MiniMaxH3CachedBlocks)
        assert _supports_reference_kv_compaction(self.blocks)
        assert len(list(iter_minimax_h3_blocks(self.blocks))) == 6
        for block in self.blocks:
            hidden_states = block(hidden_states, **kwargs)
        return hidden_states


def install(model, *, bn=1, taylor=False, static=False, steps=12):
    config = DBCacheConfig(
        Fn_compute_blocks=1,
        Bn_compute_blocks=bn,
        max_warmup_steps=0,
        num_inference_steps=steps,
        residual_diff_threshold=100.0,
        max_continuous_cached_steps=-1,
        steps_computation_mask=([1] + [0] * (steps - 1)) if static else None,
        steps_computation_policy="static" if static else "dynamic",
    )
    MiniMaxH3CachedAdapter.apply(
        BlockAdapter(
            # Cache-DiT marks pipeline classes as cached. Isolate installations.
            pipe=type("TinyPipeline", (FakeDiffusionPipeline,), {"_is_cached": False})(model),
            transformer=model,
            blocks=model.blocks,
            blocks_name="blocks",
            forward_pattern=ForwardPattern.Pattern_3,
            check_forward_pattern=False,
            has_separate_cfg=False,
        ),
        cache_config=config,
        calibrator_config=TaylorSeerCalibratorConfig(taylorseer_order=2) if taylor else None,
    )
    return model


def state_for(cls, *, device="cpu", quant="none", interval=4):
    kwargs = dict(
        num_layers=6,
        global_reference_rows=2,
        device=torch.device(device),
        refresh_interval=interval,
        pin_memory=device != "cpu",
        skip_reference_projection=True,
    )
    if cls is MiniMaxH3ReferenceKVTier2State:
        kwargs.update(ring_size=2, host_quantization=quant)
    return cls(**kwargs)


def run_steps(model, state, steps=12):
    mask = torch.tensor([True, False, True, False], device=state.device)
    state.set_global_reference_mask(mask)
    outputs = []
    for step in range(steps):
        state.begin_step(step)
        state.set_local_reference_mask(mask)
        compact = state.should_skip_reference_projection
        hidden = torch.zeros(2 if compact else 4, 3, device=state.device)
        output = model(hidden, reference_kv_tier1_state=state, reference_kv_compact=compact)
        if state.device.type != "cpu":
            torch.npu.synchronize()
        state.end_step()
        outputs.append(output)
    return outputs


@pytest.mark.parametrize("cls", [MiniMaxH3ReferenceKVTier1State, MiniMaxH3ReferenceKVTier2State])
@pytest.mark.parametrize("bn", [0, 1])
@pytest.mark.parametrize("taylor,static", [(False, False), (True, False), (True, True)])
def test_refresh_shape_change_and_skipped_middle(cls, bn, taylor, static):
    model = install(TinyTransformer(), bn=bn, taylor=taylor, static=static)
    physical = list(model.blocks)
    state = state_for(cls)
    outputs = run_steps(model, state)
    assert model.wrapper_observed
    for index, block in enumerate(physical):
        if index == 0 or (bn and index == 5):
            assert block.calls == list(range(12))
        else:
            assert block.calls == [0, 1, 4, 5, 8, 9]
    assert state.stats.refresh_steps == 3
    assert state.stats.captured_layers == 18
    assert state.stats.block_cached_layers == 6 * (5 - bn)
    for output in outputs:
        torch.testing.assert_close(output, torch.full_like(output, 0.21), atol=1e-5, rtol=1e-5)
    # A new request must not reuse old residuals/calibrator history.
    cache_dit.refresh_context(model, num_inference_steps=12, verbose=False)
    next_state = state_for(cls)
    run_steps(model, next_state)
    assert next_state.stats.captured_layers == 18
    assert next_state.stats.block_cached_layers == state.stats.block_cached_layers
    state.close()
    next_state.close()


def test_text_only_retains_normal_cache_hits():
    model = install(TinyTransformer())
    physical = list(model.blocks)
    for _ in range(5):
        torch.testing.assert_close(model(torch.zeros(4, 3)), torch.full((4, 3), 0.21))
    assert physical[0].calls == [-1] * 5
    assert physical[2].calls == [-1]
    assert physical[-1].calls == [-1] * 5


def test_physical_inspection_keeps_backend_gates():
    blocks = nn.ModuleList([TinyBlock(0)])
    wrapped = nn.ModuleList([SimpleWrapper(blocks)])
    assert _supports_reference_kv_compaction(wrapped)
    blocks[0].attn.attention.use_ring = True
    assert not _supports_reference_kv_compaction(wrapped)
    blocks[0].attn.attention.use_ring = False
    blocks[0].attn.attention.attn_backend.get_name = lambda: "FASTVIDEO_VSA"
    assert not _supports_reference_kv_compaction(wrapped)


class SimpleWrapper(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.transformer_blocks = blocks


def test_observer_still_executes_every_layer(tmp_path):
    model = install(TinyTransformer(), static=True)
    physical = list(model.blocks)
    state = MiniMaxH3ReferenceKVObserverState(
        num_layers=6,
        global_reference_rows=2,
        device=torch.device("cpu"),
        intervals=(2, 4),
        sample_rows=2,
        sample_heads=1,
        output_path=str(tmp_path / "observer.jsonl"),
    )
    run_steps(model, state, steps=4)
    assert all(block.calls == [0, 1, 2, 3] for block in physical)
    assert state.stats.sampled_layers == 24
    state.close()


def test_skip_validation_rejects_refresh_and_out_of_order():
    state = state_for(MiniMaxH3ReferenceKVTier1State)
    state.begin_step(0)
    with pytest.raises(RuntimeError, match="reuse steps"):
        state.mark_block_cache_skipped([0])
    state.abort_step()
    state.begin_step(1)
    with pytest.raises(RuntimeError, match="in order"):
        state.mark_block_cache_skipped([2])
    assert state.stats.block_cached_layers == 0
    state.abort_step()


@pytest.mark.skipif(not hasattr(torch, "npu") or not torch.npu.is_available(), reason="requires Ascend NPU")
def test_npu_int8_ring_with_cachedit_and_taylorseer():
    model = install(TinyTransformer(), taylor=True, static=True)
    state = state_for(MiniMaxH3ReferenceKVTier2State, device="npu:0", quant="int8")
    run_steps(model, state)
    assert state.effective_host_quantization == "int8"
    assert state.stats.block_cached_layers == 24
    assert state.stats.dequantized_layers == 30
    assert state.stats.prefetched_layers < 9 * 6
    assert state.host_pool.is_pinned()
    state.close()
