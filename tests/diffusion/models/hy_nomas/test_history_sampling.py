# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import replace

import pytest
import torch

from tests.diffusion.ar_diffusion.test_capability_runner import make_runner
from tests.diffusion.models.hy_nomas.test_ar import batch, execute, sampling, step_chunk
from vllm_omni.diffusion.models.hy_nomas.configuration_hy_nomas import HyNomasARConfig
from vllm_omni.diffusion.models.hy_nomas.history_sampling import (
    ARCHIVED_HISTORY_MODES,
    history_frame_indices,
    select_history,
    selected_layers,
)
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.platforms import current_omni_platform

pytest_plugins = ["tests.diffusion.models.hy_nomas.test_ar"]

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("uniform", (0, 3, 6, 9)),
        ("recent_mix_uniform", (0, 7, 8, 9)),
        ("temporal_context", (0, 4, 7, 9)),
    ],
)
def test_reference_grids(mode, expected):
    assert history_frame_indices(10, 4, mode) == expected
    for total in (0, 1, 3, 18, 36, 72, 4096, 2**63 + 17):
        for budget in (1, 2, 9, 18, 36, 72):
            frames = history_frame_indices(total, budget, mode)
            assert len(frames) == min(total, budget)
            assert frames == tuple(sorted(set(frames)))
            assert all(0 <= frame < total for frame in frames)
            if frames:
                assert frames[-1] == total - 1


@pytest.mark.parametrize("gamma", [0, -1, float("nan"), float("inf"), True, "2"])
def test_invalid_temporal_gamma(gamma):
    with pytest.raises(ValueError, match="gamma"):
        HyNomasARConfig.from_args({"ar_history_temporal_gamma": gamma})
    with pytest.raises(ValueError, match="gamma"):
        history_frame_indices(20, 9, "temporal_context", gamma=gamma)


@pytest.fixture(params=ARCHIVED_HISTORY_MODES)
def sampled_runner(pipeline, monkeypatch, request):
    pipeline.ar_config = replace(
        pipeline.ar_config, history_mode=request.param, history_latent_frames=4, history_archive_max_bytes=1 << 24
    )
    monkeypatch.setattr(current_omni_platform, "synchronize", lambda: None)
    runner = make_runner(pipeline)
    yield runner
    for session_id in tuple(runner._sessions):
        runner.close_session(session_id)


def test_sampled_pipeline_matches_dense_history_and_action_masks(sampled_runner):
    runner = sampled_runner
    pipeline = runner.pipeline
    previous: list[tuple[int, int, int]] = []
    for index in range(18):
        call_start = len(pipeline.calls)
        execute(runner, batch(index=index, prompt="longer prompt" if index % 2 else "forest"))
        session = pipeline._ar_sessions["s"]
        state = runner._sessions["s"].segmented("cond")
        expected_positions = history_frame_indices((index + 1) * 3, 4, pipeline.ar_config.history_mode)
        assert tuple(frame.position for frame in session.history_frames) == expected_positions
        expected_rows: list[int] = []
        expected_mask: list[bool] = []
        tails: list[tuple[int, int, int]] = []
        for segment in sorted({position // 3 for position in expected_positions}):
            count = sum(position // 3 == segment for position in expected_positions)
            text = 7 if segment % 2 else 4
            start = len(expected_rows)
            expected_rows.extend([segment * 3 + 1] * (count + text))
            expected_mask.extend([True] * (count + text - 1) + [False])
            tails.append((segment + 1, start + count + 2, start + count + text))
        state.prepare_layer_views(1, 0)
        for layer in range(2):
            empty = torch.empty(0, 2, 4)
            key, value = state.assemble(layer, empty, empty, torch.zeros(1, 2, 4), torch.zeros(1, 2, 4))
            expected = (torch.tensor(expected_rows).float() + layer)[:, None, None].expand(-1, 2, 4)
            torch.testing.assert_close(key[:-1], expected, rtol=0, atol=0)
            torch.testing.assert_close(value[:-1], -expected, rtol=0, atol=0)
        assert session.history_mask.tolist() == [expected_mask]
        assert session.action_tails == tails
        assert state.history_tokens <= 4 * (1 + pipeline.args.text_len)
        # Old segment action tails are hidden in attention without changing the archive.
        first_call = pipeline.calls[call_start]
        for _, start, end in previous:
            assert not first_call["mask"][:, start:end].any()
        previous = tails
        assert session.history_archive.frames == (index + 1) * 3


def test_selected_later_chunk_restores_earlier_segment_text(sampled_runner):
    runner = sampled_runner
    state = StepRequestState(request_id="multi", sampling=sampling(frames=21), prompt="forest")
    with runner._bound_ar_session("multi", description="test"):
        runner.pipeline.prepare_encode(state)
        step_chunk(runner.pipeline, state)
        step_chunk(runner.pipeline, state)
    archive = runner.pipeline._ar_sessions["multi"].history_archive
    assert [record.text_tokens for record in archive.records] == [4, 0]
    selected = select_history(archive.records, (4, 5))
    assert selected[-1].text is archive.records[0]
    layers = list(selected_layers(selected, 2))
    for layer, (key, value) in enumerate(layers):
        expected = torch.cat((archive.records[1].layers[layer][0][1:3], archive.records[0].layers[layer][0][3:]))
        torch.testing.assert_close(key, expected, rtol=0, atol=0)
        torch.testing.assert_close(value, -expected, rtol=0, atol=0)


@pytest.mark.parametrize("phase", ["restore", "peer"])
def test_failed_rebuild_keeps_committed_archive_and_pages(sampled_runner, monkeypatch, phase):
    runner = sampled_runner
    execute(runner, batch())
    pipeline = runner.pipeline
    session = pipeline._ar_sessions["s"]
    cache = runner._sessions["s"].segmented("cond")
    records = session.history_archive.records
    handles = cache.handles
    mask = session.history_mask.clone()
    free = cache.cache.manager.block_pool.get_num_free_blocks()
    original = cache.stage_replacement

    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("injected replacement failure")

    if phase == "restore":
        monkeypatch.setattr(cache, "stage_replacement", fail)
    else:
        original_check = pipeline._check_sp_error

        def peer(error, operation):
            original_check(error, operation)
            if operation == "KV staging":
                raise RuntimeError("injected replacement failure")

        monkeypatch.setattr(pipeline, "_check_sp_error", peer)
    # Inspect before runner failure cleanup intentionally destroys the session.
    with runner._bound_ar_session("s", description="test"):
        with pytest.raises(RuntimeError, match="replacement failure"):
            pipeline.forward(batch(index=1))
    assert session.history_archive.records is records
    assert cache.handles == handles
    assert cache.cache.manager.block_pool.get_num_free_blocks() == free
    torch.testing.assert_close(session.history_mask, mask, rtol=0, atol=0)
    assert not cache.commit_active


@pytest.mark.parametrize("mode", ARCHIVED_HISTORY_MODES)
def test_sampled_configuration_requires_archive_quota(mode):
    with pytest.raises(ValueError, match="positive archive byte quota"):
        HyNomasARConfig.from_args({"ar_history_mode": mode})
    config = HyNomasARConfig.from_args(
        {
            "ar_history_mode": mode,
            "ar_history_archive_max_bytes": 1024,
            "ar_history_latent_frames": 9,
            "ar_history_temporal_gamma": 0.5,
        }
    )
    assert config.history_mode == mode and config.history_temporal_gamma == 0.5
    assert config.history_latent_frames == 9
    assert HyNomasARConfig.from_args({}).history_mode == "full"
