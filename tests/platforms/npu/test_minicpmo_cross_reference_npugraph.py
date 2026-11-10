# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real Ascend CFM graph checks; a small real DiT, not checkpoint audio qualification."""

import pytest
import torch

pytest.importorskip("torch_npu")

from tests.helpers.mark import hardware_marks
from tests.platforms.npu.test_minicpmo_code2wav_npugraph import _Token2Wav
from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import BatchedToken2Wav
from vllm_omni.platforms.npu.models.minicpmo_4_5_code2wav import prepare_code2wav_graph_runtime
from vllm_omni.platforms.npu.models.step_audio2_token2wav import npu_token2wav_sdpa_context

pytestmark = [pytest.mark.core_model, pytest.mark.omni, *hardware_marks(res={"npu": "A3"}, num_cards=1)]


@pytest.mark.parametrize("batch", [1, 2, 3, 4])
@pytest.mark.parametrize("capture_width", [16, 24])
@torch.inference_mode()
def test_equal_history_cross_reference_graph_matches_independent_solves(batch, capture_width, monkeypatch):
    """Per-row conditions/caches, row/query padding and trimming survive replay."""
    decoder_dit = pytest.importorskip("cosyvoice2.flow.decoder_dit")
    prepare_code2wav_graph_runtime()
    torch.manual_seed(29)
    estimator = (
        decoder_dit.DiT(in_channels=7, out_channels=2, depth=2, num_heads=2, head_dim=8, hidden_size=16, mlp_ratio=2.0)
        .float()
        .to("npu")
        .eval()
    )
    for parameter in estimator.parameters():
        parameter.normal_(0.0, 0.08)
    token2wav = _Token2Wav(estimator)
    token2wav.flow.decoder.rand_noise = torch.randn(1, 2, 256, device="npu", dtype=torch.float32)
    token2wav.flow.decoder.inference_cfg_rate = 0.7
    token2wav.n_timesteps = 10
    backend = BatchedToken2Wav(
        token2wav,
        cfm_graph_config={
            "enabled": True,
            "fused_body": True,
            "capture_frames": capture_width,
            "micro_batch_size": 4,
            "max_graph_batch": 4,
            "max_graphs": 32,
            "offset_bucket_frames": 8,
        },
    )
    wrapper = backend._whole_euler_graph_wrapper
    assert backend._ragged_fused_body and backend._whole_euler_ragged_active()
    assert wrapper.device.type == "npu" and wrapper.enabled
    arena = wrapper.arena
    prompt_frames = 4
    # The deliberately distinct rows stand for different reference conditions.
    expected_att = [
        torch.randn(backend.n_timesteps, arena.depth, 2, arena.heads, prompt_frames, arena.att_width, device="npu")
        for _ in range(batch)
    ]
    actual_att = [value.clone() for value in expected_att]
    expected_cnn = [
        torch.randn(backend.n_timesteps, arena.depth, 2, arena.cnn_channels, arena.cnn_width, device="npu")
        for _ in range(batch)
    ]
    actual_cnn = [value.clone() for value in expected_cnn]
    speakers = torch.arange(1, batch + 1, device="npu", dtype=torch.float32).reshape(batch, 1) / 3
    accepted_batches = []
    original_replay = wrapper.replay

    def observed_replay(**kwargs):
        result = original_replay(**kwargs)
        assert result is not None, "test must exercise a real graph, not an eager fallback"
        accepted_batches.append(int(kwargs["x"].shape[0]))
        return result

    monkeypatch.setattr(wrapper, "replay", observed_replay)
    with npu_token2wav_sdpa_context(require_math=True):
        # 24 x 16 frames crosses prompt + 100 cache trimming and replays the
        # steady key. Row 3 is padded to graph batch 4, then sliced back.
        for chunk in range(24):
            mu = torch.randn(batch, 2, 16, device="npu")
            cond = torch.randn_like(mu) / 2
            wanted = []
            for row in range(batch):
                result = backend._decode_cfm(
                    mu[row : row + 1],
                    speakers[row : row + 1],
                    cond[row : row + 1],
                    cnn_cache=expected_cnn[row],
                    att_cache=[expected_att[row]],
                    att_keep=(prompt_frames, 100),
                    prompt_lengths=[prompt_frames],
                )
                wanted.append((result[0].clone(), result[1].clone(), [value.clone() for value in result[2]]))
            # Preserve each row's conditions and CFG [conditional B | zero B]
            # ordering exactly as _stack_flow_cache does in the model path.
            cnn_batch = torch.cat(
                [value[:, :, :1] for value in actual_cnn] + [value[:, :, 1:] for value in actual_cnn], dim=2
            )
            actual = backend._decode_cfm(
                mu,
                speakers,
                cond,
                cnn_cache=cnn_batch,
                att_cache=actual_att,
                att_keep=(prompt_frames, 100),
                prompt_lengths=[prompt_frames] * batch,
            )
            torch.npu.synchronize()
            assert isinstance(actual[2], list) and len(actual[2]) == batch
            new_cnn = [
                torch.cat((actual[1][:, :, row : row + 1], actual[1][:, :, batch + row : batch + row + 1]), dim=2)
                for row in range(batch)
            ]
            for row, reference in enumerate(wanted):
                torch.testing.assert_close(actual[0][row : row + 1], reference[0], rtol=1e-3, atol=1e-4)
                torch.testing.assert_close(new_cnn[row], reference[1], rtol=1e-3, atol=1e-4)
                torch.testing.assert_close(actual[2][row], reference[2][0], rtol=1e-3, atol=1e-4)
            assert len({value.data_ptr() for value in actual[2]}) == batch
            expected_cnn = [value[1] for value in wanted]
            expected_att = [value[2][0] for value in wanted]
            actual_cnn, actual_att = new_cnn, actual[2]
            if chunk >= 7:
                assert all(value.shape[4] == prompt_frames + 100 for value in actual_att)
    assert batch in accepted_batches
    stats = wrapper.stats_snapshot()
    assert stats["hits"] > 0 and stats["flushes"] == 0


@torch.inference_mode()
def test_shared_prompt_input_is_not_overwritten(monkeypatch):
    """A shared read-only initial cache must not become another row's state."""
    decoder_dit = pytest.importorskip("cosyvoice2.flow.decoder_dit")
    prepare_code2wav_graph_runtime()
    estimator = (
        decoder_dit.DiT(in_channels=7, out_channels=2, depth=2, num_heads=2, head_dim=8, hidden_size=16, mlp_ratio=2.0)
        .float()
        .to("npu")
        .eval()
    )
    token2wav = _Token2Wav(estimator)
    token2wav.flow.decoder.rand_noise = torch.randn(1, 2, 256, device="npu")
    token2wav.flow.decoder.inference_cfg_rate = 0.7
    backend = BatchedToken2Wav(token2wav, cfm_graph_config={"enabled": True, "fused_body": True})
    arena = backend._whole_euler_graph_wrapper.arena
    cache = torch.randn(backend.n_timesteps, arena.depth, 2, arena.heads, 4, arena.att_width, device="npu")
    saved = cache.clone()
    with npu_token2wav_sdpa_context(require_math=True):
        result = backend._decode_cfm(
            torch.randn(2, 2, 8, device="npu"),
            torch.tensor([[0.2], [-0.7]], device="npu"),
            torch.zeros(2, 2, 8, device="npu"),
            cnn_cache=None,
            att_cache=[cache, cache],
            att_keep=(4, 100),
            prompt_lengths=[4, 4],
        )
    torch.npu.synchronize()
    torch.testing.assert_close(cache, saved, rtol=0, atol=0)
    assert result[2][0].data_ptr() != result[2][1].data_ptr()
