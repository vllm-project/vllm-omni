# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in trained-weight CFM integration; no HiFT or audio-quality assertions.

Set VLLM_TEST_MINICPMO_4_5_MODEL to a local model directory. This test uses
the production flow configuration and weights with deterministic synthetic
conditioning, not a text-to-speech quality corpus or a model download.
"""

import os
from pathlib import Path

import pytest
import torch

pytest.importorskip("torch_npu")

from tests.helpers.mark import hardware_marks
from tests.platforms.npu.test_minicpmo_code2wav_npugraph import _Token2Wav
from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import BatchedToken2Wav
from vllm_omni.platforms.npu.models.minicpmo_4_5_code2wav import prepare_code2wav_graph_runtime
from vllm_omni.platforms.npu.models.step_audio2_token2wav import npu_token2wav_sdpa_context

pytestmark = [
    pytest.mark.full_model,
    pytest.mark.omni,
    *hardware_marks(res={"npu": "A3"}, num_cards=1),
    pytest.mark.slow,
]


@pytest.fixture(scope="module")
def trained_flow():
    model = os.environ.get("VLLM_TEST_MINICPMO_4_5_MODEL", "")
    if not model:
        pytest.skip("Set VLLM_TEST_MINICPMO_4_5_MODEL to opt into local checkpoint validation")
    assets = Path(model) / "assets" / "token2wav"
    for name in ("flow.yaml", "flow.pt"):
        assert (assets / name).is_file(), f"Missing local model asset: {assets / name}"
    from hyperpyyaml import load_hyperpyyaml

    from vllm_omni.model_executor.models.step_audio2.step_audio2_token2wav import drop_upstream_chunk_att_buffers

    prepare_code2wav_graph_runtime()
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        torch.manual_seed(37)
        with (assets / "flow.yaml").open() as handle:
            flow = load_hyperpyyaml(handle)["flow"]
        drop_upstream_chunk_att_buffers(flow)
        flow.load_state_dict(torch.load(assets / "flow.pt", map_location="cpu", weights_only=True), strict=True)
        flow = flow.float().to("npu").eval()
    finally:
        torch.set_default_dtype(previous_dtype)
    yield flow
    del flow


def _snapshot(result):
    mel, cnn, att = result
    if isinstance(att, list):
        assert len(att) == 1
        att = att[0]
    return mel.clone(), cnn.clone(), att.clone()


@pytest.mark.parametrize("batch", [1, 2, 3, 4])
@torch.inference_mode()
def test_trained_cfm_long_stream_matches_independent_graphs(trained_flow, batch, monkeypatch, record_property):
    token2wav = _Token2Wav(trained_flow.decoder.estimator)
    token2wav.flow = trained_flow
    token2wav.n_timesteps = 10
    backend = BatchedToken2Wav(
        token2wav,
        cfm_graph_config={
            "enabled": True,
            "fused_body": True,
            "capture_frames": 50,
            "micro_batch_size": 4,
            "max_graph_batch": 4,
            "max_graphs": 16,
            "offset_bucket_frames": 50,
        },
    )
    wrapper = backend._whole_euler_graph_wrapper
    assert backend._whole_euler_ragged_active() and backend._ragged_fused_body
    assert wrapper.device.type == "npu" and wrapper.dtype == torch.float32 and wrapper.att_slots == 0
    torch.manual_seed(41)
    channels = int(trained_flow.decoder.rand_noise.shape[1])
    speaker_inputs = torch.randn(batch, trained_flow.spk_embed_affine_layer.in_features, device="npu")
    speakers = torch.cat([backend._project_speakers(value[None]) for value in speaker_inputs])
    accepted = []
    replay = wrapper.replay

    def observe(**kwargs):
        result = replay(**kwargs)
        assert result is not None, "A real graph is required; eager fallback does not qualify"
        accepted.append(int(kwargs["x"].shape[0]))
        return result

    monkeypatch.setattr(wrapper, "replay", observe)
    max_mel_error = 0.0
    try:
        with npu_token2wav_sdpa_context(require_math=True):
            # Different conditions initialize each row independently, exactly
            # once. Both streams then evolve from these same initial states.
            reference = []
            for row in range(batch):
                prompt_mu = torch.randn(1, channels, 50, device="npu") * 0.1
                result = backend._decode_cfm(
                    prompt_mu,
                    speakers[row : row + 1],
                    torch.zeros_like(prompt_mu),
                    cnn_cache=None,
                    att_cache=None,
                    att_keep=(50, 100),
                )
                reference.append(_snapshot(result)[1:])
            current = [(cnn.clone(), att.clone()) for cnn, att in reference]
            accepted.clear()
            for chunk in range(24):
                # A final short query also exercises production query padding.
                width = 30 if chunk == 23 else 50
                mu = torch.randn(batch, channels, width, device="npu") * 0.1
                cond = torch.zeros_like(mu)
                expected = []
                for row in range(batch):
                    cnn, att = reference[row]
                    result = backend._decode_cfm(
                        mu[row : row + 1],
                        speakers[row : row + 1],
                        cond[row : row + 1],
                        cnn_cache=cnn,
                        att_cache=[att],
                        att_keep=(50, 100),
                        prompt_lengths=[50],
                    )
                    expected.append(_snapshot(result))
                cnn_batch = torch.cat(
                    [value[0][:, :, :1] for value in current] + [value[0][:, :, 1:] for value in current], dim=2
                )
                result = backend._decode_cfm(
                    mu,
                    speakers,
                    cond,
                    cnn_cache=cnn_batch,
                    att_cache=[value[1] for value in current],
                    att_keep=(50, 100),
                    prompt_lengths=[50] * batch,
                )
                torch.npu.synchronize()
                assert isinstance(result[2], list) and len(result[2]) == batch
                assert len({value.data_ptr() for value in result[2]}) == batch
                next_states = []
                for row in range(batch):
                    cnn = torch.cat(
                        (result[1][:, :, row : row + 1], result[1][:, :, batch + row : batch + row + 1]), dim=2
                    )
                    actual = result[0][row : row + 1], cnn, result[2][row]
                    for name, value, wanted in zip(("mel", "cnn", "att"), actual, expected[row], strict=True):
                        torch.testing.assert_close(
                            value,
                            wanted,
                            rtol=1e-3,
                            atol=1e-4,
                            msg=lambda message: f"batch={batch}, chunk={chunk}, row={row}, {name}: {message}",
                        )
                    max_mel_error = max(max_mel_error, float((actual[0] - expected[row][0]).abs().max()))
                    next_states.append((cnn.clone(), result[2][row].clone()))
                reference = [(value[1], value[2]) for value in expected]
                current = next_states
                if chunk >= 1:
                    assert all(value[1].shape[4] == 150 for value in current)
        assert accepted.count(batch) >= 24
        stats = wrapper.stats_snapshot()
        assert stats["hits"] > 0 and stats["flushes"] == 0
        record_property("max_mel_abs_error", max_mel_error)
        record_property("cfm_steps", backend.n_timesteps)
        record_property("stream_chunks", 24)
    finally:
        wrapper._flush()
