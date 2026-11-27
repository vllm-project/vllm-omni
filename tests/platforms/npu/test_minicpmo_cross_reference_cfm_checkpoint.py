# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in trained-weight CFM integration; no HiFT or audio-quality assertions.

Set VLLM_TEST_MINICPMO_4_5_MODEL to a local model directory. This test uses
the production flow configuration and weights with deterministic synthetic
conditioning, not a text-to-speech quality corpus or a model download.
"""

import os
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("torch_npu")

from tests.helpers.mark import hardware_marks
from tests.platforms.npu.test_minicpmo_code2wav_npugraph import _Token2Wav
from vllm_omni.model_executor.models.minicpmo_4_5.code2wav.token2wav import BatchedToken2Wav
from vllm_omni.model_executor.models.minicpmo_4_5.ops import fused_qkv_layer_norm
from vllm_omni.platforms.npu.models.minicpmo_4_5_code2wav import prepare_code2wav_graph_runtime
from vllm_omni.platforms.npu.models.step_audio2_token2wav import npu_token2wav_sdpa_context

pytestmark = [
    pytest.mark.full_model,
    pytest.mark.omni,
    *hardware_marks(res={"npu": "A3"}, num_cards=1),
    pytest.mark.slow,
]


@torch.inference_mode()
def test_fp16_qkv_fusion_matches_fp32_layer_norm():
    batch, frames, heads, head_dim, capacity = 8, 50, 8, 64, 450
    generator = torch.Generator().manual_seed(42)
    packed = torch.randn(batch, frames, 3 * heads * head_dim, generator=generator).half()
    initial_kv = torch.randn(2, batch, heads, capacity, head_dim, generator=generator).half()
    q_norm = torch.nn.LayerNorm(head_dim, eps=1e-5, dtype=torch.float16)
    k_norm = torch.nn.LayerNorm(head_dim, eps=1e-6, dtype=torch.float16)
    for norm in (q_norm, k_norm):
        norm.weight.copy_(torch.randn(head_dim, generator=generator))
        norm.bias.copy_(torch.randn(head_dim, generator=generator))

    q, k, v = (part.transpose(1, 2) for part in packed.view(batch, frames, 3, heads, head_dim).unbind(dim=2))
    expected_q = F.layer_norm(q.float(), (head_dim,), q_norm.weight.float(), q_norm.bias.float(), q_norm.eps).half()
    expected_k = F.layer_norm(k.float(), (head_dim,), k_norm.weight.float(), k_norm.bias.float(), k_norm.eps).half()
    packed_npu, kv = packed.to("npu"), initial_kv.to("npu")
    q_norm, k_norm = q_norm.to("npu"), k_norm.to("npu")
    assert fused_qkv_layer_norm._use_npu_triton(packed_npu, kv, q_norm, k_norm, head_dim)
    actual_q = fused_qkv_layer_norm.qkv_head_layer_norm(
        packed_npu, kv, num_heads=heads, head_dim=head_dim, q_norm=q_norm, k_norm=k_norm
    )
    actual_kv = kv.cpu()
    # Allow one FP16 rounding step for the different FP32 reductions.
    torch.testing.assert_close(actual_q.cpu(), expected_q, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(actual_kv[0, :, :, :frames], expected_k, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(actual_kv[1, :, :, :frames], v, rtol=0, atol=0)
    torch.testing.assert_close(actual_kv[:, :, :, frames:], initial_kv[:, :, :, frames:], rtol=0, atol=0)


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


_FP16_ATOL = 2e-2
_FP16_RTOL = 2e-2
# Across the 24-chunk Ascend runs, at most 6.41e-5 of cache elements exceeded
# the nominal tolerance per row/chunk, with a worst ratio of 4.624. Allow
# sparse drift between graph batch shapes while bounding its magnitude.
_FP16_CACHE_MAX_OUTLIER_FRACTION = 1e-4
_FP16_CACHE_MAX_TOLERANCE_RATIO = 5.0


def _fp16_error_stats(actual, expected):
    delta = (actual.float() - expected.float()).abs_()
    max_abs = float(delta.max())
    tolerance = expected.to(dtype=torch.float32, copy=True).abs_().mul_(_FP16_RTOL).add_(_FP16_ATOL)
    delta.div_(tolerance)
    return max_abs, float(torch.count_nonzero(delta > 1)) / delta.numel(), float(delta.max())


@pytest.mark.parametrize(
    "native_reference,precision",
    [
        pytest.param(False, "fp32", id="batching-fp32"),
        pytest.param(True, "fp32", id="fusion-off-on"),
        pytest.param(False, "fp16-sdpa", id="fp16-sdpa"),
    ],
)
@pytest.mark.parametrize("mixed_history,variable_queries", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("batch", [1, 2, 3, 4])
@torch.inference_mode()
def test_trained_cfm_long_stream_matches_independent_graphs(
    trained_flow, batch, monkeypatch, record_property, mixed_history, variable_queries, native_reference, precision
):
    fp16 = precision != "fp32"
    dtype = torch.float16 if fp16 else torch.float32
    # Preserve the shared FP32 checkpoint for subsequent parameterized cases.
    if fp16:
        decoder = trained_flow.decoder
        monkeypatch.setattr(decoder, "estimator", deepcopy(decoder.estimator))
    token2wav = _Token2Wav(trained_flow.decoder.estimator)
    token2wav.flow = trained_flow
    token2wav.n_timesteps = 10
    config = {
        "enabled": True,
        "fused_body": True,
        "capture_frames": 50,
        "micro_batch_size": 4,
        "max_graph_batch": 4,
        # Independent reference rows and merged rows share this graph cache.
        "max_graphs": 64,
        "offset_bucket_frames": 50,
    }
    backend = BatchedToken2Wav(
        token2wav,
        cfm_graph_config=config,
        cfm_npu_float16=fp16,
    )
    wrapper = backend._whole_euler_graph_wrapper
    assert backend._whole_euler_ragged_active() and backend._ragged_fused_body
    assert wrapper.device.type == "npu" and wrapper.dtype == dtype and wrapper.att_slots == 0
    torch.manual_seed(41)
    channels = int(trained_flow.decoder.rand_noise.shape[1])
    speaker_inputs = torch.randn(batch, trained_flow.spk_embed_affine_layer.in_features, device="npu")
    speakers = torch.cat([backend._project_speakers(value[None]) for value in speaker_inputs])
    wrapper.npu_mixed_history = mixed_history
    prompt_lengths = [50 + 10 * row if mixed_history else 50 for row in range(batch)]
    accepted = []
    fused_calls = 0
    fused_capture_batches = set()
    replay = wrapper.replay

    def observe(**kwargs):
        assert kwargs["x"].dtype == torch.float32
        before = fused_calls
        result = replay(**kwargs)
        assert result is not None, (
            "A real graph is required; eager fallback does not qualify; "
            f"stats={wrapper.stats_snapshot()}, max_graphs={wrapper.max_graphs}, "
            f"cached={len(wrapper._cache)}, unsupported={wrapper._unsupported}, "
            f"batch={kwargs['x'].shape[0]}, query={kwargs['x'].shape[2]}, "
            f"offsets={[int(row.shape[-2]) for row in kwargs['att_cache']] if isinstance(kwargs['att_cache'], list) else None}"
        )
        logical_batch = int(kwargs["x"].shape[0])
        accepted.append(logical_batch)
        if fused_calls > before:
            fused_capture_batches.add(logical_batch)
        return result

    monkeypatch.setattr(wrapper, "replay", observe)

    reference_backend = BatchedToken2Wav(token2wav, cfm_graph_config=config) if native_reference else backend
    reference_wrapper = reference_backend._whole_euler_graph_wrapper
    reference_wrapper.npu_mixed_history = mixed_history
    native_calls = 0
    eligibility = fused_qkv_layer_norm._use_npu_triton

    def observe_fused(*args, **kwargs):
        nonlocal fused_calls
        eligible = eligibility(*args, **kwargs)
        if eligible:
            fused_calls += 1
        return eligible

    monkeypatch.setattr(fused_qkv_layer_norm, "_use_npu_triton", observe_fused)

    def native_eligibility(*args, **kwargs):
        nonlocal native_calls
        native_calls += 1
        return False

    def reference_decode(*args, **kwargs):
        with monkeypatch.context() if native_reference else nullcontext() as patch:
            if native_reference:
                patch.setattr(fused_qkv_layer_norm, "_use_npu_triton", native_eligibility)
            return reference_backend._decode_cfm(*args, **kwargs)

    # Both paths read the same immutable noise table and fixed codec stream.
    initial_noise = trained_flow.decoder.rand_noise.clone()
    generator = torch.Generator().manual_seed(43)
    tokens = torch.randint(trained_flow.input_embedding.num_embeddings, (batch, 24, 32), generator=generator).to("npu")
    encoder_states = [(None, None) for _ in range(batch)]
    chunk_errors = []
    chunk_relative_errors = []
    max_mel_error = 0.0
    fp16_chunk_stats: dict[str, list[tuple[float, float, float]]] = {name: [] for name in ("mel", "cnn", "att")}
    fp16_failures = []
    try:
        with npu_token2wav_sdpa_context(require_math=True):
            # Different conditions initialize each row independently, exactly
            # once. Both streams then evolve from these same initial states.
            reference = []
            prompts = []
            for row in range(batch):
                prompt_mu = torch.randn(1, channels, prompt_lengths[row], device="npu") * 0.1
                prompts.append(prompt_mu.clone())
                result = reference_decode(
                    prompt_mu,
                    speakers[row : row + 1],
                    torch.zeros_like(prompt_mu),
                    cnn_cache=None,
                    att_cache=None,
                    att_keep=(max(prompt_lengths), 100),
                )
                reference.append(_snapshot(result)[1:])
            if native_reference:
                current = []
                for row in range(batch):
                    prompt_mu = prompts[row]
                    current.append(
                        _snapshot(
                            backend._decode_cfm(
                                prompt_mu,
                                speakers[row : row + 1],
                                torch.zeros_like(prompt_mu),
                                cnn_cache=None,
                                att_cache=None,
                                att_keep=(max(prompt_lengths), 100),
                            )
                        )[1:]
                    )
            else:
                current = [(cnn.clone(), att.clone()) for cnn, att in reference]
            accepted.clear()
            for chunk in range(24):
                # A final short query also exercises production query padding.
                width = 30 if chunk == 23 else 50
                if native_reference:
                    hidden_rows = []
                    for row in range(batch):
                        cnn, att = encoder_states[row]
                        hidden, cnn, att = backend._encode_chunk_eager(
                            tokens[row : row + 1, chunk],
                            last_chunk=chunk == 23,
                            cnn_cache=cnn,
                            att_cache=att,
                        )
                        encoder_states[row] = (cnn, att)
                        assert hidden.shape[1] >= width
                        hidden_rows.append(hidden[:, :width].transpose(1, 2).contiguous())
                    mu = torch.cat(hidden_rows)
                else:
                    mu = torch.randn(batch, channels, width, device="npu") * 0.1
                widths = [width - 2 * row if variable_queries else width for row in range(batch)]
                cond = torch.zeros_like(mu)
                if variable_queries:
                    # Poison invalid conditioning frames: a real ragged NPU
                    # solve must exclude them from valid outputs and caches.
                    # Independent solves below see only each row's valid slice.
                    for row, valid_width in enumerate(widths):
                        mu[row, :, valid_width:] = 7.0
                        cond[row, :, valid_width:] = -11.0
                expected = []
                for row in range(batch):
                    cnn, att = reference[row]
                    result = reference_decode(
                        mu[row : row + 1, :, : widths[row]],
                        speakers[row : row + 1],
                        cond[row : row + 1, :, : widths[row]],
                        cnn_cache=cnn,
                        att_cache=[att],
                        att_keep=(max(prompt_lengths), 100),
                        prompt_lengths=[prompt_lengths[row]],
                    )
                    expected.append(_snapshot(result))
                cnn_batch = torch.cat(
                    [value[0][:, :, :1] for value in current] + [value[0][:, :, 1:] for value in current], dim=2
                )
                same_shape_row = None
                control_row = 0 if chunk == 0 else batch - 1
                if fp16 and batch > 1 and not mixed_history and chunk in (0, 6, 14, 23):
                    # The repeated request and mixed requests use the same
                    # physical graph shape and inputs for the selected row.
                    # Cover startup, both observed cache outliers and the tail.
                    row_slice = slice(control_row, control_row + 1)
                    row_cnn, row_att = current[control_row]
                    replica_cnn = row_cnn.repeat_interleave(batch, dim=2)
                    replica_mel, replica_cnn_out, replica_att = backend._decode_cfm(
                        mu[row_slice].expand_as(mu),
                        speakers[row_slice].expand_as(speakers),
                        cond[row_slice].expand_as(cond),
                        cnn_cache=replica_cnn,
                        att_cache=[row_att] * batch,
                        att_keep=(max(prompt_lengths), 100),
                        prompt_lengths=prompt_lengths,
                    )
                    same_shape_row = (
                        replica_mel[row_slice].clone(),
                        torch.cat(
                            (
                                replica_cnn_out[:, :, row_slice],
                                replica_cnn_out[:, :, batch + control_row : batch + control_row + 1],
                            ),
                            dim=2,
                        ),
                        replica_att[control_row],
                    )
                    del replica_mel, replica_cnn_out, replica_att, replica_cnn
                result = backend._decode_cfm(
                    mu,
                    speakers,
                    cond,
                    cnn_cache=cnn_batch,
                    att_cache=[value[1] for value in current],
                    att_keep=(max(prompt_lengths), 100),
                    valid_lengths=widths if mixed_history else None,
                    prompt_lengths=prompt_lengths,
                )
                torch.npu.synchronize()
                assert isinstance(result[2], list) and len(result[2]) == batch
                assert len({value.data_ptr() for value in result[2]}) == batch
                next_states = []
                chunk_max_error = 0.0
                chunk_max_relative_error = 0.0
                chunk_stats = {name: (0.0, 0.0, 0.0) for name in fp16_chunk_stats} if fp16 else {}
                for row in range(batch):
                    cnn = torch.cat(
                        (result[1][:, :, row : row + 1], result[1][:, :, batch + row : batch + row + 1]), dim=2
                    )
                    actual = result[0][row : row + 1, :, : widths[row]], cnn, result[2][row]
                    for name, value, wanted in zip(("mel", "cnn", "att"), actual, expected[row], strict=True):
                        assert value.shape == wanted.shape
                        assert value.dtype == (torch.float32 if name == "mel" else dtype)
                        assert torch.isfinite(value).all()
                        if fp16:
                            observed = _fp16_error_stats(value, wanted)
                            chunk_stats[name] = tuple(map(max, zip(chunk_stats[name], observed, strict=True)))
                            max_abs, fraction, ratio = observed
                            if name == "mel":
                                within_tolerance = ratio <= 1.0
                            else:
                                within_tolerance = (
                                    fraction <= _FP16_CACHE_MAX_OUTLIER_FRACTION
                                    and ratio <= _FP16_CACHE_MAX_TOLERANCE_RATIO
                                )
                            if not within_tolerance:
                                fp16_failures.append(
                                    f"chunk={chunk}, row={row}, {name}: "
                                    f"max_abs={max_abs:.6g}, outlier_fraction={fraction:.6g}, max_ratio={ratio:.6g}"
                                )
                        else:
                            torch.testing.assert_close(
                                value,
                                wanted,
                                rtol=1e-3,
                                atol=1e-4,
                                msg=lambda message: f"batch={batch}, chunk={chunk}, row={row}, {name}: {message}",
                            )
                    if row == control_row and same_shape_row is not None:
                        for name, value, replica in zip(("mel", "cnn", "att"), actual, same_shape_row, strict=True):
                            assert value.shape == replica.shape
                            max_abs = float((value.float() - replica.float()).abs().max())
                            record_property(f"same_shape_{name}_chunk_{chunk}_max_abs_error", max_abs)
                            if max_abs != 0.0:
                                fp16_failures.append(
                                    f"chunk={chunk}, row={row}, {name}, repeated vs mixed: max_abs={max_abs:.6g}"
                                )
                    error = float((actual[0] - expected[row][0]).abs().max())
                    relative = float(
                        ((actual[0] - expected[row][0]).abs() / expected[row][0].abs().clamp_min(1e-6)).max()
                    )
                    chunk_max_relative_error = max(chunk_max_relative_error, relative)
                    chunk_max_error = max(chunk_max_error, error)
                    max_mel_error = max(max_mel_error, error)
                    next_states.append((cnn.clone(), result[2][row].clone()))
                chunk_errors.append(chunk_max_error)
                chunk_relative_errors.append(chunk_max_relative_error)
                for name, observed in chunk_stats.items():
                    fp16_chunk_stats[name].append(observed)
                reference = [(value[1], value[2]) for value in expected]
                current = next_states
                if chunk >= (3 if variable_queries else 1):
                    assert all(value[1].shape[-2] == prompt_lengths[row] + 100 for row, value in enumerate(current))
        assert accepted.count(batch) >= 24
        stats = wrapper.stats_snapshot()
        assert stats["hits"] > 0 and stats["flushes"] == 0
        torch.testing.assert_close(trained_flow.decoder.rand_noise, initial_noise, rtol=0, atol=0)
        if native_reference:
            assert native_calls > 0
            assert reference_wrapper.stats_snapshot()["hits"] > 0
        assert batch in fused_capture_batches, "The requested batch must capture the NPU Triton QKV path"
        record_property("cfm_precision", precision)
        record_property("qkv_fused_capture_batches", sorted(fused_capture_batches))
        record_property("chunk_mel_max_abs_errors", chunk_errors)
        record_property("chunk_mel_max_relative_errors_floor_1e_6", chunk_relative_errors)
        record_property("max_mel_abs_error", max_mel_error)
        record_property("cfm_steps", backend.n_timesteps)
        record_property("stream_chunks", 24)
        if fp16:
            for name, observed in fp16_chunk_stats.items():
                record_property(f"chunk_{name}_error_stats", observed)
            assert not fp16_failures, (
                f"{len(fp16_failures)} row/tensor comparisons exceeded FP16 bounds:\n" + "\n".join(fp16_failures[:8])
            )
    finally:
        wrapper._flush()
        if native_reference:
            reference_wrapper._flush()
