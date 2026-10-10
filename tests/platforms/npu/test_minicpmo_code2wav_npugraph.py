# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend hardware validation for MiniCPM-o Code2Wav NPUGraph replay."""

from math import prod
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

pytest.importorskip("vllm_ascend")

from tests.helpers.mark import hardware_marks
from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import (
    BatchedToken2Wav,
)
from vllm_omni.platforms.npu.graph_tools import NPUExactGraphRunner
from vllm_omni.platforms.npu.models.minicpmo_4_5_code2wav import (
    _graphable_estimator_step,
    prepare_code2wav_graph_runtime,
)
from vllm_omni.platforms.npu.models.step_audio2_token2wav import (
    npu_token2wav_sdpa_context,
)
from vllm_omni.platforms.npu.worker.npu_model_runner import OmniNPUModelRunner
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.omni,
    *hardware_marks(res={"npu": "A3"}, num_cards=1),
]


class _Flow(nn.Module):
    def __init__(self, estimator: nn.Module):
        super().__init__()
        self.encoder = nn.Identity()
        self.decoder = SimpleNamespace(estimator=estimator)


class _Token2Wav:
    def __init__(self, estimator: nn.Module):
        self.flow = _Flow(estimator).eval()
        self.hift = nn.Identity().eval()
        self.float16 = False
        self.n_timesteps = 2
        self.mel_cache_len = 1
        self.source_cache_len = 2
        self.speech_window = torch.ones(4, device="npu")


def test_npu_runner_inherits_runtime_snapshot_replacement():
    assert issubclass(OmniNPUModelRunner, OmniGPUModelRunner)
    assert OmniNPUModelRunner._replace_intermediate_buffer is OmniGPUModelRunner._replace_intermediate_buffer
    runner = object.__new__(OmniNPUModelRunner)
    request = SimpleNamespace(additional_information_cpu=None)
    runner.requests = {"req": request}
    runner.model_intermediate_buffer = {
        "req": {
            "codes": {"audio": torch.tensor([1, 2])},
            "meta": {"last_chunk": True},
        }
    }

    runner._replace_intermediate_buffer("req", {"meta": {"is_segment_finished": True}})

    assert runner.model_intermediate_buffer["req"] == {"meta": {"is_segment_finished": True}}
    assert request.additional_information_cpu == runner.model_intermediate_buffer["req"]


def test_minicpmo_cfm_estimator_npugraph_matches_eager():
    decoder_dit = pytest.importorskip("cosyvoice2.flow.decoder_dit")
    prepare_code2wav_graph_runtime()

    estimator = (
        decoder_dit.DiT(
            in_channels=6,
            out_channels=2,
            depth=2,
            num_heads=2,
            head_dim=8,
            hidden_size=16,
            mlp_ratio=2.0,
        )
        .npu()
        .eval()
    )
    graph_runner = NPUExactGraphRunner(max_graphs=4)
    adapter = BatchedToken2Wav(_Token2Wav(estimator))

    def inputs(seed: int):
        def make(shape: tuple[int, ...], offset: float):
            values = torch.arange(
                prod(shape),
                device="npu",
                dtype=torch.float32,
            )
            return torch.sin(values * 0.17 + seed + offset).reshape(shape)

        return (
            make((2, 2, 6), 0.0),
            make((2, 2, 6), 0.5),
            make((2, 1, 16), 1.0),
            make((2, 1), 1.5),
            make((2, 1, 6), 2.0),
        )

    def compute(*values, caches=None):
        cnn_cache, att_cache = (None, None) if caches is None else caches
        return _graphable_estimator_step(
            adapter,
            estimator,
            x=values[0],
            mu=values[1],
            time_embedding=values[2],
            speakers=values[3],
            cond=values[4],
            cnn_cache=cnn_cache,
            att_cache=att_cache,
        )

    with torch.inference_mode(), npu_token2wav_sdpa_context(require_math=True):
        warm_inputs = inputs(11)
        warm_outputs = graph_runner.run(
            "cfm_estimator",
            warm_inputs,
            (False,),
            lambda *values: compute(*values),
        )
        replay_inputs = inputs(13)
        expected = compute(*replay_inputs)
        actual = graph_runner.run(
            "cfm_estimator",
            replay_inputs,
            (False,),
            lambda *values: compute(*values),
        )
        for eager, replayed in zip(expected, actual, strict=True):
            torch.testing.assert_close(replayed, eager, rtol=1e-3, atol=1e-3)

        cached_inputs = (*inputs(17), warm_outputs[1], warm_outputs[2])
        graph_runner.run(
            "cfm_estimator",
            cached_inputs,
            (True,),
            lambda *values: compute(*values[:5], caches=(values[5], values[6])),
        )
        replay_cached_inputs = (
            *inputs(19),
            warm_outputs[1] + 0.1,
            warm_outputs[2] + 0.1,
        )
        expected_cached = compute(
            *replay_cached_inputs[:5],
            caches=(replay_cached_inputs[5], replay_cached_inputs[6]),
        )
        actual_cached = graph_runner.run(
            "cfm_estimator",
            replay_cached_inputs,
            (True,),
            lambda *values: compute(*values[:5], caches=(values[5], values[6])),
        )
        for eager, replayed in zip(expected_cached, actual_cached, strict=True):
            torch.testing.assert_close(replayed, eager, rtol=1e-3, atol=1e-3)
        assert graph_runner.stats == {"captures": 2, "failed": 0, "hits": 2}


@pytest.mark.parametrize("bucket_frames", [0, 8])
@torch.inference_mode()
def test_whole_euler_npugraph_matches_eager_and_owns_outputs(bucket_frames):
    """A whole solve, including masked padding and cached replay, uses one graph."""
    from vllm_omni.model_executor.models.minicpmo_4_5.cuda_graph_wrapper import WholeEulerCFMGraphWrapper

    decoder_dit = pytest.importorskip("cosyvoice2.flow.decoder_dit")
    prepare_code2wav_graph_runtime()
    torch.manual_seed(19)
    estimator = (
        decoder_dit.DiT(in_channels=7, out_channels=2, depth=2, num_heads=2, head_dim=8, hidden_size=16, mlp_ratio=2.0)
        .npu()
        .eval()
    )
    adapter = BatchedToken2Wav(_Token2Wav(estimator))
    adapter.flow.decoder.rand_noise = torch.randn(1, 2, 128, device="npu")
    adapter.flow.decoder.inference_cfg_rate = 0.7
    wrapper = WholeEulerCFMGraphWrapper(
        estimator,
        n_timesteps=2,
        micro_batch_size=2,
        max_graph_batch=2,
        query_bucket_frames=bucket_frames,
        ragged_body=adapter._ragged_body,
    )
    assert wrapper.enabled
    caches = (None, None)
    with npu_token2wav_sdpa_context(require_math=True):
        # Same cached shape twice, with different input values, exercises replay.
        for index in range(3):
            mu = torch.randn(2, 2, 6, device="npu")
            speakers = torch.randn(2, 1, device="npu")
            cond = torch.randn(2, 2, 6, device="npu")
            adapter._whole_euler_graph_wrapper = None
            expected = adapter._decode_cfm(mu, speakers, cond, cnn_cache=caches[0], att_cache=caches[1])
            adapter._whole_euler_graph_wrapper = wrapper
            actual = adapter._decode_cfm(mu, speakers, cond, cnn_cache=caches[0], att_cache=caches[1])
            for value, reference in zip(actual, expected, strict=True):
                torch.testing.assert_close(value, reference, rtol=1e-4, atol=1e-5)
            if index == 0:
                saved = tuple(value.clone() for value in actual)
                first = actual
                caches = tuple(value.clone() for value in expected[1:])
        for value, reference in zip(first, saved, strict=True):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
    assert wrapper.stats_snapshot()["captures"] == 2
    assert wrapper.stats_snapshot()["hits"] >= 1

    # Unequal query lengths exercise the mask/length inputs that the old NPU
    # estimator patch excluded from graph replay.
    with npu_token2wav_sdpa_context(require_math=True):
        adapter._whole_euler_graph_wrapper = None
        expected = adapter._decode_cfm(mu, speakers, cond, cnn_cache=None, att_cache=None, valid_lengths=[6, 4])
        adapter._whole_euler_graph_wrapper = wrapper
        actual = adapter._decode_cfm(mu, speakers, cond, cnn_cache=None, att_cache=None, valid_lengths=[6, 4])
        for row, length in enumerate((6, 4)):
            torch.testing.assert_close(actual[0][row, :, :length], expected[0][row, :, :length], rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(actual[1:], expected[1:], rtol=1e-4, atol=1e-5)


@torch.inference_mode()
def test_encoder_npugraph_shared_arena_replays_exact_shapes():
    from tests.model_executor.models.minicpmo_4_5.test_flow_encoder_graph import _graphs, _inputs

    prepare_code2wav_graph_runtime()
    graphs, encode = _graphs("npu", rows=(1, 2), frames=(8, 18))
    for rows, frames in ((2, 18), (1, 8), (2, 18)):
        tokens, cnn, att = _inputs(rows, 6, frames, "npu", seed=rows + frames)
        expected = encode(tokens, cnn_cache=torch.cat(cnn), att_cache=torch.cat(att, dim=1))
        actual = graphs.run(tokens, cnn, att)
        assert actual is not None
        for value, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(value, reference)
    assert graphs.replays == 3
    assert len(graphs._storage) == 6
    # An unknown startup shape must not start an opportunistic capture.
    graphs.capture_on_request = False
    graphs.forward = lambda tokens, *, last_chunk, **kwargs: encode(tokens, **kwargs)
    tokens, cnn, att = _inputs(1, 7, 8, "npu")
    graphs(tokens, last_chunk=False, cnn_cache=torch.cat(cnn), att_cache=torch.cat(att, dim=1))
    assert graphs._npu_runners.captures == 0


@torch.inference_mode()
def test_hift_npugraph_matches_eager_with_source_cache():
    from tests.model_executor.models.minicpmo_4_5.test_cuda_graph_wrapper import (
        _DeterministicSineGen,
        _F0Predictor,
    )
    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.hifigan import HiFTGenerator
    from vllm_omni.model_executor.models.minicpmo_4_5.cuda_graph_wrapper import HiFTGraphWrapper

    prepare_code2wav_graph_runtime()
    hift = (
        HiFTGenerator(
            base_channels=32,
            sampling_rate=24000,
            upsample_rates=[8, 5, 3],
            upsample_kernel_sizes=[16, 11, 7],
            source_resblock_kernel_sizes=[7, 7, 11],
            source_resblock_dilation_sizes=[[1, 3, 5]] * 3,
            f0_predictor=_F0Predictor(),
        )
        .npu()
        .eval()
    )
    hift.m_source.l_sin_gen = _DeterministicSineGen(hift.nb_harmonics + 1)
    token2wav = SimpleNamespace(
        hift=hift,
        flow=SimpleNamespace(encoder=nn.Identity(), token_mel_ratio=2),
        mel_cache_len=2,
        source_cache_len=960,
    )
    wrapper = HiFTGraphWrapper(token2wav, {"codec_chunk_frames": 2, "codec_left_context_frames": 3}, [1, 2])
    source = torch.randn(2, 2880, device="npu")
    for actual, expected in zip(hift._stft(source), hift._stft_on_cpu(source), strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    wrapper.capture()
    for rows, frames, cache_frames in ((2, 6, 960), (1, 4, 0), (2, 6, 960)):
        mel = torch.randn(rows, 80, frames, device="npu")
        cache = torch.randn(rows, 1, cache_frames, device="npu")
        # Compare against the original NPU eager path, including CPU STFT.
        hift._stft = hift._stft_on_cpu
        hift._istft = hift._istft_on_cpu
        legacy = hift.inference(mel, cache)
        del hift._stft
        del hift._istft
        expected = hift.inference(mel, cache)
        actual = wrapper.replay(mel, cache)
        for value, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(value, reference, rtol=1e-4, atol=1e-5)
        # DFT and CPU FFT have different FP32 reduction order; downstream
        # HF32 convolutions can amplify that rounding. Bound waveform error
        # against the old path as well as checking graph/eager equivalence.
        error = (actual[0] - legacy[0]).float()
        assert error.abs().max().item() < 1e-3
        relative_rms = error.square().mean() / legacy[0].float().square().mean().clamp_min(1e-12)
        assert relative_rms.item() < 1e-6  # waveform SNR > 60 dB
        torch.testing.assert_close(actual[1], legacy[1], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@torch.inference_mode()
def test_hift_stft_preserves_fft_precision_inside_autocast(dtype):
    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.hifigan import HiFTGenerator

    hift = HiFTGenerator.__new__(HiFTGenerator)
    nn.Module.__init__(hift)
    hift.register_parameter("device_anchor", nn.Parameter(torch.zeros(1, device="npu")))
    hift.istft_params = {"n_fft": 16, "hop_len": 4}
    hift.register_buffer("stft_window", torch.hann_window(16, dtype=torch.float32).npu())
    hift.enable_npu_graph_stft()
    source = torch.sin(torch.arange(256, device="npu", dtype=torch.float32)).reshape(2, 128).to(dtype)
    expected = hift._stft_on_cpu(source)
    with torch.autocast("npu", dtype=torch.float16):
        actual = hift._stft(source)
    for value, reference in zip(actual, expected, strict=True):
        assert value.dtype == dtype
        torch.testing.assert_close(value, reference, rtol=1e-4 if dtype == torch.float32 else 1e-3, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("batch,frames", [(1, 3), (1, 121), (4, 721)])
@torch.inference_mode()
def test_hift_npu_istft_matches_cpu_without_repeated_host_copies(monkeypatch, dtype, batch, frames):
    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.hifigan import HiFTGenerator

    hift = HiFTGenerator.__new__(HiFTGenerator)
    nn.Module.__init__(hift)
    hift.register_parameter("device_anchor", nn.Parameter(torch.zeros(1, device="npu")))
    hift.istft_params = {"n_fft": 16, "hop_len": 4}
    hift.register_buffer("stft_window", torch.hann_window(16, dtype=torch.float32).npu())
    hift.enable_npu_graph_stft()
    torch.manual_seed(7)
    magnitude = (torch.rand(batch, 9, frames, device="npu") * 200).to(dtype)
    phase = (torch.rand_like(magnitude) * 6.0 - 3.0).to(dtype)
    expected = hift._istft_on_cpu(magnitude, phase)
    # Prime the window-only envelope, just as graph startup does.
    hift._istft(magnitude, phase)
    envelope = next(iter(hift._npu_istft_envelopes.values()))

    def unexpected_host_copy(*args, **kwargs):
        raise AssertionError("warmed ISTFT must stay on device")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", unexpected_host_copy)
        patch.setattr(torch.Tensor, "item", unexpected_host_copy)
        with torch.autocast("npu", dtype=torch.float16):
            actual = hift._istft(magnitude, phase)
    assert actual.dtype == dtype and actual.shape == (batch, 4 * (frames - 1))
    torch.testing.assert_close(actual, expected, rtol=1e-4 if dtype == torch.float32 else 1e-3, atol=1e-4)
    retained = actual.clone()
    hift._istft(magnitude * 0.5, phase)
    torch.testing.assert_close(actual, retained, rtol=0, atol=0)
    assert next(iter(hift._npu_istft_envelopes.values())) is envelope
