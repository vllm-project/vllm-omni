# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import queue
import threading
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3 import chunked_cpu_output, vae, vae_parallel

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("route", ["mainline", "legacy", "compact", "mp4", "mp4_batch"])
def test_postprocess_preserves_mainline_and_chunked_output_routes(route):
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _minimax_h3_post_process,
        _minimax_h3_prepare_video_transport,
    )

    frames = torch.arange(2 * 4 * 5 * 3, dtype=torch.uint8).reshape(1, 2, 4, 5, 3)
    if route == "mainline":
        video = frames
    elif route == "legacy":
        video = frames.float().permute(0, 4, 1, 2, 3) / 255
    elif route == "compact":
        video = _minimax_h3_prepare_video_transport(
            frames.float().permute(0, 4, 1, 2, 3) / 255, enabled=True, output_type="np"
        )
    else:
        video = b"mp4" if route == "mp4" else [b"mp4", b"second"]
    audio = torch.zeros(1, 2, 6)
    result = _minimax_h3_post_process((video, audio))
    if route.startswith("mp4"):
        assert result["video"] == ([b"mp4"] if route == "mp4" else video)
        assert result["audio"] == [None] * len(result["video"])
    else:
        expected = frames.float() / 255 if route == "legacy" else frames
        torch.testing.assert_close(torch.from_numpy(result["video"][0]), expected[0])
        torch.testing.assert_close(torch.from_numpy(result["audio"]), audio)
    assert result["fps"] == 24
    assert result["audio_sample_rate"] == 32000


class _WaitingSink:
    """Model the sink's worker waiting for ownership transfer or abort."""

    def __init__(self, **kwargs):
        self.abort_calls = 0
        self.work: queue.Queue[None] = queue.Queue()
        self.thread = threading.Thread(target=self.work.get, daemon=True)
        self.thread.start()

    def abort(self):
        self.abort_calls += 1
        self.work.put(None)
        self.thread.join(timeout=2)


@pytest.fixture
def chunked_host(monkeypatch):
    sinks = []

    def create_sink(**kwargs):
        sink = _WaitingSink(**kwargs)
        sinks.append(sink)
        return sink

    monkeypatch.setattr(chunked_cpu_output, "MiniMaxH3ChunkedCpuMp4Output", create_sink)
    monkeypatch.setattr(vae.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(vae_parallel, "prepare_gather_stream", lambda *args, **kwargs: None)
    monkeypatch.setenv("VLLM_OMNI_H3_VAE_PAIR_PIPELINE", "0")
    monkeypatch.setenv("VLLM_OMNI_H3_VAE_MIXED_BATCH", "0")
    host = SimpleNamespace(
        parallel_size=1,
        config_dict={"latent_channels": 1, "latents_mean": [0.0], "latents_std": [1.0]},
        validate_chunked_output=lambda: None,
        _decoder_tile_count=lambda latent: 1,
        _finalized_chunk_frame_capacity=lambda: 1,
        model=SimpleNamespace(decode_base=lambda *args, **kwargs: None),
        _log_exact_op_chunked_decode=lambda *args, **kwargs: None,
    )
    yield host, sinks
    # Clean up the leaked worker even when running against the broken source.
    for sink in sinks:
        if sink.thread.is_alive():
            sink.abort()


def _decode(host, *, return_output=True):
    return vae.MiniMaxH3VideoVAE.decode_latent_to_chunked_cpu_mp4(
        host,
        torch.zeros(1, 1, 1, 1, 1),
        height=2,
        width=2,
        fps=24,
        audio_sample_rate=32000,
        video_codec_options=None,
        return_output=return_output,
    )


@pytest.mark.parametrize("stage", ["mean", "std", "agreement", "decode", "stats"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_chunked_decode_releases_sink_on_failure(monkeypatch, chunked_host, stage, error_type):
    host, sinks = chunked_host
    failure = error_type(f"injected {stage} failure")

    def fail(*args, **kwargs):
        raise failure

    if stage in ("mean", "std"):
        original_tensor = torch.tensor
        target = host.config_dict[f"latents_{stage}"]

        def allocate(data, **kwargs):
            if data is target:
                raise failure
            return original_tensor(data, **kwargs)

        monkeypatch.setattr(vae.torch, "tensor", allocate)
    elif stage == "agreement":
        host.parallel_size = 2
        monkeypatch.setattr(vae.dist, "is_initialized", lambda: True)
        monkeypatch.setattr(vae, "get_world_group", lambda: SimpleNamespace(device_group=object()))
        monkeypatch.setattr(vae.dist, "all_reduce", fail)
    elif stage == "decode":
        host.model.decode_base = fail
    else:
        host._log_exact_op_chunked_decode = fail

    with pytest.raises(error_type) as caught:
        _decode(host)
    assert caught.value is failure
    assert len(sinks) == 1
    assert sinks[0].abort_calls == 1
    assert not sinks[0].thread.is_alive()


def test_chunked_decode_transfers_live_sink_to_caller(chunked_host):
    host, sinks = chunked_host
    sink = _decode(host)
    assert sink is sinks[0]
    assert sink.abort_calls == 0
    assert sink.thread.is_alive()
    sink.abort()


def test_chunked_decode_without_output_does_not_create_sink(chunked_host):
    host, sinks = chunked_host
    assert _decode(host, return_output=False) is None
    assert sinks == []


def test_chunked_decode_rejects_full_output_and_releases_sink(chunked_host):
    host, sinks = chunked_host
    host.model.decode_base = lambda *args, **kwargs: torch.zeros(1)
    with pytest.raises(RuntimeError, match="unexpectedly returned a full tensor"):
        _decode(host)
    assert sinks[0].abort_calls == 1
    assert not sinks[0].thread.is_alive()


def test_chunked_decode_releases_sink_when_peer_reports_failure(monkeypatch, chunked_host):
    host, sinks = chunked_host
    host.parallel_size = 2
    monkeypatch.setattr(vae.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(vae, "get_world_group", lambda: SimpleNamespace(device_group=object()))
    monkeypatch.setattr(vae.dist, "all_reduce", lambda failed, **kwargs: failed.fill_(1))
    with pytest.raises(RuntimeError, match="sink failed on the output rank"):
        _decode(host)
    assert sinks[0].abort_calls == 1
    assert not sinks[0].thread.is_alive()


def test_chunked_decode_preserves_sink_creation_error(monkeypatch, chunked_host):
    host, sinks = chunked_host
    failure = RuntimeError("sink initialization failed")

    def fail(**kwargs):
        raise failure

    monkeypatch.setattr(chunked_cpu_output, "MiniMaxH3ChunkedCpuMp4Output", fail)
    with pytest.raises(RuntimeError, match="failed to construct") as caught:
        _decode(host)
    assert caught.value.__cause__ is failure
    assert sinks == []


def test_chunked_decode_without_output_preserves_failure(chunked_host):
    host, sinks = chunked_host
    failure = RuntimeError("decode failed")

    def fail(*args, **kwargs):
        raise failure

    host.model.decode_base = fail
    with pytest.raises(RuntimeError) as caught:
        _decode(host, return_output=False)
    assert caught.value is failure
    assert sinks == []
