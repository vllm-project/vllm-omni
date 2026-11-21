# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Exercise file-backed staging with asynchronous device transfers."""

import gc
import hashlib
import weakref

import pytest
import torch
from torch import nn

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.offloader.file_backed_offload import FileBackedBlockOffloader
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]


def _make_offloader(tmp_path, device):
    # More blocks than slots force reuse; distinct weights expose overwrites.
    model = nn.Module()
    model.blocks = nn.ModuleList([nn.Linear(8, 8, bias=False, device="meta", dtype=torch.bfloat16) for _ in range(4)])
    header = b"{}"
    data = bytearray(len(header).to_bytes(8, "little") + header)
    blocks = []
    for index in range(4):
        value = torch.full((8, 8), index + 1, dtype=torch.bfloat16)
        raw = value.view(torch.uint8).numpy().tobytes()
        offset = len(data)
        data.extend(raw)
        source = {
            "shard": "weights.bin",
            "file_offset": offset,
            "bytes": len(raw),
            "concat_offset": 0,
            "source_name": f"blocks.{index}.weight",
        }
        entry = {
            "runtime_name": f"blocks.{index}.weight",
            "shape": (8, 8),
            "dtype": "BF16",
            "bytes": len(raw),
            "buffer_offset": 0,
            "sources": [source],
        }
        blocks.append({"parameters": [entry], "bytes": len(raw)})
    (tmp_path / "weights.bin").write_bytes(data)
    manifest = {
        "model_path": str(tmp_path),
        "shards": [
            {
                "shard": "weights.bin",
                "header_bytes": 10,
                "file_bytes": len(data),
                "header_sha256": hashlib.sha256(header).hexdigest(),
            }
        ],
        "blocks": blocks,
        "resident_tensors": [],
    }
    return FileBackedBlockOffloader(model, model.blocks, manifest, device), model


@pytest.fixture
def device():
    if current_omni_platform.get_device_count() == 0:
        pytest.skip("Accelerator required for file-backed staging")
    return current_omni_platform.get_torch_device(0)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_reused_slots_preserve_in_flight_weights(tmp_path, device, monkeypatch):
    offloader, model = _make_offloader(tmp_path, device)
    outputs = []
    inflight_h2d = []
    inflight_compute = []
    submit = offloader._submit

    def observe_and_submit(index):
        if index >= offloader.num_slots:
            slot = index % offloader.num_slots
            inflight_h2d.append(not offloader.h2d_done[slot].query())
            inflight_compute.append(not offloader.compute_done[slot].query())
        submit(index)

    monkeypatch.setattr(offloader, "_submit", observe_and_submit)
    offloader.activate()
    try:
        # Keep both streams busy, then verify that reuse was actually submitted
        # while the previous transfer and computation were still in flight.
        delay = torch.ones((2048, 2048), device=device, dtype=torch.bfloat16)
        copy_scratch = torch.empty_like(delay)
        compute_scratch = torch.empty_like(delay)
        torch.accelerator.synchronize(device)
        with offloader.copy_stream:
            for _ in range(32):
                torch.mm(delay, delay, out=copy_scratch)
        for _ in range(2):
            with offloader.run():
                for index in range(4):
                    with offloader.layer(index):
                        for _ in range(32):
                            torch.mm(delay, delay, out=compute_scratch)
                        outputs.append(model.blocks[index].weight.clone())
        torch.accelerator.synchronize(device)
        assert any(inflight_h2d), "no staging slot was reused while its H2D transfer was in flight"
        assert any(inflight_compute), "no staging slot was reused while its computation was in flight"
        for index, output in enumerate(outputs):
            assert torch.equal(output.cpu(), torch.full((8, 8), index % 4 + 1, dtype=torch.bfloat16))
    finally:
        offloader.deactivate()


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_read_failure_releases_staging_even_with_traceback(tmp_path, device):
    offloader, model = _make_offloader(tmp_path, device)
    original_weights = [layer.weight for layer in model.blocks]
    offloader.activate()
    cpu_storage = [weakref.ref(slot.untyped_storage()) for slot in offloader.cpu_slots]
    gpu_storage = [weakref.ref(slot.untyped_storage()) for slot in offloader.gpu_slots]
    # Validation succeeded. Truncating now makes the background read fail
    # while it holds a view into a pinned staging slot.
    (tmp_path / "weights.bin").write_bytes(b"")
    with pytest.raises(EOFError) as error:
        with offloader.run(), offloader.layer(0):
            pass
    gc.collect()
    assert error.value.__traceback__ is not None
    assert all(ref() is None for ref in cpu_storage + gpu_storage)
    assert not offloader.cpu_slots and not offloader.gpu_slots
    assert not offloader.handles and not offloader.futures
    assert offloader.executor is None and offloader.copy_stream is None
    assert all(layer.weight is original for layer, original in zip(model.blocks, original_weights))


def test_minimax_h3_rejects_cache_dit_with_file_backed_offload():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.models.minimax_h3.time_request import MINIMAX_H3_SHAPE_PLANNER
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
    from vllm_omni.errors import OmniClientError
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.model_executor.models.minimax_h3.conditioning import MiniMaxH3EncoderConditioning

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.load_text_encoder = False
    pipeline.load_vae_encoder = False
    pipeline.partition = "fl2va"
    pipeline.supported_tasks = frozenset({"t2va"})
    pipeline.default_video_shift = 12.0
    pipeline.default_audio_shift = 3.0
    pipeline.device = torch.device("cpu")
    pipeline.od_config = SimpleNamespace(enable_file_backed_offload=True)
    pipeline._quality_policy = Mock()
    pipeline._quality_policy.resolve.return_value = SimpleNamespace(cache_dit=object())
    pipeline._cache_dit_runtime = Mock()
    pipeline.diffuse = Mock()
    pipeline.decode = Mock()

    conditioning = MiniMaxH3EncoderConditioning(
        hidden_states=torch.ones(1, 5120, dtype=torch.bfloat16),
        token_tags=torch.ones(1, dtype=torch.int64),
        task="t2va",
        height=768,
        width=1344,
        num_frames=124,
        latent_t=MINIMAX_H3_SHAPE_PLANNER.video_latent_t(124),
        audio_t=MINIMAX_H3_SHAPE_PLANNER.audio_latent_t(124 / 24),
    )
    prompt = {
        "prompt": "file-backed Cache-DiT conflict",
        "additional_information": {"encoder_output": conditioning.to_omni_payload()},
    }
    sampling = OmniDiffusionSamplingParams(
        quality="high",
        width=1344,
        height=768,
        fps=24,
        num_frames=124,
        num_inference_steps=50,
        extra_args={"task": "t2va", "aspect_ratio": "16:9"},
    )
    batch = DiffusionRequestBatch(
        [OmniDiffusionRequest(prompt=prompt, sampling_params=sampling, request_id="file-backed-cache-dit")]
    )

    with pytest.raises(OmniClientError, match="Cache-DiT cannot be combined with file-backed offload"):
        pipeline.forward(batch)

    pipeline._cache_dit_runtime.prepare.assert_not_called()
    pipeline.diffuse.assert_not_called()
