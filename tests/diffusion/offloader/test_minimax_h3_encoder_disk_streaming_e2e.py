# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Single-GPU acceptance: the MiniMax-H3 text encoder streams from disk.

This is an opt-in, real-checkpoint test. It is skipped unless
``VLLM_TEST_MINIMAX_H3_FL2VA_MODEL`` points at an authorized FL2VA partition,
because it reads the 63 GiB text encoder from disk and needs a CUDA device.

It asserts the PR3 acceptance contract: the loader produces a checkpoint-mmap
plan that includes the text encoder (with fused ``qkv_proj``/``gate_up_proj``
bindings), the backend streams it in bounded staging slots, and one encoder
forward returns ``[seq, 5120]`` BF16 finite states.

Those assertions cannot detect a permuted fused source mapping (for example
swapped K/V slices), so an optional golden comparison pins the model-specific
checkpoint mapping: set ``VLLM_TEST_MINIMAX_H3_ENCODER_REFERENCE`` to a file
holding the hidden states for this test's fixed token IDs. The reference must
come from the ordinary (non-streaming) loader on the same device kind, since
streamed and ordinary forwards are bit-identical only on the same device. Set
``VLLM_TEST_MINIMAX_H3_ENCODER_REFERENCE_RECORD`` instead to record the
reference: that run loads the ordinary pipeline, releases the DiT/VAE, moves
the encoder to the device, and saves one forward. Recording needs a device
with room for the whole 63 GiB encoder, which is exactly the memory the
streaming path avoids.
"""

import gc
import os

import pytest
import torch

from tests.helpers.mark import hardware_test

pytestmark = [
    pytest.mark.local_model,
    pytest.mark.diffusion,
    pytest.mark.parallel,
]

MODEL_ENV = "VLLM_TEST_MINIMAX_H3_FL2VA_MODEL"
REFERENCE_ENV = "VLLM_TEST_MINIMAX_H3_ENCODER_REFERENCE"
RECORD_ENV = "VLLM_TEST_MINIMAX_H3_ENCODER_REFERENCE_RECORD"
PORT = "29514"
IDS = (100, 132)


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.skipif(
    not os.environ.get(MODEL_ENV),
    reason=f"set {MODEL_ENV} to an authorized FL2VA checkpoint path",
)
def test_text_encoder_streams_from_disk() -> None:
    from vllm.config import set_current_vllm_config
    from vllm.config.load import LoadConfig

    from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
    from vllm_omni.diffusion.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )
    from vllm_omni.diffusion.forward_context import set_forward_context
    from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
    from vllm_omni.diffusion.model_loader.host_weight_plan import FusedTensorBinding
    from vllm_omni.diffusion.offloader import enable_offload_backend
    from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config

    if not torch.cuda.is_available():
        pytest.skip("MiniMax H3 encoder streaming requires a CUDA device")
    if model_parallel_is_initialized():
        pytest.skip("requires a process that has not initialized model parallel groups")

    device = torch.device("cuda:0")
    od_config = OmniDiffusionConfig(
        model=os.environ[MODEL_ENV],
        model_class_name="MiniMaxH3Pipeline",
        dtype=torch.bfloat16,
        parallel_config=DiffusionParallelConfig(ulysses_degree=1),
        diffusion_offload_config=None
        if os.environ.get(RECORD_ENV)
        else {
            "mode": "layer",
            "components": ["dit", "text_encoder"],
            "layer_options": {"dit": {"resident_layers": 1}},
        },
        enforce_eager=True,
        trust_remote_code=True,
    )
    vllm_config = create_diffusion_vllm_config(device, od_config)
    with (
        set_forward_context(vllm_config=vllm_config, omni_diffusion_config=od_config),
        set_current_vllm_config(vllm_config),
    ):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{PORT}",
        )
        initialize_model_parallel(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            sequence_parallel_size=1,
            ulysses_degree=1,
            ring_degree=1,
        )
        loader = DiffusersPipelineLoader(LoadConfig(), od_config=od_config)
        pipeline = loader.load_model(load_device="cpu", device=device)
        record_path = os.environ.get(RECORD_ENV)
        if record_path:
            _record_reference(pipeline, device, record_path)
            return
        pipeline, backend = enable_offload_backend(od_config, pipeline, device=device)
        plan = loader.host_weight_plan
        assert plan is not None, "loader did not produce a checkpoint-mmap plan"
        encoder_bindings = {
            name: binding for name, binding in plan.bindings.items() if name.startswith("text_encoder.")
        }
        assert encoder_bindings, "text encoder was not added to the mmap plan"
        qkv = encoder_bindings["text_encoder.text_model.layers.0.self_attn.qkv_proj.weight"]
        assert isinstance(qkv, FusedTensorBinding)
        assert len(qkv.checkpoint_keys) == 3

        # The VAE is not part of the acceptance; release every reference to it
        # so the staging ring has room on a 19 GiB machine.
        pipeline.video_vae = None
        pipeline.audio_vae = None
        pipeline.vae = None
        pipeline._vae_modules = []
        pipeline._model_cpu_offload_modules = []
        backend._staged_components.clear()
        gc.collect()
        torch.accelerator.empty_cache()

        encoder = pipeline.text_encoder
        encoder.load_to_device()
        ids = torch.arange(*IDS, dtype=torch.long, device=device)

        import threading
        import time

        samples: list[float] = []
        stop = threading.Event()

        def sample_rss() -> None:
            while not stop.is_set():
                samples.append(_current_rss_gib())
                time.sleep(0.02)

        sampler = threading.Thread(target=sample_rss, daemon=True)
        sampler.start()
        try:
            with torch.no_grad():
                hidden = encoder(ids)
            torch.accelerator.synchronize()
        finally:
            stop.set()
            sampler.join()

        assert tuple(hidden.shape) == (ids.numel(), 5120)
        assert hidden.dtype is torch.bfloat16
        assert torch.isfinite(hidden).all()
        # Bounded staging: the forward must not materialize the 48 GiB encoder.
        host_delta = max(samples) - samples[0]
        assert host_delta < 4.0, f"encoder forward grew host RSS by {host_delta:.2f} GiB"

        # The golden comparison pins the fused source mapping: equal shapes and
        # finite states still pass if K/V slices are swapped, but the ordinary
        # loader's hidden states for these token IDs do not.
        reference_path = os.environ.get(REFERENCE_ENV)
        if reference_path:
            expected = torch.load(reference_path, map_location=device, weights_only=True)
            assert torch.equal(hidden, expected), "streamed encoder states diverge from the ordinary-loader reference"


def _record_reference(pipeline, device: torch.device, path: str) -> None:
    """Record the ordinary-loader hidden states for the fixed token IDs.

    The ordinary path has no component isolation, so every non-encoder
    reference is dropped before the move. Recording needs a device with room
    for the whole 63 GiB encoder, which is exactly the memory the streaming
    path avoids; the host must still hold the full pipeline during the load.
    """
    pipeline.transformer = None
    pipeline.video_vae = None
    pipeline.audio_vae = None
    pipeline.vae = None
    pipeline._vae_modules = []
    pipeline._model_cpu_offload_modules = []
    gc.collect()
    torch.accelerator.empty_cache()

    encoder = pipeline.text_encoder
    encoder.to(device)
    ids = torch.arange(*IDS, dtype=torch.long, device=device)
    with torch.no_grad():
        hidden = encoder(ids)
    torch.accelerator.synchronize()
    torch.save(hidden.cpu(), path)


def _current_rss_gib() -> float:
    with open("/proc/self/status", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024 / 1024
    raise RuntimeError("VmRSS is unavailable")
