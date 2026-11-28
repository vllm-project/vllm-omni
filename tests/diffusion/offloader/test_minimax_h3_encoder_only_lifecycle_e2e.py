# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Single-GPU acceptance: encoder-only lifecycle streams the text encoder.

This is an opt-in, real-checkpoint test. It is skipped unless
``VLLM_TEST_MINIMAX_H3_FL2VA_MODEL`` points at an authorized FL2VA partition,
because it reads the 63 GiB text encoder from disk and needs a CUDA device.

It asserts the P5 acceptance contract (RFC #10943 §enc-dec):

* ``model_loaded`` selects only the text encoder, so the pipeline never
  constructs the DiT or the VAEs and declares no DiT/VAE weight source;
* the loader still produces a checkpoint-mmap plan for the encoder alone;
* one encoder forward returns ``[seq, 5120]`` BF16 finite states with bounded
  host-RSS growth.

When ``VLLM_TEST_MINIMAX_H3_BENCHMARK_CONDITIONING`` points at a previously
saved ``text_conditioning.pt``, the test also asserts the encoder output is
bit-identical ``(torch.equal)`` to that saved intermediate.
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
BENCHMARK_ENV = "VLLM_TEST_MINIMAX_H3_BENCHMARK_CONDITIONING"
PORT = "29516"

# Matches docs/dev-flow/minimax-h3-conditioning-cache/benchmark-case.md §1.
BENCHMARK_PROMPT = "A tiny orange robot walks across a sunlit wooden table, stops, and waves at the camera."


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.skipif(
    not os.environ.get(MODEL_ENV),
    reason=f"set {MODEL_ENV} to an authorized FL2VA checkpoint path",
)
def test_text_encoder_streams_without_dit_or_vae() -> None:
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
    from vllm_omni.diffusion.offloader import enable_offload_backend
    from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config
    from vllm_omni.model_executor.models.minimax_h3.preprocessing import (
        build_minimax_h3_presentation,
    )

    if not torch.cuda.is_available():
        pytest.skip("MiniMax H3 encoder streaming requires a CUDA device")
    if model_parallel_is_initialized():
        pytest.skip("requires a process that has not initialized model parallel groups")

    device = torch.device("cuda:0")
    # Encoder-only lifecycle: no DiT, no VAE. The explicit ``weight_transfer``
    # keeps the resolved strategy distributed so the loader-owned mmap plan can
    # back the encoder (rank-local zero-resident resolves to plain LAYER_WISE,
    # which has no disk backing on a 19 GiB machine).
    od_config = OmniDiffusionConfig(
        model=os.environ[MODEL_ENV],
        model_class_name="MiniMaxH3Pipeline",
        dtype=torch.bfloat16,
        parallel_config=DiffusionParallelConfig(ulysses_degree=1),
        model_loaded={
            "transformer": False,
            "text_encoder": True,
            "vae_encoder": False,
            "vae_decoder": False,
        },
        diffusion_offload_config={
            "mode": "layer",
            "components": ["text_encoder"],
            "layer_options": {"text_encoder": {"weight_transfer": "allgather"}},
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
        pipeline, backend = enable_offload_backend(od_config, pipeline, device=device)

        # Lifecycle isolation: DiT and VAEs stay unconstructed.
        assert pipeline.transformer is None
        assert pipeline.vae is None
        assert pipeline.video_vae is None
        assert pipeline.audio_vae is None
        assert pipeline._dit_modules == []

        plan = loader.host_weight_plan
        assert plan is not None, "loader did not produce a checkpoint-mmap plan"
        encoder_bindings = [name for name in plan.bindings if name.startswith("text_encoder.")]
        assert encoder_bindings, "text encoder was not added to the mmap plan"
        assert not any(name.startswith("transformer.") for name in plan.bindings), (
            "DiT must not be planned when the lifecycle excludes it"
        )

        encoder = pipeline.text_encoder
        encoder.load_to_device()

        merge_size = int(pipeline.processor.image_processor.merge_size)
        ids, tags = build_minimax_h3_presentation(
            pipeline.tokenizer,
            prompt=BENCHMARK_PROMPT,
            task="t2va",
            condition_labels=[],
            image_grid_thw=None,
            video_grid_thw=None,
            video_timestamps=None,
            merge_size=merge_size,
        )

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

        benchmark_path = os.environ.get(BENCHMARK_ENV)
        if benchmark_path:
            reference = torch.load(benchmark_path, map_location="cpu", weights_only=False)
            ref_hidden = reference["hidden_states"]
            ref_tags = reference["token_tags"]
            assert ref_hidden.shape == hidden.shape, (
                f"benchmark shape {tuple(ref_hidden.shape)} != encoder output {tuple(hidden.shape)}"
            )
            assert torch.equal(hidden, ref_hidden), "encoder output diverged from the benchmark hidden_states"
            assert torch.equal(tags, ref_tags), "encoder token_tags diverged from the benchmark token_tags"

        # Bounded staging: the forward must not materialize the 48-63 GiB encoder.
        # VmRSS includes reclaimable file-backed page cache from the mmap stream,
        # so this bound is looser than the anon budget tracked by the cgroup guard.
        host_delta = max(samples) - samples[0]
        assert host_delta < 12.0, f"encoder forward grew host RSS by {host_delta:.2f} GiB"

        # Release the offloader's mappings and host registration while the
        # objects that own them are still reachable; dropping the references
        # first would leave the lease to the garbage collector.
        backend.shutdown()
        del pipeline, backend, loader
        gc.collect()
        torch.accelerator.empty_cache()


def _current_rss_gib() -> float:
    with open("/proc/self/status", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024 / 1024
    raise RuntimeError("VmRSS is unavailable")
