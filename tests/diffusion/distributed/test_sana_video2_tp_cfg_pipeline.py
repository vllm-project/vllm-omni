# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SANA-Video 2.0 native/parallel sampler trajectories with real process groups."""

import os
import socket
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from PIL import Image
from torch import nn


class _VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            latent_channels=128, temporal_compression_ratio=8, spatial_compression_ratio=32, scaling_factor=1.0
        )
        self.register_buffer("latents_mean", torch.zeros(128))
        self.register_buffer("latents_std", torch.ones(128))

    @property
    def dtype(self):
        return torch.float32

    def encode(self, pixels):
        shape = (pixels.shape[0], 128, 1, pixels.shape[-2] // 32, pixels.shape[-1] // 32)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: torch.zeros(shape)))


def _model():
    from vllm_omni.diffusion.models.sana_video2.transformer_sana_video2 import (
        SanaVideo2TransformerConfig,
        SanaVideo2TransformerModel,
    )

    torch.manual_seed(8102)
    config = SanaVideo2TransformerConfig(
        in_channels=128,
        hidden_size=120,
        depth=4,
        num_heads=20,
        caption_channels=16,
        model_max_length=4,
        linear_head_dim=6,
        softmax_head_dim=12,
        softmax_ratio=0.25,
        mlp_ratio=1.0,
        attn_res_block_size=2,
    )
    model = SanaVideo2TransformerModel(config).eval()
    for projection in (model.attn_res.attn_proj, model.attn_res.mlp_proj, model.attn_res.final_proj):
        torch.nn.init.normal_(projection.weight, std=0.1)
    return model


def _inputs(rank):
    generator = torch.Generator().manual_seed(8103 + rank)
    latents = torch.randn(1, 128, 3, 1, 3, generator=generator)
    positive = torch.full((1, 4, 16), 0.5 + rank)
    negative = torch.full((1, 4, 16), -0.25 - rank)
    mask = torch.tensor([[True, True, False, False]])
    return latents, positive, negative, mask


def _generate(pipeline, task, guidance, rank):
    latents, positive, negative, mask = _inputs(rank)
    steps = []

    def callback(_step, _time, value):
        steps.append(value.cpu().clone())

    output = pipeline.generate(
        image=Image.new("RGB", (96, 32), color="white") if task == "ti2v" else None,
        height=32,
        width=96,
        num_frames=17,
        num_inference_steps=2,
        guidance_scale=guidance,
        flow_shift=1.5,
        latents=latents,
        prompt_embeds=positive,
        negative_prompt_embeds=negative,
        prompt_attention_mask=mask,
        negative_prompt_attention_mask=mask,
        output_type="latent",
        callback=callback,
    )
    return output.cpu(), steps


def _warmup(pipeline, rank):
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    _, positive, negative, mask = _inputs(rank)
    pipeline.encode_prompt = lambda *_, **__: (positive, mask, negative, mask)
    sampling = OmniDiffusionSamplingParams(
        height=32,
        width=96,
        num_frames=9,
        num_inference_steps=1,
        guidance_scale=8.0,
        output_type="latent",
        seed=8105 + rank,
    )
    request = SimpleNamespace(prompts=["warmup"], is_dummy_run=lambda: True, sampling_params=sampling)
    return pipeline.forward(request).output.cpu()


def _worker(rank, tp, sp, cfg, port, directory):
    from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
    from vllm_omni.diffusion.distributed.parallel_state import (
        destroy_distributed_env,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm_omni.diffusion.forward_context import set_forward_context
    from vllm_omni.diffusion.models.sana_video2.pipeline_sana_video2 import SanaVideo2Pipeline

    torch.set_num_threads(1)
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(tp * sp * cfg),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
    )
    init_distributed_environment(local_rank=rank, backend="gloo")
    try:
        initialize_model_parallel(
            tensor_parallel_size=tp,
            sequence_parallel_size=sp,
            ulysses_degree=sp,
            ring_degree=1,
            cfg_parallel_size=cfg,
            backend="gloo",
        )
        parallel = DiffusionParallelConfig(
            tensor_parallel_size=tp,
            sequence_parallel_size=sp,
            ulysses_degree=sp,
            ring_degree=1,
            ulysses_mode="advanced_uaa" if sp > 1 else "strict",
            cfg_parallel_size=cfg,
        )
        config = OmniDiffusionConfig(model=directory, dtype=torch.float32, parallel_config=parallel)
        model = _model()
        model.load_weights(torch.load(f"{directory}/weights.pt", map_location="cpu", weights_only=True).items())
        components = (object(), nn.Identity(), _VAE(), model)
        with patch.object(SanaVideo2Pipeline, "_load_components", return_value=components):
            pipeline = SanaVideo2Pipeline(od_config=config)
        results = {}
        with torch.no_grad(), set_forward_context(omni_diffusion_config=config):
            for task in ("t2v", "ti2v"):
                for guidance in (1.0, 8.0):
                    results[(task, guidance)] = _generate(pipeline, task, guidance, rank)
            results["warmup"] = _warmup(pipeline, rank)
        torch.save(results, f"{directory}/rank_{rank}.pt")
    finally:
        destroy_distributed_env()


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.parallel
@pytest.mark.cpu
@pytest.mark.parametrize("tp,sp,cfg", [(1, 1, 2), (2, 1, 2), (1, 2, 2), (2, 2, 2)])
def test_parallel_pipeline_cfg_and_rank_invariants(tp, sp, cfg, tmp_path):
    import torch.multiprocessing as mp

    from vllm_omni.diffusion.models.sana_video2.pipeline_sana_video2 import SanaVideo2Pipeline

    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    model = _model()
    pipeline = SanaVideo2Pipeline(tokenizer=object(), text_encoder=nn.Identity(), vae=_VAE(), transformer=model)
    expected = {}
    serial_branch = {}
    with torch.no_grad():
        for task in ("t2v", "ti2v"):
            for guidance in (1.0, 8.0):
                expected[(task, guidance)] = _generate(pipeline, task, guidance, 0)
        if tp == sp == 1:
            pipeline.cfg_group = SimpleNamespace(broadcast=lambda value: value)
            for task in ("t2v", "ti2v"):
                for guidance in (1.0, 8.0):
                    serial_branch[(task, guidance)] = _generate(pipeline, task, guidance, 0)
            serial_branch["warmup"] = _warmup(pipeline, 0)
            pipeline.cfg_group = None
        expected["warmup"] = _warmup(pipeline, 0)
    torch.save(model.state_dict(), tmp_path / "weights.pt")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    context = mp.spawn(_worker, args=(tp, sp, cfg, port, str(tmp_path)), nprocs=tp * sp * cfg, join=False)
    try:
        deadline = time.monotonic() + 180
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                raise TimeoutError(f"TP{tp}+SP{sp}+CFG{cfg} workers exceeded 180 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    first = None
    for rank in range(tp * sp * cfg):
        actual = torch.load(tmp_path / f"rank_{rank}.pt", map_location="cpu", weights_only=True)
        if first is None:
            first = actual
        else:
            torch.testing.assert_close(actual["warmup"], first["warmup"], rtol=0, atol=0)
            for case in ((task, guidance) for task in ("t2v", "ti2v") for guidance in (1.0, 8.0)):
                output, steps = actual[case]
                first_output, first_steps = first[case]
                torch.testing.assert_close(output, first_output, rtol=0, atol=0)
                for observed, target in zip(steps, first_steps, strict=True):
                    torch.testing.assert_close(observed, target, rtol=0, atol=0)
        warmup_targets = [("native", expected["warmup"])]
        if serial_branch:
            warmup_targets.append(("serial", serial_branch["warmup"]))
        for label, target in warmup_targets:
            warmup_delta = (actual["warmup"].double() - target.double()).flatten()
            warmup_absolute = warmup_delta.abs().max().item()
            warmup_relative = (warmup_delta.norm() / target.double().flatten().norm().clamp_min(1e-12)).item()
            if rank == 0:
                print(
                    f"TP{tp} SP{sp} CFG{cfg} warmup {label} max_abs={warmup_absolute:.3e} rel_l2={warmup_relative:.3e}"
                )
            if tp == sp == 1 and label == "serial":
                torch.testing.assert_close(actual["warmup"], target, rtol=0, atol=0)
        for case, value in expected.items():
            if case == "warmup":
                continue
            reference, reference_steps = value
            output, steps = actual[case]
            targets = [("native", reference, reference_steps)]
            if serial_branch:
                split_output, split_steps = serial_branch[case]
                targets.append(("serial", split_output, split_steps))
            for label, target_output, target_steps in targets:
                for index, (observed, target) in enumerate([*zip(steps, target_steps), (output, target_output)]):
                    delta = (observed.double() - target.double()).flatten()
                    absolute = delta.abs().max().item()
                    relative = (delta.norm() / target.double().flatten().norm().clamp_min(1e-12)).item()
                    if rank == 0:
                        print(
                            f"TP{tp} SP{sp} CFG{cfg} {case} {label} step={index} "
                            f"max_abs={absolute:.3e} rel_l2={relative:.3e}"
                        )
                    if tp == sp == 1 and label == "serial":
                        torch.testing.assert_close(observed, target, rtol=0, atol=0)
