# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CFG batching parity on the pipelines that opt in, with tiny random-weight transformers.

Each case builds the branch kwargs the way the pipeline's denoise loop does
(shared latents, per-branch prompt embeds and position ids) and runs the
pipeline class's own predict_noise_maybe_with_cfg / predict_noise once with
CFG batching and once without. The batched run must make one transformer
call and match the sequential result.
"""

from collections.abc import Callable
from types import SimpleNamespace

import pytest
import torch
from vllm.config import set_current_vllm_config

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]

DTYPE = torch.float32


def _device() -> torch.device:
    accelerator = torch.accelerator.current_accelerator()
    return torch.device(accelerator.type, 0) if accelerator is not None else torch.device("cpu")


def _randn(g: torch.Generator, *shape: int) -> torch.Tensor:
    return torch.randn(*shape, generator=g, device=g.device, dtype=DTYPE)


def _sd3(od_config, g):
    from vllm_omni.diffusion.models.sd3.pipeline_sd3 import StableDiffusion3Pipeline
    from vllm_omni.diffusion.models.sd3.sd3_transformer import SD3Transformer2DModel

    od_config.tf_model_config = TransformerConfig.from_dict(
        {
            "sample_size": 8,
            "patch_size": 2,
            "in_channels": 4,
            "out_channels": 4,
            "num_layers": 2,
            "attention_head_dim": 16,
            "num_attention_heads": 2,
            "joint_attention_dim": 32,
            "caption_projection_dim": 32,
            "pooled_projection_dim": 16,
            "pos_embed_max_size": 16,
        }
    )
    model = SD3Transformer2DModel(od_config=od_config)

    def branch(batch, latents, timestep):
        return {
            "hidden_states": latents,
            "timestep": timestep,
            "encoder_hidden_states": _randn(g, batch, 12, 32),
            "pooled_projections": _randn(g, batch, 16),
            "return_dict": False,
        }

    def make(batch):
        latents, timestep = _randn(g, batch, 4, 8, 8), torch.full((batch,), 500.0, device=g.device)
        return branch(batch, latents, timestep), branch(batch, latents, timestep)

    return StableDiffusion3Pipeline, model, make


def _sdxl(od_config, g):
    from vllm_omni.diffusion.models.sdxl.pipeline_sdxl import StableDiffusionXLPipeline
    from vllm_omni.diffusion.models.sdxl.sdxl_unet import SDXLUNet2DConditionModel

    # The UNet has the fixed SDXL-base layout (~2.6B params); build it on the device.
    with torch.device(g.device):
        model = SDXLUNet2DConditionModel(od_config=od_config)

    def make(batch):
        latents, timestep = _randn(g, batch, 4, 16, 16), torch.full((batch,), 500.0, device=g.device)
        # diffuse() shares one add_time_ids tensor between the branches.
        time_ids = torch.tensor([[64.0, 64.0, 0.0, 0.0, 64.0, 64.0]], device=g.device).repeat(batch, 1)
        return [
            {
                "hidden_states": latents,
                "timestep": timestep,
                "encoder_hidden_states": _randn(g, batch, 77, 2048),
                "added_cond_kwargs": {"text_embeds": _randn(g, batch, 1280), "time_ids": time_ids},
                "return_dict": False,
            }
            for _ in range(2)
        ]

    return StableDiffusionXLPipeline, model, make


def _flux(kontext: bool):
    def build(od_config, g):
        from vllm_omni.diffusion.models.flux.flux_transformer import FluxTransformer2DModel

        if kontext:
            from vllm_omni.diffusion.models.flux.pipeline_flux_kontext import FluxKontextPipeline as PipelineCls
        else:
            from vllm_omni.diffusion.models.flux.pipeline_flux import FluxPipeline as PipelineCls
        model = FluxTransformer2DModel(
            num_layers=1,
            num_single_layers=1,
            num_attention_heads=2,
            attention_head_dim=32,
            joint_attention_dim=64,
            pooled_projection_dim=64,
            axes_dims_rope=(8, 12, 12),
            guidance_embeds=False,
        )

        def make(batch):
            n_img = 32 if kontext else 16  # Kontext appends the reference image tokens
            latents, timestep = _randn(g, batch, n_img, 64), torch.full((batch,), 0.5, device=g.device)
            img_ids = torch.rand(n_img, 3, generator=g, device=g.device).mul(4).floor()
            out = []
            for _ in range(2):
                out.append(
                    {
                        "hidden_states": latents,
                        "timestep": timestep,
                        "guidance": None,
                        "pooled_projections": _randn(g, batch, 64),
                        "encoder_hidden_states": _randn(g, batch, 8, 64),
                        # Built per branch from that branch's prompt length, like encode_prompt does.
                        "txt_ids": torch.zeros(8, 3, device=g.device),
                        "img_ids": img_ids,
                        "joint_attention_kwargs": None,
                        "return_dict": False,
                    }
                )
            return out

        return PipelineCls, model, make

    return build


def _flux2(klein: bool):
    def build(od_config, g):
        if klein:
            from vllm_omni.diffusion.models.flux2_klein.flux2_klein_transformer import Flux2Transformer2DModel
            from vllm_omni.diffusion.models.flux2_klein.pipeline_flux2_klein import Flux2KleinPipeline as PipelineCls
        else:
            from vllm_omni.diffusion.models.flux2.flux2_transformer import Flux2Transformer2DModel
            from vllm_omni.diffusion.models.flux2.pipeline_flux2 import Flux2Pipeline as PipelineCls
        model = Flux2Transformer2DModel(
            in_channels=16,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=32,
            num_attention_heads=2,
            joint_attention_dim=48,
            timestep_guidance_channels=32,
            axes_dims_rope=(8, 8, 8, 8),
            guidance_embeds=not klein,
        )

        def ids(batch, n):
            return torch.rand(n, 4, generator=g, device=g.device).mul(4).floor().unsqueeze(0).expand(batch, -1, -1)

        def make(batch):
            latents, timestep = _randn(g, batch, 16, 16), torch.full((batch,), 0.5, device=g.device)
            guidance = None if klein else torch.full((batch,), 4.0, device=g.device)
            img_ids = ids(batch, 16)
            return [
                {
                    "hidden_states": latents,
                    "timestep": timestep,
                    "guidance": guidance,
                    "encoder_hidden_states": _randn(g, batch, 8, 48),
                    "txt_ids": ids(batch, 8),
                    "img_ids": img_ids,
                    "joint_attention_kwargs": None,
                    "return_dict": False,
                }
                for _ in range(2)
            ]

        return PipelineCls, model, make

    return build


def _wan(i2v: bool):
    def build(od_config, g):
        from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import WanTransformer3DModel

        if i2v:
            from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_i2v import Wan22I2VPipeline as PipelineCls
        else:
            from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import Wan22Pipeline as PipelineCls
        extra = {"image_dim": 16, "added_kv_proj_dim": 32} if i2v else {}
        model = WanTransformer3DModel(
            num_attention_heads=2,
            attention_head_dim=16,
            in_channels=8,
            out_channels=4,
            text_dim=16,
            freq_dim=16,
            ffn_dim=64,
            num_layers=2,
            **extra,
        )

        def make(batch):
            latents = _randn(g, batch, 8, 3, 8, 8)  # I2V-style channel concat of the condition
            timestep = torch.full((batch,), 500.0, device=g.device)
            image_embeds = _randn(g, batch, 4, 16) if i2v else None
            out = []
            for _ in range(2):
                kwargs = {
                    "hidden_states": latents,
                    "timestep": timestep,
                    # The I2V cross-attention splits image tokens off a fixed 512-token text context.
                    "encoder_hidden_states": _randn(g, batch, 512 if i2v else 12, 16),
                    "attention_kwargs": {},
                    "return_dict": False,
                    "current_model": model,
                }
                if i2v:
                    kwargs["encoder_hidden_states_image"] = image_embeds
                out.append(kwargs)
            out[1]["attention_kwargs"] = out[0]["attention_kwargs"]
            return out

        return PipelineCls, model, make

    return build


def _wan_vace(od_config, g):
    from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_vace import Wan22VACEPipeline
    from vllm_omni.diffusion.models.wan2_2.wan2_2_vace_transformer import WanVACETransformer3DModel

    model = WanVACETransformer3DModel(
        num_attention_heads=2,
        attention_head_dim=16,
        in_channels=4,
        out_channels=4,
        text_dim=16,
        freq_dim=16,
        ffn_dim=64,
        num_layers=2,
        vace_layers=[0, 1],
        vace_in_channels=8,
    )

    def make(batch):
        latents, timestep = _randn(g, batch, 4, 3, 8, 8), torch.full((batch,), 500.0, device=g.device)
        # diffuse() passes the same VACE condition and attention_kwargs to both branches.
        vace_context, attention_kwargs = _randn(g, batch, 8, 3, 8, 8), {}
        return [
            {
                "hidden_states": latents,
                "timestep": timestep,
                "encoder_hidden_states": _randn(g, batch, 12, 16),
                "attention_kwargs": attention_kwargs,
                "vace_context": vace_context,
                "vace_context_scale": 1.0,
                "return_dict": False,
                "current_model": model,
            }
            for _ in range(2)
        ]

    return Wan22VACEPipeline, model, make


CASES: dict[str, Callable] = {
    "sd3": _sd3,
    "sdxl": _sdxl,
    "flux": _flux(kontext=False),
    "flux-kontext": _flux(kontext=True),
    "flux2": _flux2(klein=False),
    "flux2-klein": _flux2(klein=True),
    "wan2.2-t2v": _wan(i2v=False),
    "wan2.2-i2v": _wan(i2v=True),
    "wan2.2-vace": _wan_vace,
}


@pytest.fixture
def diffusion_env(tmp_path):
    device = _device()
    # fp32 keeps batch-1 vs batch-2 GEMM differences far below the tolerance;
    # SDPA is the attention backend that takes fp32 everywhere.
    od_config = OmniDiffusionConfig.from_kwargs(
        dtype=DTYPE, diffusion_attention_config={"default": "TORCH_SDPA"}, linear_backend="TORCH"
    )
    vllm_config = create_diffusion_vllm_config(device, od_config)
    init_distributed_environment(
        world_size=1, rank=0, local_rank=0, distributed_init_method=f"file://{tmp_path / 'distributed'}"
    )
    initialize_model_parallel()
    try:
        with set_current_vllm_config(vllm_config), set_current_diffusion_config(od_config):
            yield device, od_config, vllm_config
    finally:
        destroy_distributed_env()


def _pipeline(cls, model: torch.nn.Module, batched: bool):
    """The pipeline class around ``model`` only: no weights, encoders or scheduler."""
    pipe = cls.__new__(cls)
    torch.nn.Module.__init__(pipe)
    # SDXL's denoiser is ``unet``; the others call it ``transformer``.
    setattr(pipe, (getattr(cls, "_dit_modules", None) or ["transformer"])[0], model)
    pipe.od_config = SimpleNamespace(enable_cfg_batching=batched, cache_backend="none")
    return pipe


@hardware_test(res={"cuda": "L4", "xpu": "B60"})
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("case", sorted(CASES))
def test_cfg_batching_matches_sequential(diffusion_env, case, batch):
    device, od_config, vllm_config = diffusion_env
    g = torch.Generator(device=device).manual_seed(0)
    cls, model, make = CASES[case](od_config, g)
    assert cls.supports_cfg_batching
    model = model.to(device=device, dtype=DTYPE).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(mean=0.0, std=0.05, generator=g)
    positive, negative = make(batch)

    calls: list[int] = []
    model.register_forward_pre_hook(
        lambda m, args, kwargs: calls.append(kwargs["hidden_states"].shape[0]), with_kwargs=True
    )
    out = {}
    with torch.inference_mode(), set_forward_context(vllm_config=vllm_config, omni_diffusion_config=od_config):
        for batched in (True, False):
            calls.clear()
            pipe = _pipeline(cls, model, batched)
            out[batched] = pipe.predict_noise_maybe_with_cfg(True, 4.0, positive, negative, cfg_normalize=False)
            assert calls == ([2 * batch] if batched else [batch, batch]), calls

    torch.testing.assert_close(out[True], out[False], atol=1e-4, rtol=1e-4)
    assert torch.isfinite(out[True]).all()
