# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.cache.teacache.extractors import CacheContext, get_extractor
from vllm_omni.diffusion.models.sd3.pipeline_sd3 import StableDiffusion3Pipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_sd3_extractor_returns_valid_context() -> None:
    class _DummyBlock(nn.Module):
        def norm1(self, hidden_states: torch.Tensor, emb: torch.Tensor):
            # Return a tuple like AdaLayerNormZero does: first item is the
            # modulated/normed hidden states used for cache decision.
            return (hidden_states + 0.1, torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1))

        def forward(
            self,
            *,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            temb: torch.Tensor,
        ):
            # Preserve shapes; just add a tiny delta so residual caching is valid.
            return encoder_hidden_states, hidden_states + 0.01

    class _DummySD3Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_size = 2
            self.out_channels = 4
            self.inner_dim = 3
            self.transformer_blocks = nn.ModuleList([_DummyBlock()])
            self.proj_out = nn.Linear(self.inner_dim, self.patch_size * self.patch_size * self.out_channels, bias=False)

        def pos_embed(self, hidden_states: torch.Tensor) -> torch.Tensor:
            # Convert (B, C, H, W) -> (B, num_patches, inner_dim).
            batch, _, height, width = hidden_states.shape
            hp = height // self.patch_size
            wp = width // self.patch_size
            return torch.zeros((batch, hp * wp, self.inner_dim), dtype=hidden_states.dtype, device=hidden_states.device)

        def time_text_embed(self, timestep: torch.Tensor, pooled_projections: torch.Tensor) -> torch.Tensor:
            _ = pooled_projections
            return torch.zeros((timestep.shape[0], self.inner_dim), dtype=timestep.dtype, device=timestep.device)

        def context_embedder(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
            return encoder_hidden_states

        def norm_out(self, hidden_states: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
            _ = temb
            return hidden_states

    extractor = get_extractor("SD3Transformer2DModel")
    module = _DummySD3Transformer()

    # Minimal inputs mirroring SD3Transformer2DModel.forward signature.
    hidden_states = torch.zeros((1, 4, 4, 4))
    encoder_hidden_states = torch.zeros((1, 2, 3))
    pooled_projections = torch.zeros((1, 3))
    timestep = torch.tensor([1])

    # return_dict=True: output is a Transformer2DModelOutput and supports output[0]
    ctx = extractor(
        module,
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        pooled_projections=pooled_projections,
        timestep=timestep,
        return_dict=True,
    )
    assert isinstance(ctx, CacheContext)
    ctx.validate()

    (token_states,) = ctx.run_transformer_blocks()
    output = ctx.postprocess(token_states)

    assert hasattr(output, "sample")
    assert output.sample.shape == (1, 4, 4, 4)
    assert output[0].shape == (1, 4, 4, 4)

    # return_dict=False: output is a 1-tuple and supports output[0] (pipeline-style)
    ctx = extractor(
        module,
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        pooled_projections=pooled_projections,
        timestep=timestep,
        return_dict=False,
    )
    (token_states,) = ctx.run_transformer_blocks()
    output = ctx.postprocess(token_states)

    assert isinstance(output, tuple)
    assert len(output) == 1
    assert output[0].shape == (1, 4, 4, 4)


@pytest.mark.parametrize("do_true_cfg", [True, False])
def test_sd3_diffuse_marks_cfg_branches_for_teacache(do_true_cfg: bool) -> None:
    """The TeaCache hook reads transformer.do_true_cfg to keep positive/negative states apart."""
    transformer = SimpleNamespace(do_true_cfg=None)
    flags_seen_by_predict: list[bool | None] = []

    def predict_noise_maybe_with_cfg(do_cfg, guidance_scale, positive_kwargs, negative_kwargs, cfg_normalize):
        flags_seen_by_predict.append(transformer.do_true_cfg)
        assert (negative_kwargs is not None) == do_cfg
        return torch.zeros_like(positive_kwargs["hidden_states"])

    pipeline = SimpleNamespace(
        scheduler=SimpleNamespace(set_begin_index=lambda index: None),
        interrupt=False,
        device=torch.device("cpu"),
        od_config=SimpleNamespace(dtype=torch.float32),
        transformer=transformer,
        predict_noise_maybe_with_cfg=predict_noise_maybe_with_cfg,
        scheduler_step_maybe_with_cfg=lambda noise_pred, t, latents, do_cfg: latents,
    )
    embeds = torch.zeros((1, 2, 3))
    pooled = torch.zeros((1, 3))

    StableDiffusion3Pipeline.diffuse(
        pipeline,
        latents=torch.zeros((1, 4, 4, 4)),
        timesteps=torch.tensor([1000.0, 500.0, 1.0]),
        prompt_embeds=embeds,
        pooled_prompt_embeds=pooled,
        negative_prompt_embeds=embeds if do_true_cfg else None,
        negative_pooled_prompt_embeds=pooled if do_true_cfg else None,
        do_true_cfg=do_true_cfg,
        guidance_scale=4.5 if do_true_cfg else 1.0,
    )

    assert flags_seen_by_predict == [do_true_cfg] * 3
