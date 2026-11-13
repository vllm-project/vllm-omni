# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The FLUX RoPE table computed once per request and passed into the forward.

The table depends only on the position ids, so the FLUX pipelines compute it
once with ``rotary_emb`` and pass it to every step as ``image_rotary_emb``; the
transformer must compute exactly what it computes from the ids itself.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_omni.diffusion.models.flux.flux_transformer import (
    FluxKontextTransformer2DModel,
    FluxTransformer2DModel,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_TXT, _IMG = 8, 16


@pytest.fixture(autouse=True)
def _cpu_single_rank(monkeypatch):
    """CPU custom-op dispatch, torch SDPA attention and a one-rank TP group."""
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

    from vllm_omni.diffusion.attention.backends import abstract
    from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend
    from vllm_omni.platforms.interface import OmniPlatformEnum, UnspecifiedOmniPlatform

    class _OutOfTreePlatform(UnspecifiedOmniPlatform):
        _omni_enum = OmniPlatformEnum.OOT

    # The attention dispatches on the platform; an out-of-tree one runs torch SDPA on CPU.
    monkeypatch.setattr(abstract, "current_omni_platform", _OutOfTreePlatform())
    tp_group = MagicMock(world_size=1, rank_in_group=0)
    with (
        set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))),
        patch("vllm_omni.diffusion.attention.layer.get_attn_backend_for_role", return_value=(SDPABackend, None)),
        patch("vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size", return_value=1),
        patch("vllm.distributed.parallel_state.get_tp_group", return_value=tp_group),
    ):
        yield


def _model(cls):
    torch.manual_seed(0)
    model = cls(
        num_layers=1,
        num_single_layers=1,
        num_attention_heads=2,
        attention_head_dim=16,
        joint_attention_dim=32,
        pooled_projection_dim=16,
        axes_dims_rope=(4, 4, 8),
    ).eval()
    # vLLM's parallel linears allocate their weights uninitialized.
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0.0, 0.02)
    return model


def _inputs():
    g = torch.Generator().manual_seed(1)
    return {
        "hidden_states": torch.randn(1, _IMG, 64, generator=g),
        "encoder_hidden_states": torch.randn(1, _TXT, 32, generator=g),
        "pooled_projections": torch.randn(1, 16, generator=g),
        "timestep": torch.tensor([0.5]),
        "img_ids": torch.randint(0, 8, (_IMG, 3), generator=g).float(),
        "txt_ids": torch.zeros(_TXT, 3),
        "guidance": torch.tensor([3.5]),
        "return_dict": False,
    }


@pytest.mark.parametrize("cls", [FluxTransformer2DModel, FluxKontextTransformer2DModel])
@torch.no_grad()
def test_passed_table_matches_table_from_ids(cls):
    model = _model(cls)
    inputs = _inputs()

    from_ids = model(**inputs)[0]
    table = model.rotary_emb(inputs["txt_ids"], inputs["img_ids"], dtype=inputs["hidden_states"].dtype)
    passed = model(**inputs, image_rotary_emb=table)[0]

    torch.testing.assert_close(passed, from_ids, rtol=0, atol=0)


@pytest.mark.parametrize("cls", [FluxTransformer2DModel, FluxKontextTransformer2DModel])
@torch.no_grad()
def test_passed_table_skips_position_embedding(cls):
    model = _model(cls)
    inputs = _inputs()
    table = model.rotary_emb(inputs["txt_ids"], inputs["img_ids"])

    with patch.object(model.pos_embed, "forward", side_effect=AssertionError("pos_embed evaluated")):
        model(**inputs, image_rotary_emb=table)


def test_rotary_emb_reads_deprecated_batched_ids():
    model = _model(FluxTransformer2DModel)
    inputs = _inputs()

    unbatched = model.rotary_emb(inputs["txt_ids"], inputs["img_ids"])
    batched = model.rotary_emb(inputs["txt_ids"][None], inputs["img_ids"][None])

    for a, b in zip(unbatched, batched):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_rotary_emb_casts_to_requested_dtype():
    model = _model(FluxTransformer2DModel)
    inputs = _inputs()

    cos, sin = model.rotary_emb(inputs["txt_ids"], inputs["img_ids"], dtype=torch.bfloat16)

    assert cos.dtype == sin.dtype == torch.bfloat16
    # pos_embed returns [S, head_dim / 2] cos and sin tables.
    assert cos.shape == sin.shape == (_TXT + _IMG, 8)


@pytest.mark.parametrize("kontext", [False, True], ids=["flux", "kontext"])
def test_cfg_batching_shares_the_table(kontext):
    """The table is unbatched like the ids, so CFG batching passes one table, not a concatenation."""
    if kontext:
        from vllm_omni.diffusion.models.flux.pipeline_flux_kontext import FluxKontextPipeline as PipelineCls
    else:
        from vllm_omni.diffusion.models.flux.pipeline_flux import FluxPipeline as PipelineCls

    pipeline = object.__new__(PipelineCls)
    table = (torch.randn(_TXT + _IMG, 8), torch.randn(_TXT + _IMG, 8))
    negative_table = (table[0].clone(), table[1].clone())
    positive = {"hidden_states": torch.randn(1, _IMG, 64), "image_rotary_emb": table}
    negative = {"hidden_states": torch.randn(1, _IMG, 64), "image_rotary_emb": negative_table}

    merged, n_pos, n_neg = PipelineCls.merge_cfg_branch_kwargs(pipeline, positive, negative)

    assert (n_pos, n_neg) == (1, 1)
    assert merged["image_rotary_emb"] is table
    assert merged["hidden_states"].shape[0] == 2
