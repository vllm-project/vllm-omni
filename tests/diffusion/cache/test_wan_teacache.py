# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig
from vllm_omni.diffusion.cache.teacache.hook import apply_teacache_hook
from vllm_omni.diffusion.data import DiffusionCacheConfig


@pytest.mark.core_model
@pytest.mark.cpu
@pytest.mark.parametrize(
    "attribute,value",
    [("has_transformer_2", True), ("expand_timesteps", True), ("is_dmd", True), ("transformer", None)],
)
def test_wan_rejects_unvalidated_variants(attribute, value):
    from vllm_omni.diffusion.cache.teacache.backend import enable_wan_teacache

    pipeline = SimpleNamespace(
        has_transformer_2=False,
        expand_timesteps=False,
        is_dmd=False,
        transformer=SimpleNamespace(
            config=SimpleNamespace(in_channels=16, num_layers=30, num_attention_heads=12, attention_head_dim=128)
        ),
        od_config=SimpleNamespace(parallel_config=None),
    )
    setattr(pipeline, attribute, value)
    with pytest.raises(ValueError, match="single-expert"):
        enable_wan_teacache(pipeline, DiffusionCacheConfig())


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(dtype, marks=hardware_marks(res={"cuda": ["B200"]}, num_cards=1))
        for dtype in (torch.float32, torch.bfloat16)
    ],
)
@pytest.mark.parametrize("reuse", [False, True])
def test_wan_hook_full_compute_matches_native_forward(dtype, reuse, monkeypatch):
    from vllm.utils.network_utils import get_file_store_init_method

    from tests.diffusion.distributed import test_pipeline_parallel as pp
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec, OmniDiffusionConfig
    from vllm_omni.diffusion.forward_context import set_forward_context
    from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import WanTransformer3DModel

    device = pp.init_dist(0, 1, get_file_store_init_method(), "cuda")
    try:
        pp.initialize_model_parallel(pipeline_parallel_size=1, backend="nccl")
        od = OmniDiffusionConfig(model="unused", dtype=dtype)
        if dtype == torch.float32:
            od.diffusion_attention_config = AttentionConfig(default=AttentionSpec(backend="TORCH_SDPA"))
        with set_current_diffusion_config(od), set_forward_context(omni_diffusion_config=od), torch.inference_mode():
            model = (
                WanTransformer3DModel(
                    num_attention_heads=2, attention_head_dim=64, text_dim=32, freq_dim=32, ffn_dim=256, num_layers=2
                )
                .to(device=device, dtype=dtype)
                .eval()
            )
            # vLLM linear parameters are allocated empty until checkpoint loading.
            torch.manual_seed(17)
            for name, parameter in model.named_parameters():
                if parameter.ndim == 1 and name.endswith("weight"):
                    torch.nn.init.ones_(parameter)
                else:
                    torch.nn.init.normal_(parameter, std=0.02)
            args = dict(
                hidden_states=torch.randn(1, 16, 1, 4, 4, device=device, dtype=dtype),
                timestep=torch.tensor([500], device=device),
                encoder_hidden_states=torch.randn(1, 4, 32, device=device, dtype=dtype),
                return_dict=False,
            )
            expected = model(**args)[0]
            assert torch.isfinite(expected).all()
            calls = []
            run_blocks = model._run_local_blocks

            def counted(*args, **kwargs):
                calls.append(model.cfg_branch)
                return run_blocks(*args, **kwargs)

            monkeypatch.setattr(model, "_run_local_blocks", counted)
            apply_teacache_hook(
                model,
                TeaCacheConfig(transformer_type="WanTransformer3DModel", coefficients=[0, 0, 0, 0, 0 if reuse else 1]),
            )
            model.cfg_branch = "positive"
            for _ in range(2):
                tolerance = 1e-5 if dtype == torch.float32 else 1e-2
                torch.testing.assert_close(model(**args)[0], expected, rtol=tolerance, atol=tolerance)
            assert calls == ["positive"] * (1 if reuse else 2)
    finally:
        pp._cleanup_distributed()
