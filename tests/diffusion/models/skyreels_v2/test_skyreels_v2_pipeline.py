# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.skyreels_v2 import (
    SkyReelsV2Pipeline,
    SkyReelsV2Transformer3DModel,
    get_skyreels_v2_post_process_func,
    get_skyreels_v2_pre_process_func,
)
from vllm_omni.diffusion.models.skyreels_v2.pipeline_skyreels_v2 import (
    SKYREELS_V2_DEFAULT_FLOW_SHIFT,
)
from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import WanTransformer3DModel
from vllm_omni.diffusion.registry import (
    _DIFFUSION_MODELS,
    _DIFFUSION_POST_PROCESS_FUNCS,
    _DIFFUSION_PRE_PROCESS_FUNCS,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_skyreels_v2_pipeline_import_and_registry() -> None:
    assert SkyReelsV2Pipeline is not None
    assert SkyReelsV2Transformer3DModel is WanTransformer3DModel
    assert _DIFFUSION_MODELS["SkyReelsV2Pipeline"] == (
        "skyreels_v2",
        "pipeline_skyreels_v2",
        "SkyReelsV2Pipeline",
    )
    assert _DIFFUSION_POST_PROCESS_FUNCS["SkyReelsV2Pipeline"] == "get_skyreels_v2_post_process_func"
    assert _DIFFUSION_PRE_PROCESS_FUNCS["SkyReelsV2Pipeline"] == "get_skyreels_v2_pre_process_func"


def test_skyreels_v2_component_discovery_declarations() -> None:
    assert SkyReelsV2Pipeline._dit_modules == ["transformer"]
    assert SkyReelsV2Pipeline._encoder_modules == ["text_encoder"]
    assert SkyReelsV2Pipeline._vae_modules == ["vae"]
    assert SkyReelsV2Pipeline.supports_request_batch is True


def test_skyreels_v2_process_hooks_reuse_wan22(monkeypatch) -> None:
    od_config = SimpleNamespace()
    sentinel_post = object()
    sentinel_pre = object()
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.skyreels_v2.pipeline_skyreels_v2.get_wan22_post_process_func",
        lambda config: sentinel_post,
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.skyreels_v2.pipeline_skyreels_v2.get_wan22_pre_process_func",
        lambda config: sentinel_pre,
    )
    assert get_skyreels_v2_post_process_func(od_config) is sentinel_post
    assert get_skyreels_v2_pre_process_func(od_config) is sentinel_pre


def test_skyreels_v2_metadata_declares_video_output() -> None:
    from vllm_omni.diffusion.io_support import get_diffusion_output_type
    from vllm_omni.diffusion.model_metadata import get_diffusion_model_metadata

    metadata = get_diffusion_model_metadata("SkyReelsV2Pipeline")
    assert metadata.final_output_type == "video"
    assert metadata.attention_mask_free is True
    assert get_diffusion_output_type("SkyReelsV2Pipeline") == "video"


def test_skyreels_v2_uses_wan22_cache_dit_enabler() -> None:
    from vllm_omni.diffusion.cache.cachedit import CUSTOM_DIT_ENABLERS
    from vllm_omni.diffusion.cache.cachedit import model_specific as cd_model_specific

    assert CUSTOM_DIT_ENABLERS["SkyReelsV2Pipeline"] is cd_model_specific.enable_cache_for_wan22


def test_skyreels_v2_applies_t2v_defaults_and_drops_moe(monkeypatch) -> None:
    captured: dict[str, float | None] = {}

    def fake_wan_init(self, *, od_config, prefix: str = "") -> None:
        del prefix
        captured["flow_shift"] = od_config.flow_shift
        captured["boundary_ratio"] = od_config.boundary_ratio
        self.transformer = SimpleNamespace(config={"patch_size": [1, 2, 2]})
        self.transformer_2 = object()
        self.has_transformer_2 = True
        self.weights_sources = [
            SimpleNamespace(prefix="transformer."),
            SimpleNamespace(prefix="transformer_2."),
            SimpleNamespace(prefix="vae."),
        ]

    monkeypatch.setattr(
        "vllm_omni.diffusion.models.skyreels_v2.pipeline_skyreels_v2.Wan22Pipeline.__init__",
        fake_wan_init,
    )

    od_config = SimpleNamespace(flow_shift=None, boundary_ratio=None)
    pipeline = SkyReelsV2Pipeline(od_config=od_config)

    assert captured["flow_shift"] == SKYREELS_V2_DEFAULT_FLOW_SHIFT
    assert captured["boundary_ratio"] == 0.0
    assert od_config.flow_shift == SKYREELS_V2_DEFAULT_FLOW_SHIFT
    assert od_config.boundary_ratio == 0.0
    assert pipeline.has_transformer_2 is False
    assert pipeline.transformer_2 is None
    assert [source.prefix for source in pipeline.weights_sources] == ["transformer.", "vae."]
    assert pipeline.transformer_config == {"patch_size": [1, 2, 2]}


def test_skyreels_v2_preserves_explicit_flow_shift(monkeypatch) -> None:
    def fake_wan_init(self, *, od_config, prefix: str = "") -> None:
        del prefix
        self.transformer = SimpleNamespace(config={})
        self.transformer_2 = None
        self.has_transformer_2 = False
        self.weights_sources = []

    monkeypatch.setattr(
        "vllm_omni.diffusion.models.skyreels_v2.pipeline_skyreels_v2.Wan22Pipeline.__init__",
        fake_wan_init,
    )

    od_config = SimpleNamespace(flow_shift=5.0, boundary_ratio=0.0)
    SkyReelsV2Pipeline(od_config=od_config)
    assert od_config.flow_shift == 5.0


def test_skyreels_v2_requires_transformer(monkeypatch) -> None:
    def fake_wan_init(self, *, od_config, prefix: str = "") -> None:
        del od_config, prefix
        self.transformer = None
        self.transformer_2 = None
        self.has_transformer_2 = False
        self.weights_sources = []

    monkeypatch.setattr(
        "vllm_omni.diffusion.models.skyreels_v2.pipeline_skyreels_v2.Wan22Pipeline.__init__",
        fake_wan_init,
    )

    with pytest.raises(RuntimeError, match="requires a `transformer`"):
        SkyReelsV2Pipeline(od_config=SimpleNamespace(flow_shift=8.0, boundary_ratio=0.0))


def test_skyreels_v2_load_weights_uses_optional_gate_loader(monkeypatch) -> None:
    expected = {"loaded"}
    pipeline = SkyReelsV2Pipeline.__new__(SkyReelsV2Pipeline)

    def fake_loader(model, weights):
        assert model is pipeline
        assert list(weights) == [("weight", torch.ones(1))]
        return expected

    monkeypatch.setattr(
        "vllm_omni.diffusion.models.skyreels_v2.pipeline_skyreels_v2.load_wan_weights_with_optional_gate",
        fake_loader,
    )

    assert SkyReelsV2Pipeline.load_weights(pipeline, iter((("weight", torch.ones(1)),))) is expected
