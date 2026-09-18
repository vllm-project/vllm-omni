# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MammothModa2 routes stage-level VAE memory flags onto ``gen_vae``.

The DiT stage runs as an ``LLM_GENERATION`` stage, so the diffusion VAE flags
(``OmniDiffusionConfig.vae_use_slicing`` / ``vae_use_tiling``) are not part of
its engine arguments, and the generic diffusion registry configures ``model.vae``
while MammothModa2 owns ``gen_vae``.  The stage's ``additional_config`` is the
channel that reaches the model; these tests pin that a requested mode is applied
rather than silently ignored, and that an unsupported request fails loudly.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.models.mammoth_moda2 import pipeline_mammothmoda2_dit as pipeline_mod
from vllm_omni.diffusion.models.mammoth_moda2.pipeline_mammothmoda2_dit import (
    MammothModa2DiTPipeline,
    apply_vae_memory_flags,
)
from vllm_omni.transformers_utils.configs.mammoth_moda2 import Mammothmoda2Config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _stub_vae(**overrides):
    attrs = {"use_slicing": False, "use_tiling": False}
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


class _ReadOnlyTilingVae:
    """VAE whose tiling attribute cannot be set (mirrors a read-only property)."""

    use_slicing = False

    @property
    def use_tiling(self) -> bool:  # noqa: D102
        return False


class TestApplyVaeMemoryFlags:
    def test_no_flags_leaves_vae_untouched(self):
        vae = _stub_vae()
        assert apply_vae_memory_flags(vae, None) == {}
        assert apply_vae_memory_flags(vae, {}) == {}
        assert vae.use_slicing is False
        assert vae.use_tiling is False

    @pytest.mark.parametrize(
        ("config", "expected"),
        [
            ({"vae_use_slicing": True}, {"use_slicing": True}),
            ({"vae_use_tiling": True}, {"use_tiling": True}),
            (
                {"vae_use_slicing": True, "vae_use_tiling": True},
                {"use_slicing": True, "use_tiling": True},
            ),
        ],
    )
    def test_requested_modes_are_applied(self, config, expected):
        vae = _stub_vae()
        effective = apply_vae_memory_flags(vae, config)
        assert effective == expected
        for attr, value in expected.items():
            assert getattr(vae, attr) is value

    def test_explicit_false_is_applied_and_reported(self):
        vae = _stub_vae(use_slicing=True)
        assert apply_vae_memory_flags(vae, {"vae_use_slicing": False}) == {"use_slicing": False}
        assert vae.use_slicing is False

    def test_unrelated_additional_config_keys_are_ignored(self):
        vae = _stub_vae()
        assert apply_vae_memory_flags(vae, {"some_other_stage_option": True}) == {}

    def test_unknown_vae_flag_raises(self):
        with pytest.raises(ValueError, match="Unsupported MammothModa2 VAE memory flag"):
            apply_vae_memory_flags(_stub_vae(), {"vae_use_parallel": True})

    def test_requested_but_unavailable_attribute_raises(self):
        vae = SimpleNamespace()  # no use_slicing / use_tiling at all
        with pytest.raises(ValueError, match="cannot be honoured"):
            apply_vae_memory_flags(vae, {"vae_use_tiling": True})

    def test_unavailable_attribute_is_tolerated_when_not_requested(self):
        vae = SimpleNamespace()
        assert apply_vae_memory_flags(vae, {"vae_use_tiling": False}) == {}

    def test_read_only_attribute_fails_loudly(self):
        with pytest.raises(ValueError, match="rejected use_tiling"):
            apply_vae_memory_flags(_ReadOnlyTilingVae(), {"vae_use_tiling": True})


def _hf_config() -> Mammothmoda2Config:
    return Mammothmoda2Config(
        llm_config={"model_type": "mammothmoda2_qwen2_5_vl", "text_config": {"hidden_size": 8}},
        gen_vae_config={"in_channels": 4, "out_channels": 4, "latent_channels": 4},
        gen_dit_config={"hidden_size": 8},
    )


class TestPipelineWiring:
    """``__init__`` must route the stage's additional_config to gen_vae."""

    @staticmethod
    def _build(monkeypatch, additional_config):
        fake_vae = _stub_vae()
        fake_transformer = SimpleNamespace(hidden_size=8, config=SimpleNamespace(hidden_size=8))

        monkeypatch.setattr(pipeline_mod.AutoencoderKL, "from_config", staticmethod(lambda cfg: fake_vae))
        monkeypatch.setattr(pipeline_mod.Transformer2DModel, "from_config", staticmethod(lambda cfg: fake_transformer))
        monkeypatch.setattr(MammothModa2DiTPipeline, "_reinit_caption_embedder", lambda self, in_features: None)
        monkeypatch.setattr(
            pipeline_mod.RotaryPosEmbedReal, "get_freqs_real", staticmethod(lambda *args, **kwargs: None)
        )

        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=_hf_config()),
            additional_config=additional_config,
        )
        pipeline = MammothModa2DiTPipeline(vllm_config=vllm_config)
        return pipeline, fake_vae

    def test_stage_additional_config_reaches_gen_vae(self, monkeypatch):
        pipeline, fake_vae = self._build(monkeypatch, {"vae_use_slicing": True, "vae_use_tiling": True})
        assert pipeline.gen_vae is fake_vae
        assert fake_vae.use_slicing is True
        assert fake_vae.use_tiling is True
        assert pipeline._vae_memory_flags == {"use_slicing": True, "use_tiling": True}

    def test_defaults_are_left_off_without_flags(self, monkeypatch):
        _, fake_vae = self._build(monkeypatch, None)
        assert fake_vae.use_slicing is False
        assert fake_vae.use_tiling is False

    def test_unsupported_request_aborts_construction(self, monkeypatch):
        monkeypatch.setattr(pipeline_mod.AutoencoderKL, "from_config", staticmethod(lambda cfg: SimpleNamespace()))
        monkeypatch.setattr(
            pipeline_mod.Transformer2DModel,
            "from_config",
            staticmethod(lambda cfg: SimpleNamespace(hidden_size=8, config=SimpleNamespace(hidden_size=8))),
        )
        monkeypatch.setattr(MammothModa2DiTPipeline, "_reinit_caption_embedder", lambda self, in_features: None)
        monkeypatch.setattr(
            pipeline_mod.RotaryPosEmbedReal, "get_freqs_real", staticmethod(lambda *args, **kwargs: None)
        )
        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=_hf_config()),
            additional_config={"vae_use_slicing": True},
        )
        with pytest.raises(ValueError, match="cannot be honoured"):
            MammothModa2DiTPipeline(vllm_config=vllm_config)
