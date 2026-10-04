# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MammothModa2 honours the stage's VAE memory modes on ``gen_vae``.

The DiT stage's ``vae_use_slicing`` / ``vae_use_tiling`` reach the diffusion
config like any other stage field, but the registry applies them to
``model.vae`` (``vllm_omni/diffusion/registry.py``) while this pipeline owns
``gen_vae``.  These tests pin that ``__init__`` reads the standard fields, so a
requested mode reaches the VAE that actually decodes the latents.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
from vllm_omni.diffusion.models.mammoth_moda2 import pipeline_mammothmoda2_dit as pipeline_mod
from vllm_omni.diffusion.models.mammoth_moda2.pipeline_mammothmoda2_dit import MammothModa2DiTPipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _stub_vae():
    return SimpleNamespace(use_slicing=False, use_tiling=False)


def _raw_config() -> dict:
    return {
        "model_type": "mammothmoda2",
        "llm_config": {
            "model_type": "mammothmoda2_qwen2_5_vl",
            "text_config": {"model_type": "mammothmoda2_qwen2_5_vl_text", "hidden_size": 8},
        },
        "gen_vae_config": {"in_channels": 4, "out_channels": 4, "latent_channels": 4},
        "gen_dit_config": {"hidden_size": 8},
    }


def _od_config(*, vae_use_slicing: bool = False, vae_use_tiling: bool = False) -> OmniDiffusionConfig:
    return OmniDiffusionConfig(
        model="/models/MammothModa2-Preview",
        model_class_name="MammothModa2DiTPipeline",
        tf_model_config=TransformerConfig.from_dict(_raw_config()),
        vae_use_slicing=vae_use_slicing,
        vae_use_tiling=vae_use_tiling,
    )


class TestPipelineWiring:
    """``__init__`` must apply the stage's VAE memory modes to ``gen_vae``."""

    @staticmethod
    def _build(monkeypatch, *, vae_use_slicing=False, vae_use_tiling=False):
        fake_vae = _stub_vae()
        fake_transformer = SimpleNamespace(hidden_size=8, config=SimpleNamespace(hidden_size=8))

        monkeypatch.setattr(pipeline_mod.AutoencoderKL, "from_config", staticmethod(lambda cfg: fake_vae))
        monkeypatch.setattr(pipeline_mod.Transformer2DModel, "from_config", staticmethod(lambda cfg: fake_transformer))
        monkeypatch.setattr(MammothModa2DiTPipeline, "_reinit_caption_embedder", lambda self, in_features: None)
        monkeypatch.setattr(
            pipeline_mod.RotaryPosEmbedReal, "get_freqs_real", staticmethod(lambda *args, **kwargs: None)
        )

        pipeline = MammothModa2DiTPipeline(
            od_config=_od_config(vae_use_slicing=vae_use_slicing, vae_use_tiling=vae_use_tiling)
        )
        return pipeline, fake_vae

    def test_defaults_are_left_off(self, monkeypatch):
        _, fake_vae = self._build(monkeypatch)
        assert fake_vae.use_slicing is False
        assert fake_vae.use_tiling is False

    @pytest.mark.parametrize(
        ("slicing", "tiling"),
        [
            (True, False),
            (False, True),
            (True, True),
        ],
    )
    def test_stage_fields_reach_gen_vae(self, monkeypatch, slicing, tiling):
        pipeline, fake_vae = self._build(monkeypatch, vae_use_slicing=slicing, vae_use_tiling=tiling)
        assert pipeline.gen_vae is fake_vae
        assert fake_vae.use_slicing is slicing
        assert fake_vae.use_tiling is tiling
