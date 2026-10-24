# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import (
    DistributedVaeMixin,
)
from vllm_omni.diffusion.models.minimax_h3.vae import (
    MiniMaxH3LightVideoVAE,
    _light_vae_opt_enabled,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, True),
        ("1", True),
        ("true", True),
        ("yes", True),
        ("0", False),
        ("off", False),
        ("false", False),
        ("no", False),
    ],
)
def test_light_vae_opt_switch(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("MINIMAX_H3_LIGHT_VAE_OPT", raising=False)
    else:
        monkeypatch.setenv("MINIMAX_H3_LIGHT_VAE_OPT", value)
    assert _light_vae_opt_enabled() is expected


def test_light_vae_keeps_decode_contract():
    assert issubclass(MiniMaxH3LightVideoVAE, DistributedVaeMixin)
    assert callable(getattr(MiniMaxH3LightVideoVAE, "decode_latent", None))
    assert callable(getattr(MiniMaxH3LightVideoVAE, "load_to_device", None))
    assert callable(getattr(MiniMaxH3LightVideoVAE, "offload_to_cpu", None))


class _FakeDecoderVAE:
    decode_dtype = torch.float16

    def __init__(self, output):
        self.output = output
        self.calls = 0
        self.use_tiling = True
        self.parallel_args = None

    def decode_latent(self, latent):
        self.calls += 1
        return self.output

    def set_parallel_size(self, *args, **kwargs):
        self.parallel_args = (args, kwargs)


@pytest.mark.parametrize("with_light_vae", [False, True])
def test_decoder_pipeline_uses_optional_light_vae(monkeypatch, tmp_path, with_light_vae):
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as main_module
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3_decoder as decoder_module

    model_dir = tmp_path / "FL2VA"
    model_dir.mkdir()
    light_dir = tmp_path / "light-vae"
    light_dir.mkdir()
    (light_dir / "decode_config.json").write_text("{}")

    video = torch.ones(1, 3, 2, 4, 4)
    audio = torch.ones(1, 2, 4)
    created = {}

    def make_vae(kind, output):
        def factory(*args, **kwargs):
            vae = _FakeDecoderVAE(output)
            created[kind] = vae
            return vae

        return factory

    monkeypatch.setattr(decoder_module, "resolve_minimax_h3_decoder_model_path", lambda *args, **kwargs: str(model_dir))
    monkeypatch.setattr(decoder_module, "get_local_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(decoder_module, "MiniMaxH3VideoVAE", make_vae("video", video))
    monkeypatch.setattr(decoder_module, "MiniMaxH3AudioVAE", make_vae("audio", audio))
    monkeypatch.setattr(decoder_module, "MiniMaxH3LightVideoVAE", make_vae("light", video))

    od_config = SimpleNamespace(
        model=str(model_dir),
        revision=None,
        task_type=None,
        trust_remote_code=False,
        parallel_config=SimpleNamespace(vae_patch_parallel_size=2, vae_parallel_mode="tile"),
        vae_use_tiling=True,
        enable_diffusion_pipeline_profiler=False,
        light_vae_path=str(light_dir) if with_light_vae else None,
    )
    pipeline = decoder_module.MiniMaxH3DecoderPipeline(od_config=od_config)
    monkeypatch.setattr(pipeline, "_release_stage_cache", lambda: None)
    monkeypatch.setattr(pipeline, "_component_on_device", lambda component: nullcontext())
    monkeypatch.setattr(
        main_module.current_omni_platform,
        "create_autocast_context",
        lambda **kwargs: nullcontext(),
    )

    video_out, audio_out = pipeline.decode(torch.zeros(1), torch.zeros(1), height=4, width=4)

    assert video_out.shape[-2:] == (4, 4)
    assert audio_out is audio
    if with_light_vae:
        assert pipeline.light_video_vae is created["light"]
        assert created["light"].calls == 1
        assert created["video"].calls == 0
        assert created["light"].use_tiling is True
        assert created["light"].parallel_args == ((2,), {"mode": "tile"})
    else:
        assert pipeline.light_video_vae is None
        assert created["video"].calls == 1
        assert "light" not in created
