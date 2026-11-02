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


@pytest.mark.parametrize(
    ("parallel_size", "world_rank", "expected_rank", "expected_ranks"),
    [
        (2, 0, 0, [0, 1]),
        (2, 3, -1, [0, 1]),
        (8, 3, 3, None),
    ],
)
def test_light_vae_decode_uses_configured_parallel_size(
    monkeypatch, parallel_size, world_rank, expected_rank, expected_ranks
):
    from vllm_omni.diffusion.models.minimax_h3 import vae as vae_module

    world_group = object()
    created = []

    def new_group(ranks):
        group = SimpleNamespace(ranks=ranks)
        created.append(group)
        return group

    def get_world_size(group):
        return len(group.ranks) if group in created else 8

    def get_rank(group):
        return world_rank if group is world_group else (0 if world_rank < parallel_size else -1)

    monkeypatch.setattr(vae_module.dist, "get_world_size", get_world_size)
    monkeypatch.setattr(vae_module.dist, "new_group", new_group)
    monkeypatch.setattr(vae_module.dist, "get_rank", get_rank)

    vae = object.__new__(MiniMaxH3LightVideoVAE)
    vae.distributed_executor = SimpleNamespace(group=world_group, parallel_size=parallel_size)

    group, size, resolved_world, world_size, rank = vae._decode_parallel_state()

    assert size == parallel_size
    assert resolved_world is world_group
    assert world_size == 8
    assert rank == expected_rank
    assert [g.ranks for g in created] == ([expected_ranks] if expected_ranks else [])
    assert group is (created[0] if created else world_group)


def _light_vae_for_decode(use_tiling):
    calls = []

    def decode_clip(z):
        calls.append(tuple(z.shape))
        return torch.zeros_like(z)

    vae = object.__new__(MiniMaxH3LightVideoVAE)
    vae.model = SimpleNamespace(
        spatial_compression_ratio=1,
        tokens_chunk_size=8,
        token_overlap=0,
        frame_pre_padding=0,
        use_tiling=True,
        decoder=torch.nn.Linear(1, 1),
        _decode_clip=decode_clip,
        _decode=decode_clip,
    )
    vae._device_target = torch.device("cpu")
    vae.use_tiling = use_tiling
    vae._tile_off = False
    vae._tile_layout = "adaptive"
    vae._tile_explicit = (2, 2, 0, 0)
    vae._light_settings = {"image_context_tokens": 4, "image_context_output_phase": 0}
    vae.is_distributed_enabled = lambda: False
    return vae, calls


@pytest.mark.parametrize(
    ("use_tiling", "decoded_tiles", "decoded_tile_shape"),
    [
        (False, 1, (4, 6)),
        (True, 6, (2, 2)),
    ],
)
def test_light_vae_vae_use_tiling_controls_spatial_tiles(use_tiling, decoded_tiles, decoded_tile_shape):
    vae, calls = _light_vae_for_decode(use_tiling)

    output = vae._decode_video(torch.zeros(1, 2, 1, 4, 6))

    assert output.shape == (1, 2, 1, 4, 6)
    assert len(calls) == 1
    assert calls[0][0] == decoded_tiles
    assert calls[0][-2:] == decoded_tile_shape


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
