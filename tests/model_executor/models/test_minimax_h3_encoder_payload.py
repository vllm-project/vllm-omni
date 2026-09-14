# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.data_entry_keys import flatten_payload
from vllm_omni.model_executor.models.minimax_h3.conditioning import (
    MiniMaxH3EncoderConditioning,
    MiniMaxH3EncoderMediaConditioning,
    MiniMaxH3EncoderMediaInput,
)
from vllm_omni.model_executor.stage_input_processors.minimax_h3 import encoder2diffusion

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_pipeline_owns_three_encoders_then_dit_and_decoders() -> None:
    from vllm_omni.model_executor.models.minimax_h3.pipeline import MINIMAX_H3_PIPELINE

    encoder, diffusion = MINIMAX_H3_PIPELINE.stages
    assert encoder.model_arch == "MiniMaxH3Encoder"
    assert encoder.custom_process_next_stage_input_func == (
        "vllm_omni.model_executor.stage_input_processors.minimax_h3.encoder2diffusion_full_payload"
    )
    assert diffusion.model_arch == "MiniMaxH3Pipeline"
    assert diffusion.stage_input_payload_keys == ("encoder_output",)
    assert diffusion.requires_multimodal_data is False
    assert diffusion.requires_full_payload_input is False


def _patch_encoder_constructors(monkeypatch, *, rank: int):
    from vllm_omni.model_executor.models.minimax_h3 import encoder as encoder_module

    group = SimpleNamespace(rank_in_group=rank, world_size=2, device_group=object())

    class FakeVideoVAE(torch.nn.Module):
        def __init__(self, path, *, device, encode_only):
            super().__init__()
            self.path = path
            self.parallel_args = None

        def set_parallel_size(self, size, *, process_group):
            self.parallel_args = (size, process_group)

    class FakeAudioVAE(torch.nn.Module):
        def __init__(self, path, *, device, encode_only):
            super().__init__()
            self.path = path

    def init_backbone(self, *, vllm_config, prefix):
        torch.nn.Module.__init__(self)

    monkeypatch.setattr(encoder_module.MiniMaxH3TextEncoderBackbone, "__init__", init_backbone)
    monkeypatch.setattr(encoder_module, "get_tp_group", lambda: group)
    monkeypatch.setattr(encoder_module, "get_local_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(encoder_module, "MiniMaxH3VideoVAE", FakeVideoVAE)
    monkeypatch.setattr(encoder_module, "MiniMaxH3AudioVAE", FakeAudioVAE)
    return encoder_module, group


def _component_root(tmp_path):
    root = tmp_path / "FL2VA"
    for name in ("text_encoder", "video_vae", "audio_vae"):
        (root / name).mkdir(parents=True)
    return root


def _encoder_config(root, *, video_mode="patch", roles=None):
    components = (
        {
            "text_encoder": {"parallel_mode": "tp"},
            "video_vae": {"parallel_mode": video_mode},
            "audio_vae": {"parallel_mode": "leader"},
        }
        if roles is None
        else roles
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model=str(root / "text_encoder"),
            hf_config=SimpleNamespace(minimax_h3_encoder_components=components),
        )
    )


def test_encoder_requires_exactly_three_components(monkeypatch, tmp_path) -> None:
    encoder_module, _ = _patch_encoder_constructors(monkeypatch, rank=0)
    root = tmp_path / "FL2VA"
    (root / "text_encoder").mkdir(parents=True)
    (root / "video_vae").mkdir()
    config = _encoder_config(root)

    with pytest.raises(RuntimeError, match=r"requires all three encoders.*audio_vae"):
        encoder_module.MiniMaxH3Encoder(vllm_config=config)


@pytest.mark.parametrize(("rank", "has_audio_vae"), [(0, True), (1, False)])
def test_encoder_uses_tp_patch_and_leader_on_the_same_rank_set(
    monkeypatch,
    tmp_path,
    rank,
    has_audio_vae,
) -> None:
    encoder_module, group = _patch_encoder_constructors(monkeypatch, rank=rank)
    root = _component_root(tmp_path)
    config = _encoder_config(root)

    model = encoder_module.MiniMaxH3Encoder(vllm_config=config)

    assert model.component_config.text_parallel_mode == "tp"
    assert model.component_config.video_parallel_mode == "patch"
    assert model.component_config.audio_parallel_mode == "leader"
    assert model.video_vae.parallel_args == (2, group.device_group)
    assert (model.audio_vae is not None) is has_audio_vae


def test_encoder_requires_all_role_policies(monkeypatch, tmp_path) -> None:
    encoder_module, _ = _patch_encoder_constructors(monkeypatch, rank=0)
    root = _component_root(tmp_path)
    config = _encoder_config(
        root,
        roles={
            "text_encoder": {"parallel_mode": "tp"},
            "video_vae": {"parallel_mode": "patch"},
        },
    )

    with pytest.raises(ValueError, match=r"requires exactly.*audio_vae"):
        encoder_module.MiniMaxH3Encoder(vllm_config=config)


def test_encoder_does_not_infer_role_policies(monkeypatch, tmp_path) -> None:
    encoder_module, _ = _patch_encoder_constructors(monkeypatch, rank=0)
    root = _component_root(tmp_path)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            model=str(root / "text_encoder"),
            hf_config=SimpleNamespace(),
        )
    )

    with pytest.raises(ValueError, match="requires explicit minimax_h3_encoder_components"):
        encoder_module.MiniMaxH3Encoder(vllm_config=config)


def test_video_leader_policy_skips_nonleader(monkeypatch, tmp_path) -> None:
    encoder_module, _ = _patch_encoder_constructors(monkeypatch, rank=1)
    root = _component_root(tmp_path)
    model = encoder_module.MiniMaxH3Encoder(vllm_config=_encoder_config(root, video_mode="leader"))

    assert model.video_vae is None
    assert model.audio_vae is None
    assert (
        model._encode_media(
            MiniMaxH3EncoderMediaInput(
                task="ref2va",
                height=32,
                width=32,
                num_frames=48,
                latent_t=1,
                audio_t=80,
                images=(torch.zeros(32, 32, 3, dtype=torch.uint8),),
            )
        )
        is None
    )


def test_encoder_runs_video_and_audio_components_on_ar_model() -> None:
    from vllm_omni.model_executor.models.minimax_h3.encoder import MiniMaxH3Encoder

    class VideoVAE:
        def encode_image(self, _image):
            return torch.ones(1, 96)

        def encode_video(self, _frames):
            return torch.full((1, 96), 2.0), (1, 2, 2)

    class AudioVAE:
        def encode_waveform(self, waveform, _sample_rate):
            return torch.ones(160, 32), waveform.shape[-1] // 800

    model = MiniMaxH3Encoder.__new__(MiniMaxH3Encoder)
    torch.nn.Module.__init__(model)
    model._component_leader = True
    model.video_vae = VideoVAE()
    model.audio_vae = AudioVAE()
    media = MiniMaxH3EncoderMediaInput(
        task="ref2va",
        height=32,
        width=32,
        num_frames=48,
        latent_t=1,
        audio_t=80,
        images=(torch.zeros(32, 32, 3, dtype=torch.uint8),),
        videos=(torch.zeros(1, 32, 32, 3, dtype=torch.uint8),),
        video_audios=((torch.zeros(64_000), 32_000),),
    )

    conditioning = model._encode_media(media)

    assert conditioning is not None
    assert conditioning.visual_condition_shapes == ((1, 2, 2), (1, 2, 2))
    assert conditioning.audio_condition_lengths == (80,)
    assert conditioning.ref_blocks[1]["kind"] == "video_audio"


def test_encoder_output_reuses_encoder_handoff_and_round_trips() -> None:
    expected = MiniMaxH3EncoderConditioning(
        hidden_states=torch.randn(3, 5120, dtype=torch.bfloat16),
        token_tags=torch.tensor([1, 0, 1], dtype=torch.int64),
        task="t2va",
        height=256,
        width=448,
        num_frames=17,
        latent_t=5,
        audio_t=10,
    )

    source = SimpleNamespace(
        finished=True,
        request_id="request-1",
        outputs=[SimpleNamespace(multimodal_output=flatten_payload(expected.to_omni_payload()))],
    )
    prompt = {
        "prompt": "hello",
        "additional_information": {"global_request_id": ["request-1"]},
    }

    result = encoder2diffusion([source], prompt)

    assert result is not None
    payload = result["additional_information"]["encoder_output"]
    actual = MiniMaxH3EncoderConditioning.from_omni_payload(payload)
    torch.testing.assert_close(actual.hidden_states, expected.hidden_states)
    torch.testing.assert_close(actual.token_tags, expected.token_tags)


def test_encoder_releases_workspace_after_cpu_payload(monkeypatch) -> None:
    from vllm_omni.model_executor.models.minimax_h3 import encoder as encoder_module

    model = encoder_module.MiniMaxH3Encoder.__new__(encoder_module.MiniMaxH3Encoder)
    torch.nn.Module.__init__(model)
    model._component_leader = True
    model._token_tags = torch.tensor([1])
    info = {
        "meta": {"minimax_h3_encoder_request": {}},
        "_minimax_h3_encoder_media_conditioning": MiniMaxH3EncoderMediaConditioning(
            task="t2va",
            height=256,
            width=448,
            num_frames=17,
            latent_t=5,
            audio_t=10,
        ),
    }
    empty_cache_calls = 0

    def empty_cache() -> None:
        nonlocal empty_cache_calls
        empty_cache_calls += 1

    monkeypatch.setattr(torch.accelerator, "empty_cache", empty_cache)
    output = model.make_omni_output(
        torch.zeros(1, 5120),
        model_intermediate_buffer=[info],
        request_sample_eligible=[True],
    )

    assert empty_cache_calls == 1
    assert output.multimodal_outputs is not None
    assert output.multimodal_outputs["embed"]["embedding"][0].device.type == "cpu"
