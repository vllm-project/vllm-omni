# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pytest
import torch
from PIL import Image

from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.models.longcat_video import longcat_video_avatar_transformer as avatar_transformer
from vllm_omni.diffusion.models.longcat_video.longcat_video_avatar_transformer import (
    LongCatVideoAvatarTransformer3DModel,
    _read_config,
    replace_linear_with_quantized,
)
from vllm_omni.diffusion.models.longcat_video.pipeline_longcat_video_avatar import (
    LongCatVideoAvatarPipeline,
    _avatar_model_allow_patterns,
    _build_multi_speaker_ref_target_masks,
    _default_at2v_shape,
    _infer_asset_root_from_path,
    _prepare_multi_speaker_audio_arrays,
    _resolve_num_segments,
    prepare_longcat_video_avatar_model_for_omni,
)
from vllm_omni.diffusion.offloader.block_discovery import get_blocks_from_dit
from vllm_omni.diffusion.offloader.module_collector import ModuleDiscovery

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@dataclass
class _FeatureExtractorOutput:
    input_features: torch.Tensor


@dataclass
class _WhisperEncoderOutput:
    hidden_states: tuple[torch.Tensor, ...]


@dataclass
class _OffloadConfigStub:
    enable_cpu_offload: bool = False
    enable_layerwise_offload: bool = False
    enable_distributed_layerwise_offload: bool = False


class _DeviceRecordingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.to_devices: list[torch.device] = []

    def to(self, device: torch.device | str):
        self.to_devices.append(torch.device(device))
        return self


def _small_avatar_transformer(depth: int = 2) -> LongCatVideoAvatarTransformer3DModel:
    return LongCatVideoAvatarTransformer3DModel(
        hidden_size=8,
        depth=depth,
        num_heads=1,
        caption_channels=8,
        intermediate_dim=8,
        output_dim=8,
        audio_channel=8,
        context_tokens=1,
    )


def _single_block_lora_state(lora_dim: int = 2) -> dict[str, torch.Tensor]:
    prefix = "lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___attn___lorahyphen___qkv"
    return {
        f"{prefix}.alpha_scale": torch.tensor(1.0),
        f"{prefix}.lora_down.weight": torch.ones(3 * lora_dim, 8),
        **{f"{prefix}.lora_up.blocks.{idx}.weight": torch.ones(8, lora_dim) for idx in range(3)},
    }


def _outer_loader_with_weights(weights: list[tuple[str, torch.Tensor]]) -> DiffusersPipelineLoader:
    loader = DiffusersPipelineLoader.__new__(DiffusersPipelineLoader)
    loader.quant_config = None
    loader.counter_before_loading_weights = 0.0
    loader.counter_after_loading_weights = 0.0
    loader.get_all_weights = lambda model: iter(weights)
    return loader


def _pipeline_with_small_transformer() -> LongCatVideoAvatarPipeline:
    pipeline = LongCatVideoAvatarPipeline.__new__(LongCatVideoAvatarPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.transformer = _small_avatar_transformer(depth=1)
    pipeline.weights_sources = [
        DiffusersPipelineLoader.ComponentSource(
            model_or_path="unused",
            subfolder=None,
            revision=None,
            prefix="transformer.",
        )
    ]
    pipeline._distill_lora_path = None
    return pipeline


def test_longcat_video_avatar_declares_all_offload_components():
    pipeline = LongCatVideoAvatarPipeline.__new__(LongCatVideoAvatarPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.transformer = torch.nn.Identity()
    pipeline.text_encoder = torch.nn.Identity()
    pipeline.audio_encoder = torch.nn.Identity()
    pipeline.vae = torch.nn.Identity()

    assert isinstance(pipeline, SupportsComponentDiscovery)

    modules = ModuleDiscovery.discover(pipeline)

    assert modules.dit_names == ["transformer"]
    assert modules.dits == [pipeline.transformer]
    assert modules.encoder_names == ["text_encoder", "audio_encoder"]
    assert modules.encoders == [pipeline.text_encoder, pipeline.audio_encoder]
    assert modules.vae_names == ["vae"]
    assert modules.vaes == [pipeline.vae]


def test_longcat_video_avatar_declares_layerwise_offload_blocks():
    model = _small_avatar_transformer()

    block_attr_names, blocks = get_blocks_from_dit(model)

    assert block_attr_names == ["blocks"]
    assert blocks == list(model.blocks)


@pytest.mark.parametrize(
    ("od_config", "build_components_on_gpu", "expected_device", "build_on_accelerator"),
    [
        (_OffloadConfigStub(), False, torch.device("cuda:7"), False),
        (_OffloadConfigStub(), True, torch.device("cuda:7"), True),
        (_OffloadConfigStub(enable_cpu_offload=True), True, torch.device("cpu"), False),
        (_OffloadConfigStub(enable_layerwise_offload=True), True, torch.device("cpu"), False),
        (_OffloadConfigStub(enable_distributed_layerwise_offload=True), True, torch.device("cpu"), False),
    ],
)
def test_longcat_video_avatar_initial_component_placement_respects_offload(
    od_config: _OffloadConfigStub,
    build_components_on_gpu: bool,
    expected_device: torch.device,
    build_on_accelerator: bool,
):
    pipeline = LongCatVideoAvatarPipeline.__new__(LongCatVideoAvatarPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.od_config = od_config
    pipeline.device = torch.device("cuda:7")
    pipeline.build_components_on_gpu = build_components_on_gpu
    pipeline.text_encoder = _DeviceRecordingModule()
    pipeline.audio_encoder = _DeviceRecordingModule()
    pipeline.transformer = _DeviceRecordingModule()
    pipeline.vae = _DeviceRecordingModule()

    pipeline._place_components_after_construction()

    for component in (pipeline.text_encoder, pipeline.audio_encoder, pipeline.transformer, pipeline.vae):
        assert component.to_devices == [expected_device]
    assert pipeline._build_components_on_accelerator() is build_on_accelerator
    assert pipeline.device == torch.device("cuda:7")


def test_longcat_video_avatar_registers_lora_with_owning_block(monkeypatch, tmp_path):
    model = _small_avatar_transformer()
    lora_state = _single_block_lora_state()
    monkeypatch.setattr(avatar_transformer, "load_file", lambda path, device: lora_state)

    model.load_lora(
        tmp_path / "dmd_lora.safetensors",
        "dmd",
        lora_network_dim=2,
        lora_network_alpha=2,
    )
    qkv_input = torch.ones(1, 8)
    base_output = model.blocks[0].attn.qkv(qkv_input)
    model.enable_loras(["dmd"])

    lora = model.lora_dict["dmd"].loras[0]
    assert torch.allclose(model.blocks[0].attn.qkv(qkv_input) - base_output, torch.full((1, 24), 16.0))
    assert dict(model.lora_dict["dmd"].named_modules())[lora.lora_name] is lora
    assert model.blocks[0].attn.qkv._longcat_lora_adapters["dmd"] is lora
    adapter_parameter_names = [name for name, _ in model.named_parameters() if "_longcat_lora_adapters" in name]
    assert adapter_parameter_names == [
        "blocks.0.attn.qkv._longcat_lora_adapters.dmd.lora_down.weight",
        "blocks.0.attn.qkv._longcat_lora_adapters.dmd.lora_up.blocks.0.weight",
        "blocks.0.attn.qkv._longcat_lora_adapters.dmd.lora_up.blocks.1.weight",
        "blocks.0.attn.qkv._longcat_lora_adapters.dmd.lora_up.blocks.2.weight",
    ]
    assert not any(name.startswith("lora_dict.") for name, _ in model.named_parameters())
    root_parameter_ids = [id(parameter) for _, parameter in model.named_parameters()]
    for _, parameter in lora.named_parameters():
        assert root_parameter_ids.count(id(parameter)) == 1
    assert not any("_longcat_lora_adapters" in name for name, _ in model.blocks[1].named_parameters())
    assert {
        "blocks.0.attn.qkv._longcat_lora_adapters.dmd.alpha_scale",
        "blocks.0.attn.qkv._longcat_lora_adapters.dmd.lora_down.weight",
        "blocks.0.attn.qkv._longcat_lora_adapters.dmd.lora_up.blocks.0.weight",
        "blocks.0.attn.qkv._longcat_lora_adapters.dmd.lora_up.blocks.1.weight",
        "blocks.0.attn.qkv._longcat_lora_adapters.dmd.lora_up.blocks.2.weight",
    }.issubset(model.state_dict())

    block_attr_names, blocks = get_blocks_from_dit(model)
    assert block_attr_names == ["blocks"]
    assert blocks[0].attn.qkv._longcat_lora_adapters["dmd"] is lora

    pipeline = LongCatVideoAvatarPipeline.__new__(LongCatVideoAvatarPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.transformer = model
    pipeline.text_encoder = torch.nn.Identity()
    pipeline.audio_encoder = torch.nn.Identity()
    pipeline.vae = torch.nn.Identity()
    discovered = ModuleDiscovery.discover(pipeline)
    assert discovered.dits == [model]
    assert any(
        "_longcat_lora_adapters.dmd.lora_down.weight" in name for name, _ in discovered.dits[0].named_parameters()
    )

    model.blocks[0].to(dtype=torch.float64)

    assert lora.lora_down.weight.dtype == torch.float64
    assert all(block.weight.dtype == torch.float64 for block in lora.lora_up.blocks)
    assert next(model.blocks[0].parameters()).device == lora.lora_down.weight.device


def test_longcat_video_avatar_adds_distill_lora_after_base_weight_accounting(monkeypatch, tmp_path):
    pipeline = LongCatVideoAvatarPipeline.__new__(LongCatVideoAvatarPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.transformer = _small_avatar_transformer(depth=1)
    pipeline._distill_lora_path = tmp_path / "dmd_lora.safetensors"
    base_parameter_names = {name for name, _ in pipeline.named_parameters()}
    assert not any("_longcat_lora_adapters" in name for name in base_parameter_names)

    loaded_base_weights = {"transformer.x_embedder.proj.weight"}

    class BaseWeightsLoaderStub:
        init_calls: list[list[str]] = []

        def __init__(self, module, *, skip_substrs):
            assert {name for name, _ in module.named_parameters()} == base_parameter_names
            self.init_calls.append(skip_substrs)

        def load_weights(self, weights):
            assert list(weights) == []
            return loaded_base_weights

    monkeypatch.setattr(
        "vllm_omni.diffusion.models.longcat_video.pipeline_longcat_video_avatar.AutoWeightsLoader",
        BaseWeightsLoaderStub,
    )
    monkeypatch.setattr(avatar_transformer, "load_file", lambda path, device: _single_block_lora_state(128))

    loaded_weights = pipeline.load_weights([])

    assert loaded_weights == loaded_base_weights
    assert BaseWeightsLoaderStub.init_calls == [["._longcat_lora_adapters."]]
    assert not any("_longcat_lora_adapters" in name for name in loaded_weights)
    assert any("_longcat_lora_adapters.dmd.lora_down.weight" in name for name, _ in pipeline.named_parameters())


def test_longcat_video_avatar_base_weight_reload_skips_registered_lora(monkeypatch, tmp_path):
    pipeline = LongCatVideoAvatarPipeline.__new__(LongCatVideoAvatarPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.transformer = _small_avatar_transformer(depth=1)
    pipeline._distill_lora_path = tmp_path / "dmd_lora.safetensors"
    monkeypatch.setattr(avatar_transformer, "load_file", lambda path, device: _single_block_lora_state(128))

    loader_calls = 0

    class ReloadableWeightsLoaderStub:
        def __init__(self, module, *, skip_substrs):
            assert skip_substrs == ["._longcat_lora_adapters."]
            assert not any(
                "_longcat_lora_adapters" in name and "._longcat_lora_adapters." not in name
                for name, _ in module.named_parameters()
            )

        def load_weights(self, weights):
            nonlocal loader_calls
            loader_calls += 1
            return {"transformer.x_embedder.proj.weight"}

    monkeypatch.setattr(
        "vllm_omni.diffusion.models.longcat_video.pipeline_longcat_video_avatar.AutoWeightsLoader",
        ReloadableWeightsLoaderStub,
    )

    first_loaded = pipeline.load_weights([])
    first_lora = pipeline.transformer.lora_dict["dmd"].loras[0]
    second_loaded = pipeline.load_weights([])
    retained_adapter_names = {name for name, _ in pipeline.named_parameters() if "._longcat_lora_adapters." in name}

    assert loader_calls == 2
    assert first_loaded == {"transformer.x_embedder.proj.weight"}
    assert second_loaded == {"transformer.x_embedder.proj.weight"} | retained_adapter_names
    assert pipeline.transformer.lora_dict["dmd"].loras[0] is first_lora


def test_longcat_video_avatar_real_outer_loader_allows_base_weight_reload(monkeypatch, tmp_path):
    pipeline = _pipeline_with_small_transformer()
    pipeline._distill_lora_path = tmp_path / "dmd_lora.safetensors"
    monkeypatch.setattr(avatar_transformer, "load_file", lambda path, device: _single_block_lora_state(128))
    base_weights = [
        (f"transformer.{name}", parameter.detach().clone())
        for name, parameter in pipeline.transformer.named_parameters()
    ]
    loader = _outer_loader_with_weights(base_weights)

    loader.load_weights(pipeline)
    first_lora = pipeline.transformer.lora_dict["dmd"].loras[0]
    first_lora_values = {name: value.detach().clone() for name, value in first_lora.named_parameters()}

    loader.load_weights(pipeline)

    assert pipeline.transformer.lora_dict["dmd"].loras[0] is first_lora
    for name, value in first_lora.named_parameters():
        assert torch.equal(value, first_lora_values[name])


def test_longcat_video_avatar_real_outer_loader_keeps_base_weight_check_strict():
    pipeline = _pipeline_with_small_transformer()
    base_weights = [
        (f"transformer.{name}", parameter.detach().clone())
        for name, parameter in pipeline.transformer.named_parameters()
    ]
    missing_name, _ = base_weights.pop()
    loader = _outer_loader_with_weights(base_weights)

    with pytest.raises(ValueError, match="were not initialized from checkpoint") as exc_info:
        loader.load_weights(pipeline)

    assert missing_name in str(exc_info.value)


def test_longcat_video_avatar_invokes_managed_whisper_encoder_directly(monkeypatch):
    class FakeFeatureExtractor:
        def __call__(self, *args, **kwargs):
            return _FeatureExtractorOutput(input_features=torch.ones(1, 80, 4))

    class FakeWhisperEncoder(torch.nn.Module):
        dtype = torch.float32

        def __init__(self):
            super().__init__()
            self.call_count = 0

        def forward(self, input_features, *, output_hidden_states):
            self.call_count += 1
            assert output_hidden_states is True
            hidden_state = torch.ones(input_features.shape[0], input_features.shape[-1], 2)
            return _WhisperEncoderOutput(hidden_states=(hidden_state,) * 33)

    pipeline = LongCatVideoAvatarPipeline.__new__(LongCatVideoAvatarPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.audio_feature_extractor = FakeFeatureExtractor()
    pipeline.audio_encoder = FakeWhisperEncoder()
    monkeypatch.setattr(pipeline, "_loudness_norm", lambda audio, *args, **kwargs: audio)

    embedding = pipeline._get_audio_embedding_whisper(
        np.ones(1600, dtype=np.float32),
        fps=25,
        sample_rate=16000,
    )

    assert pipeline.audio_encoder.call_count == 1
    assert embedding.shape == (2, 5, 2)


@pytest.mark.parametrize(
    ("resolution", "expected_shape"),
    [
        ("480p", (480, 832)),
        ("720p", (736, 1248)),
    ],
)
def test_longcat_video_avatar_at2v_default_shape_uses_resolution_bucket(
    resolution: str,
    expected_shape: tuple[int, int],
):
    assert _default_at2v_shape(resolution) == expected_shape


@pytest.mark.parametrize(
    ("use_int8", "expected_weight_dir", "unexpected_weight_dir"),
    [
        (True, "base_model_int8/*", "base_model/*"),
        (False, "base_model/*", "base_model_int8/*"),
    ],
)
def test_longcat_video_avatar_allow_patterns_download_one_weight_set(
    use_int8: bool,
    expected_weight_dir: str,
    unexpected_weight_dir: str,
):
    allow_patterns = _avatar_model_allow_patterns(use_int8)

    assert expected_weight_dir in allow_patterns
    assert unexpected_weight_dir not in allow_patterns
    assert "whisper-large-v3/*" not in allow_patterns
    assert "whisper-large-v3/model.safetensors" in allow_patterns
    assert "whisper-large-v3/pytorch_model.bin" not in allow_patterns
    assert "whisper-large-v3/flax_model.msgpack" not in allow_patterns
    assert "vocal_separator/*" in allow_patterns


def test_longcat_video_avatar_prepare_model_adds_omni_metadata(tmp_path):
    model_dir = tmp_path / "LongCat-Video-Avatar-1.5"
    base_model_dir = model_dir / "base_model_int8"
    base_model_dir.mkdir(parents=True)
    (base_model_dir / "config.json").write_text('{"model_type": "longcat_avatar"}', encoding="utf-8")
    (model_dir / "model_index.json").write_text("{}", encoding="utf-8")

    prepared = prepare_longcat_video_avatar_model_for_omni(str(model_dir), use_int8=True)

    assert prepared == str(model_dir)
    model_index = json.loads((model_dir / "model_index.json").read_text(encoding="utf-8"))
    assert model_index["_class_name"] == "LongCatVideoAvatarPipeline"
    assert model_index["_diffusers_version"] == "0.38.0"
    transformer_config = json.loads((model_dir / "transformer" / "config.json").read_text(encoding="utf-8"))
    assert transformer_config == {"model_type": "longcat_avatar"}


def test_longcat_video_avatar_infers_repo_root_from_official_asset_path(tmp_path):
    input_json = tmp_path / "assets" / "avatar" / "single_example_1.json"
    input_json.parent.mkdir(parents=True)
    input_json.write_text("{}", encoding="utf-8")

    assert _infer_asset_root_from_path(input_json) == tmp_path


def test_longcat_video_avatar_falls_back_to_json_parent_without_official_layout(tmp_path):
    input_json = tmp_path / "single_example_1.json"
    input_json.write_text("{}", encoding="utf-8")

    assert _infer_asset_root_from_path(input_json) == tmp_path


@pytest.mark.parametrize(
    ("value", "audio_duration", "expected"),
    [
        (None, 26.6, 1),
        ("1", 26.6, 1),
        (3, 26.6, 3),
        ("auto", 3.0, 1),
        ("auto", 23.8, 8),
        ("auto", 26.6, 9),
    ],
)
def test_longcat_video_avatar_resolve_num_segments(
    value,
    audio_duration: float,
    expected: int,
):
    assert (
        _resolve_num_segments(
            value,
            audio_duration=audio_duration,
            num_frames=93,
            num_cond_frames=13,
            save_fps=25,
        )
        == expected
    )


def test_longcat_video_avatar_read_config_filters_non_constructor_metadata(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "_class_name": "LongCatVideoAvatarTransformer3DModel",
                "architectures": ["LongCatVideoAvatarTransformer3DModel"],
                "_diffusers_version": "0.35.1",
                "model_max_length": 512,
                "hidden_size": 4096,
                "depth": 48,
                "num_heads": 32,
            }
        ),
        encoding="utf-8",
    )

    config = _read_config(config_path)

    assert config == {
        "hidden_size": 4096,
        "depth": 48,
        "num_heads": 32,
    }


def test_longcat_video_avatar_load_weights_supports_int8_buffers():
    model = LongCatVideoAvatarTransformer3DModel(
        hidden_size=4,
        depth=0,
        num_heads=1,
        caption_channels=4,
        intermediate_dim=4,
        output_dim=4,
        audio_channel=4,
        context_tokens=1,
    )
    replace_linear_with_quantized(model)

    buffer_name = "t_embedder.mlp.0.weight_int8"
    buffers = dict(model.named_buffers())
    loaded_weight = torch.ones_like(buffers[buffer_name])

    loaded_params = model.load_weights([(buffer_name, loaded_weight)])

    assert buffer_name in loaded_params
    assert torch.equal(dict(model.named_buffers())[buffer_name], loaded_weight)


def test_longcat_video_avatar_multi_speaker_para_audio_arrays_are_aligned():
    left = np.ones(3, dtype=np.float32)
    right = np.ones(5, dtype=np.float32) * 2

    left_out, right_out = _prepare_multi_speaker_audio_arrays(
        [left, right],
        audio_type="para",
        generate_duration=0.5,
        sample_rate=10,
    )

    assert left_out.tolist() == [1, 1, 1, 0, 0]
    assert right_out.tolist() == [2, 2, 2, 2, 2]


def test_longcat_video_avatar_multi_speaker_add_audio_arrays_are_sequential():
    left = np.ones(3, dtype=np.float32)
    right = np.ones(2, dtype=np.float32) * 2

    left_out, right_out = _prepare_multi_speaker_audio_arrays(
        [left, right],
        audio_type="add",
        generate_duration=0.1,
        sample_rate=10,
    )

    assert left_out.tolist() == [1, 1, 1, 0, 0]
    assert right_out.tolist() == [0, 0, 0, 2, 2]


def test_longcat_video_avatar_multi_speaker_masks_use_explicit_bboxes():
    image = Image.new("RGB", (8, 6))
    masks, use_background = _build_multi_speaker_ref_target_masks(
        image,
        {
            "person1": [1, 1, 4, 3],
            "person2": [2, 4, 5, 7],
        },
    )

    assert not use_background
    assert masks.shape == (3, 6, 8)
    assert masks[0, 1:4, 1:3].sum() == 6
    assert masks[1, 2:5, 4:7].sum() == 9
    assert masks[2, 0, 0] == 1
    assert masks[2, 2, 2] == 0


def test_longcat_video_avatar_multi_speaker_masks_default_to_left_right_split():
    image = Image.new("RGB", (10, 8))
    masks, use_background = _build_multi_speaker_ref_target_masks(image, None)

    assert not use_background
    assert masks.shape == (3, 8, 10)
    assert masks[0, :, :5].sum() > 0
    assert masks[0, :, 5:].sum() == 0
    assert masks[1, :, :5].sum() == 0
    assert masks[1, :, 5:].sum() > 0


def test_longcat_video_avatar_transformer_accepts_multi_speaker_masks():
    model = LongCatVideoAvatarTransformer3DModel(
        hidden_size=4,
        depth=0,
        num_heads=1,
        caption_channels=4,
        intermediate_dim=4,
        output_dim=4,
        audio_channel=4,
        context_tokens=1,
    )
    hidden_states = torch.randn(1, 16, 5, 4, 4)
    timestep = torch.zeros(1, 5)
    encoder_hidden_states = torch.randn(1, 1, 4, 4)
    audio_embs = torch.randn(2, 5, 5, 12, 4)
    ref_target_masks = torch.zeros(3, 4, 4)
    ref_target_masks[0, :, :2] = 1
    ref_target_masks[1, :, 2:] = 1
    ref_target_masks[2] = torch.where(ref_target_masks[0] + ref_target_masks[1] > 0, 0, 1)

    output = model(
        hidden_states=hidden_states,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        num_cond_latents=1,
        audio_embs=audio_embs,
        ref_target_masks=ref_target_masks,
    )

    assert output.shape == hidden_states.shape
