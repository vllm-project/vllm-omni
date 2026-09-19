# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import json
import weakref
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from vllm_omni.diffusion.models.cosmos3.lidar import (
    Cosmos3LidarEncoder,
    prepare_lidar_encoder_input,
    validate_lidar_config,
)
from vllm_omni.diffusion.models.cosmos3.multiview_flex_attention import (
    MaskItem,
    MultiviewAttentionContext,
    MultiviewLayout,
    PaddedAttentionGeometry,
    build_multiview_flex_metadata,
    get_multiview_attention_plan,
    multiview_pair_predicate,
)
from vllm_omni.diffusion.models.cosmos3.multiview_packing import (
    pack_state,
    packed_position_ids,
    patchify_sensor,
    unpack_state,
    unpatchify_sensor,
)
from vllm_omni.diffusion.models.cosmos3.multiview_prompts import control_emphasis, format_camera_caption
from vllm_omni.model_extras.cosmos3_lidar import load_lidar_frames, required_lidar_sweeps, validate_lidar_header

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def lidar_config() -> dict:
    return {
        "version": "1.2",
        "dtype": "float32",
        "sample_posterior": False,
        "apply_validity_mask": True,
        "fps": 10.0,
        "latent_channels": 128,
        "spatial_compression": [16, 16],
        "temporal_compression_factor": 1,
        "streaming_chunk_frames": 2,
        "streaming_context_frames": 3,
        "range_projection": {
            "native_height": 128,
            "semantic_width": 1800,
            "model_width": 1808,
            "model_width_transform": "circular_pad",
            "intensity_encoding": "unit",
            "min_range_m": 5.0,
            "max_range_m": 100.0,
        },
        "network_config": {
            "resolution": [128, 1808],
            "patch_size": [2, 2],
            "depths": [3, 3, 3, 3],
            "temporal_downsample": [False, False, False],
            "z_dim": 128,
            "in_channels": 3,
        },
    }


def numeric_frames(sweeps=3):
    frames = torch.zeros(3, sweeps, 128, 1800)
    frames[0] = 52.5
    frames[1:] = 1
    return frames


@pytest.mark.parametrize("frames,fps,sweeps", [(301, 30, 100), (5, 30, 2), (9, 30, 3), (9, 10, 9)])
def test_sweep_count_tracks_resolved_camera_duration(frames, fps, sweeps):
    assert required_lidar_sweeps(frames, fps, 10) == sweeps


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf")])
def test_rejects_invalid_sweep_rates(fps):
    with pytest.raises(ValueError):
        required_lidar_sweeps(9, fps, 10)


def test_numeric_file_truncates_excess_and_rejects_short_input(tmp_path):
    path = tmp_path / "input.safetensors"
    frames = numeric_frames()
    frames[:, 2] = float("nan")  # Unused sweeps must not be read or value-validated at admission.
    save_file({"frames": frames}, path)
    assert validate_lidar_header(path) == (3, 3, 128, 1800)
    actual = load_lidar_frames(path, num_sweeps=2)
    assert actual.is_contiguous() and actual.dtype == torch.float32
    torch.testing.assert_close(actual, frames[:, :2])
    with pytest.raises(ValueError, match="requires 4 sweeps"):
        load_lidar_frames(path, num_sweeps=4)
    with pytest.raises(ValueError, match="finite"):
        load_lidar_frames(path, num_sweeps=3)


def test_lidar_header_admission_does_not_materialize_tensor_values(monkeypatch):
    import safetensors

    class HeaderOnlySlice:
        def get_shape(self):
            return [3, 100, 128, 1800]

        def get_dtype(self):
            return "F32"

        def __getitem__(self, key):
            pytest.fail("Admission must not read tensor values")

    class HeaderOnlyFile:
        def keys(self):
            return ["frames"]

        def get_slice(self, key):
            assert key == "frames"
            return HeaderOnlySlice()

    monkeypatch.setattr(safetensors, "safe_open", lambda *args, **kwargs: nullcontext(HeaderOnlyFile()))
    assert validate_lidar_header("large.safetensors") == (3, 100, 128, 1800)


@pytest.mark.parametrize("invalid", ["dtype", "shape", "name", "nan", "intensity", "validity", "range", "corrupt"])
def test_rejects_invalid_numeric_uploads(tmp_path, invalid):
    path = tmp_path / "input.safetensors"
    frames = numeric_frames(1)
    if invalid == "dtype":
        frames = frames.half()
    elif invalid == "shape":
        frames = frames[..., :1799].contiguous()
    elif invalid in {"nan", "intensity", "validity", "range"}:
        channel = {"nan": 0, "intensity": 1, "validity": 2, "range": 0}[invalid]
        frames[channel, 0, 0, 0] = float("nan") if invalid == "nan" else -1
    if invalid == "corrupt":
        path.write_bytes(b"not a tensor file")
    else:
        save_file({"wrong" if invalid == "name" else "frames": frames}, path)
    if invalid in {"dtype", "shape", "name", "corrupt"}:
        with pytest.raises(ValueError):
            validate_lidar_header(path)
    else:
        assert validate_lidar_header(path) == (3, 1, 128, 1800)
    with pytest.raises(ValueError):
        load_lidar_frames(path)


def test_physical_normalization_circular_padding_and_validity():
    frames = numeric_frames(1)
    frames[0, 0, 0, :4] = torch.tensor([0.0, 4.9, 5.0, 100.1])
    frames[2, 0, 1, 0] = 0
    frames[1, 0, 2, 0] = 0.25
    normalized = prepare_lidar_encoder_input(frames, lidar_config()["range_projection"])
    assert normalized.shape == (1, 3, 1, 128, 1808)
    torch.testing.assert_close(normalized[..., :4], normalized[..., 1800:1804])
    torch.testing.assert_close(normalized[..., -4:], normalized[..., 4:8])
    assert normalized[0, :, 0, 0, 4].tolist() == [-1, -1, 0]
    assert normalized[0, :, 0, 0, 6].tolist() == [-1, 1, 1]
    assert normalized[0, :, 0, 0, 7].tolist() == [-1, -1, 0]
    assert normalized[0, :, 0, 1, 4].tolist() == [-1, -1, 0]
    assert normalized[0, 1, 0, 2, 4].item() == -0.5


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", "1"),
        ("dtype", "bfloat16"),
        ("sample_posterior", True),
        ("sample_posterior", 0),
        ("apply_validity_mask", 1),
        ("fps", 0),
        ("spatial_compression", [8, 8]),
        ("temporal_compression_factor", 4),
        ("streaming_context_frames", 1),
    ],
)
def test_rejects_incompatible_encoder_metadata(field, value):
    config = lidar_config()
    validate_lidar_config(config)
    config[field] = value
    with pytest.raises(ValueError):
        validate_lidar_config(config)


@pytest.mark.parametrize("apply_validity_mask", [False, True])
def test_encoder_uses_fp32_posterior_mean_chunk_context_and_latent_affine(apply_validity_mask):
    model = object.__new__(Cosmos3LidarEncoder)
    torch.nn.Module.__init__(model)
    model.config = lidar_config()
    model.config["apply_validity_mask"] = apply_validity_mask
    model.coords = torch.zeros(1, 2, 128, 1808)
    model.latent_mean = torch.tensor(0.25)
    model.latent_std = torch.tensor(0.5)
    calls = []

    class Encoder(torch.nn.Module):
        def forward_stream(self, pixels, coords, cache):
            assert pixels.dtype == coords.dtype == torch.float32
            assert not torch.is_autocast_enabled("cpu")
            calls.append((pixels.shape[2], 0 if cache is None else cache["temporal"][0].shape[2]))
            channels = torch.cat((pixels[:, :1, :, :1, :1], torch.full_like(pixels[:, :1, :, :1, :1], 999)), dim=1)
            # Fake log variance is deliberately huge: only the posterior mean is encoded.
            return channels, {"temporal": (torch.zeros(1, 1, 9, 1), torch.zeros(1, 1, 9, 1))}

    model.encoder = Encoder()
    model.quant_conv = torch.nn.Identity()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = model(numeric_frames(5))
    assert calls == [(2, 0), (2, 1), (1, 2)]
    assert result.shape == (1, 1, 5, 1, 1)
    torch.testing.assert_close(result, torch.full_like(result, -0.5))
    assert not hasattr(model, "decode")


def mixed_items():
    return (
        MaskItem((6, 1, 2), 2, is_control=True, seconds_per_frame=4 / 30),
        MaskItem((6, 1, 2), 2, seconds_per_frame=4 / 30),
        MaskItem((7, 1, 3), 1, view_offset=2, is_control=True, is_lidar=True, seconds_per_frame=0.1),
        MaskItem((7, 1, 3), 1, view_offset=2, is_lidar=True, seconds_per_frame=0.1),
    )


@pytest.mark.parametrize("scope", ["all_views", "same_view", "decomposed"])
def test_mixed_boundaries_caption_isolation_sensor_controls_and_time_window(scope):
    items = mixed_items()
    offsets = [6]
    for item in items:
        offsets.append(offsets[-1] + item.num_tokens)
    layout = MultiviewLayout(
        items=items,
        attention_scope=scope,
        decomposed_temporal_window_seconds=0.4,
        caption_lengths=(2, 4),
        control_attends_sensor=True,
    )
    metadata = build_multiview_flex_metadata(layout, PaddedAttentionGeometry(66, 66, 6, 6), "cpu")
    allowed = multiview_pair_predicate(
        metadata, torch.arange(metadata.q_len)[:, None], torch.arange(metadata.kv_len)[None, :]
    )
    # Camera control and target read only their own caption, including all_views.
    assert allowed[0, :2].all() and not allowed[0, 2:6].any()
    assert not allowed[6, :2].any() and allowed[6, 2:6].all()
    assert allowed[24:, :6].all()  # Both LiDAR items read every camera caption.
    assert not allowed[12, offsets[2] : offsets[3]].any()  # Camera cannot read LiDAR controls.
    assert not allowed[45, offsets[0] : offsets[1]].any()  # LiDAR cannot read camera controls.
    if scope == "decomposed":
        # Last LiDAR sweep (0.6 s) sees camera capture 0.267 s but not 0 s.
        assert not allowed[63, offsets[1]]
        assert allowed[63, offsets[1] + 4]
        # A camera at t=0 cannot see future LiDAR targets.
        assert not allowed[12, offsets[3] + 3]


def test_layout_cache_keys_include_geometry_and_caption_boundaries():
    layout = MultiviewLayout(
        items=mixed_items(), caption_lengths=(2, 4), max_und_tokens=8, decomposed_temporal_window_seconds=0.4
    )
    context = MultiviewAttentionContext(layout, {})
    plan, _ = get_multiview_attention_plan(context, device=torch.device("cpu"), real_q_len=66, real_und_len=6)
    assert plan is not None
    changed = replace(layout, caption_lengths=(3, 3))
    assert changed != layout
    other, _ = get_multiview_attention_plan(
        replace(context, layout=changed), device=torch.device("cpu"), real_q_len=66, real_und_len=6
    )
    assert other is not plan


def test_lidar_rewinds_to_camera_origin_and_advances_to_furthest_endpoint():
    items = mixed_items()
    positions, endpoint = packed_position_ids(items, text_origin=100, base_fps=30, camera_compression=4)
    torch.testing.assert_close(positions[:, :12], positions[:, 12:24])
    torch.testing.assert_close(positions[:, 24:45], positions[:, 45:66])
    assert positions[0, 24::3][:7].tolist() == [100, 100.75, 101.5, 102.25, 103, 103.75, 104.5]
    assert endpoint == 105
    assert positions[0, 6].item() == 100  # Second camera shares the origin.
    wide_camera = replace(items[0], token_shape=(6, 1, 200))
    _, endpoint = packed_position_ids((wide_camera, *items[1:]), text_origin=100, base_fps=30, camera_compression=4)
    assert endpoint == 200  # The cursor accounts for spatial endpoints too.


def test_sensor_patching_and_shared_scheduler_state_roundtrip():
    tensors = [torch.randn(1, 3, 4, 3, 5), torch.randn(1, 7, 9, 2, 3)]
    for tensor in tensors:
        torch.testing.assert_close(unpatchify_sensor(patchify_sensor(tensor, 2), tuple(tensor.shape[1:]), 2), tensor)
    packed = pack_state(tensors)
    shapes = [tuple(tensor.shape[1:]) for tensor in tensors]
    updated = unpack_state(packed + 0.125, shapes)
    for result, original in zip(updated, tensors):
        torch.testing.assert_close(result, original + 0.125)


def test_exact_reference_camera_labels_and_mode_emphasis():
    caption = format_camera_caption("Driving.", "camera_front_wide_120fov")
    assert caption == "The video is captured from a camera mounted on a car. The camera is facing forward. Driving."
    assert control_emphasis("wsm", joint=True) == (
        "Follow the wsm and lidar control videos precisely: every camera view must align with its world-scenario map, "
        "and the LiDAR rangemap must align with the HD-map rangemap, at every frame."
    )
    assert "silhouette" in control_emphasis("depth", joint=False)
    assert "lidar" not in control_emphasis("wsm", joint=False)


@pytest.mark.parametrize("backend", ["triton", "maskless"])
def test_packed_transformer_isolates_causal_captions_and_only_times_noisy_targets(monkeypatch, backend):
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3_multiview import Cosmos3MultiviewVFMTransformer

    model = object.__new__(Cosmos3MultiviewVFMTransformer)
    torch.nn.Module.__init__(model)
    model.lidar_config = lidar_config()
    model.latent_patch_size = 1
    model.temporal_compression_factor = 4
    model.base_fps = 30
    model.enable_fps_modulation = True
    model.temporal_modality_margin = 10
    model.timestep_scale = 1
    model.cached_kv = model.cached_freqs_gen = None
    model._multiview_mask_cache, model._multiview_buffer_cache = {}, {}
    model._maskless_gqa_ratio, model._maskless_fa_version = 1, 2
    model._offload_context = lambda _: nullcontext()
    model.gen_sp_prepare = lambda hidden, cos, sin: (hidden, cos, sin)
    model.gen_sp_gather = torch.nn.Identity()
    model.proj_in = torch.nn.Linear(1, 2, bias=False)
    model.proj_in.weight.data.fill_(1)
    model.lidar_proj_in = model.lidar_proj_out = torch.nn.Identity()
    model._project_video_tokens = lambda hidden: hidden[:, :, :1]
    model.norm_moe_gen = torch.nn.Identity()
    model.time_embedder = lambda time: time[:, None].expand(-1, 2)
    origins, texts, layers = [], [], []

    class Language(torch.nn.Module):
        def rotary_emb(self, dummy, position_ids):
            origins.append(position_ids.clone())
            value = position_ids[0].unsqueeze(-1).expand(-1, -1, 2).float()
            return value, value

        def forward(self, ids, freqs):
            texts.append(ids.clone())
            key = ids.cumsum(1).float().unsqueeze(-1).unsqueeze(-1)
            return [(key, key)]

    class Layer(torch.nn.Module):
        def forward(self, hidden, **kwargs):
            layers.append((hidden.clone(), kwargs))
            return hidden

    model.language_model = Language()
    model.gen_layers = torch.nn.ModuleList([Layer()])
    camera = torch.zeros(1, 1, 4, 1, 1)
    lidar = torch.zeros(1, 2, 3, 1, 1)
    controls = [torch.full_like(camera, 3)]
    items = (
        MaskItem((4, 1, 1), 2, is_control=True, seconds_per_frame=4 / 30),
        MaskItem((4, 1, 1), 2, seconds_per_frame=4 / 30),
        MaskItem((3, 1, 1), 1, view_offset=2, is_lidar=True, is_control=True, seconds_per_frame=0.1),
        MaskItem((3, 1, 1), 1, view_offset=2, is_lidar=True, seconds_per_frame=0.1),
    )
    shapes = (tuple(camera.shape[1:]), tuple(lidar.shape[1:]))
    kwargs = dict(
        hidden_states=pack_state([camera, lidar]),
        timestep=torch.tensor([10.0]),
        text_ids=torch.tensor([[2, 3, 7, 11, 13]]),
        text_mask=torch.tensor([[1, 1, 2, 2, 2]]),
        caption_lengths=(2, 3),
        packed_shapes=shapes,
        control_latents=controls,
        lidar_control_latents=torch.full_like(lidar, 2),
        noisy_frame_mask=torch.tensor([0, 1, 0, 1]).reshape(1, 1, 4, 1, 1),
        temporal_position_period=2,
        multiview_layout=MultiviewLayout(
            items=items,
            max_und_tokens=8,
            backend=backend,
            control_attends_sensor=True,
            decomposed_temporal_window_seconds=None if backend == "maskless" else 0.4,
        ),
    )
    if backend == "maskless":
        with pytest.raises(ValueError, match="B == 1"):
            model(**{**kwargs, "hidden_states": kwargs["hidden_states"].expand(2, -1)})
    prediction = model(**kwargs)
    if backend == "maskless":
        assert len(model._multiview_mask_cache) == 1
        assert len(model._multiview_buffer_cache) == 6
    with monkeypatch.context() as patch:
        patch.setattr(
            torch.Tensor, "item", lambda *args: pytest.fail("Denoising steps must not synchronize text lengths")
        )
        model(**kwargs)  # CPU position metadata was computed during cache initialization.
    assert [ids.tolist() for ids in texts] == [[[2, 3]], [[7, 11, 13]]]
    assert layers[0][1]["k_und"].flatten().tolist() == [2, 5, 7, 18, 31]
    assert origins[0][0, 0].tolist() == [0, 1]
    assert origins[1][0, 0].tolist() == [0, 1, 2]
    assert origins[2][0, 0, 0].item() == 13  # Longest caption + margin.
    assert layers[0][0][0, :, 0].tolist() == [3, 3, 3, 3, 0, 10, 0, 10, 2, 2, 2, 10, 10, 10]
    video_pred, lidar_pred = unpack_state(prediction, shapes)
    assert video_pred.flatten().tolist() == [0, 10, 0, 10]
    assert (lidar_pred == 10).all()
    model(**kwargs)
    assert len(texts) == 2  # Text cached once per branch.
    model.reset_cache()
    model(**{**kwargs, "control_latents": None})
    assert layers[-1][0].shape[1] == 7  # Both control streams are removed together for control CFG.
    assert all(not item.is_control for item in layers[-1][1]["multiview_layout"].layout.items)


@pytest.mark.parametrize("model_width", [6, 10])
def test_encoder_padding_uses_projection_widths(model_width):
    frames = torch.ones(3, 1, 1, 6)
    frames[0] = 52.5
    frames[1] = torch.linspace(0, 1, 6)
    projection = {"model_width": model_width, "semantic_width": 6, "min_range_m": 5, "max_range_m": 100}
    actual = prepare_lidar_encoder_input(frames, projection)
    half_padding = (model_width - 6) // 2
    columns = [(index - half_padding) % 6 for index in range(model_width)]
    assert actual.shape == (1, 3, 1, 1, model_width)
    torch.testing.assert_close(actual[0, 1, 0, 0], (frames[1, 0, 0] * 2 - 1)[columns])
    assert actual[0, 0].eq(0).all() and actual[0, 2].eq(1).all()


def test_missing_projection_weights_fail_even_when_some_lidar_weights_are_present():
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3_multiview import Cosmos3MultiviewVFMTransformer

    model = object.__new__(Cosmos3MultiviewVFMTransformer)
    torch.nn.Module.__init__(model)
    model.lidar_config = lidar_config()
    loaded = {
        f"transformer.lidar_proj_{direction}.{parameter}"
        for direction in ("in", "out")
        for parameter in ("weight", "bias")
    }
    model.validate_loaded_weights(loaded)
    with pytest.raises(ValueError, match="lidar_proj_out.bias"):
        model.validate_loaded_weights(loaded - {"transformer.lidar_proj_out.bias"})
    model.lidar_config = None
    model.validate_loaded_weights(set())  # Legacy WSM has no LiDAR modules.


@pytest.mark.parametrize(
    "mode,noise_source",
    [
        ("joint", "seed"),
        ("joint", "generator"),
        ("joint", "advanced_generator"),
        ("joint", "injected"),
        ("transfer", "seed"),
        ("ordinary", "seed"),
        ("completion", "seed"),
    ],
)
@pytest.mark.parametrize("emphasis", [False, True])
@pytest.mark.parametrize("return_lidar", [False, True])
def test_pipeline_shares_schedule_preserves_conditions_and_optional_lidar(
    tmp_path, monkeypatch, mode, noise_source, emphasis, return_lidar
):
    from types import SimpleNamespace

    import vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview as module
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    pipeline = object.__new__(module.Cosmos3MultiviewPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device, pipeline.dtype = torch.device("cpu"), torch.float32
    pipeline.vae_scale_factor_temporal, pipeline.vae_scale_factor_spatial = 4, 16
    pipeline.is_distilled_model = False
    cameras = tuple(module.COSMOS3_MADS_CAMERAS[1::-1])
    pipeline.multiview_cameras = cameras
    pipeline.multiview_config = {
        "schema_version": 2,
        "variable_view_count": True,
        "separate_view_text_tokenization": True,
        "inference_defaults": {"num_steps": 2, "guidance": 1},
    }
    pipeline.multiview_align_temporal_positions_across_views = True
    pipeline.multiview_attention_scope = "decomposed"
    pipeline.multiview_decomposed_temporal_window_seconds = 0.4
    pipeline.multiview_control_attends_sensor = True
    pipeline.multiview_backend = "triton"
    pipeline.transformer = SimpleNamespace(
        latent_channel_size=1,
        latent_patch_size=2,
        _pad_to_patch_size=lambda h, w: ((h + 1) // 2, (w + 1) // 2, 0, 0),
        reset_cache=lambda: None,
    )
    monkeypatch.setattr(module, "_resolve_multiview_geometry", lambda *args, **kwargs: ("480", "1,1", 32, 32))
    pipeline._set_timesteps = lambda *args, **kwargs: None
    pipeline._set_mixed_precision_step = lambda *args: None
    pipeline._reset_mixed_precision = lambda: None
    pipeline.progress_bar = lambda steps: steps
    pipeline._encode_video_tensor = lambda pixels: pixels[:, :1, ::4, ::16, ::16].clone()
    pixel_refs, latent_refs = [], []

    def prepare_pixels(views, **kwargs):
        pixels = torch.full((1, 3, 10, 32, 32), 0.5)
        pixel_refs.append(weakref.ref(pixels))
        return pixels

    pipeline._prepare_camera_major_pixels = prepare_pixels
    prepare_latents = pipeline._prepare_multiview_latents
    encode_video = pipeline._encode_multiview_video

    def record_latents(**kwargs):
        result = prepare_latents(**kwargs)
        latent_refs.extend(weakref.ref(tensor) for tensor in (result[0], result[2]))
        return result

    def record_encoded_video(*args, **kwargs):
        result = encode_video(*args, **kwargs)
        latent_refs.append(weakref.ref(result))
        return result

    pipeline._prepare_multiview_latents = record_latents
    pipeline._encode_multiview_video = record_encoded_video
    decoded = []

    def decode(latents):
        assert all(ref() is None for ref in latent_refs)
        decoded.append(latents.clone())
        assert latents.shape == (1, 1, 2, 2, 2)
        return torch.zeros(1, 3, 5, 32, 32)

    pipeline._decode_latents = decode
    texts = []

    def tokenize(prompt, *args, **kwargs):
        texts.append(prompt)
        length = 2 + len(prompt) % 3
        return torch.ones(1, length, dtype=torch.long), torch.ones(1, length, dtype=torch.long)

    pipeline._tokenize_prompt = tokenize
    extra = {
        "emphasize_control_in_prompt": emphasis,
        "multiview": {"condition_video_as_image": mode != "completion", "views": []},
    }
    for index, camera in enumerate(cameras):
        view = {"camera_key": camera, "prompt": f"Scene {index}."}
        if mode != "ordinary":
            view["control_path"] = "control.mp4"
        if mode != "completion" or index == 0:
            view["vision_path"] = "condition.mp4"
        extra["multiview"]["views"].append(view)
    if mode != "ordinary":
        extra["wsm"] = {}
    if mode == "joint":
        pipeline.multiview_config["lidar"] = lidar_config()
        path = tmp_path / "map.safetensors"
        save_file({"frames": numeric_frames(3)}, path)
        extra["lidar"] = {"control_path": str(path), "return_output": return_lidar}

        def encode(frames):
            assert frames.shape[1] == 2  # round(5 * 10 / 30); extra sweep discarded.
            return torch.full((1, 2, 2, 1, 3), 7.0)

        pipeline.lidar_encoder = encode
    lidar_decoded = []

    def decode_lidar(latents):
        from vllm_omni.diffusion.models.cosmos3.lidar_encoder.neighborhood_attention import neighborhood_attention_2d

        lidar_decoded.append(latents.clone())
        # Execute small real FlexAttention when output is requested. It must
        # leave the camera tensors and both RNG streams unchanged.
        q = latents.permute(0, 2, 3, 4, 1).reshape(2, 1, 3, 1, 2).contiguous()
        with torch.inference_mode():
            spatial = neighborhood_attention_2d(q, q, q, kernel_size=(1, 3), dilation=1, scale=1.0)
        return spatial.mean((1, 2, 3, 4)).reshape(1, 1, 2, 1, 1).expand(1, 3, 2, 128, 1800).contiguous()

    decode_lidar.config = lidar_config()
    pipeline.lidar_decoder = decode_lidar
    calls, scheduler_calls, sample_refs = [], [], []

    def predict(**kwargs):
        assert all(ref() is None for ref in pixel_refs)
        assert all(ref() is None for ref in sample_refs)
        sample_refs.append(weakref.ref(kwargs["hidden_states"]))
        # Keep snapshots for parity assertions without extending lifetimes.
        calls.append({"hidden_states": kwargs["hidden_states"].clone(), "packed_shapes": kwargs["packed_shapes"]})
        targets = unpack_state(kwargs["hidden_states"], kwargs["packed_shapes"])
        if mode == "joint":
            assert len(targets) == 2 and kwargs["lidar_control_latents"].eq(7).all()
            assert kwargs["multiview_layout"].items[-1].is_lidar
        if mode != "ordinary":
            assert kwargs["control_latents"][0].eq(0.5).all()
        return torch.ones_like(kwargs["hidden_states"])

    pipeline.predict_noise = predict

    def step(noise, timestep, latents, **kwargs):
        scheduler_calls.append(timestep.item())
        return (latents - noise * 0.125,)

    pipeline.scheduler = SimpleNamespace(timesteps=torch.tensor([1000.0, 500.0]), step=step)
    sp = OmniDiffusionSamplingParams(num_frames=5, num_inference_steps=2, seed=42, extra_args=extra)
    expected_generator = torch.Generator().manual_seed(sp.seed)
    if noise_source != "seed":
        # An explicit generator, including its current position, takes
        # precedence over the request seed for every generated sensor.
        sp.generator = torch.Generator().manual_seed(123)
        if noise_source in {"advanced_generator", "injected"}:
            torch.randn(7, generator=sp.generator)
        expected_generator.set_state(sp.generator.get_state())
    camera_shape = (1, 1, 4, 2, 2)
    if noise_source == "injected":
        sp.latents = torch.full(camera_shape, 0.25)
        expected_camera = sp.latents.clone()
    else:
        expected_camera = torch.randn(camera_shape, generator=expected_generator)
    if mode == "joint":
        # LiDAR must use the next draw, without restarting the camera stream.
        expected_lidar = torch.randn((1, 2, 2, 1, 3), generator=expected_generator)
    if mode == "completion":
        expected_camera[:, :, :2] = 0.5
    else:
        expected_camera[:, :, ::2] = 0.5
    rng_before = torch.random.get_rng_state()
    result = pipeline.forward(
        SimpleNamespace(prompts=[{"prompt": "ignored", "negative_prompt": "ignored too"}], sampling_params=sp)
    )
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert scheduler_calls == [1000, 500]
    assert len(calls) == 2 and len(decoded) == 2
    initial_targets = unpack_state(calls[0]["hidden_states"], calls[0]["packed_shapes"])
    torch.testing.assert_close(initial_targets[0], expected_camera, rtol=0, atol=0)
    if mode == "joint":
        torch.testing.assert_close(initial_targets[1], expected_lidar, rtol=0, atol=0)
    if sp.generator is not None:
        assert torch.equal(sp.generator.get_state(), expected_generator.get_state())
    assert texts[1::2] == ["", ""]
    suffix = control_emphasis("wsm", joint=mode == "joint")
    for index, caption in enumerate(texts[::2]):
        assert f"Scene {index}." in caption
        assert caption.count(suffix) == int(emphasis and mode != "ordinary")
    if mode == "completion":
        assert decoded[0].eq(0.5).all()  # Entire known view stays fixed.
    else:
        assert all(view[:, :, 0].eq(0.5).all() for view in decoded)
    assert set(result.output) == {"payload", "metadata"}
    if mode == "joint" and return_lidar:
        assert set(result.output["payload"]) == {"video", "lidar"}
        torch.testing.assert_close(lidar_decoded[0], expected_lidar - 0.25)
        assert result.output["payload"]["lidar"].shape == (1, 3, 2, 128, 1800)
        assert result.output["metadata"]["lidar"]["fps"] == 10
        assert result.output["metadata"]["lidar"]["num_frames"] == 2
    else:
        assert not lidar_decoded
        assert set(result.output["payload"]) == {"video"}
    assert result.output["metadata"]["multiview"]["cameras"] == list(cameras)


def test_view_completion_rejects_short_known_video(monkeypatch):
    import vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview as module

    pipeline = object.__new__(module.Cosmos3MultiviewPipeline)
    torch.nn.Module.__init__(pipeline)
    monkeypatch.setattr(
        module, "media_to_uint8_cthw", lambda *args, **kwargs: torch.zeros(3, 4, 32, 32, dtype=torch.uint8)
    )
    with pytest.raises(ValueError, match="complete RGB video of 5 frames"):
        pipeline._prepare_camera_major_pixels(
            [{"camera_key": "front", "vision_path": "short.mp4"}],
            field="vision",
            height=32,
            width=32,
            num_frames=5,
            keep_first=False,
            require_complete=True,
        )


class TinyLidarEncoder(Cosmos3LidarEncoder):
    def __init__(self, config):
        torch.nn.Module.__init__(self)
        self.config = config
        self.encoder = torch.nn.Linear(1, config["network_config"].get("base_channels", 1))
        self.quant_conv = torch.nn.Linear(1, 1)
        self.register_buffer("coords", torch.zeros(1, 2, 1, 1))
        self.register_buffer("latent_mean", torch.zeros(1))
        self.register_buffer("latent_std", torch.ones(1))


@pytest.fixture
def lidar_vae_artifact(tmp_path):
    config = lidar_config()
    component = {**config, "network_config": {**config["network_config"], "base_channels": 4, "decoder_depths": None}}
    folder = tmp_path / "lidar_vae"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps(component))
    model = TinyLidarEncoder(component).float()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(1.0001)  # Detect rounding through the pipeline's BF16 default.
    state = {
        **model.state_dict(),
        "decoder.weight": torch.ones(1),
        "post_quant_conv.weight": torch.ones(1),
    }
    save_file(state, folder / "diffusion_pytorch_model.safetensors")
    return tmp_path, config, component, state


def test_encoder_artifact_inventory_and_fp32_loading(lidar_vae_artifact, monkeypatch):
    import safetensors

    path, config, component, state = lidar_vae_artifact
    safe_open = safetensors.safe_open
    reads = []

    @contextmanager
    def encoder_only_open(filename, **kwargs):
        assert kwargs == {"framework": "pt", "device": "cpu"}
        with safe_open(filename, **kwargs) as handle:

            def get_slice(name):
                assert not name.startswith(("decoder.", "post_quant_conv."))
                return handle.get_slice(name)

            def get_tensor(name):
                assert not name.startswith(("decoder.", "post_quant_conv."))
                reads.append(name)
                return handle.get_tensor(name)

            yield SimpleNamespace(keys=handle.keys, get_slice=get_slice, get_tensor=get_tensor)

    monkeypatch.setattr(safetensors, "safe_open", encoder_only_open)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        loaded = TinyLidarEncoder.from_pretrained(str(path), config, torch.device("cpu"))
    finally:
        torch.set_default_dtype(previous)
    assert loaded.config == component
    assert loaded.encoder.out_features == 4
    assert set(reads) == loaded.state_dict().keys()
    for name, tensor in loaded.state_dict().items():
        assert tensor.dtype == torch.float32
        torch.testing.assert_close(tensor, state[name], rtol=0, atol=0)
    assert not loaded.training and all(not p.requires_grad for p in loaded.parameters())
    assert not hasattr(loaded, "decoder") and not hasattr(loaded, "decode")


def test_encoder_resolves_vae_from_hub(lidar_vae_artifact, monkeypatch):
    from vllm_omni.transformers_utils import repo_utils

    path, config, _, _ = lidar_vae_artifact

    def download(repo_id, *, allow_patterns):
        assert repo_id == "test-org/joint-lidar-model"
        assert allow_patterns == ["lidar_vae/config.json", "lidar_vae/diffusion_pytorch_model.safetensors"]
        return str(path)

    monkeypatch.setattr(repo_utils.hf_api(), "snapshot_download", download)
    loaded = TinyLidarEncoder.from_pretrained("test-org/joint-lidar-model", config, torch.device("cpu"))
    assert loaded.encoder.out_features == 4


@pytest.mark.parametrize("missing", ["config.json", "diffusion_pytorch_model.safetensors"])
def test_encoder_requires_vae_files(lidar_vae_artifact, missing):
    path, config, _, _ = lidar_vae_artifact
    (path / "lidar_vae" / missing).unlink()
    with pytest.raises(ValueError, match="Incomplete joint artifact: lidar_vae/config.json"):
        TinyLidarEncoder.from_pretrained(str(path), config, torch.device("cpu"))


@pytest.mark.parametrize("missing", ["encoder.weight", "quant_conv.weight", "coords", "latent_mean", "latent_std"])
def test_encoder_requires_complete_encoder_state(lidar_vae_artifact, missing):
    path, config, _, state = lidar_vae_artifact
    del state[missing]
    save_file(state, path / "lidar_vae/diffusion_pytorch_model.safetensors")
    with pytest.raises(RuntimeError, match="Missing key"):
        TinyLidarEncoder.from_pretrained(str(path), config, torch.device("cpu"))


@pytest.mark.parametrize(
    "name,value,error,message",
    [
        ("encoder.weight", torch.ones(2, 1), RuntimeError, "size mismatch"),
        ("latent_mean", torch.zeros(2), RuntimeError, "size mismatch"),
        ("encoder.weight", torch.ones(4, 1, dtype=torch.bfloat16), ValueError, "must be FP32"),
        ("quant_conv.weight", torch.ones(1, 1, dtype=torch.float16), ValueError, "must be FP32"),
        ("coords", torch.zeros(1, 2, 1, 1, dtype=torch.bfloat16), ValueError, "must be FP32"),
        ("latent_std", torch.ones(1, dtype=torch.int64), ValueError, "must be FP32"),
        ("latent_mean", torch.tensor([float("nan")]), ValueError, "positive standard deviations"),
        ("latent_mean", torch.tensor([float("inf")]), ValueError, "positive standard deviations"),
        ("latent_std", torch.tensor([float("nan")]), ValueError, "positive standard deviations"),
        ("latent_std", torch.tensor([float("inf")]), ValueError, "positive standard deviations"),
        ("latent_std", torch.tensor([0.0]), ValueError, "positive standard deviations"),
        ("latent_std", torch.tensor([-1.0]), ValueError, "positive standard deviations"),
        ("encoder.unexpected", torch.ones(1), ValueError, "Unexpected LiDAR encoder tensors"),
        ("optimizer.step", torch.ones(1), ValueError, "Unexpected LiDAR encoder tensors"),
    ],
)
def test_encoder_rejects_invalid_vae_encoder_state(lidar_vae_artifact, name, value, error, message):
    path, config, _, state = lidar_vae_artifact
    state[name] = value
    save_file(state, path / "lidar_vae/diffusion_pytorch_model.safetensors")
    with pytest.raises(error, match=message):
        TinyLidarEncoder.from_pretrained(str(path), config, torch.device("cpu"))


@pytest.mark.parametrize(
    "field", ["fps", "network_config", "range_projection", "apply_validity_mask", "decoder_depths", "missing_default"]
)
def test_encoder_rejects_conflicting_vae_metadata(lidar_vae_artifact, field):
    path, config, component, _ = lidar_vae_artifact
    if field == "network_config":
        component[field]["depths"] = [1, 1, 1, 1]
    elif field == "range_projection":
        component[field] = {**component[field], "max_range_m": 105.0}
    elif field == "decoder_depths":
        config["network_config"][field] = [1, 1, 1, 1]
    elif field == "missing_default":
        config["network_config"]["decoder_depths"] = None
        del component["network_config"]["decoder_depths"]
    else:
        component[field] = False if field == "apply_validity_mask" else 11.0
    (path / "lidar_vae/config.json").write_text(json.dumps(component))
    with pytest.raises(ValueError, match="metadata disagrees"):
        TinyLidarEncoder.from_pretrained(str(path), config, torch.device("cpu"))


@pytest.mark.parametrize("field", ["dtype", "sample_posterior", "apply_validity_mask"])
def test_encoder_requires_vae_policy_metadata(lidar_vae_artifact, field):
    path, config, component, _ = lidar_vae_artifact
    del component[field]
    (path / "lidar_vae/config.json").write_text(json.dumps(component))
    with pytest.raises(ValueError, match="missing LiDAR metadata"):
        TinyLidarEncoder.from_pretrained(str(path), config, torch.device("cpu"))


def test_encoder_loads_real_architecture_with_saved_constructor_defaults(tmp_path):
    import inspect

    from vllm_omni.diffusion.models.cosmos3.lidar_encoder.transformer_vae import Encoder

    config = lidar_config()
    config["latent_channels"] = 4
    config["network_config"].update(z_dim=4, base_channels=4, depths=[1] * 4, num_heads=[1] * 4)
    arguments = inspect.signature(Encoder).bind(**config["network_config"])
    arguments.apply_defaults()
    network = json.loads(json.dumps(arguments.arguments))
    # Constructor defaults and decoder-only overrides are absent from deployment metadata.
    component = {**config, "network_config": {**network, "decoder_depths": None, "out_channels": 3}}
    model = Cosmos3LidarEncoder(component).float()
    state = model.state_dict()
    state["latent_mean"].fill_(0.1234567)
    state["latent_std"].fill_(0.9876543)
    state["encoder.tokenizer.0.weight"].fill_(0.1234567)
    folder = tmp_path / "lidar_vae"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps(component))
    save_file({**state, "decoder.weight": torch.ones(1)}, folder / "diffusion_pytorch_model.safetensors")
    loaded = Cosmos3LidarEncoder.from_pretrained(str(tmp_path), config, torch.device("cpu"))
    assert loaded.config == component
    assert loaded.state_dict().keys() == state.keys()
    for name, tensor in loaded.state_dict().items():
        torch.testing.assert_close(tensor, state[name], rtol=0, atol=0)


def test_unipc_updates_mixed_sensor_state_with_one_sigma_schedule():
    from vllm_omni.diffusion.models.schedulers.scheduling_flow_unipc_multistep import FlowUniPCMultistepScheduler

    camera, lidar = torch.randn(1, 1, 3, 2, 2), torch.randn(1, 2, 5, 1, 3)
    shapes = (tuple(camera.shape[1:]), tuple(lidar.shape[1:]))
    schedulers = [FlowUniPCMultistepScheduler() for _ in range(3)]
    for scheduler in schedulers:
        scheduler.set_timesteps(3, device="cpu", shift=10)
    state = pack_state([camera, lidar])
    for timestep in schedulers[0].timesteps:
        camera_v, lidar_v = camera * 0.1, lidar * 0.2
        state = schedulers[0].step(pack_state([camera_v, lidar_v]), timestep, state, return_dict=False)[0]
        camera = schedulers[1].step(camera_v, timestep, camera, return_dict=False)[0]
        lidar = schedulers[2].step(lidar_v, timestep, lidar, return_dict=False)[0]
        packed_camera, packed_lidar = unpack_state(state, shapes)
        torch.testing.assert_close(packed_camera, camera)
        torch.testing.assert_close(packed_lidar, lidar)
