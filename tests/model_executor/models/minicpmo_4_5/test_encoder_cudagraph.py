# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass, field

import pytest
import torch
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.models.whisper.modeling_whisper import WhisperConfig
from vllm.config import CompilationConfig, MultiModalConfig, ParallelConfig
from vllm.v1.worker.encoder_cudagraph_defs import EncoderCudaGraphConfig

from vllm_omni.model_executor.models.minicpmo_4_5.encoder_cudagraph import (
    _AXIS_KEY,
    _ceiling,
    _MiniCPMO45EncoderCudaGraphMixin,
    _NoEncoderCudaGraph,
    _select_encoder_cudagraph_mixin,
)
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
    MiniCPMO45OmniLLMForConditionalGeneration,
    MiniCPMWhisperEncoder,
    Resampler,
    SiglipVisionConfig,
    SiglipVisionTransformer,
)
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.skipif(
    "capture_axes" not in EncoderCudaGraphConfig.__dataclass_fields__,
    reason="Requires vLLM containing capture_axes",
)
def test_serving_wrapper_exposes_concrete_thinker_protocol():
    from vllm.model_executor.models.interfaces import supports_encoder_cudagraph

    from vllm_omni.model_executor.models.minicpmo_4_5.encoder_cudagraph import bind_minicpmo_encoder_cudagraph
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    torch.nn.Module.__init__(model)
    thinker = _EncoderModel().eval()
    thinker.multimodal_config = MultiModalConfig(media_io_kwargs={"video": {"num_frames": 2}})
    assert not supports_encoder_cudagraph(model)
    bind_minicpmo_encoder_cudagraph(model, thinker)
    assert supports_encoder_cudagraph(model)
    assert model.get_encoder_cudagraph_config() == thinker.get_encoder_cudagraph_config()


def test_unsupported_stage_does_not_advertise_encoder_protocol():
    from vllm.model_executor.models.interfaces import supports_encoder_cudagraph

    from vllm_omni.model_executor.models.minicpmo_4_5.encoder_cudagraph import bind_minicpmo_encoder_cudagraph

    model = torch.nn.Module()
    bind_minicpmo_encoder_cudagraph(model, torch.nn.Module())
    assert not supports_encoder_cudagraph(model)


@dataclass
class _EncoderConfig:
    query_num: int = 4
    hidden_size: int = 16
    vision_batch_size: int = 16
    audio_chunk_length: int = 0
    audio_pool_step: int = 2


@dataclass
class _EncoderTestModelConfig:
    multimodal_config: MultiModalConfig | None = None
    max_model_len: int = 32


@dataclass
class _EncoderTestSchedulerConfig:
    max_num_batched_tokens: int = 32


@dataclass
class _EncoderTestVllmConfig:
    """Only manager-facing state; not a substitute for serving initialization."""

    compilation_config: CompilationConfig = field(default_factory=CompilationConfig)
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    model_config: _EncoderTestModelConfig = field(default_factory=_EncoderTestModelConfig)
    scheduler_config: _EncoderTestSchedulerConfig = field(default_factory=_EncoderTestSchedulerConfig)


@pytest.mark.parametrize("value,expected", [(1, 1), (3, 4), (8, 8), (9, 9)])
def test_tier_selection_preserves_out_of_range_inputs(value: int, expected: int) -> None:
    assert _ceiling(value, (1, 2, 4, 8)) == expected


def test_non_cuda_platform_does_not_create_encoder_manager(monkeypatch) -> None:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: False)
    mixin = _select_encoder_cudagraph_mixin()
    assert mixin is _NoEncoderCudaGraph

    class NativeEncoderModel(torch.nn.Module, _NoEncoderCudaGraph):
        pass

    model = NativeEncoderModel()
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.model = model
    runner.compilation_config = CompilationConfig(cudagraph_mm_encoder=True)
    runner.supports_mm_inputs = True
    assert GPUModelRunner._create_encoder_cudagraph_manager(runner) is None


@pytest.mark.parametrize("limit,expected", [(128, (4, 16)), (8, (4, 8))])
def test_default_budget_range_is_bounded(limit: int, expected: tuple[int, int]) -> None:
    model = _EncoderModel()
    config = _EncoderTestVllmConfig(
        scheduler_config=_EncoderTestSchedulerConfig(max_num_batched_tokens=limit),
        model_config=_EncoderTestModelConfig(max_model_len=limit),
    )
    assert model.get_encoder_cudagraph_budget_range(config) == expected


def test_default_budget_range_batches_more_than_one_item() -> None:
    # query_num is 4 here, so the per-item floor is 4 and the 4x ceiling is 16.
    # Equal ends would leave the manager's max_budget // min_budget at one item.
    model = _EncoderModel()
    min_budget, max_budget = model.get_encoder_cudagraph_budget_range(_EncoderTestVllmConfig())
    assert 0 < min_budget <= max_budget
    assert max_budget // min_budget > 1


def test_duplex_audio_bypasses_offline_graph_protocol_when_enabled(monkeypatch) -> None:
    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.stage0 import MiniCPMO45Stage0DuplexRuntime

    torch.manual_seed(42)
    model = _EncoderModel().eval()
    model.vllm_config = _EncoderTestVllmConfig(compilation_config=CompilationConfig(cudagraph_mm_encoder=True))
    model.audio_past_key_values = None
    data = {"audio_features": torch.randn(1, 16, 16), "audio_feature_lens": [torch.tensor([16])]}
    with torch.no_grad():
        expected = model.get_audio_embedding_streaming(
            data, use_extra_context=True, prefix_extra_frames=0, suffix_extra_frames=2
        )
    model.audio_past_key_values = None

    def unexpected_protocol(*args, **kwargs):
        pytest.fail("duplex must use the stateful streaming encoder, not the offline graph protocol")

    monkeypatch.setattr(model, "encoder_cudagraph_forward", unexpected_protocol)
    monkeypatch.setattr(model, "encoder_eager_forward", unexpected_protocol)
    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.stage_model = model
    runtime.thinker = model
    runtime.device = "cpu"
    with torch.no_grad():
        actual = runtime._stage_audio_embeddings(data)
    torch.testing.assert_close(actual, torch.cat(expected[0]))
    assert model.audio_past_key_values is not None


@pytest.mark.parametrize("modality", ["image", "video"])
def test_empty_encoder_shard_has_no_items(modality: str) -> None:
    model = _EncoderModel().eval()
    prefix = "video_" if modality == "video" else ""
    kwargs = {prefix + "pixel_values": [[torch.randn(3, 2, 12)]], prefix + "tgt_sizes": [torch.tensor([[2, 3]])]}
    selected = model.select_encoder_cudagraph_items(kwargs, [])
    assert model.get_encoder_cudagraph_item_specs(selected) == []


@pytest.mark.parametrize("grids", [((2, 3), (1, 4)), ((3, 2), (2, 3))])
def test_vision_capture_metadata_matches_eager(grids, monkeypatch) -> None:
    torch.manual_seed(42)
    config = SiglipVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=8,
        patch_size=2,
    )
    config._attn_implementation = "eager"
    vision = SiglipVisionTransformer(config).eval()
    resampler = Resampler(num_queries=4, embed_dim=16, num_heads=2, kv_dim=16).eval()
    sizes = torch.tensor(grids, dtype=torch.int32)
    patches = int(sizes.prod(-1).max())
    pixels = torch.randn(len(grids), 3, 2, patches * 2)
    mask = (torch.arange(patches)[None, :] < sizes.prod(-1)[:, None]).unsqueeze(1)

    with torch.no_grad():
        expected = resampler(vision(pixels, patch_attention_mask=mask, tgt_sizes=sizes).last_hidden_state, sizes)
        position_ids = vision.embeddings._create_position_ids(mask, sizes, device=torch.device("cpu"))
        # Always 4-D, the way prepare_encoder_cudagraph_replay_buffers builds it:
        # the captured forward requires the pair, and pinning the all-valid grid
        # to a None mask here would keep the test off the path production takes.
        attention_mask = _prepare_4d_attention_mask(mask.flatten(1), pixels.dtype)
        positions, padding_mask = resampler.prepare_metadata(sizes, device=pixels.device, dtype=pixels.dtype)

        def unexpected_metadata(*args, **kwargs):
            pytest.fail("layout metadata must not be recomputed in the captured forward")

        monkeypatch.setattr(vision.embeddings, "_create_position_ids", unexpected_metadata)
        monkeypatch.setattr(resampler, "prepare_metadata", unexpected_metadata)
        hidden = vision(
            pixels,
            patch_attention_mask=mask,
            tgt_sizes=sizes,
            position_ids=position_ids,
            encoder_attention_mask=attention_mask,
        ).last_hidden_state
        actual = resampler(hidden, pos_embed=positions, key_padding_mask=padding_mask)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class _EncoderModel(MiniCPMO45OmniLLMForConditionalGeneration, _MiniCPMO45EncoderCudaGraphMixin):
    """Use production encoder entry points without constructing the language model."""

    def __init__(self):
        torch.nn.Module.__init__(self)
        self.config = _EncoderConfig()
        self.vllm_config = _EncoderTestVllmConfig()
        vision_config = SiglipVisionConfig(
            hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2, image_size=8, patch_size=2
        )
        vision_config._attn_implementation = "eager"
        self.vpm = SiglipVisionTransformer(vision_config).eval()
        self.resampler = Resampler(num_queries=4, embed_dim=16, num_heads=2, kv_dim=16).eval()
        audio_config = WhisperConfig(
            num_mel_bins=16,
            d_model=16,
            encoder_layers=2,
            encoder_attention_heads=2,
            encoder_ffn_dim=64,
            max_source_positions=100,
            dropout=0,
            attention_dropout=0,
        )
        audio_config._attn_implementation = "eager"
        self.apm = MiniCPMWhisperEncoder(audio_config).eval()
        self.audio_projection_layer = torch.nn.Identity()
        self.audio_avg_pooler = torch.nn.AvgPool1d(2, stride=2)
        self.audio_encoder_layer = -1


def _encoder_model_without_vision() -> _EncoderModel:
    """Build the mixin as an audio-only deploy would: no vision tower, graphs on."""
    model = _EncoderModel().eval()
    model.vpm = None
    model.multimodal_config = MultiModalConfig(media_io_kwargs={"video": {"num_frames": 2}})
    model.vllm_config = _EncoderTestVllmConfig(compilation_config=CompilationConfig(cudagraph_mm_encoder=True))
    return model


@pytest.mark.parametrize("modality", ["image", "video"])
def test_protocol_buffers_match_encoder_entry_point(modality: str) -> None:
    torch.manual_seed(42)
    model = _EncoderModel().eval()
    prefix = "video_" if modality == "video" else ""
    kwargs = {
        prefix + "pixel_values": [[torch.randn(3, 2, 12), torch.randn(3, 2, 8)], [torch.randn(3, 2, 12)]],
        prefix + "tgt_sizes": [torch.tensor([[2, 3], [1, 4]]), torch.tensor([[3, 2]])],
    }
    selected = model.select_encoder_cudagraph_items(kwargs, [0, 1])
    axes = selected.pop(_AXIS_KEY)
    assert axes[0] == ("vision", 1024)
    capture = model.prepare_encoder_cudagraph_capture_inputs(
        16, 2, 2, torch.device("cpu"), torch.float32, "default", axes
    )
    replay = model.prepare_encoder_cudagraph_replay_buffers(selected, 2, 2)
    for key, buffer in capture.values.items():
        buffer.zero_()
        source = replay.values[key]
        buffer[: source.shape[0]].copy_(source)
    with torch.no_grad():
        expected = model.get_multimodal_embeddings(**kwargs)
        output = model.encoder_cudagraph_forward(capture.values)
    specs = model.get_encoder_cudagraph_item_specs(kwargs)
    dest: dict[int, torch.Tensor] = {}
    model.postprocess_encoder_output(
        {"default": output}, [0, 1], [spec.output_tokens for spec in specs], dest, batch_mm_kwargs=selected
    )
    assert len(dest) == 2
    for index, reference in enumerate(expected):
        torch.testing.assert_close(dest[index], reference, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(
    "capture_axes" not in EncoderCudaGraphConfig.__dataclass_fields__,
    reason="Requires vLLM containing capture_axes",
)
def test_audio_is_not_advertised_for_padded_graph_capture() -> None:
    model = _EncoderModel().eval()
    model.multimodal_config = MultiModalConfig()
    assert "audio" not in model.get_encoder_cudagraph_config().modalities


@pytest.mark.skipif(
    "capture_axes" not in EncoderCudaGraphConfig.__dataclass_fields__,
    reason="Requires vLLM containing capture_axes",
)
def test_deploy_without_vision_tower_does_not_advertise_the_protocol() -> None:
    # An audio-only deploy builds the Thinker without a vision tower, and the
    # manager cannot be constructed from an empty axis list (it rejects a
    # non-empty tuple wrapping an empty axis, and capture would index axis_keys).
    # The binder must leave the names off so the runner factory never builds a
    # manager, instead of failing the engine during init.
    from vllm.model_executor.models.interfaces import supports_encoder_cudagraph

    from vllm_omni.model_executor.models.minicpmo_4_5.encoder_cudagraph import bind_minicpmo_encoder_cudagraph

    model = torch.nn.Module()
    thinker = _encoder_model_without_vision()
    assert thinker.vpm is None
    assert supports_encoder_cudagraph(thinker)

    bind_minicpmo_encoder_cudagraph(model, thinker)
    assert not supports_encoder_cudagraph(model)


@pytest.mark.skipif(
    "capture_axes" not in EncoderCudaGraphConfig.__dataclass_fields__,
    reason="Requires vLLM containing capture_axes",
)
def test_serving_init_binds_the_protocol_on_a_vision_deploy(monkeypatch) -> None:
    """The serving wrapper must end up advertising the protocol.

    Every other test here names the mixin or calls the binder itself, so
    deleting the `bind_minicpmo_encoder_cudagraph` call in
    `MiniCPMO45OmniForConditionalGeneration.__init__` would leave the suite
    green with the feature off. This goes through that constructor with the
    registered-model factory stubbed out.
    """
    from types import SimpleNamespace

    from vllm.model_executor.models.interfaces import supports_encoder_cudagraph

    from vllm_omni.model_executor.models.minicpmo_4_5 import minicpmo_4_5_omni
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    thinker = _EncoderModel().eval()
    thinker.make_empty_intermediate_tensors = lambda: None
    thinker.multimodal_config = MultiModalConfig(media_io_kwargs={"video": {"num_frames": 2}})
    monkeypatch.setattr(minicpmo_4_5_omni, "init_vllm_registered_model", lambda **kwargs: thinker)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(base_model_tp_plan=None),
            multimodal_config=thinker.multimodal_config,
            model_stage="llm",
        )
    )

    wrapper = MiniCPMO45OmniForConditionalGeneration(vllm_config=vllm_config)

    assert wrapper.thinker is thinker
    assert supports_encoder_cudagraph(wrapper)
    assert wrapper.get_encoder_cudagraph_config().capture_axes


@pytest.mark.skipif(
    "capture_axes" not in EncoderCudaGraphConfig.__dataclass_fields__,
    reason="Requires vLLM containing capture_axes",
)
def test_uncaptured_patch_layout_raises_with_the_layout(monkeypatch) -> None:
    """An in-budget slice whose patch extent was never captured must not reach the manager.

    The processor's no-slice upscale path keeps extreme aspect ratios inside the
    token budget while their patch extent grows past the captured ladder; the
    manager's graph lookup then misses and its `assert graph_output is not None`
    fires. The model has to name the layout instead. 1x5000 is the shape from the
    review: one slice of 2143 patches at the defaults of the in-tree processor.
    """
    from PIL import Image

    from vllm_omni.model_executor.models.minicpmo_4_5.encoder_cudagraph import _PATCH_CAPS
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import MiniCPMVImageProcessor

    model = _EncoderModel().eval()
    slices = MiniCPMVImageProcessor().get_sliced_images(Image.new("RGB", (5000, 1)))
    grid = (slices[0].size[1] // 14, slices[0].size[0] // 14)
    patches = int(torch.tensor(grid).prod())
    assert len(slices) == 1 and patches > _PATCH_CAPS[-1]

    kwargs = {"pixel_values": [[torch.randn(3, 14, grid[1] * 14)]], "tgt_sizes": [torch.tensor([grid])]}
    with pytest.raises(ValueError, match="1 slices of 2143 patches"):
        model.select_encoder_cudagraph_items(kwargs, [0])


def test_over_budget_selection_keeps_the_manager_eager_path() -> None:
    # Seven slices of four output tokens each exceed the 16-token slice ladder
    # ((1, 2, 4) at query_num=4), so the manager picks its eager path and never
    # looks the layout key up: selecting must not raise here.
    model = _EncoderModel().eval()
    sizes = [torch.tensor([[2, 2]] * 7)]
    kwargs = {"pixel_values": [[torch.randn(3, 2, 4)] * 7], "tgt_sizes": sizes}
    selected = model.select_encoder_cudagraph_items(kwargs, [0])
    assert model.get_encoder_cudagraph_item_specs(selected)[0].output_tokens == 28


def test_unpadded_bf16_replay_matches_eager(monkeypatch) -> None:
    """The configuration the serving E2E fails on: bf16 with no patch padding.

    The tiny encoder's real patch counts are far below the production ladder, so
    shrink the ladder to make the layout exact: extent == patches, as in the
    224x224 / 1024-patch fixture that the full checkpoint replays.
    """
    import vllm_omni.model_executor.models.minicpmo_4_5.encoder_cudagraph as encoder_cudagraph

    monkeypatch.setattr(encoder_cudagraph, "_PATCH_CAPS", (4, 8))
    torch.manual_seed(42)
    dtype = torch.bfloat16
    model = _EncoderModel().eval().to(dtype)
    kwargs = {"pixel_values": [[torch.randn(3, 2, 8, dtype=dtype)]], "tgt_sizes": [torch.tensor([[1, 4]])]}
    selected = model.select_encoder_cudagraph_items(kwargs, [0])
    axes = selected.pop(_AXIS_KEY)
    assert axes[0] == ("vision", 4), "fixture must be unpadded: extent == real patch count"

    capture = model.prepare_encoder_cudagraph_capture_inputs(4, 1, 1, torch.device("cpu"), dtype, "default", axes)
    replay = model.prepare_encoder_cudagraph_replay_buffers(selected, 1, 1)
    for key, buffer in capture.values.items():
        buffer.zero_()
        source = replay.values[key]
        buffer[: source.shape[0]].copy_(source)

    with torch.no_grad():
        expected = model.get_multimodal_embeddings(**kwargs)
        output = model.encoder_cudagraph_forward(capture.values)
    torch.testing.assert_close(output, expected[0], rtol=1e-2, atol=1e-2)


def test_resampler_metadata_extends_cache_before_forward(monkeypatch) -> None:
    torch.manual_seed(42)
    resampler = Resampler(num_queries=4, embed_dim=16, num_heads=2, max_size=(2, 2)).eval()
    sizes = torch.tensor([[3, 2], [1, 4]])
    hidden = torch.randn(2, 6, 16)
    with torch.no_grad():
        expected = resampler(hidden, sizes)
        positions, mask = resampler.prepare_metadata(sizes, device=hidden.device, dtype=hidden.dtype)
        monkeypatch.setattr(resampler, "_adjust_pos_cache", lambda *args, **kwargs: pytest.fail("graph grew cache"))
        actual = resampler(hidden, pos_embed=positions, key_padding_mask=mask)
    assert tuple(resampler.max_size) == (3, 4)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif("capture_axes" not in EncoderCudaGraphConfig.__dataclass_fields__, reason="Requires capture_axes")
def test_default_capture_keys_have_one_slice_capacity_per_budget(monkeypatch):
    from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

    model = _EncoderModel().eval()
    model.config.query_num = 64
    mm_config = MultiModalConfig()
    model.multimodal_config = mm_config
    config = _EncoderTestVllmConfig(
        model_config=_EncoderTestModelConfig(multimodal_config=mm_config, max_model_len=512),
        scheduler_config=_EncoderTestSchedulerConfig(max_num_batched_tokens=512),
    )
    model.vllm_config = config
    manager = EncoderCudaGraphManager(config, torch.device("cpu"), torch.float32, model)
    captured = {}

    def record_capture(token_budget, path, axis_keys):
        inputs = model.prepare_encoder_cudagraph_capture_inputs(
            token_budget,
            manager.max_batch_size,
            manager.max_frames_per_batch,
            torch.device("cpu"),
            torch.float32,
            path,
            axis_keys,
        )
        captured[token_budget, axis_keys] = inputs.values["pixels"].shape[0]

    monkeypatch.setattr(manager, "_capture_budget_graph", record_capture)
    manager.capture(None)
    assert captured == {
        (budget, (("vision", patches),)): budget // model.config.query_num
        for budget in (64, 128, 256)
        for patches in (1024, 1152, 2048)
    }
    assert manager.get_num_graphs_to_capture() == 9


@pytest.mark.skipif(
    "capture_axes" not in EncoderCudaGraphConfig.__dataclass_fields__, reason="Requires encoder graph manager"
)
def test_flag_on_image_embeds_bypasses_manager_in_runner(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from vllm.v1.worker import gpu_model_runner

    from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

    embeddings = torch.randn(1, 4, 16)
    model = _EncoderModel().eval()
    runner = OmniGPUModelRunner.__new__(OmniGPUModelRunner)
    runner.model = model
    runner.compilation_config = CompilationConfig(cudagraph_mm_encoder=True)
    runner.encoder_cudagraph_manager = SimpleNamespace(
        supports_modality=lambda modality: modality in ("image", "video"),
        execute=lambda kwargs: pytest.fail("precomputed embeddings entered graph manager"),
    )
    data = {"image_embeds": embeddings}
    runner.requests = {
        "req": SimpleNamespace(
            mm_features=[SimpleNamespace(data=data, modality="image", identifier="image-0", mm_position=None)]
        )
    }
    runner.observability_config = None
    runner.lora_config = None
    runner.is_multimodal_pruning_enabled = False
    runner.requires_sequential_video_encoding = False
    runner.device = torch.device("cpu")
    runner.encoder_cache = {}
    runner.timed_encoder_operation = lambda *args: nullcontext()
    runner.maybe_save_ec_to_connector = lambda *args: None
    monkeypatch.setattr(
        gpu_model_runner,
        "group_and_batch_mm_kwargs",
        lambda inputs, **kwargs: iter((modality, 1, item) for modality, item in inputs),
    )
    scheduled = SimpleNamespace(
        scheduled_encoder_inputs={"req": [0]}, ec_manager_metadata=None, free_encoder_mm_hashes=[]
    )
    with torch.no_grad():
        runner._execute_mm_encoder(scheduled)
    torch.testing.assert_close(runner.encoder_cache["image-0"], embeddings[0], rtol=0, atol=0)
