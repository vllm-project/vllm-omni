# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from contextlib import contextmanager
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.hunyuan_image3.encoder_cudagraph import _pad_cu_seqlens
from vllm_omni.model_executor.models.hunyuan_image3.siglip2 import Config, Siglip2VisionEmbeddings
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

pytestmark = pytest.mark.core_model


@pytest.mark.cpu
def test_position_interpolation_preserves_eager_embeddings():
    embeddings = Siglip2VisionEmbeddings(Config(dict(hidden_size=8, patch_size=2, num_channels=3, num_patches=4)))
    pixels = torch.randn(10, 12)
    shapes = torch.tensor([[2, 3], [1, 4]])
    expected = []
    table = embeddings.position_embedding.weight.reshape(2, 2, 8).permute(2, 0, 1).unsqueeze(0)
    for height, width in shapes.tolist():
        pos = torch.nn.functional.interpolate(
            table, (height, width), mode="bilinear", align_corners=False, antialias=True
        )
        expected.append(pos.reshape(8, height * width).T)
    torch.testing.assert_close(embeddings(pixels, shapes), embeddings.patch_embedding(pixels) + torch.cat(expected))


@pytest.mark.cpu
def test_replay_padding_keeps_tail_sequence_capacity():
    buffer = torch.tensor([0, 0, 0, 12], dtype=torch.int32)
    _pad_cu_seqlens(buffer, torch.tensor([0, 3, 7], dtype=torch.int32))
    assert buffer.tolist() == [0, 3, 7, 12]
    _pad_cu_seqlens(buffer, torch.tensor([0, 2], dtype=torch.int32))
    assert buffer.tolist() == [0, 2, 2, 12]


@pytest.mark.cpu
@pytest.mark.parametrize("enabled,cuda", [(False, True), (True, False)])
def test_stage_does_not_create_manager_when_disabled_or_non_cuda(monkeypatch, enabled, cuda):
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    from vllm_omni.worker import gpu_model_runner

    factory = Mock()
    model = SimpleNamespace(create_encoder_cudagraph_manager=factory)
    runner = object.__new__(OmniGPUModelRunner)
    runner.compilation_config = SimpleNamespace(cudagraph_mm_encoder=enabled)
    runner.supports_mm_inputs = True
    monkeypatch.setattr(runner, "get_model", lambda: model)
    monkeypatch.setattr(gpu_model_runner.current_omni_platform, "is_cuda", lambda: cuda)
    monkeypatch.setattr(GPUModelRunner, "_create_encoder_cudagraph_manager", lambda self: None)
    assert runner._create_encoder_cudagraph_manager() is None
    factory.assert_not_called()


@pytest.mark.cpu
def test_stage_preserves_upstream_manager_for_other_models(monkeypatch):
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    runner = object.__new__(OmniGPUModelRunner)
    runner.compilation_config = SimpleNamespace(cudagraph_mm_encoder=True)
    runner.supports_mm_inputs = True
    monkeypatch.setattr(runner, "get_model", lambda: nn.Linear(2, 2))
    upstream = object()
    monkeypatch.setattr(GPUModelRunner, "_create_encoder_cudagraph_manager", lambda self: upstream)
    assert runner._create_encoder_cudagraph_manager() is upstream


@pytest.mark.cpu
def test_sdpa_encoder_stays_eager():
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    from vllm_omni.model_executor.models.hunyuan_image3.hunyuan_image3 import HunyuanImage3ForConditionalGeneration

    attention = SimpleNamespace(attn_backend=AttentionBackendEnum.TORCH_SDPA)
    model = SimpleNamespace(
        vision_model=SimpleNamespace(
            encoder=SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace(attn=attention))])
        )
    )
    assert HunyuanImage3ForConditionalGeneration.create_encoder_cudagraph_manager(model, None, None, None) is None


def _cpu_manager(budgets, *, max_items=2, image_base_size=1024, factor=16, tp_size=1):
    from vllm.config.multimodal import MultiModalConfig

    from vllm_omni.model_executor.models.hunyuan_image3.encoder_cudagraph import HunyuanImage3EncoderCudaGraphManager

    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            encoder_cudagraph_token_budgets=budgets,
            encoder_cudagraph_max_vision_items_per_batch=max_items,
            encoder_cudagraph_max_frames_per_batch=None,
        ),
        model_config=SimpleNamespace(
            multimodal_config=MultiModalConfig(
                limit_per_prompt={"image": 4, "video": 0}, mm_encoder_tp_mode="data" if tp_size > 1 else "weights"
            )
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp_size),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=32768),
    )
    model = SimpleNamespace(
        config=SimpleNamespace(image_base_size=image_base_size, hidden_size=2, vit_processor={"max_num_patches": 1024}),
        vae=SimpleNamespace(ffactor_spatial=factor, dtype=torch.float32, config=SimpleNamespace(latent_channels=1)),
    )
    return HunyuanImage3EncoderCudaGraphManager(config, torch.device("cpu"), torch.float32, model)


@pytest.mark.cpu
@pytest.mark.parametrize(
    "budgets, max_items, expected_vision, expected_vae",
    [
        ([1024, 2048, 4096, 8192], 2, [1024, 2048], 37),
        ([1024, 2048, 4096], 2, [1024, 2048], 37),
        (None, 0, [1024, 2048, 4096, 8192, 16384, 32768], 37),
        (None, 2, [1024, 2048], 37),
        ([1024, 3500, 4096], 2, [1024, 3500], 37),
        ([1024, 3500, 4096], 0, [1024, 3500, 4096], 37),
        ([1024, 1024, 3500, 4096], 2, [1024, 3500], 37),
        ([256, 512], 2, [], 0),
    ],
)
def test_capture_only_selectable_budgets(monkeypatch, budgets, max_items, expected_vision, expected_vae):
    from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

    manager = _cpu_manager(budgets, max_items=max_items)
    captured = []

    def record_capture(encoder, budget, path="default", axis_keys=()):
        captured.append((encoder, budget, axis_keys))

    # Exercise the real capture traversal, intercepting only the GPU operation.
    monkeypatch.setattr(EncoderCudaGraphManager, "_capture_budget_graph", record_capture)
    assert manager.get_num_graphs_to_capture() == len(expected_vision) + expected_vae
    manager.capture(object())
    vision = [(b, axes) for enc, b, axes in captured if enc is manager.vision]
    vae = [(b, axes) for enc, b, axes in captured if enc is manager.vae]
    assert sorted(b for b, _ in vision) == expected_vision
    assert len(vae) == expected_vae
    assert len({axes for _, axes in vae}) == expected_vae
    assert all(b >= h * w // 16**2 for b, ((h, w),) in vae)
    assert len(captured) == manager.get_num_graphs_to_capture()


@pytest.mark.cpu
@pytest.mark.parametrize("encoder_name", ["vision", "vae"])
@pytest.mark.parametrize("budgets", [[1024, 2048, 4096, 8192], [1024, 3500, 4096], [8192], [256, 512]])
def test_packing_uses_captured_graphs_and_preserves_outputs(monkeypatch, encoder_name, budgets):
    from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

    wrapper = _cpu_manager(budgets)
    manager = getattr(wrapper, encoder_name)
    captured = set()

    def record_capture(encoder, budget, path="default", axis_keys=()):
        captured.add((budget, axis_keys))

    monkeypatch.setattr(EncoderCudaGraphManager, "_capture_budget_graph", record_capture)
    manager.capture(object())
    ids = torch.arange(1, 4, dtype=torch.float32)
    if encoder_name == "vision":
        tokens = 1024
        inputs = {
            "pixels": ids[:, None, None].expand(-1, 1024, 1),
            "mask": torch.ones(3, 1024, dtype=torch.bool),
            "shapes": torch.tensor([[32, 32]] * 3),
        }
    else:
        tokens = 4096
        inputs = {"pixels": ids[:, None, None, None].expand(-1, 1, 1024, 1024)}
        # Capture inputs must have the same one-image shape as replay batches.
        for budget, axes in captured:
            capture = manager.model.prepare_encoder_cudagraph_capture_inputs(
                budget, manager.max_batch_size, 0, torch.device("cpu"), torch.float32, axis_keys=axes
            )
            assert capture.values["pixels"].shape[0] == 1

    buffers: dict[tuple[int, tuple], torch.Tensor] = {}
    replay_batches = []

    def values(kwargs):
        pixels = kwargs["pixels"]
        image_ids = pixels[(slice(None),) + (0,) * (pixels.ndim - 1)]
        return image_ids.repeat_interleave(tokens)[:, None].expand(-1, 2)

    def replay(kwargs, budget, path="default", axis_keys=()):
        key = (budget, axis_keys)
        assert key in captured
        count = len(kwargs["pixels"])
        replay_batches.append(count)
        capacity = min(manager.max_batch_size, budget // tokens) * tokens
        buffer = buffers.setdefault(key, torch.empty(capacity, 2))
        buffer[: count * tokens].copy_(values(kwargs))
        return buffer

    monkeypatch.setattr(manager, "_run_budget_graph", replay)
    monkeypatch.setattr(manager.model, "encoder_eager_forward", lambda kwargs, path="default": values(kwargs))
    outputs = manager.execute(inputs)
    if captured:
        assert replay_batches == ([1, 1, 1] if encoder_name == "vae" else [2, 1])
    else:
        assert not replay_batches
        assert manager.graph_misses == 3
    # Subsequent replays overwrite static buffers; earlier images must survive.
    for buffer in buffers.values():
        buffer.fill_(-1)
    assert len(outputs) == 3
    for image_id, output in zip(ids, outputs):
        torch.testing.assert_close(output, torch.full((tokens, 2), image_id.item()))


@pytest.mark.cpu
@pytest.mark.parametrize("encoder_name", ["vision", "vae"])
@pytest.mark.parametrize("num_images", [1, 3])
@pytest.mark.parametrize("rank", [0, 1])
def test_data_mode_shards_and_gathers_in_input_order(monkeypatch, encoder_name, num_images, rank):
    from vllm.v1.worker import encoder_cudagraph as upstream

    wrapper = _cpu_manager([1024, 2048, 4096, 8192], tp_size=2)
    manager = getattr(wrapper, encoder_name)
    assert manager.use_dp
    ids = torch.arange(1, num_images + 1, dtype=torch.float32)
    if encoder_name == "vision":
        inputs = {
            "pixels": ids[:, None, None].expand(-1, 1024, 3),
            "mask": torch.ones(num_images, 1024, dtype=torch.bool),
            "shapes": torch.tensor([[32, 32]] * num_images),
        }
        tokens = 1024
    else:
        inputs = {"pixels": ids[:, None, None, None].expand(-1, 3, 1024, 1024)}
        tokens = 4096
    expected = [torch.full((tokens, 2), float(i)) for i in range(1, num_images + 1)]
    assignments = [list(range(r, num_images, 2)) for r in range(2)]
    capacity = max(map(len, assignments)) * tokens
    gathered = torch.zeros(2 * capacity, 2)
    for r, indices in enumerate(assignments):
        if indices:
            values = torch.cat([expected[i] for i in indices])
            gathered[r * capacity : r * capacity + len(values)] = values
    local_ids = []

    def graph_output(kwargs, budget, path="default", axis_keys=()):
        selected = kwargs["pixels"].reshape(len(kwargs["pixels"]), -1)[:, 0].tolist()
        if encoder_name == "vae":
            assert len(selected) == 1
        local_ids.extend(selected)
        return torch.cat([expected[int(i) - 1] for i in selected])

    def all_gather(local, dim):
        assert dim == 0
        live_tokens = len(assignments[rank]) * tokens
        torch.testing.assert_close(local[:live_tokens], gathered[rank * capacity : rank * capacity + live_tokens])
        assert local.shape == (capacity, 2)
        return gathered

    monkeypatch.setattr(upstream, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(upstream, "get_tensor_model_parallel_rank", lambda: rank)
    gather = Mock(side_effect=all_gather)
    monkeypatch.setattr(upstream, "tensor_model_parallel_all_gather", gather)
    monkeypatch.setattr(manager, "_run_budget_graph", graph_output)
    # Keep the actual upstream execute/shard/gather and greedy packing paths.
    actual = manager.execute(inputs)
    assert local_ids == [float(i + 1) for i in assignments[rank]]
    gather.assert_called_once()
    assert len(actual) == num_images
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference)


@pytest.mark.cpu
def test_vision_shape_fallback_preserves_model_data_parallel_path():
    from vllm_omni.model_executor.models.hunyuan_image3.encoder_cudagraph import HunyuanImage3EncoderCudaGraphManager
    from vllm_omni.model_executor.models.hunyuan_image3.hunyuan_image3 import HunyuanImage3ForConditionalGeneration

    pixels = torch.ones(1, 2, 3)  # Not the four-patch capture shape.
    mask = torch.ones(1, 2, dtype=torch.bool)
    shapes = torch.tensor([[1, 2]])
    embeddings = torch.ones(1, 2, 4)
    model = SimpleNamespace(
        use_data_parallel=True,
        _parse_and_validate_image_input=lambda **kwargs: {
            "pixel_values": {
                "vit_pixel_values": pixels,
                "vit_pixel_attention_mask": mask,
                "vit_spatial_shapes": shapes,
                "vae_pixel_values": [],
            }
        },
        _vit_encode_dp=Mock(return_value=embeddings),
        vision_aligner=lambda x: x,
        _combine_image_embeddings=Mock(return_value=[embeddings[0]]),
    )
    model._vit_encode = lambda *args: HunyuanImage3ForConditionalGeneration._vit_encode(model, *args)
    manager = object.__new__(HunyuanImage3EncoderCudaGraphManager)
    manager.model = model
    manager.vision = SimpleNamespace(model=SimpleNamespace(min_tokens=4), graph_misses=0, execute=Mock())
    manager.vae = SimpleNamespace(model=SimpleNamespace(factor=16))
    actual = manager.execute({})
    model._vit_encode_dp.assert_called_once_with(pixels, mask, shapes)
    manager.vision.execute.assert_not_called()
    assert manager.vision.graph_misses == 1
    torch.testing.assert_close(actual[0], embeddings[0])


class _LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(4, 32)

    def embed_input_ids(self, input_ids):
        return self.embedding(input_ids)


@contextmanager
def _cuda_encoder_model(tmp_path, dtype, *, rank=0, world_size=1):
    from vllm.config import CompilationConfig, ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.config.multimodal import MultiModalConfig
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.utils.torch_utils import set_default_torch_dtype

    from vllm_omni.model_executor.models.hunyuan_image3.autoencoder_kl_3d import AutoencoderKLConv3D
    from vllm_omni.model_executor.models.hunyuan_image3.hunyuan_image3 import (
        HunyuanImage3ForConditionalGeneration,
        TimestepEmbedder,
        UNetDown,
    )
    from vllm_omni.model_executor.models.hunyuan_image3.siglip2 import Siglip2VisionTransformer

    torch.manual_seed(42)
    from vllm_omni.platforms import current_omni_platform

    current_omni_platform.set_device(rank)
    config = VllmConfig()
    config.compilation_config = CompilationConfig(
        cudagraph_mm_encoder=True,
        encoder_cudagraph_token_budgets=[256, 512],
        encoder_cudagraph_max_vision_items_per_batch=2,
    )
    config.parallel_config = ParallelConfig(tensor_parallel_size=world_size)
    config.model_config = SimpleNamespace(
        is_moe=False,
        multimodal_config=MultiModalConfig(
            limit_per_prompt={"image": 4, "video": 0}, mm_encoder_tp_mode="data" if world_size > 1 else "weights"
        ),
    )
    config.scheduler_config = SimpleNamespace(max_num_batched_tokens=512)
    vit_config = dict(
        hidden_size=288,
        patch_size=2,
        num_channels=3,
        num_patches=4,
        num_attention_heads=4,
        intermediate_size=128,
        hidden_act="gelu_pytorch_tanh",
        num_hidden_layers=1,
        layer_norm_eps=1e-6,
    )
    try:
        with set_current_vllm_config(config), set_default_torch_dtype(dtype):
            init_distributed_environment(
                world_size=world_size,
                rank=rank,
                distributed_init_method=f"file://{tmp_path / 'dist'}",
                local_rank=rank,
                timeout=timedelta(seconds=120),
            )
            initialize_model_parallel(tensor_model_parallel_size=world_size)
            model = object.__new__(HunyuanImage3ForConditionalGeneration)
            nn.Module.__init__(model)
            model.config = SimpleNamespace(
                hidden_size=32,
                image_base_size=32,
                patch_size=1,
                vae_downsample_factor=[2, 2],
                vit_processor={"max_num_patches": 4},
            )
            model.use_data_parallel = world_size > 1
            model.vision_model = Siglip2VisionTransformer(vit_config)
            model.vision_aligner = nn.Linear(288, 32)
            model.vae = AutoencoderKLConv3D(
                in_channels=3,
                out_channels=3,
                latent_channels=2,
                block_out_channels=(32, 32),
                layers_per_block=1,
                ffactor_spatial=2,
                ffactor_temporal=1,
                sample_size=32,
                sample_tsize=4,
                scaling_factor=0.5,
                shift_factor=0.1,
                only_encoder=True,
            )
            model.time_embed = TimestepEmbedder(32)
            model.patch_embed = UNetDown(
                patch_size=1, in_channels=2, emb_channels=32, hidden_channels=32, out_channels=32
            )
            model.model = _LanguageModel()
            model._mrope_joint_img_sep_token_id = 1
            model = model.to(device="cuda", dtype=dtype).eval()
            for name, parameter in model.named_parameters():
                if parameter.ndim > 1:
                    nn.init.normal_(parameter, std=0.02)
                elif name.endswith("weight"):
                    nn.init.ones_(parameter)
                else:
                    nn.init.zeros_(parameter)
            runner = object.__new__(OmniGPUModelRunner)
            runner.compilation_config = config.compilation_config
            runner.supports_mm_inputs = True
            runner.vllm_config, runner.device, runner.dtype = config, torch.device("cuda"), dtype
            runner.get_model = lambda: model
            manager = runner._create_encoder_cudagraph_manager()
            # Keep the fixture small while capturing a few real VAE resolutions.
            manager.vae.model.resolutions = ((32, 32), (16, 64), (64, 64))
            manager.vae._capture_axes = (manager.vae.model.resolutions,)
            try:
                yield model, manager
            finally:
                manager.clear()
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.fixture(params=[torch.float16, torch.bfloat16])
def cuda_model(tmp_path, request):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    with _cuda_encoder_model(tmp_path, request.param) as model_and_manager:
        yield model_and_manager


def _inputs(sizes, seed=42, dtype=torch.float16):
    batch = len(sizes)
    images = [torch.randn(3, h, w, device="cuda") for h, w in sizes]
    shapes = torch.tensor([[1, 3] if i % 2 else [2, 2] for i in range(batch)], device="cuda")
    mask = torch.arange(4, device="cuda")[None] < shapes.prod(-1)[:, None]
    result = dict(
        vit_pixel_values=torch.randn(batch, 4, 12, device="cuda", dtype=dtype),
        vit_pixel_attention_mask=mask,
        vit_spatial_shapes=shapes,
        vae_pixel_values=torch.cat([image.flatten() for image in images]),
        vae_pixel_size=torch.tensor([3 * h * w for h, w in sizes]),
        vae_token_grid_hw=torch.tensor([[h // 2, w // 2] for h, w in sizes], device="cuda"),
    )
    if seed is not None:
        result["vae_generator_seed"] = torch.full((batch,), seed, device="cuda")
    return result


def _run_stage(model, manager, inputs):
    from transformers import BatchFeature
    from vllm.multimodal.inputs import MultiModalKwargsItems

    from vllm_omni.model_executor.models.hunyuan_image3.hunyuan_image3 import HunyuanImage3MultiModalProcessor

    processor = object.__new__(HunyuanImage3MultiModalProcessor)
    fields = processor._get_mm_fields_config(inputs, {})
    items = MultiModalKwargsItems.from_hf_inputs(BatchFeature(inputs), fields)["image"]
    runner = object.__new__(OmniGPUModelRunner)
    runner.model, runner.device = model, torch.device("cuda")
    runner.encoder_cudagraph_manager = manager
    runner.observability_config = runner.lora_config = None
    runner.is_multimodal_pruning_enabled = runner.requires_sequential_video_encoding = False
    runner.encoder_cache = {}
    runner.maybe_save_ec_to_connector = lambda *args: None
    runner.requests = {
        "request": SimpleNamespace(
            mm_features=[
                SimpleNamespace(data=item, modality="image", identifier=str(i), mm_position=None)
                for i, item in enumerate(items)
            ]
        )
    }
    scheduler = SimpleNamespace(
        scheduled_encoder_inputs={"request": list(range(len(items)))},
        ec_manager_metadata=None,
        free_encoder_mm_hashes=[],
    )
    outputs = runner._execute_mm_encoder(scheduler)
    assert len(runner.encoder_cache) == len(items)
    return outputs


@pytest.mark.nvidia_only
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@torch.inference_mode()
def test_stage_graph_replay_matches_eager_and_preserves_rng(cuda_model):
    model, manager = cuda_model
    state = torch.cuda.get_rng_state()
    from vllm.distributed import graph_capture

    with graph_capture(device=torch.device("cuda")):
        manager.capture(torch.cuda.graph_pool_handle())
    assert torch.equal(state, torch.cuda.get_rng_state()), "capture must not sample VAE latents"
    assert manager.get_num_graphs_to_capture() == 3
    assert set(manager.vae.budget_graphs["default"]) == {(256, (resolution,)) for resolution in ((32, 32), (16, 64))}
    cases = [
        ([(32, 32)], 42),
        ([(32, 32), (32, 32)], 42),
        ([(32, 32), (16, 64), (32, 32)], 42),
        ([(48, 48)], 42),
        ([(64, 64)], 42),
        ([(32, 32), (16, 64)], None),
        ([(32, 32)], 101),
    ]
    for sizes, seed in cases:
        inputs = _inputs(sizes, seed, model.vae.dtype)
        state = torch.cuda.get_rng_state()
        eager = _run_stage(model, None, inputs)
        after_eager = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(state)
        graph = _run_stage(model, manager, inputs)
        assert torch.equal(after_eager, torch.cuda.get_rng_state())
        assert len(graph) == len(eager)
        for expected, actual in zip(eager, graph):
            patches = inputs["vit_pixel_values"].shape[1]
            torch.testing.assert_close(actual[:-patches], expected[:-patches], rtol=0, atol=0)
            # Splitting a vision batch can change reduced-precision GEMM rounding.
            tolerance = 2 * torch.finfo(expected.dtype).eps
            torch.testing.assert_close(actual[-patches:], expected[-patches:], rtol=tolerance, atol=tolerance)
    assert manager.vision.graph_hits > 0 and manager.vae.graph_hits > 0
    assert manager.vae.graph_misses == 2


@pytest.mark.nvidia_only
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("capture_vision", [True, False])
@torch.inference_mode()
def test_unusable_capture_budgets_fall_back_to_eager(cuda_model, capture_vision):
    from vllm.distributed import graph_capture

    model, manager = cuda_model
    manager.vae.token_budgets = [128]
    manager.vae.path_token_budgets = {"default": [128]}
    if not capture_vision:
        manager.vision.token_budgets = [2]
        manager.vision.path_token_budgets = {"default": [2]}
    assert manager.vae.get_num_graphs_to_capture() == 0
    with graph_capture(device=torch.device("cuda")):
        manager.capture(torch.cuda.graph_pool_handle())
    assert manager.is_captured()
    assert not manager.vae.budget_graphs
    inputs = _inputs([(32, 32), (16, 64)], dtype=model.vae.dtype)
    expected = _run_stage(model, None, inputs)
    actual = _run_stage(model, manager, inputs)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
    assert manager.vae.graph_hits == 0
    assert manager.vae.graph_misses == 2
    assert (manager.vision.graph_hits > 0) == capture_vision


@torch.inference_mode()
def _data_parallel_worker(rank, tmp_path, dtype):
    from vllm.distributed import graph_capture

    with _cuda_encoder_model(tmp_path, dtype, rank=rank, world_size=2) as (model, manager):
        assert manager.vision.use_dp and manager.vae.use_dp
        with graph_capture(device=torch.device("cuda", rank)):
            manager.capture(torch.cuda.graph_pool_handle())
        # One image leaves a rank empty. Three images exercise uneven sharding,
        # VAE resolution regrouping and reconstruction of the original order.
        cases = [
            ([(32, 32)], False, 42),
            ([(32, 32), (16, 64), (32, 32)], False, 42),
            ([(32, 32), (16, 64), (32, 32)], False, None),
            ([(64, 64)], False, 42),  # Known resolution, above every budget.
            ([(32, 32)], True, 42),
        ]
        for index, (sizes, shape_fallback, seed) in enumerate(cases):
            # Every rank must receive the same inputs and sampling RNG state.
            torch.manual_seed(100 + index)
            inputs = _inputs(sizes, seed=seed, dtype=dtype)
            if shape_fallback:
                inputs["vit_pixel_values"] = inputs["vit_pixel_values"][:, :2]
                inputs["vit_pixel_attention_mask"] = inputs["vit_pixel_attention_mask"][:, :2]
                inputs["vit_spatial_shapes"] = torch.tensor([[1, 2]], device="cuda")
            state = torch.cuda.get_rng_state()
            expected = _run_stage(model, None, inputs)
            after_eager = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(state)
            actual = _run_stage(model, manager, inputs)
            assert torch.equal(after_eager, torch.cuda.get_rng_state())
            assert len(actual) == len(expected)
            for result, reference in zip(actual, expected):
                patches = inputs["vit_pixel_values"].shape[1]
                torch.testing.assert_close(result[:-patches], reference[:-patches], rtol=0, atol=0)
                tolerance = 2 * torch.finfo(dtype).eps
                torch.testing.assert_close(result[-patches:], reference[-patches:], rtol=tolerance, atol=tolerance)
        assert manager.vision.graph_misses == 1  # Shape fallback on both ranks.
        if rank == 0:
            assert manager.vision.graph_hits > 0 and manager.vae.graph_hits > 0


@pytest.mark.nvidia_only
@hardware_test(res={"cuda": "L4"}, num_cards=2)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_data_mode_two_ranks_match_eager(tmp_path, dtype):
    if torch.accelerator.device_count() < 2:
        pytest.skip("Two CUDA devices required")
    torch.multiprocessing.spawn(_data_parallel_worker, args=(tmp_path, dtype), nprocs=2)
