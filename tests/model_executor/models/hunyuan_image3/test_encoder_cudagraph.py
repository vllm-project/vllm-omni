# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from contextlib import contextmanager
from datetime import timedelta
from types import SimpleNamespace

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
def test_stage_does_not_create_manager_when_disabled_or_non_cuda(monkeypatch, mocker, enabled, cuda):
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    from vllm_omni.worker import gpu_model_runner

    factory = mocker.Mock()
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


def _cpu_manager(budgets, *, max_items=2, tp_size=1):
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
        config=SimpleNamespace(hidden_size=2, vit_processor={"max_num_patches": 1024}),
    )
    return HunyuanImage3EncoderCudaGraphManager(config, torch.device("cpu"), torch.float32, model)


@pytest.mark.cpu
@pytest.mark.parametrize(
    "budgets, max_items, expected_vision",
    [
        ([1024, 2048, 4096, 8192], 2, [1024, 2048]),
        ([1024, 2048, 4096], 2, [1024, 2048]),
        (None, 0, [1024, 2048, 4096, 8192, 16384, 32768]),
        (None, 2, [1024, 2048]),
        ([1024, 3500, 4096], 2, [1024]),
        ([1024, 3500, 4096], 0, [1024]),
        ([1024, 1024, 3500, 4096], 2, [1024]),
        ([4096, 12288], 8, [4096]),
        ([1024, 2048, 4096, 16384], 16, [1024, 2048, 4096, 16384]),
        ([256, 512], 2, []),
    ],
)
def test_capture_only_selectable_budgets(monkeypatch, budgets, max_items, expected_vision):
    from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

    manager = _cpu_manager(budgets, max_items=max_items)
    captured = []

    def record_capture(encoder, budget, path="default", axis_keys=()):
        assert encoder is manager.vision
        assert path == "default" and axis_keys == ()
        captured.append(budget)

    # Exercise the real capture traversal, intercepting only the GPU operation.
    # The minimal model has no VAE: even inferred budgets must capture only vision.
    monkeypatch.setattr(EncoderCudaGraphManager, "_capture_budget_graph", record_capture)
    assert manager.get_num_graphs_to_capture() == len(expected_vision)
    manager.capture(object())
    assert sorted(captured) == expected_vision
    assert len(captured) == manager.get_num_graphs_to_capture()
    assert set(manager.get_cumulative_stats()["encoders"]) == {"vision"}
    assert manager.is_captured()
    manager.clear()
    assert not manager.is_captured()


@pytest.mark.cpu
@pytest.mark.parametrize("max_items", [1, 2, 8, 0])
@pytest.mark.parametrize(
    "budgets",
    [
        None,
        [1024, 2048, 4096, 8192],
        [1024, 3500, 4096],
        [1024, 1024, 3500, 4096],
        [4096, 12288],
        [1024, 2048, 4096, 16384],
        [8192],
        [256, 512],
    ],
)
def test_packing_uses_captured_graphs_and_preserves_outputs(monkeypatch, budgets, max_items):
    from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

    manager = _cpu_manager(budgets, max_items=max_items).vision
    captured = set()

    def record_capture(encoder, budget, path="default", axis_keys=()):
        captured.add((budget, axis_keys))

    monkeypatch.setattr(EncoderCudaGraphManager, "_capture_budget_graph", record_capture)
    manager.capture(object())
    reference = EncoderCudaGraphManager(manager.vllm_config, manager.device, manager.dtype, manager.model)
    tokens = manager.model.min_tokens
    buffers: dict[tuple[int, tuple], torch.Tensor] = {}
    expected_replays = []
    actual_replays = []
    selected = set()

    def values(kwargs):
        image_ids = kwargs["pixels"][:, 0, 0]
        return image_ids.repeat_interleave(tokens)[:, None].expand(-1, 2)

    def reference_replay(kwargs, budget, path="default", axis_keys=()):
        # Run the unfiltered upstream manager to get the reference batches.
        # Neither its packing nor this oracle calls our capture planner.
        expected_replays.append((len(kwargs["pixels"]), budget, axis_keys))
        return values(kwargs)

    def replay(kwargs, budget, path="default", axis_keys=()):
        key = (budget, axis_keys)
        assert key in captured
        selected.add(key)
        count = len(kwargs["pixels"])
        actual_replays.append((count, budget, axis_keys))
        capacity = min(manager.max_batch_size, budget // tokens) * tokens
        buffer = buffers.setdefault(key, torch.empty(capacity, 2))
        buffer[: count * tokens].copy_(values(kwargs))
        return buffer

    monkeypatch.setattr(reference, "_run_budget_graph", reference_replay)
    monkeypatch.setattr(manager, "_run_budget_graph", replay)
    monkeypatch.setattr(manager.model, "encoder_eager_forward", lambda kwargs, path="default": values(kwargs))
    # Include partial batches and a request spanning multiple replay batches.
    for num_images in range(1, manager.max_batch_size + 2):
        ids = torch.arange(1, num_images + 1, dtype=torch.float32)
        inputs = {
            "pixels": ids[:, None, None].expand(-1, tokens, 1),
            "mask": torch.ones(num_images, tokens, dtype=torch.bool),
            "shapes": torch.tensor([[32, 32]] * num_images),
        }
        expected_replays.clear()
        actual_replays.clear()
        expected = reference.execute(inputs)
        outputs = manager.execute(inputs)
        assert actual_replays == expected_replays
        # Subsequent replays overwrite static buffers; earlier images must survive.
        for buffer in buffers.values():
            buffer.fill_(-1)
        assert len(outputs) == num_images
        for image_id, output, ref in zip(ids, outputs, expected):
            torch.testing.assert_close(output, ref)
            torch.testing.assert_close(output, torch.full((tokens, 2), image_id.item()))
    assert selected == captured
    assert manager.graph_misses == reference.graph_misses


@pytest.mark.cpu
@pytest.mark.parametrize("num_images", [1, 3])
@pytest.mark.parametrize("rank", [0, 1])
def test_data_mode_shards_and_gathers_in_input_order(monkeypatch, mocker, num_images, rank):
    from vllm.v1.worker import encoder_cudagraph as upstream

    wrapper = _cpu_manager([1024, 2048, 4096, 8192], tp_size=2)
    manager = wrapper.vision
    assert manager.use_dp
    ids = torch.arange(1, num_images + 1, dtype=torch.float32)
    inputs = {
        "pixels": ids[:, None, None].expand(-1, 1024, 3),
        "mask": torch.ones(num_images, 1024, dtype=torch.bool),
        "shapes": torch.tensor([[32, 32]] * num_images),
    }
    tokens = 1024
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
    gather = mocker.Mock(side_effect=all_gather)
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
def test_vision_shape_fallback_preserves_model_data_parallel_path(mocker):
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
        _vit_encode_dp=mocker.Mock(return_value=embeddings),
        vision_aligner=lambda x: x,
        _encode_vae_tokens=mocker.Mock(return_value=[]),
        _combine_image_embeddings=mocker.Mock(return_value=[embeddings[0]]),
    )
    model._vit_encode = lambda *args: HunyuanImage3ForConditionalGeneration._vit_encode(model, *args)
    manager = object.__new__(HunyuanImage3EncoderCudaGraphManager)
    manager.model = model
    manager.vision = SimpleNamespace(model=SimpleNamespace(min_tokens=4), graph_misses=0, execute=mocker.Mock())
    actual = manager.execute({})
    model._vit_encode_dp.assert_called_once_with(pixels, mask, shapes)
    model._encode_vae_tokens.assert_called_once_with([], None)
    manager.vision.execute.assert_not_called()
    assert manager.vision.graph_misses == 1
    torch.testing.assert_close(actual[0], embeddings[0])


@pytest.mark.cpu
@pytest.mark.parametrize("seed_values", [None, [], [0], [42, 42, 42]])
def test_vae_tokens_preserve_image_order_and_generator_state(monkeypatch, seed_values):
    from vllm_omni.model_executor.models.hunyuan_image3.hunyuan_image3 import HunyuanImage3ForConditionalGeneration

    model = object.__new__(HunyuanImage3ForConditionalGeneration)
    nn.Module.__init__(model)
    images = [torch.full((3, h, w), float(i)) for i, (h, w) in enumerate([(2, 3), (3, 2), (2, 3)])]
    seed = None if seed_values is None else torch.tensor(seed_values, dtype=torch.long)
    calls = []

    def encode(image, cfg_factor=1, generator=None):
        calls.append((image, cfg_factor, generator))
        return torch.tensor([2.0]), image + torch.rand(image.shape, generator=generator)

    monkeypatch.setattr(model, "_vae_encode", encode)
    monkeypatch.setattr(model, "time_embed", lambda t: t + 5, raising=False)
    monkeypatch.setattr(model, "patch_embed", lambda latents, time: (latents + time, None, None), raising=False)
    # Independently predict the random draws, without calling either encoding
    # path. A shared bug in their helper must not make the comparison pass.
    seeded = bool(seed_values)
    reference = torch.Generator().manual_seed(seed_values[0] if seeded else 123)
    expected = [image.unsqueeze(0) + torch.rand((1, *image.shape), generator=reference) + 7 for image in images]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        before = torch.get_rng_state()
        actual = model._encode_vae_tokens(images, seed)
        assert torch.equal(torch.get_rng_state(), before if seeded else reference.get_state())
    assert len(calls) == len(actual) == len(images)
    generator = calls[0][2]
    if seeded:
        assert generator is not None
        assert torch.equal(generator.get_state(), reference.get_state())
    else:
        assert generator is None
    for (image, cfg_factor, used_generator), original, output, expected_output in zip(calls, images, actual, expected):
        assert cfg_factor == 1 and used_generator is generator
        torch.testing.assert_close(image, original.unsqueeze(0), rtol=0, atol=0)
        torch.testing.assert_close(output, expected_output, rtol=0, atol=0)


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
        sleep_mode_offload_cudagraph=False,
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
@pytest.mark.parametrize("sparse_budgets", [False, True])
@torch.inference_mode()
def test_stage_graph_replay_matches_eager_and_preserves_rng(cuda_model, mocker, sparse_budgets):
    model, manager = cuda_model
    if sparse_budgets:
        # Four-patch equivalent of the 1024/3500/4096 packing regression.
        manager.vision.token_budgets = [4, 13, 16]
        manager.vision.path_token_budgets = {"default": [4, 13, 16]}
    state = torch.cuda.get_rng_state()
    from vllm.distributed import graph_capture

    with graph_capture(device=torch.device("cuda")):
        manager.capture(torch.cuda.graph_pool_handle())
    assert torch.equal(state, torch.cuda.get_rng_state()), "capture must not sample VAE latents"
    assert manager.get_num_graphs_to_capture() == 1
    assert set(manager.vision.budget_graphs["default"]) == {4 if sparse_budgets else 256}
    vae_encode = mocker.spy(model, "_vae_encode")
    replay = mocker.spy(manager.vision, "_run_budget_graph")
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
        vae_encode.reset_mock()
        eager = _run_stage(model, None, inputs)
        assert vae_encode.call_count == len(sizes)
        vae_encode.reset_mock()
        after_eager = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(state)
        graph = _run_stage(model, manager, inputs)
        assert vae_encode.call_count == len(sizes)
        assert torch.equal(after_eager, torch.cuda.get_rng_state())
        assert len(graph) == len(eager)
        for expected, actual in zip(eager, graph):
            patches = inputs["vit_pixel_values"].shape[1]
            torch.testing.assert_close(actual[:-patches], expected[:-patches], rtol=0, atol=0)
            # Splitting a vision batch can change reduced-precision GEMM rounding.
            tolerance = 2 * torch.finfo(expected.dtype).eps
            torch.testing.assert_close(actual[-patches:], expected[-patches:], rtol=tolerance, atol=tolerance)
    assert manager.vision.graph_hits > 0
    if sparse_budgets:
        assert all(len(call.args[0]["pixels"]) == 1 for call in replay.call_args_list)


@pytest.mark.nvidia_only
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@torch.inference_mode()
def test_unusable_capture_budgets_fall_back_to_eager(cuda_model):
    from vllm.distributed import graph_capture

    model, manager = cuda_model
    manager.vision.token_budgets = [2]
    manager.vision.path_token_budgets = {"default": [2]}
    assert manager.get_num_graphs_to_capture() == 0
    with graph_capture(device=torch.device("cuda")):
        manager.capture(torch.cuda.graph_pool_handle())
    assert manager.is_captured()
    assert not manager.vision.budget_graphs
    inputs = _inputs([(32, 32), (16, 64)], dtype=model.vae.dtype)
    expected = _run_stage(model, None, inputs)
    actual = _run_stage(model, manager, inputs)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
    assert manager.vision.graph_hits == 0
    assert manager.vision.graph_misses == 2


@torch.inference_mode()
def _data_parallel_worker(rank, tmp_path, dtype):
    from vllm.distributed import graph_capture

    with _cuda_encoder_model(tmp_path, dtype, rank=rank, world_size=2) as (model, manager):
        assert manager.vision.use_dp
        with graph_capture(device=torch.device("cuda", rank)):
            manager.capture(torch.cuda.graph_pool_handle())
        # One image leaves a rank empty. Three images exercise uneven sharding,
        # SigLIP2 gather order with differently sized eager VAE inputs.
        cases = [
            ([(32, 32)], False, 42),
            ([(32, 32), (16, 64), (32, 32)], False, 42),
            ([(32, 32), (16, 64), (32, 32)], False, None),
            ([(64, 64)], False, 42),  # VAE size is independent of graph budgets.
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
            assert manager.vision.graph_hits > 0


@pytest.mark.nvidia_only
@hardware_test(res={"cuda": "L4"}, num_cards=2)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_data_mode_two_ranks_match_eager(tmp_path, dtype):
    if torch.accelerator.device_count() < 2:
        pytest.skip("Two CUDA devices required")
    torch.multiprocessing.spawn(_data_parallel_worker, args=(tmp_path, dtype), nprocs=2)
