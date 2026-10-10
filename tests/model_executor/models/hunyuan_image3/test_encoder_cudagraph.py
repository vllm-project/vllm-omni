# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

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


class _LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(4, 32)

    def embed_input_ids(self, input_ids):
        return self.embedding(input_ids)


@pytest.fixture(params=[torch.float16, torch.bfloat16])
def cuda_model(tmp_path, monkeypatch, request):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
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
    dtype = request.param
    from vllm_omni.platforms import current_omni_platform

    current_omni_platform.set_device(0)
    init_distributed_environment(
        world_size=1, rank=0, distributed_init_method=f"file://{tmp_path / 'dist'}", local_rank=0
    )
    initialize_model_parallel(tensor_model_parallel_size=1)
    config = VllmConfig()
    config.compilation_config = CompilationConfig(
        cudagraph_mm_encoder=True,
        encoder_cudagraph_token_budgets=[256, 512],
        encoder_cudagraph_max_vision_items_per_batch=2,
    )
    config.model_config = SimpleNamespace(multimodal_config=MultiModalConfig(limit_per_prompt={"image": 4, "video": 0}))
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
            model = object.__new__(HunyuanImage3ForConditionalGeneration)
            nn.Module.__init__(model)
            model.config = SimpleNamespace(
                hidden_size=32,
                image_base_size=32,
                patch_size=1,
                vae_downsample_factor=[2, 2],
                vit_processor={"max_num_patches": 4},
            )
            model.use_data_parallel = False
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
            monkeypatch.setattr(runner, "get_model", lambda: model)
            manager = runner._create_encoder_cudagraph_manager()
            # Keep the fixture small while capturing a few real VAE resolutions.
            manager.vae.model.resolutions = ((32, 32), (16, 64), (64, 64))
            manager.vae._capture_axes = (manager.vae.model.resolutions,)
            yield model, manager
            manager.clear()
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


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


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@torch.inference_mode()
def test_stage_graph_replay_matches_eager_and_preserves_rng(cuda_model):
    model, manager = cuda_model
    state = torch.cuda.get_rng_state()
    from vllm.distributed import graph_capture

    with graph_capture(device=torch.device("cuda")):
        manager.capture(torch.cuda.graph_pool_handle())
    assert torch.equal(state, torch.cuda.get_rng_state()), "capture must not sample VAE latents"
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
