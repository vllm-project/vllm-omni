# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""One-GPU tests with actual Ming transformer/ByT5 mapper, no checkpoint downloads.

Inputs: real bridge request builder and production preparation / InputBatch.
Expected: model API shape, reference isolation, finite values, batch permutation
equivariance, and B1/B2 deterministic agreement. These complement the independent
Euler oracle in the CPU suite; agreement alone is not a correctness proof.
"""

from pathlib import Path

import pytest
import torch
from PIL import Image
from transformers import ByT5Tokenizer, T5Config, T5EncoderModel

from tests.diffusion.models.ming_flash_omni.test_pipeline_ming_imagegen import (
    pipeline as pipeline,  # Re-export the shared pytest fixture.
)
from tests.diffusion.models.ming_flash_omni.test_pipeline_ming_imagegen import (
    prepared,
    request,
)
from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec, DiffusionParallelConfig
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.forward_context import get_forward_context, set_forward_context
from vllm_omni.diffusion.models.ming_flash_omni.byte5_encoder import MingByT5Encoder
from vllm_omni.diffusion.models.ming_flash_omni.ming_zimage_transformer import MingZImageTransformer2DModel
from vllm_omni.diffusion.models.ming_flash_omni.t5_block_mapper import T5EncoderBlockByT5Mapper
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.fixture
def distributed(tmp_path):
    """Production world/TP/SP initialization at world size one."""
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        backend="nccl",
        distributed_init_method=(tmp_path / "dist_init").as_uri(),
    )
    initialize_model_parallel()
    yield
    destroy_distributed_env()


def real_pipeline(pipe, dtype):
    pipe.device = pipe._execution_device = torch.device("cuda", 0)
    pipe._dtype = pipe.od_config.dtype = dtype
    pipe.vae.to(device=pipe.device, dtype=dtype)
    pipe.condition_encoder.to(device=pipe.device, dtype=dtype)
    pipe.od_config.parallel_config = DiffusionParallelConfig()
    pipe.od_config.diffusion_attention_config = AttentionConfig(default=AttentionSpec(backend="TORCH_SDPA"))
    with torch.random.fork_rng(devices=[0]), set_current_diffusion_config(pipe.od_config):
        torch.manual_seed(61)
        model = (
            MingZImageTransformer2DModel(
                in_channels=4,
                dim=32,
                n_layers=1,
                n_refiner_layers=1,
                n_heads=4,
                n_kv_heads=4,
                cap_feat_dim=4,
                axes_dims=[2, 2, 4],
                axes_lens=[2048, 512, 512],
            )
            .to(device=pipe.device, dtype=dtype)
            .eval()
        )
        # vLLM layers allocate unloaded weights. Supply finite nonzero test
        # weights instead of relying on uninitialized memory or zero predictions.
        for name, param in model.named_parameters():
            if name.endswith("norm.weight") or name.endswith("norm1.weight") or name.endswith("norm2.weight"):
                param.data.fill_(1)
            else:
                param.data.uniform_(-0.05, 0.05)
    pipe.transformer = model
    return pipe


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("capacity", [2, 4])
def test_actual_transformer_reference_rows_axis_and_b1_b2(pipeline, distributed, dtype, capacity):  # noqa: F811 - imported pytest fixture
    """Regression: wrong frame insertion, source/model dtype mismatch and reference broadcast.

    Input source: bridge -> encode -> genuine InputBatch -> actual Ming DiT.
    Why valid: same geometry, two distinct reference images, independent request seeds;
    capacity four comes from the high-throughput and stepwise deployment profiles.
    Expected source: each row's reference is private; permutation/singleton execution
    must preserve that row's prediction and the full deterministic denoise solution.
    """
    pipeline.od_config.max_num_seqs = capacity
    pipe = real_pipeline(pipeline, dtype)
    images = [Image.new("RGB", (16, 16), "red"), Image.new("RGB", (16, 16), "blue")]

    def requests():
        return [
            request(rid, reference=image, seed=seed)
            for rid, image, seed in zip(["A", "B"], images, [11, 22], strict=True)
        ]

    states = [prepared(pipe, r) for r in requests()]
    refs = torch.cat([s.extra["ming_reference_latent"] for s in states])
    assert not torch.equal(refs[0], refs[1])
    tolerance = 1e-4 if dtype == torch.float32 else 3e-2
    with set_forward_context(omni_diffusion_config=pipe.od_config), torch.inference_mode():
        batch_prediction = pipe.denoise_step(InputBatch.make_batch(states))
        singles = torch.cat([pipe.denoise_step(InputBatch.make_batch([s])) for s in states])
        assert batch_prediction.shape == (2, 4, 8, 8)
        assert torch.isfinite(batch_prediction).all() and batch_prediction.abs().max() > 0
        torch.testing.assert_close(batch_prediction, singles, rtol=tolerance, atol=tolerance)
        permuted = pipe.denoise_step(InputBatch.make_batch(list(reversed(states))))
        torch.testing.assert_close(permuted.flip(0), singles, rtol=tolerance, atol=tolerance)

        # Test sensitivity: swapping A's actual reference must change A's result.
        original = states[0].extra["ming_reference_latent"]
        states[0].extra["ming_reference_latent"] = states[1].extra["ming_reference_latent"]
        wrong_reference = pipe.denoise_step(InputBatch.make_batch([states[0]]))
        states[0].extra["ming_reference_latent"] = original
        assert not torch.equal(wrong_reference, singles[:1]), "reference conditioning must affect the real model"

        b1 = pipe.forward(DiffusionRequestBatch(requests()))
        for _ in range(3):
            prediction = pipe.denoise_step(InputBatch.make_batch(states))
            for i, state in enumerate(states):
                pipe.step_scheduler(state, prediction[i : i + 1])
        for i, state in enumerate(states):
            assert state.denoise_completed and state.latents.dtype == torch.float32
            torch.testing.assert_close(pipe.post_decode(state).output, b1[i].output, rtol=tolerance, atol=tolerance)
        assert get_forward_context().ref_latent is None


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_actual_byt5_mapper_variable_byte_length_reaches_real_dit(pipeline, distributed):  # noqa: F811 - imported pytest fixture
    """Regression: equal glyph count/different bytes must yield batchable conditions.

    Input: genuine ByT5 tokenizer -> HF T5 encoder -> TP-aware Ming mapper ->
    prepare_encode -> actual InputBatch -> actual Ming transformer.
    Expected: fixed padded length, zero masked features, finite correctly shaped noise.
    """
    pipe = real_pipeline(pipeline, torch.float32)
    cfg = T5Config(vocab_size=384, d_model=16, d_ff=32, d_kv=8, num_layers=1, num_heads=2, dropout_rate=0)
    encoder = T5EncoderModel(cfg).to("cuda").eval()
    with set_current_diffusion_config(pipe.od_config):
        mapper = T5EncoderBlockByT5Mapper(cfg, num_layers=1, sdxl_channels=4).to("cuda").eval()
    with torch.no_grad():
        for param in mapper.parameters():
            param.uniform_(-0.05, 0.05)
    pipe.byte5 = MingByT5Encoder(ByT5Tokenizer(), encoder, mapper, max_length=32)
    (Path(pipe.od_config.model) / "byt5").mkdir()
    states = [
        prepared(pipe, request(rid, extra_args={"byte5_text": [text]})) for rid, text in [("A", "A"), ("B", "中文中文")]
    ]
    batch = InputBatch.make_batch(states)
    assert batch.prompt_embeds.shape == (2, 288, 4)
    with set_forward_context(omni_diffusion_config=pipe.od_config), torch.inference_mode():
        prediction = pipe.denoise_step(batch)
    assert prediction.shape == (2, 4, 8, 8) and torch.isfinite(prediction).all()
