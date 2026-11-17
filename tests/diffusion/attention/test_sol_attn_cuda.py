# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata, PackedPaddingMetadata, VideoTokenLayout
from vllm_omni.diffusion.attention.backends.sol_attn import SolAttnImpl
from vllm_omni.diffusion.attention.capabilities import ExecutionContext, SupportStatus
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]

if not torch.cuda.is_available():
    pytest.skip("CUDA is unavailable", allow_module_level=True)
pytest.importorskip("sol_attn")


@hardware_test(res={"cuda": ["L4", "H100", "RTX5090"]}, num_cards=1)
@pytest.mark.parametrize("used,sink_tokens", [(63, 5), (64, 5), (257, 5), (4097, 951)])
@pytest.mark.parametrize("compiled", [False, True])
def test_real_sol_kernel_padding_and_sink(used, sink_tokens, compiled):
    generator = torch.Generator(device="cuda").manual_seed(17)
    query, key, value = (
        torch.randn(1, used + 17, 2, 128, device="cuda", dtype=torch.bfloat16, generator=generator) for _ in range(3)
    )
    lengths = torch.tensor([0, used], dtype=torch.int32, device="cuda")
    metadata = AttentionMetadata(packed_padding=PackedPaddingMetadata(used, used, lengths, lengths))
    impl = SolAttnImpl(
        2,
        128,
        0.125,
        backend_kwargs={"sol_attn": {"dense_steps": 0, "dense_layers": "", "sink_tokens": sink_tokens}},
    )
    forward = torch.compile(impl.forward_cuda, fullgraph=True) if compiled else impl.forward_cuda
    with torch.inference_mode():
        actual = forward(query, key, value, metadata)
        reference = torch.nn.functional.scaled_dot_product_attention(
            query[:, :used].transpose(1, 2).float(),
            key[:, :used].transpose(1, 2).float(),
            value[:, :used].transpose(1, 2).float(),
            scale=0.125,
        ).transpose(1, 2)
    assert actual.shape == query.shape
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[:, used:]) == 0
    compared_rows = used if used <= 64 else sink_tokens
    torch.testing.assert_close(actual[:, :compared_rows].float(), reference[:, :compared_rows], rtol=0.02, atol=0.02)
    torch.compiler.reset()


@hardware_test(res={"cuda": ["L4", "H100", "RTX5090"]}, num_cards=1)
def test_real_sol_production_attention_entry():
    config = OmniDiffusionConfig(
        diffusion_attention_config={
            "default": {"backend": "SOL_ATTN", "sol_attn": {"dense_steps": 0, "dense_layers": ""}},
            "per_role": {"minimax_h3.token_refiner": {"backend": "TORCH_SDPA"}},
        }
    )
    with set_current_diffusion_config(config):
        layer = Attention(2, 128, False, 0.125, prefix="blocks.4.attention", role="minimax_h3.main")
        refiner = Attention(2, 128, False, 0.125, role="minimax_h3.token_refiner")
    assert isinstance(layer.attention, SolAttnImpl)
    assert refiner.attn_backend.get_name() == "SDPA"
    generator = torch.Generator(device="cuda").manual_seed(19)
    query, key, value = (
        torch.randn(1, 274, 2, 128, device="cuda", dtype=torch.bfloat16, generator=generator) for _ in range(3)
    )
    lengths = torch.tensor([0, 257], dtype=torch.int32, device="cuda")
    metadata = AttentionMetadata(
        packed_padding=PackedPaddingMetadata(257, 257, lengths, lengths),
        video_layout=VideoTokenLayout(prefix_len=5, latent_grid=(1, 12, 21)),
    )
    context = ExecutionContext(platform="cuda", require_fullgraph=True)
    assert layer.resolve_execution_path(context, query, key, value, metadata).support.status is SupportStatus.SUPPORTED
    with torch.inference_mode():
        expected = layer(query, key, value, metadata)
        actual = torch.compile(layer, fullgraph=True)(query, key, value, metadata)
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
    assert torch.count_nonzero(actual[:, 257:]) == 0
    torch.compiler.reset()


def _ulysses_worker(rank, init_method):
    from vllm_omni.diffusion.distributed.parallel_state import (
        destroy_distributed_env,
        init_distributed_environment,
        initialize_model_parallel,
    )

    device = torch.device("cuda", rank)
    current_omni_platform.set_device(device)
    init_distributed_environment(world_size=2, rank=rank, local_rank=rank, distributed_init_method=init_method)
    initialize_model_parallel(sequence_parallel_size=2, ulysses_degree=2)
    try:
        config = OmniDiffusionConfig(
            parallel_config=DiffusionParallelConfig(ulysses_degree=2),
            diffusion_attention_config={
                "default": {"backend": "SOL_ATTN", "sol_attn": {"dense_steps": 0, "dense_layers": ""}}
            },
        )
        with set_current_diffusion_config(config), set_forward_context(omni_diffusion_config=config):
            layer = Attention(4, 128, False, 0.125, prefix="blocks.4.attention", role="minimax_h3.main")
            assert layer.parallel_strategy.name == "ulysses"
            generator = torch.Generator(device=device).manual_seed(23)
            query, key, value = (
                torch.randn(1, 274, 4, 128, device=device, dtype=torch.bfloat16, generator=generator) for _ in range(3)
            )
            lengths = torch.tensor([0, 257], dtype=torch.int32, device=device)
            metadata = AttentionMetadata(
                packed_padding=PackedPaddingMetadata(257, 257, lengths, lengths),
                video_layout=VideoTokenLayout(prefix_len=5, latent_grid=(1, 12, 21)),
            )
            with torch.inference_mode():
                expected = layer.attention.forward_cuda(query, key, value, metadata).chunk(2, dim=1)[rank]
                local_query, local_key, local_value = (
                    tensor.chunk(2, dim=1)[rank].contiguous() for tensor in (query, key, value)
                )
                actual = layer(local_query, local_key, local_value, metadata)
            torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
    finally:
        destroy_distributed_env()


@hardware_test(res={"cuda": ["L4", "H100", "RTX5090"]}, num_cards=2)
def test_real_sol_ulysses_matches_local(tmp_path):
    if torch.accelerator.device_count() < 2:
        pytest.skip("Requires two CUDA GPUs")
    torch.multiprocessing.spawn(_ulysses_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2)


@hardware_test(res={"cuda": ["L4", "H100", "RTX5090"]}, num_cards=1)
def test_real_sol_wan_transformer_forward(tmp_path, monkeypatch):
    import vllm_omni.diffusion.attention.backends.sol_attn as sol_module
    from vllm_omni.diffusion.distributed.parallel_state import (
        destroy_distributed_env,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import WanTransformer3DModel

    init_distributed_environment(
        world_size=1, rank=0, local_rank=0, distributed_init_method=f"file://{tmp_path / 'wan-rendezvous'}"
    )
    initialize_model_parallel()
    try:
        configs = {
            backend: OmniDiffusionConfig(
                diffusion_attention_config={
                    "default": {"backend": "TORCH_SDPA"},
                    "per_role": {"self": {"backend": backend}},
                }
            )
            for backend in ("CUDNN_ATTN", "SOL_ATTN")
        }
        models = {}
        for backend, config in configs.items():
            with set_current_diffusion_config(config), set_forward_context(omni_diffusion_config=config):
                model = (
                    WanTransformer3DModel(
                        num_attention_heads=2,
                        attention_head_dim=128,
                        in_channels=4,
                        out_channels=4,
                        text_dim=32,
                        freq_dim=32,
                        ffn_dim=512,
                        num_layers=3,
                        rope_max_seq_len=64,
                    )
                    .to(device="cuda", dtype=torch.bfloat16)
                    .eval()
                )
                models[backend] = model
        generator = torch.Generator(device="cuda").manual_seed(29)
        with torch.no_grad():
            for name, parameter in models["CUDNN_ATTN"].named_parameters():
                offset = 1.0 if "norm" in name and parameter.ndim == 1 else 0.0
                parameter.copy_(torch.randn(parameter.shape, device="cuda", generator=generator) * 0.02 + offset)
            models["SOL_ATTN"].load_state_dict(models["CUDNN_ATTN"].state_dict())
        latents = torch.randn(1, 4, 4, 32, 32, device="cuda", dtype=torch.bfloat16, generator=generator)
        text = torch.randn(1, 16, 32, device="cuda", dtype=torch.bfloat16, generator=generator)
        timestep = torch.tensor([500.0], device="cuda", dtype=torch.bfloat16)
        calls = []
        original_op = sol_module._sol_attention_op

        def record_sparse(*args):
            calls.append(args[0].shape)
            return original_op(*args)

        monkeypatch.setattr(sol_module, "_sol_attention_op", record_sparse)
        outputs = {}
        for backend, model in models.items():
            assert all(block.attn2.attn.attn_backend.get_name() == "SDPA" for block in model.blocks)
            for step in (0, 10):
                with (
                    torch.inference_mode(),
                    set_forward_context(omni_diffusion_config=configs[backend], denoise_step_idx=step),
                ):
                    output = model(latents, timestep, text, return_dict=False)[0]
                assert output.shape == latents.shape and torch.isfinite(output).all()
                outputs[backend, step] = output
        torch.testing.assert_close(outputs["SOL_ATTN", 0], outputs["CUDNN_ATTN", 0], atol=1e-3, rtol=1e-3)
        assert calls == [torch.Size([1, 1024, 2, 128])]
    finally:
        destroy_distributed_env()
