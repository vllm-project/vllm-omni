# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import math

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionMetadata,
    ConditioningBlockMask,
    PackedPaddingMetadata,
    VideoTokenLayout,
)
from vllm_omni.diffusion.attention.backends.sol_attn import SolAttnImpl
from vllm_omni.diffusion.attention.capabilities import ExecutionContext, SupportStatus
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.forward_context import bind_attention_schedule, get_forward_context, set_forward_context
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]

if not torch.cuda.is_available():
    pytest.skip("CUDA is unavailable", allow_module_level=True)
sol_attn = pytest.importorskip("sol_attn").sol_attn


@hardware_test(res={"cuda": ["H100", "B200", "RTX5090"]}, num_cards=1)
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("guard", ["none", "dense_blocks", "min_tokens"])
def test_real_sol_layer_tuning_and_short_sequence_guard(compiled, guard):
    used = 257
    settings: dict[str, str | int] = {"sink_conditioning": "off", "tau_profile": "0-4=1.5;-1=0.6"}
    if guard == "dense_blocks":
        settings["dense_blocks"] = "-1"
    elif guard == "min_tokens":
        settings["min_tokens"] = used + 1
    impl = SolAttnImpl(2, 128, 0.125, prefix="blocks.4.attention", layer_count=5, backend_kwargs={"sol_attn": settings})
    generator = torch.Generator(device="cuda").manual_seed(113)
    query, key, value = (
        torch.randn(1, used + 17, 2, 128, device="cuda", dtype=torch.bfloat16, generator=generator) for _ in range(3)
    )
    lengths = torch.tensor([0, used], dtype=torch.int32, device="cuda")
    metadata = AttentionMetadata(packed_padding=PackedPaddingMetadata(used, used, lengths, lengths))
    forward = torch.compile(impl.forward_cuda, fullgraph=True) if compiled else impl.forward_cuda
    with torch.inference_mode():
        actual = forward(query, key, value, metadata)
        if guard == "none":
            reference = sol_attn(
                query[:, :used].contiguous(),
                key[:, :used].contiguous(),
                value[:, :used].contiguous(),
                scale=0.125,
                tau=0.6,
                kv_splits=1,
            )
            torch.testing.assert_close(actual[:, :used], reference, rtol=0, atol=0)
        else:
            reference = torch.nn.functional.scaled_dot_product_attention(
                query[:, :used].transpose(1, 2).float(),
                key[:, :used].transpose(1, 2).float(),
                value[:, :used].transpose(1, 2).float(),
                scale=0.125,
            ).transpose(1, 2)
            torch.testing.assert_close(actual[:, :used].float(), reference, rtol=0.02, atol=0.02)
    assert torch.count_nonzero(actual[:, used:]) == 0
    torch.compiler.reset()


@hardware_test(res={"cuda": ["L4", "H100", "B200", "RTX5090"]}, num_cards=1)
@pytest.mark.parametrize("used", [63, 64, 257, 4097])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("sink_conditioning", ["off", "exact_kv_and_rows"])
def test_real_sol_kernel_padding_and_conditioning(used, compiled, sink_conditioning):
    generator = torch.Generator(device="cuda").manual_seed(17)
    query, key, value = (
        torch.randn(1, used + 17, 2, 128, device="cuda", dtype=torch.bfloat16, generator=generator) for _ in range(3)
    )
    lengths = torch.tensor([0, used], dtype=torch.int32, device="cuda")
    blocks = torch.zeros((used + 63) // 64, device="cuda", dtype=torch.bool)
    blocks[0] = blocks[-1] = True
    metadata = AttentionMetadata(
        packed_padding=PackedPaddingMetadata(used, used, lengths, lengths),
        conditioning_block_mask=ConditioningBlockMask(used, 64, blocks),
    )
    impl = SolAttnImpl(2, 128, 0.125, backend_kwargs={"sol_attn": {"tau": 1.5, "sink_conditioning": sink_conditioning}})
    forward = torch.compile(impl.forward_cuda, fullgraph=True) if compiled else impl.forward_cuda
    with torch.inference_mode():
        actual = forward(query, key, value, metadata)
        upstream = sol_attn(
            query[:, :used].contiguous(),
            key[:, :used].contiguous(),
            value[:, :used].contiguous(),
            scale=0.125,
            tau=1.5,
            kv_splits=1,
            exact_block_mask=blocks if sink_conditioning == "exact_kv_and_rows" else None,
        )
        reference = torch.nn.functional.scaled_dot_product_attention(
            query[:, :used].transpose(1, 2).float(),
            key[:, :used].transpose(1, 2).float(),
            value[:, :used].transpose(1, 2).float(),
            scale=0.125,
        ).transpose(1, 2)
    assert actual.shape == query.shape and torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[:, used:]) == 0
    torch.testing.assert_close(actual[:, :used], upstream, rtol=0, atol=0)
    if sink_conditioning == "exact_kv_and_rows":
        rows = blocks.repeat_interleave(64)[:used]
        torch.testing.assert_close(actual[:, :used][:, rows].float(), reference[:, rows], rtol=0.02, atol=0.02)
    elif used <= 64:
        torch.testing.assert_close(actual[:, :used].float(), reference, rtol=0.02, atol=0.02)
    torch.compiler.reset()


@hardware_test(res={"cuda": ["L4", "H100", "B200", "RTX5090"]}, num_cards=1)
def test_real_sol_production_attention_entry():
    config = OmniDiffusionConfig(
        diffusion_attention_config={
            "default": {"backend": "SOL_ATTN", "sol_attn": {"sink_conditioning": "off"}},
            "per_role": {"minimax_h3.token_refiner": {"backend": "TORCH_SDPA"}},
        }
    )
    with set_current_diffusion_config(config):
        layer = Attention(2, 128, False, 0.125, prefix="blocks.4.attention", role="self")
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


@hardware_test(res={"cuda": ["L4", "H100", "B200", "RTX5090"]}, num_cards=1)
def test_real_sol_compile_dynamic_sequence_without_explicit_length():
    impl = SolAttnImpl(2, 128, 0.125, backend_kwargs={"sol_attn": {"sink_conditioning": "off"}})
    forward = torch.compile(impl.forward_cuda, fullgraph=True, dynamic=True)
    metadata = AttentionMetadata()
    generator = torch.Generator(device="cuda").manual_seed(31)
    with torch.inference_mode():
        for length in (257, 513):
            query, key, value = (
                torch.randn(1, length, 2, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
                for _ in range(3)
            )
            actual = forward(query, key, value, metadata)
            expected = sol_attn(query, key, value, scale=0.125, tau=1.0, thresh_type="diag", kv_splits=1)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.compiler.reset()


@hardware_test(res={"cuda": ["L4", "H100", "B200", "RTX5090"]}, num_cards=1)
@pytest.mark.parametrize("sink_conditioning", ["off", "exact_kv_and_rows"])
@pytest.mark.parametrize(
    "used,compiled,tau",
    [(63, False, 1e6), (257, False, 1e6), (513, True, 1e6), (513, False, 0.0), (513, True, 1.0), (513, False, 1.5)],
)
@torch.inference_mode()
def test_real_sol_matches_hybrid_oracle(sink_conditioning, used, compiled, tau):
    from sol_attn.triton_ref.preprocess import prepare

    generator = torch.Generator(device="cuda").manual_seed(71)
    query, key, value = (
        torch.randn(1, used + 17, 2, 128, device="cuda", dtype=torch.bfloat16, generator=generator) for _ in range(3)
    )
    for tensor in (query, key, value):
        tensor[:, used:] = 100
    lengths = torch.tensor([0, used], device="cuda", dtype=torch.int32)
    metadata = AttentionMetadata(packed_padding=PackedPaddingMetadata(used, used, lengths, lengths))
    scale = 128**-0.5
    blocks = (used + 63) // 64
    protected = [False] * blocks
    if sink_conditioning == "exact_kv_and_rows":
        protected[0] = protected[-1] = True
        metadata.conditioning_block_mask = ConditioningBlockMask(
            used, 64, torch.tensor(protected, device=query.device, dtype=torch.bool)
        )
        metadata.video_layout = VideoTokenLayout(prefix_len=used - 1, latent_grid=(1, 1, 1))
    config = OmniDiffusionConfig(
        diffusion_attention_config={
            "default": {
                "backend": "SOL_ATTN",
                "sol_attn": {
                    "sink_conditioning": sink_conditioning,
                    "tau": tau,
                },
            }
        }
    )
    with set_current_diffusion_config(config):
        layer = Attention(2, 128, False, scale, prefix="blocks.4.attention", role="self")
    forward = torch.compile(layer, fullgraph=True) if compiled else layer
    actual = forward(query, key, value, metadata)
    key_centroids, value_sums, threshold = prepare(
        query[:, :used].contiguous(),
        key[:, :used].contiguous(),
        value[:, :used].contiguous(),
        tau=tau,
        scale=scale,
        thresh_type="diag",
        tokens=used,
    )
    blocks = (used + 63) // 64
    assert torch.isfinite(threshold).all()
    expected = torch.empty_like(actual[:, :used], dtype=torch.float32)
    for query_block in range(blocks):
        query_start, query_stop = query_block * 64, min((query_block + 1) * 64, used)
        query_mean = query[:, query_start:query_stop].float().mean(1)
        hybrid_keys, hybrid_values = [], []
        for key_block in range(blocks):
            start, stop = key_block * 64, min((key_block + 1) * 64, used)
            score = (query_mean * key_centroids[:, key_block].float()).sum(-1) * (scale * math.log2(math.e))
            exact = score > threshold[:, query_block]
            if abs(query_block - key_block) <= 1 or protected[query_block] or protected[key_block]:
                exact = torch.ones_like(exact)
            exact = exact[:, None, :, None]
            hybrid_keys.append(
                torch.where(exact, key[:, start:stop].float(), key_centroids[:, key_block : key_block + 1].float())
            )
            hybrid_values.append(
                torch.where(
                    exact,
                    value[:, start:stop].float(),
                    value_sums[:, key_block : key_block + 1].float() / (stop - start),
                )
            )
        expected[:, query_start:query_stop] = torch.nn.functional.scaled_dot_product_attention(
            query[:, query_start:query_stop].float().transpose(1, 2),
            torch.cat(hybrid_keys, dim=1).transpose(1, 2),
            torch.cat(hybrid_values, dim=1).transpose(1, 2),
            scale=scale,
        ).transpose(1, 2)
    torch.testing.assert_close(actual[:, :used].float(), expected, atol=0.02, rtol=0.02)
    assert actual[:, used:].count_nonzero() == 0
    assert torch.isfinite(actual).all()
    torch.compiler.reset()


@hardware_test(res={"cuda": ["H100", "B200", "RTX5090"]}, num_cards=1)
@torch.inference_mode()
def test_real_sol_defaults_match_explicit_text_audio_profile():
    generator = torch.Generator(device="cuda").manual_seed(73)
    query, key, value = (
        torch.randn(1, 257, 2, 128, device="cuda", dtype=torch.bfloat16, generator=generator) for _ in range(3)
    )
    metadata = AttentionMetadata(
        conditioning_block_mask=ConditioningBlockMask(
            257, 64, torch.tensor([True, False, False, False, True], device="cuda", dtype=torch.bool)
        )
    )
    default = SolAttnImpl(2, 128, 128**-0.5)
    explicit = SolAttnImpl(
        2,
        128,
        128**-0.5,
        backend_kwargs={"sol_attn": {"sink_conditioning": "exact_kv_and_rows"}},
    )
    actual = default.forward_cuda(query, key, value, metadata)
    expected = explicit.forward_cuda(query, key, value, metadata)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def _ulysses_worker(rank, init_method, sink_conditioning, world_size=2):
    from vllm_omni.diffusion.distributed.parallel_state import (
        destroy_distributed_env,
        init_distributed_environment,
        initialize_model_parallel,
    )

    device = torch.device("cuda", rank)
    current_omni_platform.set_device(device)
    init_distributed_environment(world_size=world_size, rank=rank, local_rank=rank, distributed_init_method=init_method)
    initialize_model_parallel(sequence_parallel_size=world_size, ulysses_degree=world_size)
    try:
        config = OmniDiffusionConfig(
            parallel_config=DiffusionParallelConfig(ulysses_degree=world_size),
            diffusion_attention_config={
                "default": {
                    "backend": "SOL_ATTN",
                    "sol_attn": {
                        "sink_conditioning": sink_conditioning,
                    },
                }
            },
        )
        with set_current_diffusion_config(config), set_forward_context(omni_diffusion_config=config):
            layer = Attention(4, 128, False, 128**-0.5, prefix="blocks.4.attention", role="self")
            assert layer.parallel_strategy.name == "ulysses"
            generator = torch.Generator(device=device).manual_seed(23)
            query, key, value = (
                torch.randn(1, 137 * world_size, 4, 128, device=device, dtype=torch.bfloat16, generator=generator)
                for _ in range(3)
            )
            lengths = torch.tensor([0, 257], dtype=torch.int32, device=device)
            metadata = AttentionMetadata(
                packed_padding=PackedPaddingMetadata(257, 257, lengths, lengths),
                video_layout=VideoTokenLayout(prefix_len=5, latent_grid=(1, 12, 21)),
                conditioning_block_mask=ConditioningBlockMask(
                    257, 64, torch.tensor([True, False, False, False, True], device=device, dtype=torch.bool)
                ),
            )
            with torch.inference_mode():
                expected = layer.attention.forward_cuda(query, key, value, metadata).chunk(world_size, dim=1)[rank]
                local_query, local_key, local_value = (
                    tensor.chunk(world_size, dim=1)[rank].contiguous() for tensor in (query, key, value)
                )
                actual = layer(local_query, local_key, local_value, metadata)
            torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
    finally:
        destroy_distributed_env()


@hardware_test(res={"cuda": ["L4", "H100", "B200", "RTX5090"]}, num_cards=2)
@pytest.mark.parametrize("sink_conditioning", ["off", "exact_kv_and_rows"])
def test_real_sol_ulysses_matches_local(tmp_path, sink_conditioning):
    if torch.accelerator.device_count() < 2:
        pytest.skip("Requires two CUDA GPUs")
    torch.multiprocessing.spawn(
        _ulysses_worker, args=(f"file://{tmp_path / 'rendezvous'}", sink_conditioning), nprocs=2
    )


@hardware_test(res={"cuda": ["H100", "B200", "RTX5090"]}, num_cards=4)
def test_real_text_audio_sol_sp4_matches_local(tmp_path):
    if torch.accelerator.device_count() < 4:
        pytest.skip("Requires four CUDA GPUs")
    torch.multiprocessing.spawn(
        _ulysses_worker, args=(f"file://{tmp_path / 'rendezvous'}", "exact_kv_and_rows", 4), nprocs=4
    )


@hardware_test(res={"cuda": ["H100", "B200", "RTX5090"]}, num_cards=1)
@pytest.mark.parametrize("selection", ["shorthand", "json", "category"])
def test_real_sol_h3_simple_selection_and_forward(tmp_path, selection):
    from dataclasses import replace

    from vllm_omni.diffusion.distributed.parallel_state import (
        destroy_distributed_env,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_blocks import MiniMaxH3DiTArchConfig, MiniMaxH3DiTBlock
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3TokenRefinerBlock

    if selection == "shorthand":
        config = OmniDiffusionConfig.from_kwargs(diffusion_attention_backend="SOL_ATTN")
    elif selection == "json":
        config = OmniDiffusionConfig.from_kwargs(diffusion_attention_config={"default": {"backend": "SOL_ATTN"}})
    else:
        config = OmniDiffusionConfig.from_kwargs(diffusion_attention_config={"per_role": {"self": "SOL_ATTN"}})
    attention_config = config.diffusion_attention_config
    explicit = OmniDiffusionConfig(
        diffusion_attention_config=replace(
            attention_config, per_role={**attention_config.per_role, "minimax_h3.token_refiner": "TORCH_SDPA"}
        )
    )
    arch = MiniMaxH3DiTArchConfig(
        hidden_size=256, num_attention_heads=2, ffn_hidden_size=512, time_embed_dim=32, adaln_out_features=18 * 256
    )
    init_distributed_environment(
        world_size=1, rank=0, local_rank=0, distributed_init_method=f"file://{tmp_path / 'h3-rendezvous'}"
    )
    initialize_model_parallel()
    try:
        models = []
        for settings in (config, explicit):
            with set_current_diffusion_config(settings), torch.device("cuda"):
                main = MiniMaxH3DiTBlock(arch, None, prefix="blocks.0", layer_index=0)
                refiner = MiniMaxH3TokenRefinerBlock(arch, None, prefix="token_refiner.blocks.0")
            assert main.attn.attention.role == "self"
            assert isinstance(main.attn.attention.attention, SolAttnImpl)
            assert refiner.attn.attention.attn_backend.get_name() == "SDPA"
            assert refiner.attn.attention.skip_sequence_parallel
            assert main.attn.attention.attention.config.sink_conditioning == "exact_kv_and_rows"
            assert main.attn.attention.attention.requires_conditioning_mask
            models.append((main.eval(), refiner.eval()))
        with torch.inference_mode():
            generator = torch.Generator(device="cuda").manual_seed(79)
            for model in models[0]:
                for parameter in model.parameters():
                    parameter.normal_(std=0.02, generator=generator)
            for automatic, manual in zip(models[0], models[1]):
                manual.load_state_dict(automatic.state_dict())
            inputs = torch.randn(257, 256, device="cuda", dtype=torch.bfloat16, generator=generator)
            lengths = torch.tensor([0, 257], device="cuda", dtype=torch.int32)
            mask = ConditioningBlockMask(257, 64, torch.tensor([True, False, False, False, False], device="cuda"))
            outputs = []
            for settings, (main, refiner) in zip((config, explicit), models):
                with set_forward_context(omni_diffusion_config=settings):
                    main_output = main.attn(
                        inputs, rope_table=None, cu_seqlens=lengths, max_seqlen=257, conditioning_block_mask=mask
                    )
                    refiner_output = refiner(inputs, cu_seqlens=lengths, max_seqlen=257)
                    with pytest.raises(ValueError, match="conditioning_block_mask"):
                        main.attn(inputs, rope_table=None, cu_seqlens=lengths, max_seqlen=257)
                assert torch.isfinite(main_output).all() and torch.isfinite(refiner_output).all()
                outputs.append((main_output, refiner_output))
            for automatic, manual in zip(outputs[0], outputs[1]):
                torch.testing.assert_close(automatic, manual, atol=0, rtol=0)
        assert "minimax_h3.token_refiner" not in config.diffusion_attention_config.per_role
    finally:
        destroy_distributed_env()


@hardware_test(res={"cuda": ["L4", "H100", "B200", "RTX5090"]}, num_cards=1)
def test_real_sol_wan_transformer_forward(tmp_path, monkeypatch):
    from vllm_omni.diffusion.attention.ops import sol_attn as sol_ops
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
        dense_backend = "CUDNN_ATTN" if current_omni_platform.get_device_capability().major == 12 else "TORCH_SDPA"
        configs = {}
        for backend in (dense_backend, "SOL_ATTN"):
            configs[backend] = OmniDiffusionConfig(
                diffusion_attention_config={
                    "default": {"backend": "TORCH_SDPA"},
                    "per_role": {"self": {"backend": dense_backend}},
                },
                diffusion_attention_schedule={
                    "profiles": {
                        "sparse": {
                            "default": {"backend": "TORCH_SDPA"},
                            "per_role": {
                                "self": {
                                    "backend": "SOL_ATTN",
                                    "sol_attn": {
                                        "dense_blocks": "0,1",
                                    },
                                }
                            },
                        }
                    },
                    "default": [{"start": 10, "end": None, "profile": "sparse"}],
                }
                if backend == "SOL_ATTN"
                else None,
            )
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
            for name, parameter in models[dense_backend].named_parameters():
                offset = 1.0 if "norm" in name and parameter.ndim == 1 else 0.0
                parameter.copy_(torch.randn(parameter.shape, device="cuda", generator=generator) * 0.02 + offset)
            models["SOL_ATTN"].load_state_dict(models[dense_backend].state_dict())
        latents = torch.randn(1, 4, 4, 32, 32, device="cuda", dtype=torch.bfloat16, generator=generator)
        text = torch.randn(1, 16, 32, device="cuda", dtype=torch.bfloat16, generator=generator)
        timestep = torch.tensor([500.0], device="cuda", dtype=torch.bfloat16)
        calls = []
        original_op = sol_ops.sol_attention

        def record_sparse(*args):
            calls.append(args[0].shape)
            return original_op(*args)

        monkeypatch.setattr(sol_ops, "sol_attention", record_sparse)
        outputs = {}
        for backend, model in models.items():
            assert all(block.attn2.attn.attn_backend.get_name() == "SDPA" for block in model.blocks)
            for step in (0, 10):
                with (
                    torch.inference_mode(),
                    set_forward_context(omni_diffusion_config=configs[backend], denoise_step_idx=step),
                    bind_attention_schedule(
                        configs[backend].diffusion_attention_schedule.default if backend == "SOL_ATTN" else None
                    ),
                ):
                    get_forward_context().total_denoise_steps = 11
                    output = model(latents, timestep, text, return_dict=False)[0]
                assert output.shape == latents.shape and torch.isfinite(output).all()
                outputs[backend, step] = output
        torch.testing.assert_close(outputs["SOL_ATTN", 0], outputs[dense_backend, 0], atol=1e-3, rtol=1e-3)
        assert calls == [torch.Size([1, 1024, 2, 128])]
    finally:
        destroy_distributed_env()
