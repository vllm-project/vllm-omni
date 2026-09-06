# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from types import SimpleNamespace

import cache_dit
import pytest
import torch

from vllm_omni.diffusion.cache.cachedit.model_specific import enable_cache_for_magi2
from vllm_omni.diffusion.data import DiffusionCacheConfig
from vllm_omni.diffusion.models.magi2 import mh_moe
from vllm_omni.diffusion.models.magi2.attention import VarlenHandler
from vllm_omni.diffusion.models.magi2.configuration_magi2 import (
    Magi2MHCConfig,
    Magi2MoEConfig,
    Magi2PreviewConfig,
)
from vllm_omni.diffusion.models.magi2.layers import MultiModalityRMSNorm
from vllm_omni.diffusion.models.magi2.mh_moe import Magi2MultiHeadMoE
from vllm_omni.diffusion.models.magi2.modeling_magi2 import (
    Magi2PreviewTransformer,
    Modality,
)
from vllm_omni.diffusion.models.magi2.preview_data_proxy import (
    Magi2DataProxy,
    Magi2PreviewDataProxyConfig,
    ModelInput,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


def _tiny_config(
    params_dtype: torch.dtype = torch.float32,
    *,
    num_layers: int = 1,
) -> Magi2PreviewConfig:
    layer_indices = tuple(range(num_layers))
    return Magi2PreviewConfig(
        num_layers=num_layers,
        hidden_size=16,
        head_dim=8,
        num_query_groups=2,
        video_in_channels=4,
        audio_in_channels=4,
        text_in_channels=4,
        intermediate_factor=2,
        multimodal_layers=layer_indices,
        params_dtype=params_dtype,
        mhc=Magi2MHCConfig(num_streams=2),
        moe=Magi2MoEConfig(
            num_heads=2,
            num_experts=4,
            top_k=2,
            expert_intermediate_size=8,
            shared_expert_intermediate_size=8,
            modality_shared_expert_intermediate_size=8,
            layers=layer_indices,
        ),
    )


def _initialize_tiny_model(model: Magi2PreviewTransformer, seed: int) -> None:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(
                torch.randn(
                    parameter.shape,
                    generator=generator,
                    dtype=parameter.dtype,
                )
                * 0.02
            )
        for module in model.modules():
            if isinstance(module, MultiModalityRMSNorm):
                module.weight.zero_()
            elif isinstance(module, Magi2MultiHeadMoE):
                module.router.expert_bias.zero_()
                module.router.expert_bias_ema.zero_()


def test_hsdp_policy_shards_individual_preview_layers():
    model = Magi2PreviewTransformer(_tiny_config(num_layers=2))
    condition = model._hsdp_shard_conditions[0]

    assert condition("block.layers.0", model.block.layers[0])
    assert condition("block.layers.1", model.block.layers[1])
    assert not condition("block", model.block)
    assert not condition("pre_adapter", model.pre_adapter)


def test_data_proxy_keeps_output_layout_request_scoped() -> None:
    proxy = Magi2DataProxy(Magi2PreviewDataProxyConfig(time_channel_dim=0))

    def make_input(width: int) -> ModelInput:
        return ModelInput(
            x_t=torch.arange(2 * width, dtype=torch.float32).reshape(1, 2, 1, 1, width),
            audio_x_t=torch.zeros(1, 1, 2),
            audio_feat_len=torch.tensor([1]),
            txt_feat=torch.zeros(1, 1, 2),
            txt_feat_len=torch.tensor([1]),
            t=torch.tensor([0.5]),
        )

    first = proxy.process_input(make_input(2))
    second = proxy.process_input(make_input(3))

    first_video, first_audio = proxy.process_output(first.token_sequence, first.output_layout)
    second_video, second_audio = proxy.process_output(second.token_sequence, second.output_layout)

    assert first_video.shape == (1, 2, 1, 1, 2)
    assert second_video.shape == (1, 2, 1, 1, 3)
    assert first_audio.shape == second_audio.shape == (1, 1, 2)


def test_tiny_native_preview_runs_through_nested_cachedit_adapter() -> None:
    model = Magi2PreviewTransformer(_tiny_config(num_layers=3))
    _initialize_tiny_model(model, seed=11)
    pipeline = type("Magi2TestPipeline", (), {"transformer": model})()
    result = enable_cache_for_magi2(
        pipeline,
        DiffusionCacheConfig(
            Fn_compute_blocks=1,
            Bn_compute_blocks=0,
            max_warmup_steps=1,
            residual_diff_threshold=1.0,
        ),
    )
    result.refresh(pipeline, 4, verbose=False)
    packed = torch.randn(6, 4)
    coordinates = torch.ones(6, 9)
    modalities = torch.tensor(
        [
            Modality.VIDEO,
            Modality.VIDEO,
            Modality.AUDIO,
            Modality.AUDIO,
            Modality.TEXT,
            Modality.TEXT,
        ]
    )
    cumulative = torch.tensor([0, 6], dtype=torch.int32)
    varlen = VarlenHandler(cumulative, cumulative, 6, 6)

    execution_counts = [0, 0, 0]
    hooks = [
        layer.register_forward_hook(
            lambda _module, _args, _output, layer_index=index: execution_counts.__setitem__(
                layer_index, execution_counts[layer_index] + 1
            )
        )
        for index, layer in enumerate(model.block.layers)
    ]
    try:
        with torch.no_grad():
            first = model(packed, coordinates, modalities, varlen)
            second = model(packed, coordinates, modalities, varlen)
        assert torch.isfinite(first).all()
        assert torch.isfinite(second).all()
        # The first layer is the configured Fn block and always executes. The
        # zero-residual repeated input hits the cache on the second call, so
        # the two middle layers are skipped.
        assert execution_counts == [2, 1, 1]
    finally:
        for hook in hooks:
            hook.remove()
        cache_dit.disable_cache(result.targets[0])

    assert not getattr(model.block, "_is_cached", False)


def test_mate_bf16_moe_route_adapter_matches_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The grouped adapter preserves head/expert routing and accumulation.

    Use a CPU reference stub for the MATE call so this test exercises the
    packing/scatter logic in ordinary CI as well as the real S5000 probe.
    """

    tokens, heads, experts, d_head, d_expert = 3, 2, 3, 8, 16
    x = torch.randn(tokens, heads, d_head, dtype=torch.bfloat16)
    weights = [
        torch.randn(heads * experts, d_head, d_expert, dtype=torch.bfloat16),
        torch.randn(heads * experts, d_head, d_expert, dtype=torch.bfloat16),
        torch.randn(heads * experts, d_expert, d_head, dtype=torch.bfloat16),
    ]
    # Flat experts 1 and 4 have no routes; token 0 is routed to two heads.
    gather_ids = torch.tensor([0, 1, 0, 2], dtype=torch.int64)
    probs = torch.tensor([0.2, 0.3, 0.4, 0.1], dtype=torch.float32)
    expert_offsets = torch.tensor([0, 1, 1, 2, 3, 3, 4], dtype=torch.int64)

    def fake_grouped(input_a, weight, token_counts, *, major_b_mode, backend):
        assert major_b_mode == "N"
        assert backend == "mubin"
        chunks = []
        cursor = 0
        for expert, count in enumerate(token_counts.tolist()):
            if count:
                chunks.append(input_a[cursor : cursor + count] @ weight[expert])
            cursor += count
        return torch.cat(chunks, dim=0)

    monkeypatch.setattr(mh_moe, "_mate_bf16_grouped_linear", fake_grouped)
    expected = mh_moe.torch_mh_moe_forward(
        x, gather_ids, probs, expert_offsets, weights[0], weights[1], weights[2]
    )
    actual = mh_moe.mate_bf16_mh_moe_forward(
        x, gather_ids, probs, expert_offsets, weights[0], weights[1], weights[2]
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_global_sort_routes_with_head_ids_preserves_stable_order() -> None:
    """The optional head metadata must be derived from the same stable sort."""

    heads, tokens, top_k, experts = 2, 3, 2, 4
    topk_probs = torch.tensor(
        [
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            [[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]],
        ],
        dtype=torch.float32,
    )
    topk_indices = torch.tensor(
        [
            [[2, 0], [1, 3], [0, 2]],
            [[1, 0], [3, 2], [2, 1]],
        ],
        dtype=torch.int64,
    )

    expected = mh_moe.global_sort_routes(topk_probs, topk_indices, experts)
    actual = mh_moe.global_sort_routes_with_head_ids(topk_probs, topk_indices, experts)

    for expected_value, actual_value in zip(expected, actual[:3]):
        torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)

    head_offset = torch.arange(heads).view(heads, 1, 1) * experts
    flattened_experts = (topk_indices + head_offset).reshape(-1)
    order = flattened_experts.argsort(stable=True)
    expected_head_ids = order // (tokens * top_k)
    torch.testing.assert_close(actual[3], expected_head_ids, rtol=0, atol=0)


def test_global_sort_routes_with_head_ids_and_counts_reuses_csr_histogram() -> None:
    """The MATE metadata helper exposes the exact int32 CSR histogram."""

    heads, tokens, top_k, experts = 2, 3, 2, 4
    topk_probs = torch.arange(heads * tokens * top_k, dtype=torch.float32).view(
        heads, tokens, top_k
    )
    topk_indices = torch.tensor(
        [
            [[2, 0], [1, 3], [0, 2]],
            [[1, 0], [3, 2], [2, 1]],
        ],
        dtype=torch.long,
    )
    expected = mh_moe.global_sort_routes_with_head_ids(
        topk_probs, topk_indices, experts
    )
    actual = mh_moe.global_sort_routes_with_head_ids_and_counts(
        topk_probs, topk_indices, experts
    )
    for expected_value, actual_value in zip(expected, actual[:4]):
        torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
    expected_counts = torch.bincount(
        (topk_indices + torch.arange(heads).view(heads, 1, 1) * experts).reshape(-1),
        minlength=heads * experts,
    ).to(torch.int32)
    torch.testing.assert_close(actual[4], expected_counts, rtol=0, atol=0)


@pytest.mark.parametrize(
    "topk_ids, num_experts, block_size",
    [
        (torch.tensor([], dtype=torch.int32), 8, 4),
        (torch.tensor([0, 0, 0, 1, 3, 3, 7], dtype=torch.int32), 8, 4),
        (torch.tensor([2, 0, 1, 2, 1, 0, 2], dtype=torch.int32), 4, 4),
    ],
)
def test_fixed_capacity_moe_align_preserves_published_prefix(
    topk_ids: torch.Tensor, num_experts: int, block_size: int
) -> None:
    """The no-sync MUSA metadata path matches the dynamic reference prefix."""

    sorted_ids, expert_ids, num_padded = (
        mh_moe._magi2_align_block_size_fixed_capacity(
            topk_ids, num_experts, block_size
        )
    )
    order = torch.argsort(topk_ids)
    sorted_experts = topk_ids[order]
    counts = torch.bincount(topk_ids, minlength=num_experts)
    padded_counts = ((counts + block_size - 1) // block_size) * block_size
    expected_total = int(padded_counts.sum().item())
    expected_sorted = torch.full(
        (expected_total,), topk_ids.numel(), dtype=torch.int32
    )
    starts = torch.cumsum(padded_counts, 0) - padded_counts
    ends = torch.cumsum(counts, 0)
    begins = ends - counts
    positions = torch.arange(topk_ids.numel(), dtype=torch.int64)
    experts = sorted_experts.to(torch.int64)
    destinations = starts[experts] + positions - begins[experts]
    expected_sorted[destinations] = order.to(torch.int32)
    expected_experts = torch.repeat_interleave(
        torch.arange(num_experts, dtype=torch.int32), padded_counts // block_size
    )

    torch.testing.assert_close(sorted_ids[:expected_total], expected_sorted)
    torch.testing.assert_close(expert_ids[: expected_experts.numel()], expected_experts)
    assert int(num_padded.item()) == expected_total
    assert torch.all(expert_ids[expected_experts.numel() :] == -1)


def test_moe_forward_uses_request_split_sizes_without_all_gather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The transformer-provided split vector avoids per-layer scalar syncs."""

    moe = object.__new__(mh_moe.Magi2MultiHeadMoE)
    moe.ep_group = SimpleNamespace(world_size=2, rank=0, replicated_sequence=False)
    moe.ep_pad_heads = 0
    moe.num_heads = 2
    moe.d_head = 2
    moe.has_real_moe_heads = True
    calls: dict[str, object] = {}

    def fake_dispatch(tensor, group, split_sizes):
        calls["dispatch"] = split_sizes
        return tensor

    def fake_undispatch(tensor, group, split_sizes):
        calls["undispatch"] = split_sizes
        return tensor

    monkeypatch.setattr(mh_moe, "ep_dispatch", fake_dispatch)
    monkeypatch.setattr(mh_moe, "ep_undispatch", fake_undispatch)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather",
        lambda *args, **kwargs: pytest.fail("request split sizes should bypass all_gather"),
    )
    monkeypatch.setattr(moe, "_local_forward", lambda tensor: tensor, raising=False)

    x = torch.randn(2, 4)
    actual = moe.forward(x, sequence_split_sizes=[2, 2])
    torch.testing.assert_close(actual, x)
    assert calls == {"dispatch": [2, 2], "undispatch": [2, 2]}


def test_global_sort_routes_builds_exact_csr_counts() -> None:
    """The MUSA scatter-count implementation must preserve every offset."""

    heads, tokens, top_k, experts = 2, 5, 3, 4
    topk_probs = torch.arange(heads * tokens * top_k, dtype=torch.float32).view(
        heads, tokens, top_k
    )
    topk_indices = torch.tensor(
        [
            [[0, 3, 1], [2, 0, 3], [1, 1, 0], [3, 2, 2], [0, 3, 1]],
            [[2, 2, 3], [1, 0, 0], [3, 1, 2], [0, 2, 3], [1, 3, 0]],
        ],
        dtype=torch.long,
    )
    _gather_ids, _sorted_probs, offsets = mh_moe.global_sort_routes(
        topk_probs, topk_indices, experts
    )
    head_offsets = torch.arange(heads, dtype=torch.long).view(heads, 1, 1) * experts
    flattened_experts = (topk_indices + head_offsets).reshape(-1)
    expected_counts = torch.bincount(flattened_experts, minlength=heads * experts)
    expected_offsets = torch.cat(
        (torch.zeros(1, dtype=torch.long), expected_counts.cumsum(0))
    )
    torch.testing.assert_close(offsets, expected_offsets, rtol=0, atol=0)


def test_mate_bf16_moe_route_adapter_accepts_sorted_head_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Supplying sorted head IDs avoids the per-route repeat_interleave path."""

    tokens, heads, experts, d_head, d_expert = 3, 2, 3, 8, 16
    x = torch.randn(tokens, heads, d_head, dtype=torch.bfloat16)
    weights = [
        torch.randn(heads * experts, d_head, d_expert, dtype=torch.bfloat16),
        torch.randn(heads * experts, d_head, d_expert, dtype=torch.bfloat16),
        torch.randn(heads * experts, d_expert, d_head, dtype=torch.bfloat16),
    ]
    gather_ids = torch.tensor([0, 1, 0, 2], dtype=torch.int64)
    probs = torch.tensor([0.2, 0.3, 0.4, 0.1], dtype=torch.float32)
    expert_offsets = torch.tensor([0, 1, 1, 2, 3, 3, 4], dtype=torch.int64)
    sorted_head_ids = torch.tensor([0, 0, 1, 1], dtype=torch.int64)

    def fake_grouped(input_a, weight, token_counts, *, major_b_mode, backend):
        assert major_b_mode == "N"
        assert backend == "mubin"
        chunks = []
        cursor = 0
        for expert, count in enumerate(token_counts.tolist()):
            if count:
                chunks.append(input_a[cursor : cursor + count] @ weight[expert])
            cursor += count
        return torch.cat(chunks, dim=0)

    monkeypatch.setattr(mh_moe, "_mate_bf16_grouped_linear", fake_grouped)
    expected = mh_moe.torch_mh_moe_forward(
        x, gather_ids, probs, expert_offsets, weights[0], weights[1], weights[2]
    )

    def fail_repeat_interleave(*args, **kwargs):
        raise AssertionError("sorted head metadata should bypass repeat_interleave")

    monkeypatch.setattr(torch, "repeat_interleave", fail_repeat_interleave)

    def fail_diff(*args, **kwargs):
        raise AssertionError("supplied counts should bypass diff")

    monkeypatch.setattr(torch, "diff", fail_diff)
    actual = mh_moe.mate_bf16_mh_moe_forward(
        x,
        gather_ids,
        probs,
        expert_offsets,
        weights[0],
        weights[1],
        weights[2],
        sorted_head_ids=sorted_head_ids,
        token_counts=torch.tensor([1, 0, 1, 1, 0, 1], dtype=torch.int32),
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_tiny_native_preview_matches_pinned_reference_golden() -> None:
    """Full-model golden from SandAI reference f68a0f9bbccb.

    The reference's unavailable compiler/FA3/MoE CUDA boundaries were replaced
    by their eager PyTorch equations when generating this tensor.  Keeping the
    resulting golden local makes this regression independent of that runtime.
    """

    model = Magi2PreviewTransformer(_tiny_config(torch.bfloat16))
    with torch.no_grad():
        trainable_parameters = (parameter for parameter in model.parameters() if parameter.requires_grad)
        for index, parameter in enumerate(trainable_parameters):
            values = (torch.arange(parameter.numel(), dtype=torch.float32) % 17 - 8) * 0.002 + (index % 5 - 2) * 0.0001
            parameter.copy_(values.reshape(parameter.shape).to(parameter.dtype))
        moe = model.block.layers[0].mlp.moe_mlp
        moe.router.expert_bias.zero_()
        moe.router.expert_bias_ema.zero_()

    packed = torch.tensor(
        [
            [0.1, -0.2, 0.3, -0.4],
            [0.5, 0.6, -0.7, 0.8],
            [-0.9, 1.0, -0.1, 0.2],
            [0.3, -0.4, 0.5, -0.6],
            [0.7, 0.8, 0.9, 1.0],
            [-0.2, 0.4, -0.6, 0.8],
            [0.9, -0.7, 0.5, -0.3],
        ]
    )
    coordinates = torch.tensor(
        [
            [0, 0, 0, 2, 2, 2, 2, 2, 2],
            [1, 1, 1, 2, 2, 2, 2, 2, 2],
            [0, 0, 0, 1, 2, 2, 1, 2, 2],
            [0, 1, 1, 1, 2, 2, 1, 2, 2],
            [0, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.float32,
    )
    modalities = torch.tensor(
        [
            Modality.VIDEO,
            Modality.VIDEO,
            Modality.AUDIO,
            Modality.AUDIO,
            Modality.TEXT,
            Modality.TEXT,
            Modality.TIME,
        ]
    )
    cumulative = torch.tensor([0, 4, 7], dtype=torch.int32)
    time_tokens = torch.tensor(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [1.0, 0.9, 0.8],
            [0.7, 0.6, 0.5],
            [0.4, 0.3, 0.2],
            [0.1, 0.0, -0.1],
        ]
    )
    expected = torch.tensor(
        [
            [-0.093289732933, 0.133922040462, 0.149612516165, 0.069642566144],
            [-0.118302434683, 0.129068359733, 0.156465008855, 0.084789708257],
            [-0.114637844265, 0.137497439981, 0.159646511078, 0.088356778026],
            [-0.104430131614, 0.146924808621, 0.152607530355, 0.092736266553],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    varlen = VarlenHandler(cumulative, cumulative, 4, 4)

    with torch.no_grad():
        actual = model(packed, coordinates, modalities, varlen, time_tokens)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_strict_loader_covers_all_keys_and_uses_ema_router_bias() -> None:
    source = Magi2PreviewTransformer(_tiny_config())
    _initialize_tiny_model(source, seed=11)
    source_moe = source.block.layers[0].mlp.moe_mlp
    source_moe.router.expert_bias.fill_(2.0)
    source_moe.router.expert_bias_ema.fill_(3.0)
    checkpoint = [(name, value.detach().clone()) for name, value in source.state_dict().items()]

    target = Magi2PreviewTransformer(_tiny_config())
    loaded = target.load_weights(checkpoint)

    assert loaded == set(target.state_dict())
    target_moe = target.block.layers[0].mlp.moe_mlp
    torch.testing.assert_close(
        target_moe.router.expert_bias,
        torch.full_like(target_moe.router.expert_bias, 3.0),
    )
    torch.testing.assert_close(
        target_moe.router.expert_bias_ema, torch.full_like(target_moe.router.expert_bias_ema, 3.0)
    )
