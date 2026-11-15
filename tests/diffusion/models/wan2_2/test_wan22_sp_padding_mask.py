# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import copy
from types import SimpleNamespace

import pytest
import torch
from vllm.config import set_current_vllm_config

from vllm_omni.diffusion.forward_context import ForwardContext
from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import _build_sp_padding_mask

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _device() -> torch.device:
    accelerator = torch.accelerator.current_accelerator()
    return torch.device(accelerator.type, 0) if accelerator is not None else torch.device("cpu")


def _parallel_config(sequence_parallel_size: int = 2):
    return SimpleNamespace(sequence_parallel_size=sequence_parallel_size, mask_sp_padding=True)


def test_sp_padding_mask_covers_full_padded_sequence() -> None:
    ctx = ForwardContext(sp_padding_size=1, sp_original_seq_len=7)
    mask = _build_sp_padding_mask(_parallel_config(), ctx, batch_size=2, device=torch.device("cpu"))

    expected = torch.tensor([[True] * 7 + [False]] * 2)
    assert mask is not None
    assert torch.equal(mask, expected)


@pytest.mark.parametrize(
    ("sequence_parallel_size", "padding", "original_len"),
    [
        (2, 0, 8),  # divisible sequence, e.g. the 512x512 warmup
        (2, 0, None),  # no SP auto-padding recorded
        (1, 1, 7),  # no sequence parallelism
    ],
)
def test_sp_padding_mask_none_without_padding(sequence_parallel_size, padding, original_len) -> None:
    ctx = ForwardContext(sp_padding_size=padding, sp_original_seq_len=original_len)
    mask = _build_sp_padding_mask(
        _parallel_config(sequence_parallel_size), ctx, batch_size=2, device=torch.device("cpu")
    )

    assert mask is None


def test_sp_padding_mask_matches_ulysses_key_length(monkeypatch: pytest.MonkeyPatch) -> None:
    # After the Ulysses all-to-all every rank attends over keys from all ranks,
    # so the mask must span the whole padded sequence, not one rank's shard.
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
    from vllm_omni.diffusion.attention.parallel import ulysses

    def fake_all_to_all(_group, tensor, _scatter_idx, _gather_idx, _use_sync):
        # Stand-in for 2 ranks: gather the sequence (4 -> 8 tokens) and scatter the heads (2 -> 1).
        return torch.cat([tensor, tensor], dim=1)[:, :, :1]

    monkeypatch.setattr(ulysses.SeqAllToAll4D, "apply", staticmethod(fake_all_to_all))
    strategy = ulysses.UlyssesParallelAttention(
        sp_group=SimpleNamespace(ulysses_group=object(), ulysses_world_size=2, ulysses_rank=1, ring_world_size=1),
        scatter_idx=2,
        gather_idx=1,
        use_sync=False,
    )
    # 7 tokens padded to 8, sharded into 4 per rank.
    query = torch.zeros(1, 4, 2, 4)
    mask = _build_sp_padding_mask(
        _parallel_config(), ForwardContext(sp_padding_size=1, sp_original_seq_len=7), 1, torch.device("cpu")
    )

    _, key_out, _, metadata_out, _ = strategy.pre_attention(
        query, torch.zeros_like(query), torch.zeros_like(query), AttentionMetadata(attn_mask=mask)
    )

    assert key_out.shape[1] == 8
    assert metadata_out.attn_mask.tolist() == [[True] * 7 + [False]]


@pytest.fixture
def diffusion_env(monkeypatch):
    from vllm.model_executor.layers.utils import default_unquantized_gemm

    from tests.helpers.runtime import assign_ephemeral_master_port
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.distributed.parallel_state import (
        destroy_distributed_env,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config

    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.dispatch_unquantized_gemm",
        lambda *_args, **_kwargs: default_unquantized_gemm,
    )
    od_config = OmniDiffusionConfig.from_kwargs(
        dtype=torch.float32, diffusion_attention_config={"default": "TORCH_SDPA"}, linear_backend="TORCH"
    )
    vllm_config = create_diffusion_vllm_config(_device(), od_config)
    assign_ephemeral_master_port()
    init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo")
    initialize_model_parallel(backend="gloo")
    try:
        with set_current_vllm_config(vllm_config), set_current_diffusion_config(od_config):
            yield od_config, vllm_config
    finally:
        destroy_model_parallel()
        destroy_distributed_env()


def test_forward_with_mask_sp_padding_and_no_padding(diffusion_env) -> None:
    # mask_sp_padding=True on a sequence that needs no SP padding (e.g. the
    # warmup) must run unmasked instead of failing on a missing mask.
    from vllm_omni.diffusion.forward_context import set_forward_context
    from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import WanTransformer3DModel

    od_config, vllm_config = diffusion_env
    model = WanTransformer3DModel(
        num_attention_heads=2,
        attention_head_dim=16,
        in_channels=4,
        out_channels=4,
        text_dim=16,
        freq_dim=16,
        ffn_dim=64,
        num_layers=1,
    )
    device = _device()
    model = model.to(device).eval()
    # A single process with an SP size of 2 in the config: no SP hooks run, so nothing is
    # padded, which is the unpadded case the forward has to handle.
    masked_config = copy.copy(od_config)
    masked_config.parallel_config = SimpleNamespace(sequence_parallel_size=2, mask_sp_padding=True)
    g = torch.Generator().manual_seed(0)

    with torch.inference_mode(), set_forward_context(vllm_config=vllm_config, omni_diffusion_config=masked_config):
        out = model(
            hidden_states=torch.randn(1, 4, 3, 8, 8, generator=g).to(device),
            timestep=torch.full((1,), 500.0, device=device),
            encoder_hidden_states=torch.randn(1, 12, 16, generator=g).to(device),
            return_dict=False,
        )[0]

    assert out.shape == (1, 4, 3, 8, 8)
