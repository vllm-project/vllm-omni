# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.helpers.runtime import get_distributed_init_method
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.attention.parallel.ulysses import UlyssesParallelAttention
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.forward_context import get_forward_context, set_forward_context
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.core_model
@pytest.mark.cpu
def test_uaa_gqa_head_padding_preserves_the_query_to_kv_ratio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Q must be padded to a multiple of the *KV* padding, not of world_size.

    Rounding each up independently is the obvious implementation and is wrong:
    28Q/7KV at SP2 would become 28/8, i.e. 14/4 per rank, whose 3.5 ratio is not
    a valid GQA shape. Deriving Q from the padded KV count gives 32/8 -> 16/4.
    """
    from vllm_omni.diffusion.attention.parallel import ulysses

    class FakeGroup:
        ulysses_group = object()
        ulysses_world_size = 2
        ulysses_rank = 0
        ring_world_size = 1

    observed = []

    def fake_all_to_all(pg, tensor, **kwargs):
        observed.append(kwargs["padded_head_cnt"])
        return tensor, tensor.shape[2]

    monkeypatch.setattr(ulysses, "_ulysses_all_to_all_any_qkv", fake_all_to_all)
    monkeypatch.setattr(ulysses, "get_ulysses_mode", lambda **kwargs: "advanced_uaa")
    monkeypatch.setattr(ulysses, "_all_gather_int", lambda *args, **kwargs: [3, 3])
    strategy = ulysses.UlyssesParallelAttention(
        FakeGroup(),
        scatter_idx=2,
        gather_idx=1,
        use_sync=False,
    )

    with set_forward_context():
        strategy.pre_attention(
            torch.zeros(1, 3, 28, 4),
            torch.zeros(1, 3, 7, 4),
            torch.zeros(1, 3, 7, 4),
            None,
        )

    q_padded, k_padded, v_padded = observed
    assert (q_padded, k_padded, v_padded) == (32, 8, 8)
    # The per-rank shape must still be a valid GQA shape.
    assert (q_padded // 2) % (k_padded // 2) == 0


@pytest.mark.core_model
@pytest.mark.cpu
def test_uaa_rejects_head_counts_that_are_not_a_gqa_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_omni.diffusion.attention.parallel import ulysses

    class FakeGroup:
        ulysses_group = object()
        ulysses_world_size = 2
        ulysses_rank = 0
        ring_world_size = 1

    monkeypatch.setattr(ulysses, "get_ulysses_mode", lambda **kwargs: "advanced_uaa")
    monkeypatch.setattr(ulysses, "_all_gather_int", lambda *args, **kwargs: [3, 3])
    strategy = ulysses.UlyssesParallelAttention(
        FakeGroup(),
        scatter_idx=2,
        gather_idx=1,
        use_sync=False,
    )

    with set_forward_context(), pytest.raises(ValueError, match="multiple of KV heads"):
        strategy.pre_attention(
            torch.zeros(1, 3, 10, 4),
            torch.zeros(1, 3, 4, 4),
            torch.zeros(1, 3, 4, 4),
            None,
        )


@pytest.mark.core_model
@pytest.mark.cpu
@pytest.mark.parametrize(
    "q_heads, kv_heads, world_size, expected_warn",
    [
        # 28Q/7KV @ U=2 -> pad to 32Q, 32/28 = 1.14x, below threshold, no warn.
        (28, 7, 2, False),
        # 32Q/1KV @ U=8 -> pad KV to 8 -> pad Q to 256, 256/32 = 8x, warn.
        (32, 1, 8, True),
        # 16Q/2KV @ U=4 -> pad KV to 4 -> pad Q to 32, 32/16 = 2x, warn.
        (16, 2, 4, True),
    ],
)
def test_uaa_padding_ratio_warning_fires_when_blowup_is_large(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    q_heads: int,
    kv_heads: int,
    world_size: int,
    expected_warn: bool,
) -> None:
    """advanced_uaa must warn once when GQA padding materially inflates Q heads.

    MQA-style shapes at high ulysses_degree pad Q by U x, which grows attention
    FLOPs and temporary VRAM by the same factor. We do not fall back (there is
    no cheap alternative -- Ulysses needs KV divisible by U for the head-dim
    split), but we surface a one-shot warning so the user can pick a
    ulysses_degree that divides KV_heads if the overhead is unacceptable.
    """
    from vllm_omni.diffusion.attention.parallel import ulysses

    # Reset the warn-once dedup set so parametrized cases do not shadow each other.
    ulysses._uaa_pad_ratio_warned.clear()

    class FakeGroup:
        ulysses_group = object()
        ulysses_world_size = world_size
        ulysses_rank = 0
        ring_world_size = 1

    def fake_all_to_all(pg, tensor, **kwargs):
        return tensor, tensor.shape[2]

    monkeypatch.setattr(ulysses, "_ulysses_all_to_all_any_qkv", fake_all_to_all)
    monkeypatch.setattr(ulysses, "get_ulysses_mode", lambda **kwargs: "advanced_uaa")
    monkeypatch.setattr(ulysses, "_all_gather_int", lambda *a, **kw: [3] * world_size)
    strategy = ulysses.UlyssesParallelAttention(
        FakeGroup(),
        scatter_idx=2,
        gather_idx=1,
        use_sync=False,
    )

    caplog.set_level("WARNING", logger="vllm_omni.diffusion.attention.parallel.ulysses")
    with set_forward_context():
        strategy.pre_attention(
            torch.zeros(1, 3, q_heads, 4),
            torch.zeros(1, 3, kv_heads, 4),
            torch.zeros(1, 3, kv_heads, 4),
            None,
        )

    warn_msgs = [r.message for r in caplog.records if "GQA padding inflates" in r.message]
    if expected_warn:
        assert len(warn_msgs) == 1, warn_msgs
        # Second call with the same shape must not re-warn (warn-once dedup).
        with set_forward_context():
            strategy.pre_attention(
                torch.zeros(1, 3, q_heads, 4),
                torch.zeros(1, 3, kv_heads, 4),
                torch.zeros(1, 3, kv_heads, 4),
                None,
            )
        warn_msgs = [r.message for r in caplog.records if "GQA padding inflates" in r.message]
        assert len(warn_msgs) == 1, "warn-once dedup should suppress the second call"
    else:
        assert warn_msgs == [], f"unexpected warning for {q_heads}Q/{kv_heads}KV @ U={world_size}: {warn_msgs}"


@pytest.mark.core_model
@pytest.mark.cpu
def test_uaa_joint_gqa_head_padding_preserves_the_query_to_kv_ratio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Joint Q/K/V must also preserve the GQA ratio through Ulysses padding.

    MMDiT-style models (SD3, FLUX, Qwen-Image, Hunyuan-Video, ...) pass a
    separate text stream via ``AttentionMetadata.joint_*``. Before the fix the
    joint block padded K/V by Q's amount and integer-divided K/V by world_size,
    which for 28Q/7KV at SP=2 produces mismatched joint vs main head counts and
    the downstream ring-attention concat blows up. Post-fix the joint block
    mirrors the main-path rule: pad KV to a world_size multiple, derive Q as
    ``padded_kv * (Q // KV)``.
    """
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
    from vllm_omni.diffusion.attention.parallel import ulysses

    class FakeGroup:
        ulysses_group = object()
        ulysses_world_size = 2
        ulysses_rank = 0
        ring_world_size = 1

    def fake_all_to_all(pg, tensor, **kwargs):
        # Simulate the head-dim sharding so the post-a2a joint concat has
        # matching per-rank head counts on Q and joint_Q.
        heads_per_rank = kwargs["padded_head_cnt"] // 2
        return tensor[:, :, :heads_per_rank, :].contiguous(), tensor.shape[2]

    monkeypatch.setattr(ulysses, "_ulysses_all_to_all_any_qkv", fake_all_to_all)
    monkeypatch.setattr(ulysses, "get_ulysses_mode", lambda **kwargs: "advanced_uaa")
    monkeypatch.setattr(ulysses, "_all_gather_int", lambda *args, **kwargs: [3, 3])
    strategy = ulysses.UlyssesParallelAttention(
        FakeGroup(),
        scatter_idx=2,
        gather_idx=1,
        use_sync=False,
    )

    attn_metadata = AttentionMetadata(
        joint_query=torch.zeros(1, 5, 28, 4),
        joint_key=torch.zeros(1, 5, 7, 4),
        joint_value=torch.zeros(1, 5, 7, 4),
    )

    with set_forward_context():
        strategy.pre_attention(
            torch.zeros(1, 3, 28, 4),
            torch.zeros(1, 3, 7, 4),
            torch.zeros(1, 3, 7, 4),
            attn_metadata,
        )

    # 28Q/7KV at SP=2 -> pad to 32/8 -> per-rank slice is 16Q/4KV.
    assert attn_metadata.joint_key.shape[-2] == 4
    assert attn_metadata.joint_value.shape[-2] == 4
    # Per-rank joint shape must remain a valid GQA shape (Q multiple of KV).
    per_rank_joint_q = 32 // 2
    per_rank_joint_kv = attn_metadata.joint_key.shape[-2]
    assert per_rank_joint_q % per_rank_joint_kv == 0


@pytest.mark.core_model
@pytest.mark.cpu
def test_uaa_rejects_joint_head_counts_that_are_not_a_gqa_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
    from vllm_omni.diffusion.attention.parallel import ulysses

    class FakeGroup:
        ulysses_group = object()
        ulysses_world_size = 2
        ulysses_rank = 0
        ring_world_size = 1

    monkeypatch.setattr(ulysses, "get_ulysses_mode", lambda **kwargs: "advanced_uaa")
    monkeypatch.setattr(ulysses, "_all_gather_int", lambda *args, **kwargs: [3, 3])
    strategy = ulysses.UlyssesParallelAttention(
        FakeGroup(),
        scatter_idx=2,
        gather_idx=1,
        use_sync=False,
    )

    attn_metadata = AttentionMetadata(
        joint_query=torch.zeros(1, 5, 10, 4),
        joint_key=torch.zeros(1, 5, 4, 4),
        joint_value=torch.zeros(1, 5, 4, 4),
    )

    with set_forward_context(), pytest.raises(ValueError, match="multiple of joint KV heads"):
        strategy.pre_attention(
            torch.zeros(1, 3, 8, 4),
            torch.zeros(1, 3, 4, 4),
            torch.zeros(1, 3, 4, 4),
            attn_metadata,
        )


@pytest.mark.core_model
@pytest.mark.cpu
def test_advanced_uaa_hybrid_rejects_non_gqa_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hybrid Ulysses+Ring still rejects shapes that are not a GQA shape.

    Valid GQA shapes are safe under advanced_uaa head padding (the derived-Q
    rule keeps real Q heads paired with real K/V heads after the split, see
    pre_attention), but a query count that is not a multiple of the K/V count
    cannot be padded while preserving the ratio, so it must be rejected.
    """
    from vllm_omni.diffusion.attention.parallel import ulysses

    sp_group = SimpleNamespace(
        ulysses_group=None,
        ulysses_world_size=2,
        ulysses_rank=0,
        ring_world_size=2,
        ring_group=object(),
    )
    monkeypatch.setattr(ulysses, "get_ulysses_mode", lambda **kwargs: "advanced_uaa")
    monkeypatch.setattr(ulysses, "_all_gather_int", lambda *args, **kwargs: [3, 3])
    strategy = UlyssesParallelAttention(
        sp_group=sp_group,
        scatter_idx=2,
        gather_idx=1,
        use_sync=False,
    )
    query = torch.zeros(1, 2, 10, 4)
    key = torch.zeros(1, 2, 4, 4)
    value = torch.zeros_like(key)

    with pytest.raises(ValueError, match="multiple of KV heads"):
        strategy.pre_attention(query, key, value, None)


def _run_attention_case(
    local_rank: int,
    world_size: int,
    init_method: str,
    input_file: str,
    output_file: str,
    num_heads: int,
    head_size: int,
    ulysses_degree: int,
    ulysses_mode: str,
    ring_degree: int = 1,
    split_sizes: list[int] | None = None,
    sdp_kernel_mode: str = "math",
    num_kv_heads: int | None = None,
    has_joint: bool = False,
    joint_strategy: str = "front",
) -> None:
    device = torch.device(f"{current_omni_platform.device_type}:{local_rank}")
    current_omni_platform.set_device(device)

    os.environ["DIFFUSION_ATTENTION_BACKEND"] = "TORCH_SDPA"

    init_distributed_environment(world_size=world_size, rank=local_rank, distributed_init_method=init_method)
    initialize_model_parallel(
        data_parallel_size=1,
        cfg_parallel_size=1,
        sequence_parallel_size=world_size,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )

    parallel_config = DiffusionParallelConfig(
        pipeline_parallel_size=1,
        data_parallel_size=1,
        tensor_parallel_size=1,
        sequence_parallel_size=world_size,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        cfg_parallel_size=1,
        ulysses_mode=ulysses_mode,
    )
    od_config = OmniDiffusionConfig(model="test", dtype=torch.float32, parallel_config=parallel_config)

    with set_forward_context(omni_diffusion_config=od_config), set_current_diffusion_config(od_config):
        attn = Attention(
            num_heads=num_heads,
            head_size=head_size,
            causal=False,
            softmax_scale=1.0 / (head_size**0.5),
            num_kv_heads=num_kv_heads,
        ).to(device=device, dtype=torch.float32)

        with np.load(input_file, allow_pickle=False) as payload:
            q_full = torch.from_numpy(payload["q"]).to(device=device)
            k_full = torch.from_numpy(payload["k"]).to(device=device)
            v_full = torch.from_numpy(payload["v"]).to(device=device)
            if has_joint:
                joint_q = torch.from_numpy(payload["joint_q"]).to(device=device)
                joint_k = torch.from_numpy(payload["joint_k"]).to(device=device)
                joint_v = torch.from_numpy(payload["joint_v"]).to(device=device)

        attn_metadata = None
        if world_size == 1:
            if has_joint:
                # Baseline: SP is inactive so pre_attention won't concat joint for us.
                # Build the full sequence directly and skip the joint metadata.
                if joint_strategy == "front":
                    q = torch.cat([joint_q, q_full], dim=1)
                    k = torch.cat([joint_k, k_full], dim=1)
                    v = torch.cat([joint_v, v_full], dim=1)
                else:
                    q = torch.cat([q_full, joint_q], dim=1)
                    k = torch.cat([k_full, joint_k], dim=1)
                    v = torch.cat([v_full, joint_v], dim=1)
            else:
                q, k, v = q_full, k_full, v_full
        else:
            if split_sizes is None:
                # NOTE: torch.chunk may return fewer than `world_size` chunks for some
                # uneven lengths (e.g. seq_len=9, world_size=4 -> 3 chunks of len 3).
                # We need an exact `world_size`-way split to simulate uneven SP shards.
                q = torch.tensor_split(q_full, world_size, dim=1)[local_rank].contiguous()
                k = torch.tensor_split(k_full, world_size, dim=1)[local_rank].contiguous()
                v = torch.tensor_split(v_full, world_size, dim=1)[local_rank].contiguous()
            else:
                if len(split_sizes) != world_size:
                    raise ValueError(f"split_sizes length ({len(split_sizes)}) must equal world_size ({world_size}).")
                if sum(int(x) for x in split_sizes) != q_full.shape[1]:
                    raise ValueError(
                        "split_sizes must sum to full seq_len "
                        f"(got sum={sum(int(x) for x in split_sizes)}, seq_len={q_full.shape[1]})."
                    )
                q = torch.split(q_full, split_sizes, dim=1)[local_rank].contiguous()
                k = torch.split(k_full, split_sizes, dim=1)[local_rank].contiguous()
                v = torch.split(v_full, split_sizes, dim=1)[local_rank].contiguous()
            # The Attention layer only enables SP communication when ForwardContext.sp_active is True.
            # In production this is managed by SequenceParallelSplitHook/GatherHook, but here we
            # shard manually for testing.
            get_forward_context()._sp_shard_depth = 1

            if has_joint:
                from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata

                attn_metadata = AttentionMetadata(
                    joint_query=joint_q,
                    joint_key=joint_k,
                    joint_value=joint_v,
                    joint_strategy=joint_strategy,
                )

        if sdp_kernel_mode == "math":
            sdp_ctx = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
        elif sdp_kernel_mode == "mem_efficient":
            sdp_ctx = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=True, enable_math=False)
        else:
            raise ValueError(f"Invalid sdp_kernel_mode: {sdp_kernel_mode!r}")

        with sdp_ctx:
            out_local = attn(q, k, v, attn_metadata=attn_metadata).contiguous()

        if world_size == 1:
            out_full = out_local
        else:
            if has_joint:
                joint_len = int(joint_q.shape[1])
                if joint_strategy == "front":
                    joint_out_local = out_local[:, :joint_len]
                    img_out_local = out_local[:, joint_len:]
                else:
                    img_out_local = out_local[:, : out_local.shape[1] - joint_len]
                    joint_out_local = out_local[:, -joint_len:]
            else:
                joint_out_local = None
                img_out_local = out_local

            local_len = torch.tensor([img_out_local.shape[1]], device=device, dtype=torch.int64)
            gathered_lens = [torch.empty_like(local_len) for _ in range(world_size)]
            torch.distributed.all_gather(gathered_lens, local_len)
            lens = [int(t.item()) for t in gathered_lens]
            max_len = max(lens)

            if img_out_local.shape[1] < max_len:
                pad = max_len - img_out_local.shape[1]
                img_out_local = torch.nn.functional.pad(img_out_local, (0, 0, 0, 0, 0, pad)).contiguous()
            else:
                img_out_local = img_out_local.contiguous()

            gathered = [torch.empty_like(img_out_local) for _ in range(world_size)]
            torch.distributed.all_gather(gathered, img_out_local)
            if local_rank == 0:
                img_full = torch.cat([t[:, : lens[i]].contiguous() for i, t in enumerate(gathered)], dim=1)
                if has_joint:
                    out_full = (
                        torch.cat([joint_out_local, img_full], dim=1)
                        if joint_strategy == "front"
                        else torch.cat([img_full, joint_out_local], dim=1)
                    )
                else:
                    out_full = img_full
            else:
                out_full = None

        if local_rank == 0:
            np.save(output_file, out_full.detach().cpu().numpy())

    destroy_distributed_env()


@pytest.mark.parametrize(
    "sp_world_size,seq_len,joint_len,num_heads,num_kv_heads",
    [
        # MHA (num_kv_heads=None, joint_len=None)
        (2, 6, None, 3, None),  # head_cnt not divisible by P=2
        (2, 5, None, 4, None),  # seq_len not divisible by P=2
        (4, 9, None, 30, None),  # Z-Image-like: head_cnt not divisible by P=4
        (4, 10, None, 8, None),  # seq_len not divisible by P=4
        # GQA (joint_len=None)
        (2, 6, None, 28, 7),  # BOOGU-like GQA: neither Q nor KV divisible by P=2
        (2, 5, None, 6, 3),  # KV divisible by P, Q is not
        (4, 8, None, 12, 3),  # KV not divisible by P=4
        # Joint GQA (MMDiT-style text stream via AttentionMetadata.joint_*)
        (2, 6, 5, 28, 7),  # BOOGU-like GQA joint: KV=7 not divisible by P=2
        (2, 5, 4, 6, 3),  # KV divisible by P, Q is not
    ],
)
def test_ulysses_uaa_matches_baseline(
    sp_world_size: int,
    seq_len: int,
    joint_len: int | None,
    num_heads: int,
    num_kv_heads: int | None,
) -> None:
    """End-to-end SP-vs-single-GPU parity under UAA.

    Covers MHA, GQA and joint (MMDiT-style) shapes. The joint variant catches
    the pre-fix bug where the joint block padded K/V by Q's head count and
    integer-divided K/V by world_size, giving per-rank shapes that mismatched
    the main-tensor slice (e.g. 14Q/3KV vs. 16Q/4KV for 28Q/7KV at SP=2).
    """
    if current_omni_platform.get_device_count() < sp_world_size:
        pytest.skip(f"Test requires {sp_world_size} GPUs")

    batch_size = 2
    head_size = 8
    kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
    has_joint = joint_len is not None

    base_init_method = get_distributed_init_method()
    sp_init_method = get_distributed_init_method()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as f_in:
        input_file = f_in.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as f_base:
        baseline_file = f_base.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as f_sp:
        sp_file = f_sp.name

    try:
        torch.manual_seed(0)
        q = torch.randn(batch_size, seq_len, num_heads, head_size, dtype=torch.float32)
        k = torch.randn(batch_size, seq_len, kv_heads, head_size, dtype=torch.float32)
        v = torch.randn(batch_size, seq_len, kv_heads, head_size, dtype=torch.float32)
        payload = {"q": q.numpy(), "k": k.numpy(), "v": v.numpy()}
        if has_joint:
            payload["joint_q"] = torch.randn(batch_size, joint_len, num_heads, head_size, dtype=torch.float32).numpy()
            payload["joint_k"] = torch.randn(batch_size, joint_len, kv_heads, head_size, dtype=torch.float32).numpy()
            payload["joint_v"] = torch.randn(batch_size, joint_len, kv_heads, head_size, dtype=torch.float32).numpy()
        np.savez(input_file, **payload)

        common_tail = (1, None, "math", num_kv_heads, has_joint)

        # Baseline (no SP)
        torch.multiprocessing.spawn(
            _run_attention_case,
            args=(1, base_init_method, input_file, baseline_file, num_heads, head_size, 1, "strict", *common_tail),
            nprocs=1,
        )
        # SP (Ulysses-P with UAA)
        torch.multiprocessing.spawn(
            _run_attention_case,
            args=(
                sp_world_size,
                sp_init_method,
                input_file,
                sp_file,
                num_heads,
                head_size,
                sp_world_size,
                "advanced_uaa",
                *common_tail,
            ),
            nprocs=sp_world_size,
        )

        baseline_t = torch.from_numpy(np.load(baseline_file, allow_pickle=False))
        sp_t = torch.from_numpy(np.load(sp_file, allow_pickle=False))
        assert baseline_t.shape == sp_t.shape
        torch.testing.assert_close(sp_t, baseline_t, atol=1e-5, rtol=1e-5)
    finally:
        for path in (input_file, baseline_file, sp_file):
            try:
                os.remove(path)
            except OSError:
                pass


@pytest.mark.parametrize(
    "num_heads,num_kv_heads,joint_len",
    [
        (3, None, None),  # MHA baseline (head_cnt not divisible by ulysses_degree=2)
        (28, 7, None),  # GQA: exercises Ring pytorch_attn_forward KV expansion (Q!=K heads)
        (28, 7, 5),  # GQA + MMDiT joint stream: joint K/V ride the padded Ulysses layout and Ring SDPA
    ],
)
def test_ulysses_uaa_hybrid_ring_matches_baseline(
    num_heads: int, num_kv_heads: int | None, joint_len: int | None
) -> None:
    sp_world_size = 4
    ulysses_degree = 2
    ring_degree = 2

    if current_omni_platform.get_device_count() < sp_world_size:
        pytest.skip(f"Test requires {sp_world_size} GPUs")

    batch_size = 2
    head_size = 8
    seq_len = 10
    kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
    has_joint = joint_len is not None

    # Ensure ring ranks see equal post-Ulysses seq_len:
    # rank0/1 -> 3+2=5, rank2/3 -> 3+2=5
    split_sizes = [3, 2, 3, 2]

    base_init_method = get_distributed_init_method()
    sp_init_method = get_distributed_init_method()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as f_in:
        input_file = f_in.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as f_base:
        baseline_file = f_base.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as f_sp:
        sp_file = f_sp.name

    try:
        torch.manual_seed(0)
        q = torch.randn(batch_size, seq_len, num_heads, head_size, dtype=torch.float32)
        k = torch.randn(batch_size, seq_len, kv_heads, head_size, dtype=torch.float32)
        v = torch.randn(batch_size, seq_len, kv_heads, head_size, dtype=torch.float32)
        payload = {"q": q.numpy(), "k": k.numpy(), "v": v.numpy()}
        if has_joint:
            payload["joint_q"] = torch.randn(batch_size, joint_len, num_heads, head_size, dtype=torch.float32).numpy()
            payload["joint_k"] = torch.randn(batch_size, joint_len, kv_heads, head_size, dtype=torch.float32).numpy()
            payload["joint_v"] = torch.randn(batch_size, joint_len, kv_heads, head_size, dtype=torch.float32).numpy()
        np.savez(input_file, **payload)

        # Baseline (no SP)
        torch.multiprocessing.spawn(
            _run_attention_case,
            args=(
                1,
                base_init_method,
                input_file,
                baseline_file,
                num_heads,
                head_size,
                1,
                "strict",
                1,
                None,
                "mem_efficient",
                num_kv_heads,
                has_joint,
            ),
            nprocs=1,
        )

        # Hybrid SP: Ulysses (P=2) + Ring (P=2) with advanced_uaa
        torch.multiprocessing.spawn(
            _run_attention_case,
            args=(
                sp_world_size,
                sp_init_method,
                input_file,
                sp_file,
                num_heads,
                head_size,
                ulysses_degree,
                "advanced_uaa",
                ring_degree,
                split_sizes,
                "mem_efficient",
                num_kv_heads,
                has_joint,
            ),
            nprocs=sp_world_size,
        )

        baseline = np.load(baseline_file, allow_pickle=False)
        sp = np.load(sp_file, allow_pickle=False)

        baseline_t = torch.from_numpy(baseline)
        sp_t = torch.from_numpy(sp)
        assert baseline_t.shape == sp_t.shape
        # Hybrid (Ulysses+Ring) typically has slightly larger numerical differences
        # than pure Ulysses due to different communication/reduction order and
        # the SDPA kernel path used by Ring attention. Use a looser tolerance to
        # keep the test stable across GPUs/kernels while still catching regressions.
        torch.testing.assert_close(sp_t, baseline_t, atol=5e-4, rtol=5e-4)
    finally:
        for path in (input_file, baseline_file, sp_file):
            try:
                os.remove(path)
            except OSError:
                pass
