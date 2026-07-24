from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.parallel.base import ParallelAttentionContext
from vllm_omni.diffusion.distributed.group_coordinator import SequenceParallelGroupCoordinator


@dataclass(slots=True)
class AsyncSequenceAllGather:
    """An asynchronous all-gather whose result is concatenated on sequence."""

    output: torch.Tensor
    work: dist.Work
    batch_size: int
    local_seq_len: int

    def wait(self) -> torch.Tensor:
        self.work.wait()
        world_size = self.output.shape[0] // self.batch_size
        return (
            self.output.view(
                world_size,
                self.batch_size,
                self.local_seq_len,
                *self.output.shape[2:],
            )
            .permute(1, 0, 2, *range(3, self.output.ndim + 1))
            .reshape(
                self.batch_size,
                world_size * self.local_seq_len,
                *self.output.shape[2:],
            )
            .contiguous()
        )


def async_all_gather_sequence(
    tensor: torch.Tensor,
    group: dist.ProcessGroup,
) -> AsyncSequenceAllGather:
    """Start an equal-shape all-gather for a ``(B, S_local, ...)`` tensor."""
    if tensor.ndim < 2:
        raise ValueError(f"Context-parallel all-gather expects at least 2 dimensions, got {tensor.shape}.")
    tensor = tensor.contiguous()
    world_size = dist.get_world_size(group)
    output = torch.empty(
        (world_size * tensor.shape[0], *tensor.shape[1:]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    work = dist.all_gather_into_tensor(output, tensor, group=group, async_op=True)
    return AsyncSequenceAllGather(
        output=output,
        work=work,
        batch_size=int(tensor.shape[0]),
        local_seq_len=int(tensor.shape[1]),
    )


@dataclass(frozen=True, slots=True)
class _ContextParallelCtx(ParallelAttentionContext):
    joint_len: int = 0
    joint_strategy: str = "front"


class ContextParallelAttention:
    """KV-gather context parallelism.

    Q remains sequence-local. K/V are all-gathered over the outer SP group,
    after which the regular local attention backend computes local-Q outputs.
    """

    def __init__(self, sp_group: SequenceParallelGroupCoordinator) -> None:
        self._sp_group = sp_group
        self._group = sp_group.device_group

    @property
    def enabled(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "context_parallel"

    def pre_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ):
        if attn_metadata is not None and (
            attn_metadata.attn_mask is not None or attn_metadata.joint_attn_mask is not None
        ):
            raise ValueError("KV-gather context parallelism currently requires equal, unmasked sequence shards.")

        extra = attn_metadata.extra if attn_metadata is not None else {}
        kv_already_gathered = bool(extra.get("context_parallel_kv_gathered", False))
        if not kv_already_gathered:
            key_handle = async_all_gather_sequence(key, self._group)
            value_handle = async_all_gather_sequence(value, self._group)
            key = key_handle.wait()
            value = value_handle.wait()

        joint_query = joint_key = joint_value = None
        joint_strategy = "front"
        if attn_metadata is not None:
            joint_query = attn_metadata.joint_query
            joint_key = attn_metadata.joint_key
            joint_value = attn_metadata.joint_value
            joint_strategy = attn_metadata.joint_strategy

        joint_len = 0
        joint_values = (joint_query, joint_key, joint_value)
        if any(x is not None for x in joint_values):
            if not all(x is not None for x in joint_values):
                raise ValueError("joint_query, joint_key, and joint_value must be provided together.")
            if joint_strategy not in {"front", "rear"}:
                raise ValueError(f"Unsupported joint_strategy={joint_strategy!r}.")
            joint_len = int(joint_query.shape[1])
            if joint_strategy == "front":
                query = torch.cat([joint_query, query], dim=1)
                key = torch.cat([joint_key, key], dim=1)
                value = torch.cat([joint_value, value], dim=1)
            else:
                query = torch.cat([query, joint_query], dim=1)
                key = torch.cat([key, joint_key], dim=1)
                value = torch.cat([value, joint_value], dim=1)

        ctx = _ContextParallelCtx(
            name=self.name,
            joint_len=joint_len,
            joint_strategy=joint_strategy,
        )
        return query, key, value, attn_metadata, ctx

    def post_attention(
        self,
        attn_output: torch.Tensor,
        ctx: ParallelAttentionContext | None,
    ) -> torch.Tensor:
        return attn_output
