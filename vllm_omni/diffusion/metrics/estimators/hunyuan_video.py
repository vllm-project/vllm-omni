from __future__ import annotations

from dataclasses import dataclass

from vllm_omni.diffusion.metrics.common import (
    ceil_to_multiple,
    estimate_cross_attention_dit_layer_flops,
    require_non_negative,
    require_positive,
    video_patch_seq_len,
)


@dataclass(frozen=True)
class HunyuanVideoDiTForwardStats:
    """Runtime shape for HunyuanVideo estimated/theoretical denoiser FLOPs."""

    batch_size: int
    latent_num_frames: int
    latent_height: int
    latent_width: int
    patch_size_t: int
    patch_size: int
    context_len: int
    hidden_dim: int
    ffn_dim: int
    num_layers: int
    num_steps: int
    forwards_per_step_per_gpu: int = 1
    tensor_parallel_size: int = 1
    sequence_parallel_size: int = 1
    pipeline_parallel_size: int = 1

    def __post_init__(self) -> None:
        for field_name, value in self.__dict__.items():
            require_non_negative(field_name, value)
        require_positive("patch_size_t", self.patch_size_t)
        require_positive("patch_size", self.patch_size)
        require_positive("forwards_per_step_per_gpu", self.forwards_per_step_per_gpu)
        require_positive("tensor_parallel_size", self.tensor_parallel_size)
        require_positive("sequence_parallel_size", self.sequence_parallel_size)
        require_positive("pipeline_parallel_size", self.pipeline_parallel_size)

    @property
    def logical_seq_len(self) -> int:
        return video_patch_seq_len(
            latent_num_frames=self.latent_num_frames,
            latent_height=self.latent_height,
            latent_width=self.latent_width,
            patch_size=(self.patch_size_t, self.patch_size, self.patch_size),
        )

    @property
    def padded_seq_len(self) -> int:
        return ceil_to_multiple(self.logical_seq_len, self.sequence_parallel_size)

    @property
    def local_seq_len(self) -> int:
        if self.sequence_parallel_size == 1:
            return self.padded_seq_len
        return self.padded_seq_len // self.sequence_parallel_size


def estimate_hunyuan_video_dit_forward_flops_per_gpu(stats: HunyuanVideoDiTForwardStats) -> int:
    """Estimate HunyuanVideo denoiser FLOPs for the current GPU."""

    flops_per_layer = estimate_cross_attention_dit_layer_flops(
        query_seq_len=stats.local_seq_len,
        key_seq_len=stats.padded_seq_len,
        context_len=stats.context_len,
        hidden_dim=stats.hidden_dim,
        ffn_dim=stats.ffn_dim,
        tensor_parallel_size=stats.tensor_parallel_size,
    )
    return stats.batch_size * flops_per_layer * stats.num_layers * stats.num_steps * stats.forwards_per_step_per_gpu
