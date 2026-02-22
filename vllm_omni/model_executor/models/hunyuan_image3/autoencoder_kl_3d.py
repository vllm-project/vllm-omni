"""
Reference code
[FLUX] https://github.com/black-forest-labs/flux/blob/main/src/flux/modules/autoencoder.py
[DCAE] https://github.com/mit-han-lab/efficientvit/blob/master/efficientvit/models/efficientvit/dc_ae.py
"""

import math
import os
import random
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from safetensors import safe_open
from torch import Tensor, nn
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
)


class DiagonalGaussianDistribution:
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        if parameters.ndim == 3:
            dim = 2  # (B, L, C)
        elif parameters.ndim == 5 or parameters.ndim == 4:
            dim = 1  # (B, C, T, H ,W) / (B, C, H, W)
        else:
            raise NotImplementedError
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=dim)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, generator: torch.Generator | None = None) -> torch.FloatTensor:
        # make sure sample is on the same device as the parameters and has same dtype
        sample = randn_tensor(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        x = self.mean + self.std * sample
        return x

    def mode(self) -> torch.Tensor:
        return self.mean


@dataclass
class DecoderOutput(BaseOutput):
    sample: torch.FloatTensor
    posterior: DiagonalGaussianDistribution | None = None


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


def forward_with_checkpointing(module, *inputs, use_checkpointing=False):
    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)

        return custom_forward

    if use_checkpointing:
        return torch.utils.checkpoint.checkpoint(create_custom_forward(module), *inputs, use_reentrant=False)
    else:
        return module(*inputs)


class Conv3d(nn.Conv3d):
    """Perform Conv3d on patches with numerical differences from nn.Conv3d
    within 1e-5. Only symmetric padding is supported."""

    def forward(self, input):
        B, C, T, H, W = input.shape
        memory_count = (C * T * H * W) * 2 / 1024**3
        if memory_count > 2:
            n_split = math.ceil(memory_count / 2)
            assert n_split >= 2
            chunks = torch.chunk(input, chunks=n_split, dim=-3)
            padded_chunks = []
            for i in range(len(chunks)):
                if self.padding[0] > 0:
                    padded_chunk = F.pad(
                        chunks[i],
                        (0, 0, 0, 0, self.padding[0], self.padding[0]),
                        mode="constant" if self.padding_mode == "zeros" else self.padding_mode,
                        value=0,
                    )
                    if i > 0:
                        padded_chunk[:, :, : self.padding[0]] = chunks[i - 1][:, :, -self.padding[0] :]
                    if i < len(chunks) - 1:
                        padded_chunk[:, :, -self.padding[0] :] = chunks[i + 1][:, :, : self.padding[0]]
                else:
                    padded_chunk = chunks[i]
                padded_chunks.append(padded_chunk)
            padding_bak = self.padding
            self.padding = (0, self.padding[1], self.padding[2])
            outputs = []
            for i in range(len(padded_chunks)):
                outputs.append(super().forward(padded_chunks[i]))
            self.padding = padding_bak
            return torch.cat(outputs, dim=-3)
        else:
            return super().forward(input)


class ColumnParallelConv3d(nn.Module):
    """Column parallel Conv3d layer for tensor parallelism.

    Similar to ColumnParallelLinear, this layer splits the output channels
    across tensor parallel ranks. The input is all-gathered before convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        gather_output: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gather_output = gather_output

        # Get tensor parallel size
        tp_size = get_tensor_model_parallel_world_size()

        # Split output channels across TP ranks
        assert out_channels % tp_size == 0, (
            f"out_channels ({out_channels}) must be divisible by tensor_parallel_size ({tp_size})"
        )
        self.output_size_per_partition = out_channels // tp_size

        # Create the local conv layer
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=self.output_size_per_partition,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, input):
        # All-gather input if needed (for TP > 1)
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size > 1:
            # Input channels should be the same across ranks, so no all-gather needed
            # But we need to ensure input is properly distributed
            pass

        # Perform local convolution
        output = self.conv(input)

        # Gather output if needed
        if self.gather_output and tp_size > 1:
            output = tensor_model_parallel_all_gather(output, dim=1)

        return output


class RowParallelConv3d(nn.Module):
    """Row parallel Conv3d layer for tensor parallelism.

    Similar to RowParallelLinear, this layer splits the input channels
    across tensor parallel ranks and gathers the output.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        input_is_parallel: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_is_parallel = input_is_parallel

        # Get tensor parallel size
        tp_size = get_tensor_model_parallel_world_size()

        # Split input channels across TP ranks
        assert in_channels % tp_size == 0, (
            f"in_channels ({in_channels}) must be divisible by tensor_parallel_size ({tp_size})"
        )
        self.input_size_per_partition = in_channels // tp_size

        # Create the local conv layer
        self.conv = nn.Conv3d(
            in_channels=self.input_size_per_partition,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, input):
        # All-gather input if not already parallel
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size > 1 and not self.input_is_parallel:
            # Split input along channel dimension
            tp_rank = get_tensor_model_parallel_rank()
            input_chunks = torch.chunk(input, tp_size, dim=1)
            input = input_chunks[tp_rank]

        # Perform local convolution
        output = self.conv(input)

        # All-reduce output (sum across TP ranks)
        if tp_size > 1:
            tp_group = get_tp_group()
            # get_tp_group() may return GroupCoordinator or ProcessGroup
            # If it's GroupCoordinator, use device_group; otherwise use directly
            if hasattr(tp_group, "device_group"):
                dist.all_reduce(output, op=dist.ReduceOp.SUM, group=tp_group.device_group)
            else:
                dist.all_reduce(output, op=dist.ReduceOp.SUM, group=tp_group)

        return output


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        # Get tensor parallel size
        tp_size = get_tensor_model_parallel_world_size()

        # Adjust GroupNorm groups for TP
        # GroupNorm requires num_channels to be divisible by num_groups
        # We need to ensure this works with TP
        num_groups = 32
        if tp_size > 1:
            # For TP, we need to ensure the local channel count is divisible by num_groups
            local_in_channels = in_channels  # Input is not sharded
            # Adjust num_groups if needed
            while local_in_channels % num_groups != 0 and num_groups > 1:
                num_groups //= 2
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

        # Use ColumnParallelConv3d for Q, K, V (output channels sharded)
        # gather_output=False to keep output sharded for attention computation
        self.q = ColumnParallelConv3d(in_channels, in_channels, kernel_size=1, gather_output=False)
        self.k = ColumnParallelConv3d(in_channels, in_channels, kernel_size=1, gather_output=False)
        self.v = ColumnParallelConv3d(in_channels, in_channels, kernel_size=1, gather_output=False)
        # Use RowParallelConv3d for proj_out (input channels sharded, output gathered)
        self.proj_out = RowParallelConv3d(in_channels, in_channels, kernel_size=1, input_is_parallel=True)

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # For TP, Q, K, V are sharded along channel dimension
        # We need to all-gather them for attention computation
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size > 1:
            q = tensor_model_parallel_all_gather(q, dim=1)
            k = tensor_model_parallel_all_gather(k, dim=1)
            v = tensor_model_parallel_all_gather(v, dim=1)

        b, c, f, h, w = q.shape
        q = rearrange(q, "b c f h w -> b 1 (f h w) c").contiguous()
        k = rearrange(k, "b c f h w -> b 1 (f h w) c").contiguous()
        v = rearrange(v, "b c f h w -> b 1 (f h w) c").contiguous()
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)

        h_ = rearrange(h_, "b 1 (f h w) c -> b c f h w", f=f, h=h, w=w, c=c, b=b)

        # Shard output back for RowParallelConv3d
        if tp_size > 1:
            tp_rank = get_tensor_model_parallel_rank()
            h_chunks = torch.chunk(h_, tp_size, dim=1)
            h_ = h_chunks[tp_rank]

        return h_

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        # Get tensor parallel size
        tp_size = get_tensor_model_parallel_world_size()

        # Adjust GroupNorm groups for TP
        num_groups = 32
        if tp_size > 1:
            # Adjust num_groups to ensure divisibility
            while in_channels % num_groups != 0 and num_groups > 1:
                num_groups //= 2
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

        # conv1: ColumnParallelConv3d (output channels sharded)
        self.conv1 = ColumnParallelConv3d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, gather_output=False
        )

        # For norm2, we need to handle sharded channels
        # We'll use the local output size for norm2
        local_out_channels = out_channels // tp_size if tp_size > 1 else out_channels
        num_groups2 = 32
        if tp_size > 1:
            while local_out_channels % num_groups2 != 0 and num_groups2 > 1:
                num_groups2 //= 2
        self.norm2 = nn.GroupNorm(num_groups=num_groups2, num_channels=local_out_channels, eps=1e-6, affine=True)

        # conv2: RowParallelConv3d (input channels sharded, output gathered)
        self.conv2 = RowParallelConv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, input_is_parallel=True
        )

        if self.in_channels != self.out_channels:
            # nin_shortcut: RowParallelConv3d (input channels sharded, output gathered)
            self.nin_shortcut = RowParallelConv3d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0, input_is_parallel=False
            )

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)  # Output is sharded along channels

        # For norm2, we work with sharded channels
        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)  # Output is gathered (full channels)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)  # Output is gathered (full channels)
        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int, add_temporal_downsample: bool = True):
        super().__init__()
        self.add_temporal_downsample = add_temporal_downsample
        stride = (2, 2, 2) if add_temporal_downsample else (1, 2, 2)  # THE
        # no asymmetric padding in torch conv, must do it ourselves
        self.conv = Conv3d(in_channels, in_channels, kernel_size=3, stride=stride, padding=0)

    def forward(self, x: Tensor):
        spatial_pad = (0, 1, 0, 1, 0, 0)  # WHAT
        x = nn.functional.pad(x, spatial_pad, mode="constant", value=0)

        temporal_pad = (0, 0, 0, 0, 0, 1) if self.add_temporal_downsample else (0, 0, 0, 0, 1, 1)
        x = nn.functional.pad(x, temporal_pad, mode="replicate")

        x = self.conv(x)
        return x


class DownsampleDCAE(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, add_temporal_downsample: bool = True):
        super().__init__()
        factor = 2 * 2 * 2 if add_temporal_downsample else 1 * 2 * 2
        assert out_channels % factor == 0
        # Use ColumnParallelConv3d (output channels sharded)
        self.conv = ColumnParallelConv3d(
            in_channels, out_channels // factor, kernel_size=3, stride=1, padding=1, gather_output=False
        )

        self.add_temporal_downsample = add_temporal_downsample
        self.group_size = factor * in_channels // out_channels

    def forward(self, x: Tensor):
        r1 = 2 if self.add_temporal_downsample else 1
        h = self.conv(x)
        h = rearrange(h, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
        shortcut = rearrange(x, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)

        B, C, T, H, W = shortcut.shape
        shortcut = shortcut.view(B, h.shape[1], self.group_size, T, H, W).mean(dim=2)
        return h + shortcut


class Upsample(nn.Module):
    def __init__(self, in_channels: int, add_temporal_upsample: bool = True):
        super().__init__()
        self.add_temporal_upsample = add_temporal_upsample
        self.scale_factor = (2, 2, 2) if add_temporal_upsample else (1, 2, 2)  # THE
        self.conv = Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor):
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        x = self.conv(x)
        return x


class UpsampleDCAE(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, add_temporal_upsample: bool = True):
        super().__init__()
        factor = 2 * 2 * 2 if add_temporal_upsample else 1 * 2 * 2
        # Use RowParallelConv3d (input channels sharded, output gathered)
        self.conv = RowParallelConv3d(
            in_channels, out_channels * factor, kernel_size=3, stride=1, padding=1, input_is_parallel=True
        )

        self.add_temporal_upsample = add_temporal_upsample
        self.repeats = factor * out_channels // in_channels

    def forward(self, x: Tensor):
        r1 = 2 if self.add_temporal_upsample else 1
        h = self.conv(x)
        h = rearrange(h, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
        shortcut = x.repeat_interleave(repeats=self.repeats, dim=1)
        shortcut = rearrange(shortcut, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
        return h + shortcut


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        z_channels: int,
        block_out_channels: tuple[int, ...],
        num_res_blocks: int,
        ffactor_spatial: int,
        ffactor_temporal: int,
        downsample_match_channel: bool = True,
    ):
        super().__init__()
        assert block_out_channels[-1] % (2 * z_channels) == 0

        self.z_channels = z_channels
        self.block_out_channels = block_out_channels
        self.num_res_blocks = num_res_blocks

        # downsampling
        # conv_in: ColumnParallelConv3d (output channels sharded)
        self.conv_in = ColumnParallelConv3d(
            in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1, gather_output=False
        )

        self.down = nn.ModuleList()
        block_in = block_out_channels[0]
        for i_level, ch in enumerate(block_out_channels):
            block = nn.ModuleList()
            block_out = ch
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block

            add_spatial_downsample = bool(i_level < np.log2(ffactor_spatial))
            add_temporal_downsample = add_spatial_downsample and bool(
                i_level >= np.log2(ffactor_spatial // ffactor_temporal)
            )
            if add_spatial_downsample or add_temporal_downsample:
                assert i_level < len(block_out_channels) - 1
                block_out = block_out_channels[i_level + 1] if downsample_match_channel else block_in
                down.downsample = DownsampleDCAE(block_in, block_out, add_temporal_downsample)
                block_in = block_out
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # end
        # Adjust GroupNorm for TP
        tp_size = get_tensor_model_parallel_world_size()
        num_groups = 32
        if tp_size > 1:
            local_block_in = block_in // tp_size
            while local_block_in % num_groups != 0 and num_groups > 1:
                num_groups //= 2
        self.norm_out = nn.GroupNorm(
            num_groups=num_groups, num_channels=block_in if tp_size == 1 else block_in // tp_size, eps=1e-6, affine=True
        )
        # conv_out: RowParallelConv3d (input channels sharded, output gathered)
        self.conv_out = RowParallelConv3d(
            block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1, input_is_parallel=True
        )

        self.gradient_checkpointing = False

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            use_checkpointing = bool(self.training and self.gradient_checkpointing)

            # downsampling
            h = self.conv_in(x)
            for i_level in range(len(self.block_out_channels)):
                for i_block in range(self.num_res_blocks):
                    h = forward_with_checkpointing(
                        self.down[i_level].block[i_block], h, use_checkpointing=use_checkpointing
                    )
                if hasattr(self.down[i_level], "downsample"):
                    h = forward_with_checkpointing(
                        self.down[i_level].downsample, h, use_checkpointing=use_checkpointing
                    )

            # middle
            h = forward_with_checkpointing(self.mid.block_1, h, use_checkpointing=use_checkpointing)
            h = forward_with_checkpointing(self.mid.attn_1, h, use_checkpointing=use_checkpointing)
            h = forward_with_checkpointing(self.mid.block_2, h, use_checkpointing=use_checkpointing)

            # end
            # For TP, h is sharded along channels, so we need to handle shortcut differently
            tp_size = get_tensor_model_parallel_world_size()
            if tp_size > 1:
                # All-gather h for shortcut computation
                h_full = tensor_model_parallel_all_gather(h, dim=1)
                group_size = self.block_out_channels[-1] // (2 * self.z_channels)
                shortcut = rearrange(h_full, "b (c r) f h w -> b c r f h w", r=group_size).mean(dim=2)
            else:
                group_size = self.block_out_channels[-1] // (2 * self.z_channels)
                shortcut = rearrange(h, "b (c r) f h w -> b c r f h w", r=group_size).mean(dim=2)
            h = self.norm_out(h)  # Works with sharded channels
            h = swish(h)
            h = self.conv_out(h)  # Output is gathered (full channels)
            h += shortcut
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        z_channels: int,
        out_channels: int,
        block_out_channels: tuple[int, ...],
        num_res_blocks: int,
        ffactor_spatial: int,
        ffactor_temporal: int,
        upsample_match_channel: bool = True,
    ):
        super().__init__()
        assert block_out_channels[0] % z_channels == 0

        self.z_channels = z_channels
        self.block_out_channels = block_out_channels
        self.num_res_blocks = num_res_blocks

        # z to block_in
        block_in = block_out_channels[0]
        # conv_in: ColumnParallelConv3d (output channels sharded)
        self.conv_in = ColumnParallelConv3d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1, gather_output=False
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # upsampling
        self.up = nn.ModuleList()
        for i_level, ch in enumerate(block_out_channels):
            block = nn.ModuleList()
            block_out = ch
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block

            add_spatial_upsample = bool(i_level < np.log2(ffactor_spatial))
            add_temporal_upsample = bool(i_level < np.log2(ffactor_temporal))
            if add_spatial_upsample or add_temporal_upsample:
                assert i_level < len(block_out_channels) - 1
                block_out = block_out_channels[i_level + 1] if upsample_match_channel else block_in
                up.upsample = UpsampleDCAE(block_in, block_out, add_temporal_upsample)
                block_in = block_out
            self.up.append(up)

        # end
        # Adjust GroupNorm for TP
        tp_size = get_tensor_model_parallel_world_size()
        num_groups = 32
        if tp_size > 1:
            local_block_in = block_in // tp_size
            while local_block_in % num_groups != 0 and num_groups > 1:
                num_groups //= 2
        self.norm_out = nn.GroupNorm(
            num_groups=num_groups, num_channels=block_in if tp_size == 1 else block_in // tp_size, eps=1e-6, affine=True
        )
        # conv_out: RowParallelConv3d (input channels sharded, output gathered)
        self.conv_out = RowParallelConv3d(
            block_in, out_channels, kernel_size=3, stride=1, padding=1, input_is_parallel=True
        )

        self.gradient_checkpointing = False

    def forward(self, z: Tensor) -> Tensor:
        with torch.no_grad():
            use_checkpointing = bool(self.training and self.gradient_checkpointing)
            # z to block_in
            repeats = self.block_out_channels[0] // (self.z_channels)
            h = self.conv_in(z)  # Output is sharded along channels
            # For TP, we need to handle the shortcut differently
            tp_size = get_tensor_model_parallel_world_size()
            if tp_size > 1:
                # Shard z.repeat_interleave output
                z_repeated = z.repeat_interleave(repeats=repeats, dim=1)
                tp_rank = get_tensor_model_parallel_rank()
                z_chunks = torch.chunk(z_repeated, tp_size, dim=1)
                h = h + z_chunks[tp_rank]
            else:
                h = h + z.repeat_interleave(repeats=repeats, dim=1)
            # middle
            h = forward_with_checkpointing(self.mid.block_1, h, use_checkpointing=use_checkpointing)
            h = forward_with_checkpointing(self.mid.attn_1, h, use_checkpointing=use_checkpointing)
            h = forward_with_checkpointing(self.mid.block_2, h, use_checkpointing=use_checkpointing)
            # upsampling
            for i_level in range(len(self.block_out_channels)):
                for i_block in range(self.num_res_blocks + 1):
                    h = forward_with_checkpointing(
                        self.up[i_level].block[i_block], h, use_checkpointing=use_checkpointing
                    )
                if hasattr(self.up[i_level], "upsample"):
                    h = forward_with_checkpointing(self.up[i_level].upsample, h, use_checkpointing=use_checkpointing)
            # end
            h = self.norm_out(h)  # Works with sharded channels
            h = swish(h)
            h = self.conv_out(h)  # Output is gathered (full channels)
        return h


class AutoencoderKLConv3D(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_channels: int,
        block_out_channels: tuple[int, ...],
        layers_per_block: int,
        ffactor_spatial: int,
        ffactor_temporal: int,
        sample_size: int,
        sample_tsize: int,
        scaling_factor: float = None,
        shift_factor: float | None = None,
        downsample_match_channel: bool = True,
        upsample_match_channel: bool = True,
        only_encoder: bool = False,
        only_decoder: bool = False,
    ):
        super().__init__()
        self.ffactor_spatial = ffactor_spatial
        self.ffactor_temporal = ffactor_temporal
        self.scaling_factor = scaling_factor
        self.shift_factor = shift_factor

        if not only_decoder:
            self.encoder = Encoder(
                in_channels=in_channels,
                z_channels=latent_channels,
                block_out_channels=block_out_channels,
                num_res_blocks=layers_per_block,
                ffactor_spatial=ffactor_spatial,
                ffactor_temporal=ffactor_temporal,
                downsample_match_channel=downsample_match_channel,
            )
        if not only_encoder:
            self.decoder = Decoder(
                z_channels=latent_channels,
                out_channels=out_channels,
                block_out_channels=list(reversed(block_out_channels)),
                num_res_blocks=layers_per_block,
                ffactor_spatial=ffactor_spatial,
                ffactor_temporal=ffactor_temporal,
                upsample_match_channel=upsample_match_channel,
            )

        self.use_slicing = False
        self.slicing_bsz = 1
        self.use_spatial_tiling = False
        self.use_temporal_tiling = False
        self.use_tiling_during_training = False

        # only relevant if vae tiling is enabled
        self.tile_sample_min_size = sample_size
        self.tile_latent_min_size = sample_size // ffactor_spatial
        self.tile_sample_min_tsize = sample_tsize
        self.tile_latent_min_tsize = sample_tsize // ffactor_temporal
        self.tile_overlap_factor = 0.125

        self.use_compile = False

        self.empty_cache = torch.empty(0, device="cuda")

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (Encoder, Decoder)):
            module.gradient_checkpointing = value

    def enable_tiling_during_training(self, use_tiling: bool = True):
        self.use_tiling_during_training = use_tiling

    def disable_tiling_during_training(self):
        self.enable_tiling_during_training(False)

    def enable_temporal_tiling(self, use_tiling: bool = True):
        self.use_temporal_tiling = use_tiling

    def disable_temporal_tiling(self):
        self.enable_temporal_tiling(False)

    def enable_spatial_tiling(self, use_tiling: bool = True):
        self.use_spatial_tiling = use_tiling

    def disable_spatial_tiling(self):
        self.enable_spatial_tiling(False)

    def enable_tiling(self, use_tiling: bool = True):
        self.enable_spatial_tiling(use_tiling)

    def disable_tiling(self):
        self.disable_spatial_tiling()

    def enable_slicing(self):
        self.use_slicing = True

    def disable_slicing(self):
        self.use_slicing = False

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int):
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
                x / blend_extent
            )
        return b

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int):
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
                y / blend_extent
            )
        return b

    def blend_t(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int):
        blend_extent = min(a.shape[-3], b.shape[-3], blend_extent)
        for x in range(blend_extent):
            b[:, :, x, :, :] = a[:, :, -blend_extent + x, :, :] * (1 - x / blend_extent) + b[:, :, x, :, :] * (
                x / blend_extent
            )
        return b

    def spatial_tiled_encode(self, x: torch.Tensor):
        B, C, T, H, W = x.shape
        # 256 * (1 - 0.25) = 192
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        # 8 * 0.25 = 2
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent  # 8 - 2 = 6

        rows = []
        for i in range(0, H, overlap_size):
            row = []
            for j in range(0, W, overlap_size):
                tile = x[:, :, :, i : i + self.tile_sample_min_size, j : j + self.tile_sample_min_size]
                tile = self.encoder(tile)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=-1))
        moments = torch.cat(result_rows, dim=-2)
        return moments

    def temporal_tiled_encode(self, x: torch.Tensor):
        B, C, T, H, W = x.shape
        # 64 * (1 - 0.25) = 48
        overlap_size = int(self.tile_sample_min_tsize * (1 - self.tile_overlap_factor))
        # 8 * 0.25 = 2
        blend_extent = int(self.tile_latent_min_tsize * self.tile_overlap_factor)
        t_limit = self.tile_latent_min_tsize - blend_extent  # 8 - 2 = 6

        row = []
        for i in range(0, T, overlap_size):
            tile = x[:, :, i : i + self.tile_sample_min_tsize, :, :]
            if self.use_spatial_tiling and (
                tile.shape[-1] > self.tile_sample_min_size or tile.shape[-2] > self.tile_sample_min_size
            ):
                tile = self.spatial_tiled_encode(tile)
            else:
                tile = self.encoder(tile)
            row.append(tile)
        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_extent)
            result_row.append(tile[:, :, :t_limit, :, :])
        moments = torch.cat(result_row, dim=-3)
        return moments

    def spatial_tiled_decode(self, z: torch.Tensor):
        B, C, T, H, W = z.shape
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))  # 24 * (1 - 0.125) = 21
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)  # 384 * 0.125 = 48
        row_limit = self.tile_sample_min_size - blend_extent  # 384 - 48 = 336

        # Distributed/multi-GPU: no padding on input -> each rank pads decoded output to right/bottom 
        # -> GPU all_gather -> rank0 reconstructs/blends/crops
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            rank = dist.get_rank()
            world_size = dist.get_world_size()

            # Count tiles
            num_rows = math.ceil(H / overlap_size)
            num_cols = math.ceil(W / overlap_size)
            total_tiles = num_rows * num_cols
            tiles_per_rank = math.ceil(total_tiles / world_size)

            print(f"==={torch.distributed.get_rank()},  {total_tiles=}, {tiles_per_rank=}, {world_size=}")

            # This rank's tile indices (round-robin allocation): rank, rank+world_size,
            my_linear_indices = list(range(rank, total_tiles, world_size))
            if my_linear_indices == []:
                my_linear_indices = [0]
            print(f"==={torch.distributed.get_rank()},  {my_linear_indices=}")
            decoded_tiles = []  # tiles
            decoded_metas = []  # (ri, rj, pad_w, pad_h)
            H_out_std = self.tile_sample_min_size
            W_out_std = self.tile_sample_min_size
            for lin_idx in my_linear_indices:
                ri = lin_idx // num_cols
                rj = lin_idx % num_cols
                i = ri * overlap_size
                j = rj * overlap_size
                tile = z[
                    :,
                    :,
                    :,
                    i : i + self.tile_latent_min_size,
                    j : j + self.tile_latent_min_size,
                ]
                dec = self.decoder(tile)
                # Pad boundary tile outputs to standard size in right/bottom direction
                pad_h = max(0, H_out_std - dec.shape[-2])
                pad_w = max(0, W_out_std - dec.shape[-1])
                if pad_h > 0 or pad_w > 0:
                    dec = F.pad(dec, (0, pad_w, 0, pad_h, 0, 0), "constant", 0)
                decoded_tiles.append(dec)
                decoded_metas.append(torch.tensor([ri, rj, pad_w, pad_h], device=z.device, dtype=torch.int64))

            # Each rank may have different counts, pad to same length
            T_out = decoded_tiles[0].shape[2] if len(decoded_tiles) > 0 else (T - 1) * self.ffactor_temporal + 1
            while len(decoded_tiles) < tiles_per_rank:
                decoded_tiles.append(
                    torch.zeros(
                        [1, 3, T_out, self.tile_sample_min_size, self.tile_sample_min_size],
                        device=z.device,
                        dtype=dec.dtype,
                    )
                )
                decoded_metas.append(
                    torch.tensor(
                        [-1, -1, self.tile_sample_min_size, self.tile_sample_min_size],
                        device=z.device,
                        dtype=torch.int64,
                    )
                )

            # Perform GPU all_gather
            decoded_tiles = torch.stack(decoded_tiles, dim=0)
            decoded_metas = torch.stack(decoded_metas, dim=0)

            tiles_gather_list = [torch.empty_like(decoded_tiles) for _ in range(world_size)]
            metas_gather_list = [torch.empty_like(decoded_metas) for _ in range(world_size)]

            dist.all_gather(tiles_gather_list, decoded_tiles)
            dist.all_gather(metas_gather_list, decoded_metas)

            if rank != 0:
                # Non-rank0 returns empty placeholder, results only valid on rank0
                return torch.empty(0, device=z.device)

            # rank0: reconstruct tile grid based on (ri, rj) metadata; skip placeholders where (ri, rj) == (-1, -1)
            rows = [[None for _ in range(num_cols)] for _ in range(num_rows)]
            for r in range(world_size):
                # [tiles_per_rank, B, C, T, H, W]
                gathered_tiles_r = tiles_gather_list[r]
                # [tiles_per_rank, 4], elements: (ri, rj, pad_w, pad_h)
                gathered_metas_r = metas_gather_list[r]
                for k in range(gathered_tiles_r.shape[0]):
                    ri = int(gathered_metas_r[k][0])
                    rj = int(gathered_metas_r[k][1])
                    if ri < 0 or rj < 0:
                        continue
                    if ri < num_rows and rj < num_cols:
                        # Remove padding
                        pad_w = int(gathered_metas_r[k][2])
                        pad_h = int(gathered_metas_r[k][3])
                        h_end = None if pad_h == 0 else -pad_h
                        w_end = None if pad_w == 0 else -pad_w
                        rows[ri][rj] = gathered_tiles_r[k][:, :, :, :h_end, :w_end]

            result_rows = []
            for i, row in enumerate(rows):
                result_row = []
                for j, tile in enumerate(row):
                    if tile is None:
                        continue
                    if i > 0:
                        tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                    if j > 0:
                        tile = self.blend_h(row[j - 1], tile, blend_extent)
                    result_row.append(tile[:, :, :, :row_limit, :row_limit])
                result_rows.append(torch.cat(result_row, dim=-1))

            dec = torch.cat(result_rows, dim=-2)
            return dec

        # Single GPU: original sequential logic
        rows = []
        for i in range(0, H, overlap_size):
            row = []
            for j in range(0, W, overlap_size):
                tile = z[
                    :,
                    :,
                    :,
                    i : i + self.tile_latent_min_size,
                    j : j + self.tile_latent_min_size,
                ]
                decoded = self.decoder(tile)
                row.append(decoded)
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=-1))
        dec = torch.cat(result_rows, dim=-2)
        return dec

    def temporal_tiled_decode(self, z: torch.Tensor):
        B, C, T, H, W = z.shape
        # 8 * (1 - 0.25) = 6
        overlap_size = int(self.tile_latent_min_tsize * (1 - self.tile_overlap_factor))
        # 64 * 0.25 = 16
        blend_extent = int(self.tile_sample_min_tsize * self.tile_overlap_factor)
        t_limit = self.tile_sample_min_tsize - blend_extent  # 64 - 16 = 48
        assert 0 < overlap_size < self.tile_latent_min_tsize

        row = []
        for i in range(0, T, overlap_size):
            tile = z[:, :, i : i + self.tile_latent_min_tsize, :, :]
            if self.use_spatial_tiling and (
                tile.shape[-1] > self.tile_latent_min_size or tile.shape[-2] > self.tile_latent_min_size
            ):
                decoded = self.spatial_tiled_decode(tile)
            else:
                decoded = self.decoder(tile)
            row.append(decoded)

        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_extent)
            result_row.append(tile[:, :, :t_limit, :, :])
        dec = torch.cat(result_row, dim=-3)
        return dec

    def encode(self, x: Tensor, return_dict: bool = True):
        def _encode(x):
            if self.use_temporal_tiling and x.shape[-3] > self.tile_sample_min_tsize:
                return self.temporal_tiled_encode(x)
            if self.use_spatial_tiling and (
                x.shape[-1] > self.tile_sample_min_size or x.shape[-2] > self.tile_sample_min_size
            ):
                return self.spatial_tiled_encode(x)

            if self.use_compile:

                @torch.compile
                def encoder(x):
                    return self.encoder(x)

                return encoder(x)
            return self.encoder(x)

        if len(x.shape) != 5:  # (B, C, T, H, W)
            x = x[:, :, None]
        assert len(x.shape) == 5  # (B, C, T, H, W)
        if x.shape[2] == 1:
            x = x.expand(-1, -1, self.ffactor_temporal, -1, -1)
        else:
            assert x.shape[2] != self.ffactor_temporal and x.shape[2] % self.ffactor_temporal == 0

        if self.use_slicing and x.shape[0] > 1:
            if self.slicing_bsz == 1:
                encoded_slices = [_encode(x_slice) for x_slice in x.split(1)]
            else:
                sections = [self.slicing_bsz] * (x.shape[0] // self.slicing_bsz)
                if x.shape[0] % self.slicing_bsz != 0:
                    sections.append(x.shape[0] % self.slicing_bsz)
                encoded_slices = [_encode(x_slice) for x_slice in x.split(sections)]
            h = torch.cat(encoded_slices)
        else:
            h = _encode(x)
        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, z: Tensor, return_dict: bool = True, generator=None):
        def _decode(z):
            if self.use_temporal_tiling and z.shape[-3] > self.tile_latent_min_tsize:
                return self.temporal_tiled_decode(z)
            if self.use_spatial_tiling and (
                z.shape[-1] > self.tile_latent_min_size or z.shape[-2] > self.tile_latent_min_size
            ):
                return self.spatial_tiled_decode(z)
            return self.decoder(z)

        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [_decode(z_slice) for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = _decode(z)
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return self.empty_cache

        if z.shape[-3] == 1:
            decoded = decoded[:, :, -1:]
        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def decode_dist(self, z: Tensor, return_dict: bool = True, generator=None):
        z = z.cuda()
        self.use_spatial_tiling = True
        decoded = self.decode(z)
        self.use_spatial_tiling = False
        return decoded

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """
        Load weights into the VAE model.
        This method accepts weights with "vae." prefix and handles the prefix removal internally.

        Args:
            weights: Iterable of (name, weight) tuples. Names may include "vae." prefix.

        Returns:
            Set of loaded parameter names (with "vae." prefix preserved).
        """
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader

        vae_params_dict = dict(self.named_parameters())
        loaded_params = set()

        for name, weight in weights:
            # Remove "vae." prefix if present for internal parameter lookup
            internal_name = name[4:] if name.startswith("vae.") else name
            original_name = name  # Keep original name for return value

            if internal_name in vae_params_dict:
                param = vae_params_dict[internal_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, weight)
                loaded_params.add(original_name)
            else:
                # Log warning if parameter not found (but don't raise exception)
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(f"VAE parameter '{internal_name}' not found in model (from '{original_name}')")

        return loaded_params

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_posterior: bool = True,
        return_dict: bool = True,
    ):
        posterior = self.encode(sample).latent_dist
        z = posterior.sample() if sample_posterior else posterior.mode()
        dec = self.decode(z).sample
        return DecoderOutput(sample=dec, posterior=posterior) if return_dict else (dec, posterior)

    def random_reset_tiling(self, x: torch.Tensor):
        if x.shape[-3] == 1:
            self.disable_spatial_tiling()
            self.disable_temporal_tiling()
            return

        # Tiling has many restrictions on input_shape and sample_size, arbitrary input_shape 
        # and sample_size may not meet conditions, so fixed values are used here
        min_sample_size = int(1 / self.tile_overlap_factor) * self.ffactor_spatial
        min_sample_tsize = int(1 / self.tile_overlap_factor) * self.ffactor_temporal
        sample_size = random.choice([None, 1 * min_sample_size, 2 * min_sample_size, 3 * min_sample_size])
        if sample_size is None:
            self.disable_spatial_tiling()
        else:
            self.tile_sample_min_size = sample_size
            self.tile_latent_min_size = sample_size // self.ffactor_spatial
            self.enable_spatial_tiling()

        sample_tsize = random.choice([None, 1 * min_sample_tsize, 2 * min_sample_tsize, 3 * min_sample_tsize])
        if sample_tsize is None:
            self.disable_temporal_tiling()
        else:
            self.tile_sample_min_tsize = sample_tsize
            self.tile_latent_min_tsize = sample_tsize // self.ffactor_temporal
            self.enable_temporal_tiling()


def load_sharded_safetensors(model_dir):
    """
    Manually load sharded safetensors files
    Args:
        model_dir: directory path containing shard files
    Returns:
        merged complete weight dictionary
    """
    # Get all shard files and sort by number
    shard_files = []
    for file in os.listdir(model_dir):
        if file.endswith(".safetensors"):
            shard_files.append(file)

    # Sort by shard number
    shard_files.sort(key=lambda x: int(x.split("-")[1]))

    print(f"Found {len(shard_files)} shard files")

    # Merge all weights
    merged_state_dict = dict()

    for shard_file in shard_files:
        shard_path = os.path.join(model_dir, shard_file)
        print(f"Loading shard: {shard_file}")

        # Load current shard using safetensors
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                merged_state_dict[key] = tensor

    print(f"Merging complete, total key count: {len(merged_state_dict)}")
    return merged_state_dict


def load_weights(model, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    def update_state_dict(state_dict: dict[str, torch.Tensor], name, weight):
        if name not in state_dict:
            raise ValueError(f"Unexpected weight {name}")

        model_tensor = state_dict[name]
        if model_tensor.shape != weight.shape:
            raise ValueError(
                f"Shape mismatch for weight {name}: "
                f"model tensor shape {model_tensor.shape} vs. "
                f"loaded tensor shape {weight.shape}"
            )
        if isinstance(weight, torch.Tensor):
            model_tensor.data.copy_(weight.data)
        else:
            raise ValueError(f"Unsupported tensor type in load_weights for {name}: {type(weight)}")

    loaded_params = set()
    for name, load_tensor in weights.items():
        updated = True
        name = name.replace("vae.", "")
        if name in model.state_dict():
            update_state_dict(model.state_dict(), name, load_tensor)
        else:
            updated = False

        if updated:
            loaded_params.add(name)

    return loaded_params
