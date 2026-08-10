"""
Modified from https://github.com/Gen-Verse/MMaDA/blob/main/models/modeling_magvitv2.py
"""

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch import Tensor

from vllm_omni.model_executor.models.omni_diffusion.third_party.magvit.common_modules import (
    AttnBlock,
    Downsample,
    Normalize,
    ResnetBlock,
    Upsample,
    nonlinearity,
)
from vllm_omni.model_executor.models.omni_diffusion.third_party.magvit.modeling_utils import ModelMixin


class Updateable:
    def do_update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        for attr in self.__dir__():
            if attr.startswith("_"):
                continue
            try:
                module = getattr(self, attr)
            except Exception:
                continue  # ignore attributes like property, which can't be retrieved using getattr?
            if isinstance(module, Updateable):
                module.do_update_step(epoch, global_step, on_load_weights=on_load_weights)
        self.update_step(epoch, global_step, on_load_weights=on_load_weights)

    def do_update_step_end(self, epoch: int, global_step: int):
        for attr in self.__dir__():
            if attr.startswith("_"):
                continue
            try:
                module = getattr(self, attr)
            except Exception:
                continue  # ignore attributes like property, which can't be retrieved using getattr?
            if isinstance(module, Updateable):
                module.do_update_step_end(epoch, global_step)
        self.update_step_end(epoch, global_step)

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # override this method to implement custom update logic
        # if on_load_weights is True, you should be careful doing things related to model evaluations,
        # as the models and tensors are not guaranteed to be on the same device
        pass

    def update_step_end(self, epoch: int, global_step: int):
        pass


class VQGANEncoder(ModelMixin, ConfigMixin):
    @dataclass
    class Config:
        ch: int = 128
        ch_mult: list[int] = field(default_factory=lambda: [1, 2, 2, 4, 4])
        num_res_blocks: list[int] = field(default_factory=lambda: [4, 3, 4, 3, 4])
        attn_resolutions: list[int] = field(default_factory=lambda: [5])
        dropout: float = 0.0
        in_ch: int = 3
        out_ch: int = 3
        resolution: int = 256
        z_channels: int = 13
        double_z: bool = False

    def __init__(
        self,
        ch: int = 128,
        ch_mult: list[int] = [1, 2, 2, 4, 4],
        num_res_blocks: list[int] = [4, 3, 4, 3, 4],
        attn_resolutions: list[int] = [5],
        dropout: float = 0.0,
        in_ch: int = 3,
        out_ch: int = 3,
        resolution: int = 256,
        z_channels: int = 13,
        double_z: bool = False,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_ch = in_ch
        # downsampling
        self.conv_in = torch.nn.Conv2d(self.in_ch, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = self.resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = self.ch * in_ch_mult[i_level]
            block_out = self.ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks[i_level]):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, True)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.quant_conv = torch.nn.Conv2d(z_channels, z_channels, 1)
        # for param in self.parameters():
        #     broadcast(param, src=0)

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks[i_level]):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        h = self.quant_conv(h)
        return h


class LFQuantizer(nn.Module):
    def __init__(
        self,
        num_codebook_entry: int = -1,
        codebook_dim: int = 13,
        beta: float = 0.25,
        entropy_multiplier: float = 0.1,
        commit_loss_multiplier: float = 0.1,
    ):
        super().__init__()
        self.codebook_size = 2**codebook_dim
        print(f"Look-up free quantizer with codebook size: {self.codebook_size}")
        self.e_dim = codebook_dim
        self.beta = beta

        indices = torch.arange(self.codebook_size)

        binary = (indices.unsqueeze(1) >> torch.arange(codebook_dim - 1, -1, -1, dtype=torch.long)) & 1

        embedding = binary.float() * 2 - 1
        self.register_buffer("embedding", embedding)
        self.register_buffer("power_vals", 2 ** torch.arange(codebook_dim - 1, -1, -1))
        self.commit_loss_multiplier = commit_loss_multiplier
        self.entropy_multiplier = entropy_multiplier

    def get_indices(self, z_q):
        return (self.power_vals.reshape(1, -1, 1, 1) * (z_q > 0).float()).sum(1, keepdim=True).long()

    def get_codebook_entry(self, indices, shape=None):
        if shape is None:
            h, w = int(math.sqrt(indices.shape[-1])), int(math.sqrt(indices.shape[-1]))
        else:
            h, w = shape
        b, _ = indices.shape
        indices = indices.reshape(-1)
        z_q = self.embedding[indices]
        z_q = z_q.view(b, h, w, -1)

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q

    def forward(self, z, get_code=False):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        """
        if get_code:
            return self.get_codebook_entry(z)

        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        ge_zero = (z_flattened > 0).float()
        ones = torch.ones_like(z_flattened)
        z_q = ones * ge_zero + -ones * (1 - ge_zero)

        # preserve gradients
        z_q = z_flattened + (z_q - z_flattened).detach()

        # compute entropy loss
        CatDist = torch.distributions.categorical.Categorical
        logit = torch.stack(
            [
                -(z_flattened - torch.ones_like(z_q)).pow(2),
                -(z_flattened - torch.ones_like(z_q) * -1).pow(2),
            ],
            dim=-1,
        )
        cat_dist = CatDist(logits=logit)
        entropy = cat_dist.entropy().mean()
        mean_prob = cat_dist.probs.mean(0)
        mean_entropy = CatDist(probs=mean_prob).entropy().mean()

        # compute loss for embedding
        commit_loss = torch.mean((z_q.detach() - z_flattened) ** 2) + self.beta * torch.mean(
            (z_q - z_flattened.detach()) ** 2
        )

        # reshape back to match original input shape
        z_q = z_q.view(z.shape)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return {
            "z": z_q,
            "quantizer_loss": commit_loss * self.commit_loss_multiplier,
            "entropy_loss": (entropy - mean_entropy) * self.entropy_multiplier,
            "indices": self.get_indices(z_q),
        }


class VQGANDecoder(ModelMixin, ConfigMixin):
    def __init__(
        self,
        ch: int = 128,
        ch_mult: list[int] = [1, 1, 2, 2, 4],
        num_res_blocks: list[int] = [4, 4, 3, 4, 3],
        attn_resolutions: list[int] = [5],
        dropout: float = 0.0,
        in_ch: int = 3,
        out_ch: int = 3,
        resolution: int = 256,
        z_channels: int = 13,
        double_z: bool = False,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_ch = in_ch
        self.give_pre_end = False

        self.z_channels = z_channels
        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = self.resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print(f"Working with z of shape {self.z_shape} = {np.prod(self.z_shape)} dimensions.")

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks[i_level]):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, True)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)
        self.post_quant_conv = torch.nn.Conv2d(z_channels, z_channels, 1)

    def forward(self, z):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape
        # timestep embedding
        temb = None
        output = dict()
        z = self.post_quant_conv(z)

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks[i_level]):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        output["output"] = h
        if self.give_pre_end:
            return output

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        output["output"] = h
        return output


class MAGVITv2(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
    ):
        super().__init__()

        self.encoder = VQGANEncoder()
        self.decoder = VQGANDecoder()
        self.quantize = LFQuantizer()

    def forward(self, pixel_values, return_loss=False):
        pass

    def encode(self, pixel_values, return_loss=False):
        hidden_states = self.encoder(pixel_values)
        quantized_states = self.quantize(hidden_states)["z"]
        codebook_indices = self.quantize.get_indices(quantized_states).reshape(pixel_values.shape[0], -1)
        output = (quantized_states, codebook_indices)
        return output

    def get_code(self, pixel_values: Tensor) -> Tensor:
        hidden_states = self.encoder(pixel_values)
        codebook_indices = self.quantize.get_indices(self.quantize(hidden_states)["z"])
        codebook_indices = codebook_indices.reshape(pixel_values.shape[0], -1)

        return codebook_indices

    def decode_code(self, codebook_indices: Tensor, shape=None) -> Tensor:
        z_q = self.quantize.get_codebook_entry(codebook_indices, shape=shape)
        z_q = z_q.to(dtype=self.decoder.post_quant_conv.weight.dtype)

        reconstructed_pixel_values = self.decoder(z_q)["output"]
        return reconstructed_pixel_values
