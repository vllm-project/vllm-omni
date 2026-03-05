from typing import List

import torch
from torch import nn

from nemo.collections.asr.models.configs import common_config as common_cfg
from nemo.collections.asr.models.spec2vec.spec2vec_config import ConvTransformerBlock
from nemo.collections.asr.parts.convolution_layers import ConvNormAct, create_pad_mask, Conv
from nemo.collections.asr.parts.wav2vec import TransformerEncoder


class FeatureEncoder(nn.Module):
    def __init__(self, feat_in, use_conv_mask, conv2d_block: common_cfg.Conv2dBlock,
                 conv_transformer_blocks: List[ConvTransformerBlock],
                 use_tf_pad: bool, ln_eps: float = 1e-5):
        super().__init__()

        self.use_conv_mask = use_conv_mask

        self.bn_moudles = []

        if conv2d_block:
            prev_out_channels = 1
            self.conv2d_block = nn.ModuleList()
            for conv2d_cfg_i in conv2d_block.layers:
                layer = ConvNormAct(in_channels=prev_out_channels,
                                    conv_type='2d',
                                    use_tf_pad=use_tf_pad,
                                    ln_eps=ln_eps,
                                    **conv2d_cfg_i)

                if isinstance(layer.norm, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    self.bn_moudles.append(layer.norm)

                prev_out_channels = conv2d_cfg_i.filters
                self.conv2d_block.append(layer)
            prev_out_channels = conv2d_block.output_dim
            self.conv2d_block.apply(kaiming_init_conv_weights)
        else:
            self.conv2d_block = None
            prev_out_channels = feat_in

        self.block_modules = nn.ModuleList()
        for block_cfg in conv_transformer_blocks:
            for conv_cfg_i in block_cfg.conv_layers:
                layer = ConvNormAct(in_channels=prev_out_channels,
                                    conv_type='1d',
                                    use_tf_pad=use_tf_pad,
                                    ln_eps=ln_eps,
                                    **conv_cfg_i)

                if isinstance(layer.norm, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    self.bn_moudles.append(layer.norm)

                prev_out_channels = conv_cfg_i.filters
                layer.apply(kaiming_init_conv_weights)
                self.block_modules.append(layer)

            if block_cfg.transformer_block is not None:
                block = TransformerEncoder(block_cfg.transformer_block)
                self.block_modules.append(block)
                prev_out_channels = block_cfg.transformer_block.encoder.embedding_dim

        self.output_dim = prev_out_channels

    def forward(self, audio_signal, length):
        # [B, F/D, T]
        output = audio_signal

        if self.use_conv_mask:
            pad_mask = create_pad_mask(length, max_len=output.size(2))
        else:
            pad_mask = None

        if self.conv2d_block is not None:
            # [B, F, T] => [B, T, F] =>[B, C, T, F]
            output = torch.transpose(output, 1, 2).unsqueeze(1)
            for module in self.conv2d_block:
                output, length, pad_mask = module(output, length, pad_mask=pad_mask)
            b, c, t, f = output.size()
            # [B, C, T, F] => [B, F, C, T] => [B, FxC/D, T]
            output = output.permute(0, 3, 1, 2).reshape(b, f * c, t)

        for module in self.block_modules:
            if isinstance(module, ConvNormAct):
                output, length, pad_mask = module(output, length, pad_mask=pad_mask)
            else:
                assert isinstance(module, TransformerEncoder)
                # [B, D, T] => [B, T, D]
                output = output.transpose(1, 2)
                output = module(output, padding_mask=pad_mask)
                # [B, T, D] => [B, D, T]
                output = output.transpose(1, 2)

        return output, length, None

    def bn_eval(self):
        for m in self.bn_moudles:
            m.eval()

    def get_subsampled_lens(self, lens):
        if self.conv2d_block is not None:
            for module in self.conv2d_block:
                lens = module.update_out_seq_lens(lens)

        for module in self.block_modules:
            if isinstance(module, ConvNormAct):
                lens = module.update_out_seq_lens(lens)

        return lens


class Projector(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.use_conv_mask = cfg.use_conv_mask

        prev_out_channels = cfg.input_dim

        if cfg.conv_layers is not None:
            self.conv_layers = nn.ModuleList()
            for conv_cfg_i in cfg.conv_layers:
                assert conv_cfg_i.stride == (1,)
                layer = ConvNormAct(in_channels=prev_out_channels,
                                    conv_type='1d',
                                    use_tf_pad=cfg.use_tf_pad,
                                    ln_eps=cfg.ln_eps,
                                    **conv_cfg_i)
                prev_out_channels = conv_cfg_i.filters
                self.conv_layers.append(layer)
            self.conv_layers.apply(kaiming_init_conv_weights)
        else:
            self.conv_layers = None

        self.transformer = None if cfg.transformer is None else TransformerEncoder(cfg.transformer)

        if cfg.output_dim is not None:
            self.output_proj = nn.Linear(prev_out_channels, cfg.output_dim)
            self.output_dim = cfg.output_dim
        else:
            self.output_proj = None
            self.output_dim = prev_out_channels

    def forward(self, inputs, length):
        # [B, T, D]
        assert inputs.shape[0] == length.shape[0]
        output = inputs

        if (self.conv_layers is not None and self.use_conv_mask) or self.transformer is not None:
            pad_mask = create_pad_mask(length, max_len=output.size(1))
        else:
            pad_mask = None

        if self.conv_layers is not None:
            # [B, T, D] => [B, D, T]
            output = output.transpose(1, 2)
            for conv_i in self.conv_layers:
                output, length, pad_mask = conv_i(output, length, pad_mask=pad_mask)
            # [B, D, T] => [B, T, D]
            output = output.transpose(1, 2)

        if self.transformer is not None:
            assert pad_mask is not None
            output = self.transformer(output, padding_mask=pad_mask)

        if self.output_proj is not None:
            output = self.output_proj(output)

        return output


def kaiming_init_conv_weights(m):
    if isinstance(m, (nn.Conv1d, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
        pass  # use default init
    elif isinstance(m, (nn.Dropout, nn.ReLU, nn.Sequential, nn.ModuleList)):
        pass  # ignore modules do not need init
    elif isinstance(m, (FeatureEncoder, ConvNormAct, Conv)):
        pass  # ignore wrapper modules
    else:
        raise ValueError('initializing unknown module type {}'.format(type(m)))


class RandomMask(nn.Module):
    def __init__(self, prob, mask_value=None, mask_dim=None):
        super().__init__()
        assert 0 <= prob < 1
        self.prob = prob
        if mask_value is not None:
            assert mask_dim is None
            self.mask_value = mask_value
            self.embedding_mask = False
        else:
            assert isinstance(mask_dim, int)
            self.mask_value = nn.Parameter(torch.FloatTensor(mask_dim).uniform_())
            self.embedding_mask = True

    def forward(self, inputs: torch.Tensor):
        if not self.training:
            return inputs

        if self.embedding_mask:
            mask_shape = inputs.size()[:-1]
        else:
            mask_shape = inputs.size()
        mask_indices = torch.bernoulli(torch.full(mask_shape, self.prob, device=inputs.device)).type(torch.bool)
        inputs[mask_indices] = self.mask_value
        return inputs
