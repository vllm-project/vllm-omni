from typing import List

import omegaconf
import torch
import torch.nn as nn

from nemo.collections.asr.models.configs import convtt_models_config as cfg
from nemo.collections.asr.parts.convolution_layers import ConvNormAct, create_pad_mask, Conv
from nemo.collections.common.parts.mem_transformer import RelTransformerBlock
import nemo.collections.common.parts.mem_transformer as mem_transformer


class ConvTransformerEncoder(nn.Module):
    """
    Args:
        feat_in (int): the size of feature channels
        feat_out (int): the size of the output features
            Defaults to -1 (means feat_out is d_model)
    """

    def __init__(self, feat_in, use_conv_mask, conv2d_block: cfg.Conv2dBlock, conv_transformer_blocks: List[cfg.ConvTransformerBlock],
                 use_tf_pad: bool, ln_eps: float = 1e-5,
                 init_mode='xavier_uniform', bias_init_mode='zero'):
        super().__init__()

        freeze_config(conv2d_block, conv_transformer_blocks)

        self.use_conv_mask = use_conv_mask

        if conv2d_block:
            prev_out_channels = 1
            self.conv2d_block = nn.ModuleList()
            for conv2d_cfg_i in conv2d_block.layers:
                layer = ConvNormAct(in_channels=prev_out_channels,
                                    conv_type='2d',
                                    use_tf_pad=use_tf_pad,
                                    ln_eps=ln_eps,
                                    **conv2d_cfg_i)
                prev_out_channels = conv2d_cfg_i.filters
                self.conv2d_block.append(layer)
            prev_out_channels = conv2d_block.output_dim
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
                prev_out_channels = conv_cfg_i.filters
                self.block_modules.append(layer)

            if block_cfg.transformer_block is not None:
                block = RelTransformerBlock(**block_cfg.transformer_block, ln_eps=ln_eps)
                self.block_modules.append(block)

        self.apply(lambda x: init_weights(x, mode=init_mode, bias_mode=bias_init_mode))

    def forward(self, audio_signal, length, match_output_len=True):
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

        adaptive_ffn_loss = None
        for module in self.block_modules:
            if isinstance(module, ConvNormAct):
                output, length, pad_mask = module(output, length, pad_mask=pad_mask)
            else:
                assert isinstance(module, RelTransformerBlock)
                # [B, D, T] => [T, B, D]
                output = output.permute(2, 0, 1)
                output, _, extra = module(output, lens=length)
                # [T, B, D] => [B, D, T]
                output = output.permute(1, 2, 0)

                if extra[0] is not None:
                    if adaptive_ffn_loss is None:
                        adaptive_ffn_loss = extra[0]
                    else:
                        adaptive_ffn_loss = adaptive_ffn_loss + extra[0]

        if match_output_len:
            # Ensure that shape mismatch does not occur due to padding
            # Due to padding and subsequent downsampling, it may be possible that
            # max sequence length computed does not match the actual max sequence length
            max_output_len = length.max()
            if output.shape[2] != max_output_len:
                output = output.narrow(dim=2, start=0, length=max_output_len).contiguous()

        return output, length, (adaptive_ffn_loss,)


def init_weights(m, mode='xavier_uniform', bias_mode='zero', embedding_mode='xavier_uniform'):
    if mode == 'xavier_uniform':
        init_ = nn.init.xavier_uniform_
    elif mode == 'xavier_normal':
        init_ = nn.init.xavier_normal_
    else:
        assert mode == 'torch_default'
        init_ = lambda x: x

    if bias_mode == 'zero':
        init_bias_ = nn.init.zeros_
    else:
        assert bias_mode == 'torch_default'
        init_bias_ = lambda x: x

    from nemo.collections.asr.modules import TransformerTDecoder, RNNTJoint
    from nemo.collections.common.parts.normalization import LayerVarNorm
    from nemo.collections.asr.modules.text_embed import TextEmbed
    from nemo.collections.asr.modules import TextConvTransformerEncoder
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
        init_(m.weight)
        if m.bias is not None:
            init_bias_(m.bias)
    elif isinstance(m, nn.Embedding):
        assert embedding_mode == mode
        init_(m.weight)
        if mode != 'torch_default':
            if m.padding_idx is not None:
                with torch.no_grad():
                    m.weight[m.padding_idx].fill_(0)
    elif isinstance(m, RelTransformerBlock):
        if m.r_w_bias is not None:
            init_(m.r_w_bias)
        if m.r_r_bias is not None:
            init_(m.r_r_bias)
    elif isinstance(m, mem_transformer.GateNet):
        init_(m.weight)
        # GateNet itself will handle bias init
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, LayerVarNorm, nn.GroupNorm)):
        pass  # use default init
    elif isinstance(m, (nn.Dropout, nn.ReLU, nn.Sequential, nn.ModuleList)):
        pass  # ignore modules do not need init
    elif isinstance(m, (ConvNormAct, Conv, mem_transformer.RelPartialLearnableMultiHeadAttn,
                        mem_transformer.PositionwiseFF, mem_transformer.RelPartialLearnableDecoderLayer,
                        mem_transformer.PositionalEmbedding, ConvTransformerEncoder,
                        TransformerTDecoder, RNNTJoint, mem_transformer.AdaptiveFFN, TextEmbed, TextConvTransformerEncoder)):
        pass  # ignore wrapper modules
    elif hasattr(m, 'HH_skip_init'):
        pass
    else:
        raise ValueError('initializing unknown module type {}'.format(type(m)))


def freeze_config(*configs):
    for conf in configs:
        if isinstance(conf, omegaconf.Container):
            omegaconf.OmegaConf.set_struct(conf, True)
