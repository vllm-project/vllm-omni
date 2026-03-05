import omegaconf
import torch.nn as nn

from hmodels.modules.gaussian_upsampling import GaussianUpsampling
from nemo.collections.asr.models.configs import convtt_models_config as cfg
from nemo.collections.asr.models.configs.common_config import TextEmbed as TextEmbedCfg
from nemo.collections.asr.modules.conv_transformer_encoder import init_weights
from nemo.collections.asr.modules.text_embed import TextEmbed
from nemo.collections.asr.parts.convolution_layers import ConvNormAct, create_pad_mask
from nemo.collections.common.parts.mem_transformer import RelTransformerBlock


class TextConvTransformerEncoder(nn.Module):
    def __init__(self, text_embed: TextEmbedCfg, use_conv_mask, encoder: cfg.ConvTransformerBlock,
                 use_tf_pad: bool, ln_eps: float = 1e-5,
                 init_mode='xavier_uniform', bias_init_mode='zero', embedding_init_mode='xavier_uniform',
                 pad_to=0, pad_value=0.0, upsampling_args=None):
        super().__init__()

        freeze_config(text_embed, encoder)

        self.text_embed = TextEmbed(**text_embed, ln_eps=ln_eps, init_mode=init_mode, bias_init_mode=bias_init_mode,
                                    embedding_init_mode=embedding_init_mode)

        self.use_conv_mask = use_conv_mask
        self.pad_to = pad_to
        self.pad_value = pad_value

        self.block_modules = nn.ModuleList()
        for conv_cfg_i in encoder.conv_layers:
            layer = ConvNormAct(in_channels=self.text_embed.embed_size,
                                conv_type='1d',
                                use_tf_pad=use_tf_pad,
                                ln_eps=ln_eps,
                                **conv_cfg_i)
            self.block_modules.append(layer)

        if upsampling_args:
            upsampling_layer = GaussianUpsampling(upsampling_args.variance)
            self.block_modules.append(upsampling_layer)

        if encoder.transformer_block is not None:
            block = RelTransformerBlock(**encoder.transformer_block, ln_eps=ln_eps)
            self.block_modules.append(block)

        self.apply(lambda x: init_weights(x, mode=init_mode, bias_mode=bias_init_mode))

    def forward(self, text, length, match_output_len=True, upsampling_lens=None,
                upsampling_max_len=None):
        output = self.text_embed(text)

        # [B, T, D] => [B, D, T]
        output = output.permute(0, 2, 1)

        if self.pad_to > 0:
            # pad to multiple of pad_to (to avoid issues caused by downsampling and for efficiency)
            pad_amt = output.size(-1) % self.pad_to
            if pad_amt != 0:
                output = nn.functional.pad(output, (0, self.pad_to - pad_amt), value=self.pad_value)

        if self.use_conv_mask:
            pad_mask = create_pad_mask(length, max_len=output.size(2))
        else:
            pad_mask = None

        for module in self.block_modules:
            if isinstance(module, ConvNormAct):
                output, length, pad_mask = module(output, length, pad_mask=pad_mask)
            elif isinstance(module, GaussianUpsampling):
                assert upsampling_max_len is not None
                assert upsampling_lens is not None
                upsampling_durations = (upsampling_lens.float() / length.float()).unsqueeze(1).expand(-1, output.shape[2])
                # [B, D, T] => [B, T, D]
                output = output.permute(0, 2, 1)
                output = module(output, length, upsampling_durations, upsampling_max_len)
                # [B, T, D] => [B, D, T]
                output = output.permute(0, 2, 1)
                length = upsampling_lens
                pad_mask = create_pad_mask(length, max_len=output.size(2))
            else:
                assert isinstance(module, RelTransformerBlock)
                # [B, D, T] => [T, B, D]
                output = output.permute(2, 0, 1)
                output, _, extra = module(output, lens=length)
                # [T, B, D] => [B, D, T]
                output = output.permute(1, 2, 0)

        if match_output_len:
            # Ensure that shape mismatch does not occur due to padding
            # Due to padding and subsequent downsampling, it may be possible that
            # max sequence length computed does not match the actual max sequence length
            max_output_len = length.max()
            if output.shape[2] != max_output_len:
                output = output.narrow(dim=2, start=0, length=max_output_len).contiguous()

        return output, length, None


def freeze_config(*configs):
    for conf in configs:
        if isinstance(conf, omegaconf.Container):
            omegaconf.OmegaConf.set_struct(conf, True)
