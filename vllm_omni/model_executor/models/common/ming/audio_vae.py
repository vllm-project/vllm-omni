# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel, Qwen2Config, Qwen2Model
from transformers.utils import is_flash_attn_2_available
from vllm.logger import init_logger

from .audio_dsp import ISTFTHead

logger = init_logger(__name__)


class AudioVAEConfig(PretrainedConfig):
    model_type = "audio_vae"

    def __init__(
        self,
        sample_rate=44100,
        enc_kwargs=None,
        semantic_module_kwargs=None,
        dec_kwargs=None,
        hifi_gan_disc_kwargs=None,
        spec_disc_kwargs=None,
        lambda_disc=1.0,
        lambda_mel_loss=15,
        lambda_adv=1.0,
        lambda_feat_match_loss=1.0,
        lambda_semantic=5.0,
        init_method="kaiming",
        patch_size=4,
        **kwargs,
    ):
        self.sample_rate = sample_rate
        self.enc_kwargs = enc_kwargs or {}
        self.semantic_module_kwargs = semantic_module_kwargs
        self.dec_kwargs = dec_kwargs or {}
        self.hifi_gan_disc_kwargs = hifi_gan_disc_kwargs
        self.spec_disc_kwargs = spec_disc_kwargs
        self.lambda_disc = lambda_disc
        self.lambda_mel_loss = lambda_mel_loss
        self.lambda_adv = lambda_adv
        self.lambda_feat_match_loss = lambda_feat_match_loss
        self.lambda_semantic = lambda_semantic
        self.init_method = init_method
        self.patch_size = patch_size
        super().__init__(**kwargs)


def _qwen2_config(backbone):
    config = Qwen2Config.from_dict(config_dict=backbone)
    if (getattr(config, "_attn_implementation", None) or getattr(config, "attn_implementation", None)) not in (
        None,
        "flash_attention_2",
    ):
        return config
    if is_flash_attn_2_available():
        config._attn_implementation_autoset = True
        config._attn_implementation = "flash_attention_2"
    else:
        config._attn_implementation = "sdpa"
    return config


class StreamingLinearUpsample(nn.Module):
    def __init__(self, scale_factor=4):
        super().__init__()
        self.scale_factor = scale_factor
        self.upsampler = nn.Upsample(scale_factor=scale_factor, mode="linear", align_corners=False)

    def forward(self, x, state=None, is_last=False):
        if x is None and is_last and (state is None or state.get("prev_chunk") is None):
            raise ValueError("Received end-of-stream without any latent chunk to upsample.")
        if state is None:
            state = {"prev_chunk": None, "history_last": None, "is_first": True}

        if x is None and not is_last:
            return None, state

        if state["is_first"] and is_last:
            out = self.upsampler(x.transpose(1, 2)).transpose(1, 2)
            return out, None

        output_chunks = []

        if state["is_first"]:
            state["prev_chunk"] = x
            state["is_first"] = False
            if not is_last:
                return None, state

        if state["prev_chunk"] is not None:
            p = state["prev_chunk"].transpose(1, 2)

            if state["history_last"] is None:
                lookahead = x[:, :1, :].transpose(1, 2)
                inp = torch.cat([p, lookahead], dim=2)
                up = self.upsampler(inp)
                out_prev = up[:, :, : p.size(2) * self.scale_factor]
            else:
                lookahead = x[:, :1, :].transpose(1, 2)
                inp = torch.cat([state["history_last"], p, lookahead], dim=2)
                up = self.upsampler(inp)
                start = self.scale_factor
                end = start + p.size(2) * self.scale_factor
                out_prev = up[:, :, start:end]

            output_chunks.append(out_prev.transpose(1, 2))
            state["history_last"] = p[:, :, -1:]
            state["prev_chunk"] = x

        if is_last:
            p = state["prev_chunk"].transpose(1, 2)
            inp = torch.cat([state["history_last"], p], dim=2)
            up = self.upsampler(inp)
            out_last = up[:, :, self.scale_factor :]
            output_chunks.append(out_last.transpose(1, 2))
            state = None

        final_out = torch.cat(output_chunks, dim=1) if output_chunks else None
        return final_out, state


class Encoder(nn.Module):
    def __init__(self, encoder_args, input_dim=320, hop_size=320, latent_dim=64, patch_size=-1):
        super().__init__()
        config = _qwen2_config(encoder_args)
        logger.info("AudioVAE Encoder: using attn_implementation=%r", config._attn_implementation)
        self.encoder = Qwen2Model(config)
        self.input_dim = input_dim
        self.hop_size = hop_size
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(input_dim, config.hidden_size, bias=False)
        self.fc2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.fc3 = nn.Linear(config.hidden_size, latent_dim * 2)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.patch_size = patch_size
        if patch_size != -1:
            config.num_hidden_layers = 4
            self.aggregator = Qwen2Model(config)
            self.cls_embed = nn.Parameter(torch.rand(1, 1, config.hidden_size))
            self.cls_embed.data.normal_(0, 0.02)

    def get_frames(self, x):
        num_frames_total = (x.size(-1) + self.hop_size - 1) // self.hop_size
        expected_len = (num_frames_total - 1) * self.hop_size + self.input_dim
        padding_needed = expected_len - x.size(-1)
        waveform = F.pad(x, (0, padding_needed), value=0.0)
        frames = waveform.unfold(dimension=-1, size=self.input_dim, step=self.hop_size)
        return frames

    def pad_patch_insert_cls(self, x):
        bsz, _, dim = x.size()
        num_frame = x.size(1)
        r = num_frame % self.patch_size
        pad_num = self.patch_size - r if r else 0
        x = F.pad(x, (0, 0, 0, pad_num), value=0.0)
        # [Batch, Time, Dimension] -> [Batch*PatchGroups, Patch, Dimension].
        x = x.reshape(-1, self.patch_size, dim)
        x = torch.cat((x, self.cls_embed.expand(x.size(0), -1, -1)), dim=1)
        # [Batch*PatchGroups, Patch+1, Dimension] -> [Batch, Time, Dimension].
        x = x.reshape(bsz, -1, dim)
        return x

    def forward(self, waveform):
        x = self.get_frames(waveform)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.encoder(inputs_embeds=x)
        x = x.last_hidden_state

        if self.patch_size != -1:
            x = self.pad_patch_insert_cls(x)
            x = self.aggregator(inputs_embeds=x)
            x = x.last_hidden_state
            bsz, _, dim = x.size()
            # [Batch, Time, Dimension] -> [Batch*PatchGroups, Patch+1, Dimension].
            x = x.reshape(-1, self.patch_size + 1, dim)
            # [Batch*PatchGroups, 1, Dimension] -> [Batch, PatchGroups, Dimension].
            x = x[:, -1:, :].reshape(bsz, -1, dim)

        x = self.fc3(x)
        return x, waveform.unsqueeze(1)


class Decoder(nn.Module):
    def __init__(self, decoder_args, output_dim=320, latent_dim=64, patch_size=-1):
        super().__init__()
        config = _qwen2_config(decoder_args)
        logger.info("AudioVAE Decoder: using attn_implementation=%r", config._attn_implementation)
        self.decoder = Qwen2Model(config)
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(latent_dim, config.hidden_size)
        self.hop_length = output_dim
        self.head = ISTFTHead(
            dim=config.hidden_size, n_fft=self.hop_length * 4, hop_length=self.hop_length, padding="same"
        )
        self.patch_size = patch_size
        if self.patch_size != -1:
            self.upsampling = StreamingLinearUpsample(scale_factor=patch_size)

    def forward(self, x):
        x = self.fc1(x)

        if self.patch_size != -1:
            # [Batch, Time, Dimension] -> [Batch, Dimension, Time] -> [Batch, Time, Dimension].
            x = self.upsampling.upsampler(x.transpose(1, 2)).transpose(1, 2)

        x = self.decoder(inputs_embeds=x)
        x = x.last_hidden_state
        x, _ = self.head(x)
        return x, None

    def low_level_reconstruct(self, x, past_key_values=None, use_cache=False, stream_state=None, last_chunk=False):
        if stream_state is None:
            stream_state = (None, None, None)
        upsample_state, audio_buffer, window_buffer = stream_state
        bsz, device, dtype = x.size(0), x.device, x.dtype
        x = self.fc1(x)
        if self.patch_size != -1:
            if use_cache:
                x, upsample_state = self.upsampling(x, state=upsample_state, is_last=last_chunk)
                if x is None:
                    stream_state = (upsample_state, audio_buffer, window_buffer)
                    return torch.empty(bsz, 1, 0, device=device, dtype=dtype), stream_state, past_key_values
            else:
                # [Batch, Time, Dimension] -> [Batch, Dimension, Time] -> [Batch, Time, Dimension].
                x = self.upsampling.upsampler(x.transpose(1, 2)).transpose(1, 2)

        hidden_states_list = []

        if use_cache and getattr(self.decoder.config, "sliding_window", None) is not None:
            sw_size = self.decoder.config.sliding_window
            target_len = sw_size - 1
            if past_key_values is None:
                past_len = 0
            elif hasattr(past_key_values, "get_seq_length"):
                past_len = past_key_values.get_seq_length()
            elif isinstance(past_key_values, tuple) and len(past_key_values) > 0:
                past_len = past_key_values[0][0].shape[-2]
            else:
                past_len = 0

            curr_len = x.shape[1]
            if past_len < target_len and (past_len + curr_len) >= sw_size:
                fill_len = target_len - past_len
                x_fill = x[:, :fill_len, :]
                outputs = self.decoder(inputs_embeds=x_fill, past_key_values=past_key_values, use_cache=use_cache)
                hidden_states_list.append(outputs.last_hidden_state)
                past_key_values = outputs.past_key_values
                x = x[:, fill_len:, :]

        outputs = self.decoder(inputs_embeds=x, past_key_values=past_key_values, use_cache=use_cache)
        hidden_states_list.append(outputs.last_hidden_state)
        past_key_values = outputs.past_key_values

        if len(hidden_states_list) > 1:
            full_hidden_state = torch.cat(hidden_states_list, dim=1)
        else:
            full_hidden_state = hidden_states_list[0]

        x_out, _, audio_buffer, window_buffer = self.head(
            full_hidden_state,
            streaming=use_cache,
            audio_buffer=audio_buffer,
            window_buffer=window_buffer,
            last_chunk=last_chunk,
        )

        stream_state = (upsample_state, audio_buffer, window_buffer)
        return x_out, stream_state, past_key_values


class AudioVAE(PreTrainedModel):
    config_class = AudioVAEConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        enc_kwargs = config.enc_kwargs
        dec_kwargs = config.dec_kwargs
        for key in ("backbone", "input_dim", "latent_dim"):
            if key not in enc_kwargs:
                raise ValueError(f"AudioVAE.enc_kwargs missing required key: {key}")
        for key in ("backbone", "output_dim", "latent_dim"):
            if key not in dec_kwargs:
                raise ValueError(f"AudioVAE.dec_kwargs missing required key: {key}")

        hop_size = enc_kwargs.get("hop_size", enc_kwargs["input_dim"])
        if enc_kwargs["input_dim"] != hop_size:
            raise ValueError(f"AudioVAE encoder input_dim ({enc_kwargs['input_dim']}) != hop_size ({hop_size}).")
        if hop_size != dec_kwargs["output_dim"]:
            raise ValueError(
                f"AudioVAE encoder hop_size ({hop_size}) != decoder output_dim ({dec_kwargs['output_dim']})."
            )
        self.encoder = Encoder(
            encoder_args=enc_kwargs["backbone"],
            input_dim=enc_kwargs["input_dim"],
            hop_size=hop_size,
            latent_dim=enc_kwargs["latent_dim"],
            patch_size=config.patch_size,
        )
        self.decoder = Decoder(
            decoder_args=dec_kwargs["backbone"],
            output_dim=dec_kwargs["output_dim"],
            latent_dim=dec_kwargs["latent_dim"],
            patch_size=config.patch_size,
        )
        self.post_init()

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            if self.config.init_method == "kaiming":
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            else:
                module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @torch.inference_mode()
    def encode_latent(self, waveform, waveform_length):
        if waveform.ndim != 2:
            raise ValueError(f"Expected waveform rank-2 [Batch, Time], got {tuple(waveform.shape)}")
        if waveform_length.ndim != 1:
            raise ValueError(f"Expected waveform_length rank-1 [Batch], got {tuple(waveform_length.shape)}")
        if waveform.shape[0] != waveform_length.shape[0]:
            raise ValueError(
                "Batch mismatch: "
                f"waveform batch={waveform.shape[0]} vs "
                f"waveform_length batch={waveform_length.shape[0]}"
            )
        if torch.any(waveform_length <= 0):
            raise ValueError("waveform_length must be strictly positive.")

        frame_num = torch.ceil(waveform_length / self.config.enc_kwargs["input_dim"]).to(torch.int32)
        if self.config.patch_size != -1:
            frame_num = torch.ceil(frame_num / self.config.patch_size)
        h, _ = self.encoder(waveform)
        h = h.transpose(1, 2)

        mean, scale = torch.chunk(h, 2, dim=1)
        std = F.softplus(scale) + 1e-4
        latent = (mean + std * torch.randn_like(mean)).transpose(1, 2)
        return latent, frame_num

    @torch.inference_mode()
    def decode(self, latent, past_key_values=None, use_cache=False, stream_state=(None, None, None), last_chunk=False):
        if latent.dim() != 3:
            raise ValueError(f"Expected latent rank-3 [B,T,D], got shape={tuple(latent.shape)}")
        if latent.shape[0] <= 0:
            raise ValueError("latent batch size must be positive.")

        target_dtype = next(self.decoder.parameters()).dtype
        target_device = next(self.decoder.parameters()).device
        if latent.dtype != target_dtype or latent.device != target_device:
            latent = latent.to(device=target_device, dtype=target_dtype)

        expected_latent_dim = self.config.dec_kwargs["latent_dim"]
        if latent.shape[-1] != expected_latent_dim:
            raise ValueError(f"Latent dim mismatch in decode(): got {latent.shape[-1]}, expected {expected_latent_dim}")

        waveform, stream_state, past_key_values = self.decoder.low_level_reconstruct(
            latent,
            past_key_values=past_key_values,
            use_cache=use_cache,
            stream_state=stream_state,
            last_chunk=last_chunk,
        )
        return waveform, stream_state, past_key_values
