# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
# Copyright (c) Ant Group. All rights reserved.
# Adapted from:
# https://github.com/inclusionAI/Ming/tree/e58533db227031990c5a6864dcf5f08fb53ed0d2/AudioVAE

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel, Qwen2Config, Qwen2Model
from transformers.utils import is_flash_attn_2_available
from vllm.logger import init_logger

logger = init_logger(__name__)


class AudioVAEConfig(PretrainedConfig):
    model_type = "audio_vae"

    def __init__(
        self,
        sample_rate: int = 44100,
        enc_kwargs: dict | None = None,
        dec_kwargs: dict | None = None,
        init_method: str = "kaiming",
        patch_size: int = 4,
        **kwargs,
    ):
        self.sample_rate = sample_rate
        self.enc_kwargs = enc_kwargs or {}
        self.dec_kwargs = dec_kwargs or {}
        self.init_method = init_method
        self.patch_size = patch_size
        super().__init__(**kwargs)


class ISTFT(nn.Module):
    def __init__(self, n_fft: int, hop_length: int, win_length: int, padding: str = "same"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)
        self.buffer_len = self.win_length - self.hop_length

    def _buffer_process(self, x, buffer, pad, last_chunk=False, streaming=False):
        if streaming:
            if buffer is None:
                x = x[:, pad:]
            if buffer is not None:
                x[:, : self.buffer_len] += buffer
            buffer = x[:, -self.buffer_len :]
            if not last_chunk:
                x = x[:, : -self.buffer_len]
            else:
                x = x[:, :-pad]
        else:
            x = x[:, pad:-pad]
        return x, buffer

    def forward(self, spec, audio_buffer=None, window_buffer=None, streaming=False, last_chunk=False):
        if self.padding == "center":
            return torch.istft(spec, self.n_fft, self.hop_length, self.win_length, self.window, center=True)
        elif self.padding == "same":
            pad = (self.win_length - self.hop_length) // 2
        else:
            raise ValueError("Padding must be 'center' or 'same'.")

        B, N, T = spec.shape
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]

        output_size = (T - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(
            ifft,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, :]

        y, audio_buffer = self._buffer_process(y, audio_buffer, pad, last_chunk=last_chunk, streaming=streaming)

        window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)
        window_envelope = (
            torch.nn.functional.fold(
                window_sq,
                output_size=(1, output_size),
                kernel_size=(1, self.win_length),
                stride=(1, self.hop_length),
            )
            .squeeze(0)
            .squeeze(0)
        )

        window_envelope, window_buffer = self._buffer_process(
            window_envelope, window_buffer, pad, last_chunk=last_chunk, streaming=streaming
        )
        window_envelope = window_envelope.squeeze()

        assert (window_envelope > 1e-11).all()
        y = y / window_envelope

        return y, audio_buffer, window_buffer


class ISTFTHead(nn.Module):
    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "same"):
        super().__init__()
        out_dim = n_fft + 2
        self.out = nn.Linear(dim, out_dim)
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding)

    def forward(self, x, audio_buffer=None, window_buffer=None, streaming=False, last_chunk=False):
        x_pred = self.out(x)
        x_pred = x_pred.transpose(1, 2)
        mag, p = x_pred.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(mag, max=1e2)
        x = torch.cos(p)
        y = torch.sin(p)
        S = mag * (x + 1j * y)
        audio, audio_buffer, window_buffer = self.istft(
            S, audio_buffer=audio_buffer, window_buffer=window_buffer, streaming=streaming, last_chunk=last_chunk
        )
        return audio.unsqueeze(1), x_pred, audio_buffer, window_buffer


class StreamingLinearUpsample(nn.Module):
    def __init__(self, scale_factor=4):
        super().__init__()
        self.scale_factor = scale_factor
        self.upsampler = nn.Upsample(scale_factor=scale_factor, mode="linear", align_corners=False)

    def forward(self, x, state=None, is_last=False):
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


class Decoder(nn.Module):
    def __init__(self, decoder_args, output_dim=320, latent_dim=64, patch_size=-1):
        super().__init__()
        config = Qwen2Config.from_dict(config_dict=decoder_args)
        if is_flash_attn_2_available():
            config._attn_implementation_autoset = True
            config._attn_implementation = "flash_attention_2"
        else:
            config._attn_implementation = "sdpa"

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

    def low_level_reconstruct(self, x, past_key_values=None, use_cache=False, stream_state=None, last_chunk=False):
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


class Encoder(nn.Module):
    def __init__(self, encoder_args, input_dim=320, hop_size=320, latent_dim=64, patch_size=-1):
        super().__init__()
        config = Qwen2Config.from_dict(config_dict=encoder_args)
        if is_flash_attn_2_available():
            config._attn_implementation_autoset = True
            config._attn_implementation = "flash_attention_2"
        else:
            config._attn_implementation = "sdpa"

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
        x = x.reshape(-1, self.patch_size, dim)
        x = torch.cat((x, self.cls_embed.expand(x.size(0), -1, -1)), dim=1)
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
            x = x.reshape(-1, self.patch_size + 1, dim)
            x = x[:, -1:, :].reshape(bsz, -1, dim)

        x = self.fc3(x)
        return x, waveform.unsqueeze(1)


class AudioVAE(PreTrainedModel):
    config_class = AudioVAEConfig

    def __init__(self, config: AudioVAEConfig):
        super().__init__(config)
        self.encoder = Encoder(
            encoder_args=config.enc_kwargs["backbone"],
            input_dim=config.enc_kwargs["input_dim"],
            hop_size=config.enc_kwargs.get("hop_size", 320),
            latent_dim=config.enc_kwargs["latent_dim"],
            patch_size=config.patch_size,
        )
        self.decoder = Decoder(
            decoder_args=config.dec_kwargs["backbone"],
            output_dim=config.dec_kwargs["output_dim"],
            latent_dim=config.dec_kwargs["latent_dim"],
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

    def encode_latent(self, waveform, waveform_length):
        from diffusers.models.autoencoders.autoencoder_oobleck import OobleckDiagonalGaussianDistribution

        frame_num = torch.ceil(waveform_length / self.config.enc_kwargs["input_dim"]).to(torch.int32)
        if self.config.patch_size != -1:
            frame_num = torch.ceil(frame_num / self.config.patch_size)
        h, y = self.encoder(waveform)
        h = h.transpose(1, 2)

        posterior = OobleckDiagonalGaussianDistribution(h)
        latent = posterior.sample()
        latent = latent.transpose(1, 2)
        return latent, frame_num

    def decode(self, latent, past_key_values=None, use_cache=False, stream_state=(None, None, None), last_chunk=False):
        waveform, stream_state, past_key_values = self.decoder.low_level_reconstruct(
            latent,
            past_key_values=past_key_values,
            use_cache=use_cache,
            stream_state=stream_state,
            last_chunk=last_chunk,
        )
        return waveform, stream_state, past_key_values
