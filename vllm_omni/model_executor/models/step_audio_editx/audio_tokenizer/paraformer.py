import logging
import time
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.nn.utils.rnn import pad_sequence
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.model_executor.models.step_audio_editx.utils import load_audio_text_image_video, to_device

from .transformer_utils import (
    LayerNorm,
    MultiHeadedAttentionSANMwithMask,
    PositionwiseFeedForward,
    StreamSinusoidalPositionEncoder,
    repeat,
)

logger = logging.getLogger(__name__)


def extract_fbank(data, data_len=None, data_type: str = "sound", frontend=None, **kwargs):
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
        if len(data.shape) < 2:
            data = data[None, :]  # data: [batch, N]
        data_len = [data.shape[1]] if data_len is None else data_len
    elif isinstance(data, torch.Tensor):
        if len(data.shape) < 2:
            data = data[None, :]  # data: [batch, N]
        data_len = [data.shape[1]] if data_len is None else data_len
    elif isinstance(data, (list, tuple)):
        data_list, data_len = [], []
        for data_i in data:
            if isinstance(data_i, np.ndarray):
                data_i = torch.from_numpy(data_i)
            data_list.append(data_i)
            data_len.append(data_i.shape[0])
        data = pad_sequence(data_list, batch_first=True)  # data: [batch, N]
    data, data_len = frontend(data, data_len, **kwargs)

    if isinstance(data_len, (list, tuple)):
        data_len = torch.tensor([data_len])
    return data.to(torch.float32), data_len.to(torch.int32)


class EncoderLayerSANM(nn.Module):
    def __init__(
        self,
        in_size,
        size,
        self_attn,
        feed_forward,
        normalize_before=True,
        concat_after=False,
        stochastic_depth_rate=0.0,
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(in_size)
        self.norm2 = LayerNorm(size)
        self.in_size = in_size
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        self.stochastic_depth_rate = stochastic_depth_rate

    def forward(self, x, mask, cache=None, mask_shift_chunk=None, mask_att_chunk_encoder=None):
        """Compute encoded features.

        Args:
            x_input (torch.Tensor): Input tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).

        """
        skip_layer = False
        # with stochastic depth, residual connection `x + f(x)` becomes
        # `x <- x + 1 / (1 - p) * f(x)` at training time.
        stoch_layer_coeff = 1.0
        if self.training and self.stochastic_depth_rate > 0:
            skip_layer = torch.rand(1).item() < self.stochastic_depth_rate
            stoch_layer_coeff = 1.0 / (1 - self.stochastic_depth_rate)

        if skip_layer:
            if cache is not None:
                x = torch.cat([cache, x], dim=1)
            return x, mask

        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        if self.concat_after:
            x_concat = torch.cat(
                (
                    x,
                    self.self_attn(
                        x,
                        mask,
                        mask_shift_chunk=mask_shift_chunk,
                        mask_att_chunk_encoder=mask_att_chunk_encoder,
                    ),
                ),
                dim=-1,
            )
            if self.in_size == self.size:
                x = residual + stoch_layer_coeff * self.concat_linear(x_concat)
            else:
                x = stoch_layer_coeff * self.concat_linear(x_concat)
        else:
            if self.in_size == self.size:
                x = residual + stoch_layer_coeff * self.self_attn(
                    x,
                    mask,
                    mask_shift_chunk=mask_shift_chunk,
                    mask_att_chunk_encoder=mask_att_chunk_encoder,
                )
            else:
                x = stoch_layer_coeff * self.self_attn(
                    x,
                    mask,
                    mask_shift_chunk=mask_shift_chunk,
                    mask_att_chunk_encoder=mask_att_chunk_encoder,
                )
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + stoch_layer_coeff * self.feed_forward(x)
        if not self.normalize_before:
            x = self.norm2(x)

        return x, mask, cache, mask_shift_chunk, mask_att_chunk_encoder

    def forward_chunk(self, x, cache=None, chunk_size=None, look_back=0):
        x = x.to(torch.bfloat16)
        residual = x
        if self.normalize_before:
            x = self.norm1(x)
        if self.in_size == self.size:
            attn, cache = self.self_attn.forward_chunk(x, cache, chunk_size, look_back)
            x = residual + attn
        else:
            x, cache = self.self_attn.forward_chunk(x, cache, chunk_size, look_back)
        if not self.normalize_before:
            x = self.norm1(x)
        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        ffn = self.feed_forward(x)
        x = residual + ffn
        if not self.normalize_before:
            x = self.norm2(x)
        return x, cache


class SANMEncoderChunkOpt(nn.Module):
    """
    Author: Shiliang Zhang, Zhifu Gao, Haoneng Luo, Ming Lei, Jie Gao, Zhijie Yan, Lei Xie
    SCAMA: Streaming chunk-aware multihead attention for online end-to-end speech recognition
    https://arxiv.org/abs/2006.01712
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        normalize_before: bool = True,
        concat_after: bool = False,
        interctc_layer_idx: list[int] = [],
        interctc_use_conditioning: bool = False,
        kernel_size: int = 11,
        sanm_shift: int = 0,
    ):
        super().__init__()
        self._output_size = output_size

        self.embed = StreamSinusoidalPositionEncoder()
        self.normalize_before = normalize_before
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (
            output_size,
            linear_units,
        )

        encoder_selfattn_layer = MultiHeadedAttentionSANMwithMask
        encoder_selfattn_layer_args0 = (
            attention_heads,
            input_size,
            output_size,
            kernel_size,
            sanm_shift,
        )

        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            output_size,
            kernel_size,
            sanm_shift,
        )
        self.encoders0 = repeat(
            1,
            lambda lnum: EncoderLayerSANM(
                input_size,
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args0),
                positionwise_layer(*positionwise_layer_args),
                normalize_before,
                concat_after,
            ),
        )

        self.encoders = repeat(
            num_blocks - 1,
            lambda lnum: EncoderLayerSANM(
                output_size,
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                normalize_before,
                concat_after,
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

        self.interctc_layer_idx = interctc_layer_idx
        if len(interctc_layer_idx) > 0:
            assert 0 < min(interctc_layer_idx) and max(interctc_layer_idx) < num_blocks
        self.interctc_use_conditioning = interctc_use_conditioning
        self.conditioning_layer = None
        # shift_fsmn = (kernel_size - 1) // 2

    def output_size(self) -> int:
        return self._output_size

    def _add_overlap_chunk(self, feats: np.ndarray, cache: dict = {}):
        if len(cache) == 0:
            return feats
        cache["feats"] = to_device(cache["feats"], device=feats.device)
        overlap_feats = torch.cat((cache["feats"], feats), dim=1)
        cache["feats"] = overlap_feats[:, -(cache["chunk_size"][0] + cache["chunk_size"][2]) :, :]
        return overlap_feats

    def forward_chunk(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        cache: dict = None,
    ):
        xs_pad *= self.output_size() ** 0.5
        xs_pad = self.embed(xs_pad, cache)
        if cache["tail_chunk"]:
            xs_pad = to_device(cache["feats"], device=xs_pad.device)
        else:
            xs_pad = self._add_overlap_chunk(xs_pad, cache)
        if cache["opt"] is None:
            cache_layer_num = len(self.encoders0) + len(self.encoders)
            new_cache = [None] * cache_layer_num
        else:
            new_cache = cache["opt"]

        for layer_idx, encoder_layer in enumerate(self.encoders0):
            encoder_outs = encoder_layer.forward_chunk(
                xs_pad,
                new_cache[layer_idx],
                cache["chunk_size"],
                cache["encoder_chunk_look_back"],
            )
            xs_pad, new_cache[0] = encoder_outs[0], encoder_outs[1]

        for layer_idx, encoder_layer in enumerate(self.encoders):
            encoder_outs = encoder_layer.forward_chunk(
                xs_pad,
                new_cache[layer_idx + len(self.encoders0)],
                cache["chunk_size"],
                cache["encoder_chunk_look_back"],
            )
            xs_pad, new_cache[layer_idx + len(self.encoders0)] = (
                encoder_outs[0],
                encoder_outs[1],
            )

        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)
        if cache["encoder_chunk_look_back"] > 0 or cache["encoder_chunk_look_back"] == -1:
            cache["opt"] = new_cache

        return xs_pad, ilens, None


class Paraformer(torch.nn.Module):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
    https://arxiv.org/abs/2206.08317
    """

    def __init__(
        self,
        normalize: str = None,
        encoder_conf: dict | None = None,
        input_size: int = 80,
        vocab_size: int = -1,
        ignore_id: int = -1,
        blank_id: int = 0,
        sos: int = 1,
        eos: int = 2,
        sampling_ratio: float = 0.2,
        share_embedding: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.encoder = SANMEncoderChunkOpt(input_size=input_size, **encoder_conf)

        self.blank_id = blank_id
        self.sos = sos if sos is not None else vocab_size - 1
        self.eos = eos if eos is not None else vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.normalize = normalize
        self.sampling_ratio = sampling_ratio
        self.share_embedding = share_embedding


class ParaformerStreaming(Paraformer):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
    https://arxiv.org/abs/2206.08317
    """

    def __init__(
        self,
        encoder_conf: dict = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs, encoder_conf=encoder_conf)

    def encode_chunk(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        cache: dict = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                ind: int
        """
        with autocast(device_type="cuda", enabled=False):
            # Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                speech, speech_lengths = self.normalize(speech, speech_lengths)

        # Forward encoder
        encoder_out, encoder_out_lens, _ = self.encoder.forward_chunk(speech, speech_lengths, cache=cache["encoder"])
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        return encoder_out, torch.tensor([encoder_out.size(1)])

    def init_cache(self, cache: dict = {}, **kwargs):
        chunk_size = kwargs.get("chunk_size", [0, 10, 5])
        encoder_chunk_look_back = kwargs.get("encoder_chunk_look_back", 0)
        decoder_chunk_look_back = kwargs.get("decoder_chunk_look_back", 0)
        batch_size = 1

        enc_output_size = kwargs["encoder_conf"]["output_size"]
        feats_dims = kwargs["frontend_conf"]["n_mels"] * kwargs["frontend_conf"]["lfr_m"]
        cache_encoder = {
            "start_idx": 0,
            "cif_hidden": torch.zeros((batch_size, 1, enc_output_size)),
            "cif_alphas": torch.zeros((batch_size, 1)),
            "chunk_size": chunk_size,
            "encoder_chunk_look_back": encoder_chunk_look_back,
            "last_chunk": False,
            "opt": None,
            "feats": torch.zeros((batch_size, chunk_size[0] + chunk_size[2], feats_dims)),
            "tail_chunk": False,
        }
        cache["encoder"] = cache_encoder

        cache_decoder = {
            "decode_fsmn": None,
            "decoder_chunk_look_back": decoder_chunk_look_back,
            "opt": None,
            "chunk_size": chunk_size,
        }
        cache["decoder"] = cache_decoder
        cache["frontend"] = {}
        cache["prev_samples"] = torch.empty(0)

        return cache

    def infer_encoder(
        self,
        data_in,
        key: list = None,
        frontend=None,
        cache: dict = {},
        **kwargs,
    ):
        if len(cache) == 0:
            self.init_cache(cache, **kwargs)

        meta_data = {}
        chunk_size = kwargs.get("chunk_size", [0, 10, 5])
        chunk_stride_samples = int(chunk_size[1] * 960)  # 600ms

        time1 = time.perf_counter()
        cfg = {"is_final": kwargs.get("is_final", False)}
        if isinstance(data_in[0], torch.Tensor):
            audio_sample_list = data_in
        else:
            audio_sample_list = load_audio_text_image_video(
                data_in,
                fs=frontend.fs,
                audio_fs=kwargs.get("fs", 16000),
                data_type=kwargs.get("data_type", "sound"),
                cache=cfg,
            )
        _is_final = cfg["is_final"]  # if data_in is a file or url, set is_final=True
        time2 = time.perf_counter()
        meta_data["load_data"] = f"{time2 - time1:0.3f}"
        assert len(audio_sample_list) == 1, "batch_size must be set 1"

        audio_sample = torch.cat((cache["prev_samples"], audio_sample_list[0]))

        n = int(len(audio_sample) // chunk_stride_samples + int(_is_final))
        m = int(len(audio_sample) % chunk_stride_samples * (1 - int(_is_final)))
        encoder_outs = []
        meta_data["batch_data_time"] = 0.0
        meta_data["extract_feat"] = 0.0
        for i in range(n):
            kwargs["is_final"] = _is_final and i == n - 1
            audio_sample_i = audio_sample[i * chunk_stride_samples : (i + 1) * chunk_stride_samples]
            time2 = time.perf_counter()
            # extract fbank feats
            if kwargs["is_final"] and len(audio_sample_i) == 0:
                break
            try:
                speech, speech_lengths = extract_fbank(
                    [audio_sample_i],
                    data_type=kwargs.get("data_type", "sound"),
                    frontend=frontend,
                    cache=cache["frontend"],
                    is_final=kwargs["is_final"],
                )
            except Exception as e:
                if i == n - 1 and audio_sample_i.shape[0] < 480:
                    print(f"Warning!!!, skip {audio_sample_i.shape[0]} samples")
                    break
                else:
                    raise RuntimeError("infer failed") from e
            time3 = time.perf_counter()
            if len(speech) == 0 and kwargs["is_final"]:
                break
            meta_data["extract_feat"] = meta_data["extract_feat"] + time3 - time2
            meta_data["batch_data_time"] = (
                meta_data["batch_data_time"]
                + speech_lengths.sum().item() * frontend.frame_shift * frontend.lfr_n / 1000
            )
            speech = speech.to(device=kwargs["device"])
            speech_lengths = speech_lengths.to(device=kwargs["device"])
            encoder_out, encoder_out_lens = self.encode_chunk(
                speech,
                speech_lengths,
                cache=cache,
                is_final=kwargs.get("is_final", False),
            )
            encoder_outs.append(encoder_out[:, (-speech_lengths[0]) :])

            if i == n - 1:
                break
        speech_out = []
        if len(encoder_outs) > 0:
            speech_out = torch.cat(encoder_outs, dim=1)
        result_i = {"key": key[0], "enc_out": speech_out}
        result = [result_i]

        if m > 0:  # tail exists
            cache["prev_samples"] = audio_sample[-m:]
        else:
            cache["prev_samples"] = torch.empty(0)

        if _is_final:
            self.init_cache(cache, **kwargs)

        return result, meta_data, cache

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded = set()

        for name, w in weights:
            if name.startswith("module."):
                name = name[len("module.") :]

            if name not in params:
                continue

            p = params[name]
            wl = getattr(p, "weight_loader", default_weight_loader)
            wl(p, w)
            loaded.add(name)

        return loaded
