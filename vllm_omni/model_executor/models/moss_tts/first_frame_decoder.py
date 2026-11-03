# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MOSS Local first-frame decoding in the Talker process.

Empty-history attention needs no persistent stream state. Every upsampling
block is retained; the codec stage independently primes its streaming state
with the same codes.
"""

from __future__ import annotations

import torch
from torch import nn
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.logger import init_logger

from .first_frame_special import StatelessFirstGraphs, specialize
from .modeling_moss_tts_codec import load_codec

logger = init_logger(__name__)


class MossFirstFrameDecoder(nn.Module):
    def __init__(self, codec_path: str, num_quantizers: int):
        super().__init__()
        self._codec_path = codec_path
        self._num_quantizers = num_quantizers

    def load(self, config: VllmConfig) -> set[str]:
        # First-frame rows are requests, while Talker graph sizes count tokens.
        # Reuse its resolved buckets up to the request capacity.
        compilation = config.compilation_config
        capacity = config.scheduler_config.max_num_seqs
        batch_sizes = tuple(
            sorted({size for size in (compilation.cudagraph_capture_sizes or []) if 0 < size <= capacity})
        )
        if config.model_config.enforce_eager or compilation.cudagraph_mode == CUDAGraphMode.NONE:
            batch_sizes = ()
        logger.info("MOSS first-frame codec follows Talker graph buckets: B=%s T=1", batch_sizes)
        codec_config, self._codec = load_codec(
            self._codec_path,
            device=config.device_config.device,
            load_config=config.load_config,
            num_quantizers=self._num_quantizers,
            attention_backend="sdpa",
        )
        # Reference encoding belongs to the API processor, never this decoder.
        self._codec.encoder = None
        self._sr_tensor = torch.tensor(int(codec_config.sampling_rate), dtype=torch.int32)
        specialize(self._codec)
        self._special_graphs = StatelessFirstGraphs(
            self._codec,
            self._num_quantizers,
            batch_sizes,
            warmups=compilation.cudagraph_num_of_warmups,
        )
        return set(dict(self.named_parameters()))

    @property
    def sample_rate(self) -> torch.Tensor:
        return self._sr_tensor

    @torch.inference_mode()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """[B, NQ] codes -> owned float32 [B, channels, samples] PCM."""
        return self._special_graphs(codes)
