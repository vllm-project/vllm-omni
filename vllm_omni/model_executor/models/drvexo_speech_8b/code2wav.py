# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Native Qwen3-TTS codec with the DGX experiment's eager/cuDNN policy."""

import logging

import torch

from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code2wav import Qwen3TTSCode2Wav

from .artifacts import resolve_codec_quantization


class DRVEXOSpeech8BCode2Wav(Qwen3TTSCode2Wav):
    def __init__(self, *, vllm_config, prefix=""):
        config = vllm_config.model_config.hf_config
        self.codec_compute_dtype = getattr(config, "drvexo_codec_compute_dtype", "float32")
        if self.codec_compute_dtype not in ("float32", "bfloat16"):
            raise ValueError("drvexo_codec_compute_dtype must be float32 or bfloat16")
        self.codec_quantization = resolve_codec_quantization(
            vllm_config.model_config.model, getattr(config, "drvexo_codec_quantization", None)
        )
        self._drvexo_weights_loaded = False
        torch.backends.cudnn.enabled = False
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def _maybe_enable_decoder_cudagraph(self, **kwargs):
        # Keep Codec eager. In particular, never capture the parent's FP32
        # decoder before installing/finalizing low-precision kernels.
        return None

    def load_weights(self, weights):
        if self._drvexo_weights_loaded:
            raise RuntimeError("DRVEXO Codec weights already loaded")
        loaded = super().load_weights(weights)
        dtype = getattr(torch, self.codec_compute_dtype)
        if self.codec_quantization:
            from .codec_quantization import install_packed_codec_linears

            # Calibration used native BF16 loading followed by FP32 promotion.
            # Current Qwen3-TTS keeps the configured load dtype; restore that
            # boundary before checking exact reference values or replacing linears.
            self.decoder.float()
            precision, packed, reference = self.codec_quantization
            layers = install_packed_codec_linears(self.decoder, precision, packed, reference, dtype)
            logging.getLogger(__name__).info(
                "DRVEXO Codec: linears=%d precision=%s compute_dtype=%s",
                len(layers),
                precision,
                self.codec_compute_dtype,
            )
            # The outer loader sees replacement parameter names and finalizes
            # native quantized kernels once after this method returns.
            loaded = {name for name, _ in self.named_parameters()}
        else:
            self.decoder.to(dtype=dtype)
        self.decoder.precompute_snake_caches()
        self._drvexo_weights_loaded = True
        logging.getLogger(__name__).info(
            "DRVEXO Codec: compute_dtype=%s eager=true cudnn=false", self.codec_compute_dtype
        )
        return loaded
