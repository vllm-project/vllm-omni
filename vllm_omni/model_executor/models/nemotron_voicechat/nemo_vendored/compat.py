# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Vendored from NVIDIA-NeMo/Speech (branch nemotron-labs-voicechat); compatibility shims
# replacing nemo.core / nemo.utils machinery for dependency-free inference.
# Each vendored function below is an exact copy from its annotated original source path;
# only imports were adjusted. Stub classes replace NeMo base classes whose functionality
# is not needed at inference time.

import importlib
import logging as _stdlib_logging
import math
import warnings
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM


# ==============================================================================
# Logging shim (replaces the nemo.utils logging import)
# ==============================================================================
class logging_mode:
    """Stub of nemo.utils.logging's dedup modes (used as `mode=logging_mode.ONCE`)."""

    EACH = 0
    ONCE = 1


class _NemoLoggerShim:
    """stdlib-logging based replacement for the nemo.utils logging module.

    NeMo's logger accepts an extra `mode=` kwarg (deduplication); it is accepted
    and ignored here, with a simple set-based dedup for `mode=logging_mode.ONCE`.
    """

    def __init__(self, logger):
        self._logger = logger
        self._once_seen = set()

    def _log(self, level, msg, args, mode=None):
        if mode == logging_mode.ONCE:
            key = (level, str(msg))
            if key in self._once_seen:
                return
            self._once_seen.add(key)
        self._logger.log(level, msg, *args)

    def debug(self, msg, *args, mode=None, **kwargs):
        self._log(_stdlib_logging.DEBUG, msg, args, mode=mode)

    def info(self, msg, *args, mode=None, **kwargs):
        self._log(_stdlib_logging.INFO, msg, args, mode=mode)

    def warning(self, msg, *args, mode=None, **kwargs):
        self._log(_stdlib_logging.WARNING, msg, args, mode=mode)

    def error(self, msg, *args, mode=None, **kwargs):
        self._log(_stdlib_logging.ERROR, msg, args, mode=mode)

    def critical(self, msg, *args, mode=None, **kwargs):
        self._log(_stdlib_logging.CRITICAL, msg, args, mode=mode)


logging = _NemoLoggerShim(_stdlib_logging.getLogger("nemo_vendored"))


# ==============================================================================
# typecheck no-op (replaces nemo.core.classes.common.typecheck)
# ==============================================================================
class typecheck:
    """No-op replacement for NeMo's typecheck decorator.

    Supports the two usages found in the vendored files:
        @typecheck()
        def forward(...)

        @typecheck.disable_checks()
        def forward(...)

        with typecheck.disable_checks():
            ...
    """

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, func):
        return func

    @staticmethod
    @contextmanager
    def disable_checks():
        yield


# ==============================================================================
# Neural type stubs (replace nemo.core.neural_types)
# ==============================================================================


# ==============================================================================
# from_config_dict target resolver (replaces hydra-based NeMo Serialization)
# ==============================================================================
# Maps the `_target_` strings that appear in the NVIDIA-NemotronLabs-VoiceChat-11B
# checkpoint config.json to the vendored classes. Extend as needed.
_TARGET_REGISTRY = {
    "nemo.collections.asr.modules.ConformerEncoder": (".asr.conformer_encoder", "ConformerEncoder"),
    "nemo.collections.asr.modules.conformer_encoder.ConformerEncoder": (
        ".asr.conformer_encoder",
        "ConformerEncoder",
    ),
    "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor": (
        ".asr.audio_preprocessing",
        "AudioToMelSpectrogramPreprocessor",
    ),
    "nemo.collections.asr.modules.audio_preprocessing.AudioToMelSpectrogramPreprocessor": (
        ".asr.audio_preprocessing",
        "AudioToMelSpectrogramPreprocessor",
    ),
    "nemo.collections.speechlm2.modules.perception.IdentityConnector": (".perception", "IdentityConnector"),
    "nemo.collections.speechlm2.modules.perception.AudioPerceptionModule": (".perception", "AudioPerceptionModule"),
}

_PACKAGE = __name__.rsplit(".", 1)[0]  # ...nemotron_voicechat.nemo_vendored


def from_config_dict(config):
    """Instantiate a vendored class from an OmegaConf/dict config with a `_target_` key.

    Hydra-free replacement for nemo.core.classes.common.Serialization.from_config_dict.
    Like hydra.utils.instantiate with the default `_convert_="none"`, all keys except
    `_target_` are passed as keyword arguments (nested containers stay OmegaConf nodes).
    """
    cfg = OmegaConf.create(config) if not OmegaConf.is_config(config) else config
    if "_target_" not in cfg:
        raise ValueError(f"from_config_dict requires a '_target_' key in the config; got keys: {list(cfg.keys())}")
    target = cfg["_target_"]
    if target not in _TARGET_REGISTRY:
        raise ValueError(
            f"Unknown _target_ '{target}' — not in the vendored nemo_vendored.compat._TARGET_REGISTRY. "
            f"Known targets: {sorted(_TARGET_REGISTRY)}. Add a mapping if this class was vendored."
        )
    module_name, class_name = _TARGET_REGISTRY[target]
    cls = getattr(importlib.import_module(module_name, package=_PACKAGE), class_name)
    kwargs = {str(k): cfg[k] for k in cfg if k != "_target_"}
    return cls(**kwargs)


# ==============================================================================
# NeuralModule / Exportable / mixin stubs (replace nemo.core.classes.*)
# ==============================================================================
class NeuralModule(nn.Module):
    """Slim replacement of nemo.core.classes.module.NeuralModule."""

    @property
    def num_weights(self):
        num: int = 0
        for p in self.parameters():
            if p.requires_grad:
                num += p.numel()
        return num

    @staticmethod
    def from_config_dict(config):
        return from_config_dict(config)

    def unfreeze(self, partial: bool = False) -> None:
        """Unfreeze all parameters for training. (exact copy of NeuralModule.unfreeze)"""
        if partial and not hasattr(self, "_frozen_grad_map"):
            raise ValueError("Cannot unfreeze partially without first freezing the module with `freeze()`")
        for pname, param in self.named_parameters():
            if not partial:
                param.requires_grad = True
            elif pname in self._frozen_grad_map:
                param.requires_grad = self._frozen_grad_map[pname]
        if hasattr(self, "_frozen_grad_map"):
            delattr(self, "_frozen_grad_map")
        self.train()


class Exportable:
    """Stub of nemo.core.classes.exportable.Exportable (ONNX export machinery removed)."""

    @property
    def input_names(self):
        return []

    @property
    def output_names(self):
        return []


class AccessMixin:
    """Stub of nemo.core.classes.mixins.access_mixins.AccessMixin.

    Tensor-capture registry is training/interctc-only; at inference access is
    always disabled so guarded blocks are skipped.
    """

    @property
    def access_cfg(self):
        return {}

    def is_access_enabled(self, guid=None):
        return False

    def register_accessible_tensor(self, name, tensor):
        pass


class AttentionAdapterModuleMixin:
    """Stub of nemo.collections.asr.parts.submodules.adapters.attention_adapter_mixin.
    Adapters (PEFT) are training-only; report none available."""

    def is_adapter_available(self) -> bool:
        return False


class AdapterModuleMixin(AttentionAdapterModuleMixin):
    """Stub of nemo.core.classes.mixins.adapter_mixins.AdapterModuleMixin."""

    pass


# ==============================================================================
# StreamingEncoder — exact copy of
# nemo/collections/asr/parts/mixins/streaming.py (imports adjusted only)
# ==============================================================================
from abc import ABC, abstractmethod  # noqa: E402


class StreamingEncoder(ABC):
    @abstractmethod
    def setup_streaming_params(
        self,
        max_look_ahead: int = 10000,
    ):
        """
        This function sets the needed values and parameters to perform streaming. The configuration (CacheAwareStreamingConfig) need to be stored in self.streaming_cfg.
        The streaming configuration is needed to simulate streaming inference. It would set the following
        """
        pass


# ==============================================================================
# CacheAwareStreamingConfig — exact copy of the dataclass from
# nemo/collections/asr/models/configs/asr_models_config.py
# ==============================================================================
@dataclass
class CacheAwareStreamingConfig:
    chunk_size: int = 0  # the size of each chunk at each step, it can be a list of two integers to specify different chunk sizes for the first step and others
    shift_size: int = 0  # the size of the shift in each step, it can be a list of two integers to specify different shift sizes for the first step and others

    cache_drop_size: int = 0  # the number of steps to drop from the cache
    last_channel_cache_size: int = 0  # the size of the needed cache for last channel layers

    valid_out_len: int = (
        0  # the number of the steps in the final output which are valid (have the same value as in the offline mode)
    )

    pre_encode_cache_size: int = 0  # the size of the needed cache for the pre-encoding part of the model to avoid caching inside the pre-encoding layers
    drop_extra_pre_encoded: int = 0  # the number of steps to get dropped after the pre-encoding layer

    last_channel_num: int = 0  # number of the last channel layers (like MHA layers) which need caching in the model
    last_time_num: int = 0  # number of the last time layers (like convolutions) which need caching in the model


# ==============================================================================
# avoid_float16_autocast_context / avoid_bfloat16_autocast_context — exact
# copies of NeMo nemo/utils/cast_utils.py
# ==============================================================================


def avoid_float16_autocast_context():
    """
    If the current autocast context is float16, cast it to bfloat16
    if available (unless we're in jit) or float32
    """

    if torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() == torch.float16:
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return torch.amp.autocast("cuda", dtype=torch.float32)

        if torch.cuda.is_bf16_supported():
            return torch.amp.autocast("cuda", dtype=torch.bfloat16)
        else:
            return torch.amp.autocast("cuda", dtype=torch.float32)
    else:
        return nullcontext()


# ==============================================================================
# fp32_precision — exact copy of nemo/collections/speechlm2/parts/precision.py
# ==============================================================================
@contextmanager
def fp32_precision():
    """
    Workaround for precision related issues when training with bf16-true PyTorch Lightning precision setting.
    In bf16-true, PTL changes PyTorch's default dtype, which may break implicit assumptions for some models.
    This context manager restores default float32 precision and runs the computation in float32 autocast context.
    """
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float32):
            yield
    finally:
        torch.set_default_dtype(default_dtype)


# ==============================================================================
# get_pad_id — exact copy of nemo/collections/speechlm2/data/utils.py
# ==============================================================================
def get_pad_id(tokenizer) -> int:
    pad_id = tokenizer.pad
    if pad_id is not None:
        return pad_id
    pad_id = tokenizer.unk_id
    if pad_id is not None:
        return pad_id
    warnings.warn(
        "The text tokenizer has no <pad> or <unk> tokens available, using ID 0 for padding (this may lead to silent bugs)."
    )
    return 0


# ==============================================================================
# resample — exact copies of resample/_get_sinc_resample_kernel/_apply_sinc_resample_kernel
# of nemo/collections/audio/parts/utils/resampling.py (itself vendored from
# torchaudio 2.6.0, BSD 2-Clause; see that file for the license text).
# nemo/collections/audio/parts/utils/transforms.py re-exports `resample` from there.
# ==============================================================================
def resample(
    waveform: torch.Tensor,
    orig_freq: int,
    new_freq: int,
    lowpass_filter_width: int = 6,
    rolloff: float = 0.99,
    resampling_method: str = "sinc_interp_hann",
    beta: float | None = None,
) -> torch.Tensor:
    r"""Resamples the waveform at the new frequency using bandlimited interpolation. :cite:`RESAMPLE`."""

    if orig_freq <= 0.0 or new_freq <= 0.0:
        raise ValueError("Original frequency and desired frequency should be positive")

    if orig_freq == new_freq:
        return waveform

    gcd = math.gcd(int(orig_freq), int(new_freq))

    kernel, width = _get_sinc_resample_kernel(
        orig_freq,
        new_freq,
        gcd,
        lowpass_filter_width,
        rolloff,
        resampling_method,
        beta,
        waveform.device,
        waveform.dtype,
    )
    resampled = _apply_sinc_resample_kernel(waveform, orig_freq, new_freq, gcd, kernel, width)
    return resampled


def _get_sinc_resample_kernel(
    orig_freq: int,
    new_freq: int,
    gcd: int,
    lowpass_filter_width: int = 6,
    rolloff: float = 0.99,
    resampling_method: str = "sinc_interp_hann",
    beta: float | None = None,
    device: torch.device = "cpu",
    dtype: torch.dtype | None = None,
):
    if not (int(orig_freq) == orig_freq and int(new_freq) == new_freq):
        raise Exception(
            "Frequencies must be of integer type to ensure quality resampling computation. "
            "To work around this, manually convert both frequencies to integer values "
            "that maintain their resampling rate ratio before passing them into the function. "
            "Example: To downsample a 44100 hz waveform by a factor of 8, use "
            "`orig_freq=8` and `new_freq=1` instead of `orig_freq=44100` and `new_freq=5512.5`. "
            "For more information, please refer to https://github.com/pytorch/audio/issues/1487."
        )

    if resampling_method not in ["sinc_interp_hann", "sinc_interp_kaiser"]:
        raise ValueError("Invalid resampling method: {}".format(resampling_method))

    orig_freq = int(orig_freq) // gcd
    new_freq = int(new_freq) // gcd

    if lowpass_filter_width <= 0:
        raise ValueError("Low pass filter width should be positive.")
    base_freq = min(orig_freq, new_freq)
    # This will perform antialiasing filtering by removing the highest frequencies.
    # At first I thought I only needed this when downsampling, but when upsampling
    # you will get edge artifacts without this, as the edge is equivalent to zero padding,
    # which will add high freq artifacts.
    base_freq *= rolloff

    # The key idea of the algorithm is that x(t) can be exactly reconstructed from x[i] (tensor)
    # using the sinc interpolation formula:
    #   x(t) = sum_i x[i] sinc(pi * orig_freq * (i / orig_freq - t))
    # We can then sample the function x(t) with a different sample rate:
    #    y[j] = x(j / new_freq)
    # or,
    #    y[j] = sum_i x[i] sinc(pi * orig_freq * (i / orig_freq - j / new_freq))

    # We see here that y[j] is the convolution of x[i] with a specific filter, for which
    # we take an FIR approximation, stopping when we see at least `lowpass_filter_width` zeros crossing.
    # But y[j+1] is going to have a different set of weights and so on, until y[j + new_freq].
    # Indeed:
    # y[j + new_freq] = sum_i x[i] sinc(pi * orig_freq * ((i / orig_freq - (j + new_freq) / new_freq))
    #                 = sum_i x[i] sinc(pi * orig_freq * ((i - orig_freq) / orig_freq - j / new_freq))
    #                 = sum_i x[i + orig_freq] sinc(pi * orig_freq * (i / orig_freq - j / new_freq))
    # so y[j+new_freq] uses the same filter as y[j], but on a shifted version of x by `orig_freq`.
    # This will explain the F.conv1d after, with a stride of orig_freq.
    width = math.ceil(lowpass_filter_width * orig_freq / base_freq)
    # If orig_freq is still big after GCD reduction, most filters will be very unbalanced, i.e.,
    # they will have a lot of almost zero values to the left or to the right...
    # There is probably a way to evaluate those filters more efficiently, but this is kept for
    # future work.
    idx_dtype = dtype if dtype is not None else torch.float64

    idx = torch.arange(-width, width + orig_freq, dtype=idx_dtype, device=device)[None, None] / orig_freq

    t = torch.arange(0, -new_freq, -1, dtype=dtype, device=device)[:, None, None] / new_freq + idx
    t *= base_freq
    t = t.clamp_(-lowpass_filter_width, lowpass_filter_width)

    # we do not use built in torch windows here as we need to evaluate the window
    # at specific positions, not over a regular grid.
    if resampling_method == "sinc_interp_hann":
        window = torch.cos(t * math.pi / lowpass_filter_width / 2) ** 2
    else:
        # sinc_interp_kaiser
        if beta is None:
            beta = 14.769656459379492
        beta_tensor = torch.tensor(float(beta))
        window = torch.i0(beta_tensor * torch.sqrt(1 - (t / lowpass_filter_width) ** 2)) / torch.i0(beta_tensor)

    t *= math.pi

    scale = base_freq / orig_freq
    kernels = torch.where(t == 0, torch.tensor(1.0).to(t), t.sin() / t)
    kernels *= window * scale

    if dtype is None:
        kernels = kernels.to(dtype=torch.float32)

    return kernels, width


def _apply_sinc_resample_kernel(
    waveform: torch.Tensor,
    orig_freq: int,
    new_freq: int,
    gcd: int,
    kernel: torch.Tensor,
    width: int,
):
    if not waveform.is_floating_point():
        raise TypeError(f"Expected floating point type for waveform tensor, but received {waveform.dtype}.")

    orig_freq = int(orig_freq) // gcd
    new_freq = int(new_freq) // gcd

    # pack batch
    shape = waveform.size()
    waveform = waveform.view(-1, shape[-1])

    num_wavs, length = waveform.shape
    waveform = torch.nn.functional.pad(waveform, (width, width + orig_freq))
    resampled = torch.nn.functional.conv1d(waveform[:, None], kernel, stride=orig_freq)
    resampled = resampled.transpose(1, 2).reshape(num_wavs, -1)
    target_length = torch.ceil(torch.as_tensor(new_freq * length / orig_freq)).long()
    resampled = resampled[..., :target_length]

    # unpack batch
    resampled = resampled.view(shape[:-1] + resampled.shape[-1:])
    return resampled


# ==============================================================================
# Swish — exact copy of nemo/collections/asr/parts/utils/activations.py
# ==============================================================================
class Swish(nn.SiLU):
    """
    Swish activation function introduced in 'https://arxiv.org/abs/1710.05941'
    Mathematically identical to SiLU. See note in nn.SiLU for references.
    """


# ==============================================================================
# activation_registry — exact copy of nemo/collections/common/parts/utils.py
# ==============================================================================
activation_registry = {
    "identity": nn.Identity,
    "hardtanh": nn.Hardtanh,
    "relu": nn.ReLU,
    "selu": nn.SELU,
    "swish": nn.SiLU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
}


# ==============================================================================
# FusedBatchNorm1d — exact copy of nemo/collections/asr/parts/submodules/batchnorm.py
# ==============================================================================
class FusedBatchNorm1d(nn.Module):
    """
    Fused BatchNorm to use in Conformer to improve accuracy in finetuning with TTS scenario
    Drop-in replacement for BatchNorm1d with simple affine projection
    """

    def __init__(self, num_features: int):
        """
        Args:
            num_features: number of channels, see original BatchNorm1d documentation
        """
        super().__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor):
        if x.dim() == 3:
            return x * self.weight.unsqueeze(-1) + self.bias.unsqueeze(-1)
        assert x.dim() == 2
        return x * self.weight + self.bias


# ==============================================================================
# load_pretrained_hf / set_model_dict_for_partial_init / load_checkpoint —
# exact copies of nemo/collections/speechlm2/parts/pretrained.py
# (only the functions used by the kept inference paths were vendored)
# ==============================================================================
def load_pretrained_hf(
    model_path_or_name: str, pretrained_weights: bool = True, dtype=torch.float32, trust_remote_code: bool = True
):
    """
    Load pretrained HuggingFace AutoModelForCausalLM.

    Setting ``pretrained_weights=False`` returns a model that has identical architecture with the checkpoint,
    but is randomly initialized.

    Args:
        model_path_or_name: Path or name of the model to load
        pretrained_weights: Whether to load pretrained weights (True) or random init (False)
        dtype: Data type for the model
        trust_remote_code: Whether to trust remote code when loading model (needed for some models like Nemotron)
    """
    if pretrained_weights:
        return AutoModelForCausalLM.from_pretrained(
            model_path_or_name, torch_dtype=dtype, trust_remote_code=trust_remote_code
        )
    else:
        config = AutoConfig.from_pretrained(model_path_or_name, trust_remote_code=trust_remote_code)
        return AutoModelForCausalLM.from_config(config, torch_dtype=dtype, trust_remote_code=trust_remote_code)


def set_model_dict_for_partial_init(
    pretrained_dict: Dict[str, torch.Tensor], model_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Partially initialize a model's state dictionary with a pretrained state dictionary.
    This function safely copies compatible layers from a pretrained model into a new model,
    ignoring layers with mismatched shapes or missing keys.

    Steps:
        1. Remove layers from the pretrained dictionary if their shape does not match the target model.
        2. Keep only keys that exist in the target model.
        3. Update the model dictionary with the filtered pretrained weights.

    Args:
        pretrained_dict (Dict[str, torch.Tensor]):
            The state dictionary of the pretrained model.
        model_dict (Dict[str, torch.Tensor]):
            The state dictionary of the target model to be partially initialized.

    Returns:
        Dict[str, torch.Tensor]:
            The updated model state dictionary with compatible layers loaded from the pretrained dictionary.
    """
    # 1. Remove layers where pretrained shape differs from model shape
    for k, v in list(pretrained_dict.items()):
        if k in model_dict and hasattr(model_dict[k], "numel") and v.numel() != model_dict[k].numel():
            del pretrained_dict[k]
            logging.info(f" | > Layer with shape mismatch in the model definition: {k}")

    # 2. Keep only keys that exist in the target model
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # 3. Update model dictionary with filtered pretrained layers
    model_dict.update(pretrained_dict)
    logging.info(f" | > {len(pretrained_dict)} / {len(model_dict)} layers are restored.")

    return model_dict


def load_checkpoint(checkpoint_path):
    """
    Load a model checkpoint from disk.

    Supports loading checkpoints stored in either PyTorch (`.ckpt`, `.pt`) or
    SafeTensors (`.safetensors`) formats. All parameters are loaded onto CPU
    regardless of the original device.

    Args:
        checkpoint_path (str):
            Path to the checkpoint file. If the filename contains `.safetensors`,
            it is loaded using the SafeTensors backend; otherwise, it is assumed
            to be a PyTorch checkpoint containing a `state_dict` field.

    Returns:
        dict:
            A state dictionary mapping parameter names to tensors.
    """
    if ".safetensors" in checkpoint_path:
        checkpoint_state = load_file(checkpoint_path, device="cpu")
    else:
        checkpoint_state = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
    return checkpoint_state
