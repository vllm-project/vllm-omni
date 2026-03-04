# coding=utf-8
# Copyright 2024 The EMOVA team and The HuggingFace Inc. team. All rights reserved.
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
"""EMOVASpeechTokenizer model (encode+decode runtime)."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel

try:
    from .emova_decode_runtime import (
        S2U_IMPORT_ERROR,
        SynthesizerTrn,
        VQCTCFinetuneModel,
        get_S2U_ckpt_config_path,
        get_U2S_config_checkpoint_file,
        load_U2S_config,
        load_config,
        s2u_extract_unit_demo,
        synthesis,
    )
except Exception as e:
    raise ImportError(
        "Dependencies of emova speech tokenizer are not installed properly. "
        "Check local dynin_omni EMOVA runtime module under models/speech/emova_decode_runtime.py."
    ) from e

from .configuration_emova_speech_tokenizer import EMOVASpeechTokenizerConfig

_S2U_IMPORT_ERROR = S2U_IMPORT_ERROR


class EMOVASpeechTokenizer(PreTrainedModel):
    config_class = EMOVASpeechTokenizerConfig
    base_model_prefix = "emova_speech_tokenizer"

    def __init__(self, config: EMOVASpeechTokenizerConfig):
        super().__init__(config)
        self.config = config
        self.s2u_config = None
        self.encoder = None

        if (
            get_S2U_ckpt_config_path is not None
            and load_config is not None
            and VQCTCFinetuneModel is not None
        ):
            _, s2u_config_path = get_S2U_ckpt_config_path(config.s2u_unit_type)
            s2u_cfg = load_config(config=s2u_config_path)
            s2u_cfg.model.pretrain_chkpt_path = None
            self.s2u_config = s2u_cfg.model
            self.encoder = VQCTCFinetuneModel(s2u_cfg.model, trainer=None)

        u2s_config_file, _ = get_U2S_config_checkpoint_file(config.u2s_unit_type)
        u2s_cfg = load_U2S_config(u2s_config_file)
        self.u2s_config = u2s_cfg
        self.decoder = SynthesizerTrn(
            u2s_cfg.num_symbols,
            u2s_cfg.data.filter_length // 2 + 1,
            u2s_cfg.train.segment_size // u2s_cfg.data.hop_length,
            n_speakers=u2s_cfg.data.n_speakers,
            **u2s_cfg.model,
        )
        self.style_embedding = nn.Embedding(config.u2s_num_styles, config.u2s_dim_styles)

    @property
    def device(self):
        if self.encoder is not None:
            return next(self.encoder.parameters()).device
        return next(self.decoder.parameters()).device

    @property
    def dtype(self):
        if self.encoder is not None:
            return next(self.encoder.parameters()).dtype
        return next(self.decoder.parameters()).dtype

    def encode(self, wav_file):
        if self.encoder is None or s2u_extract_unit_demo is None:
            msg = (
                "EMOVASpeechTokenizer encode path is unavailable. "
                "Install EMOVA tokenizer runtime dependencies first "
                "(see vllm_omni/model_executor/models/omada_omni/tokenizers/init_tokenizers.sh)."
            )
            if _S2U_IMPORT_ERROR is not None:
                msg += f" Root cause: {_S2U_IMPORT_ERROR!r}"
            raise RuntimeError(msg)
        speech_unit = s2u_extract_unit_demo(
            self.encoder,
            wav_file,
            model_name="SPIRAL-FSQ-CTC",
            reduced=True,
        )
        unit_numbers = speech_unit.replace("<|speech_", "").replace("|>", " ").strip()
        unit_ids = [int(unit) for unit in unit_numbers.split(" ") if unit]
        return torch.LongTensor(unit_ids).unsqueeze(0)

    def decode(self, speech_unit, condition=None, output_wav_file="output.wav"):
        content_unit = speech_unit.replace("<|speech_", "").replace("|>", " ").strip()
        style_centroid_embedding = None
        if condition:
            if condition not in self.config.u2s_style2idx:
                raise KeyError(f"Unknown speech condition '{condition}'")
            style_centroid_embedding = self.style_embedding(
                torch.tensor([self.config.u2s_style2idx[condition]], dtype=torch.long, device=self.device)
            ).unsqueeze(-1)

        return synthesis(
            content_unit,
            style_centroid_embedding,
            self.u2s_config,
            self.decoder,
            output_wav_file,
        )

    @staticmethod
    def _fix_state_dict_key_on_load(key: str) -> tuple[str, bool]:
        """Force EMOVA checkpoint key normalization for current UVITS modules.

        transformers>=4.57 may bypass model.load_state_dict() during from_pretrained
        (meta loading path), so this hook is the reliable place for key remapping.
        """
        if key.endswith("LayerNorm.beta"):
            return key.replace("LayerNorm.beta", "LayerNorm.bias"), True
        if key.endswith("LayerNorm.gamma"):
            return key.replace("LayerNorm.gamma", "LayerNorm.weight"), True

        if key.endswith("parametrizations.weight.original0"):
            return key.replace("parametrizations.weight.original0", "weight_g"), True
        if key.endswith("parametrizations.weight.original1"):
            return key.replace("parametrizations.weight.original1", "weight_v"), True

        # Keep legacy weight_norm keys as-is for this model.
        # (HF base implementation rewrites them to parametrizations.* when
        # nn.utils.parametrizations.weight_norm exists, which mismatches UVITS.)
        if key.endswith("weight_g") or key.endswith("weight_v"):
            return key, False

        return key, False

    @staticmethod
    def _remap_weight_norm_keys_for_current_model(
        state_dict: dict[str, torch.Tensor],
        model_keys: set[str],
    ) -> dict[str, torch.Tensor]:
        """Normalize weight-norm key style between legacy and parametrization formats.

        - legacy:      *.weight_g / *.weight_v
        - new format:  *.parametrizations.weight.original0 / original1
        """
        remapped: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if ".parametrizations.weight.original0" in key:
                legacy_key = key.replace(
                    ".parametrizations.weight.original0", ".weight_g"
                )
                if legacy_key in model_keys and key not in model_keys:
                    remapped[legacy_key] = value
                    continue
            elif ".parametrizations.weight.original1" in key:
                legacy_key = key.replace(
                    ".parametrizations.weight.original1", ".weight_v"
                )
                if legacy_key in model_keys and key not in model_keys:
                    remapped[legacy_key] = value
                    continue
            elif key.endswith(".weight_g"):
                param_key = key[:-len(".weight_g")] + ".parametrizations.weight.original0"
                if param_key in model_keys and key not in model_keys:
                    remapped[param_key] = value
                    continue
            elif key.endswith(".weight_v"):
                param_key = key[:-len(".weight_v")] + ".parametrizations.weight.original1"
                if param_key in model_keys and key not in model_keys:
                    remapped[param_key] = value
                    continue
            remapped[key] = value
        return remapped

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        model_keys = set(super().state_dict().keys())
        normalized_state_dict = self._remap_weight_norm_keys_for_current_model(
            dict(state_dict),
            model_keys=model_keys,
        )
        if self.encoder is None:
            # Decode-only runtime: drop S2U encoder weights from checkpoint.
            normalized_state_dict = {
                k: v for k, v in normalized_state_dict.items() if not k.startswith("encoder.")
            }
        try:
            return super().load_state_dict(
                normalized_state_dict, strict=strict, assign=assign
            )
        except TypeError:
            return super().load_state_dict(normalized_state_dict, strict=strict)
