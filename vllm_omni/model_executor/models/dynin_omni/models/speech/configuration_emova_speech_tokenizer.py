# coding=utf-8
# Copyright 2024 The EMOVA team and The HuggingFace Inc. team.
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

"""EMOVASpeechTokenizer configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from transformers import PretrainedConfig

__all__ = ["EMOVASpeechTokenizerConfig"]


_EMOVA_RUNTIME_ROOT = Path(__file__).resolve().parent
_EMOVA_CONFIG_ROOT = _EMOVA_RUNTIME_ROOT.parent / "configs" / "speech"
_CONDITION2STYLE_FILE = _EMOVA_CONFIG_ROOT / "condition2style_centroid.txt"
_LEGACY_CONDITION2STYLE_FILE = _EMOVA_RUNTIME_ROOT / "condition2style_centroid.txt"
_U2S_UNIT_CONFIG_RELPATH: Dict[str, str] = {
    "40ms_multilingual_8888_xujing_cosyvoice_FT": "config.json",
}
_DEFAULT_U2S_NUM_STYLES = 126
_DEFAULT_U2S_DIM_STYLES = 256


def _load_style2idx_from_package() -> Dict[str, int]:
    if not _EMOVA_RUNTIME_ROOT.exists():
        return {}
    condition2style_path = _CONDITION2STYLE_FILE
    if not condition2style_path.exists():
        condition2style_path = _LEGACY_CONDITION2STYLE_FILE
    if not condition2style_path.exists():
        return {}

    mapping: Dict[str, int] = {}
    for line in condition2style_path.read_text(encoding="utf-8").splitlines()[1:]:
        if not line:
            continue
        condition = line.split("|", 1)[0]
        if condition not in mapping:
            mapping[condition] = len(mapping)
    return mapping


def _resolve_u2s_runtime_config_path(unit_type: str) -> Optional[Path]:
    rel_path = _U2S_UNIT_CONFIG_RELPATH.get(unit_type)
    if not rel_path:
        return None
    candidate = (_EMOVA_CONFIG_ROOT / rel_path).resolve()
    if candidate.exists():
        return candidate
    return (_EMOVA_RUNTIME_ROOT / rel_path).resolve()


def _infer_style_dim_from_u2s_config(unit_type: str) -> Optional[int]:
    cfg_path = _resolve_u2s_runtime_config_path(unit_type)
    if cfg_path is None or not cfg_path.exists():
        return None
    try:
        cfg_json = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    model_cfg = cfg_json.get("model")
    if not isinstance(model_cfg, dict):
        return None
    style_dim = model_cfg.get("gin_channels")
    if style_dim is None:
        return None
    try:
        parsed = int(style_dim)
        return parsed if parsed > 0 else None
    except Exception:
        return None


class EMOVASpeechTokenizerConfig(PretrainedConfig):
    model_type = "emova_speech_tokenizer"

    def __init__(
        self,
        s2u_unit_type: str = "40ms_multilingual_8888",
        u2s_unit_type: str = "40ms_multilingual_8888_xujing_cosyvoice_FT",
        u2s_num_styles: Optional[int] = None,
        u2s_dim_styles: Optional[int] = None,
        u2s_style2idx: Optional[Dict[str, int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if u2s_style2idx is None:
            u2s_style2idx = _load_style2idx_from_package()

        if u2s_num_styles is None:
            u2s_num_styles = len(u2s_style2idx) if u2s_style2idx else _DEFAULT_U2S_NUM_STYLES

        if u2s_dim_styles is None:
            u2s_dim_styles = (
                _infer_style_dim_from_u2s_config(u2s_unit_type) or _DEFAULT_U2S_DIM_STYLES
            )

        self.s2u_unit_type = s2u_unit_type
        self.u2s_unit_type = u2s_unit_type
        self.u2s_num_styles = u2s_num_styles
        self.u2s_dim_styles = u2s_dim_styles
        self.u2s_style2idx = u2s_style2idx or {}
