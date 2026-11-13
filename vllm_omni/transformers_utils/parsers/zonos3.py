# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""vLLM config_format=zonos3 parser for the release's params.json layout."""

import json
from pathlib import Path
from typing import Any

from transformers import PretrainedConfig
from vllm.transformers_utils.config import register_config_parser
from vllm.transformers_utils.config_parser_base import ConfigParserBase

from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific
from vllm_omni.transformers_utils.configs.zonos3 import Zonos3Config


@register_config_parser("zonos3")
class Zonos3ConfigParser(ConfigParserBase):
    def parse(
        self,
        model: str | Path,
        trust_remote_code: bool,
        revision: str | None = None,
        code_revision: str | None = None,
        *,
        hf_overrides: dict[str, Any] | None = None,
        cache_dir: str | None = None,
        **kwargs: Any,
    ) -> tuple[dict, PretrainedConfig]:
        del trust_remote_code, code_revision, kwargs
        if hf_overrides is not None and not isinstance(hf_overrides, dict):
            raise ValueError("Zonos3 config path overrides must be a dictionary")
        overrides = hf_overrides or {}
        subdir = overrides.get("zonos3_lm_subdir", "lm/zonos3_mini")
        if not isinstance(subdir, str) or Path(subdir).is_absolute() or ".." in Path(subdir).parts:
            raise ValueError("zonos3_lm_subdir must be a relative release directory")
        root = Path(model)
        if not root.is_dir():
            root = Path(
                download_weights_from_hf_specific(
                    str(model), cache_dir, [f"{subdir}/params.json"], revision=revision, require_all=True
                )
            )
        params_file = root / "params.json" if (root / "params.json").is_file() else root / subdir / "params.json"
        with params_file.open() as handle:
            params = json.load(handle)
        config = Zonos3Config(
            zonos3_params=params, zonos3_lm_subdir=subdir, zonos3_decoder=overrides.get("zonos3_decoder", "keep2_plain")
        )
        return config.to_dict(), config
