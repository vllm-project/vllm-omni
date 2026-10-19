# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Validation for Ming-Image component configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_mlp_config(model_path: str | Path) -> dict[str, Any]:
    path = Path(model_path) / "mlp" / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Ming-Image checkpoint is missing MLP config: {path}")
    data = json.loads(path.read_text())
    required = {
        "diffusion_c_input_dim",
        "diffusion_inner_dim",
        "img_gen_scales",
        "selected_hidden_states_layers",
        "use_identity_mlp",
        "use_learnable_token_condition",
        "use_vlm_directvlm_condition",
    }
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError(f"Ming-Image MLP config is missing keys: {missing}")
    if data["img_gen_scales"] != [16]:
        raise ValueError(f"Only img_gen_scales=[16] is supported, got {data['img_gen_scales']!r}.")
    if data["selected_hidden_states_layers"] != [5, 12, 20]:
        raise ValueError("Ming-Image requires selected hidden layers [5, 12, 20].")
    if not data["use_identity_mlp"]:
        raise NotImplementedError("Ming-Image non-identity diffusion MLP is not supported.")
    if not data["use_learnable_token_condition"] or not data["use_vlm_directvlm_condition"]:
        raise ValueError("Ming-Image requires both query-token and direct-VLM conditioning.")
    return data
