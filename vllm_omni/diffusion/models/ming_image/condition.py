# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ming-Image query and direct-VLM condition projectors."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file

from vllm_omni.diffusion.models.ming_flash_omni.condition_encoder import MingConditionEncoder
from vllm_omni.transformers_utils.configs.ming_flash_omni import MingImageGenConfig

from .component_config import load_mlp_config


class MingImageConditioning(nn.Module):
    def __init__(
        self,
        model_path: str | Path,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        model_path = Path(model_path)
        mlp_config = load_mlp_config(model_path)
        config = MingImageGenConfig(
            diffusion_c_input_dim=int(mlp_config["diffusion_c_input_dim"]),
            thinker_hidden_size=2048,
            img_gen_scales=list(mlp_config["img_gen_scales"]),
            text_encoder_norm=bool(mlp_config.get("text_encoder_norm", False)),
        )
        self.query_encoder = MingConditionEncoder(
            config,
            thinker_hidden_size=2048,
            device=device,
            dtype=dtype,
            normalize_output=bool(mlp_config.get("connector_norm", False)),
            strict_loading=True,
        )
        self.query_encoder.load_from_checkpoint(model_path)

        selected_layers = mlp_config["selected_hidden_states_layers"]
        direct_input_dim = 2048 * len(selected_layers)
        direct_output_dim = int(mlp_config["diffusion_inner_dim"])
        self.direct_projector = nn.Sequential(
            nn.RMSNorm(direct_input_dim, eps=1e-5),
            nn.Linear(direct_input_dim, direct_output_dim, bias=True),
        ).to(device=device, dtype=dtype)
        self._load_direct_weights(model_path / "mlp" / "model.safetensors")

    def _load_direct_weights(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Missing Ming-Image direct projector weights: {path}")
        state = load_file(str(path))
        mapping = {
            "0.weight": "proj_directvlm.0.weight",
            "1.weight": "proj_directvlm.1.weight",
            "1.bias": "proj_directvlm.1.bias",
        }
        own_state = self.direct_projector.state_dict()
        projected: dict[str, torch.Tensor] = {}
        for target, source in mapping.items():
            if source not in state:
                raise ValueError(f"Missing Ming-Image MLP weight {source!r}.")
            if own_state[target].shape != state[source].shape:
                raise ValueError(
                    f"Ming-Image MLP shape mismatch for {source}: "
                    f"expected {tuple(own_state[target].shape)}, got {tuple(state[source].shape)}."
                )
            projected[target] = state[source]
        self.direct_projector.load_state_dict(projected, strict=True)

    def forward(
        self,
        query_hidden_states: torch.Tensor,
        direct_hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.query_encoder(query_hidden_states),
            self.direct_projector(direct_hidden_states),
        )


__all__ = ["MingImageConditioning"]
