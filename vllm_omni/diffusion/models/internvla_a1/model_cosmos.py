from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file

from .cosmos_ci_torch import build_cosmos_ci_torch_model

_SAFETENSORS_EXTENSION = ".safetensors"


def is_safetensors_checkpoint(path: str | Path) -> bool:
    return Path(path).suffix.lower() == _SAFETENSORS_EXTENSION


def infer_cosmos_ci_spatial_compression(path: str | Path) -> int:
    normalized = str(path).upper()
    if "CI8X8" in normalized or "8X8" in normalized:
        return 8
    if "CI16X16" in normalized or "16X16" in normalized:
        return 16
    raise ValueError(f"Unable to infer Cosmos CI spatial compression from checkpoint path: {path}")


def _load_safetensors_checkpoint(path: str | Path, device: str) -> dict[str, torch.Tensor]:
    return load_file(str(path), device=device)


def _load_native_cosmos_component(
    checkpoint_filepath: str | Path,
    *,
    component: str,
    device: str,
) -> torch.nn.Module:
    full_model = build_cosmos_ci_torch_model(
        spatial_compression=infer_cosmos_ci_spatial_compression(checkpoint_filepath)
    )
    if component == "encoder":
        model = full_model.encoder_module()
    elif component == "decoder":
        model = full_model.decoder_module()
    else:
        raise ValueError(f"Unsupported Cosmos component: {component}")

    if not is_safetensors_checkpoint(checkpoint_filepath):
        raise ValueError(f"Native Cosmos torch backend expects `.safetensors` checkpoints. Got: {checkpoint_filepath}")
    state_dict = _load_safetensors_checkpoint(checkpoint_filepath, device)
    model.load_state_dict(state_dict, strict=False)
    return model.eval().to(device)


def load_cosmos_component(
    checkpoint_filepath: str | Path,
    *,
    component: str,
    device: str = "cuda",
) -> torch.nn.Module:
    return _load_native_cosmos_component(
        checkpoint_filepath,
        component=component,
        device=device,
    )


class ImageTokenizer(torch.nn.Module):
    def __init__(
        self,
        checkpoint_enc: str,
        checkpoint_dec: str,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self._enc_model = load_cosmos_component(
            checkpoint_enc,
            component="encoder",
            device=device,
        )
        self._dec_model = load_cosmos_component(
            checkpoint_dec,
            component="decoder",
            device=device,
        )

    def decode(self, input_latent: torch.Tensor) -> torch.Tensor:
        original_dtype = input_latent.dtype
        model_dtype = next(self.parameters()).dtype
        output_tensor = self._dec_model(input_latent.to(model_dtype))
        return output_tensor.to(original_dtype)

    def encode(self, input_tensor: torch.Tensor) -> torch.Tensor:
        original_dtype = input_tensor.dtype
        model_dtype = next(self.parameters()).dtype
        output_latent = self._enc_model(input_tensor.to(model_dtype))
        if isinstance(output_latent, torch.Tensor):
            return output_latent.to(original_dtype)
        return output_latent[0].to(original_dtype)
