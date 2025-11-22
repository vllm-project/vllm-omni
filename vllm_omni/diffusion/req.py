from dataclasses import dataclass, field

import torch

@dataclass
class OmniDiffusionRequest:
    request_id: str | None
    prompt: str | list[str] | None = None
    negative_prompt: str | list[str] | None = None
    height: list[int] | int | None = None
    width: list[int] | int | None = None
    num_inference_steps: int = 50
    true_cfg_scale: float = 4.0
    generator: torch.Generator | list[torch.Generator] | None = None

    prompt_embeds: list[torch.Tensor] | torch.Tensor = field(default_factory=list)
    negative_prompt_embeds: list[torch.Tensor] | None = None
    prompt_embeds_mask: list[torch.Tensor] | torch.Tensor | None = None
    prompt_attention_mask: list[torch.Tensor] | None = None
    negative_attention_mask: list[torch.Tensor] | None = None