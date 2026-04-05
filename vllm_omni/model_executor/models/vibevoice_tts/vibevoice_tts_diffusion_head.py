"""VibeVoice diffusion head: adaLN-modulated FFN blocks conditioned on
timestep + LLM hidden state, with DPM-Solver for fast inference.

Ported from microsoft/VibeVoice modular_vibevoice_diffusion_head.py.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

logger = init_logger(__name__)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        output = output.type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(t.dtype)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class FeedForwardNetwork(nn.Module):
    """SwiGLU feed-forward."""

    def __init__(self, embed_dim: int, ffn_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class HeadLayer(nn.Module):
    """Single adaLN-modulated FFN layer in the diffusion head."""

    def __init__(self, embed_dim: int, ffn_dim: int, cond_dim: int, norm_eps: float = 1e-5) -> None:
        super().__init__()
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)
        self.norm = RMSNorm(embed_dim, eps=norm_eps)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 3 * embed_dim, bias=False),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.adaLN_modulation(c).chunk(3, dim=-1)
        return x + gate * self.ffn(_modulate(self.norm(x), shift, scale))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, cond_size: int, norm_eps: float = 1e-5) -> None:
        super().__init__()
        self.norm_final = RMSNorm(hidden_size, eps=norm_eps, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, output_size, bias=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_size, 2 * hidden_size, bias=False),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        return self.linear(_modulate(self.norm_final(x), shift, scale))


class VibeVoiceDiffusionHead(nn.Module):
    """Diffusion head that predicts noise/velocity from noisy latent + condition.

    Config reference: VibeVoiceDiffusionHeadConfig with fields
    hidden_size, head_layers, head_ffn_ratio, rms_norm_eps, latent_size.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        latent_size = config.latent_size

        self.noisy_images_proj = nn.Linear(latent_size, hidden_size, bias=False)
        self.cond_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.t_embedder = TimestepEmbedder(hidden_size)

        ffn_dim = int(hidden_size * config.head_ffn_ratio)
        self.layers = nn.ModuleList([
            HeadLayer(
                embed_dim=hidden_size,
                ffn_dim=ffn_dim,
                cond_dim=hidden_size,
                norm_eps=config.rms_norm_eps,
            )
            for _ in range(config.head_layers)
        ])
        self.final_layer = FinalLayer(
            hidden_size=hidden_size,
            output_size=latent_size,
            cond_size=hidden_size,
            norm_eps=config.rms_norm_eps,
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for layer in self.layers:
            nn.init.constant_(layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        x = self.noisy_images_proj(noisy_latent)
        t = self.t_embedder(timesteps)
        c = self.cond_proj(condition) + t
        for layer in self.layers:
            x = layer(x, c)
        return self.final_layer(x, c)

    def load_weight(self, weight: tuple[str, torch.Tensor]) -> str:
        params_dict = dict(self.named_parameters())
        name, loaded_weight = weight
        if name not in params_dict:
            logger.warning("Diffusion head weight %s not found (UNUSED)", name)
            return name
        param = params_dict[name]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)
        return name


# ---------------------------------------------------------------------------
# Noise scheduler – minimal DPM-Solver for inference only
# ---------------------------------------------------------------------------

def _cosine_beta_schedule(num_steps: int) -> torch.Tensor:
    """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672."""
    s = 0.008
    steps = torch.arange(num_steps + 1, dtype=torch.float64)
    f_t = torch.cos(((steps / num_steps) + s) / (1 + s) * (math.pi / 2)) ** 2
    alphas_cumprod = f_t / f_t[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0, 0.999).float()


class DPMSolverScheduler:
    """Minimal DPM-Solver++ 2M scheduler for inference.

    Supports prediction_type = "epsilon" or "v_prediction".
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_schedule: str = "cosine",
        prediction_type: str = "v_prediction",
    ) -> None:
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type

        if beta_schedule == "cosine":
            betas = _cosine_beta_schedule(num_train_timesteps)
        else:
            betas = torch.linspace(1e-4, 0.02, num_train_timesteps)

        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.final_alpha_cumprod = torch.tensor(1.0)

        self.timesteps: torch.Tensor | None = None
        self._prev_model_output: torch.Tensor | None = None

    def set_timesteps(self, num_inference_steps: int, device: torch.device | None = None) -> None:
        step_ratio = self.num_train_timesteps / num_inference_steps
        timesteps = (torch.arange(num_inference_steps) * step_ratio).round().long()
        self.timesteps = timesteps.flip(0)
        if device is not None:
            self.timesteps = self.timesteps.to(device)
        self._prev_model_output = None

    def _get_alpha_prod(self, t: int) -> torch.Tensor:
        if t >= 0:
            return self.alphas_cumprod[t]
        return self.final_alpha_cumprod

    def add_noise(
        self,
        original: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        acp = self.alphas_cumprod.to(original.device, original.dtype)
        sqrt_alpha = acp[timesteps].sqrt()
        sqrt_one_minus_alpha = (1 - acp[timesteps]).sqrt()
        while sqrt_alpha.dim() < original.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
        return sqrt_alpha * original + sqrt_one_minus_alpha * noise

    def get_velocity(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        acp = self.alphas_cumprod.to(sample.device, sample.dtype)
        sqrt_alpha = acp[timesteps].sqrt()
        sqrt_one_minus_alpha = (1 - acp[timesteps]).sqrt()
        while sqrt_alpha.dim() < sample.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
        return sqrt_alpha * noise - sqrt_one_minus_alpha * sample

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int | torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        t = int(timestep)
        alpha_prod_t = self._get_alpha_prod(t).to(sample.device, sample.dtype)

        idx = (self.timesteps == t).nonzero(as_tuple=True)[0]
        if idx.numel() > 0 and idx.item() < len(self.timesteps) - 1:
            t_prev = int(self.timesteps[idx.item() + 1])
        else:
            t_prev = -1
        alpha_prod_t_prev = self._get_alpha_prod(t_prev).to(sample.device, sample.dtype)

        sqrt_alpha_t = alpha_prod_t.sqrt()
        sqrt_one_minus_alpha_t = (1 - alpha_prod_t).sqrt()

        if self.prediction_type == "epsilon":
            pred_x0 = (sample - sqrt_one_minus_alpha_t * model_output) / sqrt_alpha_t.clamp(min=1e-8)
        elif self.prediction_type == "v_prediction":
            pred_x0 = sqrt_alpha_t * sample - sqrt_one_minus_alpha_t * model_output
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        sqrt_alpha_prev = alpha_prod_t_prev.sqrt()
        sqrt_one_minus_alpha_prev = (1 - alpha_prod_t_prev).sqrt()
        prev_sample = sqrt_alpha_prev * pred_x0 + sqrt_one_minus_alpha_prev * model_output

        return prev_sample
