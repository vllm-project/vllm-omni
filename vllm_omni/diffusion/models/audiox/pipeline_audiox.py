# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable
from typing import Any, ClassVar

import torch
import torch.nn.functional as F
import torchsde
from diffusers import AutoencoderOobleck
from einops import rearrange
from torch import einsum, nn
from torchvision import transforms
from transformers import AutoConfig, CLIPVisionModelWithProjection, T5EncoderModel, T5TokenizerFast
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.audiox.audiox_transformer import MMDiffusionTransformer
from vllm_omni.diffusion.models.interface import SupportAudioOutput
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.transformers_utils.processors import audiox as _audiox_transforms

_VIDEO_ONLY_TASKS = _audiox_transforms.VIDEO_ONLY_TASKS
_TEXT_VIDEO_TASKS = _audiox_transforms.TEXT_VIDEO_TASKS
_VIDEO_CONDITIONED_TASKS = _audiox_transforms.VIDEO_CONDITIONED_TASKS
_normalize_prompts = _audiox_transforms.normalize_prompts
prepare_audio_reference = _audiox_transforms.prepare_audio_reference
prepare_video_reference = _audiox_transforms.prepare_video_reference

# Polyexponential sigma schedule defaults; mirror upstream AudioX gradio interface
# (``audiox/interface/gradio.py`` ``generate_cond`` defaults: ``sigma_min=0.03``, ``sigma_max=1000``).
_DEFAULT_UPSTREAM_SIGMA_MIN = 0.03
_DEFAULT_UPSTREAM_SIGMA_MAX = 1000.0

logger = init_logger(__name__)


def _load_audiox_bundle_config(model_root: str) -> dict[str, Any]:
    with open(os.path.join(os.path.abspath(model_root), "config.json"), encoding="utf-8") as f:
        return json.load(f)


def _audio_conditioning_input_samples(model_config: dict[str, Any]) -> int:
    configs = model_config["model"]["conditioning"]["configs"]
    cfg = next(c["config"] for c in configs if c["id"] == "audio_prompt")
    return int(cfg["latent_seq_len"]) * int(cfg["pretransform_config"]["config"]["downsampling_ratio"])


def get_audiox_post_process_func(od_config: OmniDiffusionConfig):
    """Convert the pipeline's float audio tensor to a CPU numpy array for serving."""

    def post_process_func(audio: torch.Tensor) -> Any:
        if isinstance(audio, torch.Tensor):
            return audio.detach().cpu().float().numpy()
        return audio

    return post_process_func


class SA_PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class SA_FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        # Dropout p=0 preserves upstream ``net.{2,4}`` state-dict keys so the upstream weights
        # load into the right slots; inference is identical to no dropout.
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(0.0),
        )

    def forward(self, x):
        return self.net(x)


# Manual einsum+softmax only. SDPA/diffusion Attention here degrades conditioning vs upstream.
class SA_Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(0.0),
            )
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=h) for t in qkv)

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class SA_Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        SA_PreNorm(dim, SA_Attention(dim, heads=heads, dim_head=dim_head)),
                        SA_PreNorm(dim, SA_FeedForward(dim, mlp_dim)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


_AUDIOX_OOBLECK_CONFIG = {
    "audio_channels": 2,
    "channel_multiples": [1, 2, 4, 8, 16],
    "decoder_channels": 128,
    "decoder_input_channels": 64,
    "downsampling_ratios": [2, 4, 4, 8, 8],
    "encoder_hidden_size": 128,
    "sampling_rate": 44100,
}


def _build_audiox_oobleck(scaling_factor: float = 1.0) -> AutoencoderOobleck:
    vae = AutoencoderOobleck(**_AUDIOX_OOBLECK_CONFIG)
    vae.audiox_scaling_factor = float(scaling_factor)  # type: ignore[attr-defined]
    return vae.eval().requires_grad_(False)


class AudioVaePromptAdapter(nn.Module):
    def __init__(self, *, cond_dim: int, latent_seq_len: int = 215):
        super().__init__()
        self.pretransform = _build_audiox_oobleck()
        in_ch = int(self.pretransform.config.decoder_input_channels)
        self.proj_features_128 = nn.Linear(latent_seq_len, 128)
        self.proj_out = nn.Linear(in_ch, cond_dim) if in_ch != cond_dim else nn.Identity()

    def forward(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.pretransform.encode(audio, return_dict=True).latent_dist.sample()
        latents = z / float(self.pretransform.audiox_scaling_factor)
        latents = rearrange(self.proj_features_128(latents), "b c s -> b s c")
        latents = self.proj_out(latents)
        ones = torch.ones(latents.shape[0], latents.shape[2], device=latents.device)
        return latents, ones


class _MAFCrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self._num_heads = num_heads
        self._head_dim = dim // num_heads
        self._scale = self._head_dim**-0.5
        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_kv = nn.Linear(dim, dim * 2, bias=True)
        self.to_out = nn.Linear(dim, dim, bias=True)
        nn.init.zeros_(self.to_out.weight)
        if self.to_out.bias is not None:
            nn.init.zeros_(self.to_out.bias)

    def forward(self, experts: torch.Tensor, full_context: torch.Tensor) -> torch.Tensor:
        nh, hd = self._num_heads, self._head_dim
        q = rearrange(self.to_q(experts), "b n (h d) -> b h n d", h=nh, d=hd)
        k, v = (rearrange(t, "b n (h d) -> b h n d", h=nh, d=hd) for t in self.to_kv(full_context).chunk(2, dim=-1))
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False, scale=self._scale)
        return self.to_out(rearrange(out, "b h n d -> b n (h d)"))


class _MAFFusionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        head_dim = dim // num_heads
        self._num_heads = num_heads
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)
        self.self_attn = Attention(
            num_heads=num_heads,
            head_size=head_dim,
            softmax_scale=head_dim**-0.5,
            causal=False,
        )
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        q, k, v = self.to_qkv(h).chunk(3, dim=-1)
        nh = self._num_heads
        q_bsn = rearrange(q, "b n (h d) -> b n h d", h=nh).contiguous()
        k_bsn = rearrange(k, "b n (h d) -> b n h d", h=nh).contiguous()
        v_bsn = rearrange(v, "b n (h d) -> b n h d", h=nh).contiguous()
        out = self.self_attn(q_bsn, k_bsn, v_bsn, attn_metadata=None)
        out = rearrange(out, "b n h d -> b n (h d)")
        out = self.out_proj(out)
        x = x + out
        x = x + self.ff(self.norm2(x))
        return x


class MAF_Block(nn.Module):
    DIM = 768
    MLP_RATIO = 4.0

    def __init__(
        self,
        *,
        dim: int = 768,
        num_experts_per_modality: int = 64,
        num_heads: int = 24,
        num_fusion_layers: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        total_experts = num_experts_per_modality * 3

        self.gating_network = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Linear(dim, 3),
            nn.Sigmoid(),
        )

        self.unified_experts = nn.Parameter(torch.randn(total_experts, dim))

        self.cross_block = _MAFCrossAttentionBlock(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.fusion_blocks = nn.ModuleList(
            [_MAFFusionBlock(dim, num_heads, mlp_ratio) for _ in range(num_fusion_layers)]
        )

        self.norm_v2 = nn.LayerNorm(dim)
        self.norm_t2 = nn.LayerNorm(dim)
        self.norm_a2 = nn.LayerNorm(dim)
        self.bypass_gate_v = nn.Parameter(torch.tensor(-10.0))
        self.bypass_gate_t = nn.Parameter(torch.tensor(-10.0))
        self.bypass_gate_a = nn.Parameter(torch.tensor(-10.0))

    def forward(
        self,
        video_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        audio_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size = video_tokens.shape[0]

        v_global = video_tokens.mean(dim=1)
        t_global = text_tokens.mean(dim=1)
        a_global = audio_tokens.mean(dim=1)

        all_global = torch.cat([v_global, t_global, a_global], dim=1)
        gates = self.gating_network(all_global)
        w_v, w_t, w_a = gates.chunk(3, dim=-1)

        gated_v = video_tokens * w_v.unsqueeze(-1)
        gated_t = text_tokens * w_t.unsqueeze(-1)
        gated_a = audio_tokens * w_a.unsqueeze(-1)

        full_context = torch.cat([gated_v, gated_t, gated_a], dim=1)

        experts = self.unified_experts.unsqueeze(0).expand(batch_size, -1, -1)
        cross_out = self.cross_block(experts, full_context)
        updated_experts = self.norm1(experts + cross_out)

        for blk in self.fusion_blocks:
            updated_experts = blk(updated_experts)

        fused_v_experts, fused_t_experts, fused_a_experts = updated_experts.chunk(3, dim=1)

        refinement_v = fused_v_experts.mean(dim=1)
        refinement_t = fused_t_experts.mean(dim=1)
        refinement_a = fused_a_experts.mean(dim=1)

        alpha_v = torch.sigmoid(self.bypass_gate_v)
        alpha_t = torch.sigmoid(self.bypass_gate_t)
        alpha_a = torch.sigmoid(self.bypass_gate_a)

        final_v = video_tokens + alpha_v * self.norm_v2(refinement_v).unsqueeze(1)
        final_t = text_tokens + alpha_t * self.norm_t2(refinement_t).unsqueeze(1)
        final_a = audio_tokens + alpha_a * self.norm_a2(refinement_a).unsqueeze(1)

        return {
            "video": final_v,
            "text": final_t,
            "audio": final_a,
        }


class _BrownianTreeNoiseSampler:
    """Brownian-tree noise sampler for DPM-Solver++ SDE, ported from k-diffusion.

    Returns a scaled Brownian increment between two sigma levels; the tree is indexed by
    transformed ``sigma`` (here linear, same as k-diffusion's default), so re-querying
    identical (sigma, sigma_next) pairs returns the same noise. Entropy must come from
    the user's seed — otherwise the unseeded global RNG would make sampling non-deterministic.
    """

    def __init__(self, x: torch.Tensor, sigma_min: torch.Tensor, sigma_max: torch.Tensor, entropy: int):
        t0 = torch.as_tensor(sigma_min)
        t1 = torch.as_tensor(sigma_max)
        self._tree = torchsde.BrownianTree(t0, torch.zeros_like(x), t1, entropy=entropy)

    def __call__(self, sigma: torch.Tensor, sigma_next: torch.Tensor) -> torch.Tensor:
        # AudioX denoises from sigma_max → 0, so sigma > sigma_next throughout the loop.
        t0 = torch.as_tensor(sigma_next)
        t1 = torch.as_tensor(sigma)
        return -self._tree(t0, t1) / (t1 - t0).sqrt()


class AudioXPipeline(nn.Module, SupportAudioOutput, DiffusionPipelineProfilerMixin):
    support_audio_output: ClassVar[bool] = True
    audio_sample_rate: ClassVar[int] = 44100
    audio_channels: ClassVar[int] = 2
    _PROFILER_TARGETS: ClassVar[list[str]] = ["diffuse"]
    _CLIP_SYNC_DURATION_SEC: ClassVar[float] = 10.0
    _VIDEO_SYNC_FRAME_COUNT: ClassVar[int] = 240

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        if od_config.model is None:
            raise ValueError(
                "AudioXPipeline requires od_config.model (directory with unified safetensors; "
                "see https://huggingface.co/zhangj1an/AudioX)."
            )

        if os.path.exists(od_config.model):
            self._model_root = os.path.abspath(od_config.model)
        else:
            from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

            self._model_root = download_weights_from_hf_specific(od_config.model, None, ["*"])
        self._model_config = _load_audiox_bundle_config(self._model_root)

        model_config = self._model_config["model"]
        diffusion_config = model_config["diffusion"]

        self.model = MMDiffusionTransformer(**dict(diffusion_config["config"]))

        cond_configs = {c["id"]: c["config"] for c in model_config["conditioning"]["configs"]}
        self.audio_vae_adapter = AudioVaePromptAdapter(
            cond_dim=int(model_config["conditioning"]["cond_dim"]),
            latent_seq_len=int(cond_configs["audio_prompt"]["latent_seq_len"]),
        )

        t5_name = cond_configs["text_prompt"]["t5_model_name"]
        self._t5_max_length = int(cond_configs["text_prompt"]["max_length"])
        self.tokenizer = T5TokenizerFast.from_pretrained(t5_name)
        t5_config = AutoConfig.from_pretrained(t5_name)
        self.text_encoder = T5EncoderModel(t5_config).train(False).requires_grad_(False).to(torch.float16)

        clip_name = cond_configs["video_prompt"]["clip_model_name"]
        clip_config = AutoConfig.from_pretrained(clip_name)
        self.clip_encoder = CLIPVisionModelWithProjection(clip_config.vision_config)
        _CLIP_PATCH_TOKENS, _VIDEO_FPS, _DURATION_SEC, _DIM = 50, 5, 10, 768
        _in_features = _CLIP_PATCH_TOKENS * _VIDEO_FPS * _DURATION_SEC
        self._clip_in_features = _in_features
        self._clip_out_features = 128
        self.clip_proj = nn.Linear(_in_features, self._clip_out_features)
        self.clip_proj_sync = nn.Linear(240, self._clip_out_features)
        self.clip_sync_weight = nn.Parameter(torch.tensor(0.0))
        self.clip_temp_transformer = SA_Transformer(_DIM, depth=4, heads=16, dim_head=64, mlp_dim=_DIM * 4)
        self.clip_temp_pos_embedding = nn.Parameter(torch.randn(1, _VIDEO_FPS * _DURATION_SEC, _DIM))
        self.clip_empty_visual_feat = nn.Parameter(torch.zeros(1, self._clip_out_features, _DIM), requires_grad=False)
        _CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
        _CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
        self._clip_normalize = transforms.Compose([transforms.Normalize(mean=list(_CLIP_MEAN), std=list(_CLIP_STD))])

        self.pretransform = _build_audiox_oobleck(
            scaling_factor=float(model_config["pretransform"].get("scale", 1.0)),
        )

        self.io_channels = model_config["io_channels"]
        self.diffusion_objective = "v"

        gate_type_config = diffusion_config["gate_type_config"]
        self.maf_block = MAF_Block(
            dim=768,
            num_experts_per_modality=int(gate_type_config["num_experts_per_modality"]),
            num_heads=int(gate_type_config["num_heads"]),
            num_fusion_layers=int(gate_type_config["num_fusion_layers"]),
        )

        logger.debug("AudioX model built from %s", self._model_root)

        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=self._model_root,
                subfolder="transformer",
                revision=getattr(od_config, "revision", None),
                prefix="",
            ),
        ]
        sample_rate = int(self._model_config.get("sample_rate", 48000))
        self._sample_rate = sample_rate
        self._sample_size = int(self._model_config.get("sample_size", sample_rate * 10))
        self._target_fps = int(self._model_config.get("video_fps", 5))
        self._audio_conditioning_samples = _audio_conditioning_input_samples(self._model_config)

        self.setup_diffusion_pipeline_profiler(
            profiler_targets=list(self._PROFILER_TARGETS),
            enable_diffusion_pipeline_profiler=od_config.enable_diffusion_pipeline_profiler,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        _legacy_prefix = "conditioner.conditioners.audio_prompt."

        # DiT self-attn QKV: bundle stores weights with last dim (h, d, qkv) interleaved
        # (i.e. Q/K/V rows for the same head sit next to each other), whereas
        # QKVParallelLinear expects stacked [Q|K|V] blocks along the output axis.
        # Reshape once at load time so the parallel loader can consume them.
        nheads = int(self._model_config["model"]["diffusion"]["config"]["num_heads"])
        embed_dim = int(self._model_config["model"]["diffusion"]["config"]["embed_dim"])
        total_heads = embed_dim // nheads
        head_dim = embed_dim // total_heads
        qkv_mid = ".attn.qkv."
        to_kv_mid = ".cross_attn.to_kv."

        def _restack_interleaved(tensor: torch.Tensor, n_slots: int) -> torch.Tensor:
            """Turn bundle's (h, d, slot) interleaved-last-dim layout into stacked [slot|slot|...]."""
            if tensor.dim() == 2:  # weight: (out, in)
                out, inp = tensor.shape
                assert out == n_slots * total_heads * head_dim, (out, n_slots, total_heads, head_dim)
                return (
                    tensor.view(total_heads, head_dim, n_slots, inp).permute(2, 0, 1, 3).reshape(out, inp).contiguous()
                )
            out = tensor.shape[0]
            assert out == n_slots * total_heads * head_dim
            return tensor.view(total_heads, head_dim, n_slots).permute(2, 0, 1).reshape(out).contiguous()

        def _remap(items):
            for name, tensor in items:
                if name.startswith(_legacy_prefix):
                    name = "audio_vae_adapter." + name[len(_legacy_prefix) :]
                if qkv_mid in name and (name.endswith(".weight") or name.endswith(".bias")):
                    tensor = _restack_interleaved(tensor, 3)
                elif to_kv_mid in name and (name.endswith(".weight") or name.endswith(".bias")):
                    tensor = _restack_interleaved(tensor, 2)
                yield name, tensor

        loaded = AutoWeightsLoader(self).load_weights(_remap(weights))

        self.to(torch.float32)
        self.eval().requires_grad_(False)

        return loaded

    def _conditioning_dtype(self) -> torch.dtype:
        p = next(self.model.parameters())
        return p.dtype if p.dtype.is_floating_point else torch.float32

    @staticmethod
    def _normalize_task(task: str | None) -> str | None:
        if task is None:
            return None
        t = str(task).strip().lower()
        return t or None

    @staticmethod
    def _text_for_task(task_norm: str | None, prompt: str) -> str:
        if task_norm in _VIDEO_ONLY_TASKS:
            return ""
        return prompt

    @staticmethod
    def _ensure_text_video_prompts(task_norm: str | None, prompts: list[str]) -> None:
        if task_norm not in _TEXT_VIDEO_TASKS:
            return
        for i, p in enumerate(prompts):
            if not str(p).strip():
                raise ValueError(
                    f"audiox_task={task_norm!r} requires a non-empty text prompt for item {i}; "
                    "use v2a/v2m for video-only generation."
                )

    def _audio_prompt_tensors(
        self,
        *,
        raw_prompts: list[Any],
        extra: dict[str, Any],
        device: torch.device,
        cond_dtype: torch.dtype = torch.float32,
    ) -> list[torch.Tensor]:
        target_len = self._audio_conditioning_samples
        sample_rate = self._sample_rate
        seconds_start = float(extra.get("seconds_start", 0.0))
        seconds_total = float(target_len) / float(sample_rate)
        out: list[torch.Tensor] = []
        for raw in raw_prompts:
            src = extra.get("audio_path")
            if src is None:
                out.append(torch.zeros(2, target_len, device=device, dtype=cond_dtype))
                continue
            wav = prepare_audio_reference(
                src,
                model_sample_rate=sample_rate,
                seconds_start=seconds_start,
                seconds_total=seconds_total,
                device=device,
            )
            out.append(wav.to(dtype=cond_dtype))
        return out

    def _video_feature_tensors(
        self,
        *,
        task_norm: str | None,
        raw_prompts: list[Any],
        extra: dict[str, Any],
        seconds_start: float,
        target_fps: int,
        device: torch.device,
        cond_dtype: torch.dtype = torch.float32,
    ) -> list[torch.Tensor]:
        clip_frames = int(round(self._CLIP_SYNC_DURATION_SEC * target_fps))
        if task_norm not in _VIDEO_CONDITIONED_TASKS:
            empty = torch.zeros(clip_frames, 3, 224, 224, device=device, dtype=cond_dtype)
            return [empty for _ in raw_prompts]

        tensors: list[torch.Tensor] = []
        for _ in raw_prompts:
            src = extra.get("video_path")
            if src is None:
                raise ValueError(f"audiox_task={task_norm!r} requires video input: set extra_args['video_path'].")
            vt = prepare_video_reference(
                src,
                duration=float(self._CLIP_SYNC_DURATION_SEC),
                target_fps=target_fps,
                seek_time=seconds_start,
            )
            tensors.append(vt.to(device=device, dtype=cond_dtype))
        return tensors

    def get_conditioning_inputs(self, conditioning_tensors: dict[str, Any], negative: bool = False) -> dict[str, Any]:
        video_feature, video_mask = conditioning_tensors["video_prompt"]
        text_feature, text_mask = conditioning_tensors["text_prompt"]
        audio_feature, audio_mask = conditioning_tensors["audio_prompt"]

        refined = self.maf_block(text_feature, video_feature, audio_feature)
        fused = torch.cat(list(refined.values()), dim=1)
        masks = torch.cat([video_mask, text_mask, audio_mask], dim=1)

        if negative:
            return {"negative_cross_attn_cond": fused, "negative_cross_attn_mask": masks}
        return {"cross_attn_cond": fused}

    def diffuse(
        self,
        *,
        steps: int,
        guidance_scale: float,
        conditioning_tensors: dict[str, Any],
        negative_conditioning_tensors: dict[str, Any] | None,
        batch_size: int,
        sigma_min: float,
        sigma_max: float,
        generator: torch.Generator,
        cfg_rescale: float,
    ) -> torch.Tensor:
        device = self.device
        model_dtype = next(self.model.parameters()).dtype

        # Match upstream AudioX: disable TF32 matmul + fp16 reduced precision + cudnn benchmark
        # for numerical parity with audiox/inference/generation.py:152-156.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cudnn.benchmark = False

        latent_len = self._sample_size // int(self.pretransform.hop_length)
        noise = torch.randn(
            [batch_size, self.io_channels, latent_len], device=device, generator=generator, dtype=model_dtype
        )

        def _cast(d: dict[str, Any]) -> dict[str, Any]:
            return {k: (v.type(model_dtype) if isinstance(v, torch.Tensor) else v) for k, v in d.items()}

        cond = _cast(self.get_conditioning_inputs(conditioning_tensors))
        neg = (
            _cast(self.get_conditioning_inputs(negative_conditioning_tensors, negative=True))
            if negative_conditioning_tensors is not None
            else {}
        )

        # Inlined k-diffusion VDenoiser + sample_dpmpp_3m_sde, matching upstream AudioX exactly.
        # diffusers' EDMDPMSolverMultistepScheduler uses different v-prediction preconditioning
        # and a different stochastic update rule, which here produces a fixed ~861 Hz resonance
        # in the decoded audio regardless of conditioning.
        def denoise(x_in: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            s2 = sigma * sigma
            c_skip = 1.0 / (s2 + 1.0)
            c_out = -sigma / (s2 + 1.0).sqrt()
            c_in = 1.0 / (s2 + 1.0).sqrt()
            t_cond = sigma.atan() * (2.0 / math.pi)
            v = self.model(
                x_in * c_in,
                t_cond,
                cross_attn_cond=cond["cross_attn_cond"],
                negative_cross_attn_cond=neg.get("negative_cross_attn_cond"),
                negative_cross_attn_mask=neg.get("negative_cross_attn_mask"),
                cfg_scale=guidance_scale,
                scale_phi=cfg_rescale,
            )
            return v * c_out + x_in * c_skip

        ramp = torch.linspace(1.0, 0.0, steps, device=device)
        sigmas = torch.cat(
            [
                torch.exp(ramp * (math.log(sigma_max) - math.log(sigma_min)) + math.log(sigma_min)),
                torch.zeros(1, device=device),
            ]
        )
        x = noise * sigmas[0]

        # Match upstream AudioX: sampler runs under fp16 autocast (see audiox/inference/sampling.py:184).
        with torch.cuda.amp.autocast():
            if steps <= 1:
                s_in = x.new_ones([x.shape[0]])
                sampled = denoise(x, sigmas[0] * s_in)
            else:
                # DPM-Solver++(3M) SDE loop (k-diffusion sample_dpmpp_3m_sde), eta=1.0, s_noise=1.0.
                noise_sampler = _BrownianTreeNoiseSampler(
                    x,
                    sigmas[sigmas > 0].min(),
                    sigmas.max(),
                    entropy=generator.initial_seed(),
                )
                s_in = x.new_ones([x.shape[0]])
                denoised_1 = denoised_2 = None
                h_1 = h_2 = None
                eta = 1.0
                for i in range(len(sigmas) - 1):
                    denoised = denoise(x, sigmas[i] * s_in)
                    if sigmas[i + 1] == 0:
                        x = denoised
                    else:
                        t_, s_ = -sigmas[i].log(), -sigmas[i + 1].log()
                        h = s_ - t_
                        h_eta = h * (eta + 1)
                        x = torch.exp(-h_eta) * x + (-h_eta).expm1().neg() * denoised
                        if h_2 is not None:
                            r0 = h_1 / h
                            r1 = h_2 / h
                            d1_0 = (denoised - denoised_1) / r0
                            d1_1 = (denoised_1 - denoised_2) / r1
                            d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                            d2 = (d1_0 - d1_1) / (r0 + r1)
                            phi_2 = h_eta.neg().expm1() / h_eta + 1
                            phi_3 = phi_2 / h_eta - 0.5
                            x = x + phi_2 * d1 - phi_3 * d2
                        elif h_1 is not None:
                            r = h_1 / h
                            d = (denoised - denoised_1) / r
                            phi_2 = h_eta.neg().expm1() / h_eta + 1
                            x = x + phi_2 * d
                        x = (
                            x
                            + noise_sampler(sigmas[i], sigmas[i + 1])
                            * sigmas[i + 1]
                            * (-2 * h * eta).expm1().neg().sqrt()
                        )
                    denoised_1, denoised_2 = denoised, denoised_1
                    h_1, h_2 = h, h_1
                sampled = x

        vae = self.pretransform.to(device=sampled.device, dtype=torch.float32).eval()
        return vae.decode(sampled.to(torch.float32) * float(vae.audiox_scaling_factor), return_dict=True).sample

    def _encode_text(self, texts: list[str], device: torch.device) -> list[torch.Tensor]:
        self.text_encoder.to(device)
        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self._t5_max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device).to(torch.bool)

        self.text_encoder.eval()
        with torch.no_grad():
            embeddings = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)["last_hidden_state"]

        embeddings = embeddings.float() * attention_mask.unsqueeze(-1).float()
        return [embeddings, attention_mask]

    def _encode_video(self, video_list: list[dict], device: torch.device) -> list[torch.Tensor]:
        self.clip_encoder.to(device).eval()

        video_tensors = [item["video_tensors"] for item in video_list]
        video_sync_frames = torch.cat([item["video_sync_frames"] for item in video_list], dim=0).to(device)

        original_videos = torch.cat(video_tensors, dim=0).to(device)
        batch_size, time_length, _, _, _ = original_videos.size()
        is_zero = torch.all(original_videos == 0, dim=(1, 2, 3, 4))

        frames = original_videos.flatten(0, 1)
        pixel_values = self._clip_normalize(frames).to(device)

        with torch.no_grad():
            outputs = self.clip_encoder(pixel_values=pixel_values)
        hidden = outputs.last_hidden_state
        hidden = rearrange(hidden, "(b t) p d -> (b p) t d", b=batch_size, t=time_length)
        hidden = hidden + self.clip_temp_pos_embedding
        hidden = self.clip_temp_transformer(hidden)
        hidden = rearrange(hidden, "(b p) t d -> b (t p) d", b=batch_size)
        hidden = self.clip_proj(hidden.view(-1, self._clip_in_features))
        hidden = hidden.view(batch_size, self._clip_out_features, -1)

        sync = self.clip_proj_sync(video_sync_frames.view(-1, 240))
        sync = sync.view(batch_size, self._clip_out_features, -1)
        hidden = hidden + self.clip_sync_weight * sync

        empty = self.clip_empty_visual_feat.expand(batch_size, -1, -1)
        hidden = torch.where(is_zero.view(batch_size, 1, 1), empty, hidden)
        return [hidden, torch.ones(batch_size, 1, device=device)]

    def _encode_conditioning_tensors(self, batch_metadata: list[dict[str, Any]]) -> dict[str, Any]:
        device = self.device
        audio = torch.cat([item["audio_prompt"] for item in batch_metadata], dim=0).to(device)
        return {
            "audio_prompt": list(self.audio_vae_adapter(audio)),
            "text_prompt": self._encode_text([item["text_prompt"] for item in batch_metadata], device),
            "video_prompt": self._encode_video([item["video_prompt"] for item in batch_metadata], device),
        }

    def _build_conditioning_batch(
        self,
        *,
        texts: list[str],
        video_tensors_list: list[torch.Tensor],
        audio_prompt_list: list[torch.Tensor],
        sync_features: torch.Tensor,
        seconds_start: float,
        seconds_model: float,
        num_outputs_per_prompt: int,
        task_norm: str | None,
    ) -> list[dict[str, Any]]:
        batch: list[dict[str, Any]] = []
        for i, text in enumerate(texts):
            for _ in range(num_outputs_per_prompt):
                batch.append(
                    {
                        "video_prompt": {
                            "video_tensors": video_tensors_list[i].unsqueeze(0),
                            "video_sync_frames": sync_features,
                        },
                        "text_prompt": self._text_for_task(task_norm, text),
                        "audio_prompt": audio_prompt_list[i].unsqueeze(0),
                        "seconds_start": seconds_start,
                        "seconds_total": seconds_model,
                    }
                )
        return batch

    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        if req.prompts is None or len(req.prompts) == 0:
            raise ValueError("AudioXPipeline requires at least one prompt.")
        normalized_prompts = _normalize_prompts(list(req.prompts))
        prompts = [p["prompt"] for p in normalized_prompts]

        sampling_params = req.sampling_params
        if sampling_params.num_inference_steps is None:
            raise ValueError("AudioXPipeline requires sampling_params.num_inference_steps.")
        num_inference_steps = int(sampling_params.num_inference_steps)
        extra_args = sampling_params.extra_args or {}
        task_norm = self._normalize_task(extra_args.get("audiox_task"))
        self._ensure_text_video_prompts(task_norm, prompts)

        neg: list[str] | None = None
        if not all(p.get("negative_prompt") is None for p in normalized_prompts):
            neg = [str(p.get("negative_prompt") or "") for p in normalized_prompts]

        guidance_scale = float(sampling_params.guidance_scale)
        if sampling_params.num_outputs_per_prompt <= 0:
            raise ValueError("AudioXPipeline requires sampling_params.num_outputs_per_prompt > 0.")
        num_outputs_per_prompt = int(sampling_params.num_outputs_per_prompt)
        batch_size = len(prompts) * num_outputs_per_prompt

        seconds_start = float(extra_args.get("seconds_start", 0.0))
        seconds_model = self._sample_size / self._sample_rate
        seconds_total = float(extra_args.get("seconds_total", seconds_model))
        sigma_min = float(extra_args.get("sigma_min", _DEFAULT_UPSTREAM_SIGMA_MIN))
        sigma_max = float(extra_args.get("sigma_max", _DEFAULT_UPSTREAM_SIGMA_MAX))
        cfg_rescale = float(extra_args.get("cfg_rescale", 0.0))
        device = self.device
        generator = sampling_params.generator
        if generator is None:
            raise ValueError("AudioXPipeline requires sampling_params.generator.")
        target_fps = self._target_fps
        cond_dtype = self._conditioning_dtype()

        sync_features = torch.zeros(1, self._VIDEO_SYNC_FRAME_COUNT, 768, device=device, dtype=cond_dtype)

        audio_prompt_list = self._audio_prompt_tensors(
            raw_prompts=normalized_prompts,
            extra=extra_args,
            device=device,
            cond_dtype=cond_dtype,
        )

        video_tensors_list = self._video_feature_tensors(
            task_norm=task_norm,
            raw_prompts=normalized_prompts,
            extra=extra_args,
            seconds_start=seconds_start,
            target_fps=target_fps,
            device=device,
            cond_dtype=cond_dtype,
        )

        conditioning_batch = self._build_conditioning_batch(
            texts=prompts,
            video_tensors_list=video_tensors_list,
            audio_prompt_list=audio_prompt_list,
            sync_features=sync_features,
            seconds_start=seconds_start,
            seconds_model=seconds_model,
            num_outputs_per_prompt=num_outputs_per_prompt,
            task_norm=task_norm,
        )

        negative_conditioning_batch: list[dict[str, Any]] | None = None
        if neg is not None and guidance_scale > 1.0:
            negative_conditioning_batch = self._build_conditioning_batch(
                texts=neg,
                video_tensors_list=video_tensors_list,
                audio_prompt_list=audio_prompt_list,
                sync_features=sync_features,
                seconds_start=seconds_start,
                seconds_model=seconds_model,
                num_outputs_per_prompt=num_outputs_per_prompt,
                task_norm=task_norm,
            )

        conditioning_tensors = self._encode_conditioning_tensors(conditioning_batch)
        negative_conditioning_tensors: dict[str, Any] | None = None
        if negative_conditioning_batch is not None:
            negative_conditioning_tensors = self._encode_conditioning_tensors(negative_conditioning_batch)

        audio = self.diffuse(
            steps=num_inference_steps,
            guidance_scale=guidance_scale,
            conditioning_tensors=conditioning_tensors,
            negative_conditioning_tensors=negative_conditioning_tensors,
            batch_size=batch_size,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            generator=generator,
            cfg_rescale=cfg_rescale,
        )

        # Trim decoded audio to the requested duration (matches upstream AudioX sample script).
        if 0.0 < seconds_total < seconds_model:
            target_samples = int(seconds_total * self._sample_rate)
            audio = audio[..., :target_samples]

        return DiffusionOutput(
            output=audio,
            custom_output={"audiox_task": task_norm},
            stage_durations=self.stage_durations
            if getattr(self, "enable_diffusion_pipeline_profiler", False)
            else None,
        )
