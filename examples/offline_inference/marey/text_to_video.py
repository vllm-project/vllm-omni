# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Marey MMDiT text-to-video generation example.

Loads the Marey Flux-30B model from raw opensora-format checkpoints and
generates a video using the vllm_omni MareyTransformer.

Usage (single GPU):
    python text_to_video.py --prompt "A cat walking on the moon"

Usage (tensor parallel across N GPUs):
    torchrun --nproc_per_node=N text_to_video.py --tp N --prompt "A cat walking on the moon"

Examples:
    python text_to_video.py --prompt "A serene lakeside sunrise" --num-frames 64 --height 720 --width 1280
    torchrun --nproc_per_node=8 text_to_video.py --tp 8 --prompt "A cat walking on the moon" --steps 20

Weight paths (defaults point to the distilled checkpoint):
    --transformer-weights  : safetensors checkpoint for the transformer
    --vae-weights          : VAE checkpoint (.ckpt)
    --config               : training config.yaml with model/text_encoder/vae specs
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import safetensors.torch
import torch
import yaml

# ---------------------------------------------------------------------------
# Default weight paths
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = "/app/wlam/models/checkpoints/marey/distilled-0001/config.yaml"
DEFAULT_TRANSFORMER_WEIGHTS = (
    "/app/wlam/models/checkpoints/marey/distilled-0001/epoch0-global_step5000_distilled/ema_inference_ckpt.safetensors"
)
DEFAULT_VAE_WEIGHTS = "/app/wlam/models/checkpoints/marey/vae/epoch_4_step2819000.ckpt"


DEFAULT_NEGATIVE_PROMPT = (
    "<synthetic> <scene cut> gopro, bright, contrast, static, overexposed, bright, "
    "vignette, artifacts, still, noise, texture, scanlines, videogame, 360 camera, "
    "VR, transition, flare, saturation, distorted, warped, wide angle, contrast, "
    "saturated, vibrant, glowing, cross dissolve, texture, videogame, saturation, "
    "cheesy, ugly hands, mutated hands, mutant, disfigured, extra fingers, blown out, "
    "horrible, blurry, worst quality, bad, transition, dissolve, cross-dissolve, melt, "
    "fade in, fade out, wobbly, weird, low quality, plastic, stock footage, video camera, "
    "boring, static"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate video with Marey MMDiT (vllm_omni).")
    parser.add_argument("--prompt", default="A serene lakeside sunrise with mist over the water.", help="Text prompt.")
    parser.add_argument("--negative-prompt", default=None, help="Negative prompt for CFG (default: built-in).")
    parser.add_argument("--no-cfg", action="store_true", help="Disable classifier-free guidance.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to training config.yaml.")
    parser.add_argument(
        "--transformer-weights", default=DEFAULT_TRANSFORMER_WEIGHTS, help="Path to transformer safetensors."
    )
    parser.add_argument("--vae-weights", default=DEFAULT_VAE_WEIGHTS, help="Path to VAE checkpoint.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--height", type=int, default=720, help="Video height.")
    parser.add_argument("--width", type=int, default=1280, help="Video width.")
    parser.add_argument("--num-frames", type=int, default=33, help="Number of output frames.")
    parser.add_argument(
        "--steps", type=int, default=100, help="Number of denoising steps (reference uses 100 for distilled)."
    )
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="CFG guidance scale.")
    parser.add_argument(
        "--use-guidance-schedule",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use warmup/cooldown guidance schedule (matching reference RFLOW scheduler).",
    )
    parser.add_argument(
        "--skip-uncond",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip unconditional prediction when guidance scale is 1.0. "
        "Default: auto-detect from checkpoint path (True for distilled models).",
    )
    parser.add_argument("--warmup-steps", type=int, default=4, help="Number of guidance warmup steps.")
    parser.add_argument("--cooldown-steps", type=int, default=18, help="Number of guidance cooldown steps.")
    parser.add_argument(
        "--guidance-every-n-steps", type=int, default=2, help="Apply guidance every N steps during the middle phase."
    )
    parser.add_argument(
        "--clip-value", type=float, default=10.0, help="Clamp predicted x0 to [-clip, clip]. 0 to disable."
    )
    parser.add_argument(
        "--quality-guidance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable quality guidance (cond=0, uncond=9 for dover/aesthetics scores). Matches reference.",
    )
    parser.add_argument(
        "--flow-shift",
        type=float,
        default=3.0,
        help="Rectified flow shift (default 3.0 matches reference inference CLI).",
    )
    parser.add_argument("--fps", type=int, default=24, help="Output video FPS.")
    parser.add_argument("--output", type=str, default="marey_output.mp4", help="Output video path.")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"], help="Compute dtype.")
    parser.add_argument(
        "--tp", type=int, default=1, help="Tensor parallel size. Use torchrun --nproc_per_node=N for tp > 1."
    )
    parser.add_argument(
        "--diag", action="store_true", help="Enable DIAG / BLOCK_DIAG diagnostic output from the transformer."
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------


def init_distributed(tp_size: int) -> tuple[int, int]:
    """Initialize torch.distributed and model-parallel groups for tensor parallelism.

    For tp=1 without torchrun, sets up a single-process group automatically.
    For tp>1, expects torchrun to have set RANK/WORLD_SIZE/LOCAL_RANK env vars.

    Returns (rank, world_size).
    """
    from vllm_omni.diffusion.distributed import parallel_state as dist_state

    if tp_size > 1 and "RANK" not in os.environ:
        raise RuntimeError(
            f"--tp {tp_size} requires torchrun. Launch with:\n"
            f"  torchrun --nproc_per_node={tp_size} {os.path.basename(__file__)} ..."
        )

    if "RANK" not in os.environ:
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("", 0))
        port = sock.getsockname()[1]
        sock.close()
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    if world_size != tp_size:
        raise RuntimeError(
            f"World size ({world_size}) must equal --tp ({tp_size}). Use: torchrun --nproc_per_node={tp_size} ..."
        )

    dist_state.init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
    )
    dist_state.initialize_model_parallel(tensor_parallel_size=tp_size)

    return rank, world_size


def cleanup_distributed():
    """Destroy distributed groups."""
    from vllm_omni.diffusion.distributed import parallel_state as dist_state

    if dist_state.model_parallel_is_initialized():
        dist_state.destroy_model_parallel()
    dist_state.destroy_distributed_environment()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_yaml_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Text encoding
# ---------------------------------------------------------------------------


def load_text_encoders(config: dict, device: torch.device, dtype: torch.dtype):
    """Load UL2 (T5-like) and CLIP text encoders from config."""
    te_cfg = config.get("text_encoder", {})
    ul2_name = te_cfg.get("ul2_pretrained", "google/ul2")
    clip_name = te_cfg.get("clip_pretrained", "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K")
    byt5_name = te_cfg.get("byt5_pretrained", "google/byt5-large")
    ul2_max_len = te_cfg.get("ul2_max_length", 300)
    clip_max_len = te_cfg.get("clip_max_length", 77)
    byt5_max_len = te_cfg.get("byt5_max_length", 70)

    from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer, T5EncoderModel

    print(f"Loading UL2 text encoder: {ul2_name}")
    ul2_tokenizer = AutoTokenizer.from_pretrained(ul2_name)
    # ul2_model = AutoModel.from_pretrained(ul2_name, torch_dtype=dtype).encoder.to(device).eval()
    ul2_model = T5EncoderModel.from_pretrained(ul2_name, torch_dtype=dtype).to(device).eval()

    print(f"Loading CLIP text encoder: {clip_name}")
    clip_model = CLIPTextModel.from_pretrained(clip_name, torch_dtype=dtype).to(device).eval()
    clip_tokenizer = CLIPTokenizer.from_pretrained(clip_name)

    print(f"Loading ByT5 text encoder: {byt5_name}")
    byt5_tokenizer = AutoTokenizer.from_pretrained(byt5_name)
    byt5_model = T5EncoderModel.from_pretrained(byt5_name, torch_dtype=dtype).to(device).eval()

    return {
        "ul2_tokenizer": ul2_tokenizer,
        "ul2_model": ul2_model,
        "ul2_max_length": ul2_max_len,
        "clip_model": clip_model,
        "clip_tokenizer": clip_tokenizer,
        "clip_max_length": clip_max_len,
        "byt5_tokenizer": byt5_tokenizer,
        "byt5_model": byt5_model,
        "byt5_max_length": byt5_max_len,
    }


def _extract_quotes(text: str) -> str:
    """Extract text between quotes for ByT5 encoding. Returns the full
    prompt if no quotes are found."""
    import re

    matches = re.findall(r'["\u201c\u201d](.*?)["\u201c\u201d]', text)
    return " ".join(matches) if matches else text


def encode_text(
    prompt: str,
    encoders: dict,
    device: torch.device,
    dtype: torch.dtype,
    quote_override: str | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor | None]:
    """Encode text using UL2 (sequence), CLIP (vector), and ByT5 (quotes).

    Returns (seq_cond, seq_cond_masks, vector_cond) where seq_cond_masks
    are boolean attention masks aligned with the sequence embeddings.

    *quote_override*: if provided, ByT5 encodes this text instead of
    extracting quotes from *prompt*.  The reference scheduler reuses the
    positive prompt's quote text for the negative (unconditional) encoding.
    """
    # UL2 encoding (sequence tokens)
    ul2_tokenizer = encoders["ul2_tokenizer"]
    ul2_model = encoders["ul2_model"]
    max_len = encoders["ul2_max_length"]

    inputs = ul2_tokenizer(
        prompt,
        padding="max_length",
        max_length=max_len,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    ul2_mask = inputs["attention_mask"].to(device, torch.bool)
    with torch.no_grad():
        ul2_output = ul2_model(
            input_ids=inputs.input_ids.to(device),
            attention_mask=inputs.attention_mask.to(device),
        )
        ul2_seq = ul2_output.last_hidden_state.to(dtype)  # [1, 300, 4096]

    # CLIP encoding (pooled vector — must use CLIPTextModel.pooler_output,
    # NOT CLIPModel.get_text_features() which applies an extra text_projection)
    clip_model = encoders["clip_model"]
    clip_tokenizer = encoders["clip_tokenizer"]
    clip_max_len = encoders["clip_max_length"]

    clip_inputs = clip_tokenizer(
        [prompt],
        padding="max_length",
        max_length=clip_max_len,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        clip_output = clip_model(input_ids=clip_inputs["input_ids"].to(device))
        vector_cond = clip_output.pooler_output.to(dtype)  # [1, clip_dim]

    # ByT5 encoding (quote / full prompt)
    byt5_tokenizer = encoders["byt5_tokenizer"]
    byt5_model = encoders["byt5_model"]
    byt5_max_len = encoders["byt5_max_length"]

    quote_text = quote_override if quote_override is not None else _extract_quotes(prompt)
    byt5_inputs = byt5_tokenizer(
        quote_text,
        padding="max_length",
        max_length=byt5_max_len,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    byt5_mask = byt5_inputs["attention_mask"].to(device, torch.bool)
    with torch.no_grad():
        byt5_output = byt5_model(
            input_ids=byt5_inputs.input_ids.to(device),
            attention_mask=byt5_inputs.attention_mask.to(device),
        )
        byt5_seq = byt5_output.last_hidden_state.to(dtype)  # [1, 70, 1536]

    seq_cond = [ul2_seq, byt5_seq]
    seq_cond_masks = [ul2_mask, byt5_mask]
    return seq_cond, seq_cond_masks, vector_cond


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------


def _setup_opensora_imports():
    """Prepare sys.modules so opensora VAE can be imported without the full
    opensora dependency tree (datasets, stdit models, dask, wandb, etc.).

    The opensora package lives under ``moonvalley_ai/open_sora`` and depends on
    a sibling ``moonvalley_ai/common`` package.  We add ``moonvalley_ai/`` to
    ``sys.path`` so that ``common`` resolves, and we stub out the heavyweight
    ``opensora.models`` / ``opensora.datasets`` top-level ``__init__`` modules
    that would otherwise pull in every model + data dependency.
    """
    import types

    repo_root = Path(__file__).resolve().parents[4]
    moonvalley_dir = str(repo_root / "moonvalley_ai")
    if moonvalley_dir not in sys.path:
        sys.path.insert(0, moonvalley_dir)

    for mod_name in (
        "opensora.models",
        "opensora.datasets",
        "opensora.datasets.utils",
        "opensora.datasets.video_transforms",
        "opensora.datasets.datasets",
    ):
        if mod_name not in sys.modules:
            stub = types.ModuleType(mod_name)
            stub.__path__ = []
            stub.__package__ = mod_name
            sys.modules[mod_name] = stub

    opensora_models_path = str(repo_root / "moonvalley_ai" / "open_sora" / "opensora" / "models")
    sys.modules["opensora.models"].__path__ = [opensora_models_path]


def load_vae(
    vae_config: dict,
    device: torch.device,
    dtype: torch.dtype,
):
    """Load the opensora spatiotemporal VAE from a .ckpt file.

    *vae_config* should be the ``vae`` section from the training
    ``config.yaml`` (contains cp_path, scaling_factor, bias_factor,
    frame_chunk_len, etc.).

    Falls back to a latent visualisation stub if the full VAE cannot be loaded.
    """
    vae_path = vae_config.get("cp_path", "")
    if not os.path.exists(vae_path):
        print(f"WARNING: VAE checkpoint not found at {vae_path}. Skipping VAE decode.")
        return None

    print(f"Loading VAE from {vae_path}")
    try:
        _setup_opensora_imports()
        from opensora.models.vae.vae_adapters import PretrainedSpatioTemporalVAETokenizer

        vae = PretrainedSpatioTemporalVAETokenizer(
            cp_path=vae_path,
            strict_loading=vae_config.get("strict_loading", False),
            extra_kwargs=vae_config.get("extra_kwargs", {"no_losses": True}),
            scaling_factor=vae_config.get("scaling_factor", 1.0),
            bias_factor=vae_config.get("bias_factor", 0.0),
            frame_chunk_len=vae_config.get("frame_chunk_len"),
            max_batch_size=vae_config.get("max_batch_size"),
            reuse_as_spatial_vae=vae_config.get("reuse_as_spatial_vae", False),
            extra_context_and_drop_strategy=vae_config.get("extra_context_and_drop_strategy", False),
        )
        vae = vae.to(device, dtype).eval()
        print(
            f"Loaded opensora VAE successfully  (out_channels={vae.out_channels}, downsample={vae.downsample_factors})"
        )
        return vae
    except Exception as e:
        print(f"Could not load opensora VAE ({e}). Output will be raw latents.")
        import traceback

        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------


def build_extra_features(
    config: dict,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Build extra feature tensors from config eval values for inference.

    Returns a dict mapping feature names to tensors suitable for passing
    to MareyTransformer.forward(extra_features=...).
    """
    model_cfg = config.get("model", {})
    extra_cfg = model_cfg.get("extra_features_embedders", {})
    features: dict[str, torch.Tensor] = {}
    for feat, params in extra_cfg.items():
        if feat == "fps":
            continue
        ftype = params.get("type", "")
        if ftype == "SizeEmbedder":
            if feat == "ar":
                val = float(height) / float(width)
            else:
                val = 1.0
            features[feat] = torch.tensor([val], device=device, dtype=dtype)
        elif ftype in ("LabelEmbedder", "OrderedEmbedder"):
            eval_val = params.get("eval_value", -1)
            features[feat] = torch.tensor([eval_val], device=device, dtype=torch.long)
    return features


def build_transformer(
    config: dict, in_channels: int, caption_channels: int | list[int], vector_cond_channels: int | None
):
    """Build MareyTransformer from config.yaml model section."""
    from vllm_omni.diffusion.models.marey.marey_transformer import MareyTransformer

    model_cfg = config.get("model", {})
    te_cfg = config.get("text_encoder", {})
    depth = model_cfg.get("depth", 42)
    num_heads = model_cfg.get("num_heads", 40)
    head_dim = model_cfg.get("head_dim", 128)
    hidden_size = model_cfg.get("hidden_size", num_heads * head_dim)
    patch_size = tuple(model_cfg.get("patch_size", [1, 2, 2]))
    depth_single_blocks = model_cfg.get("depth_single_blocks", 28)
    mlp_ratio = model_cfg.get("mlp_ratio", 4.0)
    rope_dim = model_cfg.get("rope_dim", -1)
    rope_channels_ratio = model_cfg.get("rope_channels_ratio", 0.5)
    qk_norm = model_cfg.get("qk_norm", True)
    add_pos_embed_at_every_block = model_cfg.get("add_pos_embed_at_every_block", True)
    learned_pe = model_cfg.get("learned_pe", True)

    ul2_max_length = te_cfg.get("ul2_max_length", 300)
    byt5_max_length = te_cfg.get("byt5_max_length", 0)
    if isinstance(caption_channels, list) and byt5_max_length > 0:
        model_max_length = [ul2_max_length, byt5_max_length]
    else:
        model_max_length = ul2_max_length

    out_channels = model_cfg.get("out_channels", 2 * in_channels)

    extra_features_config = model_cfg.get("extra_features_embedders", None)
    camera_dim = model_cfg.get("camera_dim") if model_cfg.get("sequence_camera_condition", False) else None

    transformer = MareyTransformer(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_size=hidden_size,
        depth=depth,
        depth_single_blocks=depth_single_blocks,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        patch_size=patch_size,
        caption_channels=caption_channels,
        model_max_length=model_max_length,
        vector_cond_channels=vector_cond_channels,
        qk_norm=qk_norm,
        rope_channels_ratio=rope_channels_ratio,
        rope_dim=rope_dim,
        add_pos_embed_at_every_block=add_pos_embed_at_every_block,
        class_dropout_prob=0.0,
        learned_pe=learned_pe,
        extra_features_config=extra_features_config,
        camera_dim=camera_dim,
    )
    return transformer


_CHECKPOINT_KEY_REMAP = [
    ("y_embedder.vector_embedding_0", "y_embedder.vector_embedding"),
]


def load_transformer_weights(
    transformer: torch.nn.Module,
    weights_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Load safetensors weights into the transformer."""
    print(f"Loading transformer weights from {weights_path}")
    state_dict = safetensors.torch.load_file(weights_path)

    remapped_dict: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        remapped = name
        for old, new in _CHECKPOINT_KEY_REMAP:
            if old in remapped:
                remapped = remapped.replace(old, new)
                break
        remapped_dict[remapped] = tensor

    if hasattr(transformer, "load_weights"):
        loaded = transformer.load_weights(remapped_dict.items())
        print(f"Loaded {len(loaded)} weight tensors via load_weights()")
    else:
        missing, unexpected = transformer.load_state_dict(remapped_dict, strict=False)
        if missing:
            print(f"Missing keys: {len(missing)}")
        if unexpected:
            print(f"Unexpected keys: {len(unexpected)}")

    # Verify critical embedder weights were loaded (not random)
    params = dict(transformer.named_parameters())
    fps_key = "extra_features_embedders.fps.mlp.0.weight"
    if fps_key not in params:
        fps_key = "fps_embedder.mlp.0.weight"
    fps_w = params.get(fps_key)
    if fps_w is not None:
        print(f"{fps_key} loaded: std={fps_w.std().item():.6f}")

    transformer.to(device, dtype).eval()
    del state_dict, remapped_dict
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


def create_flow_timesteps(
    num_steps: int,
    shift: float | None = None,
    num_train_timesteps: int = 1000,
    tmin: float = 0.001,
    tmax: float = 1.0,
    teacher_steps: int = 100,
    device: torch.device | None = None,
) -> list[torch.Tensor]:
    """Create rectified-flow timesteps for distilled inference.

    Produces the teacher's schedule sub-sampled to *num_steps* entries.
    Optionally applies timestep shift (only when use_timestep_transform=True in config).

    Matches the reference RFLOW.draw_time + RFlowScheduler.timestep_shift
    computation order exactly (float64 intermediate via .tolist()) to avoid
    floating-point divergence in the DDPM denoising loop.
    """
    nt = num_train_timesteps
    sigmas_f64 = torch.linspace(tmax, tmin, teacher_steps).tolist()
    timesteps_all = [torch.tensor([t * nt], device=device) for t in sigmas_f64]

    if shift is not None and shift > 0:
        for i, t in enumerate(timesteps_all):
            t_norm = t / nt
            timesteps_all[i] = (shift * t_norm / (1.0 + (shift - 1.0) * t_norm)) * nt

    stride = max(1, teacher_steps // num_steps)
    return [timesteps_all[i * stride] for i in range(num_steps)]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def build_guidance_schedule(
    num_steps: int,
    guidance_scale: float,
    warmup_steps: int = 4,
    cooldown_steps: int = 18,
    guidance_every_n_steps: int = 2,
) -> list[float]:
    """Build an oscillating guidance schedule matching the reference RFLOW scheduler.

    During warmup: CFG is always active.
    During middle: CFG oscillates (active every ``guidance_every_n_steps``).
    During cooldown: CFG is off (scale=1.0).
    """
    cooldown_start = num_steps - cooldown_steps

    schedule = []
    for i in range(num_steps):
        if i < warmup_steps:
            schedule.append(guidance_scale)
        elif i >= cooldown_start:
            schedule.append(1.0)
        else:
            middle_idx = i - warmup_steps
            if middle_idx > 0 and (middle_idx - 1) % guidance_every_n_steps == 0:
                schedule.append(guidance_scale)
            else:
                schedule.append(1.0)
    return schedule


@torch.inference_mode()
def generate(
    transformer,
    timesteps: list[torch.Tensor],
    prompt_embeds: torch.Tensor | list[torch.Tensor],
    vector_cond: torch.Tensor | None,
    num_frames: int,
    height: int,
    width: int,
    num_steps: int,
    guidance_scale: float,
    negative_prompt_embeds: torch.Tensor | list[torch.Tensor] | None,
    negative_vector_cond: torch.Tensor | None = None,
    prompt_embeds_mask: torch.Tensor | list[torch.Tensor] | None = None,
    negative_prompt_embeds_mask: torch.Tensor | list[torch.Tensor] | None = None,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.bfloat16,
    generator: torch.Generator | None = None,
    vae_downsample_factors: tuple[int, int, int] = (4, 16, 16),
    clip_value: float | None = None,
    extra_features: dict[str, torch.Tensor] | None = None,
    uncond_extra_features: dict[str, torch.Tensor] | None = None,
    guidance_schedule: list[float] | None = None,
    skip_uncond: bool = False,
) -> torch.Tensor:
    """Run the DDPM flow-matching denoising loop and return latents.

    When *guidance_schedule* is provided it overrides the constant *guidance_scale*
    on a per-step basis.  Combined with *skip_uncond=True* this matches the
    reference RFLOW scheduler behaviour for distilled models: the unconditional
    forward pass is skipped entirely whenever the per-step scale is 1.0.
    """
    vae_scale_factor_temporal = vae_downsample_factors[0]
    vae_scale_factor_spatial = vae_downsample_factors[1]
    num_train_timesteps = 1000

    time_pad = (
        0
        if (num_frames % vae_scale_factor_temporal == 0)
        else (vae_scale_factor_temporal - num_frames % vae_scale_factor_temporal)
    )
    num_latent_frames = (num_frames + time_pad) // vae_scale_factor_temporal

    latent_h = math.ceil(height / vae_scale_factor_spatial)
    latent_w = math.ceil(width / vae_scale_factor_spatial)
    in_channels = transformer.in_channels
    shape = (1, in_channels, num_latent_frames, latent_h, latent_w)

    z_zeros = torch.zeros(shape, device=device, dtype=dtype)
    z = torch.randn_like(z_zeros)
    del z_zeros

    height_t = torch.tensor([height], device=device, dtype=dtype)
    width_t = torch.tensor([width], device=device, dtype=dtype)
    fps_t = torch.tensor([24.0], device=device, dtype=dtype)

    has_neg = negative_prompt_embeds is not None
    is_main = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

    if is_main:
        sched_desc = "scheduled" if guidance_schedule else f"constant scale={guidance_scale}"
        print(
            f"Denoising: {len(timesteps)} steps, latent shape {shape}, "
            f"CFG={'on' if has_neg else 'off'}"
            f"{f' ({sched_desc}, skip_uncond={skip_uncond})' if has_neg else ''}"
        )

    _uncond_ef = uncond_extra_features if uncond_extra_features is not None else extra_features

    def _model_forward(z_in, t_in, text_emb, vec_cond, text_mask=None, ef=None):
        raw = transformer(
            hidden_states=z_in.to(dtype),
            timestep=t_in,
            encoder_hidden_states=text_emb,
            encoder_hidden_states_mask=text_mask,
            vector_cond=vec_cond,
            height=height_t,
            width=width_t,
            fps=fps_t,
            extra_features=ef if ef is not None else extra_features,
            return_dict=False,
        )[0]
        if raw.shape[1] != in_channels:
            raw = raw[:, :in_channels]
        return raw

    for i, t in enumerate(timesteps):
        t_input = t.expand(1)

        # Determine per-step guidance scale from schedule or constant
        gs_i = guidance_schedule[i] if guidance_schedule is not None else guidance_scale

        # Match reference RFLOW logic:
        #   use_uncond = (not skip_uncond) or (guidance_scale_ > 1.0)
        # When skip_uncond=True and scale is 1.0, skip the unconditional pass entirely.
        use_uncond = has_neg and ((not skip_uncond) or (gs_i > 1.0))

        pred_cond = _model_forward(z, t_input, prompt_embeds, vector_cond, prompt_embeds_mask, ef=extra_features)

        if use_uncond:
            pred_uncond = _model_forward(
                z, t_input, negative_prompt_embeds, negative_vector_cond, negative_prompt_embeds_mask, ef=_uncond_ef
            )
            v_pred = pred_uncond + gs_i * (pred_cond - pred_uncond)
        else:
            v_pred = pred_cond

        # DDPM flow-matching step (matches RFLOW reference sampler: ddpm)
        sigma_t = (t / num_train_timesteps).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x0 = z - sigma_t * v_pred
        if clip_value is not None and clip_value > 0:
            x0 = torch.clamp(x0, -clip_value, clip_value)
            v_pred = (z - x0) / sigma_t

        if i < len(timesteps) - 1:
            sigma_s = (timesteps[i + 1] / num_train_timesteps).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            alpha_t = 1.0 - sigma_t
            alpha_s = 1.0 - sigma_s
            alpha_ts = alpha_t / alpha_s
            alpha_ts_sq = alpha_ts**2
            sigma_s_div_t_sq = (sigma_s / sigma_t) ** 2
            sigma_ts_div_t_sq = 1.0 - alpha_ts_sq * sigma_s_div_t_sq

            mean = alpha_ts * sigma_s_div_t_sq * z + alpha_s * sigma_ts_div_t_sq * x0
            variance = sigma_ts_div_t_sq * sigma_s**2

            noise = torch.randn_like(z)
            z = mean + torch.sqrt(variance) * noise
        else:
            z = x0

        if is_main and ((i + 1) % 5 == 0 or i == 0 or i == len(timesteps) - 1):
            zf = z.float()
            print(
                f"  Step {i + 1}/{len(timesteps)}: sigma={sigma_t.item():.4f} "
                f"v_pred std={v_pred.float().std().item():.4f} "
                f"x0 std={x0.float().std().item():.4f} "
                f"z std={zf.std().item():.4f}"
            )

    return z


# ---------------------------------------------------------------------------
# Video saving
# ---------------------------------------------------------------------------


def save_video(frames: np.ndarray, output_path: str, fps: int = 24) -> None:
    """Save frames as MP4 using diffusers export_to_video."""
    try:
        from diffusers.utils import export_to_video
    except ImportError:
        raise ImportError("diffusers is required for export_to_video: pip install diffusers")

    if frames.ndim == 5:
        frames = frames[0]
    if frames.ndim == 4 and frames.shape[-1] not in (1, 3, 4):
        # Likely [T, C, H, W] -> [T, H, W, C]
        frames = np.transpose(frames, (0, 2, 3, 1))

    frame_list = list(frames)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frame_list, output_path, fps=fps)
    print(f"Saved video to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    # Initialize distributed environment and TP groups
    rank, world_size = init_distributed(args.tp)

    import random as _random

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    _random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    generator = None

    # Resolve skip_uncond early so the banner shows the actual value
    skip_uncond = args.skip_uncond
    if skip_uncond is None:
        skip_uncond = "distilled" in args.transformer_weights.lower()

    if rank == 0:
        print(f"\n{'=' * 60}")
        print("Marey MMDiT Video Generation")
        print(f"  Config:     {args.config}")
        print(f"  Weights:    {args.transformer_weights}")
        print(f"  VAE:        {args.vae_weights}")
        print(f"  Prompt:     {args.prompt}")
        print(f"  Resolution: {args.width}x{args.height}, {args.num_frames} frames")
        print(f"  Steps:      {args.steps}, CFG: {args.guidance_scale}")
        print(
            f"  Schedule:   use_guidance_schedule={args.use_guidance_schedule}, "
            f"skip_uncond={skip_uncond}, clip_value={args.clip_value}"
        )
        print(f"  Flow shift: {args.flow_shift}")
        print(f"  Dtype:      {args.dtype}")
        print(f"  TP:         {args.tp}")
        print(f"  Diag:       {args.diag}")
        print(f"{'=' * 60}\n")

    # Load config
    config = load_yaml_config(args.config)

    # Text encoding: rank 0 encodes, then broadcasts embeddings to all ranks
    use_cfg = args.guidance_scale > 1.0 and not args.no_cfg
    negative_prompt_text = args.negative_prompt if args.negative_prompt is not None else DEFAULT_NEGATIVE_PROMPT

    t_enc_load = 0.0
    t_encode = 0.0

    if rank == 0:
        print("Pre-loading VAE")
        vae_cfg = config.get("vae", {})
        vae = load_vae(vae_cfg, device, dtype)
    if rank == 0:
        t0 = time.perf_counter()
        encoders = load_text_encoders(config, device, dtype)
        t_enc_load = time.perf_counter() - t0
        print(f"Text encoders loaded in {t_enc_load:.1f}s")

        t0 = time.perf_counter()
        # Reference passes quote_prompts="" to the scheduler (the line is commented
        # out in marey_inference.py), so ByT5 encodes an empty string.
        prompt_embeds, prompt_masks, vector_cond = encode_text(
            args.prompt,
            encoders,
            device,
            dtype,
            quote_override="",
        )
        t_encode = time.perf_counter() - t0
        seq_shapes = [s.shape for s in prompt_embeds] if isinstance(prompt_embeds, list) else prompt_embeds.shape
        print(f"Prompt encoded in {t_encode:.1f}s  seq={seq_shapes}  vec={vector_cond.shape}")

        negative_prompt_embeds = None
        negative_prompt_masks = None
        negative_vector_cond = None
        if use_cfg:
            negative_prompt_embeds, negative_prompt_masks, negative_vector_cond = encode_text(
                negative_prompt_text,
                encoders,
                device,
                dtype,
                quote_override="",
            )
            print(f"Negative prompt encoded  (CFG scale={args.guidance_scale})")

        del encoders
        torch.cuda.empty_cache()
    else:
        prompt_embeds = None
        prompt_masks = None
        vector_cond = None
        negative_prompt_embeds = None
        negative_prompt_masks = None
        negative_vector_cond = None

    if world_size > 1:
        obj_list = [
            prompt_embeds,
            prompt_masks,
            vector_cond,
            negative_prompt_embeds,
            negative_prompt_masks,
            negative_vector_cond,
        ]
        torch.distributed.broadcast_object_list(obj_list, src=0)
        (
            prompt_embeds,
            prompt_masks,
            vector_cond,
            negative_prompt_embeds,
            negative_prompt_masks,
            negative_vector_cond,
        ) = obj_list
        if isinstance(prompt_embeds, list):
            prompt_embeds = [p.to(device) for p in prompt_embeds]
        else:
            prompt_embeds = prompt_embeds.to(device)
        if prompt_masks is not None:
            if isinstance(prompt_masks, list):
                prompt_masks = [m.to(device) for m in prompt_masks]
            else:
                prompt_masks = prompt_masks.to(device)
        if vector_cond is not None:
            vector_cond = vector_cond.to(device)
        if negative_prompt_embeds is not None:
            if isinstance(negative_prompt_embeds, list):
                negative_prompt_embeds = [n.to(device) for n in negative_prompt_embeds]
            else:
                negative_prompt_embeds = negative_prompt_embeds.to(device)
        if negative_prompt_masks is not None:
            if isinstance(negative_prompt_masks, list):
                negative_prompt_masks = [m.to(device) for m in negative_prompt_masks]
            else:
                negative_prompt_masks = negative_prompt_masks.to(device)
        if negative_vector_cond is not None:
            negative_vector_cond = negative_vector_cond.to(device)

    # Build transformer (all ranks — TP shards weights automatically)
    t0 = time.perf_counter()
    if isinstance(prompt_embeds, list):
        caption_channels = [p.shape[-1] for p in prompt_embeds]
    else:
        caption_channels = prompt_embeds.shape[-1]
    vector_cond_channels = vector_cond.shape[-1] if vector_cond is not None else None

    vae_cfg = config.get("vae", {})
    in_channels = len(vae_cfg.get("scaling_factor", [0] * 16))

    transformer = build_transformer(config, in_channels, caption_channels, vector_cond_channels)
    load_transformer_weights(transformer, args.transformer_weights, device, dtype)

    extra_features = build_extra_features(config, args.height, args.width, device, dtype)
    if rank == 0 and extra_features:
        print(f"Extra features for inference: {list(extra_features.keys())}")

    uncond_extra_features = None
    if args.quality_guidance and use_cfg:
        uncond_extra_features = dict(extra_features)
        for qkey in ("dover_technical", "aesthetics_score_total"):
            if qkey in extra_features:
                uncond_extra_features[qkey] = torch.tensor([9], device=device, dtype=torch.long)
                extra_features[qkey] = torch.tensor([0], device=device, dtype=torch.long)
        if rank == 0:
            print("Quality guidance enabled: cond quality=0, uncond quality=9")

    if rank == 0 and args.diag:
        transformer._diag = True
    t_model = time.perf_counter() - t0
    if rank == 0:
        num_params = sum(p.numel() for p in transformer.parameters()) / 1e9
        print(f"Transformer loaded in {t_model:.1f}s  ({num_params:.1f}B params)")

    # Build flow timesteps from scheduler config
    sched_cfg = config.get("scheduler", {})
    use_ts_transform = sched_cfg.get("use_timestep_transform", False)
    shift_value = sched_cfg.get("shift_value", None)
    if args.flow_shift is not None:
        shift_value = args.flow_shift if args.flow_shift > 0 else None
    elif not use_ts_transform:
        shift_value = None
    use_ts_transform = shift_value is not None
    tmin = sched_cfg.get("tmin", 0.001)
    tmax = sched_cfg.get("tmax", 1.0)
    teacher_steps = sched_cfg.get("num_sampling_steps", 100)
    if rank == 0:
        print(
            f"Timestep schedule: use_transform={use_ts_transform}, shift={shift_value}, "
            f"tmin={tmin}, tmax={tmax}, teacher_steps={teacher_steps}"
        )
    timesteps = create_flow_timesteps(
        num_steps=args.steps,
        shift=shift_value,
        tmin=tmin,
        tmax=tmax,
        teacher_steps=teacher_steps,
        device=device,
    )

    # Read actual VAE downsample factors from the loaded VAE (or config)
    vae_ds = tuple(vae_cfg.get("downsample_factors", (4, 16, 16)))
    if rank == 0:
        print(f"Using VAE downsample factors: {vae_ds}")

    # Build guidance schedule (matches reference RFLOW warmup/cooldown oscillation)
    guidance_sched: list[float] | None = None
    if args.use_guidance_schedule and use_cfg:
        guidance_sched = build_guidance_schedule(
            num_steps=args.steps,
            guidance_scale=args.guidance_scale,
            warmup_steps=args.warmup_steps,
            cooldown_steps=args.cooldown_steps,
            guidance_every_n_steps=args.guidance_every_n_steps,
        )
        if rank == 0:
            active = sum(1 for g in guidance_sched if g > 1.0)
            print(
                f"Guidance schedule: {active}/{len(guidance_sched)} steps active "
                f"(warmup_steps={args.warmup_steps}, cooldown_steps={args.cooldown_steps}, "
                f"every_n={args.guidance_every_n_steps})"
            )

    clip_val = args.clip_value if args.clip_value > 0 else None

    # Re-seed immediately before generation so the first torch.randn_like
    # call (initial noise) matches the reference, where models are loaded in
    # the constructor and _set_seed(42) is called at the start of each infer()
    # with no RNG-consuming operations before noise generation.
    _random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Generate (all ranks participate in TP forward passes)
    t0 = time.perf_counter()
    latents = generate(
        transformer=transformer,
        timesteps=timesteps,
        prompt_embeds=prompt_embeds,
        vector_cond=vector_cond,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        num_steps=args.steps,
        guidance_scale=args.guidance_scale,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_vector_cond=negative_vector_cond,
        prompt_embeds_mask=prompt_masks,
        negative_prompt_embeds_mask=negative_prompt_masks,
        device=device,
        dtype=dtype,
        generator=generator,
        vae_downsample_factors=vae_ds,
        clip_value=clip_val,
        extra_features=extra_features,
        uncond_extra_features=uncond_extra_features,
        guidance_schedule=guidance_sched,
        skip_uncond=skip_uncond,
    )
    t_gen = time.perf_counter() - t0
    if rank == 0:
        print(f"Generation completed in {t_gen:.1f}s")

    # VAE decode and save (rank 0 only)
    del transformer
    torch.cuda.empty_cache()

    if rank == 0:
        lf = latents.float()
        print(
            f"Final latents: shape={list(latents.shape)} "
            f"mean={lf.mean().item():.4f} std={lf.std().item():.4f} "
            f"min={lf.min().item():.4f} max={lf.max().item():.4f}"
        )
        for ch in range(min(latents.shape[1], 4)):
            print(f"  ch{ch}: mean={lf[0, ch].mean().item():.4f} std={lf[0, ch].std().item():.4f}")

        latent_path = str(Path(args.output).with_suffix(".latents.pt"))
        torch.save(latents.float().cpu(), latent_path)
        print(f"Saved latents to {latent_path}")

        vae = load_vae(vae_cfg, device, dtype)
        if vae is not None:
            t0 = time.perf_counter()

            # The VAE's decode expects pixel-space num_frames aligned to
            # frame_chunk_len.  Compute it from the latent temporal dim.
            vae_t_ds = vae.downsample_factors[0]
            num_latent_t = latents.shape[2]
            num_pixel_frames = num_latent_t * vae_t_ds
            chunk = vae.frame_chunk_len
            if num_latent_t <= 1:
                # Single latent frame → use the VAE's single-frame decode path
                num_pixel_frames = 1
                print("Single latent frame detected, using single-frame decode")
            elif chunk is not None and num_pixel_frames % chunk != 0:
                num_pixel_frames = (num_pixel_frames // chunk) * chunk
                print(f"Aligned pixel frames to {num_pixel_frames} (frame_chunk_len={chunk})")

            with torch.no_grad():
                video = vae.decode(
                    latents.to(dtype),
                    num_frames=num_pixel_frames,
                    spatial_size=(args.height, args.width),
                )
            if isinstance(video, tuple):
                video = video[0]

            t_vae = time.perf_counter() - t0
            print(f"VAE decoded in {t_vae:.1f}s")

            if isinstance(video, torch.Tensor):
                vf = video.float()
                print(
                    f"VAE raw output: shape={list(vf.shape)} "
                    f"mean={vf.mean().item():.4f} std={vf.std().item():.4f} "
                    f"min={vf.min().item():.4f} max={vf.max().item():.4f}"
                )
                video = vf.cpu()
                if video.dim() == 5 and video.shape[1] in (3, 4):
                    video = video[0].permute(1, 2, 3, 0)  # [T, H, W, C]
                video = video.clamp(-1, 1) * 0.5 + 0.5
                print(f"After clamp+rescale: mean={video.mean().item():.4f} std={video.std().item():.4f}")
                video = video.numpy()
        else:
            print("No VAE available. Saving latent visualization.")
            vis = latents[0, :3].float().cpu()
            vis = (vis - vis.min()) / (vis.max() - vis.min() + 1e-8)
            video = vis.permute(1, 2, 3, 0).numpy()  # [T, H, W, 3]

        save_video(video, args.output, fps=args.fps)

        print(f"\nTotal time: {t_enc_load + t_encode + t_model + t_gen:.1f}s")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
