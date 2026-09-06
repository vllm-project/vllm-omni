# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""PiD network & sampling config registry for all supported backbones.

All backbones share the same PixDiT_T2I architecture; only the LQ-related
constructor args differ per VAE.

Backbone -> VAE characteristics:
    Qwen-Image:  16ch latent, 8x spatial compression
    Flux1:       16ch latent, 8x spatial compression
    SD3:         16ch latent, 8x spatial compression
    SDXL:         4ch latent, 8x spatial compression
    Flux2:      128ch latent, 16x spatial compression (2x2 patchify -> 32 raw ch)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Shared PixDiT_T2I backbone args (identical across all backbones)
# ---------------------------------------------------------------------------

_SHARED_BACKBONE = dict(
    in_channels=3,
    num_groups=24,
    hidden_size=1536,
    pixel_hidden_size=16,
    pixel_attn_hidden_size=1152,
    pixel_num_groups=16,
    patch_depth=14,
    pixel_depth=2,
    num_text_blocks=4,
    patch_size=16,
    txt_embed_dim=2304,
    txt_max_length=300,
    use_text_rope=True,
    text_rope_theta=10000.0,
    rope_mode="ntk_aware",
    rope_ref_h=2048,
    rope_ref_w=2048,
    repa_encoder_index=6,
)

# ---------------------------------------------------------------------------
# Shared PidNet SR args (same across all backbones, except LQ channels/spacing)
# ---------------------------------------------------------------------------

_SHARED_PID_SR = dict(
    lq_inject_mode="controlnet",
    lq_in_channels=0,
    lq_hidden_dim=1024,
    lq_num_res_blocks=4,
    lq_latent_unpatchify_factor=1,
    lq_conv_padding_mode="replicate",
    lq_gate_type="sigma_aware_per_token",
    lq_interval=2,
    zero_init_lq=True,
    sr_scale=4,
    pit_lq_inject=True,
)


def _make_net_config(
    lq_latent_channels: int,
    latent_spatial_down_factor: int,
    lq_latent_unpatchify_factor: int = 1,
) -> dict:
    """Build a complete net config for a given VAE."""
    cfg = dict(_SHARED_BACKBONE)
    cfg.update(_SHARED_PID_SR)
    cfg.update(
        lq_latent_channels=lq_latent_channels,
        latent_spatial_down_factor=latent_spatial_down_factor,
        lq_latent_unpatchify_factor=lq_latent_unpatchify_factor,
    )
    return cfg


# ---------------------------------------------------------------------------
# Per-backbone net configs
# ---------------------------------------------------------------------------

QWENIMAGE_PID_NET_CONFIG = _make_net_config(lq_latent_channels=16, latent_spatial_down_factor=8)
FLUX_PID_NET_CONFIG = _make_net_config(lq_latent_channels=16, latent_spatial_down_factor=8)
SD3_PID_NET_CONFIG = _make_net_config(lq_latent_channels=16, latent_spatial_down_factor=8)
SDXL_PID_NET_CONFIG = _make_net_config(lq_latent_channels=4, latent_spatial_down_factor=8)
FLUX2_PID_NET_CONFIG = _make_net_config(
    lq_latent_channels=128, latent_spatial_down_factor=16, lq_latent_unpatchify_factor=2
)

# ---------------------------------------------------------------------------
# Sampling config (shared across all distill checkpoints)
# ---------------------------------------------------------------------------

PID_SAMPLING_CONFIG = dict(
    student_sample_steps=4,
    student_sample_type="sde",
    student_t_list=[0.999, 0.866, 0.634, 0.342, 0.0],
    prediction_type="velocity",
    fm_timescale=1000.0,
)

# ---------------------------------------------------------------------------
# Checkpoint registry
# ---------------------------------------------------------------------------
# For each vllm-omni supported backbone, the latest official distilled
# checkpoint published under the ``nvidia/PiD`` HF repo. ``checkpoint_path``
# is the in-repo relative path; a missing ``--pid-checkpoint`` is resolved to
# this path and auto-downloaded at load time (see ``checkpoint.py``).

_PID_HF_REPO = "nvidia/PiD"
_PID_CKPT_ROOT = "checkpoints"

# backbone -> (experiment, in-repo relative path, pid_scale)
PID_CHECKPOINT_REGISTRY: dict[str, tuple[str, str, int]] = {
    "qwenimage": (
        "PiD_v1pt5_res2kto4k_sr4x_official_qwenimage_distill_4step",
        f"{_PID_CKPT_ROOT}/PiD_v1pt5_res2kto4k_sr4x_official_qwenimage_distill_4step/model_ema_bf16.pth",
        4,
    ),
    "flux": (
        "PiD_v1pt5_res2kto4k_sr4x_official_flux_distill_4step",
        f"{_PID_CKPT_ROOT}/PiD_v1pt5_res2kto4k_sr4x_official_flux_distill_4step/model_ema_bf16.pth",
        4,
    ),
    "flux2": (
        "PiD_v1pt5_res2kto4k_sr4x_official_flux2_distill_4step",
        f"{_PID_CKPT_ROOT}/PiD_v1pt5_res2kto4k_sr4x_official_flux2_distill_4step/model_ema_bf16.pth",
        4,
    ),
    "sd3": (
        "PiD_res2kto4k_sr4x_official_sd3_distill_4step",
        f"{_PID_CKPT_ROOT}/PiD_res2kto4k_sr4x_official_sd3_distill_4step/model_ema_bf16.pth",
        4,
    ),
    "sdxl": (
        "PiD_res2kto4k_sr4x_official_sdxl_distill_4step",
        f"{_PID_CKPT_ROOT}/PiD_res2kto4k_sr4x_official_sdxl_distill_4step/model_ema_bf16.pth",
        4,
    ),
}


def get_pid_net_config(backbone: str) -> dict:
    """Return the net config dict for ``backbone``."""
    mapping = {
        "qwenimage": QWENIMAGE_PID_NET_CONFIG,
        "flux": FLUX_PID_NET_CONFIG,
        "sd3": SD3_PID_NET_CONFIG,
        "sdxl": SDXL_PID_NET_CONFIG,
        "flux2": FLUX2_PID_NET_CONFIG,
    }
    if backbone not in mapping:
        raise ValueError(f"Unknown backbone: {backbone}. Choose from {list(mapping.keys())}.")
    return dict(mapping[backbone])
