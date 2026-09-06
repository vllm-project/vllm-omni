# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import logging
from dataclasses import replace as _dc_replace
from typing import Any

import torch
import torch.distributed as dist

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.forward_context import is_forward_context_available, set_forward_context
from vllm_omni.diffusion.pid.config import get_pid_net_config
from vllm_omni.diffusion.pid.decoder import PidDecodeConfig, PidDecoder
from vllm_omni.diffusion.pid.latent_forms import LatentForm, lookup_latent_form

logger = logging.getLogger(__name__)

__all__ = [
    "init_pid_decoder_on",
    "decode_with_pid",
    "maybe_pid_passthrough",
    "PidPassthrough",
    "stepwise_pid_active",
    "decode_stepwise_output",
    "validate_pid_override",
]

# -- module init ------------------------------------------------------------


def init_pid_decoder_on(model: Any, od_config: Any) -> None:
    """Mount a PiD decoder on the pipeline (called from ``registry.initialize_model``).

    The backbone key comes from the LatentForm table (pipelines declare nothing).
    No-op when --pid-enable is off or the family is unregistered.
    """
    config = _resolve_pid_config(od_config)
    if config is None or not config.enabled:
        return

    form = lookup_latent_form(model)
    if form is None:
        logger.warning(
            "PiD decode is enabled but pipeline family %s is not registered in "
            "LATENT_FORMS; PiD is disabled for this model.",
            type(model).__name__,
        )
        return

    decoder = PidDecoder(
        config=config,
        backbone=form.backbone,
        enforce_eager=bool(getattr(od_config, "enforce_eager", False)),
        od_config=od_config,
    )
    decoder.load_weights()
    # nn.Module.__setattr__ registers the submodule; declare it as resident so
    # the offloader never CPU-offloads PiD weights.
    model._pid_decoder = decoder
    model._pid_config = config
    existing = list(getattr(model, "_resident_modules", []))
    if "_pid_decoder" not in existing:
        existing.append("_pid_decoder")
    model._resident_modules = existing


def _resolve_pid_config(od_config: Any) -> PidDecodeConfig | None:
    """Normalize ``od_config.pid_decode`` to a ``PidDecodeConfig`` or None."""
    raw = getattr(od_config, "pid_decode", None)
    if raw is None:
        return None
    if isinstance(raw, PidDecodeConfig):
        return raw
    if isinstance(raw, dict):
        return PidDecodeConfig(**raw)
    raise TypeError(f"pid_decode must be PidDecodeConfig, dict, or None, got {type(raw)!r}")


def _pid_sp_world_size() -> int:
    """Sequence-parallel world size for the PiD net (1 = no SP)."""
    try:
        from vllm_omni.diffusion.distributed.parallel_state import get_sequence_parallel_world_size

        return int(get_sequence_parallel_world_size())
    except Exception:
        return 1


# -- decode core ------------------------------------------------------------

_PID_OVERRIDE_KEYS = ("enabled", "scale", "num_steps", "seed", "degrade_sigma")
_PID_OVERRIDE_TYPES: dict[str, tuple[type, ...]] = {
    "enabled": (bool,),
    "scale": (int,),
    "num_steps": (int,),
    "seed": (int,),
    "degrade_sigma": (int, float),
}


def validate_pid_override(pid_override: Any) -> None:
    if pid_override is None:
        return
    if not isinstance(pid_override, dict):
        raise ValueError(f"pid_decode must be a dict, got {type(pid_override).__name__}")
    for key, value in pid_override.items():
        if key not in _PID_OVERRIDE_KEYS:
            raise ValueError(f"pid_decode has unknown key {key!r}; allowed keys: {sorted(_PID_OVERRIDE_KEYS)}")
        expected = _PID_OVERRIDE_TYPES[key]
        if isinstance(value, bool) and bool not in expected:
            raise ValueError(f"pid_decode.{key} must be {expected[0].__name__}, got {type(value).__name__}")
        if not isinstance(value, expected):
            raise ValueError(f"pid_decode.{key} must be {expected[0].__name__}, got {type(value).__name__}")
    if "scale" in pid_override and pid_override["scale"] < 1:
        raise ValueError(f"pid_decode.scale must be >= 1, got {pid_override['scale']}")
    if "num_steps" in pid_override and pid_override["num_steps"] < 1:
        raise ValueError(f"pid_decode.num_steps must be >= 1, got {pid_override['num_steps']}")


def decode_with_pid(
    decoder: PidDecoder,
    config: PidDecodeConfig,
    lq_latent: torch.Tensor,
    pid_h: int,
    pid_w: int,
    caption: str | None,
    pid_override: dict[str, Any] | None = None,
) -> torch.Tensor | None:
    """Run one PiD super-resolution decode (override / rank gating here).

    Returns the super-resolved RGB tensor ``[B, 3, pid_h*scale, pid_w*scale]``
    in [-1, 1], or None on non-SP non-main ranks (results are not gathered).
    """
    if pid_override is not None:
        overrides = {k: pid_override[k] for k in ("scale", "num_steps", "seed", "degrade_sigma") if k in pid_override}
        if overrides:
            config = _dc_replace(config, **overrides)

    # Without SP, PiD decode runs single-card on global rank 0. With SP the
    # PiD net is sharded across the SP group, so every rank must participate
    # (all-to-all / final gather are collectives).
    if _pid_sp_world_size() <= 1:
        if dist.is_initialized() and dist.get_rank() != 0:
            return None

    if caption is None:
        logger.warning("PiD decode is enabled but no caption was provided; falling back to an empty prompt.")
        caption = ""

    if lq_latent.dim() == 5:
        lq_latent = lq_latent.squeeze(2)

    if _pid_sp_world_size() > 1 and not is_forward_context_available():
        with set_forward_context(omni_diffusion_config=getattr(decoder, "_od_config", None)):
            return decoder.decode(
                lq_latent=lq_latent,
                caption=caption,
                output_size=(int(pid_h * config.scale), int(pid_w * config.scale)),
                degrade_sigma=config.degrade_sigma,
                num_steps=config.num_steps,
                seed=config.seed,
            )

    return decoder.decode(
        lq_latent=lq_latent,
        caption=caption,
        output_size=(int(pid_h * config.scale), int(pid_w * config.scale)),
        degrade_sigma=config.degrade_sigma,
        num_steps=config.num_steps,
        seed=config.seed,
    )


# -- request-mode passthrough ------------------------------------------------


class PidPassthrough:
    """Batch-level PiD passthrough orchestrator (owned by the Runner, lifetime = one forward)."""

    def __init__(self, pipeline: Any, form: LatentForm, decoder: PidDecoder, config: PidDecodeConfig):
        self.pipeline = pipeline
        self.form = form
        self.decoder = decoder
        self.config = config
        self._saved_output_types: list[str | None] = []

    def force_latent_output(self, reqs: list[Any]) -> None:
        """Temporarily force batch output_type to "latent" (originals saved for restore)."""
        self._saved_output_types = []
        for req in reqs:
            self._saved_output_types.append(req.sampling_params.output_type)
            req.sampling_params.output_type = "latent"

    def restore_output_type(self, reqs: list[Any]) -> None:
        for req, saved in zip(reqs, self._saved_output_types):
            req.sampling_params.output_type = saved
        self._saved_output_types = []

    def decode_outputs(self, outputs: list[DiffusionOutput], reqs: list[Any]) -> list[DiffusionOutput]:
        """Replace latent outputs with PiD super-resolved images (per request)."""
        if len(outputs) != len(reqs):
            raise ValueError(
                f"PiD passthrough expects one output per request, got {len(outputs)} outputs / {len(reqs)} reqs"
            )
        for idx, (out, req) in enumerate(zip(outputs, reqs)):
            if out.error is not None:
                continue
            outputs[idx] = self._decode_one(out, req)
        return outputs

    def _decode_one(self, out: DiffusionOutput, req: Any) -> DiffusionOutput:
        sp = req.sampling_params
        x0, pid_h, pid_w = self._to_x0(out.output, sp)
        image = decode_with_pid(
            self.decoder,
            self.config,
            x0,
            pid_h,
            pid_w,
            caption=_prompt_text(req.prompt),
            pid_override=sp.pid_decode,
        )
        if image is None:
            return out  # non-SP non-main rank: keep the latent (not gathered).
        out.output = image
        return out

    def _to_x0(self, latent: torch.Tensor, sp: Any) -> tuple[torch.Tensor, int, int]:
        vsf = int(getattr(self.pipeline, "vae_scale_factor", 8))
        height, width = _resolve_target_size(self.pipeline, sp, vsf)
        x0, pid_h, pid_w = self.form.to_x0(latent, height, width, vsf, pipeline=self.pipeline)
        _validate_x0(x0, pid_h, pid_w, self.form.backbone)
        return x0, pid_h, pid_w


def maybe_pid_passthrough(pipeline: Any, reqs: list[Any], od_config: Any) -> PidPassthrough | None:
    """Decide whether this batch goes through PiD passthrough (None = regular VAE decode).

    Gating rules:
    - Globally disabled: None; an explicit per-request ``pid_decode.enabled=True``
      raises (PiD weights are not lazily loaded; restart with --pid-enable).
    - Family unregistered: warning + None (explicit request raises).
    - Request output_type == "latent" (user actually wants latents): None.
    - Initial latents present (img2img/edit: strength<1 or latents/image_latent
      set): unsupported non-canonical token grids, batch falls back to VAE.
    - Mixed per-request intent (some disabled): whole batch falls back to VAE
      (batch output_type is uniform).
    """
    config = _resolve_pid_config(od_config)
    # Strict per-request validation for every request carrying a pid_decode
    # dict (enabled or not): unknown keys / bad types must fail the request,
    # not be silently filtered.
    for ov in (getattr(req.sampling_params, "pid_decode", None) for req in reqs):
        validate_pid_override(ov)
    overrides = [getattr(req.sampling_params, "pid_decode", None) for req in reqs]
    requested = [ov for ov in overrides if ov is not None and ov.get("enabled") is True]

    decoder = getattr(pipeline, "_pid_decoder", None)
    if decoder is None:
        if requested:
            raise RuntimeError(
                "PiD decode was requested per-request (pid_decode.enabled=True) but the "
                "pipeline was not configured with --pid-enable at startup (or its family is "
                "not registered in LATENT_FORMS). PiD weights are not lazily loaded on "
                "request; restart the service with --pid-enable to enable this feature."
            )
        return None

    form = lookup_latent_form(pipeline)
    if form is None:
        # Already warned at init; fail-loud only on explicit requests.
        if requested:
            raise RuntimeError(
                f"Pipeline family {type(pipeline).__name__} is not registered in "
                "LATENT_FORMS; per-request PiD cannot be served."
            )
        return None

    disabled = [ov for ov in overrides if ov is not None and ov.get("enabled") is False]
    if disabled:
        if requested:
            logger.warning(
                "PiD batch mixes enabled/disabled requests; the batch output_type is "
                "uniform, so the whole batch falls back to VAE decode."
            )
        return None

    for req in reqs:
        sp = req.sampling_params
        if (sp.output_type or "pil") == "latent":
            return None  # the user asked for latents; PiD stays out.
        if _has_initial_latent(sp):
            logger.warning(
                "PiD passthrough does not support requests carrying initial latents "
                "(img2img/edit); batch %s falls back to VAE decode.",
                getattr(req, "request_id", "?"),
            )
            return None

    return PidPassthrough(pipeline, form, decoder, getattr(pipeline, "_pid_config", config))


def _has_initial_latent(sp: Any) -> bool:
    """img2img / edit request detection (non-canonical token grids, unsupported)."""
    if getattr(sp, "latents", None) is not None:
        return True
    if getattr(sp, "image_latent", None) is not None:
        return True
    strength = getattr(sp, "strength", None)
    return strength is not None and strength < 1.0


# -- stepwise (streaming) path ------------------------------------------------


def stepwise_pid_active(pipeline: Any, state: Any) -> bool:
    """Whether stepwise post_decode should use PiD for this state."""
    decoder = getattr(pipeline, "_pid_decoder", None)
    sampling = getattr(state, "sampling", None)
    override = getattr(sampling, "pid_decode", None) if sampling is not None else None
    validate_pid_override(override)
    if decoder is None:
        if override is not None and override.get("enabled") is True:
            raise RuntimeError(
                "PiD decode was requested per-request (pid_decode.enabled=True) but the "
                "pipeline was not configured with --pid-enable at startup. Restart the "
                "service with --pid-enable to enable this feature."
            )
        return False
    if sampling is None:
        return False
    if (getattr(sampling, "output_type", None) or "pil") == "latent":
        return False
    if _has_initial_latent(sampling):
        return False
    if override is not None and override.get("enabled") is False:
        return False
    return True


def decode_stepwise_output(pipeline: Any, state: Any, result: DiffusionOutput) -> DiffusionOutput:
    """Replace a stepwise post_decode latent result with a PiD super-resolved image."""
    decoder = getattr(pipeline, "_pid_decoder", None)
    config = getattr(pipeline, "_pid_config", None)
    form = lookup_latent_form(pipeline)
    if decoder is None or config is None or form is None:
        return result
    sampling = state.sampling
    vsf = int(getattr(pipeline, "vae_scale_factor", 8))
    height, width = _resolve_target_size(pipeline, sampling, vsf)
    x0, pid_h, pid_w = form.to_x0(result.output, height, width, vsf, pipeline=pipeline)
    _validate_x0(x0, pid_h, pid_w, form.backbone)
    image = decode_with_pid(
        decoder,
        config,
        x0,
        pid_h,
        pid_w,
        caption=_prompt_text(getattr(state, "prompt", None)),
        pid_override=getattr(sampling, "pid_decode", None),
    )
    if image is not None:
        result.output = image
    return result


# -- helpers -----------------------------------------------------------------


def _prompt_text(prompt: Any) -> str | None:
    """OmniPromptType (str | dict | None) -> caption text."""
    if prompt is None:
        return None
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, dict):
        return prompt.get("prompt") or ""
    return ""


def _resolve_target_size(pipeline: Any, sp: Any, vae_scale_factor: int) -> tuple[int, int]:
    height = getattr(sp, "height", None)
    width = getattr(sp, "width", None)
    if height is None:
        default = getattr(pipeline, "default_sample_size", None)
        height = int(default * vae_scale_factor) if default else 1024
    if width is None:
        default = getattr(pipeline, "default_sample_size", None)
        width = int(default * vae_scale_factor) if default else 1024
    return int(height), int(width)


def _validate_x0(x0: torch.Tensor, pid_h: int, pid_w: int, backbone: str) -> None:
    """Fail-loud shape validation: x0 must match the backbone net config."""
    net_config = get_pid_net_config(backbone)
    expected_c = net_config["lq_latent_channels"]
    down = net_config["latent_spatial_down_factor"]
    # latent_spatial_down_factor already describes x0's full spatial compression
    # (e.g. flux2's patchified grid is 16x). lq_latent_unpatchify_factor is a
    # channel-level op inside LQProjection2D, not a spatial term.
    if x0.dim() != 4:
        raise ValueError(f"PiD x0 must be 4D [B, C, zH, zW], got shape {tuple(x0.shape)}")
    _, c, z_h, z_w = x0.shape
    if c != expected_c:
        raise ValueError(
            f"PiD x0 channels {c} != backbone '{backbone}' lq_latent_channels {expected_c}; "
            "the LatentForm for this family is inconsistent with the PiD net config."
        )
    if z_h * down != pid_h or z_w * down != pid_w:
        raise ValueError(
            f"PiD x0 grid ({z_h}x{z_w}) inconsistent with LDM size {pid_h}x{pid_w} "
            f"under latent_spatial_down_factor={down}."
        )
