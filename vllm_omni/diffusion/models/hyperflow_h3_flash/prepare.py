# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Attach HyperFlow H3 Flash onto MiniMaxH3Pipeline (FastH3-style, no separate serve stack)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from vllm.logger import init_logger

from vllm_omni.diffusion.models.hyperflow_h3_flash.checkpoint import (
    HYPERFLOW_BASE_MODEL,
    HyperFlowH3FlashLayout,
    looks_like_hyperflow_h3_flash,
    resolve_hyperflow_layout,
    write_omni_model_index,
)
from vllm_omni.diffusion.models.hyperflow_h3_flash.schedule import (
    DEFAULT_STEPS,
    SUPPORTED_STEPS,
    VIDEO_SHIFT,
    hyperflow_schedule,
)
from vllm_omni.diffusion.sched.sigma_schedule import DMD2SigmaSchedule
from vllm_omni.errors import OmniClientError

logger = init_logger(__name__)

# Official README t2va example prompt (Video-Rebirth/hyperflow-h3-flash).
OFFICIAL_T2VA_PROMPT = "A chef slides a steaming bowl of ramen across a wooden counter; rain taps on the window."


def _symlink_tree(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        return
    os.symlink(src, dest, target_is_directory=src.is_dir())


def materialize_hyperflow_omni_snapshot(layout: HyperFlowH3FlashLayout, *, staging_root: Path | None = None) -> Path:
    """Build a MiniMax-H3-shaped tree: HyperFlow DiT + base TE/VAE, no 31B transformer."""
    root = Path(staging_root or tempfile.mkdtemp(prefix="hyperflow-h3-flash-omni-"))
    partition_dir = root / ("Ref2VA" if layout.partition == "ref2va" else "FL2VA")
    partition_dir.mkdir(parents=True, exist_ok=True)
    write_omni_model_index(layout, partition_dir, steps=DEFAULT_STEPS)
    _symlink_tree(layout.transformer_root, partition_dir / "transformer")
    base = layout.base_partition_root
    for name in ("video_vae", "audio_vae", "text_encoder", "tokenizer", "processor"):
        src = base / name
        if src.exists():
            _symlink_tree(src, partition_dir / name)
    (root / ".hyperflow_h3_flash").write_text(layout.partition + "\n", encoding="utf-8")
    return root


def _apply_hyperflow_tf_model_config(od_config: Any, transformer_config: dict[str, Any]) -> None:
    """Install HyperFlow DiT ``config.json`` into ``od_config.tf_model_config``.

    Omni often resolves ``tf_model_config`` before staging (empty / missing
    ``transformer/config.json`` on the HyperFlow hub layout). Without this,
    ``MiniMaxH3DiTArchConfig`` keeps Omni defaults (``time_embed_dim=2688``)
    and AdaLN fails to load HyperFlow's ``[*, 16]`` weights.
    """
    from vllm_omni.diffusion.data import TransformerConfig

    tf = TransformerConfig.from_dict(dict(transformer_config))
    if hasattr(od_config, "set_tf_model_config"):
        od_config.set_tf_model_config(tf)
    else:
        od_config.tf_model_config = tf
    logger.info(
        "HyperFlow H3 Flash: tf_model_config time_embed_dim=%s (AdaLN / time MLP width)",
        transformer_config.get("time_embed_dim"),
    )


def prepare_hyperflow_on_od_config(od_config: Any) -> HyperFlowH3FlashLayout | None:
    """If ``od_config.model`` is HyperFlow, rewrite it to a staged MiniMax-shaped snapshot.

    Returns the layout when HyperFlow was attached; otherwise ``None``.
    """
    model = str(getattr(od_config, "model", "") or "")
    staged = Path(model)
    if staged.is_dir() and (staged / ".hyperflow_h3_flash").is_file():
        # Already staged (e.g. thin alias re-entered). Still refresh tf config.
        partition = (staged / ".hyperflow_h3_flash").read_text(encoding="utf-8").strip() or "fl2va"
        cfg_path = staged / ("Ref2VA" if partition == "ref2va" else "FL2VA") / "transformer" / "config.json"
        if cfg_path.is_file():
            import json

            _apply_hyperflow_tf_model_config(od_config, json.loads(cfg_path.read_text(encoding="utf-8")))
        return None
    if not looks_like_hyperflow_h3_flash(model):
        return None

    extras = dict(getattr(od_config, "extras", None) or {})
    base_model = extras.get("base_model") or os.environ.get("HYPERFLOW_BASE_MODEL") or HYPERFLOW_BASE_MODEL
    layout = resolve_hyperflow_layout(
        model,
        task_type=getattr(od_config, "task_type", None),
        revision=getattr(od_config, "revision", None),
        base_model=str(base_model),
        base_revision=extras.get("base_revision"),
        load_text_encoder=bool(getattr(getattr(od_config, "model_loaded", None), "text_encoder", True)),
        subfolder=extras.get("hyperflow_subfolder"),
    )
    staging = materialize_hyperflow_omni_snapshot(layout)
    logger.info(
        "HyperFlow H3 Flash: staging MiniMax-shaped layout at %s (DiT=%s, base=%s, partition=%s)",
        staging,
        layout.transformer_root,
        layout.base_partition_root,
        layout.partition,
    )
    od_config.model = str(staging)
    if hasattr(od_config, "task_type"):
        od_config.task_type = layout.partition
    _apply_hyperflow_tf_model_config(od_config, layout.transformer_config)
    extras.update(
        {
            "hyperflow_partition": layout.partition,
            "hyperflow_dropped_blocks": list(layout.dropped_blocks()),
            "hyperflow_gate": layout.hyperflow_gate(),
            "hyperflow_quantization": layout.quantization() or {},
            "base_model": str(base_model),
        }
    )
    od_config.extras = extras
    return layout


def hyperflow_schedule_for_steps(steps: int | None) -> DMD2SigmaSchedule:
    resolved = DEFAULT_STEPS if steps is None else int(steps)
    if resolved not in SUPPORTED_STEPS:
        raise OmniClientError(
            f"HyperFlow H3 Flash requires num_inference_steps in {sorted(SUPPORTED_STEPS)} "
            f"(default {DEFAULT_STEPS}), got {resolved}"
        )
    return hyperflow_schedule(resolved)


def resolve_hyperflow_sigma_positions(sampling: Any) -> tuple[tuple[float, ...], int]:
    if sampling.guidance_scale is not None and float(sampling.guidance_scale) > 1.0:
        raise OmniClientError("HyperFlow H3 Flash disables CFG; guidance_scale must be omitted or <= 1")
    if sampling.timesteps is not None or sampling.sigmas is not None:
        raise OmniClientError("HyperFlow H3 Flash does not support custom timesteps or sigmas")
    schedule = hyperflow_schedule_for_steps(sampling.num_inference_steps)
    return schedule.base_schedule, schedule.num_inference_steps


def apply_hyperflow_runtime_pins(pipeline: Any, layout: HyperFlowH3FlashLayout) -> None:
    """Pin schedule / tasks / shifts after MiniMaxH3Pipeline finishes construction."""
    schedule = hyperflow_schedule(DEFAULT_STEPS)
    pipeline._base_schedule_by_partition = {layout.partition: schedule}
    pipeline.default_video_shift = VIDEO_SHIFT
    pipeline.supported_tasks = frozenset({"ref2va"} if layout.partition == "ref2va" else {"t2va", "fl2va"})
    logger.info(
        "HyperFlow H3 Flash on MiniMaxH3Pipeline: tasks=%s default_steps=%s",
        sorted(pipeline.supported_tasks),
        DEFAULT_STEPS,
    )


__all__ = [
    "OFFICIAL_T2VA_PROMPT",
    "apply_hyperflow_runtime_pins",
    "hyperflow_schedule_for_steps",
    "materialize_hyperflow_omni_snapshot",
    "prepare_hyperflow_on_od_config",
    "resolve_hyperflow_sigma_positions",
]
