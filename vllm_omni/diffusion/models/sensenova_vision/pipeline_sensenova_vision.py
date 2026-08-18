# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SenseNova-Vision-7B-MoT diffusion pipeline.

SenseNova-Vision is a Bagel fork; the denoising model, VAE, and ViT are
weight-compatible with the BAGEL integration.  This pipeline subclasses
:class:`BagelPipeline` and overrides only the SenseNovaVision checkpoint defaults:

- ``max_latent_size=64`` (BAGEL ships 32)
- ``vit_max_num_patch_per_side=70``
- VAE image transform ``ImageTransform(1024, 512, 16)``
- ViT image transform ``ImageTransform(980, 224, 14)``
- per-mode ``BASE_PARAMS`` from
  ``SenseNova-Vision/inference/sensenova_vision.py``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.bagel.pipeline_bagel import BagelPipeline
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch


@dataclass
class SenseNovaVisionGenParams:
    """SenseNovaVision per-mode generation parameters (BASE_PARAMS equivalents)."""

    num_timesteps: int = 50
    timestep_shift: float = 3.0
    cfg_text_scale: float = 4.0
    cfg_img_scale: float = 1.0
    cfg_interval: tuple = (0.4, 1.0)
    cfg_renorm_min: float = 1.0
    cfg_renorm_type: str = "global"
    # SenseNovaVision-specific additive flags (not consumed by the BAGEL core).
    think: bool = False
    caption: bool = False
    understanding_output: bool = False
    max_think_token_n: int = 1000
    do_sample: bool = False
    text_temperature: float = 0.3
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_base_params(cls, mode: str) -> SenseNovaVisionGenParams:
        """Build defaults for a SenseNovaVision inference mode.

        Mirrors ``BASE_PARAMS`` in
        ``SenseNova-Vision/inference/sensenova_vision.py``.
        """
        base = dict(_BASE_PARAMS.get(mode, {}))
        cfg_interval = base.pop("cfg_interval", (0.4, 1.0))
        if not isinstance(cfg_interval, (tuple, list)) or len(cfg_interval) != 2:
            cfg_interval = (0.4, 1.0)
        cfg_interval = (float(cfg_interval[0]), float(cfg_interval[1]))
        return cls(
            num_timesteps=int(base.pop("num_timesteps", 50)),
            timestep_shift=float(base.pop("timestep_shift", 3.0)),
            cfg_text_scale=float(base.pop("cfg_text_scale", 4.0)),
            cfg_img_scale=float(base.pop("cfg_img_scale", 1.0)),
            cfg_interval=cfg_interval,
            cfg_renorm_min=float(base.pop("cfg_renorm_min", 1.0)),
            cfg_renorm_type=str(base.pop("cfg_renorm_type", "global")),
            think=bool(base.pop("think", False)),
            caption=bool(base.pop("caption", False)),
            understanding_output=bool(base.pop("understanding_output", False)),
            max_think_token_n=int(base.pop("max_think_token_n", 1000)),
            do_sample=bool(base.pop("do_sample", False)),
            text_temperature=float(base.pop("text_temperature", 0.3)),
            extra=base,
        )


# Per-mode defaults, transcribed verbatim from
# SenseNova-Vision/inference/sensenova_vision.py (BASE_PARAMS).
_BASE_PARAMS: dict[str, dict[str, Any]] = {
    "generate": {
        "cfg_text_scale": 4.0,
        "cfg_img_scale": 1.0,
        "cfg_interval": [0.4, 1.0],
        "timestep_shift": 3.0,
        "num_timesteps": 50,
        "cfg_renorm_min": 1.0,
        "cfg_renorm_type": "global",
    },
    "think_generate": {
        "max_think_token_n": 1000,
        "do_sample": False,
        "cfg_text_scale": 4.0,
        "cfg_img_scale": 1.0,
        "cfg_interval": [0.4, 1.0],
        "timestep_shift": 3.0,
        "num_timesteps": 50,
        "cfg_renorm_min": 1.0,
        "cfg_renorm_type": "global",
        "think": True,
    },
    "caption_generate": {
        "max_think_token_n": 8192,
        "do_sample": False,
        "cfg_text_scale": 4.0,
        "cfg_img_scale": 1.0,
        "cfg_interval": [0.0, 1.0],
        "timestep_shift": 4.0,
        "num_timesteps": 50,
        "cfg_renorm_min": 1.0,
        "cfg_renorm_type": "global",
        "caption": True,
    },
    "dense_perception": {
        "cfg_text_scale": 4.0,
        "cfg_img_scale": 1.0,
        "cfg_interval": [0.0, 1.0],
        "timestep_shift": 4.0,
        "num_timesteps": 50,
        "cfg_renorm_min": 1.0,
        "cfg_renorm_type": "text_channel",
    },
    "edit": {
        "cfg_text_scale": 4.0,
        "cfg_img_scale": 2.0,
        "cfg_interval": [0.0, 1.0],
        "timestep_shift": 4.0,
        "num_timesteps": 50,
        "cfg_renorm_min": 1.0,
        "cfg_renorm_type": "text_channel",
    },
    "think_edit": {
        "max_think_token_n": 1000,
        "do_sample": False,
        "cfg_text_scale": 4.0,
        "cfg_img_scale": 2.0,
        "cfg_interval": [0.4, 1.0],
        "timestep_shift": 3.0,
        "num_timesteps": 50,
        "cfg_renorm_min": 0.0,
        "cfg_renorm_type": "text_channel",
        "think": True,
    },
    "understanding": {
        "max_think_token_n": 8192,
        "do_sample": False,
        "understanding_output": True,
    },
    "think_understanding": {
        "max_think_token_n": 8192,
        "do_sample": False,
        "understanding_output": True,
        "think": True,
    },
    "dense_detection": {
        "max_think_token_n": 8192,
        "do_sample": False,
        "understanding_output": True,
    },
    "dense_OCR": {
        "max_think_token_n": 20000,
        "do_sample": False,
        "understanding_output": True,
    },
    "recon3d": {
        "cfg_text_scale": 1.0,
        "cfg_img_scale": 1.0,
        "cfg_interval": [0.0, 1.0],
        "timestep_shift": 4.0,
        "num_timesteps": 50,
        "cfg_renorm_min": 1.0,
        "cfg_renorm_type": "text_channel",
    },
}


def get_sensenova_vision_post_process_func(od_config: OmniDiffusionConfig):
    """SenseNovaVision post-processing: pipelines return PIL images directly."""
    del od_config  # unused

    def post_process_func(x):
        return x

    return post_process_func


def build_sensenova_vision_diffusion_output(
    *,
    text: str | None = None,
    image: Any = None,
    think_text: str | None = None,
    stage_durations: dict[str, float] | None = None,
) -> DiffusionOutput:
    """Build a canonical SenseNovaVision ``DiffusionOutput`` envelope.

    Mirrors the envelope contract used by :class:`BagelPipeline`:

    * text-only   -> ``{"payload": {"text": ...}, "metadata": {"text": {"text_output": ...}}}``
    * image-only  -> ``{"payload": {"image": ...}, "metadata": {...}}``
    * mixed       -> ``{"payload": {"text": ..., "image": ...}, "metadata": {...}}``

    ``text``/``image`` are raw producer values (a decoded ``str`` caption and a
    ``PIL.Image`` respectively). ``think_text`` is any reasoning/caption string
    produced alongside the image (e.g. ``think_generate``) and is recorded under
    the shared ``text`` metadata group. This is deliberately generic: it carries
    only the existing ``TEXT``/``IMAGE`` output-modality contract and never
    introduces a SenseNovaVision-specific modality key.
    """
    payload: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    text_meta: dict[str, str] = {}

    if text is not None:
        payload["text"] = text
        text_meta["text_output"] = text
    if think_text is not None:
        text_meta["think_text"] = think_text
    if text_meta:
        metadata["text"] = text_meta
    if image is not None:
        payload["image"] = image

    return DiffusionOutput(
        output={
            "payload": payload,
            "metadata": metadata,
        },
        stage_durations=stage_durations,
    )


class SenseNovaVisionPipeline(BagelPipeline):
    """SenseNova-Vision-7B-MoT diffusion pipeline.

    Subclasses :class:`BagelPipeline` and reuses the entire BAGEL weight
    loading / denoising machinery.  Only the SenseNovaVision checkpoint defaults
    differ; these are applied in :meth:`__init__` and per-request mode
    defaults are applied in :meth:`forward` via ``extra_args``.
    """

    # SenseNovaVision checkpoint overrides applied on top of BAGEL.
    _sensenova_vision_max_latent_size = 64
    _sensenova_vision_vit_max_num_patch_per_side = 70
    # Official SenseNovaVision image transforms (vae, vit).
    _sensenova_vision_vae_transform = (1024, 512, 16)
    _sensenova_vision_vit_transform = (980, 224, 14)

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        self._apply_sensenova_vision_defaults()

    def _apply_sensenova_vision_defaults(self) -> None:
        """Force SenseNovaVision defaults on the loaded Bagel core after init."""
        bagel = self.bagel
        bagel.max_latent_size = self._sensenova_vision_max_latent_size
        if hasattr(bagel.config, "max_latent_size"):
            bagel.config.max_latent_size = self._sensenova_vision_max_latent_size
        if hasattr(bagel.config, "vit_max_num_patch_per_side"):
            bagel.config.vit_max_num_patch_per_side = self._sensenova_vision_vit_max_num_patch_per_side
        if hasattr(bagel.latent_pos_embed, "max_num_patch_per_side"):
            bagel.latent_pos_embed.max_num_patch_per_side = self._sensenova_vision_max_latent_size

    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        """Run SenseNovaVision image/text generation with per-mode defaults."""
        self._apply_mode_defaults(req)
        output = super().forward(req)
        return self._merge_mixed_task_text(req, output)

    def _merge_mixed_task_text(self, req: DiffusionRequestBatch, output: DiffusionOutput) -> DiffusionOutput:
        """Lift an available caption/think string into the mixed text+image payload.

        The BAGEL core returns image generations as ``payload["image"]`` and, for
        thinking modes, records the generated text only under
        ``metadata["text"]["think_text"]``. SenseNovaVision mixed tasks (e.g.
        ``caption_generate``/``think_generate``) must represent both under the
        existing ``TEXT | IMAGE`` output-modality contract, so when an image
        payload carries an available text it is also exposed as
        ``payload["text"]``. Additive only: outputs that already carry a text
        payload, or that have no available text, are returned unchanged.
        """
        raw_output = output.output
        if not isinstance(raw_output, dict):
            return output
        payload = raw_output.get("payload")
        if not isinstance(payload, dict) or "image" not in payload or "text" in payload:
            return output

        text = None
        text_meta = raw_output.get("metadata") or {}
        if isinstance(text_meta, dict):
            text_group = text_meta.get("text")
            if isinstance(text_group, dict):
                text = text_group.get("text_output") or text_group.get("think_text")
        if text is None:
            extra_args = getattr(req.sampling_params, "extra_args", None) or {}
            text = extra_args.get("text_output") or extra_args.get("caption")
        if text is None:
            return output

        payload["text"] = text
        text_meta = raw_output.setdefault("metadata", {})
        text_group = text_meta.get("text")
        if not isinstance(text_group, dict):
            text_meta["text"] = text_group = {}
        text_group.setdefault("text_output", text)
        return output

    def _apply_mode_defaults(self, req: DiffusionRequestBatch) -> None:
        """Inject SenseNovaVision ``BASE_PARAMS`` defaults into sampling params.

        The BAGEL core reads CFG/timestep knobs from ``extra_args``; SenseNovaVision
        defines those defaults per mode.  User-supplied values always win.
        """
        if not req.requests:
            return
        params = req.requests[0].sampling_params
        if params is None:
            return

        mode = None
        prompt = req.prompts[0] if req.prompts else None
        if isinstance(prompt, dict):
            mode = prompt.get("mode") or prompt.get("sensenova_vision_mode")

        if not mode:
            # No explicit mode: text-only output uses understanding-style
            # defaults, otherwise use "generate" image defaults.
            modalities = prompt.get("modalities", []) if isinstance(prompt, dict) else []
            if "text" in modalities and "image" not in modalities:
                mode = "understanding"
            else:
                mode = "generate"

        defaults = SenseNovaVisionGenParams.from_base_params(mode)
        extra_args = dict(getattr(params, "extra_args", None) or {})

        # Only fill unset knobs; explicit user args win.
        extra_args.setdefault("cfg_text_scale", defaults.cfg_text_scale)
        extra_args.setdefault("cfg_img_scale", defaults.cfg_img_scale)
        extra_args.setdefault("cfg_interval", defaults.cfg_interval)
        extra_args.setdefault("timestep_shift", defaults.timestep_shift)
        extra_args.setdefault("cfg_renorm_min", defaults.cfg_renorm_min)
        extra_args.setdefault("cfg_renorm_type", defaults.cfg_renorm_type)
        extra_args.setdefault("think", defaults.think)
        extra_args.setdefault("caption", defaults.caption)
        extra_args.setdefault("max_think_tokens", defaults.max_think_token_n)
        extra_args.setdefault("do_sample", defaults.do_sample)
        extra_args.setdefault("text_temperature", defaults.text_temperature)
        extra_args.setdefault("sensenova_vision_mode", mode)
        if defaults.extra:
            for k, v in defaults.extra.items():
                extra_args.setdefault(k, v)

        params.extra_args = extra_args
        if params.num_inference_steps is None:
            params.num_inference_steps = defaults.num_timesteps
