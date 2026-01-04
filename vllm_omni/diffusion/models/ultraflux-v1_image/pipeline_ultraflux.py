# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import time
import inspect
from pathlib import Path
from typing import Any, Iterable, Tuple, Dict, List

import torch
from torch import nn
from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler

from vllm.logger import init_logger
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest

from .pipeline_flux import FluxPipeline
from .transformer_flux import FluxTransformer2DModel
from .autoencoder_kl import AutoencoderKL

logger = init_logger(__name__)


_DTYPE_MAP: dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _normalize_dtype(dtype: str | torch.dtype | None) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype is None:
        return torch.bfloat16
    k = str(dtype).lower()
    if k not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {dtype}. Supported: {sorted(_DTYPE_MAP.keys())}")
    return _DTYPE_MAP[k]


def _device_from_config(_: OmniDiffusionConfig) -> torch.device:
    if torch.cuda.is_available():
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def _ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_images(
    images: list[Image.Image],
    out_dir: str | Path,
    basename: str,
    ext: str = "png",
) -> list[str]:
    out_dir = _ensure_dir(out_dir)
    paths: list[str] = []
    for i, im in enumerate(images):
        fp = out_dir / f"{basename}_{i:02d}.{ext}"
        im.save(fp)
        paths.append(str(fp))
    return paths


class UltraFluxPipeline(nn.Module):
    """
    vLLM-Omni native UltraFlux diffusion pipeline.

    IMPORTANT:
    ◦ FluxPipeline may NOT be an nn.Module and may NOT implement state_dict().

    ◦ vLLM DiffusersLoader still expects model.load_weights(weights_iter).

    ◦ Therefore we MUST load weights into submodules (vae/transformer/encoders) individually.

    """

    # Common component names you might have in UltraFlux/Flux-style pipelines.
    # We will probe these on self.pipe and load weights by prefix matching.
    _COMPONENT_ATTRS: tuple[str, ...] = (
        "transformer",
        "vae",
        "text_encoder",
        "text_encoder_2",
        "tokenizer",      # not a module, ignored
        "tokenizer_2",    # not a module, ignored
        "image_encoder",
        "unet",           # if present
    )

    def __init__(self, od_config: OmniDiffusionConfig):
        super().__init__()
        self.od_config = od_config

        model_id = getattr(od_config, "model", None)
        if not model_id:
            raise ValueError("OmniDiffusionConfig.model must be set for UltraFluxPipeline.")

        # device & dtype
        self.device: torch.device = _device_from_config(od_config)
        self.torch_dtype: torch.dtype = _normalize_dtype(getattr(od_config, "dtype", "bf16"))

        vae = AutoencoderKL.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=self.torch_dtype,
        )

        transformer = FluxTransformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            torch_dtype=self.torch_dtype,
        )

        pipe = FluxPipeline.from_pretrained(
            model_id,
            vae=vae,
            transformer=transformer,
            torch_dtype=self.torch_dtype,
        )

        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.config.use_dynamic_shifting = False
        pipe.scheduler.config.time_shift = 4

        # FluxPipeline might implement .to(), but it may only move internal modules.
        self.pipe = pipe
        if hasattr(self.pipe, "to"):
            self.pipe = self.pipe.to(self.device)

        # vllm-omni expects these in initialize_model()
        self.vae = getattr(self.pipe, "vae", None)
        self.transformer = getattr(self.pipe, "transformer", None)

        # tokenizer max length
        for tok_name in ("tokenizer", "tokenizer_2"):
            tok = getattr(self.pipe, tok_name, None)
            if tok is not None and hasattr(tok, "model_max_length"):
                tok.model_max_length = 512

        self._apply_vae_optimizations()
        self.eval()

    def _iter_submodules(self) -> List[tuple[str, nn.Module]]:
        mods: List[tuple[str, nn.Module]] = []
        # Prefer explicit known attributes on pipe
        for attr in self._COMPONENT_ATTRS:
            obj = getattr(self.pipe, attr, None)
            if isinstance(obj, nn.Module):
                mods.append((attr, obj))
        # Also include any nn.Modules registered directly on self (unlikely, but safe)
        for name, obj in super().named_children():
            if isinstance(obj, nn.Module):
                mods.append((name, obj))
        # De-duplicate by id
        seen = set()
        uniq: List[tuple[str, nn.Module]] = []
        for n, m in mods:
            if id(m) in seen:
                continue
            seen.add(id(m))
            uniq.append((n, m))
        return uniq

    def parameters(self, recurse: bool = True):
        # vLLM frequently calls model.parameters()
        for _, m in self._iter_submodules():
            yield from m.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        for name, m in self._iter_submodules():
            pfx = f"{prefix}.{name}" if prefix else name
            yield from m.named_parameters(prefix=pfx, recurse=recurse)


    def load_weights(self, weights_iter: Iterable[Tuple[str, torch.Tensor]]):
        """
        vLLM loader yields (name, tensor) pairs.

        Because FluxPipeline may not be an nn.Module (no state_dict),
        we dispatch incoming weights into component modules based on prefix:
            transformer.xxx -> self.pipe.transformer.load_state_dict({xxx: ...})
            vae.xxx         -> self.pipe.vae.load_state_dict({xxx: ...})
            text_encoder.xxx, text_encoder_2.xxx, etc.

        Returns:
            set of loaded fully-qualified keys for diagnostics.
        """
        # Build module map from pipeline components
        module_map: Dict[str, nn.Module] = {}
        for attr, mod in self._iter_submodules():
            module_map[attr] = mod

        # Accumulate per-module tensors with stripped prefix
        bucket: Dict[str, Dict[str, torch.Tensor]] = {k: {} for k in module_map.keys()}
        loaded: set[str] = set()

        # Prefixes sometimes used by loaders
        strip_prefixes = ("pipe.", "model.", "module.", "")

        for raw_name, tensor in weights_iter:
            if tensor is None:
                continue

            # Normalize name by stripping common leading prefixes
            name = raw_name
            for pfx in strip_prefixes:
                if pfx and name.startswith(pfx):
                    name = name[len(pfx):]
                    break

            # Expect form "component.subkey..."
            if "." not in name:
                # Not a component param, skip
                continue

            comp, subkey = name.split(".", 1)

            # Some loaders may use "vae_decoder"/"vae_encoder" patterns; best-effort remap
            if comp not in module_map:
                if comp.startswith("vae") and "vae" in module_map:
                    comp = "vae"
                elif comp.startswith("transformer") and "transformer" in module_map:
                    comp = "transformer"
                elif comp.startswith("text_encoder_2") and "text_encoder_2" in module_map:
                    comp = "text_encoder_2"
                elif comp.startswith("text_encoder") and "text_encoder" in module_map:
                    comp = "text_encoder"
                else:
                    continue

            bucket[comp][subkey] = tensor
            loaded.add(f"{comp}.{subkey}")

        # Load each module (non-strict to tolerate buffers/extras)
        for comp, mod in module_map.items():
            sd_part = bucket.get(comp, {})
            if not sd_part:
                continue
            missing, unexpected = mod.load_state_dict(sd_part, strict=False)
            if missing:
                logger.debug("UltraFlux load_weights: %s missing=%d (first=%s)", comp, len(missing), missing[:5])
            if unexpected:
                logger.debug("UltraFlux load_weights: %s unexpected=%d (first=%s)", comp, len(unexpected), unexpected[:5])

        return loaded


    def _apply_vae_optimizations(self):
        vae = getattr(self, "vae", None)
        if vae is None:
            return

        use_slicing = bool(getattr(self.od_config, "vae_use_slicing", False))
        use_tiling = bool(getattr(self.od_config, "vae_use_tiling", False))

        if hasattr(vae, "use_slicing"):
            try:
                vae.use_slicing = use_slicing
            except Exception:
                pass

        if hasattr(vae, "use_tiling"):
            try:
                vae.use_tiling = use_tiling
            except Exception:
                pass

        # UltraFlux partitioned decode (4K/8K critical) if available
        if hasattr(vae, "decode"):
            try:
                sig = inspect.signature(vae.decode)
                if hasattr(vae, "config"):
                    if "partitioned" in sig.parameters:
                        setattr(vae.config, "use_partitioned_decode", use_tiling)
                    else:
                        setattr(vae.config, "use_partitioned_decode", False)
            except Exception:
                pass

    @torch.no_grad()
    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | list[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float = 4.0,
        max_sequence_length: int | None = None,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> DiffusionOutput:
        t0 = time.perf_counter()

        p = getattr(req, "prompt", None) if req is not None else None
        p = p if p is not None else prompt
        if p is None:
            return DiffusionOutput(error="Prompt is required.")

        h = (getattr(req, "height", None) or height or 1024)
        w = (getattr(req, "width", None) or width or 1024)
        steps = (getattr(req, "num_inference_steps", None) or num_inference_steps or 50)

        cfg = getattr(req, "guidance_scale", None)
        cfg = cfg if cfg is not None else guidance_scale

        max_len = (getattr(req, "max_sequence_length", None) or max_sequence_length or 512)

        num_images_per_prompt = int(
            getattr(req, "num_outputs_per_prompt", None)
            or getattr(req, "batch_size", None)
            or 1
        )

        if isinstance(p, str):
            prompts = [p]
        else:
            prompts = list(p)

        prompts_expanded: list[str] = []
        for pp in prompts:
            prompts_expanded.extend([pp] * num_images_per_prompt)

        seed = getattr(req, "seed", None)
        if generator is None and seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))

        device_type = self.device.type
        autocast_dtype = self.torch_dtype if device_type == "cuda" else None

        # Optional offline saving (default OFF; only if explicitly provided)
        save_dir = kwargs.get("save_dir") or getattr(req, "save_dir", None)
        save_dir = str(save_dir) if save_dir else None
        save_name = kwargs.get("save_name") or getattr(req, "save_name", None) or "ultraflux"
        save_ext = kwargs.get("save_ext") or getattr(req, "save_ext", None) or "png"

        try:
            if device_type == "cuda":
                with torch.autocast(device_type=device_type, dtype=autocast_dtype):
                    out = self.pipe(
                        prompts_expanded,
                        height=int(h),
                        width=int(w),
                        num_inference_steps=int(steps),
                        guidance_scale=float(cfg),
                        max_sequence_length=int(max_len),
                        generator=generator,
                    )
            else:
                out = self.pipe(
                    prompts_expanded,
                    height=int(h),
                    width=int(w),
                    num_inference_steps=int(steps),
                    guidance_scale=float(cfg),
                    max_sequence_length=int(max_len),
                    generator=generator,
                )

            images: list[Image.Image] = list(out.images)

            if save_dir:
                _save_images(images, save_dir, str(save_name), str(save_ext))

            dt_ms = (time.perf_counter() - t0) * 1000.0
            logger.info(
                "UltraFlux done: batch=%d steps=%d cfg=%.2f size=%dx%d time=%.1fms",
                len(prompts_expanded),
                int(steps),
                float(cfg),
                int(w),
                int(h),
                dt_ms,
            )

            return DiffusionOutput(output=images)

        except Exception as e:
            logger.exception("UltraFlux inference failed")
            return DiffusionOutput(error=f"{type(e).__name__}: {e}")