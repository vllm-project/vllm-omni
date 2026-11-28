# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""``hub`` FastWAM kernel backend: AMD's CAE-FastWAM kernel package from the Hugging Face Kernel Hub.

Enabled with ``VLLM_OMNI_FASTWAM_KERNELS=hub`` (off by default; see
:mod:`~vllm_omni.diffusion.models.fastwam.kernel_backend`).

The package (``amd/cae-fastwam-kernels``, loaded with ``kernels.get_kernel`` at a pinned version)
is *model-bound*: it ships optimised replacements of four CAE-FastWAM model modules of the
reference implementation (AMD-AGI/faster-action-models ``FastWAM/src``) together with their Triton
kernels, and ``install()`` routes the reference modules to them. Its measured path is the
reference ``FastWAM.infer_action(..., compile_action_infer=True)``, which compiles its own regions
(reduce-overhead + fullgraph) around the package's custom ops. This backend therefore:

1. checks that it can run: ROCm, an AMD Instinct MI355X-class GPU (gfx950, the package's only
   validated architecture), the CAE-FastWAM preset, the ``kernels`` library, and the pinned reference
   tree importable as ``fastwam`` (the package verifies the tree's sha256 pins itself);
2. loads the package, calls ``install()`` before the reference model modules are imported and checks
   the checkpoint layout with ``verify_checkpoint``;
3. builds the reference model from the same assets as the stock path (``policy.pt`` and
   ``action_vae.pt`` of the FastWAM checkpoint; the Wan2.2 VAE and umT5 tensors from the Diffusers
   base repository, renamed to the reference layout, which holds the identical tensors);
4. serves ``infer_action`` through it. Observation parsing, normalisation and the LIBERO gripper
   post-processing stay in vLLM-Omni and are shared with the stock path.

Anything missing (no ``kernels``, no Hub access to the package, a non-gfx950 GPU, ...) makes
:func:`~vllm_omni.diffusion.models.fastwam.kernel_backend.load_fastwam_kernel_backend` log the
reason once and serve the stock path. The runner's own ``torch.compile`` is not applied on top of
this backend: the package documents that an outer compile only adds guards and dispatch.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from collections import OrderedDict
from typing import Any

import regex as re
import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.models.fastwam.kernel_backend import FastWAMKernelContext

logger = init_logger(__name__)

KERNEL_REPO = "amd/cae-fastwam-kernels"
KERNEL_VERSION = 1
# The package's validated architecture (its build.toml ``rocm_archs``).
SUPPORTED_ARCHS = ("gfx950",)
SUPPORTED_VARIANTS = ("cae_fastwam_libero",)
# The reference package the kernels plug into, and the module install() must precede.
REFERENCE_PACKAGE = "fastwam"
REFERENCE_MODEL_MODULE = "fastwam.models.wan22.fastwam"
_PROMPT_CACHE_SIZE = 256


def gpu_arch(device: torch.device) -> str | None:
    """The GPU architecture name without feature flags (``"gfx950"``), or ``None`` when unknown."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    arch = getattr(torch.cuda.get_device_properties(device), "gcnArchName", "") or ""
    return arch.split(":")[0] or None


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


# ------------------------------------------------------------------ Diffusers -> reference key layout
_VAE_RES = {"norm1": "residual.0", "conv1": "residual.2", "norm2": "residual.3", "conv2": "residual.6"}
_VAE_RES["conv_shortcut"] = "shortcut"
_VAE_FIXED = {
    "encoder.conv_in.weight": "model.encoder.conv1.weight",
    "encoder.conv_in.bias": "model.encoder.conv1.bias",
    "encoder.norm_out.gamma": "model.encoder.head.0.gamma",
    "encoder.conv_out.weight": "model.encoder.head.2.weight",
    "encoder.conv_out.bias": "model.encoder.head.2.bias",
    "quant_conv.weight": "model.conv1.weight",
    "quant_conv.bias": "model.conv1.bias",
}


def vae_key_to_reference(name: str, num_res_blocks: int = 2) -> str | None:
    """Diffusers ``AutoencoderKLWan`` encoder key -> reference ``WanVideoVAE38`` key (``None``: not needed)."""
    m = re.fullmatch(r"encoder\.down_blocks\.(\d+)\.resnets\.(\d+)\.(\w+)\.(weight|bias|gamma)", name)
    if m:
        i, j, part, leaf = m.groups()
        return f"model.encoder.downsamples.{i}.downsamples.{j}.{_VAE_RES[part]}.{leaf}"
    m = re.fullmatch(r"encoder\.down_blocks\.(\d+)\.downsampler\.(.+)", name)
    if m:
        return f"model.encoder.downsamples.{m.group(1)}.downsamples.{num_res_blocks}.{m.group(2)}"
    m = re.fullmatch(r"encoder\.mid_block\.resnets\.(\d)\.(\w+)\.(weight|bias|gamma)", name)
    if m:
        j, part, leaf = m.groups()
        return f"model.encoder.middle.{0 if j == '0' else 2}.{_VAE_RES[part]}.{leaf}"
    m = re.fullmatch(r"encoder\.mid_block\.attentions\.0\.(.+)", name)
    if m:
        return f"model.encoder.middle.1.{m.group(1)}"
    return _VAE_FIXED.get(name)


_T5_LAYER = (
    {
        "SelfAttention.q.weight": "attn.q.weight",
        "SelfAttention.k.weight": "attn.k.weight",
        "SelfAttention.v.weight": "attn.v.weight",
        "SelfAttention.o.weight": "attn.o.weight",
        "SelfAttention.relative_attention_bias.weight": "pos_embedding.embedding.weight",
        "layer_norm.weight": "norm1.weight",
    },
    {
        "DenseReluDense.wi_0.weight": "ffn.gate.0.weight",
        "DenseReluDense.wi_1.weight": "ffn.fc1.weight",
        "DenseReluDense.wo.weight": "ffn.fc2.weight",
        "layer_norm.weight": "norm2.weight",
    },
)


def t5_key_to_reference(name: str) -> str | None:
    """HF ``UMT5EncoderModel`` key -> reference ``WanTextEncoder`` key (``None``: not needed)."""
    if name in ("shared.weight", "encoder.embed_tokens.weight"):
        return "token_embedding.weight"
    if name == "encoder.final_layer_norm.weight":
        return "norm.weight"
    m = re.fullmatch(r"encoder\.block\.(\d+)\.layer\.([01])\.(.+)", name)
    if not m:
        return None
    i, layer, rest = m.groups()
    target = _T5_LAYER[int(layer)].get(rest)
    return None if target is None else f"blocks.{i}.{target}"


def _read_safetensors(files: list[str], rename) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    out: dict[str, torch.Tensor] = {}
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                target = rename(key)
                if target is not None:
                    out[target] = handle.get_tensor(key)
    return out


def _text_encoder_files(text_encoder_dir: str) -> list[str]:
    index = os.path.join(text_encoder_dir, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index, encoding="utf-8") as f:
            names = sorted(set(json.load(f)["weight_map"].values()))
        return [os.path.join(text_encoder_dir, n) for n in names]
    return [os.path.join(text_encoder_dir, "model.safetensors")]


# ------------------------------------------------------------------ the loaded backend
class HubKernelsFastWAM:
    """The reference CAE-FastWAM model running AMD's Kernel Hub package (``infer_action`` only)."""

    name = "hub"

    def __init__(
        self, model: nn.Module, kernel: Any, *, action_horizon: int, prompt_cache_size: int = _PROMPT_CACHE_SIZE
    ) -> None:
        self.model = model
        self.kernel = kernel
        self.action_horizon = action_horizon
        self.components: dict[str, nn.Module | None] = {
            "mot": model.mot,
            "text_encoder": model.text_encoder,
            "vae_encoder": model.vae,
            "proprio_encoder": model.proprio_encoder,
            "action_codec": model.action_vae,
        }
        self._prompt_cache: OrderedDict[str, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._prompt_cache_size = prompt_cache_size
        ns = getattr(kernel, "OPS_NAMESPACE", "?")
        self.description = f"{KERNEL_REPO} version {KERNEL_VERSION} (op namespace {ns}), compile_action_infer=True"

    def _context(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._prompt_cache.get(prompt)
        if cached is None:
            cached = self.model.encode_prompt(prompt)
            self._prompt_cache[prompt] = cached
            if len(self._prompt_cache) > self._prompt_cache_size:
                self._prompt_cache.popitem(last=False)
        else:
            self._prompt_cache.move_to_end(prompt)
        return cached

    def infer_action(
        self,
        image: torch.Tensor,
        prompt: str,
        proprio: torch.Tensor,
        *,
        num_steps: int,
        generator: torch.Generator | None,
        suite_id: int | None,
    ) -> torch.Tensor:
        # FastWAMPipeline.forward runs under torch.inference_mode(). The package's compiled regions (CUDA-graph replay,
        # in-place K/V buffers) need tensors that track version counters, so leave inference mode and only disable
        # autograd. Every tensor the reference model sees is created inside this block.
        with torch.inference_mode(False), torch.no_grad():
            return self._infer_action(
                image, prompt, proprio, num_steps=num_steps, generator=generator, suite_id=suite_id
            )

    def _infer_action(
        self,
        image: torch.Tensor,
        prompt: str,
        proprio: torch.Tensor,
        *,
        num_steps: int,
        generator: torch.Generator | None,
        suite_id: int | None,
    ) -> torch.Tensor:
        model = self.model
        image, proprio = image.clone(), proprio.clone()  # plain tensors, even if the caller made them in inference mode
        # Same pixel scaling as the stock path (and the reference evaluator): uint8 -> [-1, 1] in the model dtype.
        pixels = image.to(device=model.device, dtype=model.torch_dtype) * (2.0 / 255.0) - 1.0
        context, context_mask = self._context(prompt)
        # The reference draws the initial noise from a CPU generator it seeds itself; the pipeline hands
        # over a freshly seeded CPU generator, whose seed reproduces that draw.
        seed = None if generator is None else int(generator.initial_seed())
        model.action_vae_suite_id = suite_id
        out = model.infer_action(
            prompt=None,
            input_image=pixels,
            action_horizon=self.action_horizon,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt="",
            text_cfg_scale=1.0,
            num_inference_steps=num_steps,
            sigma_shift=None,
            seed=seed,
            rand_device="cpu",
            tiled=False,
            compile_action_infer=True,
        )
        return out["action"].detach().to(device="cpu", dtype=torch.float32)


class HubKernelsBackend:
    """Registry entry for ``VLLM_OMNI_FASTWAM_KERNELS=hub``."""

    name = "hub"
    repo_id = KERNEL_REPO
    version = KERNEL_VERSION

    def unsupported_reason(self, ctx: FastWAMKernelContext) -> str | None:
        if ctx.cfg.variant not in SUPPORTED_VARIANTS:
            return f"{KERNEL_REPO} supports {', '.join(SUPPORTED_VARIANTS)} only, not {ctx.cfg.variant}"
        if ctx.dtype != torch.bfloat16:
            return f"{KERNEL_REPO} runs bfloat16 only, not {ctx.dtype}"
        if torch.version.hip is None or ctx.device.type != "cuda":
            return f"{KERNEL_REPO} needs a ROCm GPU (device {ctx.device}, HIP {torch.version.hip})"
        arch = gpu_arch(ctx.device)
        if arch not in SUPPORTED_ARCHS:
            return f"{KERNEL_REPO} is built for {', '.join(SUPPORTED_ARCHS)} only; this GPU is {arch or 'unknown'}"
        if not _module_available("kernels"):
            return "the Hugging Face `kernels` package is not installed"
        if not _module_available(REFERENCE_PACKAGE):
            return (
                f"the reference implementation the package plugs into (AMD-AGI/faster-action-models FastWAM/src, "
                f"importable as `{REFERENCE_PACKAGE}`) is not installed"
            )
        if REFERENCE_MODEL_MODULE in sys.modules:
            return f"{REFERENCE_MODEL_MODULE} was imported before the kernels could be installed"
        return None

    def load(self, ctx: FastWAMKernelContext) -> HubKernelsFastWAM:
        from kernels import get_kernel

        kernel = get_kernel(self.repo_id, version=self.version)
        kernel.install()  # before anything imports the reference model modules; it checks the reference tree pins
        policy_pt = os.path.join(ctx.model_dir, ctx.cfg.policy_checkpoint)
        codec_pt = os.path.join(ctx.model_dir, ctx.cfg.action_codec_checkpoint)
        kernel.verify_checkpoint(policy_pt, codec_pt)
        model = build_reference_model(ctx, policy_pt, codec_pt)
        return HubKernelsFastWAM(model, kernel, action_horizon=ctx.cfg.action_horizon)


def build_reference_model(ctx: FastWAMKernelContext, policy_pt: str, codec_pt: str) -> nn.Module:
    """Build the reference CAE-FastWAM model (after ``install()``) from the stock path's assets."""
    from fastwam.models.action_ae import ActionAE, ActionAEConfig
    from fastwam.models.wan22.action_dit import ActionDiT
    from fastwam.models.wan22.fastwam import FastWAM
    from fastwam.models.wan22.mot import MoT
    from fastwam.models.wan22.wan_video_dit import WanVideoDiT
    from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer, WanTextEncoder
    from fastwam.models.wan22.wan_video_vae import WanVideoVAE38

    cfg, dev, dtype = ctx.cfg, ctx.device, ctx.dtype
    # Built on the CPU and then moved, like the reference loader (its RoPE tables are computed on the CPU).
    video = WanVideoDiT(
        hidden_dim=cfg.video_hidden_dim,
        in_dim=cfg.video_in_dim,
        ffn_dim=cfg.video_ffn_dim,
        out_dim=cfg.video_in_dim,
        text_dim=cfg.text_dim,
        freq_dim=cfg.freq_dim,
        eps=cfg.eps,
        patch_size=tuple(cfg.patch_size),
        num_heads=cfg.num_heads,
        attn_head_dim=cfg.attn_head_dim,
        num_layers=cfg.num_layers,
        has_image_input=False,
        seperated_timestep=True,
        require_clip_embedding=False,
        require_vae_embedding=False,
        fuse_vae_embedding_in_latents=True,
        video_attention_mask_mode="first_frame_causal",
        action_conditioned=False,
        action_dim=cfg.action_dim,
    )
    action = ActionDiT(
        hidden_dim=cfg.action_hidden_dim,
        action_dim=cfg.action_token_dim,
        ffn_dim=cfg.action_ffn_dim,
        text_dim=cfg.text_dim,
        freq_dim=cfg.freq_dim,
        eps=cfg.eps,
        num_heads=cfg.num_heads,
        attn_head_dim=cfg.attn_head_dim,
        num_layers=cfg.num_layers,
    )
    video, action = video.to(dev, dtype), action.to(dev, dtype)
    mot = MoT({"video": video, "action": action})

    vae = WanVideoVAE38()
    vae_file = os.path.join(ctx.base_model_dir, "vae", "diffusion_pytorch_model.safetensors")
    result = vae.load_state_dict(_read_safetensors([vae_file], vae_key_to_reference), strict=False)
    missing = [k for k in result.missing_keys if ".decoder." not in k and not k.startswith("model.conv2")]
    if result.unexpected_keys or missing:
        raise ValueError(f"Wan VAE encoder: unexpected {result.unexpected_keys[:5]}, missing {missing[:5]}")
    vae = vae.to(dev, dtype)
    text_dir = os.path.join(ctx.base_model_dir, "text_encoder")
    text_encoder = WanTextEncoder()
    text_encoder.load_state_dict(_read_safetensors(_text_encoder_files(text_dir), t5_key_to_reference), strict=True)
    text_encoder = text_encoder.to(dev, dtype).eval()
    tokenizer = HuggingfaceTokenizer(
        name=os.path.join(ctx.base_model_dir, "tokenizer"), seq_len=cfg.text_len, clean="whitespace"
    )

    payload = torch.load(codec_pt, map_location="cpu", weights_only=True)
    action_vae = ActionAE(ActionAEConfig(**payload["config"]))
    action_vae.load_state_dict(payload["model"], strict=True)
    action_vae = action_vae.to(dev, torch.float32).eval()
    model = FastWAM(
        video_expert=video,
        action_expert=action,
        mot=mot,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        text_dim=cfg.text_dim,
        proprio_dim=cfg.proprio_dim,
        action_vae=action_vae,
        action_vae_suite_id=0,
        device=str(dev),
        torch_dtype=dtype,
        action_infer_shift=cfg.action_shift,
    )
    ckpt = torch.load(policy_pt, map_location="cpu", weights_only=True, mmap=True)
    result = model.mot.load_state_dict(ckpt["mot"], strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(f"{policy_pt} mot: missing {result.missing_keys[:5]}, unexpected {result.unexpected_keys[:5]}")
    model.proprio_encoder.load_state_dict(ckpt["proprio_encoder"], strict=True)
    return model.to(dev).eval().requires_grad_(False)


HUB_KERNELS_BACKEND = HubKernelsBackend()

__all__ = [
    "HUB_KERNELS_BACKEND",
    "KERNEL_REPO",
    "KERNEL_VERSION",
    "HubKernelsBackend",
    "HubKernelsFastWAM",
    "t5_key_to_reference",
    "vae_key_to_reference",
]
