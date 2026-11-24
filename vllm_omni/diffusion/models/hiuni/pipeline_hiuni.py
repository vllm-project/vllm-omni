# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""HiUni pipeline: text-to-image, instruction-based editing and image understanding from one checkpoint.

HiUni (``ernie-research/HiUni``) is a small unified multimodal model. A lightweight pixel-space diffusion head
reads the shared backbone's hidden states of the noisy image tokens and predicts clean pixels directly, so there
is no VAE and no separate text encoder.

Requests are routed by their inputs:

* ``modalities`` containing ``"text"``: autoregressive text generation (image understanding when an image is
  given, text chat otherwise).
* ``multi_modal_data["image"]`` (or ``"img2img"``): instruction-based image editing.
* otherwise: text-to-image.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from typing import Any, ClassVar

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from transformers import AutoTokenizer, Qwen2VLImageProcessor, Qwen3VLConfig
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.hiuni.hiuni_transformer import HiUniKVCache, HiUniTransformer
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

logger = init_logger(__name__)

IMG_PAD_ID, BOI_ID, EOI_ID, TIME_TOKEN_ID = 151655, 151652, 151653, 151661
IM_START_ID, IM_END_ID, NL_ID, USER_ID, ASSISTANT_ID = 151644, 151645, 198, 872, 77091
# Pixels per merged image token side: a 2x2 block of 16x16 patches.
BLOCK_SIZE = 32
# Instruction prefixes seen in training; text-to-image picks one with random.Random(seed).
HIUNI_INSTRUCTIONS = [
    "请根据以下描述生成一张图片：",
    "根据下面的描述生成图片：",
    "生成图片：",
    "请画一张图：",
    "帮我生成一张图片：",
    "根据描述创建图像：",
    "请根据文字描述绘制图片：",
    "按照以下内容生成图片：",
    "生成一张符合描述的图片：",
    "请为以下内容创作一张图片：",
    "根据提示生成图像：",
    "图片生成：",
    "Generate an image based on the following description: ",
    "Create an image: ",
    "Please generate an image: ",
    "Draw an image based on: ",
    "Create a picture: ",
    "Generate: ",
    "Please draw: ",
    "Create an image from the description: ",
    "Generate a picture: ",
    "Please create an image based on: ",
]
THINK_PREFILLS = {"think": "<think>\n", "empty": "<think>\n\n</think>\n\n", "none": ""}
SIGMA_MIN = 0.003 / 1.002


def build_positions(ids: list[int], merged_grids: list[list[int]]) -> torch.Tensor:
    """Training M-RoPE layout ``[3, L]``: linear text positions and a (t, h, w) ramp per image pad block."""
    positions = torch.zeros(3, len(ids), dtype=torch.long)
    grids, pos, index = iter(merged_grids), 0, 0
    while index < len(ids):
        if ids[index] != IMG_PAD_ID:
            positions[:, index] = pos
            pos, index = pos + 1, index + 1
            continue
        t, h, w = next(grids)
        grid = torch.stack(torch.meshgrid(torch.arange(t), torch.arange(h), torch.arange(w), indexing="ij"))
        positions[:, index : index + t * h * w] = pos + grid.reshape(3, -1)
        pos, index = pos + max(t, h, w), index + t * h * w
    return positions


def build_startend(ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Per column (first visible row, first hidden row).

    Attention is causal, image blocks are bidirectional, and a noisy block (one with a time token) is hidden from
    every row after its end-of-image token.
    """
    ids = torch.as_tensor(ids)
    length = ids.numel()
    ute, lts = torch.arange(length), torch.full((length,), length)
    starts = ids.eq(BOI_ID).nonzero(as_tuple=True)[0].tolist()
    ends = ids.eq(EOI_ID).nonzero(as_tuple=True)[0].tolist()
    for start, end in zip(starts, ends):
        if start >= end:
            continue
        times = ids[start + 1 : end].eq(TIME_TOKEN_ID).nonzero(as_tuple=True)[0]
        patch_start = start + 2 + int(times[0]) if times.numel() else start + 1
        ute[patch_start:end] = patch_start
        if times.numel():
            lts[start : end + 1] = lts[start : end + 1].clamp(max=end + 1)
    return ute, lts


def build_attention_mask(
    ute: torch.Tensor, lts: torch.Tensor, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Additive ``[1, 1, L, L]`` mask: row ``r`` sees column ``c`` iff ``ute[c] <= r < lts[c]``."""
    rows = torch.arange(ute.numel(), device=device)[:, None]
    allowed = (ute.to(device)[None, :] <= rows) & (rows < lts.to(device)[None, :])
    mask = torch.zeros(allowed.shape, dtype=dtype, device=device)
    return mask.masked_fill_(~allowed, torch.finfo(dtype).min)[None, None]


def t2i_layout(
    ids: torch.Tensor, height: int, width: int, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Text-to-image layout: one image block at the fixed position ``start`` with row/column ramps, causal elsewhere."""
    pad_index = ids.eq(IMG_PAD_ID).nonzero(as_tuple=True)[0]
    start, end, length = int(pad_index[0]), int(pad_index[-1]) + 1, ids.numel()
    positions = torch.arange(length).view(1, 1, -1).repeat(3, 1, 1)
    positions[0, 0, start:end] = start
    positions[1, 0, start:end] = start + torch.arange(height).repeat_interleave(width)
    positions[2, 0, start:end] = start + torch.arange(width).repeat(height)
    positions[:, 0, end:] = start + max(height, width) + torch.arange(length - end)
    allowed = torch.ones(length, length, dtype=torch.bool).tril()
    allowed[start:end, start:end] = True
    mask = torch.zeros(length, length, dtype=dtype).masked_fill_(~allowed, torch.finfo(dtype).min)
    return positions.to(device), mask[None, None].to(device)


def t2i_input_ids(tokenizer, prompt: str | None, seed: int, height: int, width: int) -> list[int]:
    """Positive branch: chat prompt with a training instruction prefix. Negative (``prompt=None``): the image block."""
    image = [BOI_ID, TIME_TOKEN_ID] + [IMG_PAD_ID] * (height * width) + [EOI_ID, IM_END_ID]
    if prompt is None:
        return image

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    instruction = random.Random(seed).choice(HIUNI_INSTRUCTIONS)
    user = [IM_START_ID] + encode("user") + encode("\n") + encode(instruction) + encode(prompt)
    return user + [IM_END_ID] + encode("\n") + [IM_START_ID] + encode("assistant") + encode("\n") + image


def edit_input_ids(
    tokenizer, prompt: str | None, ref_size: tuple[int, int] | None, height: int, width: int
) -> tuple[list[int], list[list[int]]]:
    """Editing branch: user turn ``[BOI, reference pads, EOI] + instruction``, assistant ``[BOI, TIME, pads, EOI]``.

    ``ref_size=None`` (the unconditional branch) keeps only the assistant image block.
    """
    image = [BOI_ID, TIME_TOKEN_ID] + [IMG_PAD_ID] * (height * width) + [EOI_ID, IM_END_ID]
    if ref_size is None:
        return image, [[1, height, width]]
    ref_h, ref_w = ref_size
    user = [BOI_ID] + [IMG_PAD_ID] * (ref_h * ref_w) + [EOI_ID]
    user += tokenizer.encode(prompt, add_special_tokens=False) if prompt else []
    ids = [IM_START_ID, USER_ID, NL_ID] + user + [IM_END_ID, NL_ID, IM_START_ID, ASSISTANT_ID, NL_ID] + image
    return ids, [[1, ref_h, ref_w], [1, height, width]]


def hiuni_sigmas(num_inference_steps: int, shift: float) -> torch.Tensor:
    """FLUX.1 sigma floor with N+1 points and the last dropped; the final Euler step then lands on sigma 0."""
    sigmas = torch.linspace(1.0, SIGMA_MIN, num_inference_steps + 1)[:-1]
    return shift * sigmas / (1 + (shift - 1) * sigmas)


def initial_noise(height: int, width: int, seed: int, device: torch.device) -> torch.Tensor:
    """Raster pixel patches ``[4*h*w, 768]`` in float32, scaled by ``sqrt(tokens / 256)`` as in training."""
    tokens = 4 * height * width
    generator = torch.Generator(device).manual_seed(seed)
    noise = torch.randn(tokens, 768, generator=generator, device=device, dtype=torch.float32)
    return noise * math.sqrt(tokens / 256)


def decode_image(latents: torch.Tensor, height: int, width: int, size: tuple[int, int] | None = None) -> Image.Image:
    """Raster patches in [-1, 1] -> PIL, optionally resized (bicubic + antialias) to ``size=(W, H)``."""
    h, w = height // BLOCK_SIZE, width // BLOCK_SIZE
    image = latents.float().reshape(2 * h, 2 * w, 3, 16, 16).permute(2, 0, 3, 1, 4).reshape(3, height, width)
    image = image.mul(0.5).add(0.5).clamp(0, 1)
    if size is not None and (size[1], size[0]) != (height, width):
        image = F.interpolate(image[None], size=(size[1], size[0]), mode="bicubic", align_corners=False, antialias=True)
        image = image[0]
    return Image.fromarray(image.clamp(0, 1).mul(255).add(0.5).to(torch.uint8).permute(1, 2, 0).cpu().numpy())


def reference_raster(pixel_values: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    """Processor output (block-major, two identical frames) -> raster patches ``[grid_h*grid_w, 768]``."""
    patches = pixel_values.float().reshape(grid_h // 2, grid_w // 2, 2, 2, 3, 2, 16, 16)[:, :, :, :, :, 0]
    return patches.permute(0, 2, 1, 3, 4, 5, 6).reshape(grid_h * grid_w, 768)


def sample_token(
    logits: torch.Tensor, temperature: float, top_k: int, top_p: float, generator: torch.Generator | None
) -> int:
    """Greedy for ``temperature <= 0``, otherwise top-k / top-p sampling."""
    if temperature <= 0:
        return int(logits.argmax())
    logits = logits.float() / temperature
    if top_k > 0:
        logits = logits.masked_fill(logits < torch.topk(logits, min(top_k, logits.numel())).values[-1], float("-inf"))
    if 0 < top_p < 1:
        sorted_logits, order = torch.sort(logits, descending=True)
        probs = sorted_logits.softmax(-1)
        logits[order[probs.cumsum(-1) - probs > top_p]] = float("-inf")
    return int(torch.multinomial(logits.softmax(-1), 1, generator=generator))


def expand_image_pads(ids: list[int], merged_grids: list[list[int]]) -> list[int]:
    """Each chat-template ``<|image_pad|>`` becomes a time token followed by one pad per merged ViT token."""
    counts = iter([t * h * w for t, h, w in merged_grids])
    expanded: list[int] = []
    for token in ids:
        expanded += [TIME_TOKEN_ID] + [IMG_PAD_ID] * next(counts) if token == IMG_PAD_ID else [token]
    return expanded


def chat_visible_from(ids: torch.Tensor) -> torch.Tensor:
    """Causal ``ute`` for understanding; image patch columns are visible from the first patch row of their block."""
    ute = torch.arange(ids.numel())
    starts = ids.eq(BOI_ID).nonzero(as_tuple=True)[0].tolist()
    ends = ids.eq(EOI_ID).nonzero(as_tuple=True)[0].tolist()
    for start, end in zip(starts, ends):
        ute[start + 2 : end] = start + 2
    return ute


def load_hiuni_config(model: str, revision: str | None = None) -> tuple[Qwen3VLConfig, dict[str, Any]]:
    """Build the backbone config from HiUni's ``config.json`` (``model_type: pixel_dit_qwen3_vl``)."""
    config_dict, _ = Qwen3VLConfig.get_config_dict(model, revision=revision)
    backbone = Qwen3VLConfig(
        text_config=config_dict["text_config"],
        vision_config=config_dict["vision_config"],
        image_token_id=config_dict.get("image_token_id", IMG_PAD_ID),
        vision_start_token_id=config_dict.get("vision_start_token_id", BOI_ID),
        vision_end_token_id=config_dict.get("vision_end_token_id", EOI_ID),
        tie_word_embeddings=config_dict.get("tie_word_embeddings", True),
    )
    return backbone, config_dict


def get_hiuni_post_process_func(od_config: OmniDiffusionConfig):
    """The pipeline already returns a PIL image (or text) payload."""
    del od_config

    def post_process_func(output):
        return output

    return post_process_func


class HiUniPipeline(nn.Module, DiffusionPipelineProfilerMixin, ProgressBarMixin):
    supports_request_batch = False
    dummy_run_num_frames: ClassVar[int] = 0
    _dit_modules: ClassVar[list[str]] = ["model.model.language_model"]

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.model_dir = od_config.model
        self.dtype = od_config.dtype if od_config.dtype is not None else torch.bfloat16
        self.device = get_local_device()
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder=None,
                revision=od_config.revision,
                prefix="model.",
                fall_back_to_pt=True,
            )
        ]
        if self.dtype != torch.bfloat16:
            raise NotImplementedError(f"HiUni only supports bfloat16; got {self.dtype!r}")
        cfg_parallel_size = int(self.od_config.parallel_config.cfg_parallel_size)
        if cfg_parallel_size > 1:
            raise NotImplementedError(f"HiUni does not support cfg_parallel_size={cfg_parallel_size}")
        self.setup_diffusion_pipeline_profiler(
            profiler_targets=["forward"],
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler,
        )

        backbone_config, config_dict = load_hiuni_config(self.model_dir, od_config.revision)
        self.model = HiUniTransformer(
            backbone_config,
            num_diffusion_layers=int(config_dict.get("num_diffusion_layers", 10)),
            gate_inject_layer=int(config_dict.get("pixel_dit_gate_inject_layer", 3)),
            image_token_id=int(config_dict.get("image_token_id", IMG_PAD_ID)),
            time_token_id=int(config_dict.get("time_token_id", TIME_TOKEN_ID)),
            quant_config=od_config.quantization_config,
            prefix=prefix,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, revision=od_config.revision)
        # The slow Qwen2-VL image processor matches the training preprocessing numerically.
        self.image_processor = Qwen2VLImageProcessor.from_pretrained(self.model_dir, revision=od_config.revision)

    # ------------------------------------------------------------------
    # Request parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _first_prompt(req: DiffusionRequestBatch) -> tuple[str | dict, str]:
        if not req.prompts:
            raise ValueError("req.prompts is empty")
        if len(req.prompts) != 1:
            raise NotImplementedError(f"HiUni handles one prompt per request; got {len(req.prompts)}")
        first = req.prompts[0]
        if isinstance(first, str):
            return first, first
        if isinstance(first, dict):
            prompt = first.get("prompt") or ""
            if not isinstance(prompt, str):
                raise TypeError(f"dict prompt must contain a str 'prompt'; got {type(prompt).__name__}")
            return first, prompt
        raise TypeError(f"prompt must be str or dict; got {type(first).__name__}")

    @staticmethod
    def _input_images(first_prompt: str | dict) -> list[Image.Image]:
        if isinstance(first_prompt, str):
            return []
        mm_data = first_prompt.get("multi_modal_data") or {}
        raw = mm_data.get("image") or mm_data.get("img2img")
        if raw is None:
            return []
        images = []
        for item in raw if isinstance(raw, list) else [raw]:
            if isinstance(item, str):
                images.append(Image.open(item).convert("RGB"))
            elif isinstance(item, Image.Image):
                images.append(item.convert("RGB"))
            else:
                raise TypeError(f"Unsupported image type: {type(item).__name__}")
        return images

    @staticmethod
    def _seed(sampling_params) -> int:
        if sampling_params.seed is not None:
            return int(sampling_params.seed)
        generator = sampling_params.generator
        if isinstance(generator, list):
            generator = generator[0] if generator else None
        if generator is not None and hasattr(generator, "initial_seed"):
            return int(generator.initial_seed())
        return 42

    @staticmethod
    def _cfg_scale(sampling_params, extra_args: dict[str, Any]) -> float:
        if "cfg_scale" in extra_args:
            return float(extra_args["cfg_scale"])
        if sampling_params.true_cfg_scale is not None:
            return float(sampling_params.true_cfg_scale)
        if sampling_params.guidance_scale_provided and sampling_params.guidance_scale is not None:
            return float(sampling_params.guidance_scale)
        return 4.0

    def _preprocess(self, images: list[Image.Image], min_pixels: int, max_pixels: int):
        inputs = self.image_processor(
            images=images, return_tensors="pt", min_pixels=int(min_pixels), max_pixels=int(max_pixels)
        )
        pixel_values = inputs["pixel_values"].to(self.device)
        grid = inputs["image_grid_thw"].long().to(self.device)
        return pixel_values, grid, self.model.visual(pixel_values, grid)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _denoise(
        self,
        branches: list[dict[str, Any]],
        cfg_scale: float,
        img_cfg_scale: float,
        height: int,
        width: int,
        seed: int,
        num_inference_steps: int,
        shift: float,
        reference: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Euler flow sampling with CFG on the x0 predictions.

        ``branches`` is ``[posi, nega]`` or, for editing with ``img_cfg_scale != 1``, ``[posi, nega, full]``.
        """
        h, w = height // BLOCK_SIZE, width // BLOCK_SIZE
        ref = reference or {}
        sigmas = hiuni_sigmas(num_inference_steps, shift)
        latents = initial_noise(h, w, seed, self.device)
        with self.progress_bar(total=num_inference_steps) as pbar:
            for index, sigma in enumerate(sigmas):
                # The model embeds the timestep on the [0, 1000] scale; keep it in float32.
                timestep = (sigma * 1000).reshape(1).to(dtype=torch.float32, device=self.device)
                preds = [
                    self.model(
                        branch["input_ids"],
                        branch["position_ids"],
                        branch["attention_mask"],
                        latents,
                        timestep,
                        h,
                        w,
                        ref_features=ref.get("features") if branch["has_ref"] else None,
                        ref_raster=ref.get("raster") if branch["has_ref"] else None,
                        ref_height=ref.get("height"),
                        ref_width=ref.get("width"),
                    )
                    for branch in branches
                ]
                if len(preds) == 3:
                    posi, nega, full = preds
                    x0 = full + cfg_scale * (posi - nega) + img_cfg_scale * (nega - full)
                else:
                    posi, nega = preds
                    x0 = nega + cfg_scale * (posi - nega)
                # Divide by a device tensor: CUDA divides by a CPU scalar through its reciprocal (one ulp off).
                velocity = (latents - x0) / sigma.to(self.device)
                sigma_next = sigmas[index + 1] if index + 1 < len(sigmas) else 0
                latents = latents + velocity * (sigma_next - sigma)
                pbar.update()
        if not torch.isfinite(latents).all():
            raise RuntimeError("Non-finite pixels produced during denoising")
        return latents

    @staticmethod
    def _round_up(value: int) -> int:
        return (int(value) + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE

    def _sampling(self, req: DiffusionRequestBatch) -> dict[str, Any]:
        sp = req.sampling_params
        extra_args = getattr(sp, "extra_args", None) or {}
        return {
            "extra_args": extra_args,
            "seed": self._seed(sp),
            "cfg_scale": self._cfg_scale(sp, extra_args),
            "img_cfg_scale": float(extra_args.get("img_cfg_scale", extra_args.get("image_cfg_scale", 1.0))),
            "shift": float(extra_args.get("shift", extra_args.get("flow_shift", 4.0))),
            "num_inference_steps": int(sp.num_inference_steps or 25),
            "height": int(sp.height) if sp.height else None,
            "width": int(sp.width) if sp.width else None,
        }

    def _forward_t2i(self, prompt: str, s: dict[str, Any]) -> DiffusionOutput:
        height, width = self._round_up(s["height"] or 1024), self._round_up(s["width"] or 1024)
        h, w = height // BLOCK_SIZE, width // BLOCK_SIZE
        branches = []
        for text in (prompt, None):
            ids = torch.tensor(t2i_input_ids(self.tokenizer, text, s["seed"], h, w))
            position_ids, attention_mask = t2i_layout(ids, h, w, self.dtype, self.device)
            branches.append(
                {
                    "input_ids": ids[None].to(self.device),
                    "position_ids": position_ids,
                    "attention_mask": attention_mask,
                    "has_ref": False,
                }
            )
        latents = self._denoise(
            branches, s["cfg_scale"], 1.0, height, width, s["seed"], s["num_inference_steps"], s["shift"]
        )
        return DiffusionOutput(output={"payload": {"image": decode_image(latents, height, width)}, "metadata": {}})

    def _edit_branch(self, prompt: str | None, ref_size: tuple[int, int] | None, h: int, w: int) -> dict[str, Any]:
        ids, grids = edit_input_ids(self.tokenizer, prompt, ref_size, h, w)
        return {
            "input_ids": torch.tensor([ids], device=self.device),
            "position_ids": build_positions(ids, grids)[:, None].to(self.device),
            "attention_mask": build_attention_mask(*build_startend(ids), self.dtype, self.device),
            "has_ref": ref_size is not None,
        }

    def _forward_edit(self, prompt: str, image: Image.Image, s: dict[str, Any]) -> DiffusionOutput:
        cfg_scale, img_cfg_scale = s["cfg_scale"], s["img_cfg_scale"]
        if img_cfg_scale != 1 and cfg_scale == 1:
            raise ValueError("img_cfg_scale != 1 requires cfg_scale != 1")
        extra_args = s["extra_args"]
        min_pixels = int(extra_args.get("min_pixels", 262144))
        max_pixels = int(extra_args.get("max_pixels", 262144))
        pixel_values, grid, features = self._preprocess([image], min_pixels, max_pixels)
        _, grid_h, grid_w = grid[0].tolist()
        reference = {
            "features": features,
            "raster": reference_raster(pixel_values, grid_h, grid_w),
            "height": grid_h // 2,
            "width": grid_w // 2,
        }
        # By default the output keeps the preprocessed reference resolution.
        height = self._round_up(s["height"] or grid_h * 16)
        width = self._round_up(s["width"] or grid_w * 16)
        h, w = height // BLOCK_SIZE, width // BLOCK_SIZE
        ref_size = (reference["height"], reference["width"])
        branches = [self._edit_branch(prompt, ref_size, h, w), self._edit_branch(None, ref_size, h, w)]
        if img_cfg_scale != 1:
            branches.append(self._edit_branch(None, None, h, w))
        latents = self._denoise(
            branches,
            cfg_scale,
            img_cfg_scale,
            height,
            width,
            s["seed"],
            s["num_inference_steps"],
            s["shift"],
            reference=reference,
        )
        # Without an explicit size the output is resized back to the input image, as in the reference implementation.
        explicit_size = s["height"] is not None or s["width"] is not None
        resize = bool(extra_args.get("resize_to_edit_image", not explicit_size))
        output = decode_image(latents, height, width, image.size if resize else None)
        return DiffusionOutput(output={"payload": {"image": output}, "metadata": {}})

    # ------------------------------------------------------------------
    # Understanding and text chat
    # ------------------------------------------------------------------

    def _forward_text(self, prompt: str, images: list[Image.Image], s: dict[str, Any]) -> DiffusionOutput:
        extra_args = s["extra_args"]
        think_mode = str(extra_args.get("think_mode", "think"))
        if think_mode not in THINK_PREFILLS:
            raise ValueError(f"think_mode must be one of {sorted(THINK_PREFILLS)}; got {think_mode!r}")
        max_new_tokens = int(extra_args.get("max_new_tokens", extra_args.get("max_tokens", 1024)))
        max_length = int(extra_args.get("max_length", 9216))
        temperature = float(extra_args.get("temperature", 0.0))
        top_k, top_p = int(extra_args.get("top_k", 0)), float(extra_args.get("top_p", 1.0))
        system = str(extra_args.get("system", ""))

        prefill = THINK_PREFILLS[think_mode]
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": content if images else prompt})
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, add_vision_id=False
        )
        ids = self.tokenizer.encode(text + prefill, add_special_tokens=False)
        features, merged_grids = None, []
        if images:
            min_pixels = int(extra_args.get("min_pixels", 262144))
            max_pixels = int(extra_args.get("max_pixels", 4194304))
            _, grid, features = self._preprocess(images, min_pixels, max_pixels)
            merged_grids = [[t, h // 2, w // 2] for t, h, w in grid.tolist()]
            ids = expand_image_pads(ids, merged_grids)
        positions = build_positions(ids, merged_grids)
        ids_tensor = torch.tensor(ids)
        ute = chat_visible_from(ids_tensor) if images else torch.arange(ids_tensor.numel())
        # Prefill with the training mask; every generated token then sees all earlier columns.
        mask = build_attention_mask(ute, torch.full_like(ute, ute.numel()), self.dtype, self.device)
        generator = torch.Generator(self.device).manual_seed(s["seed"]) if temperature > 0 else None
        eos_id = self.tokenizer.eos_token_id
        stop_ids = {IM_END_ID} | ({int(eos_id)} if eos_id is not None else set())

        kv_cache = HiUniKVCache()
        logits = self.model.text_logits(
            ids_tensor[None].to(self.device),
            positions[:, None].to(self.device),
            mask,
            features,
            len(merged_grids),
            kv_cache=kv_cache,
        )[0]
        position = positions[:, -1:].to(self.device)
        length, new_tokens = len(ids), []
        while len(new_tokens) < max_new_tokens and length < max_length:
            token = sample_token(logits, temperature, top_k, top_p, generator)
            if token in stop_ids:
                break
            new_tokens.append(token)
            length += 1
            if len(new_tokens) >= max_new_tokens or length >= max_length:
                break
            position = position + 1
            logits = self.model.text_logits(
                torch.tensor([[token]], device=self.device), position[:, None], None, kv_cache=kv_cache
            )[0]

        raw = prefill + self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        # The model may reason again after an empty think block, so keep only the text after the last </think>.
        answer = raw.rsplit("</think>", 1)[-1].strip()
        text_output = raw if extra_args.get("return_raw") else answer
        logger.info("HiUni text generation: %d new tokens", len(new_tokens))
        return DiffusionOutput(
            output={"payload": {"text": text_output}, "metadata": {"text": {"text_output": text_output}}}
        )

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        first_prompt, prompt = self._first_prompt(req)
        images = self._input_images(first_prompt)
        modalities = first_prompt.get("modalities", []) if isinstance(first_prompt, dict) else []
        s = self._sampling(req)
        if "text" in (modalities or []):
            return self._forward_text(prompt, images, s)
        if images:
            if len(images) != 1:
                raise NotImplementedError(f"HiUni editing takes one reference image; got {len(images)}")
            return self._forward_edit(prompt, images[0], s)
        return self._forward_t2i(prompt, s)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(weights)
