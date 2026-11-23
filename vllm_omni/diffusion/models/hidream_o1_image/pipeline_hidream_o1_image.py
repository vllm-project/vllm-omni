# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Iterable
from typing import ClassVar, TypedDict

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from torch import nn
from transformers import AutoProcessor, PreTrainedTokenizerBase, Qwen3VLConfig
from transformers.processing_utils import ProcessorMixin
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.hidream_o1_image.hidream_o1_image_transformer import (
    HiDreamO1ImageTransformer,
)
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.models.schedulers import FlowUniPCMultistepScheduler
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import (
    DiffusionPipelineProfilerMixin,
)
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

logger = init_logger(__name__)


# upstream: HiDream-O1-Image models/pipeline.py L14-18 @21bcd30471ac
PATCH_SIZE = 32
TIMESTEP_TOKEN_NUM = 1
NOISE_SCALE = 8.0
T_EPS = 0.001

# upstream: HiDream-O1-Image models/utils.py L7-19 @21bcd30471ac
PREDEFINED_RESOLUTIONS: tuple[tuple[int, int], ...] = (
    (2048, 2048),
    (2304, 1728),
    (1728, 2304),
    (2560, 1440),
    (1440, 2560),
    (2496, 1664),
    (1664, 2496),
    (3104, 1312),
    (1312, 3104),
    (2304, 1792),
    (1792, 2304),
)


class HiDreamO1TextSample(TypedDict):
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    attention_mask: torch.Tensor | None
    full_attn_spans: list[list[tuple[int, int]]]
    vinput_mask: torch.Tensor


def find_closest_resolution(width: int, height: int) -> tuple[int, int]:
    img_ratio = width / height
    best_res: tuple[int, int] | None = None
    min_diff = float("inf")
    for w, h in PREDEFINED_RESOLUTIONS:
        diff = abs(w / h - img_ratio)
        if diff < min_diff:
            min_diff = diff
            best_res = (w, h)
    assert best_res is not None
    return best_res


def get_rope_index_fix_point(
    *,
    spatial_merge_size: int,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
    input_ids: torch.Tensor | None = None,
    image_grid_thw: torch.Tensor | None = None,
    video_grid_thw: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    skip_vision_start_token: list[int],
    fix_point: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list[torch.Tensor] = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                text_len -= skip_vision_start_token[image_index - 1]
                text_len = max(0, text_len)

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()

                if skip_vision_start_token[image_index - 1]:
                    if fix_point > 0:
                        fix_point = fix_point - st_idx
                        llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + fix_point + st_idx)
                        fix_point = 0
                else:
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
        return position_ids, mrope_position_deltas


def build_t2i_text_sample(
    *,
    prompt: str,
    height: int,
    width: int,
    tokenizer: PreTrainedTokenizerBase,
    processor: ProcessorMixin,
    model_config: Qwen3VLConfig,
) -> dict[str, torch.Tensor]:
    image_token_id = model_config.image_token_id
    video_token_id = model_config.video_token_id
    vision_start_token_id = model_config.vision_start_token_id
    image_len = (height // PATCH_SIZE) * (width // PATCH_SIZE)

    boi_token = getattr(tokenizer, "boi_token", "<|boi_token|>")
    tms_token = getattr(tokenizer, "tms_token", "<|tms_token|>")

    messages = [{"role": "user", "content": prompt}]
    template_caption = (
        processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        + boi_token
        + tms_token * TIMESTEP_TOKEN_NUM
    )
    input_ids = tokenizer.encode(template_caption, return_tensors="pt", add_special_tokens=False)

    image_grid_thw = torch.tensor([1, height // PATCH_SIZE, width // PATCH_SIZE], dtype=torch.int64).unsqueeze(0)

    vision_tokens = torch.zeros((1, image_len), dtype=input_ids.dtype) + image_token_id
    vision_tokens[0, 0] = vision_start_token_id
    input_ids_pad = torch.cat([input_ids, vision_tokens], dim=-1)

    position_ids, _ = get_rope_index_fix_point(
        spatial_merge_size=1,
        image_token_id=image_token_id,
        video_token_id=video_token_id,
        vision_start_token_id=vision_start_token_id,
        input_ids=input_ids_pad,
        image_grid_thw=image_grid_thw,
        video_grid_thw=None,
        attention_mask=None,
        skip_vision_start_token=[1],
    )

    txt_seq_len = input_ids.shape[-1]
    all_seq_len = position_ids.shape[-1]

    token_types = torch.zeros((1, all_seq_len), dtype=input_ids.dtype)
    bgn = txt_seq_len - TIMESTEP_TOKEN_NUM
    token_types[0, bgn : bgn + image_len + TIMESTEP_TOKEN_NUM] = 1
    token_types[0, txt_seq_len - TIMESTEP_TOKEN_NUM : txt_seq_len] = 3

    vinput_mask = token_types == 1
    token_types_bin = (token_types > 0).to(token_types.dtype)

    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "token_types": token_types_bin,
        "vinput_mask": vinput_mask,
    }


def build_hidream_o1_attention_mask(
    token_types: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the mixed causal/full mask once while preparing a request.

    Text rows stay causal; image and timestep rows get full attention.
    """
    if token_types.ndim == 1:
        token_types = token_types.unsqueeze(0)
    elif token_types.ndim == 2 and token_types.shape[1] == 1 and token_types.shape[0] != 1:
        token_types = token_types.transpose(0, 1)
    if token_types.ndim != 2:
        raise ValueError(f"token_types must have one or two dimensions, got shape {tuple(token_types.shape)}")

    batch_size, seq_len = token_types.shape
    min_value = torch.finfo(dtype).min
    attention_mask = torch.triu(
        torch.full(
            (batch_size, seq_len, seq_len),
            min_value,
            dtype=dtype,
            device=token_types.device,
        ),
        diagonal=1,
    )
    attention_mask[token_types.bool()] = 0
    return attention_mask.unsqueeze(1)


def build_hidream_o1_full_attn_spans(
    token_types: torch.Tensor,
) -> list[list[tuple[int, int]]]:
    """Encode the mixed causal/full rows without allocating an ``S x S`` mask."""
    if token_types.ndim == 1:
        token_types = token_types.unsqueeze(0)
    elif token_types.ndim == 2 and token_types.shape[1] == 1 and token_types.shape[0] != 1:
        token_types = token_types.transpose(0, 1)
    if token_types.ndim != 2:
        raise ValueError(f"token_types must have one or two dimensions, got shape {tuple(token_types.shape)}")

    spans: list[list[tuple[int, int]]] = []
    for sample_types in token_types:
        full_rows = sample_types.bool().nonzero(as_tuple=False).flatten()
        if full_rows.numel() == 0:
            spans.append([])
            continue

        start = int(full_rows[0].item())
        end = int(full_rows[-1].item()) + 1
        expected = torch.arange(start, end, device=full_rows.device)
        if not torch.equal(full_rows, expected):
            raise ValueError("HiDream-O1 full-attention rows must form one contiguous span")
        spans.append([(start, end)])
    return spans


def build_hidream_o1_scheduler(
    *,
    scheduler_name: str = "default",
    num_inference_steps: int = 50,
    shift: float = 3.0,
    device: torch.device | str | None = None,
    timesteps_list: list[int] | None = None,
) -> FlowUniPCMultistepScheduler:
    if scheduler_name != "default":
        raise NotImplementedError(f"unsupported scheduler_name={scheduler_name!r}")

    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000,
        shift=shift,
        prediction_type="flow_prediction",
        use_dynamic_shifting=False,
    )
    scheduler.set_timesteps(num_inference_steps, device=device)

    if timesteps_list is not None:
        scheduler.timesteps = torch.tensor(timesteps_list, device=device, dtype=torch.long)
        sigmas = [t.item() / 1000.0 for t in scheduler.timesteps]
        sigmas.append(0.0)
        scheduler.sigmas = torch.tensor(sigmas, device=device)

    return scheduler


def get_hidream_o1_image_post_process_func(od_config: OmniDiffusionConfig):
    del od_config

    def post_process_func(
        output_data: tuple[torch.Tensor, int, int],
    ) -> Image.Image:
        """Convert HiDream-O1 patch-space output to a PIL RGB image."""
        if not isinstance(output_data, tuple) or len(output_data) != 3:
            raise TypeError(
                f"HiDreamO1 postprocess expects (patch_tensor, height, width), got {type(output_data).__name__}"
            )

        z, height, width = output_data

        if not torch.is_tensor(z):
            raise TypeError(
                f"HiDreamO1 postprocess expects the first tuple element to be a torch.Tensor, got {type(z).__name__}"
            )

        height = int(height)
        width = int(width)

        if height <= 0 or width <= 0:
            raise ValueError(
                f"HiDreamO1 postprocess requires positive height/width, got height={height}, width={width}"
            )

        if height % PATCH_SIZE != 0 or width % PATCH_SIZE != 0:
            raise ValueError(
                f"HiDreamO1 output resolution must be divisible by PATCH_SIZE={PATCH_SIZE}, got {width}x{height}"
            )

        h_patches = height // PATCH_SIZE
        w_patches = width // PATCH_SIZE
        expected_image_len = h_patches * w_patches
        expected_patch_dim = 3 * PATCH_SIZE * PATCH_SIZE

        if z.ndim != 3:
            raise ValueError(
                f"HiDreamO1 patch tensor must have shape (B, image_len, patch_dim), got shape={tuple(z.shape)}"
            )

        if z.shape[0] != 1:
            raise ValueError(f"HiDreamO1 postprocess currently expects batch size 1, got batch_size={z.shape[0]}")

        if z.shape[1] != expected_image_len:
            raise ValueError(
                "HiDreamO1 patch count mismatch: "
                f"got {z.shape[1]}, expected {expected_image_len} "
                f"for resolution {width}x{height}"
            )

        if z.shape[2] != expected_patch_dim:
            raise ValueError(f"HiDreamO1 patch dimension mismatch: got {z.shape[2]}, expected {expected_patch_dim}")

        # Preserve upstream decode order exactly:
        #   1. map patch values from [-1, 1] to [0, 1]
        #   2. restore BCHW image layout
        #   3. convert to float32 on CPU
        img = (z + 1) / 2

        img = rearrange(
            img.cpu().float(),
            "B (H W) (C p1 p2) -> B C (H p1) (W p2)",
            H=h_patches,
            W=w_patches,
            p1=PATCH_SIZE,
            p2=PATCH_SIZE,
        )

        arr = np.round(
            np.clip(
                img[0].numpy().transpose(1, 2, 0) * 255,
                0,
                255,
            )
        ).astype(np.uint8)

        return Image.fromarray(arr).convert("RGB")

    return post_process_func


class HiDreamO1ImagePipeline(nn.Module, DiffusionPipelineProfilerMixin, ProgressBarMixin):
    supports_request_batch = False
    dummy_run_num_frames: ClassVar[int] = 0
    _dit_modules: ClassVar[list[str]] = ["model.model.language_model"]

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.prefix = prefix
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

        self._validate_static_config()

        self.setup_diffusion_pipeline_profiler(
            profiler_targets=["forward"],
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler,
        )

        self._init_processor_and_model()
        self._validate_tms_token_id()

    def _validate_static_config(self) -> None:
        if self.dtype != torch.bfloat16:
            raise NotImplementedError(f"only bfloat16 is supported; got {self.dtype!r}")
        cfg_parallel_size = int(self.od_config.parallel_config.cfg_parallel_size)
        if cfg_parallel_size > 1:
            raise NotImplementedError(f"cfg_parallel_size={cfg_parallel_size} not supported")

    def _init_processor_and_model(self) -> None:
        logger.info(
            "HiDreamO1ImagePipeline: loading processor and model config from %s",
            self.model_dir,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_dir,
            revision=self.od_config.revision,
        )
        model_config = Qwen3VLConfig.from_pretrained(
            self.model_dir,
            revision=self.od_config.revision,
        )
        self.model = HiDreamO1ImageTransformer(
            model_config,
            quant_config=self.od_config.quantization_config,
            prefix=self.prefix,
        )

        self._add_special_tokens(self.processor)
        self.tokenizer = (
            self.processor if isinstance(self.processor, PreTrainedTokenizerBase) else self.processor.tokenizer
        )

    def _validate_tms_token_id(self) -> None:
        actual_tms_ids = self.tokenizer.encode(self.tokenizer.tms_token, add_special_tokens=False)
        expected_tms_id = int(self.model.model.tms_token_id)
        if actual_tms_ids != [expected_tms_id]:
            raise RuntimeError(f"tms_token maps to {actual_tms_ids}, model expects [{expected_tms_id}]")

    @staticmethod
    def _add_special_tokens(processor: ProcessorMixin | PreTrainedTokenizerBase) -> None:
        tok = processor if isinstance(processor, PreTrainedTokenizerBase) else processor.tokenizer
        tok.boi_token = "<|boi_token|>"
        tok.bor_token = "<|bor_token|>"
        tok.eor_token = "<|eor_token|>"
        tok.bot_token = "<|bot_token|>"
        tok.tms_token = "<|tms_token|>"

    def _resolve_generation_params(
        self,
        req: DiffusionRequestBatch,
    ) -> tuple[str, int, int, int, int, float]:
        sp = req.sampling_params

        if sp.timesteps is not None:
            raise NotImplementedError("sp.timesteps not supported")
        if sp.latents is not None:
            raise NotImplementedError("sp.latents not supported")
        if sp.num_outputs_per_prompt != 1:
            raise NotImplementedError(f"num_outputs_per_prompt={sp.num_outputs_per_prompt} not supported")

        if not req.prompts:
            raise ValueError("req.prompts is empty")
        if len(req.prompts) != 1:
            raise NotImplementedError(f"only one prompt per request; got {len(req.prompts)}")
        first_prompt = req.prompts[0]
        if isinstance(first_prompt, str):
            prompt = first_prompt
        elif isinstance(first_prompt, dict):
            raw_prompt = first_prompt.get("prompt")
            if not isinstance(raw_prompt, str):
                raise TypeError(f"dict prompt must contain a str 'prompt' field; got {type(raw_prompt).__name__}")
            prompt = raw_prompt
        else:
            raise TypeError(f"prompt must be str or dict; got {type(first_prompt).__name__}")

        raw_h = 2048 if sp.height is None else int(sp.height)
        raw_w = 2048 if sp.width is None else int(sp.width)
        if raw_h <= 0 or raw_w <= 0:
            raise ValueError(f"h/w must be positive, got h={raw_h} w={raw_w}")
        snapped_w, snapped_h = find_closest_resolution(raw_w, raw_h)
        if (snapped_w, snapped_h) != (raw_w, raw_h):
            logger.info(
                "HiDreamO1: resolution snapped from %dx%d to %dx%d",
                raw_w,
                raw_h,
                snapped_w,
                snapped_h,
            )

        steps = 50 if sp.num_inference_steps is None else int(sp.num_inference_steps)
        if steps <= 0:
            raise ValueError(f"num_inference_steps must be positive, got {steps}")

        generator = sp.generator
        if isinstance(generator, list):
            if len(generator) != 1:
                raise NotImplementedError("generator list only supported for single-output requests with one generator")
            generator = generator[0]

        generator_seed: int | None = None
        if generator is not None:
            if not hasattr(generator, "initial_seed"):
                raise TypeError(f"sp.generator must expose initial_seed(); got {type(generator).__name__}")
            generator_seed = int(generator.initial_seed())

        if sp.seed is not None and generator_seed is not None and int(sp.seed) != generator_seed:
            raise ValueError(f"seed/generator mismatch: seed={int(sp.seed)} generator.initial_seed()={generator_seed}")

        if sp.seed is None and generator_seed is None:
            raise RuntimeError("expected request initialization to resolve a concrete seed")
        seed = int(sp.seed) if sp.seed is not None else generator_seed
        assert seed is not None
        guidance_scale = float(sp.guidance_scale)
        return prompt, snapped_h, snapped_w, steps, seed, guidance_scale

    def _prepare_noise_and_patchify(
        self,
        height: int,
        width: int,
        seed: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        gen = torch.Generator("cpu").manual_seed(seed + 1)
        noise = NOISE_SCALE * torch.randn((1, 3, height, width), generator=gen)
        noise = noise.to(device, dtype)
        return rearrange(
            noise,
            "B C (H p1) (W p2) -> B (H W) (C p1 p2)",
            p1=PATCH_SIZE,
            p2=PATCH_SIZE,
        )

    def _forward_once(
        self,
        sample: HiDreamO1TextSample,
        z_in: torch.Tensor,
        t_pixeldit: torch.Tensor,
    ) -> torch.Tensor:
        with torch.autocast(self.device.type, dtype=self.dtype, cache_enabled=False):
            outputs = self.model(
                input_ids=sample["input_ids"],
                position_ids=sample["position_ids"],
                vinputs=z_in,
                timestep=t_pixeldit.reshape(-1).to(self.device),
                attention_mask=sample["attention_mask"],
                full_attn_spans=sample["full_attn_spans"],
            )

        if outputs.x_pred is None:
            raise RuntimeError("model returned x_pred=None")
        if outputs.x_pred.shape[-1] != z_in.shape[-1]:
            raise RuntimeError(f"x_pred patch_dim {outputs.x_pred.shape[-1]} != z_in {z_in.shape[-1]}")

        mask = sample["vinput_mask"][0]
        return outputs.x_pred[0, mask].unsqueeze(0)

    def _build_and_validate_sample(
        self,
        prompt: str,
        height: int,
        width: int,
        expected_image_len: int,
    ) -> HiDreamO1TextSample:
        sample = build_t2i_text_sample(
            prompt=prompt,
            height=height,
            width=width,
            tokenizer=self.tokenizer,
            processor=self.processor,
            model_config=self.model.config,
        )
        actual_image_len = int(sample["vinput_mask"].sum().item())
        if actual_image_len != expected_image_len:
            raise RuntimeError(
                f"vinput_mask has {actual_image_len} slots, expected {expected_image_len} for {height}x{width}"
            )
        full_attn_spans = build_hidream_o1_full_attn_spans(sample["token_types"])
        inputs = {key: value.to(self.device) for key, value in sample.items()}
        attention_mask = None
        if not self.model.supports_piecewise_attention:
            attention_mask = build_hidream_o1_attention_mask(
                inputs["token_types"],
                dtype=self.dtype,
            )
        return {
            "input_ids": inputs["input_ids"],
            "position_ids": inputs["position_ids"],
            "attention_mask": attention_mask,
            "full_attn_spans": full_attn_spans,
            "vinput_mask": inputs["vinput_mask"],
        }

    @torch.inference_mode()
    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        prompt, height, width, num_inference_steps, seed, guidance_scale = self._resolve_generation_params(req)

        expected_image_len = (height // PATCH_SIZE) * (width // PATCH_SIZE)
        cond_sample = self._build_and_validate_sample(
            prompt,
            height,
            width,
            expected_image_len,
        )
        samples: list[HiDreamO1TextSample] = [cond_sample]
        if guidance_scale > 1.0:
            uncond_sample = self._build_and_validate_sample(
                " ",
                height,
                width,
                expected_image_len,
            )
            samples.append(uncond_sample)

        z = self._prepare_noise_and_patchify(
            height,
            width,
            seed,
            self.dtype,
            self.device,
        )

        sched = build_hidream_o1_scheduler(
            scheduler_name="default",
            num_inference_steps=num_inference_steps,
            shift=3.0,
            device=self.device,
        )

        with self.progress_bar(total=len(sched.timesteps)) as pbar:
            for step_t in sched.timesteps:
                step_t_f32 = step_t.float()
                t_pixeldit = 1.0 - step_t_f32 / 1000.0
                sigma = (step_t_f32 / 1000.0).to(dtype=torch.float32).clamp_min(T_EPS)

                x_pred_cond = self._forward_once(samples[0], z, t_pixeldit)
                v_cond = (x_pred_cond.to(torch.float32) - z.to(torch.float32)) / sigma

                if len(samples) > 1:
                    x_pred_uncond = self._forward_once(
                        samples[1],
                        z,
                        t_pixeldit,
                    )
                    v_uncond = (x_pred_uncond.to(torch.float32) - z.to(torch.float32)) / sigma
                    v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)
                else:
                    v_guided = v_cond

                model_output = -v_guided

                z = sched.step(
                    model_output,
                    step_t_f32,
                    z.float(),
                    return_dict=False,
                )[0].to(self.dtype)
                pbar.update()

        return DiffusionOutput(
            output=(z, height, width),
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
