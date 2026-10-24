# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar

import torch
from omegaconf import DictConfig
from torch import nn

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.media import (
    DiffusionMediaOutput,
    VideoMediaOutput,
    VideoTensorEncoding,
    VideoTensorLayout,
    VideoTensorSpec,
    VideoValueRange,
)
from vllm_omni.diffusion.models.hy_nomas.request import normalize_request
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.offloader.config import offload_enabled
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific
from vllm_omni.model_extras.hy_nomas import HY_NOMAS_DECODER_FILE, HY_NOMAS_DIT_FILE


def build_reference_config(model_root: Path, text_encoder: str) -> DictConfig:
    from configs import config as reference_config
    from omegaconf import OmegaConf

    directory = Path(reference_config.__file__).parent
    base = OmegaConf.load(directory / "config.yaml")
    inference = OmegaConf.load(directory / "experiment" / "inference.yaml")
    # 固定版本的配置仅合并这两个文件，不读取进程命令行或修改 Hydra 全局状态。
    del base["defaults"]
    del inference["defaults"]
    config = OmegaConf.merge(base, inference)
    config.dit_weight = str(model_root / HY_NOMAS_DIT_FILE)
    config.model.vae.checkpoint = str(model_root / HY_NOMAS_DECODER_FILE)
    config.model.text_encoder.model_name = text_encoder
    config.model.dit.seg_ar_attention_backend = "flex"
    config.parallelism.sp_size = 1
    config.inference.save_latents = False
    config.inference.save_prompt_txt = False
    config.inference.video_size = [704, 1280]
    config = reference_config._flatten_cfg(config)
    OmegaConf.set_struct(config, True)
    return config


def validate_config(config: OmniDiffusionConfig) -> None:
    if config.parallel_config.world_size != 1 or config.parallel_config.use_hsdp:
        raise ValueError("Hy-Nomas currently supports a single CUDA device.")
    if config.dtype != torch.bfloat16:
        raise ValueError("Hy-Nomas requires dtype=bfloat16.")
    if config.cache_backend not in (None, "none"):
        raise ValueError("Hy-Nomas does not support a diffusion cache backend.")
    if config.quantization_config is not None or config.lora_path:
        raise ValueError("Hy-Nomas currently supports the unquantized distilled checkpoint without LoRA.")
    if offload_enabled(config):
        raise ValueError("Hy-Nomas CPU offload is not supported by this integration.")
    custom = config.custom_pipeline_args or {}
    if custom.keys() - {"text_encoder"}:
        raise ValueError("Hy-Nomas custom_pipeline_args only supports text_encoder.")


class HyNomasPipeline(nn.Module, SupportsComponentDiscovery):
    # 通用预热使用两步和 CFG=0，不属于该蒸馏模型的有效采样配置。
    dummy_run_num_frames: ClassVar[int] = 0
    _dit_modules: ClassVar[list[str]] = ["pipe.transformer"]
    _encoder_modules: ClassVar[list[str]] = ["pipe.text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["pipe.vae"]

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        validate_config(od_config)
        self.device = get_local_device()
        if self.device.type != "cuda":
            raise ValueError("Hy-Nomas requires CUDA.")
        # 必需依赖直接导入；安装固定参考版本的方法见模型使用文档。
        from hymm.pipeline.causal_ar import CPTCausalInferencePipeline

        root = Path(od_config.model)
        required = [HY_NOMAS_DIT_FILE, HY_NOMAS_DECODER_FILE]
        if not root.is_dir():
            root = Path(
                download_weights_from_hf_specific(
                    od_config.model,
                    cache_dir=None,
                    allow_patterns=required,
                    revision=od_config.revision,
                    require_all=True,
                )
            )
        for relative in required:
            if not (root / relative).is_file():
                raise FileNotFoundError(f"Missing Hy-Nomas checkpoint: {root / relative}")
        text_encoder = (od_config.custom_pipeline_args or {}).get("text_encoder", "Qwen/Qwen2.5-VL-7B-Instruct")
        if not isinstance(text_encoder, str) or not text_encoder:
            raise ValueError("Hy-Nomas text_encoder must be a nonempty model ID or local path.")
        config = build_reference_config(root, text_encoder)
        self.pipe = CPTCausalInferencePipeline(args=config, device=self.device, dit_model_path=config.dit_weight).eval()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # 参考实现已在构造时加载三个组件，禁止标准 loader 再传入另一组权重。
        if next(iter(weights), None) is not None:
            raise ValueError("Hy-Nomas loads weights through its reference component loaders.")
        return set(dict(self.named_parameters()))

    def _reset_cache(self) -> None:
        from hymm.models.dit.modules.attention.range_attention import clear_range_cache

        self.pipe.kv_cache = None
        self.pipe.cache_mask_list = None
        self.pipe.cache_valid_len = 0
        self.pipe.global_frame_counter = 0
        self.pipe.segment_token_lengths = []
        self.pipe._interactive_cached_action_tails = []
        self.pipe._interactive_cached_seg_video_ranges = []
        self.pipe._interactive_segment_count = 0
        self.pipe._interactive_segment_cache_tokens = []
        self.pipe._interactive_segment_latent_lengths = []
        self.pipe._reset_interactive_noise_state()
        self.pipe._reset_interactive_pixelumm_decode_state()
        clear_range_cache()

    @torch.no_grad()
    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        if req.num_reqs != 1:
            raise ValueError("Hy-Nomas processes one request at a time.")
        request = normalize_request(req.prompts[0], req.sampling_params)
        self._reset_cache()
        try:
            video = self.pipe.inference(
                prompts=request.prompts,
                num_frames=request.num_frames,
                height=request.height,
                width=request.width,
                guidance_scale=1.0,
                camera_guidance_scale=1.0,
                num_inference_steps=request.steps,
                flow_shift=5.0,
                seed=request.seed,
                negative_prompt=request.negative_prompt,
                segment_frame_boundaries=request.boundaries,
                camera_trajectory=request.camera_trajectory,
                segment_has_camera=request.segment_has_camera,
                preserve_kv_cache=False,
            )
        finally:
            self._reset_cache()
        expected = (1, 3, request.num_frames, request.height, request.width)
        if tuple(video.shape) != expected:
            raise RuntimeError(f"Hy-Nomas returned video shape {tuple(video.shape)}; expected {expected}.")
        return DiffusionOutput(
            media=DiffusionMediaOutput(
                video=VideoMediaOutput(
                    tensor=video,
                    spec=VideoTensorSpec(
                        layout=VideoTensorLayout.BCTHW,
                        encoding=VideoTensorEncoding.NORMALIZED_FLOAT,
                        value_range=VideoValueRange.ZERO_TO_ONE,
                    ),
                )
            )
        )
