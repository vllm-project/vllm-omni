# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar

import torch
from omegaconf import DictConfig
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

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
from vllm_omni.diffusion.models.hy_nomas.configuration_hy_nomas import DECODER_FILE, DIT_FILE, REQUIRED_FILES
from vllm_omni.diffusion.models.hy_nomas.parallel import initialize_reference_parallelism
from vllm_omni.diffusion.models.hy_nomas.request import normalize_request
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.offloader.config import (
    DIT_COMPONENT,
    TEXT_ENCODER_COMPONENT,
    OffloadStrategy,
    resolve_offload_strategy,
)
from vllm_omni.diffusion.offloader.sequential_backend import apply_sequential_offload, remove_sequential_offload
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.errors import OmniClientError
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific


def build_reference_config(model_root: Path, text_encoder: str, sp_size: int = 1) -> DictConfig:
    from configs import config as reference_config
    from omegaconf import OmegaConf

    directory = Path(reference_config.__file__).parent
    base = OmegaConf.load(directory / "config.yaml")
    inference = OmegaConf.load(directory / "experiment" / "inference.yaml")
    # 固定版本的配置仅合并这两个文件，不读取进程命令行或修改 Hydra 全局状态。
    del base["defaults"]
    del inference["defaults"]
    config = OmegaConf.merge(base, inference)
    config.dit_weight = str(model_root / DIT_FILE)
    config.model.vae.checkpoint = str(model_root / DECODER_FILE)
    config.model.vae.parallel_mode = "none"
    config.model.text_encoder.model_name = text_encoder
    config.model.dit.seg_ar_attention_backend = "flex"
    config.parallelism.sp_size = sp_size
    config.parallelism.shard_text_encoder = False
    config.inference.save_latents = False
    config.inference.save_prompt_txt = False
    config.inference.video_size = [704, 1280]
    config = reference_config._flatten_cfg(config)
    OmegaConf.set_struct(config, True)
    return config


def validate_config(config: OmniDiffusionConfig) -> None:
    parallel = config.parallel_config
    if parallel.world_size != parallel.ulysses_degree or parallel.use_hsdp:
        raise ValueError("Hy-Nomas only supports Ulysses SP, without TP, PP, DP, CFG parallelism or HSDP.")
    if 16 % parallel.ulysses_degree:
        raise ValueError("Hy-Nomas ulysses_degree must divide its 16 attention heads.")
    if parallel.ulysses_mode != "strict" or parallel.ulysses_a2a_permute:
        raise ValueError("Hy-Nomas requires strict Ulysses SP with the reference all-to-all implementation.")
    if parallel.vae_patch_parallel_size != 1 or parallel.text_encoder_tp_size != 1 or parallel.enable_expert_parallel:
        raise ValueError("Hy-Nomas does not support VAE, text encoder or expert parallelism.")
    if config.dtype != torch.bfloat16:
        raise ValueError("Hy-Nomas requires dtype=bfloat16.")
    if config.cache_backend not in (None, "none"):
        raise ValueError("Hy-Nomas does not support a diffusion cache backend.")
    if config.quantization_config is not None or config.lora_path:
        raise ValueError("Hy-Nomas currently supports the unquantized distilled checkpoint without LoRA.")
    if resolve_offload_strategy(config) not in (OffloadStrategy.NONE, OffloadStrategy.MODEL_LEVEL):
        raise ValueError("Hy-Nomas supports module CPU offload; layer offload is not supported.")
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
        from vllm_omni.diffusion.models.hy_nomas.runtime import HyNomasRuntime

        root = Path(od_config.model)
        if not root.is_dir():
            root = Path(
                download_weights_from_hf_specific(
                    od_config.model,
                    cache_dir=None,
                    allow_patterns=list(REQUIRED_FILES),
                    revision=od_config.revision,
                    require_all=True,
                )
            )
        for relative in REQUIRED_FILES:
            if not (root / relative).is_file():
                raise FileNotFoundError(f"Missing Hy-Nomas checkpoint: {root / relative}")
        text_encoder = (od_config.custom_pipeline_args or {}).get("text_encoder", "Qwen/Qwen2.5-VL-7B-Instruct")
        if not isinstance(text_encoder, str) or not text_encoder:
            raise ValueError("Hy-Nomas text_encoder must be a nonempty model ID or local path.")
        self.sp_size = od_config.parallel_config.ulysses_degree
        mesh = initialize_reference_parallelism(self.sp_size, self.device)
        config = build_reference_config(root, text_encoder, self.sp_size)
        self.pipe = HyNomasRuntime(
            args=config, device=self.device, dit_model_path=config.dit_weight, world_mesh=mesh
        ).eval()

    def enable_omni_model_cpu_offload(
        self,
        *,
        device: torch.device,
        pin_memory: bool = True,
        use_hsdp: bool = False,
        offload_components: frozenset[str] | None = None,
    ) -> None:
        if use_hsdp:
            raise ValueError("Hy-Nomas does not support HSDP.")
        if self.pipe.omni_cpu_offload:
            return
        dit = self.pipe.transformer
        stages = [self.pipe.text_encoder, self.pipe.vae]
        selected_dits = [dit] if offload_components is None or DIT_COMPONENT in offload_components else []
        selected_stages = (
            stages
            if offload_components is None
            else ([self.pipe.text_encoder] if TEXT_ENCODER_COMPONENT in offload_components else [])
        )
        apply_sequential_offload(
            dit_modules=[dit],
            encoder_modules=stages,
            device=device,
            pin_memory=pin_memory,
            offload_initial_dits=bool(selected_dits),
            offload_dit_modules=selected_dits,
            offload_encoder_modules=selected_stages,
        )
        self.pipe.omni_cpu_offload = True

    def disable_omni_model_cpu_offload(self) -> None:
        if self.pipe.omni_cpu_offload:
            remove_sequential_offload([self.pipe.transformer, self.pipe.text_encoder, self.pipe.vae])
            self.pipe.omni_cpu_offload = False

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
        if self.sp_size > 1:
            tokens_per_frame = (request.height // self.pipe.s_ratio // self.pipe.patch_size[1]) * (
                request.width // self.pipe.s_ratio // self.pipe.patch_size[2]
            )
            if tokens_per_frame % self.sp_size:
                raise OmniClientError(
                    f"Hy-Nomas spatial tokens per frame ({tokens_per_frame}) must be divisible by "
                    f"ulysses_degree ({self.sp_size})."
                )
        self._reset_cache()
        try:
            # vLLM 导入时关闭 cuDNN SDPA；此处恢复参考实现的后端集合，退出后还原。
            with sdpa_kernel(
                [
                    SDPBackend.CUDNN_ATTENTION,
                    SDPBackend.FLASH_ATTENTION,
                    SDPBackend.EFFICIENT_ATTENTION,
                    SDPBackend.MATH,
                ]
            ):
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
