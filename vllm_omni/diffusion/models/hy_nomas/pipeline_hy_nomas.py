# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from vllm.logger import init_logger
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.parallel_state import get_sp_group
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.interaction.mixin import InteractionMixin
from vllm_omni.diffusion.interaction.types import ChunkMediaSpec
from vllm_omni.diffusion.media import (
    DiffusionMediaOutput,
    VideoMediaOutput,
    VideoTensorEncoding,
    VideoTensorLayout,
    VideoTensorSpec,
    VideoValueRange,
)
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.hy_nomas.configuration_hy_nomas import (
    REQUIRED_FILES,
    HyNomasARConfig,
)
from vllm_omni.diffusion.models.hy_nomas.hy_nomas_transformer import (
    build_transformer,
    commit_cache,
    initialize_reference_parallelism,
    load_component_weights,
    load_transformer_weights,
    reference_cache_views,
)
from vllm_omni.diffusion.models.hy_nomas.interaction import HyNomasInteractionSession, validate_camera_payload
from vllm_omni.diffusion.models.hy_nomas.processing_hy_nomas import (
    action_conditioning,
    normalize_request,
    plan_chunks,
    prepare_camera,
)
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.offloader.config import (
    DIT_COMPONENT,
    TEXT_ENCODER_COMPONENT,
    OffloadStrategy,
    resolve_offload_strategy,
)
from vllm_omni.diffusion.offloader.sequential_backend import (
    apply_sequential_offload,
    remove_sequential_offload,
    sequential_offload_component,
)
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.errors import OmniClientError
from vllm_omni.experimental.ar_diffusion.capability import (
    ARDiffusionKVBranchSpec,
    ARDiffusionKVCacheSpec,
    ARDiffusionSegmentedKVSpec,
)
from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState
from vllm_omni.experimental.ar_diffusion.tick_protocol import ARDiffusionChunkMetadata, ARDiffusionTickRequest
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

logger = init_logger(__name__)


def build_reference_config(model_root: Path, text_encoder: str, sp_size: int = 1) -> DictConfig:
    from configs import config as reference_config
    from omegaconf import OmegaConf

    directory = Path(reference_config.__file__).parent
    base = OmegaConf.load(directory / "config.yaml")
    inference = OmegaConf.load(directory / "experiment" / "inference.yaml")
    # Compose component settings without reading CLI arguments or changing Hydra's global state.
    del base["defaults"]
    del inference["defaults"]
    config = OmegaConf.merge(base, inference)
    config.dit_weight = str(model_root / "transformer")
    config.model.vae.checkpoint = str(model_root / "decoder")
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


def read_component_config(root: Path, component: str, filename: str, class_name: str) -> dict[str, Any]:
    path = root / component / filename
    config = json.loads(path.read_text())
    if not isinstance(config, dict) or config.get("_class_name") != class_name:
        raise ValueError(f"Hy-Nomas expects {class_name} in {path}.")
    return config


def build_decoder(config: dict[str, Any], device: torch.device) -> nn.Module:
    from hymm.models.vae.pixelumm_decoder.autoencoder import PixelUMMDCAELatentDecoder
    from hymm.models.vae.pixelumm_decoder.model import DCAEVPixelUMMDecoder, GenExpertArch

    arch = GenExpertArch(**{name: config[name] for name in GenExpertArch.__dataclass_fields__})
    local_pixels = config["local_attention_tile_pixels"]
    if type(local_pixels) is not int or local_pixels < 0 or local_pixels % 16:
        raise ValueError("Hy-Nomas local_attention_tile_pixels must be a nonnegative multiple of 16.")
    with set_default_torch_dtype(torch.bfloat16), torch.device(device):
        decoder = DCAEVPixelUMMDecoder(arch, local_attention_tile=local_pixels // 16)
    vae = PixelUMMDCAELatentDecoder(
        decoder,
        **{
            name: config[name]
            for name in (
                "denoising_step_list",
                "num_steps",
                "seed",
                "temporal_window_latents",
                "temporal_overlap_latents",
                "decoder_tile_pixels",
                "decoder_tile_overlap_pixels",
                "tail_hold_latents",
            )
        },
    )
    return vae.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)


def validate_config(config: OmniDiffusionConfig) -> None:
    from vllm_omni.diffusion.executor.abstract import DiffusionExecutor
    from vllm_omni.diffusion.executor.uniproc_executor import UniProcDiffusionExecutor

    # Reference components initialize global distributed state; reject in-process use before importing them.
    implicit_uni = config.distributed_executor_backend is None and (config.num_gpus or 1) == 1
    if implicit_uni or issubclass(DiffusionExecutor.get_class(config), UniProcDiffusionExecutor):
        raise ValueError("Hy-Nomas requires process isolation; set distributed_executor_backend='mp' or 'ray'.")
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
    if config.flow_shift not in (None, 5.0):
        raise ValueError("Hy-Nomas requires flow_shift=5.")
    if not config.enforce_eager and config.diffusion_compile_granularity == "full":
        raise ValueError("Hy-Nomas does not support full transformer compilation; use enforce_eager=True.")
    if resolve_offload_strategy(config) not in (OffloadStrategy.NONE, OffloadStrategy.MODEL_LEVEL):
        raise ValueError("Hy-Nomas supports module CPU offload; layer offload is not supported.")
    custom = config.custom_pipeline_args or {}
    allowed = {"text_encoder", *(f"ar_{name}" for name in HyNomasARConfig.__dataclass_fields__)}
    if custom.keys() - allowed:
        raise ValueError(f"Unsupported Hy-Nomas custom_pipeline_args: {sorted(custom.keys() - allowed)}")
    HyNomasARConfig.from_args(custom)
    if config.step_execution and not config.streaming_output:
        raise ValueError("Hy-Nomas step execution requires streaming_output=true.")
    if config.step_execution and config.engine_backend not in (
        "ar_diffusion",
        "vllm_omni.experimental.ar_diffusion.engine.ARDiffusionEngine",
    ):
        raise ValueError("Hy-Nomas step execution requires the ARDiffusionEngine backend.")
    text_encoder = custom.get("text_encoder")
    if "text_encoder" in custom and (not isinstance(text_encoder, str) or not text_encoder.strip()):
        raise ValueError("Hy-Nomas text_encoder must be a nonempty model ID or local path.")


@dataclass
class HyNomasSession:
    height: int
    width: int
    seed: int
    noise: torch.Tensor
    generator: torch.Generator
    decoder: Any
    history_mask: torch.Tensor
    latent_frames: int = 0
    segments: int = 0
    tick_index: int = 0
    emitted_frames: int = 0
    # Original masks are retained separately from per-forward visibility masks.
    action_tails: list[tuple[int, int, int]] = field(default_factory=list)
    offline_cache: list[dict[str, torch.Tensor]] | None = None


class HyNomasPipeline(nn.Module, InteractionMixin, SupportsComponentDiscovery):
    supports_step_execution: ClassVar[bool] = True
    supports_chunk_step_grouping: ClassVar[bool] = True
    # Generic warmup uses two steps and CFG=0, which the distilled checkpoint does not support.
    dummy_run_num_frames: ClassVar[int] = 0
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        del prefix
        validate_config(od_config)
        self.device = get_local_device()
        if self.device.type != "cuda":
            raise ValueError("Hy-Nomas requires CUDA.")
        self.initialize_ar(od_config.custom_pipeline_args or {})
        self.sp_size = od_config.parallel_config.ulysses_degree
        self.target_dtype = torch.bfloat16
        self._cpu_offload = False
        root = Path(od_config.model)
        text_encoder_override = (od_config.custom_pipeline_args or {}).get("text_encoder")
        if not root.is_dir():
            root = Path(
                download_weights_from_hf_specific(
                    od_config.model,
                    cache_dir=None,
                    allow_patterns=[
                        *REQUIRED_FILES,
                        *([] if text_encoder_override is not None else ["text_encoder/*", "tokenizer/*"]),
                    ],
                    revision=od_config.revision,
                    require_all=True,
                )
            )
        for relative in REQUIRED_FILES:
            if not (root / relative).is_file():
                raise FileNotFoundError(f"Missing Hy-Nomas checkpoint: {root / relative}")
        index = json.loads((root / "modular_model_index.json").read_text())
        if not isinstance(index, dict) or index.get("_class_name") != "HyNomasModularPipeline":
            raise ValueError("Hy-Nomas requires a HyNomasModularPipeline checkpoint.")
        transformer_config = read_component_config(root, "transformer", "config.json", "HyNomasTransformer3DModel")
        decoder_config = read_component_config(root, "decoder", "config.json", "PixelUMMDecoder")
        scheduler_config = read_component_config(
            root, "scheduler", "scheduler_config.json", "FlowMatchEulerDiscreteScheduler"
        )
        if scheduler_config.get("shift") != 5.0 or scheduler_config.get("num_train_timesteps") != 1000:
            raise ValueError("Hy-Nomas requires a 1000-timestep Euler scheduler with shift=5.")
        if any(
            scheduler_config.get(name)
            for name in (
                "use_dynamic_shifting",
                "invert_sigmas",
                "shift_terminal",
                "stochastic_sampling",
                "use_beta_sigmas",
                "use_exponential_sigmas",
                "use_karras_sigmas",
            )
        ):
            raise ValueError("Hy-Nomas requires the checkpoint's standard deterministic Euler schedule.")
        text_encoder = text_encoder_override or str(root / "text_encoder")
        initialize_reference_parallelism(self.sp_size, self.device)
        self.args = build_reference_config(root, text_encoder, self.sp_size)
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder=component,
                revision=od_config.revision,
                prefix=prefix,
                fall_back_to_pt=False,
            )
            for component, prefix in (("transformer", "transformer."), ("decoder", "vae."))
        ]
        from configs.constants import PROMPT_TEMPLATE
        from hymm.models.text_encoder import TextEncoder

        self.transformer = build_transformer(self.args, self.device, transformer_config)
        text_config = self.args.text_encoder
        self.text_encoder = (
            TextEncoder(
                text_encoder_type=self.args.text_encoder_type,
                text_encoder_path=text_encoder,
                tokenizer_type=text_config.tokenizer,
                tokenizer_path=text_encoder_override or str(root / "tokenizer"),
                max_length=text_config.text_len if not text_config.use_ll_cap_dataset else text_config.text_len_ll_cap,
                text_encoder_precision=text_config.text_encoder_precision,
                prompt_template=PROMPT_TEMPLATE[text_config.prompt_template] if text_config.prompt_template else None,
                prompt_template_video=PROMPT_TEMPLATE[text_config.prompt_template_video]
                if text_config.prompt_template_video
                else None,
                hidden_state_skip_layer=2,
                apply_final_norm=False,
                reproduce=False,
                logger=logger,
                device=self.device,
            )
            .eval()
            .requires_grad_(False)
        )
        self.vae = build_decoder(decoder_config, self.device)
        self.s_ratio = self.vae.spatial_compression_ratio
        self.patch_size = self.transformer.patch_size
        self.in_channels = self.transformer.config.in_channels
        self.num_double_blocks = len(self.transformer.double_blocks)
        self.heads_num = self.transformer.heads_num
        self.hidden_size = self.transformer.hidden_size
        self.cache_noise_sigma = float(self.args.inference.cache_noise_sigma)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = set()
        components = set()
        for component, entries in groupby(weights, key=lambda item: item[0].partition(".")[0]):
            if component not in {"transformer", "vae"}:
                raise ValueError(f"Unexpected Hy-Nomas component weight: {component}")
            if component in components:
                raise ValueError(f"Duplicate Hy-Nomas component weight stream: {component}")
            components.add(component)
            loader = load_transformer_weights if component == "transformer" else load_component_weights
            names = loader(getattr(self, component), ((name.partition(".")[2], value) for name, value in entries))
            loaded.update(f"{component}.{name}" for name in names)
        if components != {"transformer", "vae"}:
            raise ValueError(
                f"Hy-Nomas checkpoint is missing component weights: {sorted({'transformer', 'vae'} - components)}"
            )
        loaded.update(f"text_encoder.{name}" for name, _ in self.text_encoder.named_parameters())
        return loaded

    def setup_compile(self) -> None:
        raise NotImplementedError("Hy-Nomas uses forward_with_cache; set enforce_eager=True.")

    @contextmanager
    def _component_context(self, module: nn.Module):
        if self._resident_component_id == id(module):
            yield
            return
        residency = sequential_offload_component(module) if self._cpu_offload else nullcontext()
        previous = self._resident_component_id
        # vLLM disables cuDNN SDPA on import; retain the reference component backend policy.
        try:
            with (
                residency,
                sdpa_kernel(
                    [
                        SDPBackend.CUDNN_ATTENTION,
                        SDPBackend.FLASH_ATTENTION,
                        SDPBackend.EFFICIENT_ATTENTION,
                        SDPBackend.MATH,
                    ]
                ),
            ):
                self._resident_component_id = id(module)
                yield
        finally:
            self._resident_component_id = previous

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
        if self._cpu_offload:
            return
        dits = [self.transformer] if offload_components is None or DIT_COMPONENT in offload_components else []
        stages = [self.text_encoder, self.vae]
        selected = (
            stages
            if offload_components is None
            else ([self.text_encoder] if TEXT_ENCODER_COMPONENT in offload_components else [])
        )
        apply_sequential_offload(
            dit_modules=[self.transformer],
            encoder_modules=stages,
            device=device,
            pin_memory=pin_memory,
            offload_initial_dits=bool(dits),
            offload_dit_modules=dits,
            offload_encoder_modules=selected,
        )
        self._cpu_offload = True

    def disable_omni_model_cpu_offload(self) -> None:
        if self._cpu_offload:
            remove_sequential_offload([self.transformer, self.text_encoder, self.vae])
            self._cpu_offload = False

    def _encode_prompt(self, prompt: str) -> dict[str, torch.Tensor | None]:
        with self._component_context(self.text_encoder):
            tokens = self.text_encoder.text2tokens([prompt], data_type="video", max_length=self.args.text_len)
            output = self.text_encoder.encode(tokens, data_type="video", device=self.device, is_uncond=False)
        return {
            "text_states": output.hidden_state.to(dtype=self.target_dtype, device=self.device),
            "text_mask": None
            if output.attention_mask is None
            else output.attention_mask.to(dtype=self.target_dtype, device=self.device),
        }

    def _new_scheduler(self, steps: int):
        from hymm.pipeline.solvers.euler import EulerSolver

        scheduler = EulerSolver(shift=5.0, reverse=True, solver="euler")
        scheduler.set_timesteps(steps, device=self.device)
        return scheduler

    def _prepare_vae_decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents / self.vae.config.scaling_factor
        if getattr(self.vae.config, "shift_factor", None):
            latents = latents + self.vae.config.shift_factor
        return latents.to(next(self.vae.parameters()).dtype)

    def initialize_ar(self, args: Mapping[str, Any]) -> None:
        self.ar_config = HyNomasARConfig.from_args(args)
        self._ar_state: ARDiffusionKVState | None = None
        self._ar_sessions: dict[str, HyNomasSession] = {}
        self._offline_session: HyNomasSession | None = None
        self._resident_component_id: int | None = None

    def ar_diffusion_kv_cache_spec(self) -> ARDiffusionKVCacheSpec:
        config = self.ar_config
        frame_tokens = (config.height // self.s_ratio // self.patch_size[1]) * (
            config.width // self.s_ratio // self.patch_size[2]
        )
        text_tokens = int(self.args.text_len)
        retained = config.max_latent_frames * frame_tokens + config.max_segments * text_tokens
        pending = config.chunk_latent_frames * frame_tokens + text_tokens
        latent_elements = config.max_latent_frames * (config.height // self.s_ratio) * (config.width // self.s_ratio)
        # Fixed noise, the decoder's latent timeline, pending pixels and text.
        model_bytes = latent_elements * (self.in_channels + self.transformer.config.out_channels) * 2
        model_bytes += config.height * config.width * 3 * 4 * 40 + config.max_segments * text_tokens * 3584 * 2
        return ARDiffusionKVCacheSpec(
            num_layers=self.num_double_blocks,
            num_kv_heads=self.heads_num // self.sp_size,
            head_size=self.hidden_size // self.heads_num,
            tokens_per_frame=frame_tokens,
            frames_per_block=1,
            window_frames=1,
            kv_branches=(ARDiffusionKVBranchSpec("cond", 0),),
            session_capacity=config.session_capacity,
            segmented=ARDiffusionSegmentedKVSpec(
                block_size=256,
                max_retained_tokens=retained,
                max_retained_segments=config.max_latent_frames,
                max_pending_tokens=pending,
                max_pending_segments=1,
                initial_staging_tokens=retained + pending,
            ),
            model_owned_state_bytes_per_session=model_bytes,
        )

    @contextmanager
    def bind_ar_diffusion_state(self, session_id: str, state: ARDiffusionKVState) -> Iterator[None]:
        if self._ar_state is not None or state.session_id != session_id:
            raise RuntimeError("Hy-Nomas AR state binding is nested or has a different session ID.")
        self._ar_state = state
        try:
            yield
        finally:
            self._ar_state = None

    def reset_ar_diffusion_session(self, session_id: str) -> None:
        self._ar_sessions.pop(session_id, None)

    def close_ar_diffusion_session(self, session_id: str) -> None:
        self.reset_ar_diffusion_session(session_id)

    def _bound(self) -> ARDiffusionKVState:
        if self._ar_state is None:
            raise RuntimeError("Hy-Nomas streaming and sessions require the ARDiffusionEngine backend.")
        return self._ar_state

    def _session(self) -> HyNomasSession:
        if self._offline_session is not None:
            return self._offline_session
        return self._ar_sessions[self._bound().session_id]

    @torch.no_grad()
    def prepare_encode(self, state: StepRequestState, **kwargs: Any) -> StepRequestState:
        del kwargs
        bound = self._ar_state
        tick = ARDiffusionTickRequest.from_extra_args(state.sampling.extra_args)
        prompt = tick.prompt if tick is not None and tick.prompt is not None else state.prompt
        request = normalize_request(prompt, state.sampling)
        config = self.ar_config
        if bound is not None and (request.height > config.height or request.width > config.width):
            raise OmniClientError("Hy-Nomas request exceeds ar_height/ar_width; increase the deployment limits.")
        frame_tokens = (request.height // self.s_ratio // self.patch_size[1]) * (
            request.width // self.s_ratio // self.patch_size[2]
        )
        if frame_tokens % self.sp_size:
            raise OmniClientError("Hy-Nomas spatial tokens per frame must be divisible by ulysses_degree.")
        frames = (request.num_frames - 1) // 4 + 1
        if bound is not None and frames > config.max_latent_frames:
            raise OmniClientError("Hy-Nomas request exceeds ar_max_latent_frames.")
        # Ordinary generation preserves the reference's one denoising block per prompt segment.
        chunks = plan_chunks(request, config.chunk_latent_frames if bound is not None else frames)
        if tick is not None and len(chunks) != 1:
            raise OmniClientError("An AR-Diffusion tick must fit one Hy-Nomas chunk and contain one prompt segment.")
        extra = state.sampling.extra_args
        persistent = tick is not None or "session_id" in extra
        if bound is None and extra.keys() & {"session_id", "ar_diffusion_tick", "reset", "close_session"}:
            raise OmniClientError("Hy-Nomas session requests require the ARDiffusionEngine backend.")
        if persistent and not state.extra.get("hy_nomas_request_mode"):
            raise OmniClientError(
                "Persistent Hy-Nomas ticks require step_execution=false; live streams own one request."
            )
        finish = not persistent or (tick.close_session if tick is not None else extra.get("close_session", False))
        if bound is not None and not persistent:
            bound.reset()
            self.reset_ar_diffusion_session(bound.session_id)
        session = self._ar_sessions.get(bound.session_id) if bound is not None else None
        if session is None:
            generator = torch.Generator(device="cpu").manual_seed(request.seed)
            noise = torch.randn(
                (
                    1,
                    self.in_channels,
                    config.max_latent_frames if bound is not None else frames,
                    request.height // self.s_ratio,
                    request.width // self.s_ratio,
                ),
                generator=generator,
                dtype=self.target_dtype,
            ).to(self.device)
            session = HyNomasSession(
                request.height,
                request.width,
                request.seed,
                noise,
                generator,
                self.vae.create_streaming_decode_session(to_cpu=True, retain_history=False)
                if bound is not None
                else None,
                torch.empty((1, 0), dtype=torch.bool, device=self.device),
            )
            if bound is not None:
                self._ar_sessions[bound.session_id] = session
            else:
                capacity = frames * frame_tokens + len(request.prompts) * int(self.args.text_len)
                shape = (1, capacity, self.heads_num // self.sp_size, self.hidden_size // self.heads_num)
                session.offline_cache = [
                    {name: torch.empty(shape, dtype=self.target_dtype, device=self.device) for name in ("k", "v")}
                    for _ in range(self.num_double_blocks)
                ]
                self._offline_session = session
        if (session.height, session.width, session.seed) != (request.height, request.width, request.seed):
            raise OmniClientError("A Hy-Nomas continuation must keep its height, width and seed; reset to change them.")
        if bound is not None and session.latent_frames + frames > config.max_latent_frames:
            raise OmniClientError("Hy-Nomas session exceeds ar_max_latent_frames; close or reset the session.")
        if tick is not None and (
            bound is None or tick.session_id != bound.session_id or tick.chunk_index != session.tick_index
        ):
            raise OmniClientError("Hy-Nomas tick session/chunk_index does not match its persistent state.")
        if tick is not None:
            for control in tick.controls:
                if control.track != "camera" or control.schema != "hy_nomas.camera.v1":
                    raise OmniClientError("Hy-Nomas accepts only camera controls with schema hy_nomas.camera.v1.")
                validate_camera_payload(control.data)
                state.extra["hy_nomas_camera_override"] = control.data["camera_trajectory"]
        state.extra.update(
            hy_nomas_request=request,
            hy_nomas_chunks=chunks,
            hy_nomas_tick=tick,
            hy_nomas_finish=finish,
            hy_nomas_request_latent_start=session.latent_frames,
            hy_nomas_segment_key=None,
        )
        state.total_chunks = len(chunks)
        state.chunk_index = state.step_index = state.step_in_chunk = 0
        state.chunk_num_steps = request.steps
        state.do_true_cfg = False
        return state

    def peek_chunk_media(self, state: StepRequestState) -> ChunkMediaSpec:
        chunk = state.extra["hy_nomas_chunks"][state.chunk_index]
        return ChunkMediaSpec(num_media_frames=chunk.pixel_frames, fps=24.0, num_latent_frames=chunk.end - chunk.start)

    @torch.no_grad()
    def prepare_next_chunk(self, state: StepRequestState) -> None:
        session = self._session()
        request = state.extra["hy_nomas_request"]
        chunk = state.extra["hy_nomas_chunks"][state.chunk_index]
        text = state.extra.get("hy_nomas_prompt_override", request.prompts[chunk.segment])
        segment_key = (chunk.segment, state.extra.get("hy_nomas_prompt_version", 0))
        first = segment_key != state.extra["hy_nomas_segment_key"]
        if first:
            if self._ar_state is not None and session.segments >= self.ar_config.max_segments:
                raise OmniClientError("Hy-Nomas session exceeds ar_max_segments; close or reset the session.")
            encoded = self._encode_prompt(text)
            state.prompt_embeds = encoded["text_states"]
            # The reference owns padding and action-token coordinates. The
            # generic batcher must not trim that conditioning to mask.sum().
            state.prompt_embeds_mask = None
            state.extra["hy_nomas_text_mask"] = encoded["text_mask"]
            state.txt_seq_lens = [int(state.prompt_embeds.shape[1])]
            if state.txt_seq_lens[0] > int(self.args.text_len):
                raise OmniClientError("Hy-Nomas prompt exceeds the deployed text token limit.")
            prompt_session = state.interaction_sessions.setdefault("prompt", HyNomasInteractionSession())
            assert isinstance(prompt_session, HyNomasInteractionSession)
            with prompt_session.lock:
                prompt_session.version += 1
            session.segments += 1
            state.extra["hy_nomas_segment_key"] = segment_key
            state.extra["hy_nomas_segment_id"] = session.segments
        offsets = windows = None
        if first:
            segment_start = chunk.start
            segment_end = (request.num_frames - 1) // 4 + 1
            if chunk.segment + 1 < len(request.prompts):
                segment_end = request.boundaries[chunk.segment + 1] // 4
            action_start, offsets, windows = action_conditioning(
                text, self.text_encoder, int(self.args.text_len), segment_end - segment_start
            )
            state.extra["hy_nomas_action_start"] = action_start
        state.extra["hy_nomas_intra_offsets"] = [offsets] if offsets is not None and windows is not None else None
        state.extra["hy_nomas_intra_windows"] = windows
        state.extra["hy_nomas_cache_text"] = first
        start = state.extra["hy_nomas_request_latent_start"] + chunk.start
        end = start + chunk.end - chunk.start
        state.latents = session.noise[:, : self.transformer.config.out_channels, start:end].clone()
        state.scheduler = self._new_scheduler(request.steps)
        state.timesteps = state.scheduler.timesteps
        state.step_index = state.step_in_chunk = 0
        camera = state.extra.get("hy_nomas_camera_override", request.camera_trajectory)
        overridden = "hy_nomas_camera_override" in state.extra
        camera_key = (camera, overridden)
        if state.extra.get("hy_nomas_camera_key") != camera_key:
            origin = chunk.start if overridden else 0
            with self._component_context(self.transformer):
                state.extra["hy_nomas_camera_full"] = prepare_camera(
                    self.transformer,
                    self.args,
                    camera,
                    num_frames=request.num_frames - 4 * origin,
                    latent_h=request.height // self.s_ratio,
                    latent_w=request.width // self.s_ratio,
                    boundaries=None if overridden else request.boundaries,
                    device=self.device,
                )
            state.extra["hy_nomas_camera_key"] = camera_key
            state.extra["hy_nomas_camera_origin"] = origin
        origin = state.extra["hy_nomas_camera_origin"]
        full_camera = state.extra["hy_nomas_camera_full"]
        state.extra["hy_nomas_camera"] = (
            None if full_camera is None else full_camera[:, chunk.start - origin : chunk.end - origin]
        )
        camera_off = (
            full_camera is not None
            and not overridden
            and request.segment_has_camera is not None
            and not request.segment_has_camera[chunk.segment]
        )
        state.extra["hy_nomas_camera_mask"] = (
            torch.zeros(1, dtype=torch.bool, device=self.device) if camera_off else None
        )

    def _forward_chunk(
        self, state: StepRequestState, latents: torch.Tensor, timestep: torch.Tensor, *, commit: bool
    ) -> torch.Tensor:
        session = self._session()
        cache = self._ar_state.segmented("cond") if self._ar_state is not None else None
        if state.prompt_embeds is None:
            raise RuntimeError("Hy-Nomas requires encoded text before a DiT forward.")
        first = state.extra["hy_nomas_cache_text"]
        image_tokens = (
            (latents.shape[2] // self.patch_size[0])
            * (latents.shape[3] // self.patch_size[1])
            * (latents.shape[4] // self.patch_size[2])
        )
        current_tokens = image_tokens + (state.prompt_embeds.shape[1] if first else 0)
        history = session.history_mask.shape[1]
        if cache is not None:
            if cache.history_tokens != history:
                raise RuntimeError("Hy-Nomas KV history and text visibility mask disagree.")
            views = reference_cache_views(cache, current_tokens)
        else:
            if session.offline_cache is None:
                raise RuntimeError("Hy-Nomas offline KV cache is not initialized.")
            views = [
                {name: tensor[:, : history + current_tokens] for name, tensor in layer.items()}
                for layer in session.offline_cache
            ]
        mask = torch.cat(
            (session.history_mask, torch.zeros((1, current_tokens), device=self.device, dtype=torch.bool)), 1
        )
        for segment, start, end in session.action_tails:
            if segment != state.extra["hy_nomas_segment_id"]:
                mask[:, start:end] = False
        if latents.shape[1] < self.in_channels:
            latents = torch.cat(
                (
                    latents,
                    latents.new_zeros((latents.shape[0], self.in_channels - latents.shape[1], *latents.shape[2:])),
                ),
                1,
            )
        pending = cache.begin_commit((current_tokens,)) if commit and cache is not None else ()
        try:
            with (
                self._component_context(self.transformer),
                torch.autocast(
                    device_type=self.device.type, dtype=self.target_dtype, enabled=self.device.type == "cuda"
                ),
            ):
                prediction, written = self.transformer.forward_with_cache(
                    hidden_states=latents,
                    timestep=timestep,
                    text_states=state.prompt_embeds,
                    text_mask=state.extra["hy_nomas_text_mask"],
                    kv_cache_list=views,
                    cache_valid_len=history,
                    update_cache=commit,
                    global_frame_counter=session.latent_frames,
                    guidance=None,
                    cache_txt=first,
                    cache_mask_list=[mask] * len(views),
                    camera_embedding=state.extra["hy_nomas_camera"],
                    camera_has_mask=state.extra["hy_nomas_camera_mask"],
                    intra_seg_txt_offsets=state.extra["hy_nomas_intra_offsets"],
                    intra_seg_vid_windows=state.extra["hy_nomas_intra_windows"],
                    camera_rope_time_offset=0,
                )
            if commit:
                if written != current_tokens:
                    raise RuntimeError("Hy-Nomas clean forward wrote an unexpected KV token count.")
                if cache is not None:
                    commit_cache(cache, views)
                    cache.commit(retain=(*cache.handles, *pending))
                session.history_mask = torch.cat((session.history_mask, mask[:, history:].clone()), 1)
                action_start = state.extra["hy_nomas_action_start"]
                if first and action_start is not None and 0 <= action_start < state.prompt_embeds.shape[1]:
                    session.action_tails.append(
                        (
                            state.extra["hy_nomas_segment_id"],
                            history + image_tokens + action_start,
                            history + current_tokens,
                        )
                    )
            return prediction
        except BaseException:
            if commit and cache is not None:
                cache.rollback()
            raise

    @torch.no_grad()
    def denoise_step(
        self, input_batch: Any, *, states: Sequence[StepRequestState] | None = None, **kwargs: Any
    ) -> torch.Tensor:
        del input_batch, kwargs
        if states is None or len(states) != 1:
            raise ValueError("Hy-Nomas step execution supports one request at a time.")
        state = states[0]
        if state.latents is None or state.scheduler is None:
            raise RuntimeError("Hy-Nomas prepare_next_chunk must precede denoise_step.")
        self.transformer._sparse_step_index = state.step_in_chunk
        return self._forward_chunk(
            state, state.latents, state.scheduler.timesteps[state.step_in_chunk].expand(1), commit=False
        )

    def step_scheduler(self, state: StepRequestState, noise_pred: torch.Tensor, **kwargs: Any) -> None:
        del kwargs
        if state.scheduler is None or state.latents is None:
            raise RuntimeError("Hy-Nomas prepare_next_chunk must precede step_scheduler.")
        state.latents = state.scheduler.step(
            model_output=noise_pred,
            timestep=state.scheduler.timesteps[state.step_in_chunk],
            sample=state.latents,
            return_dict=False,
        )[0].to(self.target_dtype)
        state.step_in_chunk += 1
        state.step_index += 1

    def _decode_chunk(
        self, session: HyNomasSession, latents: torch.Tensor, *, pixel_frames: int, finish: bool
    ) -> np.ndarray:
        error = None
        video = None
        try:
            with self._component_context(self.vae):
                session.decoder.final_segment = finish
                pixels = session.decoder.push(self._prepare_vae_decode_latents(latents), pixel_frames=pixel_frames)
                if finish:
                    tail = session.decoder.finish()
                    if tail.numel():
                        pixels = torch.cat((pixels, tail), 2) if pixels.numel() else tail
            if pixels.numel():
                pixels = (pixels * 0.5 + 0.5).clamp(0, 1)
                video = pixels[0].permute(1, 2, 3, 0).float().mul(255).round().to(torch.uint8).cpu().numpy()
            else:
                video = np.empty((0, session.height, session.width, 3), dtype=np.uint8)
        except Exception as exc:
            error = exc
        # PixelUMM runs independently on each SP rank. All peers must reject
        # a failed decode before any of them enters the next DiT collective.
        if self.sp_size > 1:
            failed = torch.tensor([error is not None], dtype=torch.int32, device=self.device)
            if get_sp_group().all_reduce(failed).item() and error is None:
                error = RuntimeError("Hy-Nomas PixelUMM decoding failed on another SP rank.")
        if error is not None:
            raise error
        assert video is not None
        return video

    @torch.no_grad()
    def post_decode(self, state: StepRequestState, **kwargs: Any) -> DiffusionOutput:
        del kwargs
        if state.latents is None or not state.chunk_denoise_completed:
            raise RuntimeError("Hy-Nomas must complete every denoising step before committing a chunk.")
        session = self._session()
        chunk = state.extra["hy_nomas_chunks"][state.chunk_index]
        cache_latents = state.latents
        sigma = self.cache_noise_sigma
        if sigma:
            epsilon = torch.randn(cache_latents.shape, generator=session.generator, dtype=cache_latents.dtype).to(
                self.device
            )
            cache_latents = ((1 - sigma) * cache_latents.float() + sigma * epsilon.float()).to(cache_latents.dtype)
        self._forward_chunk(
            state,
            cache_latents,
            torch.full((1,), sigma * 1000, dtype=torch.float32, device=self.device),
            commit=True,
        )
        session.latent_frames += chunk.end - chunk.start
        frame_start = session.emitted_frames
        completed = state.chunk_index
        state.chunk_index += 1
        if self._ar_state is None:
            return DiffusionOutput(output=state.latents, finished=state.request_denoise_completed)
        video = self._decode_chunk(
            session,
            state.latents,
            pixel_frames=chunk.pixel_frames,
            finish=state.request_denoise_completed and state.extra["hy_nomas_finish"],
        )
        session.emitted_frames += len(video)
        tick = state.extra["hy_nomas_tick"]
        identity = (
            ARDiffusionChunkMetadata.from_tick(tick)
            if tick is not None
            else ARDiffusionChunkMetadata(
                session_id=self._bound().session_id,
                request_id=state.request_id,
                chunk_index=completed,
            )
        )
        if state.request_denoise_completed:
            session.tick_index += 1
        return DiffusionOutput(
            output={
                "payload": {"video": video},
                "metadata": {
                    "video": {"fps": 24.0},
                    "ar_diffusion": identity.to_dict(),
                    "hy_nomas": {
                        "frame_start": frame_start,
                        "num_frames": len(video),
                        "generated_frames": session.decoder.available_pixel_frames,
                        "withheld_frames": session.decoder.available_pixel_frames - session.emitted_frames,
                    },
                },
            },
            chunk_index=completed,
            total_chunks=state.total_chunks,
            finished=state.request_denoise_completed,
        )

    @torch.no_grad()
    def forward(self, request: DiffusionRequestBatch) -> DiffusionOutput:
        if request.num_reqs != 1:
            raise ValueError("Hy-Nomas processes one request at a time.")
        req = request.requests[0]
        state = StepRequestState(request_id=req.request_id, sampling=req.sampling_params, prompt=req.prompt)
        state.extra["hy_nomas_request_mode"] = True
        try:
            self.prepare_encode(state)
            chunks = []
            frame_start = self._session().emitted_frames
            while not state.request_denoise_completed:
                self.prepare_next_chunk(state)
                assert state.chunk_num_steps is not None
                with self._component_context(self.transformer):
                    for _ in range(state.chunk_num_steps):
                        self.step_scheduler(state, self.denoise_step(None, states=[state]))
                    output = self.post_decode(state)
                chunks.append(output.output["payload"]["video"] if self._ar_state is not None else output.output)
            if self._ar_state is not None:
                output.output["payload"]["video"] = np.concatenate(chunks, axis=0)
                output.output["metadata"]["hy_nomas"].update(
                    frame_start=frame_start, num_frames=len(output.output["payload"]["video"])
                )
                return output
            inputs = state.extra["hy_nomas_request"]
            with self._component_context(self.vae):
                video = self.vae.decode(self._prepare_vae_decode_latents(torch.cat(chunks, dim=2))).sample
            video = (video[:, :, : inputs.num_frames] * 0.5 + 0.5).clamp(0, 1)
            expected = (1, 3, inputs.num_frames, inputs.height, inputs.width)
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
        finally:
            if self._ar_state is None:
                self._offline_session = None
                from hymm.models.dit.modules.attention.range_attention import clear_range_cache

                clear_range_cache()
