# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import ClassVar

import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper

from vllm_omni.diffusion.cache.cachedit import (
    CacheDiTBackend,
    CacheDiTRequestSpec,
    RequestScopedCacheDiTRuntime,
)
from vllm_omni.diffusion.data import DiffusionCacheConfig, DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.offloader.config import OffloadStrategy, resolve_offload_strategy
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs.output_metadata import DiffusionPostprocessRawOutput
from vllm_omni.transformers_utils.configs.mammoth_moda2 import Mammothmoda2Config

from .mammothmoda2_dit_model import SimpleQFormerImageRefiner, Transformer2DModel
from .rope_real import RotaryPosEmbedReal
from .schedulers import FlowMatchEulerDiscreteScheduler

logger = init_logger(__name__)

# Identifies the pipeline-owned Cache-DiT installation across requests.
_MAMMOTHMODA2_CACHE_DIT_KEY = "mammothmoda2:cache_dit"


def _first_request_value(value: object) -> object:
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def get_mammoth_moda2_post_process_func(
    od_config: OmniDiffusionConfig,
) -> Callable[[torch.Tensor], DiffusionPostprocessRawOutput]:
    if od_config.output_type not in ("pil", "np", "pt"):
        raise ValueError("MammothModa2 returns decoded images; output_type must be pil, np or pt.")
    image_processor = VaeImageProcessor()

    def post_process_func(images: torch.Tensor) -> DiffusionPostprocessRawOutput:
        # The VAE returns BCHW in [-1, 1]. Convert once at the shared runtime's
        # postprocess boundary, not in each distributed worker or the example.
        return image_processor.postprocess(images, output_type=od_config.output_type)

    return post_process_func


def _build_mammoth_config(od_config: OmniDiffusionConfig) -> Mammothmoda2Config:
    raw_config = od_config.tf_model_config.to_dict()
    if not raw_config:
        raise ValueError("MammothModa2 diffusion stage requires the root checkpoint config")
    return Mammothmoda2Config(**raw_config)


def _root_weight_source(
    od_config: OmniDiffusionConfig,
) -> DiffusersPipelineLoader.ComponentSource:
    if not od_config.model:
        raise ValueError("MammothModa2 diffusion stage requires a model path")
    return DiffusersPipelineLoader.ComponentSource(
        model_or_path=od_config.model,
        subfolder=None,
        revision=od_config.revision,
        prefix="",
        fall_back_to_pt=True,
    )


def _pad_cond_sequence(
    embeds: list[torch.Tensor],
    masks: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad per-request ``[1, T_i, H]`` embeds into ``[B, T_max, H]``.

    The transformer derives per-row ``encoder_seq_lengths`` from the boolean
    mask, so padding must sit on the right and masked positions are excluded
    from the packed joint sequence.
    """
    if not embeds:
        raise ValueError("Cannot pad an empty conditioning sequence list")
    if len(embeds) != len(masks):
        raise ValueError(f"Conditioning embeds/mask count mismatch: {len(embeds)} vs {len(masks)}")

    # Fast-path for single-item batch: bypass padding tensor allocation
    if len(embeds) == 1:
        emb0, m0 = embeds[0], masks[0]
        if emb0.ndim != 3 or emb0.shape[0] != 1:
            raise ValueError(f"Conditioning embeds[0] must be [1, T, H], got {tuple(emb0.shape)}")
        if m0.ndim != 2 or m0.shape != emb0.shape[:2]:
            raise ValueError(f"Conditioning embeds/mask length mismatch: {tuple(emb0.shape)} vs {tuple(m0.shape)}")
        return emb0, m0

    hidden = embeds[0].shape[-1]
    lengths = [int(e.shape[1]) for e in embeds]
    max_len = max(lengths)

    # Fast-path for homogeneous lengths: concatenate directly in C++
    if all(length == max_len for length in lengths):
        for i, (emb, m) in enumerate(zip(embeds, masks)):
            if emb.ndim != 3 or emb.shape[0] != 1 or emb.shape[-1] != hidden:
                raise ValueError(f"Conditioning embeds[{i}] must be [1, T, {hidden}], got {tuple(emb.shape)}")
            if m.ndim != 2 or m.shape[0] != 1 or m.shape[1] != max_len:
                raise ValueError(f"Conditioning embeds/mask length mismatch: {tuple(emb.shape)} vs {tuple(m.shape)}")
        return torch.cat(embeds, dim=0), torch.cat(masks, dim=0)

    out = embeds[0].new_zeros((len(embeds), max_len, hidden))
    mask = torch.zeros((len(embeds), max_len), dtype=torch.bool, device=out.device)
    for i, (emb, m) in enumerate(zip(embeds, masks)):
        if emb.ndim != 3 or emb.shape[0] != 1 or emb.shape[-1] != hidden:
            raise ValueError(f"Conditioning embeds[{i}] must be [1, T, {hidden}], got {tuple(emb.shape)}")
        if m.ndim != 2 or m.shape[0] != 1 or m.shape[1] != emb.shape[1]:
            raise ValueError(f"Conditioning embeds/mask length mismatch: {tuple(emb.shape)} vs {tuple(m.shape)}")
        out[i, : emb.shape[1]] = emb[0]
        mask[i, : m.shape[1]] = m[0].bool()
    return out, mask


def _resolve_request_sampling_and_dims(
    prompt: dict,
    sampling: OmniDiffusionSamplingParams | None,
    request_id: str,
) -> tuple[int, int, float, tuple[float, float], int]:
    """Shared extraction and validation of geometry and sampling knobs."""
    dimensions = []
    for name in ("height", "width"):
        value = DiffusionRequestBatch.get_prompt_field(prompt, name)
        if value is None and sampling is not None:
            value = getattr(sampling, name, None)
        if value is None:
            value = 1024
        try:
            dimensions.append(int(value))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid image size {name}={value!r} for request {request_id}") from exc
    height, width = dimensions
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image size: {height}x{width} for request {request_id}")
    if height % 16 != 0 or width % 16 != 0:
        raise ValueError(f"Image size must be multiples of 16, got {height}x{width} for request {request_id}")

    info = prompt.get("additional_information")
    request_info = info if isinstance(info, dict) else {}
    extra_args = sampling.extra_args or {} if sampling else {}

    guidance = extra_args.get("text_guidance_scale")
    if guidance is None and sampling is not None:
        guidance = sampling.guidance_scale if sampling.guidance_scale_provided else None
    if guidance is None:
        guidance = _first_request_value(request_info.get("text_guidance_scale"))
    if guidance is None:
        guidance = 9.0
    try:
        text_guidance_scale = float(guidance)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid text_guidance_scale for request {request_id}") from exc

    raw_num_inference_steps = extra_args.get("num_inference_steps")
    if raw_num_inference_steps is None and sampling is not None:
        raw_num_inference_steps = sampling.num_inference_steps
    if raw_num_inference_steps is None:
        raw_num_inference_steps = _first_request_value(request_info.get("num_inference_steps"))
    if raw_num_inference_steps is None:
        raw_num_inference_steps = 50
    try:
        num_inference_steps = int(raw_num_inference_steps)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid num_inference_steps for request {request_id}") from exc
    if num_inference_steps <= 0:
        raise ValueError(f"num_inference_steps must be positive for request {request_id}")

    cfg_range = extra_args.get("cfg_range")
    if cfg_range is None:
        cfg_range = request_info.get("cfg_range")
    if cfg_range is None:
        cfg_range = [0.0, 1.0]
    if not isinstance(cfg_range, (list, tuple)) or len(cfg_range) != 2:
        raise ValueError(f"cfg_range requires two values for request {request_id}")
    try:
        cfg_start, cfg_end = float(cfg_range[0]), float(cfg_range[1])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"cfg_range requires two values convertible to floats for request {request_id}") from exc
    if not 0 <= cfg_start <= cfg_end <= 1:
        raise ValueError(f"cfg_range must satisfy 0 <= start <= end <= 1 for request {request_id}")

    return height, width, text_guidance_scale, (cfg_start, cfg_end), num_inference_steps


def _resolve_gen_vocab_start_index(od_config: OmniDiffusionConfig | None) -> int:
    """Resolve the visual token threshold through the normalized model configuration."""
    if od_config is not None and getattr(od_config, "tf_model_config", None) is not None:
        try:
            cfg = _build_mammoth_config(od_config)
            llm_cfg = getattr(cfg, "llm_config", None)
            if llm_cfg is not None:
                val = getattr(llm_cfg, "gen_vocab_start_index", None)
                if val is not None:
                    return int(val)
                text_cfg = getattr(llm_cfg, "text_config", None)
                if text_cfg is not None and getattr(text_cfg, "gen_vocab_start_index", None) is not None:
                    return int(text_cfg.gen_vocab_start_index)
        except Exception:
            pass

        # Fallback inspection of raw dictionary if _build_mammoth_config fails (e.g. mock test objects)
        raw_cfg = od_config.tf_model_config.to_dict()
        if isinstance(raw_cfg, dict):
            llm_cfg = raw_cfg.get("llm_config")
            if isinstance(llm_cfg, dict):
                text_cfg = llm_cfg.get("text_config")
                if isinstance(text_cfg, dict) and text_cfg.get("gen_vocab_start_index") is not None:
                    return int(text_cfg["gen_vocab_start_index"])
                if llm_cfg.get("gen_vocab_start_index") is not None:
                    return int(llm_cfg["gen_vocab_start_index"])
    return 152064


def _validate_request_for_admission(
    request: OmniDiffusionRequest,
    od_config: OmniDiffusionConfig | None = None,
    gen_vocab_start_index: int | None = None,
) -> tuple[int, int, int]:
    """Validate request at admission time to fail-fast before entering scheduler queue.

    Returns (height, width, num_inference_steps).
    """
    request_id = request.request_id
    prompt = request.prompt if isinstance(request.prompt, dict) else {}
    sampling = request.sampling_params
    if sampling is not None and getattr(sampling, "num_outputs_per_prompt", 1) != 1:
        raise ValueError(
            f"MammothModa2 requires num_outputs_per_prompt == 1, got {sampling.num_outputs_per_prompt} "
            f"for request {request_id}"
        )
    if not request.is_dummy_run():
        info = prompt.get("additional_information")
        if not isinstance(info, dict):
            raise ValueError(f"Missing additional_information AR conditions for request {request_id}")
        full_hidden_states = info.get("full_hidden_states")
        full_token_ids = info.get("full_token_ids")
        if not isinstance(full_hidden_states, torch.Tensor) or not isinstance(full_token_ids, list):
            raise ValueError(f"Expected full_hidden_states tensor and full_token_ids list for request {request_id}")
        try:
            answer_start_index = int(info.get("answer_start_index"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid answer_start_index for request {request_id}") from exc
        if full_hidden_states.ndim != 2:
            raise ValueError(f"Expected 2D full_hidden_states for request {request_id}")
        if full_hidden_states.shape[0] != len(full_token_ids):
            raise ValueError(f"AR hidden-state/token-count mismatch for request {request_id}")
        if not 0 <= answer_start_index <= len(full_token_ids):
            raise ValueError(f"answer_start_index outside token range for request {request_id}")
        try:
            int_tokens = [int(token_id) for token_id in full_token_ids]
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid full_token_ids for request {request_id}") from exc

        if gen_vocab_start_index is None:
            gen_vocab_start_index = _resolve_gen_vocab_start_index(od_config)
        answer_tokens = int_tokens[answer_start_index:]
        if not any(token_id >= gen_vocab_start_index for token_id in answer_tokens):
            raise ValueError(
                f"MammothModa2 AR stage produced no visual-token hidden states for request {request_id}; "
                f"the DiT stage requires at least one generated visual token. Generated token ids: {answer_tokens[:32]}"
            )

    height, width, _, _, num_inference_steps = _resolve_request_sampling_and_dims(prompt, sampling, request_id)
    return height, width, num_inference_steps


def get_mammoth_moda2_pre_process_func(od_config: OmniDiffusionConfig | None = None):
    """Admission preprocessor: fail-fast per-request validation and grouping.

    Validates AR conditions, dimensions, and sampling knobs at admission so
    malformed requests are rejected individually with their request_id before
    entering the scheduler queue. Admitted requests are assigned a
    batch_compatibility_key based on output geometry and inference steps.
    """
    gen_vocab_start_index = _resolve_gen_vocab_start_index(od_config)

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        height, width, num_inference_steps = _validate_request_for_admission(
            request, od_config=od_config, gen_vocab_start_index=gen_vocab_start_index
        )
        request.batch_compatibility_key = (
            "mammoth_moda2_dit",
            height,
            width,
            num_inference_steps,
        )
        return request

    return pre_process_func


def _validate_sequence_parallel_runtime(od_config: OmniDiffusionConfig, config: Mammothmoda2Config) -> None:
    if od_config.parallel_config.sequence_parallel_size == 1:
        return
    if getattr(config.llm_config, "model_type", "") != "mammothmoda2_qwen2_5_vl":
        raise ValueError("MammothModa2 sequence parallelism is limited to Preview text-to-image, not Dev.")
    if od_config.step_execution or od_config.max_num_seqs != 1:
        raise ValueError("MammothModa2 SP requires request mode with max_num_seqs=1.")
    if (
        not od_config.enforce_eager
        or od_config.cache_backend != "none"
        or od_config.quantization_config is not None
        or resolve_offload_strategy(od_config) is not OffloadStrategy.NONE
    ):
        raise ValueError(
            "MammothModa2 SP requires eager execution without cache acceleration, quantization or offload."
        )


@dataclass(frozen=True)
class _MammothRequest:
    index: int
    request_id: str
    full_hidden_states: torch.Tensor
    full_token_ids: list[int]
    answer_start_index: int
    height: int
    width: int
    text_guidance_scale: float
    cfg_range: tuple[float, float]
    num_inference_steps: int
    seed: int | None
    generator: torch.Generator | list[torch.Generator] | None
    generator_device: torch.device | str | None = None


@dataclass(frozen=True)
class _RequestConditioning:
    text_embeds: torch.Tensor
    text_mask: torch.Tensor
    image_embeds: torch.Tensor
    image_mask: torch.Tensor


class MammothModa2DiTPipeline(nn.Module, SupportsComponentDiscovery):
    """
    MammothModa2 DiT + VAE generation stage (non-autoregressive).

    This stage expects "image condition token hidden states" from the upstream AR stage,
    and outputs image tensors via diffusion transformer + VAE decode.

    """

    _dit_modules: ClassVar[list[str]] = ["gen_transformer"]
    _encoder_modules: ClassVar[list[str]] = ["gen_image_condition_refiner"]
    _vae_modules: ClassVar[list[str]] = ["gen_vae"]

    supports_request_batch = True
    supports_step_execution = False

    # Load only gen_* weights; ignore llm_model.* to prevent loading the entire LLM backbone in the DiT stage.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "llm_model.": None,
            "gen_tokenizer.": None,
        }
    )

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        del prefix
        self.od_config = od_config
        self.device = get_local_device()
        self.config = _build_mammoth_config(od_config)
        _validate_sequence_parallel_runtime(od_config, self.config)
        self.weights_sources = [_root_weight_source(od_config)]

        # --- Build DiT / VAE modules (names must match checkpoint keys) ---
        if self.config.gen_vae_config is None or self.config.gen_dit_config is None:
            raise ValueError("Mammothmoda2Config.gen_vae_config / gen_dit_config must not be None")

        self.gen_vae = AutoencoderKL.from_config(self.config.gen_vae_config)
        self.gen_transformer = Transformer2DModel.from_config(self.config.gen_dit_config)

        # llm_config is a Mammothmoda2Qwen2_5_VLConfig which has nested text_config
        llm_hidden_size = 0
        text_config = self.config.get_text_config()
        if text_config is None:
            logger.warning("No text config; failed to infer llm_hidden_size.")
        elif not hasattr(text_config, "hidden_size"):
            logger.warning("Text config exists, but has no hidden_size attribute; failed to infer llm_hidden_size.")
        else:
            llm_hidden_size = int(text_config.hidden_size or 0)
        if llm_hidden_size <= 0:
            raise ValueError(
                "Failed to infer llm hidden_size from Mammothmoda2Config.llm_config.text_config.hidden_size"
            )
        self._reinit_caption_embedder(llm_hidden_size)

        # Optional image condition Q-Former. Preview stores it as a standalone
        # module; Dev stores it under the DiT timestep/caption embedder.
        llm_model_type = getattr(self.config.llm_config, "model_type", "")
        refiner_config = self.config.gen_image_condition_refiner_config
        if refiner_config is not None and llm_model_type == "mammothmoda2_qwen3_vl":
            dit_hidden_size = int(self.gen_transformer.hidden_size)
            self.gen_transformer.time_caption_embed.image_embedder = SimpleQFormerImageRefiner(
                hidden_size=llm_hidden_size,
                output_hidden_size=dit_hidden_size,
                num_heads=max(1, dit_hidden_size // 128),
                **refiner_config,
            )
            self.gen_image_condition_refiner = None
        elif refiner_config is not None:
            self.gen_image_condition_refiner = SimpleQFormerImageRefiner(
                hidden_size=llm_hidden_size,
                **refiner_config,
            )
        else:
            self.gen_image_condition_refiner = None

        # Precompute rotary freqs for diffusion transformer
        # IMPORTANT: follow upstream mammothmoda: use top-level `config.gen_axes_*`
        # (the checkpoint's `gen_dit_config.axes_lens` can be as small as 1024,
        # which is insufficient for vLLM dummy-run/cudagraph warmup).
        self.gen_freqs_cis = RotaryPosEmbedReal.get_freqs_real(
            tuple(self.config.gen_axes_dim_rope),
            tuple(self.config.gen_axes_lens),
            theta=10000,
        )

        self._llm_hidden_size = llm_hidden_size

        # Cache-DiT lifecycle: the diffusion runner enables the configured
        # backend (``cache_backend`` on the deploy YAML stage entry) at startup
        # and transfers ownership here via the request-scoped protocol;
        # forward() then reconciles per-request state (step count, CFG parity).
        # The runner also emits the cache summary when
        # ``enable_cache_dit_summary`` is set.
        self._cache_dit_runtime = RequestScopedCacheDiTRuntime(self)
        self._cache_dit_config: DiffusionCacheConfig | None = None
        if str(getattr(od_config, "cache_backend", "") or "").lower() == "cache_dit":
            cache_config = od_config.cache_config
            self._cache_dit_config = (
                cache_config
                if isinstance(cache_config, DiffusionCacheConfig)
                else DiffusionCacheConfig.from_dict(cache_config or {})
            )

    def adopt_cache_dit_backend(self, backend: CacheDiTBackend) -> None:
        """Adopt a runner-installed Cache-DiT backend (request-scoped protocol)."""

        self._cache_dit_runtime.adopt(backend, installation_key=_MAMMOTHMODA2_CACHE_DIT_KEY)

    def is_cache_dit_enabled(self) -> bool:
        """Return the request-scoped Cache-DiT installation state."""

        return self._cache_dit_runtime.is_enabled

    def _prepare_cache_dit_for_request(self, *, num_inference_steps: int, cfg_active: bool) -> None:
        """Reconcile Cache-DiT hooks with this request before the denoise loop.

        cache-dit's separate-CFG accounting assumes exactly two transformer
        forwards per denoise step (conditional then unconditional), so CFG
        requests refresh the context (per-request step counts vary) while
        no-CFG requests run with hooks disabled: their single forward per
        step cannot be represented by the parity accounting.
        """
        if self._cache_dit_config is None:
            return
        if not cfg_active:
            self._cache_dit_runtime.prepare(None)
            return
        self._cache_dit_runtime.prepare(
            CacheDiTRequestSpec(
                installation_key=_MAMMOTHMODA2_CACHE_DIT_KEY,
                cache_config=self._cache_dit_config,
                num_inference_steps=num_inference_steps,
            )
        )

    def _reinit_caption_embedder(self, in_features: int) -> None:
        # Align with upstream Mammothmoda2Model's `reinit_caption_embedder`:
        # Use Qwen2RMSNorm(in_features) + Linear(in_features -> out_features).
        out_features = int(getattr(self.gen_transformer, "hidden_size", 0) or self.gen_transformer.config.hidden_size)
        self.gen_transformer.time_caption_embed.caption_embedder = nn.Sequential(
            Qwen2RMSNorm(in_features, eps=1e-5),
            nn.Linear(in_features, out_features, bias=True),
        )

    @staticmethod
    def _group_requests(specs: list[_MammothRequest]) -> list[list[int]]:
        """Group co-scheduled requests by (height, width, num_inference_steps).

        Preserves arrival order: the first-seen request of each bucket
        determines the group's execution order.
        """
        buckets: dict[tuple[int, int, int], list[int]] = {}
        for idx, spec in enumerate(specs):
            key = (spec.height, spec.width, spec.num_inference_steps)
            buckets.setdefault(key, []).append(idx)
        return list(buckets.values())

    def _parse_request(self, req: DiffusionRequestBatch, index: int = 0) -> _MammothRequest:
        request = req.requests[index]
        request_id = request.request_id
        prompt = request.prompt if isinstance(request.prompt, dict) else {}
        sampling = request.sampling_params
        if sampling is not None and getattr(sampling, "num_outputs_per_prompt", 1) != 1:
            raise ValueError(
                f"MammothModa2 requires num_outputs_per_prompt == 1, got {sampling.num_outputs_per_prompt} "
                f"for request {request_id}"
            )
        info = prompt.get("additional_information")
        if request.is_dummy_run():
            gen_start = 152064
            if hasattr(self, "config") and hasattr(self.config, "llm_config") and self.config.llm_config is not None:
                val = getattr(self.config.llm_config, "gen_vocab_start_index", None)
                if val is not None:
                    gen_start = val
                else:
                    text_cfg = getattr(self.config.llm_config, "text_config", None)
                    if text_cfg is not None and getattr(text_cfg, "gen_vocab_start_index", None) is not None:
                        gen_start = text_cfg.gen_vocab_start_index
            full_hidden_states = torch.zeros((2, self._llm_hidden_size), dtype=torch.float32)
            full_token_ids = [0, int(gen_start)]
            answer_start_index = 1
        else:
            if not isinstance(info, dict):
                raise ValueError(f"Missing additional_information AR conditions for request {request_id}")
            full_hidden_states = info.get("full_hidden_states")
            full_token_ids = info.get("full_token_ids")
            if not isinstance(full_hidden_states, torch.Tensor) or not isinstance(full_token_ids, list):
                raise ValueError(f"Expected full_hidden_states tensor and full_token_ids list for request {request_id}")
            try:
                answer_start_index = int(info.get("answer_start_index"))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"Invalid answer_start_index for request {request_id}") from exc
            if full_hidden_states.ndim != 2:
                raise ValueError(f"Expected 2D full_hidden_states for request {request_id}")
            if full_hidden_states.shape[0] != len(full_token_ids):
                raise ValueError(f"AR hidden-state/token-count mismatch for request {request_id}")
            if not 0 <= answer_start_index <= len(full_token_ids):
                raise ValueError(f"answer_start_index outside token range for request {request_id}")
            try:
                full_token_ids = [int(token_id) for token_id in full_token_ids]
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"Invalid full_token_ids for request {request_id}") from exc

        height, width, text_guidance_scale, cfg_range, num_inference_steps = _resolve_request_sampling_and_dims(
            prompt, sampling, request_id
        )

        generator = sampling.generator if sampling else None
        if isinstance(generator, list) and len(generator) != 1:
            raise ValueError(
                f"MammothModa2 single-output request mode requires exactly one generator for request {request_id}"
            )

        return _MammothRequest(
            index=index,
            request_id=request_id,
            full_hidden_states=full_hidden_states,
            full_token_ids=full_token_ids,
            answer_start_index=answer_start_index,
            height=height,
            width=width,
            text_guidance_scale=text_guidance_scale,
            cfg_range=cfg_range,
            num_inference_steps=num_inference_steps,
            seed=sampling.seed if sampling else None,
            generator=generator,
            generator_device=getattr(sampling, "generator_device", None) if sampling else None,
        )

    def _split_ar_conditions(
        self,
        *,
        full_hidden_states: torch.Tensor,
        full_token_ids: list[int],
        answer_start_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split AR-stage hidden states into text / image condition embeds.

        The token ids that distinguish question (text) tokens, generated visual
        tokens, and multi-modal placeholder tokens are read from the model config
        (``gen_vocab_start_index`` and the vision placeholder token ids), so the
        caller no longer needs to pass them. Mirrors the masking the bespoke
        MammothModa2 example performed via ar2dit.
        """
        gen_vocab_start_index = getattr(self.config.llm_config, "gen_vocab_start_index", None)
        if gen_vocab_start_index is None:
            text_cfg = getattr(self.config.llm_config, "text_config", None)
            gen_vocab_start_index = getattr(text_cfg, "gen_vocab_start_index", 152064)
        gen_vocab_start_index = int(gen_vocab_start_index)
        cached_visual_ids = getattr(self, "_cached_visual_ids_tensor", None)
        if cached_visual_ids is None:
            cached_visual_ids = torch.tensor(
                [
                    int(self.config.image_token_id),
                    int(self.config.video_token_id),
                    int(self.config.vision_start_token_id),
                    int(self.config.vision_end_token_id),
                ],
                dtype=torch.long,
            )
            self._cached_visual_ids_tensor = cached_visual_ids

        device = full_hidden_states.device
        token_ids = torch.tensor(full_token_ids, dtype=torch.long, device=device)
        positions = torch.arange(token_ids.shape[0], device=device)
        questions_mask = positions < answer_start_index
        answers_mask = ~questions_mask
        gen_token_mask = token_ids >= gen_vocab_start_index
        visual_ids_device = cached_visual_ids.to(device=device, non_blocking=True)
        visual_token_mask = torch.isin(token_ids, visual_ids_device)
        text_mask = questions_mask & ~(visual_token_mask | gen_token_mask)
        image_mask = answers_mask & gen_token_mask

        # Keep the transferred representation compact until the final device/dtype
        # conversion in ``forward``.  The masks are dtype-independent, and an
        # unconditional host-side float32 expansion here would otherwise double both
        # the selected-condition staging footprint and BF16 H2D traffic.
        text_cond = full_hidden_states[text_mask].contiguous()
        image_cond = full_hidden_states[image_mask].contiguous()
        return text_cond, image_cond

    def _split_request_conditions(self, request: _MammothRequest) -> tuple[torch.Tensor, torch.Tensor]:
        """Split AR hidden states and fail fast before touching model params."""
        text_cond, image_cond = self._split_ar_conditions(
            full_hidden_states=request.full_hidden_states,
            full_token_ids=request.full_token_ids,
            answer_start_index=request.answer_start_index,
        )
        if image_cond.shape[0] == 0:
            answer_token_ids = request.full_token_ids[request.answer_start_index :]
            raise ValueError(
                "MammothModa2 AR stage produced no visual-token hidden states for "
                f"request {request.request_id}; the DiT stage requires at least one generated visual token. "
                f"Generated token ids: {answer_token_ids[:32]}"
            )
        return text_cond, image_cond

    def _build_conditioning(
        self,
        text_cond: torch.Tensor,
        image_cond: torch.Tensor,
        model_device: torch.device,
        target_dtype: torch.dtype,
    ) -> _RequestConditioning:
        text_embeds = text_cond.to(device=model_device, dtype=target_dtype, non_blocking=True).contiguous().unsqueeze(0)
        text_mask = torch.ones((1, text_embeds.shape[1]), dtype=torch.bool, device=model_device)
        image_embeds = (
            image_cond.to(device=model_device, dtype=target_dtype, non_blocking=True).contiguous().unsqueeze(0)
        )
        image_mask = torch.ones((1, image_embeds.shape[1]), dtype=torch.bool, device=model_device)
        return _RequestConditioning(
            text_embeds=text_embeds,
            text_mask=text_mask,
            image_embeds=image_embeds,
            image_mask=image_mask,
        )

    @staticmethod
    def _make_latent_generators(
        specs: list[_MammothRequest],
        fallback_device: torch.device,
    ) -> list[torch.Generator] | None:
        generators: list[torch.Generator] = []
        for s in specs:
            if s.generator is not None:
                gen = s.generator[0] if isinstance(s.generator, list) else s.generator
            elif s.seed is not None:
                device = s.generator_device or fallback_device
                gen = torch.Generator(device=device).manual_seed(s.seed)
            else:
                return None
            generators.append(gen)
        return generators

    def _denoise_group(
        self,
        specs: list[_MammothRequest],
        conds: list[_RequestConditioning],
        model_device: torch.device,
    ) -> list[torch.Tensor]:
        batch = len(specs)
        height = specs[0].height
        width = specs[0].width
        num_inference_steps = specs[0].num_inference_steps
        target_dtype = conds[0].text_embeds.dtype

        # Conditioning collation. Token order is text-then-image per request,
        # so concatenate first and right-pad the combined sequence.
        nested_image_embedder = getattr(self.gen_transformer.time_caption_embed, "image_embedder", None)
        if self.gen_image_condition_refiner is not None:
            image_embeds, image_mask = _pad_cond_sequence(
                [c.image_embeds for c in conds],
                [c.image_mask for c in conds],
            )
            # Apply optional refiner ONLY on image condition tokens.
            if image_embeds.shape[1] > 0:
                image_embeds = self.gen_image_condition_refiner(image_embeds, ~image_mask.bool())
            refined_image_mask = torch.ones(
                image_embeds.shape[:2],
                dtype=torch.bool,
                device=image_embeds.device,
            )
            seq_embeds = []
            seq_masks = []
            for i, c in enumerate(conds):
                cur_img_embed = image_embeds[i : i + 1]
                cur_img_mask = refined_image_mask[i : i + 1]
                seq_embeds.append(torch.cat([c.text_embeds, cur_img_embed], dim=1))
                seq_masks.append(torch.cat([c.text_mask, cur_img_mask], dim=1))
            prompt_embeds, prompt_attention_mask = _pad_cond_sequence(seq_embeds, seq_masks)
            ar_image_embeds = None
            ar_image_attention_mask = None
        elif nested_image_embedder is not None:
            prompt_embeds, prompt_attention_mask = _pad_cond_sequence(
                [c.text_embeds for c in conds],
                [c.text_mask for c in conds],
            )
            ar_image_embeds, ar_image_attention_mask = _pad_cond_sequence(
                [c.image_embeds for c in conds],
                [c.image_mask for c in conds],
            )
        else:
            prompt_embeds, prompt_attention_mask = _pad_cond_sequence(
                [torch.cat([c.text_embeds, c.image_embeds], dim=1) for c in conds],
                [torch.cat([c.text_mask, c.image_mask], dim=1) for c in conds],
            )
            ar_image_embeds = None
            ar_image_attention_mask = None

        # Empty unconditional prompt for classifier-free guidance, shared by
        # all rows; rows that never take the CFG branch select their cond
        # prediction directly via torch.where below.
        needs_uncond = any(s.text_guidance_scale > 1.0 for s in specs)
        negative_prompt_embeds = None
        negative_prompt_attention_mask = None
        if needs_uncond:
            hidden_size = int(prompt_embeds.shape[-1])
            negative_prompt_embeds = torch.zeros(
                (batch, 0, hidden_size),
                dtype=target_dtype,
                device=model_device,
            )
            negative_prompt_attention_mask = torch.zeros(
                (batch, 0),
                dtype=torch.bool,
                device=model_device,
            )

        vae_scale_factor = 16
        latent_channels = int(self.gen_transformer.config.in_channels)
        shape = (batch, latent_channels, 2 * height // vae_scale_factor, 2 * width // vae_scale_factor)
        single_shape = (1, latent_channels, 2 * height // vae_scale_factor, 2 * width // vae_scale_factor)
        generators = self._make_latent_generators(specs, model_device)
        if generators is not None:
            noise_list = []
            for gen in generators:
                gen_device = gen.device if hasattr(gen, "device") else model_device
                n = randn_tensor(single_shape, generator=gen, device=gen_device, dtype=target_dtype)
                noise_list.append(n.to(device=model_device))
            latents = torch.cat(noise_list, dim=0)
        else:
            latents = randn_tensor(shape, device=model_device, dtype=target_dtype)

        scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(
            num_inference_steps=num_inference_steps,
            device=model_device,
            num_tokens=latents.shape[-2] * latents.shape[-1],
        )

        scale_vec = torch.tensor(
            [s.text_guidance_scale for s in specs],
            device=model_device,
            dtype=target_dtype,
        ).view(batch, 1, 1, 1)
        cfg_specs = [(s.text_guidance_scale > 1.0, float(s.cfg_range[0]), float(s.cfg_range[1])) for s in specs]

        total_steps = max(1, len(scheduler.timesteps))
        any_cfg_active = any(s.text_guidance_scale > 1.0 for s in specs)
        self._prepare_cache_dit_for_request(
            num_inference_steps=total_steps,
            cfg_active=any_cfg_active,
        )
        requires_paired_cfg = self._cache_dit_runtime.is_enabled and getattr(
            self, "_cache_dit_requires_paired_cfg", False
        )

        # Precompute mixed-CFG active masks across timesteps to avoid CPU list
        # comprehensions and H2D copies in the step loop.
        precomputed_active_masks = None
        all_active_per_step = None
        any_active_per_step = None
        if needs_uncond:
            precomputed_active_masks = torch.empty((total_steps, batch, 1, 1, 1), device=model_device, dtype=torch.bool)
            all_active_per_step = []
            any_active_per_step = []
            for step_idx in range(total_steps):
                frac = step_idx / total_steps
                step_mask = [is_active and (lo <= frac <= hi) for is_active, lo, hi in cfg_specs]
                all_active_per_step.append(all(step_mask))
                any_active_per_step.append(any(step_mask))
                mask_t = torch.tensor(step_mask, device=model_device, dtype=torch.bool)
                precomputed_active_masks[step_idx] = mask_t.view(batch, 1, 1, 1)

        # Precompute timesteps in latent dtype to eliminate per-step cast overhead
        timesteps_dtype = scheduler.timesteps.to(latents.dtype)

        # Run diffusion loop (CFG supported when text_guidance_scale > 1.0)
        for i, t in enumerate(scheduler.timesteps):
            timestep = timesteps_dtype[i].expand(batch)
            model_pred = self.gen_transformer(
                hidden_states=latents,
                timestep=timestep,
                text_hidden_states=prompt_embeds,
                text_attention_mask=prompt_attention_mask,
                ref_image_hidden_states=None,
                ar_image_hidden_states=ar_image_embeds,
                ar_image_attention_mask=ar_image_attention_mask,
                freqs_cis=self.gen_freqs_cis,
            )
            run_uncond = (any_active_per_step is not None and any_active_per_step[i]) if needs_uncond else False
            if requires_paired_cfg and negative_prompt_embeds is not None:
                run_uncond = True
            if run_uncond:
                model_pred_uncond = self.gen_transformer(
                    hidden_states=latents,
                    timestep=timestep,
                    text_hidden_states=negative_prompt_embeds,
                    text_attention_mask=negative_prompt_attention_mask,
                    ref_image_hidden_states=None,
                    freqs_cis=self.gen_freqs_cis,
                )
                # Fused CFG blend: torch.lerp fuses (uncond + scale * (cond - uncond)) into 1 kernel
                blended = torch.lerp(model_pred_uncond, model_pred, scale_vec)
                if all_active_per_step is not None and all_active_per_step[i]:
                    # Fast path: all requests in the batch are active.
                    model_pred = blended
                else:
                    # Inactive rows keep their conditional prediction exactly:
                    # gating the blend with torch.where avoids propagating
                    # uncond-branch NaN/rounding into CFG-free rows.
                    active_tensor = precomputed_active_masks[i]
                    model_pred = torch.where(active_tensor, blended, model_pred)
            latents = scheduler.step(model_pred, t, latents, return_dict=False)[0]
            latents = latents.to(dtype=target_dtype)

        # VAE decode (in-place scaling to reduce peak VRAM before decode)
        if self.gen_vae.config.scaling_factor is not None:
            latents.div_(self.gen_vae.config.scaling_factor)
        if self.gen_vae.config.shift_factor is not None:
            latents.add_(self.gen_vae.config.shift_factor)
        vae_dtype = next(self.gen_vae.parameters()).dtype
        image = self.gen_vae.decode(latents.to(dtype=vae_dtype), return_dict=False)[0]  # [B, C, H, W]
        return [image[i : i + 1] for i in range(batch)]

    @torch.inference_mode()
    def forward(self, req: DiffusionRequestBatch) -> list[DiffusionOutput]:
        specs = [self._parse_request(req, i) for i in range(req.num_reqs)]

        # Validate AR conditions before touching model parameters so malformed
        # requests fail fast instead of surfacing device/dtype errors.
        raw_conds = [self._split_request_conditions(spec) for spec in specs]

        model_device = next(self.parameters()).device
        if self.gen_image_condition_refiner is not None:
            target_dtype = next(self.gen_image_condition_refiner.parameters()).dtype
        else:
            target_dtype = next(self.gen_transformer.parameters()).dtype

        conds = [
            self._build_conditioning(text_cond, image_cond, model_device, target_dtype)
            for text_cond, image_cond in raw_conds
        ]

        groups = self._group_requests(specs)
        if len(groups) == 1 and len(groups[0]) == len(specs):
            images = self._denoise_group(specs, conds, model_device)
            return [DiffusionOutput(output=img) for img in images]

        outputs: list[DiffusionOutput | None] = [None] * len(specs)
        for group in groups:
            images = self._denoise_group(
                [specs[i] for i in group],
                [conds[i] for i in group],
                model_device,
            )
            for index, image in zip(group, images):
                outputs[index] = DiffusionOutput(output=image)
        if any(output is None for output in outputs):
            raise RuntimeError("DiT batching produced no image for at least one scheduled request")
        return [output for output in outputs if output is not None]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
