# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
from typing import ClassVar, Iterable

import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.schedulers import FlowUniPCMultistepScheduler
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import (
    DiffusionPipelineProfilerMixin,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = init_logger(__name__)


def build_hidream_o1_scheduler(
    *,
    scheduler_name: str = "default",
    num_inference_steps: int = 50,
    shift: float = 3.0,
    device: torch.device | str | None = None,
    timesteps_list: list[int] | None = None,
) -> FlowUniPCMultistepScheduler:
    """Build a HiDream-O1 scheduler + set_timesteps in one call.

    Mirrors upstream `models/pipeline.py::build_scheduler` (default branch)
    @21bcd30471ac; class-vs-upstream numerical parity was validated on H100
    against a vendored copy of the upstream scheduler. Only
    scheduler_name='default' is supported today.
    """
    if scheduler_name != "default":
        raise NotImplementedError(
            f"HiDream-O1-Image currently only supports scheduler_name='default'; "
            f"got {scheduler_name!r}. 'flash' and 'flow_match' are dev-model "
            f"schedulers and are not yet supported."
        )

    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000,
        shift=shift,
        prediction_type="flow_prediction",
        use_dynamic_shifting=False,
    )
    scheduler.set_timesteps(num_inference_steps, device=device)

    if timesteps_list is not None:
        scheduler.timesteps = torch.tensor(
            timesteps_list, device=device, dtype=torch.long
        )
        sigmas = [t.item() / 1000.0 for t in scheduler.timesteps]
        sigmas.append(0.0)
        scheduler.sigmas = torch.tensor(sigmas, device=device)

    return scheduler


def get_hidream_o1_image_post_process_func(od_config: OmniDiffusionConfig):
    """Identity post-processor; `forward()` already returns a tensor."""
    del od_config

    def post_process_func(x):
        return x

    return post_process_func


class HiDreamO1ImagePipeline(nn.Module, DiffusionPipelineProfilerMixin):
    """HiDream-O1-Image pixel-DiT unified transformer pipeline.

    Current scope: text-to-image only, single GPU, bfloat16, no CFG.
    See __init__.py for the full status matrix.
    """

    # Workaround while forward() is a NotImplementedError stub: 0 makes
    # DiffusionEngine skip the dummy warmup run. Remove once forward() works.
    dummy_run_num_frames: ClassVar[int] = 0

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.prefix = prefix
        self.model_dir = od_config.model

        custom_args = od_config.custom_pipeline_args or {}
        self.model_type: str = str(custom_args.get("model_type", "full"))
        assert self.model_type in ("full", "dev"), (
            f"HiDream-O1-Image model_type must be 'full' or 'dev', got {self.model_type!r}"
        )

        self.dtype = od_config.dtype if od_config.dtype is not None else torch.bfloat16
        self.device = get_local_device()

        self.setup_diffusion_pipeline_profiler(
            profiler_targets=["forward"],
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler,
        )

        self.processor = None
        self.model = None
        self._init_processor_and_model()

    def _init_processor_and_model(self) -> None:
        """Eagerly load AutoProcessor + HiDreamO1ImageTransformer.

        Mirrors upstream inference.py L60-70 (SHA 21bcd30471ac).
        `HiDreamO1ImageTransformer` is an alias for
        `Qwen3VLForConditionalGeneration`.
        """
        from transformers import AutoProcessor
        from vllm_omni.diffusion.models.hidream_o1_image.hidream_o1_image_transformer import (
            HiDreamO1ImageTransformer,
        )

        logger.info(
            "HiDreamO1ImagePipeline: loading processor + model from %s (dtype=%s, device=%s)",
            self.model_dir, self.dtype, self.device,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_dir)
        self.model = HiDreamO1ImageTransformer.from_pretrained(
            self.model_dir,
            torch_dtype=self.dtype,
            device_map=self.device,
        ).eval()

        self._add_special_tokens(self.processor)
        logger.info(
            "HiDreamO1ImagePipeline: ready (model_type=%s, num_params=%.1fB)",
            self.model_type, sum(p.numel() for p in self.model.parameters()) / 1e9,
        )

    @staticmethod
    def _add_special_tokens(processor) -> None:
        """Attach 5 special-token literal shortcuts on the tokenizer.

        Verbatim port of upstream inference.py::add_special_tokens +
        get_tokenizer (SHA 21bcd30471ac). Semantic role:
          - boi_token: image-related sequence marker
          - bor_token / eor_token: reference region begin/end markers
          - bot_token: text-related marker
          - tms_token: timestep embedding replacement marker (denoise loop
            replaces this token's embedding with the pixel-DiT timestep vector)
        """
        from transformers import PreTrainedTokenizerBase

        tok = (
            processor
            if isinstance(processor, PreTrainedTokenizerBase)
            else processor.tokenizer
        )
        tok.boi_token = "<|boi_token|>"
        tok.bor_token = "<|bor_token|>"
        tok.eor_token = "<|eor_token|>"
        tok.bot_token = "<|bot_token|>"
        tok.tms_token = "<|tms_token|>"

    @torch.inference_mode()
    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        """Run one denoising job.

        Adapts upstream `pipeline.py::generate_image` to vllm-omni's Request
        contract. Denoise loop + CFG will be landed in follow-up commits.
        """
        raise NotImplementedError(
            "HiDreamO1ImagePipeline.forward is not yet implemented."
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Return the full param-name set (Path B Pattern 1).

        Model was loaded eagerly in __init__, but `diffusers_loader` runs
        a strict "all params covered" check that requires the returned set
        to cover every named parameter.
        """
        for _ in weights:
            pass
        return {name for name, _ in self.named_parameters()}

    def has_real_checkpoint(self) -> bool:
        """Check whether the model dir has actual weight shards."""
        if not self.model_dir:
            return False
        return (
            os.path.exists(os.path.join(self.model_dir, "model.safetensors"))
            or os.path.exists(os.path.join(self.model_dir, "model.safetensors.index.json"))
        )
