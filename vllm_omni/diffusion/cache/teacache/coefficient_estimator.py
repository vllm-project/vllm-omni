# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import numpy as np
import torch
from vllm.config import LoadConfig
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_omni.diffusion.cache.teacache.extractors import get_extractor
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.hooks import HookRegistry, ModelHook
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.bagel.pipeline_bagel import BagelPipeline
from vllm_omni.diffusion.models.flux2.pipeline_flux2 import Flux2Pipeline
from vllm_omni.diffusion.models.stable_audio.pipeline_stable_audio import StableAudioPipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


class DataCollectionHook(ModelHook):
    """Hook to collect modulated inputs and model outputs for TeaCache coefficient estimation."""

    _HOOK_NAME = "teacache_collector"

    def __init__(self, transformer_type: str):
        super().__init__()
        self.transformer_type = transformer_type
        self.extractor_fn = None
        self.current_trajectory: list[tuple[np.ndarray, np.ndarray]] = []

    def initialize_hook(self, module: torch.nn.Module) -> torch.nn.Module:
        self.extractor_fn = get_extractor(self.transformer_type)
        return module

    def new_forward(self, module: torch.nn.Module, *args: Any, **kwargs: Any) -> Any:
        ctx = self.extractor_fn(module, *args, **kwargs)
        modulated_input_cpu = ctx.modulated_input.detach().float().cpu().numpy()

        outputs = ctx.run_transformer_blocks()
        ctx.hidden_states = outputs[0]
        if len(outputs) > 1 and ctx.encoder_hidden_states is not None:
            ctx.encoder_hidden_states = outputs[1]

        model_output_cpu = ctx.hidden_states.detach().float().cpu().numpy()
        self.current_trajectory.append((modulated_input_cpu, model_output_cpu))
        return ctx.postprocess(ctx.hidden_states)

    def start_collection(self):
        self.current_trajectory = []

    def stop_collection(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return list(self.current_trajectory)


class BagelAdapter:
    """Adapter for Bagel model."""

    @staticmethod
    def load_pipeline(model_path: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> BagelPipeline:
        od_config = OmniDiffusionConfig.from_kwargs(model=model_path, dtype=dtype)
        od_config.model_class_name = "BagelPipeline"

        pipeline = BagelPipeline(od_config=od_config)
        loader = DiffusersPipelineLoader(LoadConfig())
        loader.load_weights(pipeline)
        pipeline.to(device)
        return pipeline

    @staticmethod
    def get_transformer(pipeline: Any) -> tuple[Any, str]:
        return pipeline.bagel, "Bagel"

    @staticmethod
    def install_hook(transformer: Any, hook: DataCollectionHook) -> None:
        registry = HookRegistry.get_or_create(transformer)
        registry.register_hook(hook._HOOK_NAME, hook)


class StableAudioAdapter:
    """Adapter for Stable Audio Open 1.0 coefficient estimation."""

    @staticmethod
    def load_pipeline(model_path: str, device: str = "cuda", dtype: torch.dtype = torch.float16) -> Any:
        od_config = OmniDiffusionConfig.from_kwargs(model=model_path, dtype=dtype)

        # Strictly necessary because we bypass loader.load_model()
        with set_default_torch_dtype(dtype):
            pipeline = StableAudioPipeline(od_config=od_config)

        loader = DiffusersPipelineLoader(LoadConfig())
        loader.load_weights(pipeline)
        pipeline.to(device)
        return pipeline

    @staticmethod
    def get_transformer(pipeline: Any) -> tuple[Any, str]:
        return pipeline.transformer, "StableAudioDiTModel"

    @staticmethod
    def install_hook(transformer: Any, hook: DataCollectionHook) -> None:
        registry = HookRegistry.get_or_create(transformer)
        registry.register_hook(hook._HOOK_NAME, hook)


class Flux2Adapter:
    """Adapter for Flux2 model coefficient estimation."""

    @staticmethod
    def load_pipeline(model_path: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> Flux2Pipeline:
        """Load Flux2 pipeline for coefficient estimation."""
        od_config = OmniDiffusionConfig.from_kwargs(model=model_path, dtype=dtype)
        od_config.model_class_name = "Flux2Pipeline"

        pipeline = Flux2Pipeline(od_config=od_config)
        loader = DiffusersPipelineLoader(LoadConfig())
        loader.load_weights(pipeline)
        pipeline.to(device)
        return pipeline

    @staticmethod
    def get_transformer(pipeline: Any) -> tuple[Any, str]:
        return pipeline.transformer, pipeline.transformer.__class__.__name__

    @staticmethod
    def install_hook(transformer: Any, hook: DataCollectionHook) -> None:
        registry = HookRegistry.get_or_create(transformer)
        registry.register_hook(hook._HOOK_NAME, hook)


class DefaultAdapter:
    """Default adapter for standard diffusers pipelines."""

    @staticmethod
    def load_pipeline(model_path: str, device: str, dtype: torch.dtype) -> Any:
        raise NotImplementedError("DefaultAdapter.load_pipeline not implemented")

    @staticmethod
    def get_transformer(pipeline: Any) -> tuple[Any, str]:
        return pipeline.transformer, pipeline.transformer.__class__.__name__

    @staticmethod
    def install_hook(transformer: Any, hook: DataCollectionHook) -> None:
        registry = HookRegistry.get_or_create(transformer)
        registry.register_hook(hook._HOOK_NAME, hook)


_MODEL_ADAPTERS: dict[str, type] = {
    "Bagel": BagelAdapter,
    "StableAudio": StableAudioAdapter,
    "Flux2": Flux2Adapter,
}

_EPSILON = 1e-6


def calculate_relative_l1(tensor_current: np.ndarray, tensor_next: np.ndarray) -> float:
    """Calculate relative L1 distance (Eq. 4 from TeaCache paper)."""
    diff = np.abs(tensor_current - tensor_next).sum()
    norm = np.abs(tensor_current).sum() + _EPSILON
    return diff / norm


def estimate_teacache_coefficients(
    collected_data: list[list[tuple[np.ndarray, np.ndarray]]], poly_order: int = 4
) -> list[float]:
    """Estimate polynomial coefficients for TeaCache using np.polyfit."""
    input_diffs, output_diffs = [], []

    for sample in collected_data:
        for t in range(len(sample) - 1):
            feat_in_curr, feat_out_curr = sample[t]
            feat_in_next, feat_out_next = sample[t + 1]
            input_diffs.append(calculate_relative_l1(feat_in_curr, feat_in_next))
            output_diffs.append(calculate_relative_l1(feat_out_curr, feat_out_next))

    x = np.array(input_diffs, dtype=np.float64)
    y = np.array(output_diffs, dtype=np.float64)

    print("Data statistics:")
    print(f"  Count: {len(x)}")
    print(f"  Input Diffs (x): min={x.min():.4e}, max={x.max():.4e}, mean={x.mean():.4e}")
    print(f"  Output Diffs (y): min={y.min():.4e}, max={y.max():.4e}, mean={y.mean():.4e}")

    return np.polyfit(x, y, poly_order).tolist()


class TeaCacheCoefficientEstimator:
    """Model-agnostic helper class to collect data and estimate TeaCache coefficients."""

    def __init__(
        self,
        model_path: str,
        model_type: str = "Bagel",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        # Add validation here ⬇️
        if model_type not in _MODEL_ADAPTERS:
            available_types = list(_MODEL_ADAPTERS.keys())
            raise ValueError(
                f"Unsupported model_type: '{model_type}'. "
                f"Available types: {available_types}. "
                f"To add support for a new model, add an entry to _MODEL_ADAPTERS."
            )

        adapter = _MODEL_ADAPTERS.get(model_type, DefaultAdapter)
        self.pipeline = adapter.load_pipeline(model_path, device, dtype)
        self.transformer, self.transformer_type = adapter.get_transformer(self.pipeline)
        self.hook = DataCollectionHook(self.transformer_type)
        self.collected_data: list[list[tuple[np.ndarray, np.ndarray]]] = []
        adapter.install_hook(self.transformer, self.hook)

    def collect_from_prompt(self, prompt: str, **generate_kwargs):
        self.hook.start_collection()
        req = OmniDiffusionRequest(
            prompts=[prompt],
            sampling_params=OmniDiffusionSamplingParams(
                num_inference_steps=generate_kwargs.get("num_inference_steps", 20),
                seed=generate_kwargs.get("seed", 42),
            ),
        )
        with torch.no_grad():
            self.pipeline.forward(req)
        trajectory = self.hook.stop_collection()
        if trajectory:
            self.collected_data.append(trajectory)
        torch.cuda.empty_cache()

    def estimate(self, poly_order: int = 4) -> list[float]:
        """Estimate polynomial coefficients from collected data.

        Args:
            poly_order: Order of polynomial fit (default: 4)

        Returns:
            List of polynomial coefficients [a_n, a_{n-1}, ..., a_1, a_0]

        Raises:
            RuntimeError: If no data has been collected
        """
        if not self.collected_data:
            raise RuntimeError(
                "No data collected for coefficient estimation. "
                "Call collect_from_prompt() at least once before calling estimate()."
            )
        return estimate_teacache_coefficients(self.collected_data, poly_order)
