# Summary

## Entry Points

Main entry points for vLLM-omni inference and serving.

- [vllm_omni.entrypoints.omni_llm.OmniLLM][]
- [vllm_omni.entrypoints.omni_stage.OmniStage][]

## Inputs

Input data structures for multi-modal inputs.

- [vllm_omni.inputs.data.OmniTokenInputs][]
- [vllm_omni.inputs.data.OmniTokensPrompt][]
- [vllm_omni.inputs.parse][]
- [vllm_omni.inputs.preprocess.OmniInputPreprocessor][]

## Engine

Engine classes for offline and online inference.

- [vllm_omni.engine.processor.OmniProcessor][]
- [vllm_omni.engine.output_processor.MultimodalOutputProcessor][]

## Core

Core scheduling and caching components.

- [vllm_omni.core.sched.scheduler.OmniScheduler][]
- [vllm_omni.core.sched.diffusion_scheduler.DiffusionScheduler][]
- [vllm_omni.core.sched.output.OmniNewRequestData][]
- [vllm_omni.core.dit_cache_manager.DiTCacheManager][]

## Model Executor

Model execution components.

- [vllm_omni.model_executor.models.qwen2_5_omni.Qwen2_5OmniForConditionalGeneration][]
- [vllm_omni.model_executor.models.qwen2_5_omni_talker.Qwen2_5OmniTalkerForConditionalGeneration][]
- [vllm_omni.model_executor.models.qwen2_5_omni_thinker.Qwen2_5OmniThinkerForConditionalGeneration][]
- [vllm_omni.model_executor.models.qwen2_5_omni_token2wav.Qwen2_5OmniToken2WavForConditionalGenerationVLLM][]

## Configuration

Configuration classes.

- [vllm_omni.config][]

## Workers

Worker classes for distributed inference.

- [vllm_omni.worker.gpu_ar_worker.GPUARWorker][]
- [vllm_omni.worker.gpu_diffusion_worker.GPUDiffusionWorker][]

