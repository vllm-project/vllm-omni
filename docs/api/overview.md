# Summary

## Entry Points

Main entry points for vLLM-omni inference and serving.

- [vllm_omni.entrypoints.omni_llm.OmniLLM][]
- [vllm_omni.entrypoints.omni_llm.OmniStageLLM][]
- [vllm_omni.entrypoints.omni_stage.OmniStage][]
- [vllm_omni.entrypoints.cli.serve.OmniServeCommand][]

## Inputs

Input data structures for multi-modal inputs.

- [vllm_omni.inputs.data.OmniTokenInputs][]
- [vllm_omni.inputs.data.OmniTokensPrompt][]
- [vllm_omni.inputs.data.OmniEmbedsPrompt][]
- [vllm_omni.inputs.parse.parse_singleton_prompt_omni][]
- [vllm_omni.inputs.preprocess.OmniInputPreprocessor][]

## Engine

Engine classes for offline and online inference.

- [vllm_omni.engine.arg_utils.OmniEngineArgs][]
- [vllm_omni.engine.processor.OmniProcessor][]
- [vllm_omni.engine.output_processor.OmniRequestState][]
- [vllm_omni.engine.output_processor.MultimodalOutputProcessor][]
- [vllm_omni.engine.PromptEmbedsPayload][]
- [vllm_omni.engine.AdditionalInformationPayload][]
- [vllm_omni.engine.OmniEngineCoreRequest][]

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
- [vllm_omni.model_executor.layers.mrope.MRotaryEmbedding][]
- [vllm_omni.model_executor.models.registry.OmniModelRegistry][]

## Configuration

Configuration classes.

- [vllm_omni.config.OmniModelConfig][]

## Workers

Worker classes and model runners for distributed inference.

- [vllm_omni.worker.gpu_ar_worker.GPUARWorker][]
- [vllm_omni.worker.gpu_diffusion_worker.GPUDiffusionWorker][]
- [vllm_omni.worker.gpu_model_runner.OmniGPUModelRunner][]
- [vllm_omni.worker.gpu_ar_model_runner.GPUARModelRunner][]
- [vllm_omni.worker.gpu_diffusion_model_runner.GPUDiffusionModelRunner][]

