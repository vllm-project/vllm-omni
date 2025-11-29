# Summary

## Entry Points

Main entry points for vLLM-Omni inference and serving.

- [vllm_omni.entrypoints.async_omni_llm.AsyncOmniLLM][]
- [vllm_omni.entrypoints.async_omni_llm.AsyncOmniStageLLM][]
- [vllm_omni.entrypoints.chat_utils.OmniAsyncMultiModalContentParser][]
- [vllm_omni.entrypoints.chat_utils.OmniAsyncMultiModalItemTracker][]
- [vllm_omni.entrypoints.chat_utils.parse_chat_messages_futures][]
- [vllm_omni.entrypoints.cli.serve.OmniServeCommand][]
- [vllm_omni.entrypoints.log_utils.OrchestratorMetrics][]
- [vllm_omni.entrypoints.omni.Omni][]
- [vllm_omni.entrypoints.omni_llm.OmniLLM][]
- [vllm_omni.entrypoints.omni_llm.OmniStageLLM][]
- [vllm_omni.entrypoints.omni_stage.OmniStage][]
- [vllm_omni.entrypoints.openai.serving_chat.OmniOpenAIServingChat][]

## Inputs

Input data structures for multi-modal inputs.

- [vllm_omni.inputs.data.OmniEmbedsPrompt][]
- [vllm_omni.inputs.data.OmniTokenInputs][]
- [vllm_omni.inputs.data.OmniTokensPrompt][]
- [vllm_omni.inputs.parse.parse_singleton_prompt_omni][]
- [vllm_omni.inputs.preprocess.OmniInputPreprocessor][]

## Engine

Engine classes for offline and online inference.

- [vllm_omni.diffusion.diffusion_engine.DiffusionEngine][]
- [vllm_omni.engine.AdditionalInformationEntry][]
- [vllm_omni.engine.AdditionalInformationPayload][]
- [vllm_omni.engine.OmniEngineCoreRequest][]
- [vllm_omni.engine.PromptEmbedsPayload][]
- [vllm_omni.engine.arg_utils.AsyncOmniEngineArgs][]
- [vllm_omni.engine.arg_utils.OmniEngineArgs][]
- [vllm_omni.engine.output_processor.MultimodalOutputProcessor][]
- [vllm_omni.engine.output_processor.OmniRequestState][]
- [vllm_omni.engine.processor.OmniProcessor][]

## Core

Core scheduling and caching components.

- [vllm_omni.core.dit_cache_manager.DiTCacheManager][]
- [vllm_omni.core.sched.diffusion_scheduler.DiffusionScheduler][]
- [vllm_omni.core.sched.output.OmniNewRequestData][]
- [vllm_omni.core.sched.scheduler.OmniScheduler][]

## Model Executor

Model execution components.

- [vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni.OmniOutput][]
- [vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni.Qwen2_5OmniForConditionalGeneration][]
- [vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_talker.Qwen2_5OmniTalkerForConditionalGeneration][]
- [vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_thinker.Qwen2_5OmniAudioFeatureInputs][]
- [vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_thinker.Qwen2_5OmniConditionalGenerationMixin][]
- [vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_thinker.Qwen2_5OmniThinkerDummyInputsBuilder][]
- [vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_thinker.Qwen2_5OmniThinkerForConditionalGeneration][]
- [vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_thinker.Qwen2_5OmniThinkerMultiModalDataParser][]
- [vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_thinker.Qwen2_5OmniThinkerMultiModalProcessor][]
- [vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_thinker.Qwen2_5OmniThinkerProcessingInfo][]
- [vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_token2wav.Qwen2_5OmniToken2WavBigVGANModel][]
- [vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_token2wav.Qwen2_5OmniToken2WavDiTModel][]
- [vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_token2wav.Qwen2_5OmniToken2WavForConditionalGenerationVLLM][]
- [vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_token2wav.Qwen2_5OmniToken2WavModel][]
- [vllm_omni.model_executor.models.qwen2_5_omni.qwen2_old.Qwen2EmbeddingModel][]
- [vllm_omni.model_executor.models.qwen2_5_omni.qwen2_old.Qwen2ForCausalLM][]
- [vllm_omni.model_executor.models.qwen2_5_omni.qwen2_old.Qwen2Model][]

## Configuration

Configuration classes.

- [vllm_omni.config.model.OmniModelConfig][]

## Workers

Worker classes and model runners for distributed inference.

- [vllm_omni.diffusion.worker.gpu_worker.GPUWorker][]
- [vllm_omni.diffusion.worker.gpu_worker.WorkerProc][]
- [vllm_omni.worker.gpu_ar_model_runner.GPUARModelRunner][]
- [vllm_omni.worker.gpu_ar_worker.GPUARWorker][]
- [vllm_omni.worker.gpu_diffusion_model_runner.GPUDiffusionModelRunner][]
- [vllm_omni.worker.gpu_diffusion_worker.GPUDiffusionWorker][]
- [vllm_omni.worker.gpu_model_runner.OmniGPUModelRunner][]
