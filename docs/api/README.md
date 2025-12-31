# Summary

## Entry Points

Main entry points for vLLM-Omni inference and serving.

- [vllm_omni.entrypoints.async_omni.AsyncOmni][]
- [vllm_omni.entrypoints.async_omni_diffusion.AsyncOmniDiffusion][]
- [vllm_omni.entrypoints.async_omni_llm.AsyncOmniLLM][]
- [vllm_omni.entrypoints.chat_utils.OmniAsyncMultiModalContentParser][]
- [vllm_omni.entrypoints.chat_utils.OmniAsyncMultiModalItemTracker][]
- [vllm_omni.entrypoints.chat_utils.parse_chat_messages_futures][]
- [vllm_omni.entrypoints.cli.serve.OmniServeCommand][]
- [vllm_omni.entrypoints.client_request_state.ClientRequestState][]
- [vllm_omni.entrypoints.log_utils.OrchestratorMetrics][]
- [vllm_omni.entrypoints.log_utils.StageRequestMetrics][]
- [vllm_omni.entrypoints.log_utils.StageStats][]
- [vllm_omni.entrypoints.omni.Omni][]
- [vllm_omni.entrypoints.omni.OmniBase][]
- [vllm_omni.entrypoints.omni_diffusion.OmniDiffusion][]
- [vllm_omni.entrypoints.omni_llm.OmniLLM][]
- [vllm_omni.entrypoints.omni_stage.OmniStage][]
- [vllm_omni.entrypoints.stage_utils.OmniStageTaskType][]

## Inputs

Input data structures for multi-modal inputs.

- [vllm_omni.inputs.data.OmniEmbedsPrompt][]
- [vllm_omni.inputs.data.OmniTokenInputs][]
- [vllm_omni.inputs.data.OmniTokensPrompt][]
- [vllm_omni.inputs.parse.parse_singleton_prompt_omni][]
- [vllm_omni.inputs.preprocess.OmniInputPreprocessor][]

## Engine

Engine classes for offline and online inference.

- [vllm_omni.diffusion.diffusion_engine.BackgroundResources][]
- [vllm_omni.diffusion.diffusion_engine.DiffusionEngine][]
- [vllm_omni.engine.AdditionalInformationEntry][]
- [vllm_omni.engine.AdditionalInformationPayload][]
- [vllm_omni.engine.OmniEngineCoreOutput][]
- [vllm_omni.engine.OmniEngineCoreOutputs][]
- [vllm_omni.engine.OmniEngineCoreRequest][]
- [vllm_omni.engine.PromptEmbedsPayload][]
- [vllm_omni.engine.arg_utils.AsyncOmniEngineArgs][]
- [vllm_omni.engine.arg_utils.OmniEngineArgs][]
- [vllm_omni.engine.input_processor.OmniInputProcessor][]
- [vllm_omni.engine.output_processor.MultimodalOutputProcessor][]
- [vllm_omni.engine.output_processor.OmniRequestState][]

## Core

Core scheduling and caching components.

- [vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler][]
- [vllm_omni.core.sched.omni_generation_scheduler.OmniGenerationScheduler][]
- [vllm_omni.core.sched.output.OmniNewRequestData][]

## Model Executor

Model execution components.

- [vllm_omni.model_executor.custom_process_mixin.CustomProcessMixin][]
- [vllm_omni.model_executor.models.output_templates.OmniOutput][]
- [vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni.Qwen2_5OmniForConditionalGeneration][]
- [vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_talker.Qwen2_5OmniTalkerForConditionalGeneration][]
- [vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_thinker.Qwen2_5OmniConditionalGenerationMixin][]
- [vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_thinker.Qwen2_5OmniThinkerForConditionalGeneration][]
- [vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_token2wav.Qwen2_5OmniToken2WavBigVGANModel][]
- [vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_token2wav.Qwen2_5OmniToken2WavDiTModel][]
- [vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_token2wav.Qwen2_5OmniToken2WavForConditionalGenerationVLLM][]
- [vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_token2wav.Qwen2_5OmniToken2WavModel][]
- [vllm_omni.model_executor.models.qwen2_5_omni.qwen2_old.Qwen2EmbeddingModel][]
- [vllm_omni.model_executor.models.qwen2_5_omni.qwen2_old.Qwen2ForCausalLM][]
- [vllm_omni.model_executor.models.qwen2_5_omni.qwen2_old.Qwen2Model][]
- [vllm_omni.model_executor.models.qwen3_omni.qwen3_moe.Qwen3MoeForCausalLM][]
- [vllm_omni.model_executor.models.qwen3_omni.qwen3_omni.Qwen3OmniMoeForConditionalGeneration][]
- [vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_code2wav.Qwen3OmniMoeCode2Wav][]
- [vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_code_predictor_mtp.Qwen3OmniCodePredictorBaseModel][]
- [vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_code_predictor_mtp.Qwen3OmniMoeTalkerCodePredictor][]
- [vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_talker.Qwen3OmniMoeModel][]
- [vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_talker.Qwen3OmniMoeTalkerForConditionalGeneration][]
- [vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker.Qwen3MoeLLMForCausalLM][]
- [vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker.Qwen3MoeLLMModel][]
- [vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker.Qwen3OmniMoeConditionalGenerationMixin][]
- [vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker.Qwen3OmniMoeThinkerForConditionalGeneration][]
- [vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker.Qwen3OmniMoeThinkerMultiModalProcessor][]
- [vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker.Qwen3OmniMoeThinkerProcessingInfo][]

## Configuration

Configuration classes.

- [vllm_omni.config.model.OmniModelConfig][]
- [vllm_omni.diffusion.cache.teacache.config.TeaCacheConfig][]
- [vllm_omni.distributed.omni_connectors.utils.config.ConnectorSpec][]
- [vllm_omni.distributed.omni_connectors.utils.config.OmniTransferConfig][]

## Workers

Worker classes and model runners for distributed inference.

- [vllm_omni.diffusion.worker.gpu_worker.GPUWorker][]
- [vllm_omni.diffusion.worker.gpu_worker.WorkerProc][]
- [vllm_omni.diffusion.worker.npu.npu_worker.NPUWorker][]
- [vllm_omni.diffusion.worker.npu.npu_worker.NPUWorkerProc][]
- [vllm_omni.worker.gpu_ar_model_runner.ExecuteModelState][]
- [vllm_omni.worker.gpu_ar_model_runner.GPUARModelRunner][]
- [vllm_omni.worker.gpu_ar_worker.GPUARWorker][]
- [vllm_omni.worker.gpu_generation_model_runner.GPUGenerationModelRunner][]
- [vllm_omni.worker.gpu_generation_worker.GPUGenerationWorker][]
- [vllm_omni.worker.gpu_model_runner.OmniGPUModelRunner][]
- [vllm_omni.worker.npu.npu_ar_model_runner.ExecuteModelState][]
- [vllm_omni.worker.npu.npu_ar_model_runner.NPUARModelRunner][]
- [vllm_omni.worker.npu.npu_ar_worker.NPUARWorker][]
- [vllm_omni.worker.npu.npu_generation_model_runner.NPUGenerationModelRunner][]
- [vllm_omni.worker.npu.npu_generation_worker.NPUGenerationWorker][]
- [vllm_omni.worker.npu.npu_model_runner.OmniNPUModelRunner][]
