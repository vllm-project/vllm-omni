# Summary

## Entry Points

Main entry points for vLLM-Omni inference and serving.

- [vllm_omni.entrypoints.async_omni.AsyncOmni][]
- [vllm_omni.engine.cfg_companion_tracker.CfgCompanionTracker][]
- [vllm_omni.entrypoints.cli.benchmark.base.OmniBenchmarkSubcommandBase][]
- [vllm_omni.entrypoints.cli.benchmark.main.OmniBenchmarkSubcommand][]
- [vllm_omni.entrypoints.cli.benchmark.serve.OmniBenchmarkServingSubcommand][]
- [vllm_omni.entrypoints.cli.serve.OmniServeCommand][]
- [vllm_omni.entrypoints.client_request_state.ClientRequestState][]
- [vllm_omni.entrypoints.omni.Omni][]
- [vllm_omni.entrypoints.omni_base.OmniBase][]
- [vllm_omni.entrypoints.pd_utils.PDDisaggregationMixin][]

## Inputs

Input data structures for multi-modal inputs.

- [vllm_omni.inputs.data.OmniCustomPrompt][]
- [vllm_omni.inputs.data.OmniDiffusionSamplingParams][]
- [vllm_omni.inputs.data.OmniEmbedsPrompt][]
- [vllm_omni.inputs.data.OmniTextPrompt][]
- [vllm_omni.inputs.data.OmniTokenInputs][]
- [vllm_omni.inputs.data.OmniTokensPrompt][]
- [vllm_omni.inputs.preprocess.OmniInputPreprocessor][]

## Engine

Engine classes for offline and online inference.

- [vllm_omni.diffusion.diffusion_engine.DiffusionEngine][]
- [vllm_omni.distributed.omni_connectors.connectors.mooncake_transfer_engine_connector.BufferAllocator][]
- [vllm_omni.distributed.omni_connectors.connectors.mooncake_transfer_engine_connector.ManagedBuffer][]
- [vllm_omni.distributed.omni_connectors.connectors.mooncake_transfer_engine_connector.MooncakeAgentMetadata][]
- [vllm_omni.distributed.omni_connectors.connectors.mooncake_transfer_engine_connector.MooncakeTransferEngineConnector][]
- [vllm_omni.distributed.omni_connectors.connectors.mooncake_transfer_engine_connector.QueryRequest][]
- [vllm_omni.distributed.omni_connectors.connectors.mooncake_transfer_engine_connector.QueryResponse][]
- [vllm_omni.engine.AdditionalInformationEntry][]
- [vllm_omni.engine.AdditionalInformationPayload][]
- [vllm_omni.engine.OmniEngineCoreOutput][]
- [vllm_omni.engine.OmniEngineCoreOutputs][]
- [vllm_omni.engine.OmniEngineCoreRequest][]
- [vllm_omni.engine.PromptEmbedsPayload][]
- [vllm_omni.engine.arg_utils.OmniEngineArgs][]
- [vllm_omni.engine.async_omni_engine.AsyncOmniEngine][]
- [vllm_omni.engine.mm_outputs.MultimodalCompletionOutput][]
- [vllm_omni.engine.mm_outputs.MultimodalPayload][]
- [vllm_omni.engine.orchestrator.Orchestrator][]
- [vllm_omni.engine.orchestrator.OrchestratorRequestState][]
- [vllm_omni.engine.output_modality.OutputModality][]
- [vllm_omni.engine.output_modality.TensorAccumulationStrategy][]
- [vllm_omni.engine.output_processor.MultimodalOutputProcessor][]
- [vllm_omni.engine.output_processor.OmniRequestState][]
- [vllm_omni.engine.stage_engine_core_client.StageEngineCoreClient][]
- [vllm_omni.engine.stage_init_utils.StageMetadata][]
- [vllm_omni.engine.stage_init_utils.StartedLlmStage][]

## Core

Core scheduling and caching components.

- [vllm_omni.core.sched.omni_ar_scheduler.KVCacheTransferData][]
- [vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler][]
- [vllm_omni.core.sched.omni_generation_scheduler.OmniGenerationScheduler][]
- [vllm_omni.core.sched.output.OmniCachedRequestData][]
- [vllm_omni.core.sched.output.OmniNewRequestData][]
- [vllm_omni.core.sched.output.OmniSchedulerOutput][]
- [vllm_omni.model_executor.models.cosyvoice3.code2wav_core.cfm.BASECFM][]
- [vllm_omni.model_executor.models.cosyvoice3.code2wav_core.cfm.CausalConditionalCFM][]
- [vllm_omni.model_executor.models.cosyvoice3.code2wav_core.cfm.CausalMaskedDiffWithDiT][]
- [vllm_omni.model_executor.models.cosyvoice3.code2wav_core.cfm.ConditionalCFM][]
- [vllm_omni.model_executor.models.cosyvoice3.code2wav_core.hifigan.CausalConv1d][]
- [vllm_omni.model_executor.models.cosyvoice3.code2wav_core.hifigan.CausalConv1dUpsample][]
- [vllm_omni.model_executor.models.cosyvoice3.code2wav_core.hifigan.CausalConvRNNF0Predictor][]
- [vllm_omni.model_executor.models.cosyvoice3.code2wav_core.hifigan.CausalHiFTGenerator][]
- [vllm_omni.model_executor.models.cosyvoice3.code2wav_core.hifigan.HiFTGenerator][]
- [vllm_omni.model_executor.models.cosyvoice3.code2wav_core.hifigan.SineGen][]
- [vllm_omni.model_executor.models.cosyvoice3.code2wav_core.hifigan.SineGen2][]
- [vllm_omni.model_executor.models.cosyvoice3.code2wav_core.hifigan.Snake][]
- [vllm_omni.model_executor.models.cosyvoice3.code2wav_core.hifigan.SourceModuleHnNSF][]
- [vllm_omni.model_executor.models.qwen3_tts.tokenizer_25hz.vq.core_vq.DistributedGroupResidualVectorQuantization][]
- [vllm_omni.model_executor.models.qwen3_tts.tokenizer_25hz.vq.core_vq.DistributedResidualVectorQuantization][]
- [vllm_omni.model_executor.models.qwen3_tts.tokenizer_25hz.vq.core_vq.EuclideanCodebook][]
- [vllm_omni.model_executor.models.qwen3_tts.tokenizer_25hz.vq.core_vq.VectorQuantization][]
- [vllm_omni.model_executor.models.qwen3_tts.tokenizer_25hz.vq.core_vq.preprocess][]

## Configuration

Configuration classes.

- [vllm_omni.config.model.OmniModelConfig][]
- [vllm_omni.config.stage_config.ModelPipeline][]
- [vllm_omni.config.stage_config.StageConfig][]
- [vllm_omni.config.stage_config.StageConfigFactory][]
- [vllm_omni.config.stage_config.StageType][]
- [vllm_omni.diffusion.cache.teacache.config.TeaCacheConfig][]
- [vllm_omni.distributed.omni_connectors.utils.config.ConnectorSpec][]
- [vllm_omni.distributed.omni_connectors.utils.config.OmniTransferConfig][]
- [vllm_omni.model_executor.models.cosyvoice3.config.CosyVoice3Config][]
- [vllm_omni.model_executor.models.fish_speech.configuration_fish_speech.FishSpeechConfig][]
- [vllm_omni.model_executor.models.fish_speech.configuration_fish_speech.FishSpeechFastARConfig][]
- [vllm_omni.model_executor.models.fish_speech.configuration_fish_speech.FishSpeechSlowARConfig][]
- [vllm_omni.model_executor.models.mimo_audio.config_mimo_audio.MiMoAudioConfig][]
- [vllm_omni.model_executor.models.mimo_audio.config_mimo_audio.MiMoAudioTokenizerConfig][]
- [vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts.Qwen3TTSConfig][]
- [vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts.Qwen3TTSSpeakerEncoderConfig][]
- [vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts.Qwen3TTSTalkerCodePredictorConfig][]
- [vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts.Qwen3TTSTalkerConfig][]
- [vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2.Qwen3TTSTokenizerV2Config][]
- [vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2.Qwen3TTSTokenizerV2DecoderConfig][]
- [vllm_omni.model_executor.models.qwen3_tts.tokenizer_25hz.configuration_qwen3_tts_tokenizer_v1.Qwen3TTSTokenizerV1Config][]
- [vllm_omni.model_executor.models.qwen3_tts.tokenizer_25hz.configuration_qwen3_tts_tokenizer_v1.Qwen3TTSTokenizerV1DecoderBigVGANConfig][]
- [vllm_omni.model_executor.models.qwen3_tts.tokenizer_25hz.configuration_qwen3_tts_tokenizer_v1.Qwen3TTSTokenizerV1DecoderConfig][]
- [vllm_omni.model_executor.models.qwen3_tts.tokenizer_25hz.configuration_qwen3_tts_tokenizer_v1.Qwen3TTSTokenizerV1DecoderDiTConfig][]
- [vllm_omni.model_executor.models.qwen3_tts.tokenizer_25hz.configuration_qwen3_tts_tokenizer_v1.Qwen3TTSTokenizerV1EncoderConfig][]
- [vllm_omni.transformers_utils.configs.mammoth_moda2.Mammothmoda2Config][]
- [vllm_omni.transformers_utils.configs.mammoth_moda2.Mammothmoda2Qwen2_5_VLConfig][]
- [vllm_omni.transformers_utils.configs.mammoth_moda2.Mammothmoda2Qwen2_5_VLTextConfig][]
- [vllm_omni.transformers_utils.configs.mammoth_moda2.Mammothmoda2Qwen2_5_VLVisionConfig][]

## Workers

Worker classes and model runners for distributed inference.

- [vllm_omni.diffusion.worker.diffusion_model_runner.DiffusionModelRunner][]
- [vllm_omni.diffusion.worker.diffusion_worker.CustomPipelineWorkerExtension][]
- [vllm_omni.diffusion.worker.diffusion_worker.DiffusionWorker][]
- [vllm_omni.diffusion.worker.diffusion_worker.WorkerProc][]
- [vllm_omni.diffusion.worker.diffusion_worker.WorkerWrapperBase][]
- [vllm_omni.diffusion.worker.utils.DiffusionRequestState][]
- [vllm_omni.diffusion.worker.utils.RunnerOutput][]
- [vllm_omni.platforms.npu.worker.npu_ar_model_runner.ExecuteModelState][]
- [vllm_omni.platforms.npu.worker.npu_ar_model_runner.NPUARModelRunner][]
- [vllm_omni.platforms.npu.worker.npu_ar_worker.NPUARWorker][]
- [vllm_omni.platforms.npu.worker.npu_generation_model_runner.NPUGenerationModelRunner][]
- [vllm_omni.platforms.npu.worker.npu_generation_worker.NPUGenerationWorker][]
- [vllm_omni.platforms.npu.worker.npu_model_runner.OmniNPUModelRunner][]
- [vllm_omni.platforms.xpu.worker.xpu_ar_model_runner.XPUARModelRunner][]
- [vllm_omni.platforms.xpu.worker.xpu_ar_worker.XPUARWorker][]
- [vllm_omni.platforms.xpu.worker.xpu_generation_model_runner.XPUGenerationModelRunner][]
- [vllm_omni.platforms.xpu.worker.xpu_generation_worker.XPUGenerationWorker][]
- [vllm_omni.worker.base.OmniGPUWorkerBase][]
- [vllm_omni.worker.gpu_ar_model_runner.ExecuteModelState][]
- [vllm_omni.worker.gpu_ar_model_runner.GPUARModelRunner][]
- [vllm_omni.worker.gpu_ar_worker.GPUARWorker][]
- [vllm_omni.worker.gpu_generation_model_runner.GPUGenerationModelRunner][]
- [vllm_omni.worker.gpu_generation_worker.GPUGenerationWorker][]
- [vllm_omni.worker.gpu_memory_utils.parse_cuda_visible_devices][]
- [vllm_omni.worker.gpu_model_runner.CUDAGraphWrapper][]
- [vllm_omni.worker.gpu_model_runner.OmniGPUModelRunner][]
- [vllm_omni.worker.mixins.OmniWorkerMixin][]
