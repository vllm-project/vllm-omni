from typing import Union, Sequence, Optional, Any
import cloudpickle
from pydantic import ValidationError

from vllm.inputs import PromptType
from vllm.sampling_params import SamplingParams
from vllm.entrypoints.llm import LLM

from vllm.v1.engine.llm_engine import LLMEngine
from vllm.engine.arg_utils import HfOverrides
from vllm.usage.usage_lib import UsageContext
from vllm.config import CompilationConfig, is_init_field
from vllm.utils import Counter
from vllm.logger import init_logger
import vllm.envs as envs

from vllm_omni.entrypoints.utils import load_stage_configs_from_model
from vllm_omni.entrypoints.stage import OmniStage
from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.engine.output_processor import MultimodalOutputProcessor
from vllm_omni.engine.processor import OmniProcessor
from vllm_omni.outputs import OmniRequestOutput


logger = init_logger(__name__)


class OmniLLM:
    def __init__(self, model: str, stage_configs = None, log_stats: bool = False, **kwargs):
        if stage_configs is None:
            self.initialize_stage_configs(model)
        else:
            self.stage_configs = stage_configs
        
        self.stage_list = []
        self.initialize_stages(model)
        
    def initialize_stage_configs(self, model: str):
        self.stage_configs = load_stage_configs_from_model(model)
    
    def initialize_stages(self, model: str):
        for stage_config in self.stage_configs:
            stage = OmniStage(stage_config)
            stage_llm = OmniStageLLM(model=model, **stage_config.engine_args)
            stage.set_engine(stage_llm)
            self.stage_list.append(stage)
    
    def generate(
        self,
        prompts: Union[PromptType, Sequence[PromptType]],
        sampling_params_list: Optional[Union[SamplingParams,
                                        Sequence[SamplingParams]]] = None,
    ) -> list[OmniRequestOutput]:
        """Generate text outputs for the given prompts."""
        final_outputs: list[OmniRequestOutput] = []
        if len(sampling_params_list) != len(self.stage_list):
            raise ValueError(f"Expected {len(self.stage_list)} sampling params, got {len(sampling_params_list)}")
        for stage_id, stage in enumerate(self.stage_list):
            if stage_id > 0:
                engine_inputs = stage.process_engine_inputs(self.stage_list, prompts)
            else:
                engine_inputs = prompts
            engine_outputs = self._run_generation(stage, sampling_params_list[stage_id], engine_inputs)
            stage.set_engine_outputs(engine_outputs)
            if hasattr(stage, 'final_output') and stage.final_output:
                final_outputs.append(OmniRequestOutput(
                    stage_id=stage_id, 
                    final_output_type=stage.final_output_type, 
                    request_output=engine_outputs))
        return final_outputs
    
    def _run_generation(self, stage: OmniStage, sampling_params: SamplingParams, prompts: Union[PromptType, Sequence[PromptType]]):
        engine_outputs = []
        for ro in stage.engine.generate(prompts, sampling_params):
            engine_outputs.append(ro)
        return engine_outputs


class OmniStageLLM(LLM): 
    def __init__(self, 
                model: str, 
                compilation_config: Optional[Union[int, dict[str, Any],
                                           CompilationConfig]] = None,
                hf_overrides: Optional[HfOverrides] = None,
                **kwargs):
        """LLM constructor."""
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True

        if "worker_cls" in kwargs:
            worker_cls = kwargs["worker_cls"]
            # if the worker_cls is not qualified string name,
            # we serialize it using cloudpickle to avoid pickling issues
            if isinstance(worker_cls, type):
                kwargs["worker_cls"] = cloudpickle.dumps(worker_cls)

        if "kv_transfer_config" in kwargs and isinstance(
                kwargs["kv_transfer_config"], dict):
            from vllm.config import KVTransferConfig
            raw_config_dict = kwargs["kv_transfer_config"]
            try:
                kwargs["kv_transfer_config"] = KVTransferConfig(
                    **raw_config_dict)
            except ValidationError as e:
                logger.error(
                    "Failed to convert 'kv_transfer_config' dict to "
                    "KVTransferConfig object. Dict: %s. Error: %s",
                    raw_config_dict, e)
                # Consider re-raising a more specific vLLM error or ValueError
                # to provide better context to the user.
                raise ValueError(
                    f"Invalid 'kv_transfer_config' provided: {e}") from e

        if hf_overrides is None:
            hf_overrides = {}

        if compilation_config is not None:
            if isinstance(compilation_config, int):
                compilation_config_instance = CompilationConfig(
                    level=compilation_config)
            elif isinstance(compilation_config, dict):
                predicate = lambda x: is_init_field(CompilationConfig, x[0])
                compilation_config_instance = CompilationConfig(
                    **dict(filter(predicate, compilation_config.items())))
            else:
                compilation_config_instance = compilation_config
        else:
            compilation_config_instance = CompilationConfig()

        engine_args = OmniEngineArgs(
            model=model,
            hf_overrides=hf_overrides,
            compilation_config=compilation_config_instance,
            **kwargs,
        )

        # Create the Engine (autoselects V0 vs V1)
        self.llm_engine = LLMEngine.from_engine_args(
            engine_args=engine_args, usage_context=UsageContext.LLM_CLASS)
        self.llm_engine.output_processor = MultimodalOutputProcessor(tokenizer=self.llm_engine.tokenizer, 
                                                                    log_stats=self.llm_engine.log_stats)
        self.llm_engine.processor = OmniProcessor(vllm_config=self.llm_engine.vllm_config,
                                                  tokenizer=self.llm_engine.tokenizer)
        self.engine_class = type(self.llm_engine)

        self.request_counter = Counter()
        self.default_sampling_params: Union[dict[str, Any], None] = None

        if envs.VLLM_USE_V1:
            supported_tasks = self.llm_engine \
                .get_supported_tasks()  # type: ignore
        else:
            supported_tasks = self.llm_engine.model_config.supported_tasks

        logger.info("Supported_tasks: %s", supported_tasks)

        self.supported_tasks = supported_tasks