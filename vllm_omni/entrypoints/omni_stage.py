"""
Stage manager for orchestrating multiple engines in vLLM-omni.
"""

import importlib
from typing import List, Union

from vllm.inputs import TextPrompt
from vllm.v1.engine import EngineCoreOutput
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.llm_engine import LLMEngine
from vllm_omni.inputs.data import OmniTokensPrompt


class OmniStage:
    def __init__(self, stage_config):
        self.stage_config = stage_config
        self.engine = None
        self.async_engine = None
        self.stage_id = stage_config.stage_id
        self.engine_args = stage_config.engine_args
        self.model_stage = stage_config.engine_args.model_stage
        if hasattr(stage_config, "engine_input_source"):
            self.engine_input_source = stage_config.engine_input_source
        else:
            self.engine_input_source = []
        self.engine_output_type = stage_config.engine_args.engine_output_type
        self.engine_outputs = None
        if hasattr(stage_config, "custom_process_input_func"):
            # Import the module specified in the config (already a full module path)
            module_path, func_name = stage_config.custom_process_input_func.rsplit(
                ".", 1
            )
            module = importlib.import_module(module_path)
            self.custom_process_input_func = getattr(module, func_name)
        else:
            self.custom_process_input_func = None

        if hasattr(stage_config, "final_output"):
            self.final_output = stage_config.final_output
        else:
            self.final_output = False

        if hasattr(stage_config, "final_output_type"):
            self.final_output_type = stage_config.final_output_type
        else:
            self.final_output_type = None

    def set_engine(self, engine: LLMEngine) -> None:
        """Initialize the engine for the stage."""
        self.engine = engine

    def set_async_engine(self, async_engine: AsyncLLM) -> None:
        """Initialize the async engine for the stage."""
        self.async_engine = async_engine

    def set_engine_outputs(self, engine_outputs: EngineCoreOutput) -> None:
        """Set the engine output for the stage."""
        self.engine_outputs = engine_outputs

    def process_engine_inputs(
        self, stage_list, prompt: Union[OmniTokensPrompt, TextPrompt] = None
    ) -> List[Union[OmniTokensPrompt, TextPrompt]]:
        """Process the engine input for the stage."""
        if self.custom_process_input_func is None:
            engine_inputs = []
            if len(self.engine_input_source) == 0:
                raise ValueError("engine_input_source is empty")
            source_stage_id = self.engine_input_source[0]
            source_outputs = stage_list[source_stage_id].engine_outputs
            multi_modal_data = {
                source_output.request_id: prompt.get("multi_modal_data", None)
                for source_output, prompt in zip(source_outputs, prompt)
            }

            for source_output in source_outputs:
                engine_input = OmniTokensPrompt(
                    prompt_token_ids=source_output.outputs[0].token_ids,
                    multi_modal_data=(
                        multi_modal_data[source_output.request_id]
                        if multi_modal_data
                        else None
                    ),
                )
                engine_inputs.append(engine_input)
            return engine_inputs

        else:
            engine_input_source = self.engine_input_source
            return self.custom_process_input_func(
                stage_list, engine_input_source, prompt
            )
