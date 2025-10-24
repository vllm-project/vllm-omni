from vllm.engine.arg_utils import EngineArgs
from typing import Literal, Optional
from dataclasses import dataclass
from vllm_omni.config import OmniModelConfig
from vllm.utils import FlexibleArgumentParser


@dataclass
class OmniEngineArgs(EngineArgs):
    stage_id: int = 0
    model_stage: str = "thinker"
    model_arch: str = "Qwen2_5OmniForConditionalGeneration"
    engine_output_type: Optional[str] = None

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        """Shared CLI arguments for vLLM engine."""
        parser.add_argument(
            "--engine-output-type",
            type=str,
            default=OmniEngineArgs.engine_output_type,
            help=(
                "Declare EngineCoreOutput.output_type (e.g., 'text', 'image', "
                "'text+image', 'latent'). This will be written into "
                "model_config.engine_output_type for schedulers to use."
            ),
        )
        parser.add_argument("--model-stage", type=str, default=EngineArgs.model_stage, 
        help="Declare model stage (e.g., 'thinker', 'talker', 'token2wav'). This will be written into model_config.model_stage for schedulers to use.")
        return parser

    def create_model_config(self) -> OmniModelConfig:
        # First, get the base ModelConfig from the parent class
        base_config = super().create_model_config()
        
        # Create OmniModelConfig by copying all base config attributes
        # and adding the new omni-specific fields
        config_dict = base_config.__dict__.copy()
        
        # Add the new omni-specific fields
        config_dict['stage_id'] = self.stage_id
        config_dict['model_stage'] = self.model_stage
        config_dict['model_arch'] = self.model_arch
        config_dict['engine_output_type'] = self.engine_output_type
        
        # Create and return the OmniModelConfig instance
        omni_config = OmniModelConfig(**config_dict)
        omni_config.hf_config.architectures = omni_config.architectures
        
        return omni_config