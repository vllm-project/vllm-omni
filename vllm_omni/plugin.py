"""
vLLM plugin system for vLLM-omni integration.
"""

from typing import Dict, Any, Optional
from vllm_omni.core.omni_llm import OmniLLM, AsyncOmniLLM
from vllm_omni.config import create_ar_stage_config, create_dit_stage_config


class OmniPlugin:
    """vLLM plugin for vLLM-omni integration."""
    
    def __init__(self):
        self.name = "omni"
        self.version = "0.1.0"
        self.description = "Multi-modality models inference and serving"
    
    def register_components(self) -> Dict[str, Any]:
        """Register vLLM-omni components with vLLM."""
        return {
            "omni_llm": OmniLLM,
            "async_omni_llm": AsyncOmniLLM,
            "create_ar_stage_config": create_ar_stage_config,
            "create_dit_stage_config": create_dit_stage_config,
        }
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Get configuration schema for the plugin."""
        return {
            "type": "object",
            "properties": {
                "stages": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "stage_id": {"type": "integer"},
                            "engine_type": {"type": "string", "enum": ["AR", "DiT"]},
                            "model_path": {"type": "string"},
                            "input_modalities": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "output_modalities": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["stage_id", "engine_type", "model_path", "input_modalities", "output_modalities"]
                    }
                }
            },
            "required": ["stages"]
        }
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate plugin configuration."""
        if "stages" not in config:
            return False
        
        stages = config["stages"]
        if not isinstance(stages, list):
            return False
        
        for stage in stages:
            required_fields = ["stage_id", "engine_type", "model_path", "input_modalities", "output_modalities"]
            for field in required_fields:
                if field not in stage:
                    return False
            
            if stage["engine_type"] not in ["AR", "DiT"]:
                return False
        
        return True
    
    def create_omni_llm(self, config: Dict[str, Any]) -> OmniLLM:
        """Create an OmniLLM instance from configuration."""
        from vllm_omni.config import OmniStageConfig, DiTConfig
        
        stage_configs = []
        for stage_config in config["stages"]:
            if stage_config["engine_type"] == "AR":
                stage_config_obj = create_ar_stage_config(
                    stage_id=stage_config["stage_id"],
                    model_path=stage_config["model_path"],
                    input_modalities=stage_config["input_modalities"],
                    output_modalities=stage_config["output_modalities"]
                )
            elif stage_config["engine_type"] == "DiT":
                dit_config = DiTConfig(
                    model_type="dit",
                    scheduler_type="ddpm",
                    num_inference_steps=50,
                    guidance_scale=7.5
                )
                
                stage_config_obj = create_dit_stage_config(
                    stage_id=stage_config["stage_id"],
                    model_path=stage_config["model_path"],
                    input_modalities=stage_config["input_modalities"],
                    output_modalities=stage_config["output_modalities"],
                    dit_config=dit_config
                )
            else:
                raise ValueError(f"Unknown engine type: {stage_config['engine_type']}")
            
            stage_configs.append(stage_config_obj)
        
        return OmniLLM(stage_configs)
    
    def create_async_omni_llm(self, config: Dict[str, Any]) -> AsyncOmniLLM:
        """Create an AsyncOmniLLM instance from configuration."""
        from vllm_omni.config import OmniStageConfig, DiTConfig
        
        stage_configs = []
        for stage_config in config["stages"]:
            if stage_config["engine_type"] == "AR":
                stage_config_obj = create_ar_stage_config(
                    stage_id=stage_config["stage_id"],
                    model_path=stage_config["model_path"],
                    input_modalities=stage_config["input_modalities"],
                    output_modalities=stage_config["output_modalities"]
                )
            elif stage_config["engine_type"] == "DiT":
                dit_config = DiTConfig(
                    model_type="dit",
                    scheduler_type="ddpm",
                    num_inference_steps=50,
                    guidance_scale=7.5
                )
                
                stage_config_obj = create_dit_stage_config(
                    stage_id=stage_config["stage_id"],
                    model_path=stage_config["model_path"],
                    input_modalities=stage_config["input_modalities"],
                    output_modalities=stage_config["output_modalities"],
                    dit_config=dit_config
                )
            else:
                raise ValueError(f"Unknown engine type: {stage_config['engine_type']}")
            
            stage_configs.append(stage_config_obj)
        
        return AsyncOmniLLM(stage_configs)
    
    def get_help_text(self) -> str:
        """Get help text for the plugin."""
        return """
vLLM-omni Plugin

This plugin enables multi-modality models inference and serving with non-autoregressive structures.

Usage:
    vllm serve model --omni [options]

Options:
    --ar-stage MODEL_PATH     AR stage model path
    --dit-stage MODEL_PATH    DiT stage model path
    --dit-steps N             Number of DiT inference steps (default: 50)
    --dit-guidance-scale F    DiT guidance scale (default: 7.5)
    --use-diffusers          Use diffusers pipeline for DiT stage

Examples:
    vllm serve Qwen/Qwen2.5-Omni-7B --omni
    vllm serve model --omni --ar-stage text-model --dit-stage image-model
        """
