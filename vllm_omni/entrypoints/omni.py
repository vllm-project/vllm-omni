"""
Omni serve command for vLLM-omni.
"""

import argparse
import asyncio
from typing import List, Optional
from .omni_llm import AsyncOmniLLM
from vllm_omni.config import create_ar_stage_config, create_dit_stage_config, DiTConfig


class OmniServeCommand:
    """Command handler for vLLM-omni serve command."""
    
    def __init__(self):
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser for omni serve command."""
        parser = argparse.ArgumentParser(
            description="vLLM-omni: Multi-modality models inference and serving"
        )
        
        # Model arguments - make it optional with default
        parser.add_argument(
            "model",
            nargs="?",
            default="Qwen/Qwen3-0.6B",
            help="Path to the model or model name (default: Qwen/Qwen3-0.6B)"
        )
        
        # Server arguments
        parser.add_argument(
            "--port",
            type=int,
            default=8000,
            help="Port to run the server on"
        )
        
        parser.add_argument(
            "--host",
            type=str,
            default="0.0.0.0",
            help="Host to run the server on"
        )
        
        # Stage configuration arguments
        parser.add_argument(
            "--ar-stage",
            type=str,
            help="AR stage model path"
        )
        
        parser.add_argument(
            "--dit-stage",
            type=str,
            help="DiT stage model path"
        )
        
        parser.add_argument(
            "--dit-steps",
            type=int,
            default=50,
            help="Number of DiT inference steps"
        )
        
        parser.add_argument(
            "--dit-guidance-scale",
            type=float,
            default=7.5,
            help="DiT guidance scale"
        )
        
        parser.add_argument(
            "--use-diffusers",
            action="store_true",
            help="Use diffusers pipeline for DiT stage"
        )
        
        # Other arguments
        parser.add_argument(
            "--log-stats",
            action="store_true",
            help="Enable logging statistics"
        )
        
        return parser
    
    def run(self, args: List[str]) -> None:
        """Run the omni serve command."""
        parsed_args = self.parser.parse_args(args)
        
        # Create stage configurations
        stage_configs = self._create_stage_configs(parsed_args)
        
        # Create AsyncOmniLLM instance
        omni_llm = AsyncOmniLLM(
            stage_configs=stage_configs,
            log_stats=parsed_args.log_stats
        )
        
        # Start the server
        asyncio.run(self._start_server(omni_llm, parsed_args))
    
    def _create_stage_configs(self, args) -> List:
        """Create stage configurations based on arguments."""
        stage_configs = []
        stage_id = 0
        
        # Add AR stage - use main model if ar_stage not specified
        ar_model = args.ar_stage if args.ar_stage else args.model
        ar_config = create_ar_stage_config(
            stage_id=stage_id,
            model_path=ar_model,
            input_modalities=["text"],
            output_modalities=["text"]
        )
        stage_configs.append(ar_config)
        stage_id += 1
        
        # Add DiT stage if specified
        if args.dit_stage:
            dit_model = args.dit_stage
        elif args.use_diffusers:
            # Use a default DiT model if diffusers is enabled
            dit_model = "stabilityai/stable-diffusion-2-1"
        else:
            dit_model = None
            
        if dit_model:
            dit_config = DiTConfig(
                model_type="dit",
                scheduler_type="ddpm",
                num_inference_steps=args.dit_steps,
                guidance_scale=args.dit_guidance_scale,
                use_diffusers=args.use_diffusers
            )
            
            dit_stage_config = create_dit_stage_config(
                stage_id=stage_id,
                model_path=dit_model,
                input_modalities=["text"],
                output_modalities=["image"],
                dit_config=dit_config
            )
            stage_configs.append(dit_stage_config)
            stage_id += 1
        
        # If no specific stages are specified, use the main model
        if not stage_configs:
            # Try to detect if it's a multimodal model
            if "omni" in args.model.lower() or "multimodal" in args.model.lower():
                # Assume it's a multimodal model that can handle both AR and DiT
                ar_config = create_ar_stage_config(
                    stage_id=0,
                    model_path=args.model,
                    input_modalities=["text"],
                    output_modalities=["text"]
                )
                stage_configs.append(ar_config)
                
                dit_config = DiTConfig(
                    model_type="dit",
                    scheduler_type="ddpm",
                    num_inference_steps=args.dit_steps,
                    guidance_scale=args.dit_guidance_scale,
                    use_diffusers=args.use_diffusers
                )
                
                dit_stage_config = create_dit_stage_config(
                    stage_id=1,
                    model_path=args.model,
                    input_modalities=["text"],
                    output_modalities=["image"],
                    dit_config=dit_config
                )
                stage_configs.append(dit_stage_config)
            else:
                # Default to AR stage
                ar_config = create_ar_stage_config(
                    stage_id=0,
                    model_path=args.model,
                    input_modalities=["text"],
                    output_modalities=["text"]
                )
                stage_configs.append(ar_config)
        
        return stage_configs
    
    async def _start_server(self, omni_llm: AsyncOmniLLM, args) -> None:
        """Start the API server."""
        try:
            # Import here to avoid circular imports
            from vllm_omni.entrypoints.api_server import run_server
            
            await run_server(
                omni_llm_instance=omni_llm,
                host=args.host,
                port=args.port
            )
        except KeyboardInterrupt:
            print("\nShutting down server...")
        except Exception as e:
            print(f"Error starting server: {e}")
            raise
