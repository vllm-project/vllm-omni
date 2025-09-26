"""
API server for vLLM-omni.
"""

import asyncio
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from ..core.omni_llm import AsyncOmniLLM


class GenerateRequest(BaseModel):
    """Request model for generation."""
    prompts: List[str]
    max_tokens: int = 100
    temperature: float = 0.7
    stage_args: List[Dict[str, Any]] = None


class GenerateResponse(BaseModel):
    """Response model for generation."""
    outputs: List[Dict[str, Any]]
    stage_outputs: List[Dict[str, Any]] = None


app = FastAPI(
    title="vLLM-omni API",
    description="Multi-modality models inference and serving",
    version="0.1.0"
)

# Global omni_llm instance
omni_llm: AsyncOmniLLM = None


@app.on_event("startup")
async def startup_event():
    """Initialize the omni_llm instance on startup."""
    global omni_llm
    # This will be set by the run_server function
    pass


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text or multimodal content."""
    try:
        if omni_llm is None:
            raise HTTPException(status_code=500, detail="OmniLLM not initialized")
        
        # Prepare stage arguments
        if request.stage_args is None:
            # Create default stage arguments - one per stage config
            # For now, we'll process all prompts in the first stage
            stage_args = [{
                "prompt": " ".join(request.prompts) if request.prompts else "",
                "max_tokens": request.max_tokens,
                "temperature": request.temperature
            }]
        else:
            stage_args = request.stage_args
        
        # Generate using omni_llm
        try:
            outputs = await omni_llm.generate_async(stage_args)
        except Exception as e:
            # Fallback to synchronous generation
            outputs = omni_llm.generate(stage_args)
        
        # Convert outputs to response format
        response_outputs = []
        for output in outputs:
            if hasattr(output, 'outputs') and output.outputs:
                for out in output.outputs:
                    response_outputs.append({
                        "text": getattr(out, 'text', ''),
                        "finished": getattr(out, 'finish_reason', 'length') != 'length',
                        "tokens": getattr(out, 'token_ids', [])
                    })
            else:
                response_outputs.append({
                    "text": "",
                    "finished": True,
                    "tokens": []
                })
        
        return GenerateResponse(
            outputs=response_outputs,
            stage_outputs=[{"stage": i, "output": "processed"} for i in range(len(stage_args))]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "vllm-omni"}


@app.get("/info")
async def get_info():
    """Get information about the service."""
    if omni_llm is None:
        return {"error": "OmniLLM not initialized"}
    
    return {
        "service": "vllm-omni",
        "version": "0.1.0",
        "num_stages": omni_llm.get_num_stages() if hasattr(omni_llm, 'get_num_stages') else 0,
        "stage_configs": [
            {
                "stage_id": config.stage_id,
                "engine_type": config.engine_type,
                "model_path": config.model_path,
                "input_modalities": config.input_modalities,
                "output_modalities": config.output_modalities
            }
            for config in omni_llm.stage_configs
        ]
    }


async def run_server(omni_llm_instance: AsyncOmniLLM, host: str = "0.0.0.0", port: int = 8000):
    """Run the API server."""
    global omni_llm
    omni_llm = omni_llm_instance
    
    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level="info"
    )
    
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    # This is for testing purposes
    asyncio.run(run_server(None))
