import os
import tempfile
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForMultimodalLM, AutoProcessor

from tests.helpers.tiny_model import build_tiny_from_configs

TINY_CONFIGS_DIR = Path(__file__).parent / "tiny_configs"
TINY_MODEL_DIR = os.path.join(tempfile.gettempdir(), "vllm-omni-tiny-models")


### Single stage diffusion models
def tiny_flux2_klein_builder() -> str:
    """Build a tiny Flux2Klein model from vendored configs."""
    return build_tiny_from_configs(
        "Flux2KleinPipeline", "black-forest-labs/FLUX.2-klein-4B", TINY_CONFIGS_DIR / "Flux2KleinPipeline"
    )


def tiny_ltx2_builder() -> str:
    """Build a tiny LTX2 model from vendored configs."""
    return build_tiny_from_configs("LTX2Pipeline", "Lightricks/LTX-2", TINY_CONFIGS_DIR / "LTX2Pipeline")


### Omni models / multi-stage Diffusion Models
def tiny_qwen3_omni_builder() -> str:
    """Build a tiny Qwen3Omni model (all 3 stages) & return saved path."""
    config = AutoConfig.from_pretrained(TINY_CONFIGS_DIR / "qwen3_omni")
    model = AutoModelForMultimodalLM.from_config(config).to(torch.bfloat16)
    proc = AutoProcessor.from_pretrained(TINY_CONFIGS_DIR / "qwen3_omni")
    outdir = os.path.join(TINY_MODEL_DIR, "qwen3_omni")
    model.save_pretrained(outdir)
    proc.save_pretrained(outdir)
    return outdir
