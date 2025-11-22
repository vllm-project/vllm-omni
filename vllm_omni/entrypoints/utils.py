import os
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf

from vllm.transformers_utils.config import get_config
from vllm_omni.utils import detect_device_type

# Get the project root directory (2 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_stage_configs_from_model(model: str):
    """Load stage configs from model."""
    hf_config = get_config(model, trust_remote_code=True)
    model_type = hf_config.model_type
    stage_config_file = f"vllm_omni/model_executor/stage_configs/{model_type}.yaml"
    stage_config_path = PROJECT_ROOT / stage_config_file
    if not os.path.exists(stage_config_path):
        raise FileNotFoundError(f"Stage config file {stage_config_path} not found")
    stage_configs = load_stage_configs_from_yaml(config_path=str(stage_config_path))
    return stage_configs


def load_stage_configs_from_yaml(config_path: str):
    """Load stage configs from yaml file."""
    config_data = OmegaConf.load(config_path)
    return config_data.stage_args


def select_worker_class(worker_cls: Optional[str], device_type: Optional[str] = None) -> Optional[str]:
    """Select appropriate worker class based on device type."""
    if worker_cls is None:
        return None

    if device_type is None:
        device_type = detect_device_type()

    if device_type == "npu":
        # Replace module path: gpu_ar_worker -> npu_ar_worker
        if "gpu_ar_worker" in worker_cls:
            worker_cls = worker_cls.replace("gpu_ar_worker", "npu_ar_worker")
        elif "gpu_diffusion_worker" in worker_cls:
            worker_cls = worker_cls.replace("gpu_diffusion_worker", "npu_diffusion_worker")

        # Replace class name: GPUARWorker -> NPUARWorker, GPUDiffusionWorker -> NPUDiffusionWorker
        if "GPUARWorker" in worker_cls:
            worker_cls = worker_cls.replace("GPUARWorker", "NPUARWorker")
        elif "GPUDiffusionWorker" in worker_cls:
            worker_cls = worker_cls.replace("GPUDiffusionWorker", "NPUDiffusionWorker")

    return worker_cls
