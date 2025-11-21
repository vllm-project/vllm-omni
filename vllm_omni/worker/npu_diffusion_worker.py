from vllm_ascend.worker.worker_v1 import NPUWorker
from vllm_omni.worker.npu_diffusion_model_runner import NPUDiffusionModelRunner


class NPUDiffusionWorker(NPUWorker):
    """NPU diffusion worker for code2wav stage in Qwen2.5-Omni."""

    def init_device(self):
        device = self._init_device()
        
        self.model_runner: NPUDiffusionModelRunner = NPUDiffusionModelRunner(
            self.vllm_config, device
        )
