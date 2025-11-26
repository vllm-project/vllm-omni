from vllm_ascend.worker.worker_v1 import NPUWorker
from vllm_omni.worker.npu_ar_model_runner import NPUARModelRunner


class NPUARWorker(NPUWorker):
    """NPU AR worker for thinker/talker stages in Qwen2.5-Omni."""

    def init_device(self):
        device = self._init_device()

        self.model_runner: NPUARModelRunner = NPUARModelRunner(
            self.vllm_config, device
        )

