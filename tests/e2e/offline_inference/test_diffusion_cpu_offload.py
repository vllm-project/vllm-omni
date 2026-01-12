import sys
import threading
import time
from pathlib import Path

import pytest
import torch

from vllm_omni.utils.platform_utils import is_npu, is_rocm

# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_omni import Omni


class GPUMemoryMonitor:
    """Poll global device memory usage via CUDA APIs."""

    def __init__(self, device_index: int, interval: float = 0.05):
        self.device_index = device_index
        self.interval = interval
        self.peak_used_mb = 0.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        def monitor_loop() -> None:
            while not self._stop_event.is_set():
                try:
                    with torch.cuda.device(self.device_index):
                        free_bytes, total_bytes = torch.cuda.mem_get_info()
                    used_mb = (total_bytes - free_bytes) / (1024**2)
                    self.peak_used_mb = max(self.peak_used_mb, used_mb)
                except Exception:
                    pass
                time.sleep(self.interval)

        self._thread = threading.Thread(target=monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=2.0)


models = ["riverclouds/qwen_image_random"]


@pytest.mark.skipif(is_npu() or is_rocm(), reason="Hardware not supported")
@pytest.mark.parametrize("model_name", models)
def test_cpu_offload_diffusion_model(model_name: str):
    def inference(offload: bool = True):
        torch.cuda.empty_cache()
        device_index = torch.cuda.current_device()
        monitor = GPUMemoryMonitor(device_index=device_index, interval=0.02)
        monitor.start()
        m = Omni(model=model_name, enable_cpu_offload=offload)
        torch.cuda.reset_peak_memory_stats(device=device_index)
        height = 256
        width = 256

        m.generate(
            "a photo of a cat sitting on a laptop keyboard",
            height=height,
            width=width,
            num_inference_steps=9,
            guidance_scale=0.0,
            generator=torch.Generator("cuda").manual_seed(42),
        )

        monitor.stop()
        torch.cuda.synchronize(device_index)
        fallback_alloc = torch.cuda.max_memory_allocated(device=device_index) / (1024**2)
        fallback_reserved = torch.cuda.max_memory_reserved(device=device_index) / (1024**2)
        peak_memory_mb = max(monitor.peak_used_mb, fallback_alloc, fallback_reserved)

        return peak_memory_mb

    offload_peak_memory = inference(offload=True)
    no_offload_peak_memory = inference(offload=False)
    print(f"Offload peak memory: {offload_peak_memory} MB")
    print(f"No offload peak memory: {no_offload_peak_memory} MB")
    assert offload_peak_memory + 2500 < no_offload_peak_memory, (
        f"Offload peak memory {offload_peak_memory} MB should be less than no offload peak memory {no_offload_peak_memory} MB"
    )
