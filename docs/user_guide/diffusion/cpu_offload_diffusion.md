# CPU Offloading for Diffusion Model

## Overview
CPU offload lets the diffusion worker move large model components between GPU and CPU memory on demand. It keeps the DiT transformer resident on GPU only while it is actively running, and swaps it out when encoders modules need the device. This reduces peak VRAM usage so bigger checkpoints run on smaller GPUs, or multiple requests can share the same GPU.

## Execution Model
1. Text encoders run on GPU while the DiT transformer is offloaded to CPU.
2. Before denoising, weights are prefetched back to GPU, honoring pinned-memory copies for speed.
3. After the diffusion step, the transformer returns to CPU and the process repeats as needed.

Transfers use pinned host buffers, and the worker coordinates swaps via mutex-style hooks so components never compete for memory.

## Configuration
You can enable CPU offload in two ways:

- **Python API**: set `enable_cpu_offload=True`.

```python
from vllm_omni import Omni

if __name__ == "__main__":

    m = Omni(model="Qwen/Qwen-Image",enable_cpu_offload=True)
```

- **CLI**: pass `--enable-cpu-offload` to the diffusion service entrypoint.

## Known Limitations
- Cold start latency increases for over one minute for some models(e.g., Qwen-Image)
