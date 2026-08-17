# Model-Level Offloading

Model-level, or sequential, offloading keeps only the pipeline component group
currently executing on the accelerator. It is the simplest offload strategy
and is selected with `--enable-cpu-offload`.

## How it works

Pre-forward hooks enforce mutual exclusion between DiT and encoder modules:

- before an encoder runs, the DiT moves to CPU;
- before a DiT runs, encoders and other DiTs move to CPU; and
- VAE modules remain on the accelerator.

Pinned host memory reduces transfer overhead. Transfers occur at phase
boundaries, so cold-start and encoder-to-denoiser transitions become slower.

## Usage

```python
from vllm_omni import Omni

omni = Omni(
    model="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    enable_cpu_offload=True,
)
```

```bash
vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --omni --enable-cpu-offload
```

## Model integration

Pipelines should implement `SupportsComponentDiscovery`:

```python
from typing import ClassVar

from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery


class MyPipeline(nn.Module, SupportsComponentDiscovery):
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder", "vision_model"]
    _vae_modules: ClassVar[list[str]] = ["vae"]
    _resident_modules: ClassVar[list[str]] = []
```

All entries may be dotted paths. DiT and encoder lists are both required for
mutual exclusion. VAE modules are pinned but not swapped; resident modules are
small modules that must stay on the accelerator for layerwise paths.

## Split-model components

Some models, such as Cosmos3, split one transformer into mutually exclusive
components that run in different phases. The pipeline exposes
`enable_omni_model_cpu_offload`, and the backend delegates to the model-local
contexts:

```python
class Cosmos3VFMTransformer(nn.Module):
    def forward(self, ...):
        with self._offload_context("reasoner"):
            ...
        with self._offload_context("generator"):
            ...
```

This preserves the same invariant—exactly one component is device resident—
while reusing sequential `.to()` movers.

## Limitations

- Single device only.
- Higher cold-start latency.
- Transfers between encoder and denoising phases add latency.

See the [model-level design](../../../design/feature/offloader/cpu_offload.md)
for lifecycle and extension invariants.
