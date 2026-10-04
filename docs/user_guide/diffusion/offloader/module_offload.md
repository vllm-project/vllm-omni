# Model-Level Offloading

Model-level, or sequential, offloading keeps only the pipeline component group
currently executing on the accelerator. It is the simplest offload strategy
and is selected with `diffusion_offload_config.mode="module"`.

## How it works

Pre-forward hooks enforce mutual exclusion between DiT and encoder modules:

- before an encoder runs, selected DiTs move to CPU;
- before a DiT runs, selected encoders and other selected DiTs move to CPU; and
- VAE modules remain on the accelerator unless the pipeline owns their
  lifecycle and selects the `vae` component.

Pinned host memory reduces transfer overhead. Transfers occur at phase
boundaries, so cold-start and encoder-to-denoiser transitions become slower.

## Usage

```python
from vllm_omni import Omni

omni = Omni(
    model="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    diffusion_offload_config={
        "mode": "module",
        "components": ["dit", "text_encoder"],
    },
)
```

```bash
vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --omni \
  --diffusion-offload-config \
  '{"mode":"module","components":["dit","text_encoder"]}'
```

List a component to select it for offload. Module mode rejects `layer_options`
such as `weight_transfer` and `resident_layers`. The
`enable_cpu_offload=True` compatibility entry point remains supported. New
integrations should prefer the explicit config; existing model-specific stage
lifecycles do not need to migrate until equivalent component coverage exists.
Pipelines that own a VAE lifecycle (for example a non-`forward` `decode` entry
point) may additionally select the `vae` component so the VAE leaves the
accelerator during denoising; it is only valid with `mode="module"`. Pipelines
without such a lifecycle keep their VAE resident and reject `vae`. For example,
MiniMax-H3's compatibility lifecycle stages its VAEs but does not expose them
through the compact selector.

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

- Transfers are rank-local; module mode does not shard host payloads or add a
  weight AllGather across ranks.
- Higher cold-start latency.
- Transfers between encoder and denoising phases add latency.

See the [model-level design](../../../design/feature/offloader/module_offload.md)
for lifecycle and extension invariants.
