# Quantization

This document describes the quantization architecture in vLLM-Omni and the
extension points for contributors. For user-facing configuration, supported
hardware, and method-specific instructions, see the
[quantization user guide](../../user_guide/quantization/overview.md).

## Goals and scope

The design has four goals:

1. Provide one `quantization_config` entry point for diffusion-only,
   multi-stage omni/TTS, and multi-stage diffusion models.
2. Reuse vLLM's quantization registry and `QuantizationConfig` contract where
   a vLLM backend already provides the required behavior.
3. Allow different components of one pipeline to use different quantization
   configurations.
4. Preserve the distinction between online quantization of a BF16/FP16
   checkpoint and loading a pre-quantized checkpoint with serialized scales.

This document covers configuration resolution, component routing, model
integration, backend extension, and validation. Kernel implementations and
user-facing support matrices are outside its scope.

## Architecture

The quantization path is split across the following layers:

| Layer | Responsibility | Source |
| --- | --- | --- |
| Public API | Exposes `build_quant_config`, component routing, and backend registration | `vllm_omni/quantization/__init__.py` |
| Factory | Normalizes method names, selects Omni overrides or vLLM registry backends, detects ModelOpt configs, and reconciles checkpoint metadata | `vllm_omni/quantization/factory.py` |
| Component router | Resolves a quantization config from a layer prefix using longest-prefix matching | `vllm_omni/quantization/component_config.py` |
| Method configs | Implement `QuantizationConfig` and select the platform-specific linear method | `vllm_omni/quantization/*_config.py` |
| Diffusion configuration | Stores the canonical runtime config and propagates checkpoint configuration | `vllm_omni/diffusion/data.py` |
| Worker and model integration | Passes the resolved config into vLLM's model configuration and model layers | `vllm_omni/diffusion/worker/diffusion_worker.py` and `vllm_omni/diffusion/models/` |

The runtime contract remains vLLM's `QuantizationConfig`:

```text
user/CLI or model config
          |
          v
OmniDiffusionConfig.quantization_config
          |
          +--> build_quant_config()
          |        |
          |        +--> Omni override
          |        +--> vLLM registry backend
          |        +--> ComponentQuantizationConfig
          |
          +--> checkpoint metadata reconciliation
          |
          +--> diffusion worker / vLLM config
          |
          +--> model layers call get_quant_method(layer, prefix)
```

The model layer receives either a method implementation or `None`. Returning
`None` leaves that layer unquantized.

## Configuration resolution

`build_quant_config()` accepts the following forms:

| Input | Resolution |
| --- | --- |
| `None` or `"none"` | Disable quantization. |
| Method string, such as `"fp8"` | Resolve an Omni override first, then the vLLM quantization registry. |
| Flat dictionary with `method` or `quant_method` | Normalize the method and construct its `QuantizationConfig`. |
| Per-component dictionary | Construct a `ComponentQuantizationConfig`. |
| Existing `QuantizationConfig` | Pass through without rebuilding it. |

Method aliases are normalized case-insensitively, with `-` and `_` treated as
equivalent. `auto-round` and `auto_round` use the Omni INC/AutoRound adapter.
ModelOpt checkpoint metadata is detected from its method, producer metadata, or
`quant_algo`, then mapped to the corresponding vLLM ModelOpt configuration.

The factory uses Omni-specific overrides for methods whose diffusion behavior
or platform dispatch is not provided by the vLLM registry. The current override
set includes INT8, BitsAndBytes, MXFP8, MXFP4, MXFP4 dual-scale, and INC/
AutoRound. Other methods, such as FP8, GGUF, and ModelOpt, are resolved through
vLLM's registry when their configuration is compatible.

## Per-component routing

Multi-stage models do not necessarily want one quantization policy for every
module. A component map expresses that policy without requiring each pipeline
to construct separate vLLM configurations:

```python
quantization_config = {
    "transformer": {"method": "fp8"},
    "vae": None,
    "default": None,
}
```

`ComponentQuantizationConfig` applies the following rules:

1. Component keys are interpreted as runtime layer-prefixes.
2. The longest matching prefix wins.
3. `None` disables quantization for the matching component.
4. `default`, when present, handles prefixes with no explicit match.
5. A matching child config delegates to its normal
   `get_quant_method(layer, prefix)` implementation.

Prefix matching happens after the model's weight-name mapping. New model
integrations must verify the actual runtime prefixes, especially when a
`WeightsMapper` changes checkpoint names. A prefix mismatch can silently route
a layer to the default configuration, so component-routing tests should cover
both positive matches and fall-through behavior.

## Online and pre-quantized checkpoints

The same method name can represent two different loading paths:

- **Online**: the source checkpoint contains BF16/FP16 weights; the method
  creates quantized weights or scales while the model is loaded.
- **Pre-quantized**: the checkpoint contains quantized weights and the metadata
  required by the method; the loader consumes those serialized tensors.

Method configuration owns the mode flag, for example
`is_checkpoint_int8_serialized` or `is_checkpoint_mxfp8_serialized`. The
configuration's `get_quant_method()` selects the corresponding linear method.

For pipelines with transformer-specific `config.json` files, such as cascade
models, `resolve_quant_config_from_disk()` reconciles the active configuration
with each transformer's checkpoint metadata. Its invariants are:

- If no active configuration exists, valid checkpoint metadata can be used for
  auto-detection.
- A method mismatch is an error rather than an implicit conversion.
- Serialized-checkpoint flags rebuild an online configuration into the
  matching offline configuration.
- Checkpoint-specific `ignored_layers` are preserved by rebuilding the active
  configuration when they differ.
- AutoRound MXFP8 metadata (`data_type="mx_fp"`) selects its offline path.

This prevents a model from interpreting serialized weights with an online
loader or silently applying the wrong method to one transformer in a
multi-transformer pipeline.

## Model and layer integration

Diffusion model constructors receive the resolved configuration and pass it to
quantizable vLLM layers. Each method configuration is responsible for:

- identifying supported layers in `get_quant_method()`;
- creating the correct parameter shapes and loaders;
- processing weights after loading when conversion is required; and
- applying the platform-specific quantized operation at runtime.

Model code must keep precision-sensitive or unsupported modules unquantized.
The shared `safe_quant_config()` helper is used by model integrations for
normalization and modulation layers where a generic FP8 configuration would be
unsafe. Text encoders, audio/vision encoders, VAEs, schedulers, and tokenizers
are not implicitly quantized just because the main transformer is quantized;
they require explicit component routing and method support.

Weight quantization and quantized KV cache are separate paths. The former is
resolved through `quantization_config`; runtime attention/KV-cache quantization
is configured by the diffusion attention configuration and must be validated
independently.

## Platform and parallelism boundaries

Platform checks belong in the method implementation rather than in the common
factory. This keeps configuration parsing portable while allowing CUDA, NPU,
and XPU methods to select different kernels or reject unsupported execution at
model construction time. The user guide is the source of truth for the current
hardware support matrix.

Quantization methods must also respect the weight layout produced by the
configured parallelism strategy:

- Tensor-parallel layers may receive partitioned weight shapes and loaders.
- Sequence-parallel execution changes activation movement but does not by
  itself change the quantization configuration.
- Data-parallel replicas build their own method state; quantization metadata
  must be available consistently on every rank.
- `WeightsMapper` transformations must be applied before matching ignored-layer
  lists or component prefixes.

Distributed layerwise offload has an additional boundary: its sharded mmap
AllGather path is not compatible with online quantization because the path
expects already materialized checkpoint weights. Use the rank-local
`--dlo-no-use-allgather` path or a compatible pre-quantized checkpoint when
combining these features. See the [DLO design document](offloader/distributed_layerwise_offload.md)
for the full compatibility matrix.

## Adding a quantization backend

Contributors should follow this sequence:

1. Check whether an existing vLLM backend can provide the required weight
   format and kernel behavior. Prefer reuse over a duplicate Omni backend.
2. If Omni-specific behavior is required, implement a `QuantizationConfig` and
   its `QuantizeMethodBase`/linear method in `vllm_omni/quantization/`.
3. Implement `get_name()`, supported activation dtypes, minimum capability,
   checkpoint parsing, ignored-layer handling, and `get_quant_method()`.
4. Add a lazy factory override in `factory.py`. Keep optional platform
   dependencies out of package import paths and normalize aliases to one
   canonical method name.
5. If the method is pre-quantized, define the checkpoint metadata and the
   serialized loading path before adding an online path.
6. Add model-specific component routing only when the model contains stages
   that must use different policies.
7. Document user-visible configuration in the quantization user guide and keep
   implementation rationale here.

The public factory registration hook,
`register_quantization_override(method, builder)`, is intended for integrations
that need to register an Omni-compatible builder without changing the vLLM
registry.

## Validation

Validation should cover the complete path, not only whether a config object can
be instantiated:

| Area | Required checks |
| --- | --- |
| Factory | Strings, aliases, flat dictionaries, invalid inputs, passthrough, and component maps |
| Method config | Online/offline selection, ignored layers, platform guards, and weight loading |
| Checkpoint metadata | Auto-detection, method mismatch errors, serialized flags, and per-transformer metadata |
| Model integration | Correct quantization scope, unquantized components, and parallel weight layouts |
| Numerical quality | Same-seed BF16 versus quantized output comparison at representative resolutions and step counts |
| Performance | Peak device memory, generation latency, and any backend-specific throughput metric |

The unit tests live under `tests/diffusion/quantization/`. The trajectory
similarity tool in
`vllm_omni/quantization/tools/compare_diffusion_trajectory_similarity.py`
provides a repeatable quality and memory comparison for diffusion models.
Method support and feature combinations remain model- and hardware-dependent;
passing factory tests alone does not establish production support.

## Design invariants

- `quantization_config` is the single runtime source of truth after config
  normalization.
- A checkpoint method mismatch fails loudly instead of risking weight
  corruption.
- Components without an explicit supported quantization path remain in their
  original precision.
- Optional backend dependencies are imported lazily.
- Method implementations own platform-specific capability checks and kernels.
- User-facing support claims require model, hardware, numerical, and memory
  validation in addition to configuration-level tests.
