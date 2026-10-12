# Archived module design pages

This directory preserves module-design pages that predate the structure
proposed in [issue #5137](https://github.com/vllm-project/vllm-omni/issues/5137).
They are historical snapshots, not active architectural contracts, and are
intentionally excluded from the documentation navigation.

| Archived page | Active destination |
| --- | --- |
| [AR Module](ar_module.md) | [Autoregressive Runtime](../ar_runtime.md) |
| [AsyncOmni Architecture](async_omni_architecture.md) | [Engine Orchestration](../engine_orchestration.md) and [Stage Runtime](../stage_runtime.md) |
| [DiT Module](dit_module.md) | [Diffusion module overview](../diffusion/index.md) |
| [Entrypoint Module](entrypoint_module.md) | [Entrypoints and Serving Boundaries](../entrypoints.md) |

Do not update these pages with current design guidance. Move still-useful
material into the active destination and review it against current code and
tests instead. No compatibility redirect pages are provided at the former
paths.
