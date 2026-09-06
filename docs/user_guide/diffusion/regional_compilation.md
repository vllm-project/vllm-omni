# Regional Compilation

Regional compilation applies `torch.compile` to the repeated transformer blocks
declared by a diffusion model. It is the default compilation scope when
diffusion inference runs without `--enforce-eager`.

## Configuration

Dynamic compilation is enabled by default so the compiled regions can handle
mixed resolutions. For a fixed-shape workload, disable it explicitly:

```bash
vllm serve <model> --omni --no-diffusion-compile-dynamic
```

The equivalent per-stage deploy configuration is:

```yaml
stages:
  - stage_id: 0
    diffusion_compile_granularity: regional
    diffusion_compile_dynamic: false
```

For an experimental whole-transformer compile scope, set
`--diffusion-compile-granularity full` or use
`diffusion_compile_granularity: full` in the deploy configuration. Full scope may
still contain graph breaks; it does not force one graph. It is rejected when
HSDP, sequence parallelism, CPU offload, or layerwise offload is enabled. Use
regional scope with those features.

## Compute/Communication Overlap

Inductor can reorder communication and compute within compiled regions to
overlap collective operations with independent computation. This optimization
is disabled by default. For default single-stage diffusion serving, enable it
with:

```bash
vllm serve <model> --omni \
  --diffusion-compile-reorder-comm-overlap
```

This setting applies to both regional and full generic compilation. It has no
effect with `--enforce-eager` or pipelines that provide their own
`setup_compile()` implementation. Models whose compiled graphs do not contain
communication operations may see no benefit.

These settings control the generic model-runner compilation path. Pipelines
that provide their own `setup_compile()` implementation manage their compilation
policy independently. Compilation is lazy, so backend or graph errors can first
surface on the initial request.

Use `--enforce-eager` to disable the model runner's generic compile setup.
Pipelines that compile internally define their own eager-mode behavior.
