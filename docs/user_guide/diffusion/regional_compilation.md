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

These settings control the generic model-runner compilation path. Pipelines
that provide their own `setup_compile()` implementation manage their compilation
policy independently. Compilation is lazy, so backend or graph errors can first
surface on the initial request.

## Pinning one packed shape (MiniMax-H3)

Compiled regions are keyed by input shape. MiniMax-H3 packs a request into a
row count that follows the request -- a longer prompt, more frames, a larger
frame -- and rounds it up to the next 64-row boundary, so two requests can land
on two different packed lengths.

Under the default `diffusion_compile_dynamic: true` a new packed length is
absorbed by the dynamic shape, so it is not by itself a recompilation. It still
hands the backend a shape it has not autotuned, which is the risk this knob
exists to remove: a fixed-shape deployment (`--no-diffusion-compile-dynamic`)
recompiles the transformer blocks per shape, and a first-seen large shape can
push autotuning into its worst-case memory use.

A request pins the length with `extra_args["pad_seq_len"]`:

```python
sampling_params = SamplingParams(extra_args={"pad_seq_len": 54080})
```

The value must be a positive multiple of 64 and must cover the rows the request
actually uses; the packed sequence is then padded to it instead of to the next
64-row boundary. Pick a bucket that covers every request dimension the
deployment accepts -- prompt length, frame count, frame size and reference
blocks all change the used row count. The server logs the effective length as
`MiniMax H3 packed sequence: ... pad_seq_len=... used=... seq_len=...` whenever
a request pins it. The padding rows are masked out, so their cost is the
attention and feed-forward work on those rows.

Use `--enforce-eager` to disable the model runner's generic compile setup.
Pipelines that compile internally define their own eager-mode behavior.
