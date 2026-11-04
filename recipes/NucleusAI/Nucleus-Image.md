# Nucleus-Image for text-to-image generation

## Summary

- Vendor: NucleusAI
- Model: `NucleusAI/Nucleus-Image`
- Task: Text-to-image generation
- Mode: Online serving with the OpenAI-compatible API
- Maintainer: Community

## When to use this recipe

Use this recipe when you want a known-good starting point for serving
`NucleusAI/Nucleus-Image` on `2x H20` and comparing the main parallel
execution modes currently supported by vLLM-Omni for this model.

This model currently supports:

- **TP (Tensor Parallelism)**
- **SP (Ulysses / Ring)**
- **Layerwise CPU offload**

Step execution uses the same default guidance scale (`4.0`) as full forward
and decodes at the aligned dimensions prepared for the request. Ulysses and
Ring use global expert-choice routing; padding does not participate in
expert selection.

Ring also accepts image token counts that are not divisible by the SP degree.
For strict padding exclusion, enable `mask_sp_padding=True` in
`DiffusionParallelConfig`. Image states and RoPE are padded together; Ring
uses the original image K/V length, and gathered outputs are cropped back to
the original token count. The padding-exclusion option defaults to `False`.

## References

- Upstream model card: <https://huggingface.co/NucleusAI/Nucleus-Image>
- Related online serving test: `tests/e2e/online_serving/test_nucleus_image_expansion.py`
- Related online example: `examples/online_serving/nucleus_image/`
- Related offline test: `tests/e2e/offline_inference/test_nucleus_image_expansion.py`
- Related offline example: `examples/offline_inference/nucleus_image/`

## Hardware Support

This recipe documents a tested CUDA configuration for `2x H20`.
AMD, Ascend NPU, and Intel GPU execution have not been validated for this model.

FP8 is optional and partial: attention projections, the router, dense MLPs,
and shared experts are quantizable. Routed SwiGLU expert weights, the text
encoder, and VAE stay in checkpoint precision. See the
[FP8 scope guide](../../docs/user_guide/quantization/fp8.md).

## GPU

### 2 x H20

#### Environment

- Platform: CUDA GPU
- GPU: `2x H20`
- Model: `NucleusAI/Nucleus-Image`
- vLLM-Omni: current working tree

#### Command

##### Baseline

```bash
vllm serve NucleusAI/Nucleus-Image --omni \
  --port 8091
```

##### TP=2

```bash
vllm serve NucleusAI/Nucleus-Image --omni \
  --tensor-parallel-size 2 \
  --port 8091
```

##### USP=2

```bash
vllm serve NucleusAI/Nucleus-Image --omni \
  --ulysses-degree 2 \
  --port 8091
```

##### RING=2

```bash
vllm serve NucleusAI/Nucleus-Image --omni \
  --ring-degree 2 \
  --port 8091
```

##### Optional layerwise offload variant

```bash
vllm serve NucleusAI/Nucleus-Image --omni \
  --tensor-parallel-size 2 \
  --enable-layerwise-offload \
  --port 8091
```

#### Verification

After the server is ready, test with a simple request:

```bash
curl -X POST http://localhost:8091/v1/images/generations   -H "Content-Type: application/json"   -d '{
    "prompt": "A weathered lighthouse on a rocky coastline at golden hour, waves crashing against the rocks below, seagulls circling overhead, dramatic clouds painted in shades of amber and violet",
    "size": "1024x1024",
    "num_inference_steps": 50,
    "guidance_scale": 4.0,
    "seed": 42
  }' | jq -r '.data[0].b64_json' | base64 -d > output.png
```

#### Historical Benchmark Summary

These numbers were reported in the original PR before the global MoE routing
fix. They have not been rerun with the corrected SP path and do not establish
a speedup or output equivalence for the current implementation.

| Metric  | Baseline | TP=2 | USP=2 | RING=2 |
|---------|---------:|-----:|------:|-------:|
| Latency |      21s |  17s |   16s |    15s |

#### Notes

- **Parallel support:** For this model, the currently relevant multi-GPU modes are
  TP (`--tensor-parallel-size`), Ulysses (`--ulysses-degree`), and Ring (`--ring`).
- **Layer offload:** `--enable-layerwise-offload` is supported as an optional
  memory-saving mode when you need extra headroom.
