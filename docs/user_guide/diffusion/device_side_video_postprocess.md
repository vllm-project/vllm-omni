# Device-side video postprocessing

vLLM-Omni can convert decoded WAN2.2 video from normalized floating-point
`[B, C, T, H, W]` tensors to contiguous `uint8` `[B, T, H, W, C]` frames before
the worker-to-engine device-to-host transfer. This reduces the transported video
payload by four times when the original IPC path widens bfloat16 to float32.

The optimization is disabled by default. Enable it with a structured diffusion
configuration:

```python
from vllm_omni import Omni

engine = Omni(
    model="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    video_output_transport={"enable_device_postprocess": True},
)
```

For the server CLI, pass the same object as JSON:

```bash
vllm serve Wan-AI/Wan2.2-TI2V-5B-Diffusers --omni \
  --video-output-transport '{"enable_device_postprocess": true}'
```

For a deploy configuration, place it on the diffusion stage:

```yaml
stages:
  - stage_id: 0
    video_output_transport:
      enable_device_postprocess: true
```

## Runtime contract

WAN2.2 emits decoded video through the typed `DiffusionOutput.media` field. The
model runner validates and splits the batch into request-local tensors before
preparing each video for transport. The worker then performs the D2H copy and
shared-memory IPC using the prepared representation. The engine uses the generic
media finalizer instead of a model-specific postprocessor.

The runtime leaves the video in normalized floating-point form when:

- `enable_device_postprocess` is false;
- the requested output type is not `np`; or
- frame interpolation still requires the floating-point tensor.

These are policy fallbacks, not malformed-contract errors. If the device cannot
allocate the temporary conversion buffer, the runner logs a warning and prepares
the request-local float representation instead. Invalid tensor layout, encoding,
value range, lifecycle state, and non-memory runtime failures still fail before
IPC.

## Precision

Device preparation converts to float32 before denormalization and quantization.
For bfloat16 WAN output this is more precise than the historical host path,
which denormalizes in bfloat16. Relative to that path, values may differ by at
most one level out of 255. Float32 inputs remain byte-identical after
quantization.

## Benchmark

Use the checked-in benchmark to compare an unmodified main checkout with the
candidate checkout. It separates cold startup from warm requests, requires at
least three measured rounds, reports process-tree host RSS and peak allocated
GPU memory, and compares the quantized output frames.

```bash
python benchmarks/diffusion/bench_device_postprocess.py \
  --baseline-checkout /path/to/vllm-omni-main \
  --candidate-checkout . \
  --warmup-runs 1 \
  --rounds 3
```

The two checkouts must use the same model cache, Python environment, visible GPU,
prompt, seed, shape, and sampling parameters. MP4 hashes are not compared because
multi-threaded video encoding is not byte-deterministic.

## Scope

WAN2.2 is the reference migration for the first contract version. Other video
pipelines continue using their existing output paths until they can declare the
same typed media contract without bypassing model-specific float consumers such
as safety checks or audio/video packaging.

See [RFC #6541](https://github.com/vllm-project/vllm-omni/issues/6541) for the
contract, lifecycle, batching rules, and migration plan.
