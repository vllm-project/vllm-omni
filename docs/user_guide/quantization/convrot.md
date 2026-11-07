# ConvRot

ConvRot ([arXiv:2512.03673](https://arxiv.org/abs/2512.03673)) is a
training-free, rotation-based quantization method for diffusion transformers.
vLLM-Omni implements it as two online methods that quantize a BF16/FP16
checkpoint while loading:

| Method | Weights | Activations | GEMM |
| --- | --- | --- | --- |
| `convrot_int8` | INT8, per output channel | INT8, dynamic per token | vLLM INT8 kernels |
| `convrot_int4` | INT4, per output channel, packed two per byte | INT4, dynamic per token | INT8 tensor cores on the INT4 values |

Before quantization, the input dimension of every linear layer is split into
groups of `rotation_group_size` channels, and each group of both the
activation and the weight is multiplied by the same normalized regular Hadamard
matrix (Kronecker powers of the paper's 4x4 regular Hadamard). The matrix is
orthogonal, so the layer output is unchanged, while channel outliers -- which
DiT activations have in large numbers -- are spread across their group before
the per-token and per-channel scales are taken. Weights are rotated once at
load time; activations are rotated with one small reshape-matmul per forward.

`convrot_int4` stores INT4 weights (a quarter of BF16 storage) and computes the
W4A4 product exactly on INT8 tensor cores, which also run on GPUs without INT4
matrix instructions such as Hopper. Its `int8_layers` option runs selected
layers on rotated INT8 instead, the paper's mixed W4A4 + INT8 setting for
sensitive layers.

## Usage

```bash
vllm serve <model> --omni --quantization convrot_int8
vllm serve <model> --omni --quantization convrot_int4
```

```python
from vllm_omni import Omni

omni = Omni(model="<model>", quantization="convrot_int8")
omni = Omni(
    model="<model>",
    quantization_config={
        "method": "convrot_int4",
        "rotation_group_size": 256,
        "int8_layers": ["attn.to_out"],
        "ignored_layers": ["proj_out"],
    },
)
```

| Option | Default | Meaning |
| --- | --- | --- |
| `rotation_group_size` | `256` | Power of four. A layer whose input it does not tile uses the largest smaller power of four that does, or no rotation. The paper uses 1024 for layers with larger outliers. |
| `ignored_layers` | `[]` | Layers kept in BF16/FP16. A name matches every module path containing it as whole components: `attn.to_out` selects it in every block, `token_refiner` every layer below it. |
| `int8_layers` | `[]` | `convrot_int4` only: layers run on rotated INT8, matched like `ignored_layers`. |

## Validation

MiniMax-H3 Ref2VA, 1344x768, 124 frames, 50 steps, seed 42, two H200 GPUs
(Ulysses SP 2, text encoder offloaded), compared with BF16 on the same request.
PSNR and spectral distance are measured against the BF16 output; H3 sampling is
chaotic, so a clean output whose motion diverges also scores a low PSNR.

| Method | Inference | Peak memory per GPU | Video PSNR (mean / last frame) | Audio spectral distance | Result |
| --- | --- | --- | --- | --- | --- |
| BF16 | 167-178 s | 75.5 GiB | -- | -- | Reference |
| `int8` (no rotation) | 149 s | 43.9 GiB | 12.6 / 22.3 dB | 0.134 | Clean |
| `convrot_int8` | 152 s | 44.1 GiB | 16.6 / 30.8 dB | 0.055 | Clean, closest to BF16 |
| `convrot_int4`, every layer W4A4 | 177 s | 32.8 GiB | 12.7 / 12.8 dB | 2.227 | Smeared video, clipped noise audio |
| `convrot_int4`, mixed (below) | 160 s | 38.8 GiB | 11.9 / 11.5 dB | 0.170 | Clean |

On synthetic layers with DiT-like outlier channels, rotation cuts the INT8
output error from 10.2% to 1.25% and the W4A4 error from 77% to 22%.

For MiniMax-H3, quantize only the attention QKV and first MLP projections to
W4A4, keep the modulation, attention output and second MLP projections on
rotated INT8, and keep the entry and exit projections in BF16:

```bash
vllm serve MiniMaxAI/MiniMax-H3 --omni --diffusion-quantization-config '{
  "method": "convrot_int4",
  "int8_layers": ["adaln_proj", "attn.out_proj", "mlp.fc2"],
  "ignored_layers": ["video_patch_proj", "audio_patch_proj", "condition_proj",
                     "time_embedder", "token_refiner", "final_layer"]}'
```

Hopper has no INT4 matrix instructions, so W4A4 runs on INT8 tensor cores
after unpacking; it saves memory but is not faster than `convrot_int8` there.

## Limitations

- CUDA only (compute capability 8.0+). Weights are quantized online; ConvRot
  checkpoints are not serialized.
- Rotation, activation quantization and, for `convrot_int4`, weight unpacking
  run as separate operations around the GEMM; they are not fused.
- INT4 activations have a coarse per-token resolution. Compare against BF16 on
  your prompts, and move layers that degrade quality to `int8_layers` or
  `ignored_layers`.
