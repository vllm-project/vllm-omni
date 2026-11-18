# Sol-Attn

`SOL_ATTN` is an opt-in, training-free sparse attention backend. By default, H3
uses it throughout denoising while keeping text/audio blocks exact. Selecting it does
not change the checkpoint, sampling steps, precision, or platform defaults.

Sol-Attn is by **Haopeng Li et al.** ([paper](https://arxiv.org/abs/2607.24027),
[upstream implementation](https://github.com/NVlabs/Sana/tree/f4035e0a976d828ed41c7b0b799e9712b47cd60d/techniques/sparse_backends)).
This backend integrates their algorithm; it does not require VSA-trained weights.

## Requirements

Install the validated upstream kernel package and the cuDNN BSA dependency:

```bash
pip install 'git+https://github.com/NVlabs/Sana.git@f4035e0a976d828ed41c7b0b799e9712b47cd60d#subdirectory=techniques/sparse_backends'
pip install 'nvidia-cudnn-frontend>=1.31.0'
```

The default cuDNN backend requires SM90 or newer and BF16 self-attention with
head dimension 128 and equal Q/KV heads. It supports one sequence at a time,
locally or with pure Ulysses; Ring, AllGather-KV, and quantized KV are unsupported.

## Quick start

Start T2VA or FL2VA with native checkpoints:

```bash
vllm serve /path/to/MiniMax-H3/FL2VA --omni \
  --task-type fl2va \
  --enforce-eager --vae-use-tiling \
  --diffusion-attention-backend SOL_ATTN
```

No JSON or model-specific attention settings are required.

- **T2VA / FL2VA:** set request `extra_params.task` to `t2va` or `fl2va`.
- **Ref2VA:** replace `/FL2VA` with `/Ref2VA` and `--task-type fl2va` with
  `--task-type ref2va`; set request `extra_params.task` to `ref2va`.

For memory requirements, parallelism, and HTTP requests, see the
[MiniMax-H3 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md).

## Advanced settings

To override defaults, replace the quick-start backend flag with:

```bash
--diffusion-attention-config '{"default":{"backend":"SOL_ATTN","sol_attn":{"thresh_type":"exact","tau":1.5}}}'
```

| Key | Default | Meaning |
| --- | --- | --- |
| `tau` | `1.0` | Controls sparsity. Higher values generally compute fewer blocks exactly; check output quality. |
| `thresh_type` | `diag` | Threshold estimator: `diag` or `exact`. Neither changes precision or makes attention dense. |

## Validation limits

Full-model video/audio quality, HTTP serving, and speedup for the default
profile remain unvalidated. More sparsity does not guarantee a faster request;
compare against a matching dense baseline before deployment. Full-model
compilation and HSDP compatibility are not guaranteed.
