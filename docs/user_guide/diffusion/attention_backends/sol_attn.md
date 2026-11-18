# Sol-Attn

`SOL_ATTN` is an opt-in, training-free sparse attention backend. Selecting it
uses **cuDNN SOL-BSA, block64 text/audio protection, `diag`, `tau=1`, and no
dense step/layer guards**. It does not change the platform's default backend.

The algorithm combines exact selected blocks with an approximation of omitted
blocks, merged using their softmax normalization. It is not VSA and does not
require VSA-trained weights.

## Installation and scope

Install the validated upstream kernel package and the cuDNN BSA dependency:

```bash
pip install 'git+https://github.com/NVlabs/Sana.git@f4035e0a976d828ed41c7b0b799e9712b47cd60d#subdirectory=techniques/sparse_backends'
pip install 'nvidia-cudnn-frontend>=1.31.0'
```

The default cuDNN provider requires SM90 or newer, BF16 Q/K/V, head dimension
128, and noncausal self-attention with equal Q/KV heads. Supported execution is
one real sequence, locally or with pure Ulysses. Structural suffix padding is
excluded; multiple packed documents, arbitrary masks, Ring, AllGather-KV,
quantized KV, and compact unequal-Q/KV attention are rejected.

## MiniMax-H3 online serving

Start T2VA or FL2VA with native checkpoints:

```bash
vllm serve /path/to/MiniMax-H3/FL2VA --omni \
  --task-type fl2va \
  --enforce-eager --vae-use-tiling \
  --diffusion-attention-config '{"per_role":{"minimax_h3.main":{"backend":"SOL_ATTN"},"minimax_h3.token_refiner":{"backend":"TORCH_SDPA"}}}'
```

No `sol_attn` option block is required. The main DiT uses the defaults above;
the text token refiner stays dense.

- **T2VA / FL2VA:** set request `extra_params.task` to `t2va` or `fl2va`.
- **Ref2VA:** replace `/FL2VA` with `/Ref2VA` and `--task-type fl2va` with
  `--task-type ref2va`; set request `extra_params.task` to `ref2va`.
- Set request `num_inference_steps=50` for a 50-step run. Attention selection
  does not change the sampler, step count, checkpoint, or precision.
- GPU selection and parallelism remain deployment choices. Without parallelism
  options this uses one GPU; provide sufficient model memory in either case.

Submit requests through the
[MiniMax-H3 online video API](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md#http-api-examples).

## Protection policy

The default adopts the
[upstream Sol-H3 Ref2VA policy](https://github.com/NVlabs/Sana/blob/cf67a3acad7511acb85053e56a05e5dd24c82ba0/models/minimax_h3/Sol-H3/h3_runtime/engine.py#L302)
for T2VA, FL2VA, and Ref2VA. Upstream uses different guards for its other tasks;
this integration deliberately uses one text/audio policy across H3 tasks.
It does not import the upstream LoRA, quantized transport, or reduced-step profile.

H3 builds one block64 mask per request branch from actual packed text/audio
positions, including reference and target audio. Marked blocks stay exact on
both Q and KV axes. Visual reference blocks remain eligible for sparse routing,
except mixed boundary blocks containing text/audio. Neighboring and
route-selected blocks also stay exact; disabling dense guards does not make
every block approximate. Pure Ulysses retains the global mask coordinates.

Missing or stale text/audio metadata fails explicitly. Other models must
publish compatible metadata or explicitly select `sink_mode="prefix"`; there
is no silent fallback to prefix protection or dense attention.

## Controls

Optional overrides go in `AttentionSpec.sol_attn` under the selected role:

| Key | Default | Meaning |
| --- | --- | --- |
| `provider` | `cudnn` | BF16 SOL-BSA; `triton` selects the other BSA provider, `native` the upstream fused kernel. |
| `sink_mode` | `text_audio` | Model-owned text/audio block64 protection; `prefix` selects the legacy prefix/range policy. |
| `tau` | `1.0` | Nonnegative threshold multiplier; higher values generally select fewer exact blocks. |
| `thresh_type` | `diag` | `diag` ignores cross-dimension covariance; `exact` includes it. Neither changes precision or means dense attention. |
| `dense_steps` | `0` | First N denoising steps stay dense; positive values also guard missing step metadata. |
| `dense_layers` | `[]` | Layer indices/ranges kept dense; nonempty selectors also guard unidentified layers. |
| `prefix_sink` | `true` | In `prefix` mode, derive the protected prefix from model-owned video layout. Ignored by `text_audio`. |
| `sink_tokens` | `0` | Additional protected range in `prefix` mode; BSA rounds outward to whole blocks. |
| `sink_start` | `0` | Range offset in `prefix` mode; `null` places the range at the valid-token suffix. |
| `kv_splits` | `auto` | BSA resolves to one; native uses four on SM90 with CuTe and at least 65,536 valid tokens, otherwise one. |

`text_audio` requires a BSA provider and rejects explicit sink ranges. The
`triton` provider requires `vllm-omni[vsa]` for its shared kernel, SM80+, and
standard head-dimension scaling; it still executes SOL, not the VSA algorithm.

To use the native fused kernel, explicitly set
`"sol_attn":{"provider":"native","sink_mode":"prefix"}`. It supports SM80+
and the upstream architecture-specific runtimes. Prefix mode conservatively
protects everything before the target video, including Ref2VA visual references;
native execution recomputes protected query rows densely. Set dense guards
explicitly if desired. Models with cross-attention must select SOL only for
compatible self-attention roles.

## Validation limits

The SOL-BSA implementation currently uses a chunked PyTorch residual, not a
fused residual kernel; routing metadata grows quadratically with block count.
Kernel and attention-forward checks do not establish full-model video/audio
quality, HTTP-serving validation, or a speedup for this default profile.
Validate against a matching dense baseline before deployment. Full-model
compilation and outer HSDP boundaries are outside the backend's guarantee.
