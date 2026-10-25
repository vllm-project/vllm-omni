# Batch Invariance (Diffusion)

!!! note
    Diffusion batch invariance is experimental. Historical SD3 results do not validate this revision; remeasure with the final code and vLLM versions. Other configurations are **not rejected** — they run, just without that evidence. See [Known Limitations](#known-limitations) for the operator gaps behind that narrow scope.

Batch invariance means a request produces bit-identical output regardless of the batch size it lands in or the position it occupies inside that batch. vLLM-Omni builds on vLLM's batch-invariant kernels; for the autoregressive side see vLLM's own guide, this page covers the diffusion stage.

## Motivation

- **Reinforcement learning**: RL rollouts must be reproducible, or batching variance enters the reward signal and cannot be separated from real policy change.
- **Framework and model debugging**: a bug that only appears at batch size 4 is easier to isolate when batch size is the only variable that changed.
- **Regression testing**: bit-identical output makes tensor-level assertions possible, so numerical drift is caught by CI instead of by eye.

## Hardware Requirements

Evidence covers exactly one NVIDIA CUDA compute capability: **8.9** (SM89, Ada — measured on an RTX 4090). No other capability has been measured.

The worker does not check compute capability. Any other CUDA GPU is **unverified, not rejected**: the switch is honoured, the engine runs, and determinism holds only for the operators vLLM actually replaces. Do not read that as "newer is safer" — vLLM *branches* on capability rather than requiring a floor, so a result in one capability bracket says nothing about another.

One negative result is already known: **on SM120 (compute capability 12.0, RTX 5090) batch invariance does not hold.** That is an observed failure, not a prediction — identical requests diverge across batch sizes with the switch on.

ROCm/HIP builds and non-CUDA devices skip silently: the diffusion bootstrap returns before it calls into vLLM, so no operator replacement happens and no message is emitted. Do not infer determinism from a clean startup there.

The seed contract, however, is not device-aware: it follows the switch alone. On those devices an enabled switch still requires an explicit integer seed and still rejects `generator`, `generator_device`, `latents` and `sigmas`, without the operator replacement that would make those restrictions pay off. If a mixed pipeline needs `VLLM_BATCH_INVARIANT=1` for its AR stage there, set `VLLM_OMNI_DIFFUSION_BATCH_INVARIANT=0` so the diffusion stage is not held to a contract it cannot benefit from. That is also the way out on any device you have not validated.

## Enabling Batch Invariance

Two environment variables control the diffusion stage:

| Variable | Effect |
| --- | --- |
| `VLLM_BATCH_INVARIANT` | vLLM's global switch. The diffusion stage follows it by default. |
| `VLLM_OMNI_DIFFUSION_BATCH_INVARIANT` | Diffusion-only override. Unset follows the global switch; `1`/`true`/`yes`/`on` forces on; `0`/`false`/`no`/`off` forces off. |

Leave the diffusion-only variable unset unless you need the two stages to differ — a mixed AR + diffusion pipeline may want deterministic text generation without subjecting the diffusion stage to the [seed contract](#seed-contract) below. An unparsable value raises `ValueError` listing the accepted values rather than defaulting to off, because a typo that silently disabled determinism would be worse than a crash.

```bash
# Whole pipeline deterministic
export VLLM_BATCH_INVARIANT=1

# AR stage deterministic, diffusion stage left alone
export VLLM_BATCH_INVARIANT=1
export VLLM_OMNI_DIFFUSION_BATCH_INVARIANT=0
```

### Seed contract

In batch-invariant mode every diffusion request must carry an explicit integer seed, and must not pass a `generator` object:

```python
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

sampling_params = OmniDiffusionSamplingParams(
    seed=1234,          # required: an explicit int in torch.Generator.manual_seed range
    height=512,
    width=512,
    num_inference_steps=8,
    guidance_scale=1.0,
)
```

Without a seed, vLLM-Omni assigns a random one so all ranks share an RNG state — so consecutive identical requests would legitimately differ, making determinism look broken while the kernels worked correctly. A `generator` object is rejected because it binds to one device and does not travel across worker processes.

Three further inputs are rejected for the same reason — each takes the RNG identity out of the seed's hands:

| Rejected input | Why |
| --- | --- |
| `generator_device` | selects the device the RNG is drawn on, so the seed no longer fixes the draw |
| `latents` | supplies the initial noise directly, so the seed does not determine it at all |
| `sigmas` | replaces the noise schedule, so the seeded trajectory follows a different path |

These are the *only* request-level rejections in batch-invariant mode. They are not recipe constraints — they are the premise of "same seed ⇒ same output" itself, which is why they stay while the configuration checks do not (see [The table is evidence, not a gate](#the-table-is-evidence-not-a-gate)). External RNG inputs are checked before replica selection and model preprocessing; internal reconstruction and warmup are exempt. OpenPI checks seed provenance before forwarding, accepting configured defaults and validating provided types before coercion.

## Historical PR Measurement Recipe

This historical recipe requires revalidation on the final code and vLLM versions:

| Dimension | Validated value |
| --- | --- |
| Pipeline | `StableDiffusion3Pipeline` |
| Attention backend | `TORCH_SDPA` |
| Resolution | 512 × 512 |
| Inference steps | 8 (the only measured value) |
| dtype | `torch.bfloat16` |

The historical [PR description](https://github.com/vllm-project/vllm-omni/pull/5935) compares eager, single-GPU SD3 at BS1 and BS4 using decoded bf16 image tensors of shape `[1, 3, 512, 512]`, not latents.

Pending validation: repeat the same seeded prompt at real BS1–4 with `output_type="latent"`, including a feature-disabled control. Fix the physical GPU and code/vLLM versions; record sample positions, shapes, dtypes and byte hashes. These measurements have not been run.

### The table is evidence, not a gate

**vLLM-Omni does not reject configurations outside this table.** A different pipeline, resolution, dtype, step count or GPU count runs normally; determinism then holds only for the operators vLLM actually replaces, and anything outside that set (see [Known Limitations](#known-limitations)) carries no guarantee. Entries outside the table are **unverified, not unsupported** — they may well be batch-invariant, nobody has checked. This matches vLLM's own approach upstream, which lists the models it has validated and notes that others may also work. If you validate another configuration, please report it so the table can grow. `num_inference_steps` is the one dimension we can reason about rather than merely leave unmeasured: it is a loop count and changes no tensor shape.

### Multi-GPU is unverified in both directions

Every measurement above is single-GPU, with all 12 parallelism degrees in `DiffusionParallelConfig` at 1. Multi-GPU diffusion batch invariance is therefore **unverified in both directions** — no evidence that it holds, none that it fails. It is not gated, so it runs. That is a structural gap rather than a sampling one: on one GPU the diffusion stage performs no collective communication at all, so the single-GPU runs never exercise the code multi-GPU determinism would rest on.

An omni-specific point needs auditing before trusting that path. vLLM sets `disable_custom_all_reduce = True` whenever `VLLM_BATCH_INVARIANT` is on, in `vllm/config/parallel.py`. That name appears zero times anywhere in `vllm_omni`, and diffusion calls `torch.distributed.all_reduce` directly rather than routing through vLLM's `CustomAllreduce`, so the upstream protection does not obviously carry over. Two upstream issues describe the failure modes to expect: vllm#50136 (custom all-reduce eligibility depends on a size threshold, so kernel selection becomes a function of batch composition) and vllm#30321 (DP + EP inconsistency).

### Why only SD3 has evidence

`StableDiffusion3Pipeline` is the only pipeline a bit-identity run covers, and that follows from operator coverage rather than a preference for SD3. Video pipelines add `Conv3d` and audio pipelines add `Conv1d`; vLLM's batch-invariant layer overrides no convolution operator, so those paths are untouched by it. `group_norm` is a separate, image-side gap — also unoverridden upstream, and SD3 is exempt only because the recipe fixes the shapes its own normalization sees. Uncovered operators require measurements of the actual pipeline.

## Implementation Details

When the diffusion stage runs batch-invariant, vLLM-Omni:

1. resolves the three-state switch during worker startup, after the CUDA device is selected and before distributed initialization, because vLLM's initialization writes NCCL environment variables that a live communicator would ignore;
2. returns silently on ROCm/non-CUDA devices, skipping steps 3 and 4; compute capability is not checked, so every CUDA device proceeds;
3. calls vLLM's `init_batch_invariance()`, which installs deterministic operator implementations, disables split-k and reduced-precision reductions, forces IEEE fp32 precision for matmul and cuDNN convolution, and pins NCCL to deterministic algorithms;
4. aligns `VLLM_BATCH_INVARIANT` for the worker process, because vLLM re-reads that variable itself — without the alignment a diffusion-only opt-in would install nothing.

Expect a throughput cost. Deterministic kernels give up optimizations that reorder floating-point reductions; that trade is the point of the feature.

## Known Limitations

**Coverage and measurements limit the evidence.** vLLM's batch-invariant layer overrides matrix multiplication (`mm`, `addmm`, `matmul`, `linear`, `bmm`), softmax variants and `mean.dim`. It does not override `scaled_dot_product_attention`, `conv2d`, or `group_norm` — exactly what a diffusion pipeline leans on: the `TORCH_SDPA` backend calls `torch.nn.functional.scaled_dot_product_attention` directly, and the VAE decoder is built from convolutions and group normalization. Their batch invariance is a property of the specific shapes the validated recipe produces, not something the kernel layer guarantees. Change the resolution and the convolution shapes change with it, which is why 512 × 512 is listed as measured rather than as a supported range.

An operator absent from the override list is not thereby non-deterministic — it is simply unguaranteed, and each case has to be argued separately rather than assumed either way.

Further operator guarantees and pipeline measurements are needed:

- the [historical recipe](#historical-pr-measurement-recipe) needs revalidation; other configurations run unverified;
- multi-GPU diffusion batch invariance is unverified; the evidence matrix is single-GPU;
- `torch.compile` paths are unverified — the validated run uses `enforce_eager=True`;
- `output_type="latent"` skips VAE decode; `"pt"` includes decode, and PIL adds postprocessing/encoding. Evidence at one boundary does not validate another.

### Internal generators

The runner creates generators from admitted seeds. Pipeline request reconstruction must preserve this internal state.
