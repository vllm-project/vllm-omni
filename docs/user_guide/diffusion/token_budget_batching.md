# Token-Budget Diffusion Batching

Diffusion step execution batches compatible requests together. A fixed
`max_num_seqs` works for one request shape, but different resolutions have
different per-step cost, so a fixed count either wastes capacity on small
requests or over-commits on large ones. Token-budget admission (the
`compute_budget` policy) admits waiting requests until the batch reaches a
budget counted in packed patch tokens.

## Token Accounting

The admission unit is the packed patch token:

```text
tokens = (height / (vae_scale_factor * latent_patch_size))
       * (width  / (vae_scale_factor * latent_patch_size))
       * num_frames * num_outputs_per_prompt
```

With the default `vae_scale_factor=8` and `latent_patch_size=2`, one
512x512 image is 1024 tokens and one 1024x1024 image is 4096 tokens. This is
the same unit as the MixFusion chunk envelope, so the budget, the envelope,
and the DRR score share one source of truth.

The CFG weight is deliberately excluded from capacity: true CFG changes how a
request executes, not how many requests fit. It only scales the DRR ordering
score through `cfg_compute_multiplier` (default `2.0`).

Requests are admitted one at a time while their token cost fits the remaining
budget. A request that does not fit waits for a later cycle. When nothing is
running and the cheapest waiting request alone exceeds the budget, that
cheapest request is admitted anyway so admission cannot deadlock; this also
covers a budget calibrated on cheaper reference requests than the ones the
server actually serves.

## Choosing the Budget

`max_num_batched_tokens` is the only DRR admission-budget entry:

- Explicit `--max-num-batched-tokens N`: used as the admission budget and
  clamped to the paged KV token capacity when that capacity is known.
- Unset (the default): `DEFAULT_DIFFUSION_TOKEN_BUDGET = 16384` packed
  tokens (sixteen 512x512 images or four 1024x1024 images), optionally
  replaced by startup calibration.

The default is not written back into `max_num_batched_tokens`, because that
field also sizes native MoE LoRA token buffers and the paged KV block table.
In paged mode the DRR budget is clamped to the block-table capacity, so a
larger budget requires a larger paged KV capacity as well.

`max_num_seqs` is a literal request-count cap, independent of the budget. Under
DRR, unset `max_num_seqs` means no count cap and the token budget alone bounds
the batch; a set value caps the count at that number. Other scheduling paths
resolve unset `max_num_seqs` to 1.

Unknown keys in `--diffusion-batching-config` are rejected. The removed
`compute_unit_budget` key raises an error that points at
`max_num_batched_tokens` instead of being silently renamed: the unit is
unchanged for image requests, so the old numeric value can be passed as the
budget directly.

## First-Time Use on a New Setup

First run the profiler policy. Keep `--max-num-seqs` large enough to cover the
batch sizes you want to sample; it is a literal count cap here.

```bash
python3 -m vllm_omni.entrypoints.cli.main serve /data/Qwen-Image \
  --omni \
  --step-execution \
  --max-num-seqs 128 \
  --diffusion-batching-policy profile
```

Send high-concurrency benchmark traffic using the reference request shape
(512x512 for Qwen-Image). The server log then contains `[DiffusionBatchProfiler]`
records with batch size, request tokens, total tokens, and denoise-step time.

Generate the budget from the server log. The profiler groups the 512x512
records by batch size, filters outliers with IQR, uses the median denoise-step
time per batch size, and applies a two-segment linear fit to locate the
sweet-spot batch size.

```bash
python3 benchmarks/diffusion/diffusion_batching_profiler.py \
  --log-file /path/to/server.log \
  --reference-tokens 1024 \
  --output-json /tmp/diffusion_batching_config.json
```

The generated JSON reports `max_num_batched_tokens`, computed as:

```text
max_num_batched_tokens = reference_tokens * sweet_spot_batch_size
```

Use the generated value for later serving:

```bash
TOKEN_BUDGET=<generated_max_num_batched_tokens>

python3 -m vllm_omni.entrypoints.cli.main serve /data/Qwen-Image \
  --served-model-name Qwen/Qwen-Image \
  --omni \
  --port 8081 \
  --step-execution \
  --max-num-seqs 128 \
  --max-num-batched-tokens "${TOKEN_BUDGET}" \
  --diffusion-batching-policy compute_budget \
  --diffusion-batching-config '{"drr_alpha":1.0,"drr_beta":1.0,"log_stats":true}' \
  --vae-use-tiling \
  --vae-use-slicing
```

`drr_alpha`, `drr_beta`, `drr_reserve_time_s`, and `log_stats` are optional
policy knobs; `log_stats` only enables verbose batching-policy logs.

## Use the Built-In Reference Profile

Use this only when the environment matches the built-in reference: Qwen-Image
on one Ascend 910B2 NPU, profiled with 512x512 requests. The built-in budget is
ten reference images, `10240` packed tokens:

```bash
python3 -m vllm_omni.entrypoints.cli.main serve /data/Qwen-Image \
  --served-model-name Qwen/Qwen-Image \
  --omni \
  --port 8081 \
  --step-execution \
  --max-num-batched-tokens 10240 \
  --diffusion-batching-policy compute_budget \
  --vae-use-tiling \
  --vae-use-slicing
```

For 1024x1024 requests the token cost is 4096, so two fit the budget and the
third waits for the next cycle.

## Startup Calibration

Instead of profiling manually, the engine can calibrate the DRR budget at
startup. Enable it in the batching config:

```bash
--diffusion-batching-policy compute_budget \
--diffusion-batching-config '{"warmup_auto_calibrate":true,"warmup_reference_height":512,"warmup_reference_width":512}'
```

Calibration is an explicit startup phase that runs after the executor and
scheduler exist, because it needs a real forward pass. It sweeps a batch-size
ladder at the reference shape, picks the sweet spot, and applies the result
through the scheduler, which clamps it to the mode-defined capacity: the paged
KV block-table capacity in paged mode, unclamped in dense mode (dense mode has
no token-counted admission capacity). It runs only when `warmup_auto_calibrate`
is enabled, and is skipped when `--max-num-batched-tokens` is set explicitly,
when step execution is off, or when DRR is not the resolved policy. The
calibrated profile is cached so only the first startup pays the sweep;
`warmup_profile_refresh` forces a re-sweep.

## Mixed-Shape Scheduling

Step-wise continuous batching requires requests in the same running batch to
have compatible `SamplingParamsKey` values. FIFO stops at an incompatible
queue head; DRR instead reorders waiting requests at every scheduling cycle,
so a large head request does not block cheaper requests that could fill the
batch.

DRR scores each waiting request as:

```text
score = age_s^drr_alpha / (cost / 1024)^drr_beta
```

where `cost` is the request's packed-token count multiplied by
`cfg_compute_multiplier` when true CFG is enabled, and 1024 is the
normalization reference (one 512x512 image). Only the ordering uses the CFG
weight; the budget does not.

The admission loop fills the remaining budget in score order. The oldest
request that has waited longer than `drr_reserve_time_s` (default 60 seconds)
becomes the reservation target: while it cannot join the running batch yet,
only candidates that fit the gap left after reserving its tokens are admitted,
and admission stops there so the gap survives. Once it fits, it is admitted
first, so younger cheaper requests cannot keep taking the slot ahead of it.
