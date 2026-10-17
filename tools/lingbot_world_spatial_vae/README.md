# Validate LingBot-World spatial VAE decode

This manual correctness tool decodes saved real LingBot-World latents using the
existing four-rank width-sharded Wan VAE. It does not run DiT, benchmark latency
or throughput, compare scheduling strategies, or change production serving.
The underlying spatial-shard backend is already part of vLLM-Omni.

## Run

Use the normal vLLM-Omni CUDA environment, four visible GPUs and a LingBot-World
v2 causal checkpoint with an untiled FP32 Wan VAE (`patch_size=None`). Validation
also loads an unpatched single-GPU reference; allow memory for it on rank zero.
The tool does not allocate or reserve GPUs. Obtain them through your scheduler
and choose an unused local rendezvous port.

Provide ten trusted normalized FP32 BCTHW CPU tensors named
`steady_latent_00.pt` through `steady_latent_09.pt`, each with three latent
frames. The names match the saved outputs from the original LingBot analysis.
Loading uses `weights_only=True`.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/lingbot_world_spatial_vae/validate.py \
  --model /path/to/lingbot-world-v2-14b-causal-fast-diffusers \
  --latents /path/to/saved/latents \
  --output /path/to/new/validation-result --port 29500
```

## Checks and artifacts

Two sessions exercise state reset. Each chunk must be finite and yield nine
output frames initially, then twelve for each subsequent chunk. The first four
chunks of session zero are compared, before uint8 conversion, against:

- concatenated four-rank decode with a fresh cache (`atol=rtol=1e-5`);
- unpatched, untiled single-GPU FP32 decode (maximum absolute error 0.03).

This prefix check does not establish whole-rollout equivalence or bitwise
spatial-versus-single-GPU equivalence. The second session exercises reset but
is not an independent reference or an exact cross-session equality check.

The output directory contains `hardware.json`, per-chunk shapes and hashes in
`decode.jsonl`, every decoded uint8 chunk, `validation.json`, `summary.json`
and the FP32 comparison operands in `validation_tensors.pt`. Input hashes are
included for reproducibility. No timing, peak-memory benchmark, profiler trace,
media encoding or network delivery is measured.

The supervisor terminates sibling ranks after a rank failure and reaps child
processes on exit. The parent stops the supervisor on timeout or error. This is
a validation utility, not an inference service or cancellation API.
