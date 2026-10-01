# WaveServe Wan / Noisy PP (experimental)

> Experimental offline Noisy PP on WaveServe Wan 2.1 1.3B (rectified-flow).
> Deviations from [`TEMPLATE.md`](../TEMPLATE.md): this is an opt-in AR-Diffusion
> feature recipe (not a supported-models claim); hardware is stated as a
> qualification profile for #102 acceptance, not a GA hardware matrix.

## Summary

- Vendor: Physis-AI
- Model: `Physis-AI/waveserve-wan2.1-1.3b-diffusers-rf-dev` (Wan 2.1 T2V 1.3B, rectified-flow)
- Task: **Noisy PP** — chunk-autoregressive text-to-video with Serial / Latest-KV within one request
- Mode: Offline Omni with `deploy/waveserve_wan.yaml` + `ARDiffusionEngine`
- Hardware: multi-GPU CUDA (qualify with `S = num_denoise_steps + 1`, `G ≥ 1`)
- Tracking: [project #102](https://github.com/JiusiServe/vllm-omni-project-manage/issues/102) (feature, v0.31); [#103](https://github.com/JiusiServe/vllm-omni-project-manage/issues/103) (model quality, v0.32)
- Maintainer: Community (experimental)

## When to use this recipe

Use this recipe to exercise **Noisy PP** on WaveServe Wan weights through Omni:
same-request multi-chunk Serial (baseline) or Latest-KV diagonal overlap.
DreamZero / LingBot default session paths are unchanged; this deploy is opt-in.

This recipe intentionally avoids a model-specific Python example under
`examples/` (see contributing examples policy). Commands use the shared Omni
Python API and the noisy_pp bench script.

## Supported model contract

| Item | Contract |
| --- | --- |
| Task | Offline text-to-video via Omni `generate` |
| Checkpoint | `Physis-AI/waveserve-wan2.1-1.3b-diffusers-rf-dev` (experimental distilled RF) |
| Entrypoint | Omni + `vllm_omni/deploy/waveserve_wan.yaml` (not `vllm serve --omni` GA path) |
| `chunk_schedule` | Request `extra_args`: `serial` \| `latest` (unknown values raise) |
| Topology | `pipeline_parallel_size = S · G`, `S ∈ {1, T+1}`, `G ≥ 1` |
| Acceptance (#102) | Reproducible basic function + config logging; no FPS / realtime SLA |

## References

- Design: [`docs/design/feature/noisy_pp.md`](../../docs/design/feature/noisy_pp.md)
- Deploy: [`vllm_omni/deploy/waveserve_wan.yaml`](../../vllm_omni/deploy/waveserve_wan.yaml)
- Pipeline: `WaveServeWanPipeline` (`vllm_omni/diffusion/models/waveserve_wan/`)
- Project issue: [#102 Noisy PP feature support](https://github.com/JiusiServe/vllm-omni-project-manage/issues/102)
- Related RFC: [Unified KV Cache Management for the AR-Diffusion Engine](https://github.com/vllm-project/vllm-omni/issues/4366)

## Hardware

- Accelerator: NVIDIA CUDA GPUs (NVLink or PCIe); qualify with enough cards for `S · G`
- Profile used for #102 smoke: 5× GPU with `S=5`, `G=1` (`T=4` denoise + clean)
- Qualification scope: experimental; not a GA hardware claim

## Software environment

- OS: Linux
- Python: 3.10+
- Driver / runtime: NVIDIA driver with a CUDA runtime supported by your PyTorch build
- vLLM / vLLM-Omni: match the repository checkout you are validating

## Command

### Offline generate (Latest-KV, S=5)

```bash
python - <<'PY'
from pathlib import Path
from omegaconf import OmegaConf
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

repo = Path(".").resolve()
cfg = OmegaConf.load(repo / "vllm_omni/deploy/waveserve_wan.yaml")
stage0 = cfg.stages[0]
OmegaConf.update(stage0, "parallel_config.pipeline_parallel_size", 5, force_add=True)
OmegaConf.update(stage0, "model_config.ar_diffusion_stage_config.stage_parallel_size", 5, force_add=True)
deploy = repo / "outputs/waveserve_wan_s5.yaml"
deploy.parent.mkdir(parents=True, exist_ok=True)
deploy.write_text(OmegaConf.to_yaml(cfg))

omni = Omni(
    model="/data/models/waveserve-wan2.1-1.3b-diffusers-rf-dev",
    deploy_config=str(deploy),
)
try:
    outputs = omni.generate(
        "A corgi running along the beach at sunset, waves rolling in, golden light.",
        OmniDiffusionSamplingParams(
            extra_args={
                "num_chunks": 7,
                "num_denoise_steps": 4,
                "kv_history_chunks": 6,
                "chunk_schedule": "latest",
                "reset": True,
            }
        ),
        use_tqdm=False,
    )
    print("ok", type(outputs[0]).__name__, getattr(outputs[0], "metrics", None))
finally:
    omni.close()
PY
```

### Serial vs Latest bench

```bash
python benchmarks/noisy_pp/waveserve_serial_vs_latest.py \
  --model /data/models/waveserve-wan2.1-1.3b-diffusers-rf-dev \
  --deploy-config vllm_omni/deploy/waveserve_wan.yaml \
  --world-size 5 --gpus-per-stage 1 \
  --denoise-steps 4 --chunks 7 --history 6 \
  --repeat 3 --regimes serial,latest \
  --json-out outputs/omni_s5_c7.json
```

### 1× / 2× GPU smoke

```bash
# S=1 serial (single GPU)
python benchmarks/noisy_pp/waveserve_serial_vs_latest.py \
  --model /data/models/waveserve-wan2.1-1.3b-diffusers-rf-dev \
  --world-size 1 --denoise-steps 0 \
  --chunks 1 --history 2 --repeat 1 --regimes serial

# S=2 (T=1 denoise + clean), two GPUs
python benchmarks/noisy_pp/waveserve_serial_vs_latest.py \
  --model /data/models/waveserve-wan2.1-1.3b-diffusers-rf-dev \
  --world-size 2 --denoise-steps 1 \
  --chunks 1 --history 2 --repeat 1 --regimes serial
```

## Verification

```bash
# Expect JSON with serial/latest timings and no traceback
test -f outputs/omni_s5_c7.json && python -c 'import json; print(sorted(json.load(open("outputs/omni_s5_c7.json"))))'
```

Note: `denoise-steps` must satisfy `world-size == (denoise-steps + 1) * gpus-per-stage`.
For a pure S=1 smoke the bench currently requires `denoise-steps + 1 == world-size`;
use a one-off deploy YAML with `pipeline_parallel_size: 1` / `stage_parallel_size: 1`
and `num_denoise_steps: 1` if you need S=1 with T≥1 on a single card (Omni allows
`S ∈ {1, T+1}`).

## Notes

- **Noisy PP** topology: `pipeline_parallel_size = S · G` with `S = num_denoise_steps + 1`
  (denoise stages + clean). `G = 1` keeps a full DiT per stage rank; `G > 1`
  splits layers within each stage (`forward_latent_step` + activation pack
  `{latent[, hidden_states]}`).
- Do **not** wrap Omni in `torchrun`; use the mp executor from deploy YAML.
- The test checkpoint was distilled with timestep **shift 5.0**.
- FPS / speedup / realtime SLA are out of scope for this phase (#102).

## Supported features

| Feature | Status |
| --- | --- |
| Offline Omni generate | Yes (experimental) |
| Serial / Latest-KV `chunk_schedule` | Yes |
| Vertical PP `S=T+1` | Yes |
| Layer groups `G>1` | Code path present; device evidence tracked under #102 |
| Online `vllm serve --omni` | Not claimed |
| DreamZero / LingBot session path | Unchanged (separate deploy) |
