# CausVid Wan 2.1 1.3B (Noisy KV)

> Experimental: run [CausVid](https://github.com/tianweiy/CausVid) Wan 2.1 T2V
> 1.3B student weights on Omni's Cell / Noisy KV path (Self-Forcing / Latest-KV).

## Summary

- Student DiT: `tianweiy/CausVid` → `autoregressive_checkpoint/model.pt`
- Codec (T5 / VAE): official `Wan-AI/Wan2.1-T2V-1.3B` via Diffusers layout
  `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` (Omni needs Diffusers; same Wan-AI weights)
- Engine: `ARDiffusionEngine` + `CausVidWanPipeline` (Cell / Noisy KV stack)
- Defaults: `num_denoise_steps=3` (`D`), `shift=8.0`
- Schedule ↔ topology:

| Schedule | Deploy YAML | Topology |
| --- | --- | --- |
| Self-Forcing | `vllm_omni/deploy/causvid_wan_self_forcing.yaml` | `S=1`, `G=2` (2 GPUs, layer PP) |
| Latest-KV | `vllm_omni/deploy/causvid_wan_latest_kv.yaml` | `S=D+1=4`, `G=1` (4 GPUs, vertical) |

CausVid trains causal few-step DMD on Wan 2.1 1.3B. Official README uses native
`wan_models/Wan2.1-T2V-1.3B/`; Omni loads the Diffusers packaging of that same
base, then remaps CausVid `model.pt` onto the DiT.

## Layout

| Piece | Source |
| --- | --- |
| Tokenizer / UMT5 / VAE | `--model` → `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` |
| DiT student | `model_config.causvid_ckpt` → CausVid `autoregressive_checkpoint/model.pt` |
| Deploy (Self-Forcing) | `vllm_omni/deploy/causvid_wan_self_forcing.yaml` |
| Deploy (Latest-KV) | `vllm_omni/deploy/causvid_wan_latest_kv.yaml` |

## Prepare checkpoints

```bash
# Official Wan 2.1 1.3B base (Diffusers layout for Omni)
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --local-dir /data/models/Wan2.1-T2V-1.3B-Diffusers

# CausVid student DiT
huggingface-cli download tianweiy/CausVid \
  --include "autoregressive_checkpoint/*" \
  --local-dir /data/models/CausVid
```

## Offline smoke (2 GPUs, Self-Forcing)

```python
from pathlib import Path
from omegaconf import OmegaConf
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

repo = Path("/path/to/vllm-omni")
cfg = OmegaConf.load(repo / "vllm_omni/deploy/causvid_wan_self_forcing.yaml")
cfg.stages[0].model_config.causvid_ckpt = (
    "/data/models/CausVid/autoregressive_checkpoint/model.pt"
)

omni = Omni(
    model="/data/models/Wan2.1-T2V-1.3B-Diffusers",
    deploy_config=cfg,
)
out = omni.generate(
    "A corgi running on the beach at sunset.",
    OmniDiffusionSamplingParams(
        height=480,
        width=832,
        num_frames=21,  # aligns with CausVid latent T≈3 per chunk × chunks
        extra_args={
            "num_chunks": 7,
            "num_denoise_steps": 3,
            "cell_schedule": "self_forcing",
            "shift": 8.0,
            "reset": True,
        },
    ),
)
```

For Latest-KV, load `causvid_wan_latest_kv.yaml` instead (4 GPUs) and keep
`cell_schedule: "latest_kv"` as in that file.

## Notes

- **Self-Forcing** (`Ordering.SELF_FORCING`) matches CausVid clean-history
  semantics on **S=1** layer PP. **Latest-KV** needs vertical **S=D+1** and
  reads the newest finished history step (may P2P KV). CausVidWan rejects
  `same_step`.
- Exact official DMD timestep table parity (`1000/757/522`) vs the shared
  FlowEuler grid is still converging; defaults use CausVid **shift=8** and
  **D=3** (S=1 Self-Forcing uses clean when `H>0`; Latest-KV YAML uses
  `S=D+1` with an explicit clean replica).
- Do not wrap in `torchrun`; use deploy `distributed_executor_backend: mp`.
