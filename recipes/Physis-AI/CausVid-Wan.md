# CausVid Wan 2.1 1.3B (Noisy KV)

> Experimental: run [CausVid](https://github.com/tianweiy/CausVid) Wan 2.1 T2V
> 1.3B student weights on Omni's Cell / Noisy KV path (Self-Forcing / Latest-KV).

## Summary

- Student DiT: `tianweiy/CausVid` → `autoregressive_checkpoint/model.pt`
- Codec (T5 / VAE): official `Wan-AI/Wan2.1-T2V-1.3B` via Diffusers layout
  `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` (Omni needs Diffusers; same Wan-AI weights)
- Engine: `ARDiffusionEngine` + `CausVidWanPipeline` (Cell / Noisy KV stack)
- Default schedule: `num_denoise_steps=3`, `shift=8.0`, `S=T+1=4`

CausVid trains causal few-step DMD on Wan 2.1 1.3B. Official README uses native
`wan_models/Wan2.1-T2V-1.3B/`; Omni loads the Diffusers packaging of that same
base, then remaps CausVid `model.pt` onto the DiT.

## Layout

| Piece | Source |
| --- | --- |
| Tokenizer / UMT5 / VAE | `--model` → `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` |
| DiT student | `model_config.causvid_ckpt` → CausVid `autoregressive_checkpoint/model.pt` |
| Deploy | `vllm_omni/deploy/causvid_wan.yaml` |

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

## Offline smoke (4 GPUs, Self-Forcing)

```python
from pathlib import Path
from omegaconf import OmegaConf
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

repo = Path("/path/to/vllm-omni")
cfg = OmegaConf.load(repo / "vllm_omni/deploy/causvid_wan.yaml")
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
            "cell_schedule": "self_forcing",  # or latest_kv / same_step (canonical names only; serial/latest/interleaved removed)
            "shift": 8.0,
            "reset": True,
        },
    ),
)
```

## Notes

- **Self-Forcing** (`Ordering.SELF_FORCING`) matches CausVid clean-history
  semantics; **Latest-KV** and **Same-Step** (`same_step`)
  share the diagonal placement but differ in which history KV step is read —
  Same-Step stays local (no KV P2P). Neither matches how CausVid was trained.
- Exact official DMD timestep table parity (`1000/757/522`) vs the shared
  FlowEuler grid is still converging; defaults use CausVid **shift=8** and
  **T=3** on the Euler cell path.
- Do not wrap in `torchrun`; use deploy `distributed_executor_backend: mp`.
