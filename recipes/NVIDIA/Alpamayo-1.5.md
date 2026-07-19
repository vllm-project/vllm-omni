# Alpamayo-1.5 for vision-language-action driving trajectories

## Summary

- Vendor: NVIDIA
- Model: `nvidia/Alpamayo-1.5-10B` (Qwen3-VL-8B backbone + flow-matching action expert)
- Task: Vision-language-action — multi-camera images + ego history → chain-of-thought
  reasoning + a 64-waypoint future trajectory
- Mode: Online serving via the OpenAI-compatible `/v1/chat/completions` API; the
  predicted trajectory is returned under `choices[*].message.multimodal_output["actions"]`
- Maintainer: Community

> ⚠️ The Alpamayo weights carry a **non-commercial** license (NVIDIA). Review the
> upstream license before any deployment.

## When to use this recipe

Use this recipe to stand up Alpamayo-1.5 as a single-stage autoregressive pipeline:
the Qwen3-VL backbone generates reasoning text until the `<|traj_future_start|>`
trigger token, then `forward()` dispatches inline to a flow-matching expert that
samples action trajectories — all inside one model class (the sglang single-model
pattern), no cross-stage KV transfer.

## References

- Upstream model card: <https://huggingface.co/nvidia/Alpamayo-1.5-10B>
- Related examples under `examples/`:
  [`examples/online_serving/alpamayo/`](../../examples/online_serving/alpamayo/README.md)
  (HTTP path) and
  [`examples/offline_inference/alpamayo/`](../../examples/offline_inference/alpamayo/README.md)
  (in-process engine eval)
- Related issue or discussion:
  [vllm-project/vllm-omni#2873](https://github.com/vllm-project/vllm-omni/issues/2873)

## Hardware Support

## GPU

### 1x H200 141GB

#### Environment

- OS: Ubuntu 22.04
- Python: 3.12
- Driver / runtime: CUDA 12.x
- GPU: NVIDIA H200, 141 GB (≥ 90 GB free required)
- vLLM version: 0.22.x
- vLLM-Omni version or commit: this branch

#### Command

Alpamayo-1.5-10B ships **no tokenizer files**, so bake an extended tokenizer once
(borrows Qwen3-VL-8B's tokenizer + Alpamayo's 4000 trajectory tokens + special
tokens):

```bash
export ALPAMAYO_WEIGHTS=nvidia/Alpamayo-1.5-10B            # HF id or local dir
export ALPAMAYO_VLM_BASE=Qwen/Qwen3-VL-8B-Instruct        # tokenizer base
export ALPAMAYO_TOKENIZER_DIR=/tmp/alpamayo-tokenizer
export ALPAMAYO_MODEL=alpamayo-1.5                         # public served name
```

```python
# Bake the extended tokenizer once, then point --tokenizer at the output dir.
import os
from transformers import AutoProcessor
from vllm_omni.model_executor.models.alpamayo.processing import add_alpamayo_tokens

processor = AutoProcessor.from_pretrained(os.environ["ALPAMAYO_VLM_BASE"], trust_remote_code=True)
add_alpamayo_tokens(processor.tokenizer)  # +4000 <i*> traj tokens + ~28 special tokens
processor.save_pretrained(os.environ["ALPAMAYO_TOKENIZER_DIR"])
```

Start the server (the stage config is auto-loaded from
`vllm_omni/deploy/alpamayo1_5.yaml`):

```bash
CUDA_VISIBLE_DEVICES=0 \
vllm-omni serve "$ALPAMAYO_WEIGHTS" \
  --served-model-name "$ALPAMAYO_MODEL" \
  --omni \
  --port 8765 \
  --tokenizer "$ALPAMAYO_TOKENIZER_DIR" \
  --trust-remote-code \
  --trust-request-chat-template \
  --dtype bfloat16 \
  --enforce-eager \
  --gpu-memory-utilization 0.6 \
  --max-model-len 32768 \
  --limit-mm-per-prompt '{"image": 16}'
```

Wait for `Application startup complete.` (~50 s including model load).

#### Verification

Run the bundled HTTP client (decodes the returned actions to an xyz trajectory and
prints minADE/meanADE vs the clip's GT future):

```bash
export ALPAMAYO_CLIP_PKL=/path/to/clip.pkl   # see the example README for the clip schema
ALPAMAYO_SERVER=http://localhost:8765 \
ALPAMAYO_N_SAMPLES=4 \
python3 examples/online_serving/alpamayo/http_client.py
```

Expected output:

```
[http] status=200 ~10s
[actions] shape=(4, 64, 2)
clip=<clip-id>  n=4
  minADE@4  ≈ 0.44 m
  meanADE@4 ≈ 2–3 m
```

Reference result on a representative clip: minADE@4 = 0.44 m, minADE@16 = 0.28 m
(vs the upstream NVIDIA Alpamayo-1.5 reference 0.585 m at the same N). Set
`ALPAMAYO_N_SAMPLES=16` for minADE@16.

The client is a plain OpenAI `/v1/chat/completions` request: multi-camera images +
`extra_body={"robot_obs": {"ego_history_xyz": ..., "ego_history_rot": ...}, "n_samples": N}`.
The server fuses the ego history into the prompt's `<|traj_history|>` placeholders and
returns the predicted trajectory under
`choices[0].message.multimodal_output["actions"]`. See
[`examples/online_serving/alpamayo/README.md`](../../examples/online_serving/alpamayo/README.md)
for the full payload shape and clip format.

#### Notes

- Memory usage: target ~84 GB on a 141-GB H200 with `--gpu-memory-utilization 0.6`.
  Bump up on a dedicated GPU; lower (e.g. 0.4) on a shared one. vLLM's default 0.9
  (~125 GB) fails on most partially-used GPUs.
- Key flags: `--omni` is **required** (routes to the stage-pipeline engine; without
  it the `Alpamayo1_5` architecture is rejected). `--enforce-eager` is required —
  the inline flow-matching path does not CUDA-graph-compile cleanly.
  `--limit-mm-per-prompt '{"image": 16}'` covers the 16-camera input.
- Known limitations: graph capture is unsupported (use `--enforce-eager`). The
  per-request flow-matching sample count is set via `n_samples` in the request body,
  not a server flag.
