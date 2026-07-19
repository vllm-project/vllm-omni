# Alpamayo — Online Serving

End-to-end HTTP path: `vllm-omni serve` + a pure-HTTP client. Reference
result on a representative clip: minADE@4 = 0.44 m, minADE@16 = 0.28 m
(vs upstream NVIDIA Alpamayo 1.5 reference 0.585 m at the same N).

For the offline (in-process engine) eval path, see
[`examples/offline_inference/alpamayo/`](../../offline_inference/alpamayo/).

---

## 0. Prerequisites

Export these once so the rest of the commands stay generic:

```bash
export ALPAMAYO_REPO=/path/to/vllm-omni-alpamayo              # this repo (clone)
export ALPAMAYO_WEIGHTS=nvidia/Alpamayo-1.5-10B               # HF id or local dir — SERVER ONLY
export ALPAMAYO_VLM_BASE=Qwen/Qwen3-VL-8B-Instruct            # HF id or local dir — SERVER ONLY
export ALPAMAYO_MODEL=alpamayo-1.5                            # public name (--served-model-name)
export ALPAMAYO_TOKENIZER_DIR=/tmp/alpamayo-tokenizer         # baked in step 1 — SERVER ONLY
export ALPAMAYO_CLIP_PKL=/path/to/clip.pkl                    # see "clip format" below
```

`ALPAMAYO_WEIGHTS` is where the weights actually live (passed to
`vllm-omni serve`). `ALPAMAYO_MODEL` is the public name the server
advertises and clients send in the `model` field of their HTTP request —
it has nothing to do with where the weights live. Keep them decoupled
so you can move/rename the weight dir without breaking clients.

The HTTP client only needs `ALPAMAYO_SERVER`, `ALPAMAYO_MODEL`,
`ALPAMAYO_CLIP_PKL` and (optionally) `ALPAMAYO_N_SAMPLES` — no weights,
no tokenizer, no config download. It decodes the returned actions to an
xyz trajectory client-side (the action-space normalization constants are
embedded in the script, copied from the checkpoint's `config.json`) and
prints minADE / meanADE against the clip's GT future.

You also need:
- `vllm-omni` installed (`pip install -e .` from the repo root).
- A free GPU with ≥ 90 GB free (`nvidia-smi`).

**Clip format.** `ALPAMAYO_CLIP_PKL` is a `pickle.load`-able dict with
keys `image_frames` (tensor `[C, F, 3, H, W]`),
`camera_indices` (length-C int list), `ego_history_xyz` (tensor
`[n_traj, T, 3]`), `ego_history_rot` (`[n_traj, T, 3, 3]`) and
`ego_future_xyz` (`[1, 1, T_future, 3]` GT used for ADE).

---

## 1. One-time setup: bake the extended tokenizer

Alpamayo-1.5-10B ships **no tokenizer files**. We borrow Qwen3-VL-8B's
tokenizer and append Alpamayo's 4000 discrete trajectory tokens
(`<i0>..<i3999>`) + 28 special tokens (`<|traj_history|>`,
`<|traj_future_start|>`, `<|cot_start|>`, ...) to it, then save the
extended tokenizer to a fresh dir.

```python
import os
from transformers import AutoProcessor
from vllm_omni.model_executor.models.alpamayo.processing import add_alpamayo_tokens

processor = AutoProcessor.from_pretrained(os.environ["ALPAMAYO_VLM_BASE"], trust_remote_code=True)
info = add_alpamayo_tokens(processor.tokenizer)  # +4000 <i*> traj tokens + ~28 special tokens
processor.save_pretrained(os.environ["ALPAMAYO_TOKENIZER_DIR"])
print("vocab is now", len(processor.tokenizer), "| future_start id:", info["traj_token_ids"]["future_start"])
```

Expected: `vocab is now 155697 | future_start id: 155681` (and `traj_token_start_idx` 151669),
written to `$ALPAMAYO_TOKENIZER_DIR`.

Run once per host. Output is self-contained (does NOT touch the base
tokenizer directory).

---

## 2. Start the server

Pick a free GPU (replace `0` below):

```bash
CUDA_VISIBLE_DEVICES=0 \
vllm-omni serve $ALPAMAYO_WEIGHTS \
  --served-model-name $ALPAMAYO_MODEL \
  --omni \
  --port 8765 \
  --tokenizer $ALPAMAYO_TOKENIZER_DIR \
  --trust-remote-code \
  --trust-request-chat-template \
  --dtype bfloat16 \
  --enforce-eager \
  --gpu-memory-utilization 0.6 \
  --max-model-len 32768 \
  --limit-mm-per-prompt '{"image": 16}'
```

Wait for `Application startup complete.` (~50 s including model load).

### Flag cheatsheet

| Flag | Why |
|---|---|
| `--omni` | **Required.** Routes to vllm-omni's stage-pipeline engine. Without it falls back to vllm's plain engine and rejects `Alpamayo1_5` architecture. |
| `--served-model-name $ALPAMAYO_MODEL` | Public name clients send in the HTTP `model` field. Decouples the wire name from the weight path. |
| `--tokenizer $ALPAMAYO_TOKENIZER_DIR` | The extended tokenizer dir from step 1. |
| `--trust-remote-code` | Alpamayo config / pipeline registration uses custom Python. |
| `--trust-request-chat-template` | The HTTP client passes a pass-through chat template so the prompt string isn't wrapped with extra role markers. |
| `--limit-mm-per-prompt '{"image": 16}'` | 16 cameras × 4 frames worth of images per request. |
| `--enforce-eager` | Disables CUDA-graph capture (Alpamayo's inline flow-matching path doesn't graph-compile cleanly). |
| `--gpu-memory-utilization 0.6` | Target ~84 GB on a 140-GB H200. Bump up on a dedicated GPU; drop (e.g. 0.4) on a shared GPU. vLLM's default 0.9 = 125 GB which fails on most partially-used GPUs. |

Flow-matching sample count is per-request (see `ALPAMAYO_N_SAMPLES`
below), not a server flag.

---

## 3. Run the HTTP client

In another terminal:

```bash
ALPAMAYO_SERVER=http://localhost:8765 \
ALPAMAYO_N_SAMPLES=4 \
python3 $ALPAMAYO_REPO/examples/online_serving/alpamayo/http_client.py
```

(`ALPAMAYO_MODEL` and `ALPAMAYO_CLIP_PKL` are inherited from the export
block above.)

Expected output:
```
[http] status=200 ~10s
[actions] shape=(4, 64, 2)
clip=<clip-id>  n=4
  minADE@4  ≈ 0.44 m
  meanADE@4 ≈ 2–3 m
```

Server-side log (in the server terminal) will show:
```
Alpamayo15: history fusion done for req chatcmpl-... (48 delta tokens)
Alpamayo15: trajectory trigger fired for 1 request(s)
Alpamayo15: flow matching completed — sampled actions shape (4, 64, 2)
```

For minADE@16 set `ALPAMAYO_N_SAMPLES=16`.

---

## 4. Writing your own HTTP client

The client is a normal OpenAI `/v1/chat/completions` request. Key
payload shape (full version in `http_client.py`):

```python
import json, requests, base64

payload = {
    "model": "alpamayo-1.5",                       # = server's --served-model-name
    "messages": [{"role": "user", "content": [
        # 16 images, each as a base64 data URL
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
        # ... (15 more)
        # Final text block = the fully-formatted Alpamayo prompt
        {"type": "text", "text": prompt},
    ]}],
    "max_tokens": 400,
    "temperature": 0.6,
    "top_p": 0.98,
    # Pass-through template so the prompt string we built isn't wrapped.
    "chat_template": "{% for m in messages %}"
                      "{% for c in m.content %}"
                      "{% if c.type == 'text' %}{{ c.text }}{% endif %}"
                      "{% endfor %}{% endfor %}",
    # robot_obs is the per-request ego history. The vllm_xargs protocol
    # restricts values to flat primitives, so we JSON-encode the nested
    # dict; the server's prepare_runner_inputs accepts either a dict
    # (Python clients) or a JSON string (HTTP) under the same key.
    "vllm_xargs": {
        "robot_obs": json.dumps({
            "ego_history_xyz": <nested list, shape (1, n_traj, T, 3)>,
            "ego_history_rot": <nested list, shape (1, n_traj, T, 3, 3)>,
        }),
        "n_samples": 4,                            # per-request FM rolls
    },
}
resp = requests.post("http://localhost:8765/v1/chat/completions", json=payload)
actions = resp.json()["choices"][0]["message"]["multimodal_output"]["actions"]
# actions is a nested list, shape (n_samples, n_waypoints=64, action_dim=2)
```

`actions` are normalized (acceleration, curvature) controls, not
positions. To turn them into a 64-waypoint xyz trajectory, run them
through `UnicycleAccelCurvatureActionSpace.action_to_traj(actions,
ego_history_xyz, ego_history_rot)` with the normalization constants from
the checkpoint's `config.json` — see the `ACTION_SPACE_CFG` block in
`http_client.py` for the exact values and call.

The `prompt` string must include:
- Per-camera blocks: `"Front camera: frame 0 <|vision_start|><|image_pad|><|vision_end|>frame 1 <|vision_start|><|image_pad|><|vision_end|>..."`
  (camera name + frame index + image placeholder, repeated for each
  camera × each frame; the order of `<|image_pad|>` matches the order
  of `image_url` blocks in `messages`)
- 48 `<|traj_history|>` placeholders inside `<|traj_history_start|>...<|traj_history_end|>` — the server's
  `prepare_runner_inputs` swaps them with delta-encoded ego history
- Standard system + user wrapper + `<|cot_start|>` to start AR

See `processing.build_alpamayo_prompt(camera_indices, num_frames_per_camera)`
for the canonical builder. Pure string templating, no tokenizer needed
on the client side.
