# Deploying vllm-omni

Build-and-deploy runbook for the `rekaai/vllm-omni-fork` Docker image. Similar to how we deploy our vLLM fork, but this image layers on the upstream `vllm/vllm-openai` base (no multi-hour CUDA compile) and bundles two siblings: `vllm-reka` (plugin) and `moonvalley_ai` (opensora source for Marey's VAE).

References
- [Deploying our vLLM fork](https://www.notion.so/moonvalley/Deploying-our-vLLM-fork-321cef6731968073b8a4f3b4c903cf58)

## Prerequisites

- Access to Reka's Docker Hub (`rekaai` org)
- `docker` with BuildKit enabled (on Linux set `DOCKER_BUILDKIT=1`)
- Sibling repo layout on the machine you're building from:

  ```
  <parent>/
      vllm-omni/        ← this repo
      vllm-reka/        ← https://github.com/reka-ai/vllm-reka
      moonvalley_ai/    ← opensora source (https://github.com/reka-ai/moonvalley_ai)
  ```

  The build script fails fast if any sibling is missing.

- For production deploy: access to the infra repo (`reka-code/infra`) and `tk` CLI set up for the target cluster.

### Running locally (outside Docker)

If you want to run the server directly on your machine instead of via Docker, you need to install `vllm` first — it's an expected host dependency that isn't declared in this repo's own requirements.

```bash
pip install vllm==0.17.0
pip install -e .          # then install vllm-omni
```

The Docker workflow doesn't need this because the base image (`vllm/vllm-openai:v0.17.0`) already includes vllm.

## 1. Sync the three repos

On whatever machine you're building from, make sure all three sibling repos
are at the commits you want baked into the image.

```bash
cd <parent>
( cd vllm-omni     && git pull && git log -1 --oneline )
( cd vllm-reka     && git pull && git log -1 --oneline )
( cd moonvalley_ai && git pull && git log -1 --oneline )
```

You don't need to write SHAs down — `build-serving.sh` attaches per-repo git state as OCI labels on the resulting image (HEAD SHA, HEAD commit summary, last 10 commits, and a dirty-working-tree flag for each repo, plus build time/host/user). See "Verify what's inside a built image" below for how to read them back.

## 2. Build the image

Use the `build-serving.sh` wrapper. It discovers the sibling layout from its own location, symlinks the correct `.dockerignore` into place for the build, runs `docker build`, and cleans up.

```bash
./docker/build-serving.sh \
    --build-arg torch_cuda_arch_list="9.0" \
    --build-arg max_jobs=28 \
    --build-arg nvcc_threads=4 \
    -t rekaai/vllm-omni-fork:$(git rev-parse HEAD) \
    -t rekaai/vllm-omni-fork:latest
```

### Notes

- **CPU build is fine.** Unlike the full vLLM fork build, this layers on top of `vllm/vllm-openai:v0.17.0` which already has vLLM + CUDA kernels compiled. No GPU needed at build time; expect a few minutes, not hours.
- **If building on macOS**, add `--platform linux/amd64` so the resulting image runs on Linux/amd64 GPU nodes. Builds via QEMU will be noticeably slower — prefer a Linux amd64 build host if you have one.
- **Git SHA as the tag** is the `vllm-omni` commit only. The image also depends on `vllm-reka` and `moonvalley_ai`, so you should **always pin by`@sha256:` when deploying** (step 5) — the content digest covers all three.

### Verify what's inside a built image

All per-repo git state is attached as OCI labels — no need to run the
image. Inspect via `docker inspect`:

```bash
# All labels
docker inspect --format '{{json .Config.Labels}}' rekaai/vllm-omni-fork:${VLLM_OMNI_SHA} | jq

# Just the HEAD commit summary for each repo
docker inspect --format '{{index .Config.Labels "org.reka.vllm-omni.head"}}'     rekaai/vllm-omni-fork:${VLLM_OMNI_SHA}
docker inspect --format '{{index .Config.Labels "org.reka.vllm-reka.head"}}'     rekaai/vllm-omni-fork:${VLLM_OMNI_SHA}
docker inspect --format '{{index .Config.Labels "org.reka.moonvalley_ai.head"}}' rekaai/vllm-omni-fork:${VLLM_OMNI_SHA}

# Recent log for a repo (last 10 commits)
docker inspect --format '{{index .Config.Labels "org.reka.vllm-omni.log"}}' rekaai/vllm-omni-fork:${VLLM_OMNI_SHA}

# Dirty-tree check — if any of these are "true", the image contains
# uncommitted changes from somebody's working copy. Don't push to prod.
for repo in vllm-omni vllm-reka moonvalley_ai; do
    echo "$repo: $(docker inspect --format "{{index .Config.Labels \"org.reka.${repo}.dirty\"}}" rekaai/vllm-omni-fork:${VLLM_OMNI_SHA})"
done
```

Labels attached:

| Label | Contents |
|---|---|
| `org.opencontainers.image.revision` | `vllm-omni` HEAD commit SHA (OCI standard) |
| `org.opencontainers.image.created` | Build timestamp (OCI standard) |
| `org.reka.build-time` / `-host` / `-user` | Build provenance |
| `org.reka.<repo>.commit-sha` | HEAD commit SHA (full 40-char git hash) |
| `org.reka.<repo>.head` | HEAD commit summary (short commit sha, date, author, subject) |
| `org.reka.<repo>.log` | Last 10 commits (multiline; each line is `<short-commit-sha> <date> <subject>`) |
| `org.reka.<repo>.dirty` | `"true"` if working tree had uncommitted changes |

for `<repo>` ∈ `{vllm-omni, vllm-reka, moonvalley_ai}`.

## 3. Smoke test on a GPU node

Don't push an image you haven't run. SSH into an H100 node (or any CUDA node), pull the image, and run a quick `/health` check.

```bash
# On the GPU node
docker pull rekaai/vllm-omni-fork:${VLLM_OMNI_SHA}

# vllm-reka mode (e.g. reka-edge-2603)
# This exercises the vllm-reka plugin bundled in the image.
docker run --gpus all --rm -p 8000:8000 \
    -v /path/to/reka-edge-2603:/model \
    -e USE_IMAGE_PATCHING=1 \
    -e VLLM_USE_V1=1 \
    -e VLLM_FLASH_ATTN_VERSION=3 \
    -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=300 \
    -e VLLM_VIDEO_LOADER_BACKEND=yasa \
    rekaai/vllm-omni-fork:latest \
    vllm serve /model \
        --served-model-name reka-edge-2603 \
        --tokenizer-mode yasa \
        --gpu-memory-utilization 0.95 \
        --max-model-len 8192 \
        --max-num-batched-tokens 20000 \
        --limit-mm-per-prompt '{"image": 6, "video": 3}' \
        --media-io-kwargs '{"video": {"num_frames": 6, "sampling": "chunk"}}' \
        --tensor-parallel-size 1 \
        --dtype bfloat16 \
        --chat-template-content-format openai \
        --enable-auto-tool-choice \
        --tool-call-parser hermes \
        --trust-remote-code \
        --quantization bitsandbytes

# Using HF directly
docker run --gpus all --rm -p 8000:8000 \
    -e USE_IMAGE_PATCHING=1 \
    -e VLLM_USE_V1=1 \
    -e VLLM_FLASH_ATTN_VERSION=3 \
    -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=300 \
    -e VLLM_VIDEO_LOADER_BACKEND=yasa \
    rekaai/vllm-omni-fork:latest \
    vllm serve rekaai/reka-edge-2603 \
        --served-model-name reka-edge-2603 \
        --tokenizer-mode yasa \
        --gpu-memory-utilization 0.95 \
        --max-model-len 8192 \
        --max-num-batched-tokens 20000 \
        --limit-mm-per-prompt '{"image": 6, "video": 3}' \
        --media-io-kwargs '{"video": {"num_frames": 6, "sampling": "chunk"}}' \
        --tensor-parallel-size 1 \
        --dtype bfloat16 \
        --chat-template-content-format openai \
        --enable-auto-tool-choice \
        --tool-call-parser hermes \
        --trust-remote-code \
        --quantization bitsandbytes

# In another shell — health check
curl http://localhost:8000/health
# → 200 OK

# Send a completion request to verify the model loads and responds
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"reka-edge-2603","messages":[{"role":"user","content":"Hello!"}]}'
# → should return a JSON response with a non-empty assistant message

# vllm-omni diffusion mode (marey) — single-stage pipeline, not --stage-configs-path.
# The MODEL dir must contain config.yaml, ema_inference_ckpt.safetensors, and vae.ckpt.
# Make sure to APPEND `:/model` to the checkpoint path - this is needed for Docker to mount
# it to the /model inside the container, which is where vllm-omni expects it
docker run --gpus all --rm -p 8000:8000 \
    --shm-size=8g \
    -v /path/to/marey-distilled-0100:/model \
    -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    rekaai/vllm-omni-fork:latest \
    vllm serve /model --omni \
        --port 8000 \
        --model-class-name MareyPipeline \
        --flow-shift 3.0 \
        --gpu-memory-utilization 0.98 \
        --ulysses-degree 8
```

If `/health` returns 200 in both modes, the image is good.

## 4. Push to Docker Hub

```bash
docker login                                           # Reka Docker Hub creds
docker push rekaai/vllm-omni-fork:$(git rev-parse HEAD)
docker push rekaai/vllm-omni-fork:latest
```

After the push, grab the image's content digest — this is what you'll pin
in tanka.

```bash
docker inspect --format='{{index .RepoDigests 0}}' rekaai/vllm-omni-fork:$(git rev-parse HEAD)
# → rekaai/vllm-omni-fork@sha256:abcdef012345...
```

## 5. Update tanka and deploy

Edit the relevant jsonnet environment (e.g. `infra/tanka/environments/oracle-ashburn-h100/inference.jsonnet`) and pin the image by `@sha256:` — not by tag — for reproducibility.

```jsonnet
local inference = import 'inference.libsonnet';

{
  world_model_marey: inference.vllm_omni_inference_server(
    name='world-model-marey',
    // Pin by content digest — covers vllm-omni + vllm-reka + moonvalley_ai.
    image='rekaai/vllm-omni-fork@sha256:abcdef012345...',
    gpu_count=8,
    replicas=1,
    // rclone pulls this dir from OCI models bucket into /model. It must
    // contain config.yaml, ema_inference_ckpt.safetensors, and vae.ckpt.
    model='marey/distilled-0100',
    override_entrypoint=[
      'vllm', 'serve', '/model',
      '--omni',
      '--port', '8000',
      '--model-class-name', 'MareyPipeline',
      '--flow-shift', '3.0',
      '--gpu-memory-utilization', '0.98',
      '--ulysses-degree', '8',
    ],
    env=[
      { name: 'PYTORCH_CUDA_ALLOC_CONF', value: 'expandable_segments:True' },
    ],
    instance_type=null,
    instance_product_type='NVIDIA-H100-80GB',
    shm_size='64Gi',
  ),
}
```

Then deploy through the normal tanka flow (see the infra repo's deploy guide for cluster-specific steps):

```bash
cd infra/tanka
tk apply environments/oracle-ashburn-h100
```

## Rollback

Rollbacks are a one-line jsonnet revert to a previous `@sha256:` digest, followed by `tk apply`. Keep the previous digest in commit history so this is a pure `git revert` + apply.

## FAQ

**Why not just use `:latest`?**
Tag mutation is the thing that ruins reproducibility. `@sha256:` pins the exact image bytes that served traffic when the deployment was last applied. Every prod pod should reference a digest.

**Why does the image bundle `moonvalley_ai`?**
Marey's VAE loader (`vllm_omni/diffusion/models/marey/pipeline_marey.py`) imports `opensora.models.vae.vae_adapters`. `opensora` isn't pip-installable cleanly (its `__init__` pulls in heavy optional deps), so the code stubs
parts of it at runtime and loads the VAE submodule from a sys.path entry.

The image sets `MOONVALLEY_AI_PATH=/workspace/moonvalley_ai` so the lookup succeeds.

**How do I build for a different GPU architecture (e.g. L40S)?**
No change — the base image covers modern NVIDIA archs (sm_70 through sm_90).

Just deploy to a node with that GPU type.

**Which file do I edit to change the build?**
- `docker/Dockerfile.serving` — the layers.
- `docker/build-context.dockerignore` — what gets sent to the daemon.
- `docker/build-serving.sh` — the wrapper that enforces sibling layout.

  MODEL=/app/hf_checkpoints/marey-distilled-0100 MOONVALLEY_AI_PATH=/app/kwa/moonvalley_ai ./vllm-omni/examples/online_serving/marey/run_server.sh
