# DreamZero-DROID

> Robot policy for the DROID embodiment, served over the OpenPI WebSocket protocol

## Summary

- Vendor: GEAR-Dreams
- Model: `GEAR-Dreams/DreamZero-DROID`
- Task: Vision-language-action inference for robot manipulation
- Mode: Online serving via OpenPI WebSocket endpoint
- Maintainer: Community

## When to use this recipe

Serve DreamZero-DROID as a real-time robot policy on AMD GPUs. DreamZero derives
its actions from a causal video world model, so it runs on the AR-Diffusion
engine, keeps per-session KV cache and needs a `session_id` on every
observation. The bundled deployment splits the classifier-free guidance branches
across two GPUs.

## References

- Upstream model: <https://huggingface.co/GEAR-Dreams/DreamZero-DROID>
- OpenPI client library: <https://github.com/Physical-Intelligence/openpi>
- Pipeline: `vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero.DreamZeroPipeline`
- Deploy configs: [`vllm_omni/deploy/dreamzero.yaml`](../../vllm_omni/deploy/dreamzero.yaml) (TP=1, no CFG parallel),
  [`vllm_omni/deploy/dreamzero_tp1_cfg2.yaml`](../../vllm_omni/deploy/dreamzero_tp1_cfg2.yaml) (TP=1, CFG parallel size 2)
- Examples and client: [`examples/online_serving/dreamzero/`](../../examples/online_serving/dreamzero/)
- E2E test: [`tests/e2e/online_serving/test_dreamzero_expansion.py`](../../tests/e2e/online_serving/test_dreamzero_expansion.py)

## Hardware Support

Other hardware is welcome as community validation lands.

## ROCm

### 2x AMD MI300X

#### Environment

- OS: Ubuntu 22.04.5 LTS, x86_64
- Python: 3.12
- ROCm / HIP: 7.2.53211
- PyTorch: 2.10.0+git8514f05
- vLLM version: 0.22.0+rocm722
- vLLM-Omni version or commit: 0.22.0+rocm, source tree at tag `v0.22.0`
- Docker image: `vllm/vllm-omni-rocm:v0.22.0`, digest
  `sha256:be06edae7ba9d14a89c5907da2e3631b7998765bce285c937f7d9a96e6aeb2c5`

#### Command

The v0.22.0 image does not ship `openpi-client`, which the endpoint needs, so
the command installs it before serving.

```bash
git clone --branch v0.22.0 --depth 1 https://github.com/vllm-project/vllm-omni.git
cd vllm-omni

docker run -d --name dreamzero --network host \
    --device /dev/kfd --device /dev/dri --group-add=video \
    --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --shm-size=32g \
    -v "$PWD":/work -w /work \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e HIP_VISIBLE_DEVICES=0,1 \
    -e OMP_NUM_THREADS=8 \
    -e ATTENTION_BACKEND=torch \
    -e DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA \
    -e VLLM_DISABLE_COMPILE_CACHE=1 \
    --entrypoint bash \
    vllm/vllm-omni-rocm:v0.22.0@sha256:be06edae7ba9d14a89c5907da2e3631b7998765bce285c937f7d9a96e6aeb2c5 \
    -c 'python -m pip install --no-deps openpi-client &&
        exec vllm serve GEAR-Dreams/DreamZero-DROID \
          --omni --host 127.0.0.1 --port 8091 \
          --served-model-name dreamzero-droid \
          --deploy-config vllm_omni/deploy/dreamzero_tp1_cfg2.yaml \
          --enforce-eager --disable-log-stats'
```

The server is ready when the log reaches `Application startup complete`.

#### Verification

Replay the example DROID clips against the running server:

```bash
python -m pip install --no-deps openpi-client

hf download YangshenDeng/vllm-omni-dreamzero-assets \
    --repo-type dataset --local-dir outputs/dreamzero/assets

python examples/online_serving/dreamzero/openpi_client.py \
    --host 127.0.0.1 --port 8091 --video-dir outputs/dreamzero/assets
```

```text
Server metadata: {"action_space": "joint_position", "image_resolution": [180, 320],
                  "n_external_cameras": 2, "needs_session_id": true,
                  "needs_stereo_camera": false, "needs_wrist_camera": true}
Action 0: shape=(24, 8) dtype=float32 min=-0.120401 max=0.540639
Action 1: shape=(24, 8) dtype=float32 min=-0.003906 max=0.421654
```

The e2e test covers the same contract plus a mid-session reset, and needs no
assets — it synthesizes the clips and starts its own server:

```bash
python -m pytest -q \
    tests/e2e/online_serving/test_dreamzero_expansion.py::test_dreamzero_openpi_online
```

It carries a CUDA hardware marker, so on ROCm select it by node id rather than
by marker expression.

#### Notes

- Memory usage: 42.79 GiB of weights per GPU — each CFG rank holds a full copy.
- Key flags: the two GPUs come from `cfg_parallel_size: 2`, which puts the
  conditional and unconditional branches on separate ranks; stage 0 `devices`
  must list exactly `tensor_parallel_size * cfg_parallel_size` entries.
  `--enforce-eager` overrides `enforce_eager: false` in the config.
- The WebSocket endpoint is `ws://127.0.0.1:8091/v1/realtime/robot/openpi`, not
  the root path. The root completes the handshake but returns error frames.
- Known limitations: `max_num_seqs: 1`, so one session at a time.
- `NVML init failed, will use profiling fallback` is expected on ROCm.
- Do not use the `v0.24.0` or `v0.24.1` ROCm images: their
  `dreamzero_tp1_cfg2.yaml` is missing `engine_backend` and startup fails.
  Fixed in v0.26.0, which has no ROCm image published yet.
