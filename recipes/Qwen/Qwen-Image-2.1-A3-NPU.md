# Qwen-Image-2.1 and Qwen-Image-2.1-Turbo on Ascend A3

## Summary

`QwenImage21Pipeline` serves both checkpoints through the OpenAI-compatible
Images API. Each A3 card has two NPU dies with 64 GiB of HBM per die.
[Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) ·
[Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo)

For Atlas 800I A2, use the [A2 recipe](Qwen-Image-2.1-NPU.md).

## Ascend A3 performance

| NPU dies | Physical cards | Ulysses degree | Base P50 (s) | Turbo P50 (s) | Base peak extra HBM (GiB/die) | Turbo peak extra HBM (GiB/die) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 4 | 8 | 1.83 | 1.06 | 37.23 | 37.93 |
| 4 | 2 | 4 | 2.65 | 1.36 | 37.32 | 37.32 |
| 2 | 1 | 2 | 4.19 | 1.94 | 37.24 | 37.24 |
| 1 | 1 (one die) | 1 | 6.34 | 2.72 | 36.67 | 36.66 |

The base checkpoint ran with `num_inference_steps=20`. Turbo used its fixed
eight-step distilled schedule. Both used BF16 eager mode,
`true_cfg_scale=1.0`, and seed 42 with a CPU generator. Multi-die runs used
Ulysses sequence parallelism with replicated weights. Some base runs
overlapped checkpoint downloads to shared storage.

Measured on an Atlas 900 A3 SuperPoD compute node with an arm64 host,
CANN 9.3.0, Ascend driver 26.1.1, PyTorch 2.10.0, and torch-npu
2.10.0.post7. The vLLM-Omni image was `v0.34.0-a3` with OCI index
`sha256:584e105786b352505427ef81b9080b90267e12114462221f3db41b7988eb044b`.

At concurrency 1, the client decoded each response as one 1024×1024 PNG.
Each profile ran two rounds of one warm-up and five timed requests. Round 1
used `p20` for warm-up and `p01` through `p05` for timing. Round 2 used
`p19` for warm-up and `p06` through `p10` for timing.
[Prompt corpus](../../benchmarks/diffusion/h3-image-prompts.jsonl)

P50 is the median of ten request latencies measured through receipt of the
full HTTP response. Peak extra HBM is the largest increase over each die's
pre-launch reading. The sampler waited three seconds between `npu-smi info` calls.

## Checkpoint

Download the checkpoint into a persistent model directory on the host.

```bash
export MODEL_ROOT=/path/to/models
modelscope download --model Qwen/Qwen-Image-2.1 \
  --local_dir "${MODEL_ROOT}/Qwen-Image-2.1" --max-workers 8
modelscope download --model Qwen/Qwen-Image-2.1-Turbo \
  --local_dir "${MODEL_ROOT}/Qwen-Image-2.1-Turbo" --max-workers 8
```

Use the same Diffusers layout for Hugging Face downloads. Each directory needs
`model_index.json` plus the text encoder, DiT, and VAE weights.

## Environment

Use `quay.io/ascend/vllm-omni:v0.34.0-a3` with an installed Ascend driver.
See the [NPU installation guide](../../docs/getting_started/installation/npu.md)
for the host requirements.

### Start the container

Check device use with `npu-smi info`. Use `npu-smi info -m` to map logical
IDs to physical cards. For 2, 4, or 8 dies, select both dies on each card.

```bash
export IMAGE=quay.io/ascend/vllm-omni:v0.34.0-a3
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11
IFS=, read -r -a dies <<< "${ASCEND_RT_VISIBLE_DEVICES}"
device_args=()
for die in "${dies[@]}"; do
  device_args+=(--device "/dev/davinci${die}")
done

docker run --rm -it --init --privileged --network host --shm-size 16g \
  "${device_args[@]}" \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v "${MODEL_ROOT}:/models:ro" \
  -e ASCEND_RT_VISIBLE_DEVICES -e "NPU_DIES=${#dies[@]}" \
  "${IMAGE}" bash
```

## Command

Run this in the container. Set `MODEL_ID` to `Qwen/Qwen-Image-2.1-Turbo`
to serve Turbo.

```bash
export MODEL_ID=Qwen/Qwen-Image-2.1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_TARGET_DEVICE=npu
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_NPU_SOCKET_PORT_RANGE=auto

vllm serve "/models/${MODEL_ID##*/}" --omni \
  --host 127.0.0.1 --port 18089 --served-model-name "${MODEL_ID}" \
  --dtype bfloat16 --enforce-eager --trust-remote-code --max-num-seqs 1 \
  --num-gpus "${NPU_DIES}" --usp "${NPU_DIES}" --ulysses-mode strict
```

## Request and verification

Set the same `MODEL_ID` in a second shell.

```bash
set -o pipefail
curl -fsS http://127.0.0.1:18089/v1/models | jq -r '.data[].id'

sampling='{"true_cfg_scale":1.0,"num_inference_steps":20}'
if [[ "${MODEL_ID}" == Qwen/Qwen-Image-2.1-Turbo ]]; then
  sampling='{"true_cfg_scale":1.0}'
fi
jq -nc --arg model "${MODEL_ID}" --argjson sampling "${sampling}" \
  '{model:$model,prompt:"A ceramic teapot on a wooden table",size:"1024x1024",
    n:1,seed:42,generator_device:"cpu",response_format:"b64_json",
    output_format:"png"} + $sampling' \
  | curl -fsS http://127.0.0.1:18089/v1/images/generations \
      -H 'Content-Type: application/json' --data-binary @- \
  | jq -er '.data[0].b64_json' | base64 -d > qwen-image.png

python - <<'PYTHON'
from PIL import Image

with Image.open("qwen-image.png") as image:
    image.load()
    assert image.format == "PNG" and image.size == (1024, 1024)
PYTHON
```

For image-editing requests, see the
[Qwen-Image-2.1 recipe](Qwen-Image-2.1.md#online-serving).
