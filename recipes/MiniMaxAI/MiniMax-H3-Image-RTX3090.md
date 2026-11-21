# MiniMax-H3-Image for text-to-image on two RTX 3090 GPUs

This recipe runs native 2048x2048 text-to-image generation on two 24 GiB
RTX 3090 GPUs with NVLink. It uses the released six-step schedule, tensor
parallelism, CPU offload, and exact reuse without quantization.

The historical performance campaign predates the branch rebase. A functional
smoke test also passed on the rebased implementation, as described below.
The historical results do not establish current performance or qualify a fresh
installation.

## Summary

- Vendor: MiniMax
- Model: [MiniMaxAI/MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image)
- Task: text-to-image through `POST /v1/images/generations`
- Hardware: 2x NVIDIA GeForce RTX 3090, 24 GiB each, NVLink NV4
- Profile: DiT TP2, text-encoder TP2, USP1, DP1, one request at a time
- Precision: native checkpoint precision, BF16 DiT and FP32 image VAE
- Maintainer: barkly1te

## Supported model contract

| Setting | This deployment profile |
| --- | --- |
| Input | Text prompt, without reference images |
| Output | One 2048x2048 PNG, returned as `data[0].b64_json` |
| Generation canvas | Native 2048x2048, `min_generation_area=4194304` |
| Sampling | Six steps, `guidance_scale=1` or omitted, no negative prompt |
| Noise | `generator_device=cpu`, fixed seed for comparisons |
| Concurrency | One request at a time |
| Offload | Rank-local DiT and text-encoder layers, 48 resident DiT cores |

The model supports editing and other sizes, but this RTX 3090 profile has not
been qualified for them. See the [model contract](./MiniMax-H3-Image-H200.md#supported-model-contract)
for the broader interface. A 1024x1024 response uses a larger canvas and a final
resize by default, so it is not the native 2048x2048 workload measured here.

## Hardware and software

The historical run used two 24 GiB RTX 3090 GPUs joined by NVLink NV4 and a
host with about 376 GiB total RAM. That host size is an observation, not an
established minimum. CPU offload needs host copies and pinned staging memory
for both the DiT and text encoder. TP2 uses each rank's loader-produced layout;
it does not provide the TP1 shared-checkpoint mmap memory guarantee. Leave room
for loading, the OS, and filesystem cache. Check available RAM before startup.

| Component | Historical environment |
| --- | --- |
| Python | 3.12.15 |
| NVIDIA driver | 580.65.06 |
| PyTorch | 2.14.0+cu130 |
| vLLM | 0.33.0+precompiled, source `4ba787ec9959e566d3b587a796dff07673324da5` |
| Transformers | 5.19.0 |
| vLLM-Omni baseline | `0b74da131f963ab2fd60f7eba813145672cf81e4`, plus the optimization patch |

The existing environment was reused. A clean environment rebuild was not part
of that measurement.

## Prepare the source and runtime

Use the `codex/h3-image-3090-opt` branch of the `barkly1te/vllm-omni` fork.
Its optimizations are required by the options below. The branch is rebased on
Omni `80f1ae5fb6a5e85ad58235cd10eb7a05d2082118`; a released Omni wheel alone
does not select this source.

```bash
git clone --branch codex/h3-image-3090-opt \
  https://github.com/barkly1te/vllm-omni.git
cd vllm-omni
git rev-parse HEAD
uv venv --python 3.12 --seed
source .venv/bin/activate
```

Follow the [CUDA source installation guide](../../docs/getting_started/installation/gpu.md#build-from-source)
for this checkout's matching vLLM 0.33.0, PyTorch, driver, and cuDNN runtime.
Install Omni from this checkout with `uv pip install -e .`, and keep the
guide's cuDNN library exports in the server shell. Do not use the published
Omni 0.32.0 installation commands for this branch. Finish with:

```bash
uv pip check
.venv/bin/python -c 'import vllm_omni; print(vllm_omni.__file__)'
```

The printed module path must be inside this checkout. The remaining commands
run from its root. They also require the Hugging Face `hf` CLI, `curl`, `jq`,
and `base64`.

## Prepare local weights

Use a directory on your model disk. The image checkpoint plus its external
text encoder occupy about 126 GiB before cache or temporary copies. Obtain
the model under its license with your authorized Hugging Face account.

```bash
export MODEL_ROOT="${MODEL_ROOT:-$HOME/models}"
export MODEL_DIR="$MODEL_ROOT/MiniMax-H3-Image"
export ENCODER_ROOT="$MODEL_ROOT/MiniMax-H3/FL2VA"

hf download MiniMaxAI/MiniMax-H3-Image \
  --revision 7b3d20b4659f5d4b3ad7205daa008e47d6b1e868 \
  --local-dir "$MODEL_DIR"
hf download MiniMaxAI/MiniMax-H3 \
  --revision 42ed227ee7df40d41602854ae760620d6eb651fe \
  --include 'FL2VA/text_encoder/*' \
  --local-dir "$MODEL_ROOT/MiniMax-H3"

test -f "$ENCODER_ROOT/text_encoder/config.json"
if [ ! -e "$MODEL_DIR/text_encoder" ]; then
  ln -s "$ENCODER_ROOT/text_encoder" "$MODEL_DIR/text_encoder"
fi
test -f "$MODEL_DIR/text_encoder/model.safetensors.index.json"
```

These pinned download revisions follow the [existing AMD weight setup](./MiniMax-H3-Image-AMD.md#runtime-and-model-preparation).
The image loader prefers the local `text_encoder/` directory, including a
symlink, over the external repository named in `model_index.json`. Keep the
image model index, `pdd_config.json`, fused heads, tokenizer, processor,
transformer, and VAE together. No model-index rewrite is needed.

## Start the server

Select two free GPUs and an unused port. This command matches the measured
profile. Keep its default attention and VAE decode policies when reproducing
the historical results.

```bash
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

vllm serve "$MODEL_DIR" --omni \
  --host 0.0.0.0 --port 8091 \
  --served-model-name minimax-h3-image \
  --num-gpus 2 --tensor-parallel-size 2 --usp 1 \
  --text-encoder-tp-size 2 \
  --diffusion-offload-config '{"mode":"layer","components":["dit","text_encoder"],"layer_options":{"dit":{"resident_layers":48}}}' \
  --cache-config '{"minimax_h3_adaln_cache":true,"minimax_h3_adaln_offload":true}' \
  --additional-config '{"min_generation_area":4194304,"minimax_h3_retain_resident_cores":true,"minimax_h3_reuse_refined_text":true}'
```

The positive resident-layer count selects DLO. Both components use the default
`rank-local` transfer. The exact AdaLN cache keeps projection masters on CPU
and stages them on cache misses. Retained cores keep the 48 resident core
allocations on GPU between requests, including during text encoding and VAE
decode. Text refinement reuse keeps the first result only within one denoise
invocation. All six steps and every main DiT block still run.

Keep base weights immutable while offload is enabled. Disable offload before
changing weights, then enable it again. This combination requires native CUDA
Flash Attention, TP2, USP1, DP1, and unquantized weights. HWR, AllGather, and
HSDP are outside this profile. See [exact AdaLN staging](../../docs/user_guide/diffusion/offloader/distributed_layerwise_offload.md#minimax-h3-image-exact-adaln-staging)
for the compatibility rules.

## Generate and check an image

After loading finishes, run these commands in another terminal:

```bash
curl --fail --silent --show-error http://127.0.0.1:8091/health
curl --fail --silent --show-error http://127.0.0.1:8091/v1/models

curl --fail --silent --show-error \
  http://127.0.0.1:8091/v1/images/generations \
  -H 'Content-Type: application/json' \
  --data-binary '{
    "model": "minimax-h3-image",
    "prompt": "A photorealistic wide view of California coast, turquoise ocean, golden cliffs, clear afternoon sunlight, detailed coastal vegetation.",
    "size": "2048x2048",
    "seed": 42,
    "generator_device": "cpu",
    "num_inference_steps": 6,
    "n": 1,
    "response_format": "b64_json",
    "output_format": "png"
  }' --output response.json

jq --exit-status --raw-output '.data[0].b64_json' response.json \
  | base64 --decode > image.png

.venv/bin/python - <<'PY'
from PIL import Image

with Image.open("image.png") as image:
    image.load()
    assert image.format == "PNG"
    assert image.size == (2048, 2048)
    assert any(low != high for low, high in image.convert("RGB").getextrema())
print("Valid nonconstant 2048x2048 PNG")
PY
```

Record the first HTTP request separately. The service performs an internal
dummy warmup before it becomes ready. For warm measurements, send three
full-size requests before timing further requests, as in the historical run.
Monitor both GPUs during startup, the first request, warm requests, and VAE
decode. For example, in a separate terminal:

```bash
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total \
  --format=csv --loop=1
```

## Historical qualification

The author recorded these local results on Omni `0b74da131` plus the final
optimization patch. The rebased branch has not repeated this campaign.
Raw experiment artifacts are not published with this recipe.

| Comparison | Baseline | Optimized |
| --- | ---: | ---: |
| Warm HTTP latency, median of five trial means | 36.585 s | 14.609 s |
| First HTTP latency after dummy warmup, median of five trials | 36.836 s | 27.804 s |
| Tuned baseline with 29 resident layers, median of five trial means | 28.631 s | 14.577 s |

Each five-trial comparison used three full-size warmup requests and six fixed
prompt and seed pairs per service. A separate comparison used 20 pairs after
three warmups. Across all final comparisons, 226 HTTP requests succeeded and
113 paired decoded RGB outputs matched pixel for pixel. This is equality on
the tested corpus, not a blinded semantic quality evaluation.

Timing covered POST submission through receipt of the full HTTP response,
including server encoding. It excluded client decoding, saving, loading, and
warmups. The historical bundle contained 54 assets, including the linked
text encoder, with local manifest identity
`local-sha256:420b91a89a72f3ca001dab8567030e342b0bddac51e06ad5ba23c1e9774dd073`.
That local identity is not a public model revision or a claim that a new
download has been compared with the bundle.

The optimized requests reached a sampled whole-device peak of 22,874 MiB,
leaving 1,702 MiB on the more occupied GPU. Nominal one-second polling had a
maximum observed gap of 8.441 seconds. These values describe observed memory,
not a continuous peak bound. Sampling can miss short allocations, so do not
treat the observed headroom as a guarantee for another host or workload.

### Rebased implementation smoke test

On November 12, 2026, a local service smoke test used implementation commit
`9ca183980555c0730088d4db9d9dc99402ee4b10` with documentation-only cleanup,
the existing runtime, and the same local model assets. The server selected
Flash Attention and rank-local DLO with 48 resident DiT cores on each GPU.

One full-size warmup and two measured requests completed successfully at
2048x2048, six steps, and concurrency one. The two prompts used seeds 42 and
46. All three PNG images were valid, nonconstant, and RGB-identical to their
corresponding saved historical baseline images. The service released both
GPUs after the check.

This check covers service startup, first and repeated requests, and a prompt
change. It does not repeat the five-trial performance campaign or the
20-prompt quality comparison. Raw smoke artifacts remain local. A fresh
installation, fresh weight downloads, other resolutions, concurrent requests,
editing, and other hardware require separate validation.

## References

- [Canonical model card and download layout](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image#download)
- [H200 model contract and other deployment profiles](./MiniMax-H3-Image-H200.md)
- [Distributed layerwise offloading](../../docs/user_guide/diffusion/offloader/distributed_layerwise_offload.md)
- [Images API](../../docs/serving/image_generation_api.md)
- [Shared text-to-image client](../../examples/online_serving/text_to_image/openai_chat_client.py), which does not expose `generator_device`; use the JSON request above for this workload
