# Qwen3-Omni: Online serving

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/qwen3_omni>.

## 🛠️ Installation

Please refer to [README.md](https://github.com/vllm-project/vllm-omni/tree/main/README.md)

## Run examples (Qwen3-Omni)

### Launch the Server

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091
```

For a local deployment with multiple API frontend processes sharing one set
of stage engines, add `--api-server-count`:

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni \
    --api-server-count 2 --port 8091
```

This mode supports local EngineCore stages with one local process group per
replica, including stages configured with multiple `num_replicas`. Headless,
remote, diffusion, intra-replica data-parallel, and Ray stage deployments
remain single-frontend.

The default deployment configuration situated at `vllm_omni/deploy/qwen3_omni_moe.yaml` is resolved and loaded
automatically via the model registry, obviating the necessity for the `--deploy-config` flag in standard deployment topologies.
The bundled Qwen3-Omni setup explicitly sets `moe_backend: triton` for the Thinker and Talker stages. This keeps
both stages on vLLM's Triton unquantized MoE path and avoids the unstable FlashInfer CUTLASS unquantized MoE path on
some GPUs.
Asynchronous chunk streaming is **enabled by default** within the bundled configuration.
On CUDA and NPU the same file selects model runner v2 for all three stages. The Thinker prefix cache is enabled on CUDA; NPU keeps it disabled by default (set `enable_prefix_caching: true` on stage 0 to opt in). Talker and Code2Wav keep it disabled.
The NPU Thinker and Talker use an Ascend model state for attention metadata while retaining Omni request lifecycle and M-RoPE handling. Code2Wav uses Ascend input batches and forward context with the shared V2 chunk lifecycle. It runs without outer graph padding and copies audio to host synchronously before publishing each output. Code2Wav captures up to 32 exact-shape NPU graphs at startup for its streaming decode windows; other shapes run eagerly and nothing is captured while serving. Set `enforce_eager: true` on stage 2 to disable them.
XPU, ROCm and MUSA stay on model runner v1. Set `model_runner: v1` in an overlay to opt out on CUDA or NPU.

On B200/B300-class GPUs (SM100/SM103) with cuDNN 9.20.0, Qwen3-Omni
automatically runs Code2Wav in FP32 when its configured precision resolves to
BF16 or FP16. This avoids incorrect low-precision convolutions that can make
streaming audio quiet or distorted at certain decode window lengths
([#8935](https://github.com/vllm-project/vllm-omni/issues/8935)). The fallback
also covers the optional first-frame decoder inside the Talker; Thinker and
Talker generation retain their configured precision. It takes effect before
weight loading and CUDA Graph capture, and logs a warning at startup.

The automatic guard matches **exactly cuDNN 9.20.0** (runtime version `92000`),
the version reported in the issue. Other versions retain the configured
precision; this is not a claim that all other versions are unaffected. The
issue reports successful tests with 9.21.0.82 and 9.24.0.43, but NVIDIA's
[9.21 release notes](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.21.0/release-notes.html)
do not explicitly identify this Conv1d numerical issue as fixed.

The fallback also overrides an **explicit** BF16/FP16 stage dtype. To retain
the requested precision, set the boolean `codec_cudnn_fp32_fallback: false`
in the connector's `extra` settings. For example, save this as `deploy.yaml`
and launch with `--deploy-config deploy.yaml`:

```yaml
base_config: /absolute/path/to/vllm_omni/deploy/qwen3_omni_moe.yaml
connectors:
  connector_of_shared_memory:
    extra:
      codec_cudnn_fp32_fallback: false
```

Use the MRv2 deployment as `base_config` if that is your existing profile.
The bundled profiles share this connector definition, so the setting reaches
both the Code2Wav stage and the Talker's optional decoder copy. For custom
profiles with different connectors, set it on the incoming connector used by
each of those stages. Disabling the fallback can reintroduce the reported
audio corruption; only do so after validating your stack and decode windows.

FP32 increases Code2Wav memory use; latency depends on the selected kernels.
The Instruct checkpoint's 216,016,577 Code2Wav parameters occupy about 412 MiB
in BF16/FP16 and 824 MiB in FP32, an increase of 412 MiB **per decoder copy**.
The default profile loads one copy. Profiles enabling Talker first-frame audio
load a second copy, adding another 412 MiB to the Talker process. Activations,
cuDNN workspace and CUDA Graph pools need additional memory; these weight-only
figures do not establish the peak memory or that a deployment fits its budgets.

For releases without this automatic fallback, the default separate Code2Wav
stage can be run in FP32 with
`--stage-overrides '{"2": {"dtype": "float32"}}'`.

For Thinker pipeline parallelism, configure `pipeline_parallel_size` and `devices`
on the Thinker stage. Keep Talker and Code2Wav at `pipeline_parallel_size: 1`. The
default asynchronous scheduling and audio chunk streaming can remain enabled.

To explicitly utilize a custom deployment YAML, specify the configuration path:

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091 \
    --deploy-config /path/to/deploy_config_file
```

To serve **thinker-only** (text output, no talker / code2wav loaded) with Instruct
weights, pass the bundled thinker-only deploy YAML. The `pipeline:` key
`qwen3_omni_moe_thinker_only` overrides the HF `enable_audio_output` resolver:

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091 \
    --deploy-config vllm_omni/deploy/qwen3_omni_moe_thinking.yaml
```

Captioner / Thinking checkpoints (`enable_audio_output=false`) still auto-select
the same single-stage pipeline without `--deploy-config`.

### Thinking checkpoint chat template

Serving `Qwen/Qwen3-Omni-30B-A3B-Thinking` with function tools through
`/v1/chat/completions` or the turn-based `/v1/realtime` endpoint requires the
compatibility chat template bundled with vLLM-Omni. From a source checkout:

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Thinking --omni --port 8091 \
    --chat-template vllm_omni/transformers_utils/chat_templates/qwen3_omni_thinking.jinja \
    --reasoning-parser qwen3 \
    --tool-call-parser hermes \
    --enable-auto-tool-choice
```

For an installed package, substitute the path to `qwen3_omni_thinking.jinja`
inside its `vllm_omni/transformers_utils/chat_templates` directory.

### Launch individual stages (stage-based CLI)

Adopt the stage-based CLI architecture to independently instantiate execution processes per functional stage.
The example below pins Stage 0 to GPU 0 and Stage 1/2 to GPU 1 via
`CUDA_VISIBLE_DEVICES`.

#### 1. Stage 0 (Thinker + API server)

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni \
    --port 8091 \
    --stage-id 0 \
    --omni-master-address 127.0.0.1 \
    --omni-master-port 26000
```

#### 2. Stage 1 (Talker)

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni \
    --stage-id 1 \
    --headless \
    --omni-master-address 127.0.0.1 \
    --omni-master-port 26000
```

#### 3. Stage 2 (Code2Wav)

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni \
    --stage-id 2 \
    --headless \
    --omni-master-address 127.0.0.1 \
    --omni-master-port 26000
```

Add `--deploy-config /path/to/deploy_config_file` to every command if you want
to override the bundled deploy YAML.

For the regular one-process launch, stage-specific CLI tuning is usually done
with `--stage-overrides`, for example:

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091 \
    --stage-overrides '{"1": {"gpu_memory_utilization": 0.5}}'
```

To experiment with another supported MoE backend, set `moe_backend` in the
deployment YAML for both the Thinker and Talker stages. For the stage-based
CLI, pass `--moe-backend` directly on each stage command.

For other stage settings, pass the option on that stage command. For example,
to change its GPU memory utilization:

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni \
    --stage-id 1 \
    --headless \
    --gpu-memory-utilization 0.5 \
    --omni-master-address 127.0.0.1 \
    --omni-master-port 26000
```

### Send Multi-modal Request

Get into the example folder

```bash
cd examples/online_serving/qwen3_omni
```

#### Send request via python

```bash
python openai_chat_completion_client_for_multimodal_generation.py --query-type use_image --port 8091 --host "localhost"
```

The Python client supports the following command-line arguments:

- `--query-type` (or `-q`): Query type (default: `use_video`). Options: `text`, `use_audio`, `use_image`, `use_video`
- `--model` (or `-m`): Model name/path (default: `Qwen/Qwen3-Omni-30B-A3B-Instruct`)
- `--video-path` (or `-v`): Path to local video file or URL. If not provided and query-type is `use_video`, uses default video URL. Supports local file paths (automatically encoded to base64) or HTTP/HTTPS URLs. Example: `--video-path /path/to/video.mp4` or `--video-path https://example.com/video.mp4`
- `--image-path` (or `-i`): Path to local image file or URL. If not provided and query-type is `use_image`, uses default image URL. Supports local file paths (automatically encoded to base64) or HTTP/HTTPS URLs and common image formats: JPEG, PNG, GIF, WebP. Example: `--image-path /path/to/image.jpg` or `--image-path https://example.com/image.png`
- `--audio-path` (or `-a`): Path to local audio file or URL. If not provided and query-type is `use_audio`, uses default audio URL. Supports local file paths (automatically encoded to base64) or HTTP/HTTPS URLs and common audio formats: MP3, WAV, OGG, FLAC, M4A. Example: `--audio-path /path/to/audio.wav` or `--audio-path https://example.com/audio.mp3`
- `--prompt` (or `-p`): Custom text prompt/question. If not provided, uses default prompt for the selected query type. Example: `--prompt "What are the main activities shown in this video?"`

For example, to use a local video file with custom prompt:

```bash
python openai_chat_completion_client_for_multimodal_generation.py \
    --query-type use_video \
    --video-path /path/to/your/video.mp4 \
    --prompt "What are the main activities shown in this video?"
```

#### Send request via curl

```bash
bash run_curl_multimodal_generation.sh use_image
```

## Modality control

You can control output modalities to specify which types of output the model should generate. This is useful when you only need text output and want to skip audio generation stages for better performance.

### Supported modalities

| Modalities | Output |
| ------------ | -------- |
| `["text"]` | Text only |
| `["audio"]` | Audio only |
| `["text", "audio"]` | Text + Audio |
| Not specified | Text + Audio (default) |

### Using curl

#### Text only

```bash
curl http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "messages": [{"role": "user", "content": "Describe vLLM in brief."}],
    "modalities": ["text"]
  }'
```

#### Text + Audio

```bash
response=$(curl -s http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "messages": [{"role": "user", "content": "Describe vLLM in brief."}],
    "modalities": ["text", "audio"]
  }')

echo "$response" | jq -r '.choices[0].message.content'
echo "$response" | jq -r '.choices[1].message.audio.data' | base64 -d > output.wav
```

### Using Python client

```bash
python openai_chat_completion_client_for_multimodal_generation.py \
    --query-type use_image \
    --modalities text
```

### Using OpenAI Python SDK

#### Text only

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8091/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
    messages=[{"role": "user", "content": "Describe vLLM in brief."}],
    modalities=["text"]
)
print(response.choices[0].message.content)
```

#### Text + Audio

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8091/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
    messages=[{"role": "user", "content": "Describe vLLM in brief."}],
    modalities=["text", "audio"]
)
# Response contains two choices: one with text, one with audio
print(response.choices[0].message.content)  # Text response
print(response.choices[1].message.audio)    # Audio response
```

## Streaming Output

If you want to enable streaming output, please set the argument as below. The final output will be obtained just after generated by corresponding stage. We support both text streaming output and audio streaming output. Other modalities can output normally.

```bash
python openai_chat_completion_client_for_multimodal_generation.py \
    --query-type use_image \
    --stream
```

## Run Local Web UI Demo

!!! note "Gradio is an optional dependency"
    The Gradio demo requires the `[demo]` extras. Install them first:

    ```bash
    pip install 'vllm-omni[demo]'
    ```

    For this 0.33 source checkout, follow the
    [source demo dependency instructions](https://docs.vllm.ai/projects/vllm-omni/en/latest/getting_started/installation/#source-development-and-demo-dependencies)
    from the repository root. Build the temporary Gradio wheel and include it
    with `.[demo]` in the same normal install command.

This Web UI demo allows users to interact with the model through a web browser.

### Running Gradio Demo

The Gradio demo connects to a vLLM API server. You have two options:

#### Option 1: One-step Launch Script (Recommended)

The convenience script launches both the vLLM server and Gradio demo together:

```bash
./run_gradio_demo.sh --model Qwen/Qwen3-Omni-30B-A3B-Instruct --server-port 8091 --gradio-port 7861
```

This script will:

1. Start the vLLM server in the background
2. Wait for the server to be ready
3. Launch the Gradio demo
4. Handle cleanup when you press Ctrl+C

The script supports the following arguments:

- `--model`: Model name/path (default: Qwen/Qwen3-Omni-30B-A3B-Instruct)
- `--server-port`: Port for vLLM server (default: 8091)
- `--gradio-port`: Port for Gradio demo (default: 7861)
- `--deploy-config`: Path to custom deploy config YAML file (optional)
- `--server-host`: Host for vLLM server (default: 0.0.0.0)
- `--gradio-ip`: IP for Gradio demo (default: 127.0.0.1)
- `--share`: Share Gradio demo publicly (creates a public link)

#### Option 2: Manual Launch (Two-Step Process)

##### Step 1: Launch the vLLM API server

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091
```

If you have a custom deploy config file:

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091 --deploy-config /path/to/deploy_config_file
```

##### Step 2: Run the Gradio demo

In a separate terminal:

```bash
python gradio_demo.py --model Qwen/Qwen3-Omni-30B-A3B-Instruct --api-base http://localhost:8091/v1 --port 7861
```

Then open `http://localhost:7861/` on your local browser to interact with the web UI.

The gradio script supports the following arguments:

- `--model`: Model name/path (should match the server model)
- `--api-base`: Base URL for the vLLM API server (default: <http://localhost:8091/v1>)
- `--ip`: Host/IP for Gradio server (default: 127.0.0.1)
- `--port`: Port for Gradio server (default: 7861)
- `--share`: Share the Gradio demo publicly (creates a public link)

## Experimental DFlash Thinker decoding

DFlash integration currently covers CUDA model runner v1 (MRv1), with an
ordinary, non-speculative Talker and Code2Wav. The default deployment above
uses MRv2 and is not the validated DFlash configuration. When configuring
DFlash, explicitly select `model_runner: v1`, disable Thinker prefix caching
(`enable_prefix_caching: false` on stage 0), and use non-resumable requests.
Set stage 0 `pipeline_parallel_size: 1`: DFlash with pipeline parallelism is
rejected before loading the Thinker weights because auxiliary captures are
not configured on every pipeline rank. Tensor parallelism remains supported.

For DFlash speech output, set stage 0 `enforce_eager: true` and
`speculative_config.enforce_eager: true` so both the target Thinker and the
DFlash drafter run eagerly. **CUDA Graph DFlash speech is outside the
supported scope**, including with `disable_custom_all_reduce: true`: the
graph experiment reported an unresolved generated-speech ASR regression.
Disabling custom all-reduce addresses a startup failure, not speech quality.

For text-only graph experiments, set stage 0
`disable_custom_all_reduce: true`. With the pinned vLLM 0.33.0 runtime,
DFlash with tensor parallelism and default custom all-reduce can fail during
graph memory profiling before the service becomes ready. Eager execution
of both target and drafter avoids that graph startup path. These settings
are deployment requirements; the model does not enforce them at admission.

Eager execution also does not establish strict greedy equivalence:
baseline/DFlash token differences remain unexplained. Treat this integration
as experimental, and do not infer speech quality or output equivalence from
successful requests or throughput alone. See the
[validation results and limitations](../../../design/qwen3_omni_dflash_validation.md)
for the tested configuration, accuracy evidence, and separate eager/graph
performance results.

## Example materials

??? abstract "gradio_demo.py"
    ``````py
    --8<-- "examples/online_serving/qwen3_omni/gradio_demo.py"
    ``````
??? abstract "openai_chat_completion_client_for_multimodal_generation.py"
    ``````py
    --8<-- "examples/online_serving/qwen3_omni/openai_chat_completion_client_for_multimodal_generation.py"
    ``````
??? abstract "qwen3_omni_moe_thinking.yaml"
    ``````yaml
    --8<-- "vllm_omni/deploy/qwen3_omni_moe_thinking.yaml"
    ``````
??? abstract "run_curl_multimodal_generation.sh"
    ``````sh
    --8<-- "examples/online_serving/qwen3_omni/run_curl_multimodal_generation.sh"
    ``````
??? abstract "run_gradio_demo.sh"
    ``````sh
    --8<-- "examples/online_serving/qwen3_omni/run_gradio_demo.sh"
    ``````
