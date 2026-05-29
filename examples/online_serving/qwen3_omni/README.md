# Qwen3-Omni

## 🛠️ Installation

Please refer to [README.md](../../../README.md)

## Run examples (Qwen3-Omni)

### Launch the Server

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091
```

The default deployment configuration, situated at `vllm_omni/deploy/qwen3_omni_moe.yaml`, is resolved and loaded
automatically via the model registry, obviating the `--deploy-config` flag in standard deployment topologies.
Asynchronous chunk streaming operates as **enabled by default** within this bundled configuration.
Additionally, NPU, ROCm, and XPU per-platform configuration deltas are deterministically merged from the
`platforms`: section of the corresponding YAML.

**Note:** The OpenAI-style **`/v1/realtime`** WebSocket interface (facilitating streaming PCM audio input alongside audio and transcription output)
is currently **unsupported** while the `async_chunk` configuration attribute is enabled.
It is requisite to instantiate the default omni architecture or utilize a deployment configuration specifying `async_chunk: false` to facilitate real-time streaming sessions.

To explicitly utilize a custom deployment YAML, mandate the configuration path accordingly:
```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091 \
    --deploy-config /path/to/your_deploy_config.yaml
```

For the bundled 3x-GPU multi-replica layout (talker/code2wav scale-out),
use:

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091 \
    --deploy-config vllm_omni/deploy/qwen3_omni_moe_multi_replicas.yaml
```

### Launch individual stages (stage-based CLI)

Use the stage-based CLI when you want to run one stage per process.
The example below pins Stage 0 to GPU 0 and Stage 1/2 to GPU 1 via
`CUDA_VISIBLE_DEVICES`.

**1. Stage 0 (Thinker + API server)**

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni \
    --port 8091 \
    --stage-id 0 \
    --omni-master-address 127.0.0.1 \
    --omni-master-port 26000
```

**2. Stage 1 (Talker)**

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni \
    --stage-id 1 \
    --headless \
    --omni-master-address 127.0.0.1 \
    --omni-master-port 26000
```

**3. Stage 2 (Code2Wav)**

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni \
    --stage-id 2 \
    --headless \
    --omni-master-address 127.0.0.1 \
    --omni-master-port 26000
```

Append `--deploy-config /path/to/your_deploy_config.yaml` to each node invocation if it is necessary
to explicitly override the bundled deployment YAML schema.

For standard **unified-process** launcher, stage-specific CLI configuration tuning is conventionally implemented
via the `--stage-overrides` directive, as demonstrated below:

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091 \
    --stage-overrides '{"1": {"gpu_memory_utilization": 0.5}}'
```

Conversely, within the stage-based CLI paradigm, `--stage-overrides` modifiers are typically **unnecessary**
for this category of optimization. Given that each instantiation strictly initiates a single functional stage,
parameter flags can be systematically assigned directly onto that specific stage's command sequence:

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni \
    --stage-id 1 \
    --headless \
    --gpu-memory-utilization 0.5 \
    --omni-master-address 127.0.0.1 \
    --omni-master-port 26000
```

### Tuning deployment parameters

Most engine knobs (`max_num_batched_tokens`, `max_model_len`, `enforce_eager`,
`gpu_memory_utilization`, `tensor_parallel_size`, …) can be tuned without
editing the YAML. There are three layers, in increasing specificity:

#### 1. Global CLI flags (apply to every stage)

```bash
# Tighter memory budget on a smaller GPU
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091 \
    --gpu-memory-utilization 0.85

# Disable cudagraphs (e.g. for debugging)
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091 \
    --enforce-eager

# Reduce context length
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091 \
    --max-model-len 32768

# Toggle prefix caching on every stage (yaml default: off)
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091 \
    --enable-prefix-caching
# ...or force it off if the yaml turned it on
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091 \
    --no-enable-prefix-caching

# Toggle pipeline-wide async chunked streaming between stages
# (yaml default for qwen3_omni_moe: on)
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091 \
    --no-async-chunk
```

For the TTS counterpart (synchronous codec variant), see the Qwen3-TTS
section of the online TTS hub:
[examples/online_serving/text_to_speech/README.md#qwen3-tts](../text_to_speech/README.md#qwen3-tts).

Explicit CLI flags **override** the deploy YAML (which itself overrides the
parser defaults). If you don't pass a flag, the YAML value wins.

> **Note on `--no-async-chunk`**: Flips the deploy yaml's `async_chunk:`
> bool. Pipelines that implement alternate processor functions for
> chunked vs end-to-end modes (e.g. qwen3_tts code2wav) dispatch
> automatically based on that bool — no extra flag or variant yaml is
> needed.

> ⚠️ **For multi-stage models that share GPUs (qwen3_omni_moe by default
> shares cuda:1 between stages 1 and 2), avoid using global memory flags.**
> A global `--gpu-memory-utilization 0.85` would apply to every stage and
> oversubscribe the shared device. Use per-stage overrides instead — see
> below.

#### 2. Per-stage overrides via `--stage-overrides` (recommended for memory)

```bash
# Lower stage 1's memory budget; leave others at the YAML default
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091 \
    --stage-overrides '{
        "1": {"gpu_memory_utilization": 0.5},
        "2": {"max_num_batched_tokens": 65536}
    }'
```

Per-stage values are always treated as explicit and beat YAML defaults for
the named stage. Other stages keep their YAML values.

If you switch to the stage-based CLI, the same per-stage tuning can usually be
passed directly on that stage's command instead of using `--stage-overrides`.

#### 3. Custom deploy YAML

When per-stage overrides get long, write a small overlay YAML that inherits
from the bundled default:

```yaml
# my_qwen3_omni_overrides.yaml
base_config: /path/to/vllm_omni/deploy/qwen3_omni_moe.yaml

stages:
  - stage_id: 0
    max_num_batched_tokens: 65536
    enforce_eager: true
  - stage_id: 1
    gpu_memory_utilization: 0.5
  - stage_id: 2
    max_model_len: 8192
```

Then start the server with `--deploy-config my_qwen3_omni_overrides.yaml`.
The `base_config:` line tells the loader to inherit everything else (stages,
connectors, edges, platforms section) from the bundled production YAML, so
you only need to spell out the deltas.

#### 4. Multi-node deployment (cross-host transfer connector)

The bundled `qwen3_omni_moe.yaml` uses `SharedMemoryConnector` between stages,
which only works when all stages run on the same physical host. For
**cross-node** deployments, write a small overlay YAML that swaps in a
network-capable connector (e.g. `MooncakeStoreConnector`) and re-points each
stage's connector wiring at it. The connector spec carries your own server
addresses — there is no checked-in default because every cluster is
different.

```yaml
# my_qwen3_omni_multinode.yaml
base_config: /path/to/vllm_omni/deploy/qwen3_omni_moe.yaml

connectors:
  mooncake_connector:
    name: MooncakeStoreConnector
    extra:
      host: "127.0.0.1"
      metadata_server: "http://YOUR_METADATA_HOST:8080/metadata"
      master: "YOUR_MASTER_HOST:50051"
      segment: 512000000    # 512 MB transfer segment
      localbuf: 64000000    # 64 MB local buffer
      proto: "tcp"

stages:
  - stage_id: 0
    output_connectors:
      to_stage_1: mooncake_connector
  - stage_id: 1
    input_connectors:
      from_stage_0: mooncake_connector
    output_connectors:
      to_stage_2: mooncake_connector
  - stage_id: 2
    input_connectors:
      from_stage_1: mooncake_connector
```

Then launch with `--deploy-config my_qwen3_omni_multinode.yaml`. Same
pattern works for Qwen2.5-Omni — replace `base_config:` with the path to
`vllm_omni/deploy/qwen2_5_omni.yaml`.

> ⚠️ Replace `YOUR_METADATA_HOST` / `YOUR_MASTER_HOST` with the actual
> mooncake server addresses for your cluster. The `base_config:` overlay
> inherits all stage budgets, devices, and edges from the bundled prod
> YAML — you only need to spell out the connector swap.

### Send Multi-modal Request

Get into the example folder
```bash
cd examples/online_serving/qwen3_omni
```

####  Send request via python

```bash
python examples/online_serving/openai_chat_completion_client_for_multimodal_generation.py --model Qwen/Qwen3-Omni-30B-A3B-Instruct --query-type use_image --port 8091 --host "localhost"
```

#### Realtime WebSocket client (`openai_realtime_client.py`)

[`openai_realtime_client.py`](./openai_realtime_client.py) connects to **`ws://<host>:<port>/v1/realtime`**, streams a local WAV as **PCM16 mono @ 16 kHz** in fixed-size chunks (OpenAI-style `input_audio_buffer.append` / `commit`), and receives **`response.audio.delta`** (incremental PCM for the reply) plus **`transcription.*`** events. By default it concatenates audio deltas and writes **`--output-wav`** (model output is typically **24 kHz**). Optional **`--delta-dump-dir`** saves each delta as `delta_000001.wav`, … for debugging.

Streaming input works well for translation-style use cases; if the Thinker runs while input is still incomplete, consider limiting **`max_tokens`** in your session / server defaults to avoid over-generation.

**Dependencies:**

```bash
pip install websockets
```

**From this directory** (`examples/online_serving/qwen3_omni`):

```bash
python openai_realtime_client.py \
  --url ws://localhost:8091/v1/realtime \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --input-wav /path/to/input_16k_mono.wav \
  --output-wav realtime_output.wav \
  --delta-dump-dir ./rt_delta_wavs
```

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--url` | `ws://localhost:8091/v1/realtime` | Full WebSocket URL including path |
| `--model` | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | Must match the served model (sent in `session.update`) |
| `--input-wav` | *(required)* | Input WAV: mono, 16-bit PCM, **16 kHz** |
| `--output-wav` | `realtime_output.wav` | Output path for concatenated reply audio |
| `--output-text` | *(optional)* | If set, write final transcription text to this path |
| `--chunk-ms` | `200` | Size of each uploaded audio chunk (milliseconds of audio) |
| `--send-delay-ms` | `0` | Delay between chunk sends (simulate realtime upload) |
| `--delta-dump-dir` | *(optional)* | Directory to write per-`response.audio.delta` WAV files |
| `--num-requests` | `1` | Number of sequential sessions (see `--concurrency`) |
| `--concurrency` | `1` | Max concurrent WebSocket sessions when `--num-requests` > 1 |

Ensure the server is running **without** `async_chunk` if you use `/v1/realtime`, for example:

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091
```

The Python client supports the following command-line arguments:

- `--query-type` (or `-q`): Query type (default: `use_video`). Options: `text`, `use_audio`, `use_image`, `use_video`
- `--model` (or `-m`): Model name/path (default: `Qwen/Qwen3-Omni-30B-A3B-Instruct`)
- `--video-path` (or `-v`): Path to local video file or URL. If not provided and query-type is `use_video`, uses default video URL. Supports local file paths (automatically encoded to base64) or HTTP/HTTPS URLs. Example: `--video-path /path/to/video.mp4` or `--video-path https://example.com/video.mp4`
- `--image-path` (or `-i`): Path to local image file or URL. If not provided and query-type is `use_image`, uses default image URL. Supports local file paths (automatically encoded to base64) or HTTP/HTTPS URLs and common image formats: JPEG, PNG, GIF, WebP. Example: `--image-path /path/to/image.jpg` or `--image-path https://example.com/image.png`
- `--audio-path` (or `-a`): Path to local audio file or URL. If not provided and query-type is `use_audio`, uses default audio URL. Supports local file paths (automatically encoded to base64) or HTTP/HTTPS URLs and common audio formats: MP3, WAV, OGG, FLAC, M4A. Example: `--audio-path /path/to/audio.wav` or `--audio-path https://example.com/audio.mp3`
- `--prompt` (or `-p`): Custom text prompt/question. If not provided, uses default prompt for the selected query type. Example: `--prompt "What are the main activities shown in this video?"`
- `--speaker`: TTS speaker/voice for audio output when requesting audio (e.g. `ethan`, `chelsie`, `aiden`). Omit to use the model default. Example: `--speaker "chelsie"`


For example, to use a local video file with custom prompt:

```bash
python examples/online_serving/openai_chat_completion_client_for_multimodal_generation.py \
    --query-type use_video \
    --video-path /path/to/your/video.mp4 \
    --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --prompt "What are the main activities shown in this video?"
```

####  Send request via curl

```bash
bash run_curl_multimodal_generation.sh use_image
```


### FAQ

## Modality control
You can control output modalities to specify which types of output the model should generate. This is useful when you only need text output and want to skip audio generation stages for better performance.

### Supported modalities

| Modalities | Output |
|------------|--------|
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
curl -s http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "messages": [{"role": "user", "content": "Describe vLLM in brief."}],
    "modalities": ["audio"]
  }' | jq -r '.choices[0].message.audio.data' | base64 -d > output.wav
```

### Using Python client

```bash
python examples/online_serving/openai_chat_completion_client_for_multimodal_generation.py \
    --query-type use_image \
    --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
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
import base64
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8091/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
    messages=[{"role": "user", "content": "Describe vLLM in brief."}],
    modalities=["text", "audio"]
)
# Response contains two choices: one with text, one with audio
print(response.choices[0].message.content)  # Text response

# Save audio to file
audio_data = base64.b64decode(response.choices[1].message.audio.data)
with open("output.wav", "wb") as f:
    f.write(audio_data)
```

## Speaker selection

When requesting audio output, you can choose the TTS speaker (voice) used for synthesis. If not specified, the model uses its default speaker.

### Using curl

Pass a `speaker` field in the request body:

```bash
curl http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "modalities": ["audio"],
    "speaker": "chelsie"
  }'
```

### Using Python client

Use the `--speaker` argument when generating audio:

```bash
python examples/online_serving/openai_chat_completion_client_for_multimodal_generation.py \
    --query-type use_image \
    --modalities audio \
    --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --speaker "chelsie"
```

### Using OpenAI Python SDK

Pass `speaker` in `extra_body`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8091/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
    modalities=["audio"],
    extra_body={"speaker": "chelsie"}
)
# Audio uses the specified speaker
print(response.choices[1].message.audio)
```

Supported speaker names depend on the model (e.g. `Ethan`, `Chelsie`, `Aiden`). Omit `speaker` to use the default.

## Streaming Output
If you want to enable streaming output, please set the argument as below. The final output will be obtained just after generated by corresponding stage. We support both text streaming output and audio streaming output. Other modalities can output normally.
```bash
python examples/online_serving/openai_chat_completion_client_for_multimodal_generation.py \
    --query-type use_image \
    --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --stream
```

## Run Local Web UI Demo

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

**Step 1: Launch the vLLM API server**

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091
```

If you have custom stage configs file:
```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091 --deploy-config /path/to/deploy_config_file
```

**Step 2: Run the Gradio demo**

In a separate terminal:

```bash
python gradio_demo.py --model Qwen/Qwen3-Omni-30B-A3B-Instruct --api-base http://localhost:8091/v1 --port 7861
```

Then open `http://localhost:7861/` on your local browser to interact with the web UI.

The gradio script supports the following arguments:

- `--model`: Model name/path (should match the server model)
- `--api-base`: Base URL for the vLLM API server (default: http://localhost:8091/v1)
- `--ip`: Host/IP for Gradio server (default: 127.0.0.1)
- `--port`: Port for Gradio server (default: 7861)
- `--share`: Share the Gradio demo publicly (creates a public link)
