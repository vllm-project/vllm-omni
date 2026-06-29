# Qwen3-Omni

## Setup
Please refer to the [stage configuration documentation](https://docs.vllm.ai/projects/vllm-omni/en/latest/configuration/stage_configs/) to configure memory allocation appropriately for your hardware setup.

## Run examples

### Multiple Prompts
Get into the example folder
```bash
cd examples/offline_inference/qwen3_omni
```
Then run the command below. Note: for processing large volume data, it uses py_generator mode, which will return a python generator from Omni class.
```bash
bash run_multiple_prompts.sh
```
### Single Prompt
Get into the example folder
```bash
cd examples/offline_inference/qwen3_omni
```
Then run the command below.
```bash
bash run_single_prompt.sh
```
If you have not enough memory, you can set thinker with tensor parallel. Just run the command below.
```bash
bash run_single_prompt_tp.sh
```

### Modality control
If you want to control output modalities, e.g. only output text, you can run the command below:
```bash
python end2end.py --output-wav output_audio \
                  --query-type use_audio \
                  --modalities text
```

#### Using Local Media Files
The `end2end.py` script supports local media files (audio, video, image) via command-line arguments:

```bash
# Use local video file
python end2end.py --query-type use_video --video-path /path/to/video.mp4

# Use local image file
python end2end.py --query-type use_image --image-path /path/to/image.jpg

# Use local audio file
python end2end.py --query-type use_audio --audio-path /path/to/audio.wav

# Combine multiple local media files
python end2end.py --query-type mixed_modalities \
    --video-path /path/to/video.mp4 \
    --image-path /path/to/image.jpg \
    --audio-path /path/to/audio.wav
```

If media file paths are not provided, the script will use default assets. Supported query types:
- `use_video`: Video input
- `use_image`: Image input
- `use_audio`: Audio input
- `text`: Text-only query
- `multi_audios`: Multiple audio inputs
- `mixed_modalities`: Combination of video, image, and audio inputs

### Async-chunk (offline)

For true stage-level concurrency -- where downstream stages (Talker, Code2Wav)
start **before** the upstream stage (Thinker) finishes -- use the async_chunk
example. This requires:

1. A deploy config YAML with ``async_chunk: true`` (e.g.
   ``qwen3_omni_moe.yaml``).
2. Hardware that matches the config (e.g. 2x H100 for the default 3-stage
   config).

The async_chunk example uses ``AsyncOmni`` instead of the synchronous ``Omni``
class, which enables the async orchestrator to receive stage-0 intermediate
outputs and trigger downstream stages early. Chunk data flows directly between
stage workers via the in-worker ``OmniChunkTransferAdapter`` / connector,
**not** through the orchestrator.

#### Single prompt
```bash
cd examples/offline_inference/qwen3_omni
bash run_single_prompt_async_chunk.sh
```

#### Multiple prompts with concurrency control
```bash
bash run_multiple_prompts_async_chunk.sh --max-in-flight 4
```

#### Text-only output (skip audio generation)
```bash
python end2end_async_chunk.py --query-type text --modalities text
```

#### Custom stage config
```bash
python end2end_async_chunk.py \
    --query-type use_audio \
    --deploy-config /path/to/your_deploy_config.yaml
```

> **Note**: The synchronous ``end2end.py`` (using ``Omni``) is still the
> recommended entry point for non-async-chunk workflows. Only use the
> async_chunk example when you need the stage-level concurrency semantics
> described in PR #962 / #1151.

## Ray batch inference (Thinking)

**`batch_inference_ray.py`** uses **Ray Data** and Ray's
`vLLMEngineProcessorConfig` to run a **single vLLM engine inside a Ray actor**.

### Setup

Use a **vLLM-Omni** environment (e.g. `vllm/vllm-omni-rocm:v0.20.0`).

The stock `vllm-omni-rocm` images do not ship Ray Data LLM; Thus, we need to install the following inside a container.

```bash
pip uninstall -y uvloop 2>/dev/null || true
pip install -U "ray[default,data]"
pip install --no-cache-dir grpcio pyarrow pandas datasets hydra-core omegaconf
pip install -U qwen-omni-utils librosa soundfile
pip install --no-cache-dir "uvloop>=0.21"
```

**Tensor parallelism:** the Thinking audio encoder has 20 attention heads, thus **TP must divide 20** (e.g. TP=2/4 on stock vllm/vllm-omni-rocm:v0.20.0; TP=8 require additional patches). We have created the following PRs to fix this:

- [vllm-project/vllm#45900](https://github.com/vllm-project/vllm/pull/45900) — Fix Qwen3-vLLM audio encoder TP when heads are not divisible by TP size
- [vllm-project/vllm-omni#4322](https://github.com/vllm-project/vllm-omni/pull/4322) — Fix Qwen3-Omni audio encoder TP when heads are not divisible by TP size



### Run

From this directory:

```bash
cd examples/offline_inference/qwen3_omni

python batch_inference_ray.py \
  model=qwen3_omni_30b_a3b_thinking \
  dataset=random_mm \
  query_type=mixed \
  num_prompts=1024 \
  input_len=1024 \
  output_len=4096 \
  batch_size=256 \
  n_repeats=2 \
  vllm.tensor_parallel_size=4 \
  vllm.gpu_memory_utilization=0.7 \
  vllm.max_model_len=16384 \
  vllm.max_num_seqs=512 \
  vllm.max_num_batched_tokens=65536 \
  vllm.max_concurrent_batches=8 \
  vllm.enable_expert_parallel=true
```
