# AURA Omni Native Pipeline

`aura_omni` serves AURA as a native multi-stage vLLM-Omni pipeline:

```text
Qwen3-ASR -> AURA/Qwen3-VL -> Qwen3-TTS Talker -> Qwen3-TTS Code2Wav
```

The pipeline has three semantic modules, but four engine stages because the
existing Qwen3-TTS implementation is natively split into Talker and Code2Wav.

Start the server with the deploy profile:

```bash
vllm serve aurateam/AURA \
  --omni \
  --port 8091 \
  --deploy-config vllm_omni/deploy/aura_omni.yaml \
  --trust-remote-code
```

The deploy file sets per-stage model repos:

- Stage 0 ASR: `Qwen/Qwen3-ASR-1.7B`
- Stage 1 AURA: `aurateam/AURA`
- Stage 2/3 TTS: `Qwen/Qwen3-TTS-12Hz-1.7B-Base`

For local weights, edit the `model` value on each stage in
`vllm_omni/deploy/aura_omni.yaml`. The deploy profile includes
`pipeline: aura_omni`, so the server uses this four-stage topology even when
the command-line model path points at one component checkpoint.

Expected request shape:

- Send microphone audio as the Stage 0 multimodal audio input.
- Include video frames in the original request `multi_modal_data`; the
  `asr2aura` processor carries them forward to AURA.
- Optional `additional_information` keys:
  - `aura_system_prompt`
  - `tts_language`
  - `tts_speaker`
  - `tts_instruct`
  - `tts_task_type`

If AURA emits `<|silent|>`, the `aura2tts` processor returns no TTS request, so
the TTS stages are skipped for that turn.

## Python Client

```bash
python examples/online_serving/aura_omni/openai_chat_completion_client.py \
  --host localhost \
  --port 8091 \
  --model aurateam/AURA \
  --modalities text,audio
```

Use local media:

```bash
python examples/online_serving/aura_omni/openai_chat_completion_client.py \
  --audio-path /path/to/input.wav \
  --video-path /path/to/video.mp4 \
  --output-dir output_aura_omni_online
```

## Curl

```bash
cd examples/online_serving/aura_omni
bash run_curl_multimodal_generation.sh
```

Set `PORT`, `MODEL`, or `OUTPUT_DIR` to override defaults:

```bash
PORT=8666 MODEL=/data/models/AURA bash run_curl_multimodal_generation.sh
```

## Gradio

Launch the server and Gradio UI together:

```bash
cd examples/online_serving/aura_omni
bash run_gradio_demo.sh
```

If the server is already running:

```bash
python examples/online_serving/aura_omni/gradio_demo.py \
  --model aurateam/AURA \
  --api-base http://localhost:8091/v1
```

## Offline

For offline inference, see
[`examples/offline_inference/aura_omni`](../../offline_inference/aura_omni/).
