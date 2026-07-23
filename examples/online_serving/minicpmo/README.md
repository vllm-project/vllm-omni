# MiniCPM-o 4.5: Online serving

This directory contains MiniCPM-o 4.5 online serving demos for vLLM-Omni.
Inputs can include text, image, audio, or video; outputs are text and optional
24 kHz speech.

For the experimental native duplex runtime architecture, lifecycle invariants,
capability boundary, and validation scope, see
[`vllm_omni/experimental/fullduplex/DESIGN.md`](../../../vllm_omni/experimental/fullduplex/DESIGN.md).

## Installation

Install vLLM-Omni with the MiniCPM-o talker dependencies:

```bash
pip install 'vllm-omni[minicpmo]'

# From a source checkout:
pip install -e '.[minicpmo]'
```

The `minicpmo` extra installs `stepaudio2-minicpmo` and its audio dependencies,
including `librosa`.

## Start the backend server

Pick a deploy config that matches your GPU layout:

| config | GPUs | TP | Notes |
|---|---|---|---|
| `minicpmo_4_5.yaml` | 1 | 1 | Thinker and talker+t2w co-located on GPU0. |
| `minicpmo_4_5_2gpu.yaml` | 2 | 1 | Thinker on GPU0, talker+t2w on GPU1. |
| `minicpmo_4_5_3gpu.yaml` | 3 | 2 | Thinker 2-way TP on GPU0/1, talker+t2w share GPU2. |
| `minicpmo_4_5_8x4090.yaml` | 8 | 4 | Thinker 4-way TP on GPU0-3, talker+t2w on GPU4. |
| `minicpmo_4_5_3gpu_stage1_replicas.yaml` | 3 | 1 | Experimental validation profile: Thinker on GPU0, two talker+Token2wav replicas on GPU1/2. |
| `minicpmo_4_5_4gpu_stage1_replicas.yaml` | 4 | 1 | Experimental validation profile: Thinker on GPU0, three talker+Token2wav replicas on GPU1/2/3. |
| `minicpmo_4_5_8x4090_stage1_replicas.yaml` | 8 | 4 | Experimental validation profile: Thinker 4-way TP on GPU0-3, four talker+Token2wav replicas on GPU4-7. |
| `minicpmo_4_5_duplex.yaml` | 2 | 1 | Experimental native duplex profile. |

```bash
vllm-omni serve openbmb/MiniCPM-o-4_5 \
    --omni \
    --deploy-config vllm_omni/deploy/minicpmo_4_5.yaml \
    --trust-remote-code \
    --host 0.0.0.0 --port 8099
```

For local ModelScope checkpoints, replace `openbmb/MiniCPM-o-4_5` with the
checkpoint path. To start the experimental native duplex backend, use
`vllm_omni/deploy/minicpmo_4_5_duplex.yaml`.

The `*_stage1_replicas.yaml` files exercise composite Stage1 replica routing
and failure recovery. They are validation profiles, not recommended production
entrypoints.

## Send chat requests

From this directory, run the MiniCPM-specific curl or Python clients:

```bash
bash run_curl_multimodal_generation.sh text
bash run_curl_multimodal_generation.sh use_image
bash run_curl_multimodal_generation.sh use_audio '["text"]'

python openai_chat_completion_client_for_multimodal_generation.py \
    --query-type use_image \
    --host localhost \
    --port 8099
```

Speech output requires `chat_template_kwargs.use_tts_template=true`. Put that
field at the request root for curl; the OpenAI Python SDK can merge it from
`extra_body`.

## Launch the Gradio demo

```bash
bash examples/online_serving/minicpmo/run_gradio_demo.sh

# Or run the Python entry point directly:
python examples/online_serving/minicpmo/gradio_demo.py \
    --minicpmo45-api-base http://localhost:8099/v1 \
    --minicpmo45-model openbmb/MiniCPM-o-4_5 \
    --port 7862
```

Open `http://<host>:7862` in a browser.

## Run the Realtime duplex CLI demo

After the duplex backend is running, stream one WAV through the Realtime
WebSocket endpoint:

```bash
python examples/online_serving/minicpmo/realtime_duplex_demo.py \
    --url ws://localhost:8099/v1/realtime?duplex=1 \
    --model openbmb/MiniCPM-o-4_5 \
    --input-wav /path/to/input_16k_mono_pcm16.wav \
    --ref-audio /path/to/MiniCPM-o-Demo/assets/ref_audio/ref_minicpm_signature.wav \
    --output-dir /tmp/minicpmo_realtime_duplex_demo
```

## Open the experimental browser client

The browser UI serves the page and proxies the same-origin Realtime WebSocket to
the backend:

```bash
python -m examples.online_serving.minicpmo.realtime_web \
    --port 7862 \
    --ws-backend ws://127.0.0.1:8099 \
    --ref-audio /path/to/MiniCPM-o-Demo/assets/ref_audio/ref_minicpm_signature.wav
```

Open `http://<host>:7862/`. When using a reverse proxy, open the URL mapped to
port `7862`; the browser derives its WebSocket endpoint relative to that URL.

If the page proxy serves HTTP but does not forward WebSocket upgrades, point the
browser at a separately exposed Realtime endpoint:

```bash
python -m examples.online_serving.minicpmo.realtime_web \
    --port 7862 \
    --ws-backend ws://127.0.0.1:8099 \
    --public-realtime-url wss://public.example/v1/realtime
```

## Validate soft-interrupt behavior

The soft-interrupt E2E driver defaults to `--validation-mode model-policy`,
which checks lifecycle and streaming invariants for arbitrary input audio. The
stronger `response-required` mode is diagnostic: it requires a purpose-built
two-response WAV, its `--input-sha256`, and an
`--expect-second-response-substring` value.

## Related examples

- [Offline MiniCPM-o inference](../../offline_inference/minicpmo/)
- [MiniCPM-o 4.5 recipe](../../../recipes/OpenBMB/MiniCPM-o-4_5.md)
