# vLLM-Omni · MiniCPM-o Online Demo

Gradio-based web UI for **MiniCPM-o 4.5** and **MiniCPM-o 2.6** served via
`vllm-omni`'s OpenAI-compatible endpoints.

The UI supports:

- **Inputs**: text prompt + optional image, audio (file or mic), video.
- **Outputs**: text + speech (WAV player).
- **Model switch**: dropdown to toggle between the 4.5 and 2.6 endpoints.

## 1. Start the backend server(s)

### MiniCPM-o 4.5

```bash
vllm serve openbmb/MiniCPM-o-4_5 --omni \
    --deploy-config vllm_omni/deploy/minicpmo_4_5_8x4090.yaml \
    --trust-remote-code \
    --host 0.0.0.0 --port 8099
```

### MiniCPM-o 2.6

```bash
vllm serve /path/to/MiniCPM-o-2_6 --omni \
    --deploy-config vllm_omni/deploy/minicpmo_2_6_8x4090.yaml \
    --trust-remote-code \
    --host 0.0.0.0 --port 8091
```

## 2. Launch the Gradio demo

```bash
# Both endpoints auto-detected if present:
bash examples/online_serving/minicpmo/run_gradio_demo.sh

# Or run the python entry point directly:
python examples/online_serving/minicpmo/gradio_demo.py \
    --minicpmo45-api-base http://localhost:8099/v1 \
    --minicpmo45-model openbmb/MiniCPM-o-4_5 \
    --minicpmo26-api-base http://localhost:8091/v1 \
    --minicpmo26-model /path/to/MiniCPM-o-2_6 \
    --port 7862
```

Open `http://<host>:7862` in a browser.

## Notes

- **TTS trigger** differs between versions:
  - **4.5**: sets `extra_body.chat_template_kwargs.use_tts_template=True`
  - **2.6**: sets `extra_body.mm_processor_kwargs.use_tts=True`
- Uncheck **"Generate speech output (TTS)"** to get text-only responses (faster).
- Audio output sample rate is 24 kHz.
- Video input is forwarded as a base64 `video_url` entry; the server needs
  decord/torchvision to decode it.
