# vLLM-Omni · MiniCPM-o Online Demo

Gradio-based web UI for **MiniCPM-o 4.5** and **MiniCPM-o 2.6** served via
`vllm-omni`'s OpenAI-compatible endpoints.

The UI supports:

- **Inputs**: text prompt + optional image, audio (file or mic), video.
- **Outputs**: text + speech (WAV player).
- **Model switch**: dropdown to toggle between the 4.5 and 2.6 endpoints.

## 1. Start the backend server(s)

### MiniCPM-o 4.5 (this workspace)

```bash
# Edit CUDA_VISIBLE_DEVICES / STAGE_CFG to match your GPU layout.
bash /tmp/start_minicpmo45_server.sh            # serves http://0.0.0.0:8099
```

Stage configs available:

| config | GPUs | TP | Notes |
|---|---|---|---|
| `minicpmo45_2gpu.yaml` | 2 | 1 | Thinker on GPU0, talker+t2w on GPU1. |
| `minicpmo45_3gpu.yaml` | 3 | 2 | Thinker 2-way TP on GPU0/1, talker+t2w share GPU2. |
| `minicpmo45_8x4090.yaml` | 8 | - | Full 8x4090 layout. |

### MiniCPM-o 2.6 (optional)

Any vllm-omni OpenAI server for 2.6 works; by default the launcher looks for
one at `http://localhost:8091/v1`.

## 2. Launch the Gradio demo

```bash
# Both endpoints auto-detected if present:
bash examples/online_serving/minicpmo/run_gradio_demo.sh

# Or run the python entry point directly:
python examples/online_serving/minicpmo/gradio_demo.py \
    --minicpmo45-api-base http://localhost:8099/v1 \
    --minicpmo45-model /cache/caitianchi/model/MiniCPM-o-4_5_full \
    --minicpmo26-api-base http://localhost:8091/v1 \
    --minicpmo26-model ./MiniCPM-o-2_6 \
    --port 7862
```

Open `http://<host>:7862` in a browser.

## Notes

- **TTS trigger** differs between versions:
  - **4.5**: the demo sets `extra_body.chat_template_kwargs.use_tts_template=True`, which appends `<|tts_bos|>` to the assistant prefix.
  - **2.6**: the demo sets `extra_body.mm_processor_kwargs.use_tts=True`, which injects `<|spk_bos|><|spk|><|spk_eos|><|tts_bos|>`.
- Uncheck **"Generate speech output (TTS)"** to get text-only responses (faster).
- The audio output is the raw WAV returned by the stage-1 talker + Token2Wav; sample rate is 24 kHz for 4.5.
- Video input is forwarded as a base64 `video_url` entry; the server needs decord/torchvision to decode it.
