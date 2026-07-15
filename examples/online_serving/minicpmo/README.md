# vLLM-Omni · MiniCPM-o 4.5 Online Demo

Gradio-based web UI for **MiniCPM-o 4.5** served via `vllm-omni`'s
OpenAI-compatible endpoints.

The UI supports:

- **Inputs**: text prompt + optional image, audio (file or mic), video.
- **Outputs**: text + speech (WAV player).

## 1. Start the backend server

The deploy config auto-loads via `--omni`; the default
`vllm_omni/deploy/minicpmo_4_5.yaml` targets a single-GPU layout (thinker
and talker + t2w co-located on GPU 0).  For other hardware layouts pick
one of the deploy variants below.

| deploy config | GPUs | Notes |
|---|---|---|
| `minicpmo_4_5.yaml` (default) | 1 | Thinker and talker+t2w co-located on GPU0. |
| `minicpmo_4_5_2gpu.yaml` | 2 | Thinker on GPU0, talker+t2w on GPU1. |
| `minicpmo_4_5_3gpu.yaml` | 3 | Thinker 2-way TP on GPU0/1, talker+t2w share GPU2. |
| `minicpmo_4_5_8x4090.yaml` | 8 | Full 8x4090 layout. |

Default (single-GPU):

```bash
vllm serve openbmb/MiniCPM-o-4_5 --omni \
    --trust-remote-code \
    --host 0.0.0.0 --port 8099
```

Other layouts via `--deploy-config`:

```bash
vllm serve openbmb/MiniCPM-o-4_5 --omni \
    --deploy-config vllm_omni/deploy/minicpmo_4_5_8x4090.yaml \
    --trust-remote-code \
    --host 0.0.0.0 --port 8099
```

## 2. Launch the Gradio demo

```bash
bash examples/online_serving/minicpmo/run_gradio_demo.sh

# Or run the python entry point directly:
python examples/online_serving/minicpmo/gradio_demo.py \
    --minicpmo45-api-base http://localhost:8099/v1 \
    --minicpmo45-model openbmb/MiniCPM-o-4_5 \
    --port 7862
```

Open `http://<host>:7862` in a browser.

## Notes

- **TTS trigger**: the demo sets
  `extra_body.chat_template_kwargs.use_tts_template=True`, which appends
  `<|tts_bos|>` to the assistant prefix.
- Uncheck **"Generate speech output (TTS)"** to get text-only responses
  (faster).
- The audio output is the raw WAV returned by the stage-1 talker +
  Token2Wav; sample rate is 24 kHz.
- Video input is forwarded as a base64 `video_url` entry; the server needs
  decord/torchvision to decode it.
