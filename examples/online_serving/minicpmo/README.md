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

The deploy config auto-loads via `--omni`.
The default `vllm_omni/deploy/minicpmo_4_5.yaml` keeps all three stages on
logical device 0 with memory budgets of 65%, 15%, and 15%. For throughput,
`minicpmo_4_5_batching.yaml` gives the Thinker GPU 0 (90%) and colocates the
Talker (55%) and Code2Wav (35%) on GPU 1. Each stage admits at most four
concurrent sequences.

| deploy config | GPUs | Notes |
|---|---|---|
| `minicpmo_4_5.yaml` (default) | 1 | Memory-constrained compatibility layout. |
| `minicpmo_4_5_batching.yaml` | 2 | Recommended continuous-batching layout; Talker and Code2Wav share GPU 1. |
| `minicpmo_4_5_2gpu.yaml` | 2 | Backward-compatible alias for `minicpmo_4_5_batching.yaml`. |
| `minicpmo_4_5_3gpu.yaml` | 3 | One GPU per stage. |
| `minicpmo_4_5_8x4090.yaml` | 8 | Full 8x4090 layout. |
| `minicpmo_4_5_duplex.yaml` | 1 | Experimental native full-duplex overlay. |

The split pipeline preserves native-duplex epoch/turn identity, segment text,
turn completion, reference voice, and terminal-audio metadata through
Code2Wav. Focused CPU regressions cover this envelope; run the Realtime
scenario below for live barge-in validation on the target GPU.

Default:

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

### Per-stage overrides

```bash
vllm serve openbmb/MiniCPM-o-4_5 --omni --trust-remote-code --port 8099 \
    --stage-overrides '{"0": {"gpu_memory_utilization": 0.65}}'
```

## Send multimodal requests

```bash
cd examples/online_serving/minicpmo
```

### curl

```bash
bash run_curl_multimodal_generation.sh text
bash run_curl_multimodal_generation.sh use_image
bash run_curl_multimodal_generation.sh use_audio '["text"]'

python openai_chat_completion_client_for_multimodal_generation.py \
    --query-type use_image \
    --port 8099 \
    --host localhost

# Text-only (faster; no <|tts_bos|>)
python openai_chat_completion_client_for_multimodal_generation.py \
    --query-type text \
    --modalities text \
    --prompt "Briefly introduce yourself."
```

Streaming text + audio:

```bash
python streaming_chat_completion.py \
    --base-url http://localhost:8099/v1 \
    --output minicpmo_stream.wav
```

The example prints text deltas immediately and joins the independently encoded
24 kHz WAV audio deltas into one valid WAV file. Add `--text-only` to skip
Talker and Code2Wav.

Shared helpers also work if you pass MiniCPM defaults yourself:

```bash
python ../openai_chat_completion_client_for_multimodal_generation.py \
    --model openbmb/MiniCPM-o-4_5 \
    --query-type text \
    --port 8099
```

Speech output no longer depends on a MiniCPM-specific default in the generic
serving layer. `chat_template_kwargs.use_tts_template=true` remains an
explicitly supported model option.

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

## Daily-Omni accuracy

Daily-Omni requires one A–D letter. Set
`chat_template_kwargs.enable_thinking=false` explicitly in `--extra-body`;
the generic benchmark CLI does not inject a MiniCPM-specific default.
Leaving reasoning enabled can exhaust the 256-token answer budget inside
`<think>` and make first-letter extraction score reasoning text.

For the established text benchmark, send
`--extra_body '{"modalities":["text"],"chat_template_kwargs":{"enable_thinking":false}}'`.
Requesting audio also benchmarks Talker and Code2Wav and changes the
assistant template, so it is not an apples-to-apples accuracy run.

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

## Pipeline notes

- Stage 1 performs request-owned AR continuous batching. Stage 2 keeps
  request-owned Flow/HiFT caches and batches exact-shape-compatible chunks.
- Reference audio travels with the first codec chunk; Stage 2 owns its
  temporary prompt WAV and evicts prompt features when the request finishes.
- Codec sampling reads the checkpoint `tts_config` (default deterministic
  seed 42). Stage-1 YAML sampling parameters govern only the binary
  continue/stop token exposed to vLLM.
- `StageRequestStats.batch_size` is request-scoped and does not report the
  scheduler's execution batch.
- Stage 0 and Stage 1 use vLLM CUDA Graph capture. Stage 2 remains eager until
  a dedicated exact-shape graph wrapper owns static I/O buffers and copies
  request cache state outside capture.
- Co-locating all three stages minimizes hardware requirements but makes their
  CUDA contexts contend for one GPU. Use the 8x4090 layout or a custom
  multi-GPU deploy config when throughput is the primary goal.
- Output audio is base64 WAV in `message.audio.data` (24 kHz mono).
- Offline counterpart:
  [`examples/offline_inference/minicpmo/`](../../offline_inference/minicpmo/)
- Recipe:
  [`recipes/OpenBMB/MiniCPM-o-4_5.md`](../../../recipes/OpenBMB/MiniCPM-o-4_5.md)
