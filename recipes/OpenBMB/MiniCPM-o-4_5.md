# MiniCPM-o 4.5

> Online serving and offline inference for omni multimodal chat
> (text / image / audio / video → text + 24 kHz speech)

## Summary

- Vendor: OpenBMB
- Model: [`openbmb/MiniCPM-o-4_5`](https://huggingface.co/openbmb/MiniCPM-o-4_5)
- Task: Omni multimodal chat — accepts text / image / audio / video input;
  emits text and 24 kHz mono speech in the same response
- Mode: Online serving via the OpenAI-compatible `/v1/chat/completions`
  API (plus Gradio demo), and offline inference via `Omni.generate`
- Maintainer: [`@tc-mb`](https://github.com/tc-mb) (MiniCPM-V / MiniCPM-o team)

## When to use this recipe

Use this recipe as a known-good starting point for serving
`openbmb/MiniCPM-o-4_5` on vLLM-Omni. MiniCPM-o 4.5 is the omni member
of the MiniCPM-o family — it runs a multimodal thinker, a streaming
MiniCPMTTS codec talker, and a separate batched Code2Wav stage so a single
`/v1/chat/completions` call can return text and 24 kHz speech in one
shot. The recommended batching deploy isolates the Thinker on GPU 0 and
co-locates Talker and Code2Wav on GPU 1; 1-GPU, 3-GPU, and 8x4090 layouts are
also provided.

## References

- Default deploy configs (auto-loaded by HF `model_type=minicpmo` +
  `hf_config.version="4.5"`):
  - Default single-GPU compatibility layout (auto-loaded):
    [`vllm_omni/deploy/minicpmo_4_5.yaml`](../../vllm_omni/deploy/minicpmo_4_5.yaml)
  - Recommended 2-GPU continuous-batching layout:
    [`vllm_omni/deploy/minicpmo_4_5_batching.yaml`](../../vllm_omni/deploy/minicpmo_4_5_batching.yaml)
  - 2-GPU and 3-GPU layouts:
    [`vllm_omni/deploy/minicpmo_4_5_2gpu.yaml`](../../vllm_omni/deploy/minicpmo_4_5_2gpu.yaml),
    [`vllm_omni/deploy/minicpmo_4_5_3gpu.yaml`](../../vllm_omni/deploy/minicpmo_4_5_3gpu.yaml)
  - 8x RTX 4090 layout:
    [`vllm_omni/deploy/minicpmo_4_5_8x4090.yaml`](../../vllm_omni/deploy/minicpmo_4_5_8x4090.yaml)
- Online example + Gradio demo:
  [`examples/online_serving/minicpmo/`](../../examples/online_serving/minicpmo/)
- Offline end-to-end example:
  [`examples/offline_inference/minicpmo/`](../../examples/offline_inference/minicpmo/)
- Pipeline / talker source:
  [`vllm_omni/model_executor/models/minicpmo_4_5/`](../../vllm_omni/model_executor/models/minicpmo_4_5/)
- Stage-input processors (thinker → talker and talker → Code2Wav):
  [`vllm_omni/model_executor/stage_input_processors/minicpmo_4_5_omni.py`](../../vllm_omni/model_executor/stage_input_processors/minicpmo_4_5_omni.py)
- Upstream model card:
  [`openbmb/MiniCPM-o-4_5`](https://huggingface.co/openbmb/MiniCPM-o-4_5)
- Integration PR:
  [vllm-project/vllm-omni#3642](https://github.com/vllm-project/vllm-omni/pull/3642)

## Hardware Support

Four hardware layouts ship with deploy configs. Every layout uses the
same strict three-stage topology. The Talker emits codec chunks only;
Code2Wav consumes them through a shared-memory async connector.

| Layout | Thinker | Talker | Code2Wav | Typical hardware |
| --- | --- | --- | --- | --- |
| 1-GPU (default) | GPU 0 | GPU 0 | GPU 0 | 1x large-memory GPU |
| 2-GPU | GPU 0 | GPU 1 | GPU 1 | 2x large-memory GPU |
| 3-GPU | GPU 0 | GPU 1 | GPU 2 | 3x GPU |
| 8x RTX 4090 24GB | GPU 0–3 (TP=4) | GPU 4 | GPU 5 | 8x RTX 4090 consumer |

### Migration from the fused deployment

MiniCPM-o 4.5 now requires the three-stage topology: the Talker owns
request-local codec generation and Code2Wav owns waveform state and
reference-voice prompt features. `minicpmo_4_5.yaml` remains the stable
single-GPU entry point; `minicpmo_4_5_batching.yaml` is the recommended
two-GPU profile. The removed fused two-stage implementation is not retained as
a fallback because it would duplicate state machines and correctness paths.

## GPU

### 1 x GPU (default — single command)

The default
[`vllm_omni/deploy/minicpmo_4_5.yaml`](../../vllm_omni/deploy/minicpmo_4_5.yaml)
co-locates Thinker, codec-only Talker, and Code2Wav on GPU 0. Their
`gpu_memory_utilization` budgets are 0.65, 0.15, and 0.15. This layout
minimizes the GPU count; use a large-memory accelerator and leave the
remaining 5% for runtime overhead.

#### Environment

- OS: Linux
- Python: 3.10+
- vLLM / vLLM-Omni: >= 0.21.0 (or current `main`)
- Optional Talker dep: `stepaudio2-minicpmo` (see Notes for why this is
  required and how to install it)

#### Command

```bash
vllm serve openbmb/MiniCPM-o-4_5 --omni \
    --trust-remote-code \
    --host 0.0.0.0 --port 8099
```

The deploy config is auto-loaded by the model registry — no
`--deploy-config` flag needed for this default single-GPU layout.

For the recommended two-GPU layout, add:

```bash
--deploy-config vllm_omni/deploy/minicpmo_4_5_batching.yaml
```

#### Performance comparison

Compare text-only and text+audio separately. Text-only isolates Thinker
generation; text+audio also schedules Talker and Code2Wav. The following full
Daily-Omni runs used the same two GPUs, 1197 samples, concurrency 10, and
identical `enable_thinking=false` / `use_tts_template=true` request settings.
The `origin/main` fused Talker ran eager because its graph capture copied an
unpinned CPU metadata tensor.

| Metric | `origin/main` two-stage | Three-stage batching |
| --- | ---: | ---: |
| Accuracy | 64.83% | 64.83% |
| Throughput | 0.62 req/s | 1.97 req/s |
| Mean E2EL | 16.17 s | 5.07 s |
| Mean serving TTFT | 0.92 s | 1.28 s |
| Mean audio TTFP | 16.17 s | 3.24 s |
| Mean audio RTF | 5.97 | 2.11 |
| Stage 0 mean TPOT / ITL | 8.27 / 8.27 ms | 40.08 / 40.11 ms |
| Stage 0 median TPOT / ITL | 7.23 / 7.24 ms | 7.43 / 7.53 ms |

The split pipeline improves throughput 3.19x and lowers audio TTFP by 80%.
Isolating the Thinker on GPU 0 also removes the prior single-GPU TPOT
regression: 40.08 ms is slightly better than the pre-rebase report (~44 ms).
Its median TPOT is effectively the same as main; the higher mean is queueing
tail latency because this profile bounds each stage to four sequences while
main's Thinker admits 16. Global TPOT/ITL remains zero when serving emits text
as one aggregated chunk, so the table reports Stage 0 metrics.

#### Verification

**Quick smoke test (text-only output)**:

```bash
curl http://localhost:8099/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "openbmb/MiniCPM-o-4_5",
        "messages": [{"role": "user", "content": "Briefly introduce yourself."}],
        "modalities": ["text"]
    }'
```

**Text + speech in one response** (the headline 4.5 feature). The model
bridge conditions the Talker from the generated assistant span, so the
generic serving layer does not inject MiniCPM-specific template defaults.
`use_tts_template=true` remains supported when explicitly requested:

```bash
curl http://localhost:8099/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "openbmb/MiniCPM-o-4_5",
        "messages": [{"role": "user", "content": "Say hello, then introduce vLLM in one sentence."}],
        "modalities": ["text", "audio"],
        "chat_template_kwargs": {"use_tts_template": true}
    }'
```

When using the OpenAI Python SDK, the same flag can also be sent as
`extra_body={"chat_template_kwargs": {"use_tts_template": True}}`
because the client merges `extra_body` into the request root.

Response carries text in one choice's `message.content` and base64 WAV
in another choice's `message.audio.data` (24 kHz mono, see Notes). With
`modalities: ["text", "audio"]` you typically get two `choices` entries
(one text, one audio).

**Streaming text + speech**:

```bash
python examples/online_serving/minicpmo/streaming_chat_completion.py \
    --base-url http://localhost:8099/v1 \
    --output minicpmo_stream.wav
```

The client prints text deltas as they arrive and reconstructs one valid WAV
from the independently encoded audio deltas.

**Gradio demo (text + image + audio + video UI)**:

```bash
bash examples/online_serving/minicpmo/run_gradio_demo.sh
# or run the python entry point directly:
python examples/online_serving/minicpmo/gradio_demo.py \
    --minicpmo45-api-base http://localhost:8099/v1 \
    --minicpmo45-model openbmb/MiniCPM-o-4_5 \
    --port 7862
```

Open `http://<host>:7862` and try a text prompt with the **"Generate
speech output (TTS)"** checkbox on / off.

#### Notes

- Memory budget: Thinker, Talker, and Code2Wav reserve 0.65, 0.15, and
  0.15 of GPU 0. The larger Thinker share protects its multimodal KV cache;
  all three model processes still share one CUDA device.
- `--trust-remote-code` is required — the HF repo ships a custom
  `MiniCPMO` config / model class.
- Stage 0 Thinker and Stage 1 Talker enable vLLM CUDA Graphs. Stage 2 remains
  eager because its request-owned Flow/HiFT caches and variable chunk/cache
  shapes are not yet exposed through a static exact-shape graph wrapper.
- All default stages use `max_num_seqs: 4` to reduce cross-process GPU
  contention. Talker AR
  state and Code2Wav caches are request-owned; Code2Wav batches only
  exact-shape-compatible chunks and does not fall back to serial decode.
- `StageRequestStats.batch_size` is a request-scoped placeholder, not the
  scheduler's execution batch.
- Single-GPU co-location trades throughput for hardware density: Stage 0/1
  CUDA Graph replay and eager Stage 2 vocoder kernels compete across three
  CUDA contexts. Use the 8x4090 config or a custom multi-GPU mapping for
  throughput-sensitive serving.

### 8 x RTX 4090 24GB (consumer-GPU layout)

Use
[`vllm_omni/deploy/minicpmo_4_5_8x4090.yaml`](../../vllm_omni/deploy/minicpmo_4_5_8x4090.yaml)
on an 8x RTX 4090 host. Thinker uses 4-way TP across GPUs 0–3
(`~85 %` mem each ≈ 20.4 GiB/card), Talker uses GPU 4, and Code2Wav
uses GPU 5. GPUs 6–7 are left free.

#### Command

```bash
vllm serve openbmb/MiniCPM-o-4_5 --omni \
    --deploy-config vllm_omni/deploy/minicpmo_4_5_8x4090.yaml \
    --trust-remote-code \
    --host 0.0.0.0 --port 8099
```

#### Notes

- `max_model_len` is capped at 4096 in this layout — 8192 still OOMs on
  4090s. Raise it if your cards have more headroom (e.g. 4090 D /
  custom 32 GB SKUs), but verify with a long-prompt run before
  promoting.
- All other knobs match the single-GPU section; the only difference is
  the per-card memory pressure on the thinker shards.

## Notes (applies to all layouts)

- **Code2Wav dependency**: Stage 2 loads `Token2wav` from the
  MiniCPM-o-flavored
  vocoder (PyPI package `stepaudio2-minicpmo` — NOT the upstream
  `stepfun-ai/Step-Audio2`, whose `Token2wav.__init__` signature
  rejects `n_timesteps`). Install via the published extra:

  ```bash
  pip install 'vllm-omni[minicpmo]'
  ```

  Equivalent direct install: `pip install stepaudio2-minicpmo`. A
  missing dep raises `ImportError` at first request with the same
  install hint instead of silently emitting empty audio.

- **TTS conditioning**: the MiniCPM stage bridge can condition speech from
  the generated assistant span without changing shared serving code.
  `chat_template_kwargs.use_tts_template=true` remains supported when an
  explicit `<|tts_bos|>` boundary is desired. For **curl**, put
  `chat_template_kwargs` at the request root; the OpenAI Python SDK may use
  `extra_body` because it flattens those fields into the root.

- **Reference voice**: request audio is carried on the first codec chunk.
  Code2Wav owns the temporary prompt WAV and prompt-feature cache, and removes
  both when the stream ends.

- **Talker sampling**: codec-token sampling reads the checkpoint `tts_config`
  and defaults to deterministic seed 42. Stage-1 deploy sampling parameters
  control only vLLM's binary continue/stop token.

- **Output audio**: 24 kHz mono WAV inside the OpenAI-style
  `message.audio.data` (base64). The Gradio demo's WAV player decodes
  this automatically.

- **Routing**: MiniCPM-o 4.5 and 2.6 both ship `architectures=
  ["MiniCPMO"]` in HF config; routing is disambiguated by
  `hf_config.version == "4.5"` via the
  `hf_config_predicate` on the 4.5 pipeline. A 2.6 checkpoint loaded
  with this recipe's `--deploy-config` will be rejected at startup
  rather than silently misrouted.

- **Async chunking**: enabled in all deploy configs. Talker sends
  25-code chunks with three-code left context to Code2Wav through
  `SharedMemoryConnector`; terminal chunks flush held lookahead state.
- **Response choices**: text and audio are separate choices. SDK clients
  should select the choice whose `message.audio.data` is populated rather
  than assuming `choices[0]` contains audio.
