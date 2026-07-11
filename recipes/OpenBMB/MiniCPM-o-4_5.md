# MiniCPM-o 4.5

> Online serving for omni multimodal chat (text / image / audio / video → text + 24 kHz speech)

## Summary

- Vendor: OpenBMB
- Model: [`openbmb/MiniCPM-o-4_5`](https://huggingface.co/openbmb/MiniCPM-o-4_5)
- Task: Omni multimodal chat — accepts text / image / audio / video input;
  emits text and 24 kHz mono speech in the same response
- Mode: Online serving via the OpenAI-compatible `/v1/chat/completions`
  API, plus a bundled Gradio demo (text + speech UI)
- Maintainer: [`@tc-mb`](https://github.com/tc-mb) (MiniCPM-V / MiniCPM-o team)

## When to use this recipe

Use this recipe as a known-good starting point for serving
`openbmb/MiniCPM-o-4_5` on vLLM-Omni. MiniCPM-o 4.5 is the omni member
of the MiniCPM-o family — it pairs a multimodal-understanding thinker
LLM with a streaming `MiniCPMTTS + Token2Wav` talker so a single
`/v1/chat/completions` call can return text and 24 kHz speech in one
shot. The recipe covers three shipped GPU layouts (2 / 3 / 8 GPUs)
selected via `--deploy-config`.

## References

- Default deploy configs (auto-loaded by HF `model_type=minicpmo` +
  `hf_config.version="4.5"`):
  - 2-GPU layout (default):
    [`vllm_omni/deploy/minicpmo_4_5.yaml`](../../vllm_omni/deploy/minicpmo_4_5.yaml)
  - 3-GPU layout (thinker TP=2):
    [`vllm_omni/deploy/minicpmo_4_5_3gpu.yaml`](../../vllm_omni/deploy/minicpmo_4_5_3gpu.yaml)
  - 8x RTX 4090 layout:
    [`vllm_omni/deploy/minicpmo_4_5_8x4090.yaml`](../../vllm_omni/deploy/minicpmo_4_5_8x4090.yaml)
- Online example + Gradio demo:
  [`examples/online_serving/minicpmo/`](../../examples/online_serving/minicpmo/)
- Pipeline / talker source:
  [`vllm_omni/model_executor/models/minicpmo_4_5/`](../../vllm_omni/model_executor/models/minicpmo_4_5/)
- Stage-input processor (thinker → talker bridge):
  [`vllm_omni/model_executor/stage_input_processors/minicpmo_4_5_omni.py`](../../vllm_omni/model_executor/stage_input_processors/minicpmo_4_5_omni.py)
- Upstream model card:
  [`openbmb/MiniCPM-o-4_5`](https://huggingface.co/openbmb/MiniCPM-o-4_5)
- Integration PR:
  [vllm-project/vllm-omni#3642](https://github.com/vllm-project/vllm-omni/pull/3642)

## Hardware Support

Three GPU layouts ship with default deploy configs. Pick the layout that
matches your hardware and pass it via `--deploy-config`; the talker
(`MiniCPMTTS + Token2Wav`) always lives on its own GPU because of the
in-process vocoder, and the thinker is the part that scales out via TP.

| Layout | Thinker | Talker + Token2Wav | Typical hardware |
| --- | --- | --- | --- |
| 2-GPU (default) | GPU 0 | GPU 1 | 2x A100/H100/H200 80GB |
| 3-GPU (thinker TP=2) | GPU 0,1 (TP=2) | GPU 2 | 3x mid-tier GPUs |
| 8x RTX 4090 24GB | GPU 0–3 (TP=4) | GPU 4 | 8x RTX 4090 consumer |

## GPU

### 2 x GPU (default — single command)

The default
[`vllm_omni/deploy/minicpmo_4_5.yaml`](../../vllm_omni/deploy/minicpmo_4_5.yaml)
puts the thinker on GPU 0 (`~70 %` memory, `enforce_eager: true`,
`max_num_seqs: 1`) and the talker + Token2Wav vocoder on GPU 1
(`~75 %` memory). This is the recommended starting layout — works on
any pair of 80GB-class GPUs (A100, H100, H200) and on most 40GB+
pairs as long as the thinker model weights fit.

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
`--deploy-config` flag needed for this default 2-GPU layout.

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

**Text + speech in one response** (the headline 4.5 feature). The TTS
path is gated by a Jinja flag on the chat template. Pass
`use_tts_template=true` via the **top-level** `chat_template_kwargs`
field (curl does not flatten nested `extra_body`):

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

- Memory budget: thinker weights occupy GPU 0 at `gpu_memory_utilization:
  0.7`; talker + Token2Wav vocoder share GPU 1 at `0.75`.
- `--trust-remote-code` is required — the HF repo ships a custom
  `MiniCPMO` config / model class.
- Pin: `enforce_eager: true` on both stages (CUDA graph capture is off
  by design for the talker's Token2Wav path).
- Stage 1 (talker) is hard-capped to `max_num_seqs: 1`: the talker
  only consumes `runtime_additional_information[0]`, so any value > 1
  makes concurrent requests share request-0's audio. This is the same
  cap baked into the deploy config.

### 3 x GPU (thinker TP=2)

Use
[`vllm_omni/deploy/minicpmo_4_5_3gpu.yaml`](../../vllm_omni/deploy/minicpmo_4_5_3gpu.yaml)
when you have a third GPU available and want the thinker on 2-way
tensor parallel for higher throughput; the talker stays on its own
GPU (talker has its own in-process Token2Wav vocoder, so co-locating
it with the thinker risks OOM under load).

#### Command

```bash
vllm serve openbmb/MiniCPM-o-4_5 --omni \
    --deploy-config vllm_omni/deploy/minicpmo_4_5_3gpu.yaml \
    --trust-remote-code \
    --host 0.0.0.0 --port 8099
```

Verification and Notes mirror the 2-GPU section; thinker latency
roughly halves under load thanks to TP=2.

### 8 x RTX 4090 24GB (consumer-GPU layout)

Use
[`vllm_omni/deploy/minicpmo_4_5_8x4090.yaml`](../../vllm_omni/deploy/minicpmo_4_5_8x4090.yaml)
on an 8x RTX 4090 host. Thinker uses 4-way TP across GPUs 0–3
(`~85 %` mem each ≈ 20.4 GiB/card), talker + Token2Wav lives on GPU 4
(`~90 %` mem). GPUs 5–7 are left free.

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
- All other knobs match the 2-GPU section; the only difference is the
  per-card memory pressure on the thinker shards.

## Notes (applies to all layouts)

- **Talker dependency**: the `MiniCPM-o 4.5` talker calls
  `from stepaudio2 import Token2wav` against the MiniCPM-o-flavored
  vocoder (PyPI package `stepaudio2-minicpmo` — NOT the upstream
  `stepfun-ai/Step-Audio2`, whose `Token2wav.__init__` signature
  rejects `n_timesteps`). Install via the published extra:

  ```bash
  pip install 'vllm-omni[minicpmo]'
  ```

  Equivalent direct install: `pip install stepaudio2-minicpmo`. A
  missing dep raises `ImportError` at first request with the same
  install hint instead of silently emitting empty audio.

- **TTS trigger**: speech output requires
  `chat_template_kwargs.use_tts_template=true` so the chat template
  appends `<|tts_bos|>` before generation. Without it, Stage-1 talker
  receives no TTS token span and returns silent audio (not text-only).
  For **curl**, put `chat_template_kwargs` at the request root; nested
  `extra_body.chat_template_kwargs` is ignored. The OpenAI Python SDK
  may use `extra_body` because it flattens those fields into the root.

- **Output audio**: 24 kHz mono WAV inside the OpenAI-style
  `message.audio.data` (base64). The Gradio demo's WAV player decodes
  this automatically.

- **Routing**: MiniCPM-o 4.5 and 2.6 both ship `architectures=
  ["MiniCPMO"]` in HF config; routing is disambiguated by
  `hf_config.version == "4.5"` via the
  `hf_config_predicate` on the 4.5 pipeline. A 2.6 checkpoint loaded
  with this recipe's `--deploy-config` will be rejected at startup
  rather than silently misrouted.

- **Async chunking**: disabled in all three deploy configs
  (`async_chunk: false`) — the talker batches a single full thinker
  output, not chunks.
