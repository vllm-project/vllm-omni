# SocialOmni

[SocialOmni](https://github.com/MAC-AutoML/SocialOmni) evaluates speaker
attribution (Level 1) and whether and how a target participant should enter a
conversation (Level 2). This runner sends chat completions to a vLLM-Omni
server with video and embedded audio, requesting text output.

## Setup

Install vLLM-Omni following the [installation guide](../../docs/getting_started/installation/README.md).
The client requires `aiohttp`; Level 2 also needs FFmpeg with H.264 and AAC
encoders. The runner uses `ffmpeg` on `PATH`, falling back to `imageio-ffmpeg`.
For a standalone client without a model installation, run the commands below
with `uv run --no-project --with aiohttp --with imageio-ffmpeg python`.

Download the [MIT-licensed dataset](https://huggingface.co/datasets/alexisty/SocialOmni)
at the pinned revision:

```bash
uvx --from huggingface-hub hf download alexisty/SocialOmni \
    --repo-type dataset --revision 3b76009b45090eaa54007454c93a831f3cc8e1e6 \
    --local-dir /path/to/socialomni
```

The expected layout is `data/level_1/dataset.json` and
`data/level_2/annotations.json`, with videos under each level. Results include
metadata hashes and whether they match this revision; media content is not
verified by those hashes.

The client embeds video bytes in each request, following the shared
multimodal client. The server does not need access to the client's dataset or
prefix cache. Start a text-output server:

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni \
    --host 127.0.0.1 --port 8091 \
    --deploy-config vllm_omni/deploy/qwen3_omni_moe_thinking.yaml
```

The upstream deployment configuration uses two GPUs and text output only,
including when loading Instruct weights. Choose device and context settings
that fit the videos and hardware; see the
[Qwen3-Omni serving guide](../../examples/online_serving/qwen3_omni/README.md).
The request contains one `video_url` and
`mm_processor_kwargs.use_audio_in_video=true`. The video URL contains the
base64-encoded video; no separate audio item is sent.

## Evaluation

From the repository root, run a small deterministic selection covering both
Level 1 visibility groups and Level 2 YES/NO decisions:

```bash
uv run --no-sync python -m benchmarks.socialomni.evaluate \
    --dataset-root /data/socialomni \
    --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --base-url http://127.0.0.1:8091 --mini \
    --prefix-cache-dir /data/socialomni/prefixes \
    --judge-config /path/to/judges.json
```

Remove `--mini` for all 2,000 Level 1 and 209 Level 2 samples. Use
`--level level1` or `--level level2` to select one level. The first 200 Level 2
records in source order also produce the paper subset metrics.

Create a local judge configuration with exactly these three names. Set each
`model` to the corresponding model ID served by that endpoint:

```json
{
  "judges": [
    {"name": "gpt-4o", "model": "gpt-4o", "base_url": "https://api.openai.com/v1", "api_key_env": "OPENAI_API_KEY", "max_concurrency": 4},
    {"name": "gemini-2.5-pro", "model": "gemini-2.5-pro", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", "api_key_env": "GEMINI_API_KEY", "max_concurrency": 4},
    {"name": "qwen3-omni", "model": "qwen3-omni", "base_url": "http://localhost:8092/v1", "api_key_env": null, "max_concurrency": 1}
  ]
}
```

Endpoints must accept OpenAI-compatible chat completions. API keys are read
from the named environment variables; key values are not included in results.
`--base-url` and judge URLs may include `/v1` or the complete
`/chat/completions` path. Requests respect `HTTP_PROXY`, `HTTPS_PROXY`, and
`NO_PROXY`. Set `NO_PROXY=localhost,127.0.0.1` for local servers.

## Protocol and results

Level 2 re-encodes video and audio up to the annotated timestamp before model
requests. Model prompts receive neither the reference transcript nor the
reference continuation. Responses are generated for every ground-truth YES,
even when the model predicts NO. Only the judges receive reference material.

Level 1 reports accuracy, macro-F1 (the mean F1 score over four answer
positions), and accuracy by speaker visibility. Level 2 reports YES/NO
classification and these response metrics:

| Metric | Definition |
| ------ | ---------- |
| `QGold` | Three-judge mean quality over all ground-truth YES states; missing or empty generated responses contribute zero. |
| `QEns` | Mean quality conditional on a correct YES decision and a non-empty successful response. |
| `Cov+` | Fraction of ground-truth YES states meeting that condition. |
| `QEns_joint` | `Cov+ * QEns`. |

Every eligible response requires all three integer scores in
`{0, 25, 50, 75, 100}`. Missing judges never produce a partial-panel mean.
Omitting `--judge-config` allows a classification-only Level 2 run; quality
remains unset and the run exits with status 1. Request failures and unparsable
answers remain in the denominator and also make the run incomplete.

Each run writes a unique JSON file under `benchmarks/results/socialomni/`,
including configuration, client versions, metadata hashes, per-sample outputs,
failures, and metrics. Client versions do not identify a remote server; retain
its launch command, model revision, and software versions alongside results.
Latency and throughput are client diagnostics, not paper quality metrics.

`--max-concurrency` bounds model requests. Each non-empty model phase runs one
warmup request per configured concurrent worker; `--warmup N` overrides this
count. Warmup repeats initial samples and is discarded. Model wall time
excludes prefix encoding, warmup, and judging, but includes reading and
base64-encoding the request media. Request latency starts after that media
preparation. Judges have separate
concurrency limits and no warmup. Transient requests are retried up to three
times; an invalid judge score can trigger up to three scoring attempts.

Reference: [SocialOmni paper](https://arxiv.org/abs/2603.16859).
