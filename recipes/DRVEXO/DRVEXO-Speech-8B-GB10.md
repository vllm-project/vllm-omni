# DRVEXO Speech 8B — DGX Spark GB10

## Summary

- Vendor: DRVEXO
- Model: DRVEXO Speech 8B, complete local unified safetensors checkpoint
- Task: text/audio chat, caller-text speech, explicit candidate scoring
- Mode: online, three-stage synchronous Thinker handoff with Talker-to-Codec chunks
- Hardware: one NVIDIA DGX Spark GB10
- Maintainer: DRVEXO-LLM

## When to use this recipe

Run a local speech backend for voice communications, outbound-call applications
or voice control. The application supplies committed conversation history,
authoritative product facts, telephony transport and actual tool/device execution.

## Supported model contract

| Task | Entry point | Contract |
| --- | --- | --- |
| Text or current-audio input to text | `/v1/chat/completions` | Complete committed text history; at most one current input audio item |
| Generated text plus speech | `/v1/chat/completions` | `modalities: [text, audio]`, WAV output, generated-token conditioning |
| Fixed caller-text speech | `/v1/drvexo/speech`, `/v1/drvexo/speech/stream` | 1–512 characters; Auto/Chinese/English; exact Talker tokenization, no control tokens |
| Text/audio NDJSON stream | `/v1/drvexo/chat/stream` | One response choice; local HTTP TCP listener; direct TLS/UDS unsupported |
| Explicit candidate decision | `/v1/drvexo/decision` | 2–151 caller-defined candidates; scores are non-executable |

The complete checkpoint contains the Thinker/audio frontend, Talker,
CodePredictor, Codec, tokenizers, speaker embedding and optional General head.
The code PR does not host weights. Use an authorized local bundle, including its
`components.json`, stage views and namespaced safetensors shards.

| Profile | Deployment |
| --- | --- |
| Complete BF16 | `vllm_omni/deploy/drvexo_speech_8b.yaml`, one device, eager, TP1, one sequence |
| Complete FP8 / NVFP4 | Same YAML with the corresponding independently exported bundle; component precision exceptions below |

## References

- [Integration and validation scope](../../docs/models/drvexo_speech_8b.md)
- [Pipeline deployment](../../vllm_omni/deploy/drvexo_speech_8b.yaml)
- [PR #9645](https://github.com/vllm-project/vllm-omni/pull/9645)
- Use the shared HTTP API directly; no model-specific Python example is required.

## Hardware

- Accelerator: one NVIDIA GB10, DGX Spark, shared CPU/GPU memory, approximately 121 GiB OS-visible.
- Number of devices: one; no inter-device topology qualified.
- Qualification: local functional serving; no throughput, speaker-SIM,
  microphone, telephony or physical device-execution acceptance is implied.
- Other accelerator profiles have not been qualified for this integration.

## Software environment

- OS: Ubuntu 24.04 LTS, ARM64
- Python: 3.12
- Torch / CUDA runtime: 2.13.0+cu130
- vLLM: 0.32.0
- Transformers: 5.16.1
- FlashInfer: 0.7.0.post1; Humming-kernels: 0.1.17; CUTLASS DSL: 4.8.0
- vLLM-Omni: this PR; exact measured source revisions and follow-up limitations
  are retained in the integration guide's validation section.

The local environment includes system site-packages. Development-package
version metadata can lag the code; record the source commit. First-use native
kernel compilation can take several minutes. vLLM 0.28 compatibility is not claimed.

## Command

Install this PR checkout with its documented CUDA dependencies first. Download
or copy the complete authorized checkpoint before launch; the loader is local-only.

```bash
export VLLM_API_KEY="set-a-local-api-key"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
vllm serve /absolute/path/to/complete-checkpoint --omni \
  --deploy-config vllm_omni/deploy/drvexo_speech_8b.yaml \
  --served-model-name drvexo --host 127.0.0.1 --port 8000 \
  --stage-init-timeout 1800 --init-timeout 1900
```

The serialization opt-in is for the tested local multiprocess stage transport;
checkpoint weights are safetensors. The equivalent tested module entry point is
`python -m vllm_omni.entrypoints.cli.main serve` with the same arguments.
Keep the pipeline's eager, single-sequence, no-prefix-cache settings and exact
2048-token Thinker prefill batch when enabling the General head.

## Verification

```bash
curl -f http://127.0.0.1:8000/health
curl -f http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $VLLM_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"drvexo","messages":[{"role":"user","content":"Hello"}],"modalities":["text"],"max_tokens":64}'
curl -f http://127.0.0.1:8000/v1/drvexo/speech \
  -H "Authorization: Bearer $VLLM_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"drvexo","text":"Hello, welcome back.","language":"English"}' \
  --output speech.wav
```

Expect a healthy server, a nonempty chat response and a playable WAV. Requests
to `/v1/*` without the bearer token must return 401. The integration guide
records the broader precision-specific tests and source revisions; these quick
commands alone do not establish all of those checks.

## Notes

- BF16/FP8/NVFP4 describe selected stored or quantized components, not every
  operator. Quantized bundles retain a BF16 audio encoder, FP32 CPU General head
  and other embedding/norm/convolution/reference exceptions in their manifests.
- The General head hashes every bound shard at startup. Candidate scoring
  repeats the prefix sequentially; one decision is admitted per API process,
  busy calls return 429 and the 120-second deadline returns 504 after cleanup.
- All stages share one device; separate deployments are needed for traffic
  isolation. No production concurrency guarantee has been established.
- Local shutdown evidence retains shared-memory resource-tracker warnings;
  zero process exits do not prove complete IPC-object lifecycle correctness.

## Supported features

| Feature | Status and topology | Shared guide |
| --- | --- | --- |
| Online serving | Three stages, single GB10 | [Online serving](../../docs/serving/README.md) |
| Native speech streaming | Talker-to-Codec chunks; explicit sentence stream and NDJSON wrapper | [Speech API](../../docs/serving/speech_api.md) |
| Independent stage routing | This recipe uses the complete pipeline; arbitrary external routing is not qualified | [Independent stages](../../docs/features/independent_stage_execution.md) |
| Quantization | Independently exported BF16/FP8/NVFP4 bundles with component exceptions | [Integration](../../docs/models/drvexo_speech_8b.md) |
