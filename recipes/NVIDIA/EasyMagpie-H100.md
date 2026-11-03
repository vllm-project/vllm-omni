# EasyMagpie TTS — H100

## Summary

- Vendor: NVIDIA.
- Model: a converted EasyMagpie/Nemotron-TTS checkpoint with architecture `EasyMagpieTTSForConditionalGeneration`.
- Task and mode: speaker-conditioned text-to-speech, streaming PCM over `/v1/audio/speech`.
- Hardware target: one H100 80 GB; no inter-device communication (TP=PP=DP=1).
- Maintainer: vLLM-Omni community.
- Status: configuration-only. This integration has not been benchmarked on H100 at C256.

## Supported model contract

| Item | Contract |
| --- | --- |
| Input | Complete text and a bundled speaker name; optional checkpoint-specific context text |
| Output | Mono 22,050 Hz audio; streaming signed 16-bit PCM |
| Pipeline | Nemotron-H and local codebook transformer, then stateful spectral codec |
| Required assets | Converted LM weights/tokenizers, speaker embeddings, and `codec_native` weights/configuration |

| Profile | Capacity | Precision | Cache budgets | Codec chunk sizes |
| --- | --- | --- | --- | --- |
| [`easymagpie_h100_c256.yaml`](../../vllm_omni/deploy/easymagpie_h100_c256.yaml) | 256 requests per stage | FP16 LM/codec, FP32 SSM | 8 GiB LM, 256 MiB codec | 2, then 6, then 8 stacked frames |

The profile enables request-owned decode buffers, batched feedback, separate
prefill/decode scheduling, and prewarmed codec CUDA graphs. It uses native Omni
workers and connectors. It does not install the reference deployment's patches
to vLLM, HTTP handling, or transport. Cache budgets exclude weights, activations,
and graph storage. Recompute preemption is unsupported.

## Software and checkpoint setup

Install the vLLM version required by this checkout. The converted checkpoint
contract and optional Perth dependency are described in the
[checkpoint setup](EasyMagpie-A4500.md#checkpoint-setup). Watermarking follows
the codec configuration; preserve the same setting when comparing deployments.
There is no qualified H100 software/driver combination or measured memory peak
for this integration yet.

## Command

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
TOKENIZERS_PARALLELISM=false \
vllm serve /path/to/converted-model --omni \
    --served-model-name easymagpie \
    --deploy-config vllm_omni/deploy/easymagpie_h100_c256.yaml \
    --trust-remote-code --port 8091
```

Use the [shared speech clients](../../examples/online_serving/text_to_speech/README.md)
with a bundled voice, checkpoint-specific `extra_params.context_text`, and
22,050 Hz playback. A smoke test must return nonempty audio before running C256.
Compare both deployments sequentially with identical weights, precision, request
parameters, warmup, request count, and concurrency. Report throughput, p95 TTFA,
request failures, playback underruns, and output checks. A smaller-GPU comparison
does not establish H100 parity.

## References

- [Speech API](../../docs/serving/speech_api.md)
- [Supported models](../../docs/models/supported_models.md)
- [Pipeline definition](../../vllm_omni/model_executor/models/easymagpie/pipeline.py)
