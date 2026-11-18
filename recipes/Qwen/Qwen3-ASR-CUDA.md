# Qwen3-ASR on CUDA

Serve Qwen3-ASR with optional word timestamps on one GPU. The configuration
below was tested on an NVIDIA H20 with vLLM 0.33.0, Qwen3-ASR-1.7B and
Qwen3-ForcedAligner-0.6B in eager mode, using English and Mandarin audio.

## Start the server

After [installing vLLM-Omni](../../docs/getting_started/installation/README.md),
run from the repository root. Replace the model IDs with local checkpoint paths
if the weights are already downloaded.

```bash
vllm serve Qwen/Qwen3-ASR-1.7B --omni \
    --deploy-config vllm_omni/deploy/qwen3_asr.yaml \
    --forced-aligner Qwen/Qwen3-ForcedAligner-0.6B --forced-aligner-device 0 \
    --trust-remote-code --enforce-eager --max-num-seqs 2 \
    --kv-cache-memory-bytes 2147483648 --host 127.0.0.1 --port 18473
```

For plain transcription only, omit both `--forced-aligner` and
`--forced-aligner-device` and request `json` or `text`.

## Request word timestamps

From the repository root, send the included Mandarin audio clip:

```bash
curl --fail http://127.0.0.1:18473/v1/audio/transcriptions \
    -F file=@tests/assets/semantic_vad/complete.wav \
    -F response_format=verbose_json \
    -F 'timestamp_granularities[]=word'
```

The validated run returned “不用帮我叫车了，我自己坐地铁回去。” with 15 word units;
the last ended at 3.36 s in the 3.40 s clip. See the
[Transcription API](../../docs/serving/transcription_api.md) for response fields
and text streaming.

### Word segmentation

The English and Mandarin checks used the built-in segmenter. Japanese and
Korean need the optional `qwen-asr` processor for faithful word boundaries.
Its 0.0.6 release pins Transformers 4.57.6, which conflicts with the tested
Transformers 5.19.0 environment. Those languages were not validated.
