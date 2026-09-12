# CosyVoice3 streaming TTS with flow CUDA graphs

> Voice-cloned streaming TTS, and how to reproduce the flow CUDA-graph A/B.

## Summary

- Vendor: FunAudioLLM
- Model: `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`
- Task: Streaming text-to-speech with voice cloning (`ref_audio` + `ref_text`)
- Mode: Online serving (`async_chunk`) and offline
- Maintainer: @BruceLoveDecimal

## When to use this recipe

Serving CosyVoice3 for low-latency streaming TTS, and measuring the effect of
the flow estimator CUDA graphs (`enable_flow_cuda_graph`) on latency,
throughput, and streaming continuity on your own hardware.

CosyVoice3 runs two stages: a Qwen2-0.5B talker emitting speech tokens, and a
code2wav stage that turns them into audio with a flow-matching DiT plus a HiFT
vocoder. Stage 1 keeps `enforce_eager: true` because the vocoder's dynamic
convolution shapes rule out vLLM's own CUDA graphs; `enable_flow_cuda_graph`
graphs only the flow's DiT estimator, which is the launch-bound part.

## References

- Upstream: <https://github.com/FunAudioLLM/CosyVoice>
- Speech API docs: [`docs/serving/speech_api.md`](../../docs/serving/speech_api.md)
- Offline example: [`examples/offline_inference/text_to_speech/cosyvoice3/end2end.py`](../../examples/offline_inference/text_to_speech/cosyvoice3/end2end.py)
- Deploy config: [`vllm_omni/deploy/cosyvoice3.yaml`](../../vllm_omni/deploy/cosyvoice3.yaml)

## Hardware Support

## GPU

### 1x NVIDIA H20 96GB

#### Environment

- OS: Ubuntu 22.04
- Python: 3.12
- Driver / runtime: CUDA 13.0
- PyTorch: 2.13
- vLLM version: 0.27
- vLLM-Omni version or commit: `main` at or after the flow CUDA-graph change

`tensorrt` is not installed in this environment, so the flow runs the torch
DiT estimator. The TRT estimator path is not covered by these numbers.

#### Command

```bash
vllm serve FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --omni --host 127.0.0.1 --port 8091 --trust-remote-code
```

The deploy config ships with `enable_flow_cuda_graph: true` in the
`connector_of_shared_memory` `extra` block. Set it to `false` there to serve
without flow graphs, which is also how the A/B below toggles sides.

#### Verification

```bash
curl -s http://127.0.0.1:8091/health -o /dev/null -w '%{http_code}\n'
```

Once traffic is flowing, the stage-1 log reports each capture:

```bash
SERVER_LOG=./serve.log
grep -c 'Captured flow estimator CUDA graph' "$SERVER_LOG"
```

A single-voice workload settles at a handful of captures. Zero captures with
the flag on means every request fell back to eager -- check the free-memory
floor and look for a `Disabling flow estimator CUDA graphs` warning.

#### Benchmark with `vllm bench`

Build a small seed-tts dataset once, then benchmark against the running
server. `audio_ttfp` (time to first packet) is the streaming-TTS analogue of
TTFT; continuity is reported by `vllm bench` on stdout as
`Streaming continuity OK rate` (the fraction of requests whose worst
playback underrun stays under 100 ms) and is **not** written into the result
JSON, so capture stdout if you need it.

The tokenizer lives in a subdirectory of the checkpoint, so it has to be a
local path -- appending `/CosyVoice-BlankEN` to the HF repo id does not
resolve. Download the checkpoint once, as in
[`examples/offline_inference/text_to_speech/README.md`](../../examples/offline_inference/text_to_speech/README.md):

```python
from huggingface_hub import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
                  local_dir='pretrained_models/Fun-CosyVoice3-0.5B')
```

`--dataset-path` takes the `BytedanceSpeech/seed-tts-eval` repo id or a local
root in the same `meta.lst` layout.

```bash
MODEL_DIR=pretrained_models/Fun-CosyVoice3-0.5B

vllm bench serve \
  --omni \
  --host 127.0.0.1 \
  --port 8091 \
  --model "$MODEL_DIR" \
  --tokenizer "$MODEL_DIR/CosyVoice-BlankEN" \
  --backend openai-audio-speech \
  --endpoint /v1/audio/speech \
  --dataset-name seed-tts \
  --dataset-path BytedanceSpeech/seed-tts-eval \
  --seed-tts-locale zh \
  --num-prompts 24 \
  --num-warmups 2 \
  --max-concurrency 4 \
  --request-rate inf \
  --percentile-metrics ttft,e2el,audio_rtf,audio_ttfp,audio_duration,audio_underrun \
  --metric-percentiles 50,95,99 \
  --seed 0 \
  --disable-tqdm \
  --save-result --result-dir ./bench-out --result-filename c4.json
```

#### Reproducing the flow CUDA-graph A/B

Pass the path to the shipped `vllm_omni/deploy/cosyvoice3.yaml`; the script
writes two toggled copies and serves each with `--deploy-config`, so the
installed config is never edited in place.

Single runs on this host carry roughly +/-3-5% noise, which is enough to
manufacture a convincing regression. Interleave the sides (A,B,A,B,...) so
thermal drift hits both instead of being confounded with the side, restart
the server between sides, and only call a direction when the gap clears the
combined stdev.

```bash
#!/bin/bash
# flow-graph-ab.sh <reps> <concurrency> <path-to-cosyvoice3.yaml>
# Interleaved A/B over the deploy-config flag, restarting the server per side.
set -eu
REPS="$1"; C="$2"; BASE_YAML="$3"
MODEL_DIR=pretrained_models/Fun-CosyVoice3-0.5B
DATASET=BytedanceSpeech/seed-tts-eval

# Two copies of the shipped config, so the installed one is never mutated.
for val in false true; do
  sed "s/^      enable_flow_cuda_graph: .*/      enable_flow_cuda_graph: $val/" \
    "$BASE_YAML" > "cosyvoice3-$val.yaml"
  grep -q "enable_flow_cuda_graph: $val" "cosyvoice3-$val.yaml" \
    || { echo "toggle did not apply for $val" >&2; exit 1; }
done

for r in $(seq 1 "$REPS"); do
  for side in off:false on:true; do
    tag=${side%%:*}; val=${side##*:}

    vllm serve "$MODEL_DIR" --omni --host 127.0.0.1 --port 8091 \
      --trust-remote-code --deploy-config "cosyvoice3-$val.yaml" \
      > "serve-$tag-r$r.log" 2>&1 &
    ready=0
    for _ in $(seq 1 60); do
      if curl -sf http://127.0.0.1:8091/health >/dev/null; then ready=1; break; fi
      sleep 10
    done
    if [ "$ready" != 1 ]; then
      echo "server never became ready; see serve-$tag-r$r.log" >&2
      pkill -f 'vllm serve' || true
      exit 1
    fi

    mkdir -p "ab-out/$tag"
    vllm bench serve --omni --host 127.0.0.1 --port 8091 \
      --model "$MODEL_DIR" --tokenizer "$MODEL_DIR/CosyVoice-BlankEN" \
      --backend openai-audio-speech --endpoint /v1/audio/speech \
      --dataset-name seed-tts --dataset-path "$DATASET" --seed-tts-locale zh \
      --num-prompts 24 --num-warmups 2 --max-concurrency "$C" --request-rate inf \
      --percentile-metrics ttft,e2el,audio_rtf,audio_ttfp,audio_duration,audio_underrun \
      --metric-percentiles 50,95,99 --seed 0 --disable-tqdm \
      --save-result --result-dir "ab-out/$tag" --result-filename "c$C-r$r.json" \
      | tee "bench-$tag-c$C-r$r.log"

    # grep -c exits 1 on zero matches, which is the expected result on the
    # off side, so keep it from tripping set -e.
    ncap=$(grep -c 'Captured flow estimator CUDA graph' "serve-$tag-r$r.log" || true)
    echo "$tag r$r captures=$ncap"
    pkill -f 'vllm serve' || true
    sleep 5
  done
done
```

Sanity-check every run before trusting the table: the `off` side must report
`captures=0` and the `on` side a non-zero count. Equal continuity on both
sides usually means the toggle never took effect.

#### Measured results

1x H20, seed-tts zh, 24 prompts, 3 interleaved repeats per side. All entries
below exceed the combined standard deviation unless noted.

| metric | c=1 | c=4 | c=8 |
| --- | --- | --- | --- |
| continuity OK rate | 100% -> 100% | 5.6% -> 80.6% | 0% -> 8.3% (ns) |
| throughput | +21.3% | +57.8% | +71.7% |
| RTF mean | -22.5% | -41.5% | -44.5% |
| `audio_ttfp` mean | -33.8% | -52.9% | -48.9% |
| `audio_ttfp` p99 | -35.1% | -38.4% | -38.5% |
| E2EL mean | -17.6% | -36.8% | -42.9% |
| underrun mean | n/a (0 both) | -83.9% | -69.1% |

At c=1 continuity is already 100% on both sides, so graphs buy latency, not
continuity. At c=8 this GPU is saturated -- RTF stays near 1.0 and continuity
moves from 0% to 8.3% across repeats of 16.67/0/8.33, which is not
significant; CUDA graphs do not fix queueing.

Offline stage-1 time, 3 interleaved repeats of
`examples/offline_inference/text_to_speech/cosyvoice3/end2end.py`:

| utterance | graphs off | graphs on | delta |
| --- | --- | --- | --- |
| ~10 s | 2174.1 +/- 64.7 ms | 1717.0 +/- 36.3 ms | -21.0% |
| ~25 s | 4931.6 +/- 42.0 ms | 4300.7 +/- 54.0 ms | -12.8% |

#### Notes

- Memory usage: stage 0 `gpu_memory_utilization: 0.4`, stage 1 `0.2`.
- Key flags: `enable_flow_cuda_graph` (default on in the shipped config),
  `flow_graph_max_graphs` (default 64), `codec_chunk_frames`,
  `codec_left_context_frames`.
- Streaming continuity is GPU-class sensitive. Re-measure `max_num_seqs` and
  `codec_chunk_frames` on your own card rather than assuming these settings
  transfer.
- Known limitations: the TRT estimator path is not graphed and was not
  measured here; graphs are keyed on exact input shape, so a deployment
  serving many reference voices of differing durations produces more shapes
  and, once `flow_graph_max_graphs` is reached, retires the whole cache and
  starts over.
