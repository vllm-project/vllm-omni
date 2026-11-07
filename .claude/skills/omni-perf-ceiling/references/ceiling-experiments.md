# Ceiling Experiments

## Contents
- Admission overlay
- Cell commands
- Per-stage GPU attribution
- Envelope sweep
- Device peaks
- Pitfalls

## Admission overlay

A high-concurrency cell is only valid when no stage caps admission below
the target. Write a thin overlay over the shipped profile:

```yaml
# c64.yaml
base_config: /abs/path/vllm_omni/deploy/<model>_turn.yaml
stages:
  - stage_id: 0
    max_num_seqs: 64
  - stage_id: 1
    max_num_seqs: 64
  - stage_id: 2
    max_num_seqs: 64
```

`base_config` resolves relative to the overlay file; absolute paths work.
Platform sections in the base still apply on top of `stages`. Check model-
specific limits too, for example MiniCPM-o Code2Wav `max_graph_batch` in the
connector `extra` (batches above it run eagerly) and duplex
`duplex_session.max_sessions`.

After startup, grep the server log for capture budgets and fallbacks
(`graph`, `eager`, `budget`, `capacity`) and record them.

## Cell commands

```bash
vllm bench serve --omni --host 127.0.0.1 --port $PORT --model $MODEL \
  --endpoint /v1/chat/completions --backend openai-chat-omni \
  --dataset-name seed-tts --dataset-path $SEED_TTS --seed-tts-locale en \
  --disable-shuffle --no-oversample --seed 0 --temperature 0 \
  --num-prompts $((10 * C)) --num-warmups $C --max-concurrency $C \
  --request-rate inf \
  --extra-body '{"modalities":["text","audio"]}' \
  --percentile-metrics ttft,e2el,audio_ttfp,audio_rtf,audio_duration \
  --metric-percentiles 50,90,99 --print-stage \
  --save-result --save-detailed --result-dir $OUT --result-filename c$C.json \
  > $OUT/c${C}_bench.log 2>&1
```

Sample the serving GPU for the whole measured window:

```bash
nvidia-smi -i $GPU --query-gpu=timestamp,utilization.gpu,memory.used \
  --format=csv -lms 500 > $OUT/c${C}_gpu.csv &
```

Build the ledger:

```bash
python .claude/skills/omni-perf-ceiling/scripts/ceiling_report.py \
  --cell c1=$OUT/c1.json:$OUT/c1_bench.log:$OUT/c1_gpu.csv \
  --cell c64=$OUT/c64.json:$OUT/c64_bench.log:$OUT/c64_gpu.csv \
  --json $OUT/ledger.json
```

Add physical bounds once peaks and per-stage work are known. Weight bytes
come from the checkpoint (sum tensor sizes per component prefix in the
safetensors headers); tokens per audio second from `request_stage_metrics`
in the result JSON (stage `num_tokens_out` over stage-2 `audio_duration_s`):

```bash
  --peaks $OUT/peaks.json \
  --decode-bound thinker=<weight_gib>,<text_tokens_per_audio_s> \
  --decode-bound talker=<weight_gib>,<codec_tokens_per_audio_s> \
  --compute-bound cfm=<tflop_per_audio_s>,tf32 \
  --compute-bound vocoder=<tflop_per_audio_s>,fp32
```

Count compute-stage FLOPs on CPU with `torch.utils.flop_counter.FlopCounterMode`
around one forward of the stage's module at a representative chunk shape,
then scale by frames per audio second, sampling steps and CFG rows. Remove
weight-norm parametrizations first (`torch.nn.utils.parametrize`); the
counter cannot see through them.

Model-specific `--extra-body` fields (chat template switches, reference
audio) belong to the workload freeze; copy them from the model profile.

## Per-stage GPU attribution

Choose the cheapest method that the hardware allows.

1. **One device per stage** (`*_3gpu.yaml` or equivalent overlay). Sample
   every device; each device's busy share is its stage's `u_i`. Inter-device
   transfer differs from the shared-device deployment, so use this for
   `g_i`, not for the shared-device RTF.
2. **Shared device, MPS off.** Add `platforms: {cuda: {cuda_mps: false}}` to
   the overlay so each stage process owns its context, then sample
   `nvidia-smi pmon -i $GPU -s u -d 1`, then pass
   `--pmon c64=$OUT/c64_pmon.txt:$OUT/server.log` to `ceiling_report.py`.
   It maps PIDs to stages through the `StageEngineCoreProc_stage<N>` log
   prefix. Under MPS all clients report as the MPS server and per-stage
   attribution is lost. Time-slicing without MPS changes overlap, so treat
   this as attribution only. pmon reports `-` for intervals without a
   reading, which becomes frequent under heavy load; the report warns when
   the stage sum disagrees with `utilization.gpu` by more than 20%. Then
   only the proportions between stages are usable; prefer method 1 or 3
   for absolute `g_i`.
3. **Stage isolation.** Feed one stage recorded or synthetic inputs at fixed
   batch sizes and time it. This gives the `g_i(B)` curve directly, including
   batch sizes the deployment cannot reach.

`--print-stage` `stage_gen_time` is per-request wall time inside a stage.
Under batching it overlaps across requests and is not GPU time; use it for
the low-concurrency bound and for spotting the slowest stage, not for `g_i`.

## Envelope sweep

One configuration lever per column, same commit and workload:

| Lever | Typical knob |
|-------|-------------|
| Admission | stage `max_num_seqs`, `max_sessions` |
| Runner | `model_runner: v2` per stage, or the model's MRv2 profile |
| Graph coverage | graph batch limits, graph budgets, capture sizes |
| Micro-batching | per-stage micro-batch or chunk scheduling knobs |
| Co-location | `cuda_mps`, `gpu_memory_utilization`, placement |
| Replicas | replica profiles for the binding stage |

Keep the best value per cell. Numerics-changing levers go in a separate
column with their quality delta.

## Device peaks

```bash
python .claude/skills/omni-perf-ceiling/scripts/measure_peaks.py --json $OUT/peaks.json
```

Run it on an otherwise idle device. It reports copy bandwidth and dense
matmul throughput for BF16, FP16, TF32 and IEEE FP32.

## Pitfalls

- Too few prompts at high concurrency: ramp-up and ramp-down dominate and
  Little's law fails. Use at least `10 x C`.
- Warmup shorter than `C` requests leaves first-shape graph capture inside
  the measured window.
- Mixed output modes: a single-file engine and a streaming engine differ in
  work per request.
- Profilers on: never report profiler-enabled latency or throughput.
- Shared hosts: record other processes on the device; GPU utilisation from
  another tenant invalidates the attribution.
