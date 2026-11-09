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

Sample the serving device for the whole measured window:

```bash
nvidia-smi -i $GPU --query-gpu=timestamp,utilization.gpu,memory.used \
  --format=csv -lms 500 > $OUT/c${C}_gpu.csv &
```

On Ascend use `npu-smi info watch` with a timestamp prefix instead; the command
and its caveats are in [npu-ceiling.md](npu-ceiling.md). `ceiling_report.py`
detects either format from the header.

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
  --decode-bound thinker@0=<weight_gib>,<text_tokens_per_audio_s>,<kv_gib_per_seq> \
  --decode-bound talker@1=<weight_gib>,<codec_tokens_per_audio_s>,<kv_gib_per_seq> \
  --compute-bound code2wav@2=<cfm_tflop_per_audio_s>,tf32 \
  --compute-bound code2wav@2=<vocoder_tflop_per_audio_s>,fp32 \
  --launch-bound talker@1=<kernels_per_step>,<codec_tokens_per_audio_s>,graph \
  --launch-bound code2wav@2=<kernels_per_chunk>,<chunks_per_audio_s>,eager
```

`@STAGE` is the stage id used by `--stage-util` or `--pmon`; with it the
report prints each stage's measured `g_i` beside its bound and the % of SOL.
Terms with the same name add up per class (CFM and vocoder above), and the
stage's floor is the larger of its compute and memory sums. KV GiB per
sequence is the KV cache at the mean context length of the workload (layers x
2 x KV heads x head dim x bytes x tokens); omit it for short contexts. Use
`graph` launch latency for steps that replay a captured graph at the cell's
batch and `eager` for steps that fall back (check the startup log for the
graph batch limit).

Add the per-request terms for the end-to-end floor
([bound-models.md](bound-models.md#end-to-end-rtf-fixed-and-per-audio-second-cost)).
Each is `TFLOP,DTYPE[,WEIGHT_GIB[,KERNELS,MODE]]` per request, gating `@STAGE`:

```bash
  --fixed-bound thinker_prefill@0=<2 * thinker_params * uncached_tokens / 1e12>,bf16,<thinker_weight_gib> \
  --fill-bound thinker_first_token@1=0,bf16,<thinker_weight_gib> \
  --fill-bound talker_first_chunk@2=0,bf16,<codec_chunk_frames * talker_weight_gib>
```

- Prefill FLOPs are `2 * params * tokens` for the dense part; add attention
  and encoder FLOPs when the prompt carries audio or images. Use uncached
  tokens only when prefix caching is on: `mean(input_lens) - floor(min(input_lens)
  / block_size) * block_size` from the result JSON is a safe floor.
- A fill term over `k` sequential decode steps reads the weights `k` times,
  so its `WEIGHT_GIB` is `k * weight_gib`. The first codec chunk is the
  deploy's `codec_chunk_frames` (or the request's `initial_codec_chunk_frames`
  when it is set).
- Append `,<kernels>,eager|graph` when the kernel count of that path is known.

MiniCPM-o 4.5 turn on L20X, as a worked example: Thinker 14.1 GiB bf16
(`7.57e9` params), mean input 137.9 tokens with the shortest at 130 and block
size 16, so at least 9.9 uncached tokens, `2 * 7.57e9 * 9.9 = 0.149 TFLOP`;
`codec_chunk_frames: 25` with a 0.648 GiB Talker gives `16.2` GiB for the
first chunk.

Count kernels per step from one short nsys capture of the stage, in a run
that produces no timing numbers:

```bash
nsys profile -t cuda,nvtx --cuda-graph-trace node -o $OUT/c64_nsys \
  vllm serve ... &   # same deploy, profiling only
nsys stats -r cuda_gpu_kern_sum $OUT/c64_nsys.nsys-rep
```

Divide the stage process's kernel count by the number of steps it ran in the
window. `--cuda-graph-trace node` is needed to see kernels inside replayed
graphs.

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
   every device and pass one file per stage as
   `--stage-util c64=<stage>:$OUT/c64_dev<n>.csv`; each device's busy share is
   that stage's `u_i`. Inter-device transfer differs from the shared-device
   deployment, so use this for `g_i`, not for the shared-device RTF. This is
   the only method that yields absolute `g_i` on Ascend.
2. **Shared device, MPS off (CUDA only).** Add `platforms: {cuda: {cuda_mps: false}}` to
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

4. **Per-kernel correlation (diagnosis only).** When a stage sits far below
   SOL, the nsys capture above, grouped by kernel name, shows which ops own
   the stage's time; `ncu` on its top kernels gives their compute and memory
   SOL%. This locates the gap inside the stage; it does not replace `g_i`,
   and profiled runs never supply timing numbers.

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
| Co-location | `cuda_mps` (CUDA only), `gpu_memory_utilization`, placement |
| Replicas | replica profiles for the binding stage |

Keep the best value per cell. Numerics-changing levers go in a separate
column with their quality delta.

## Device peaks

```bash
python .claude/skills/omni-perf-ceiling/scripts/measure_peaks.py --json $OUT/peaks.json
```

Run it on an otherwise idle device. It reports copy bandwidth and dense matmul
throughput for BF16, FP16, IEEE FP32, and the device's reduced-precision FP32
mode: TF32 on CUDA, HF32 on Ascend. `--device` forces a backend when both are
importable. `launch_latency_us` holds the per-op cost of tiny kernels issued
eagerly (`eager`, PyTorch dispatch included) and replayed from a captured
graph (`graph`); `graph` is null when capture is not available. Launch latency
depends on the host CPU as well as the device, so measure it on the serving
host.

## Pitfalls

- Too few prompts at high concurrency: ramp-up and ramp-down dominate and
  Little's law fails. Use at least `10 x C`.
- Warmup shorter than `C` requests leaves first-shape graph capture inside
  the measured window.
- Mixed output modes: a single-file engine and a streaming engine differ in
  work per request.
- Profilers on: never report profiler-enabled latency or throughput.
- Bounds without arithmetic: a bound column that cannot be recomputed from the
  report's substituted formulas is not evidence.
- Shared hosts: record other processes on the device; GPU utilisation from
  another tenant invalidates the attribution.
