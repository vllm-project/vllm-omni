# Ceiling Experiments

## Contents
- Admission overlay
- Cell commands
- Per-stage GPU attribution
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

The benchmark saves `benchmark_end_time` immediately after the timed run, before
metric computation or optional WER/accuracy evaluation. The report subtracts
`duration` from this timestamp to locate the measured window. The legacy `date`
is the result-creation time and must not be used: postprocessing can shift it by
minutes. Benchmark timestamps and sampler timestamps must use the same local
clock and timezone (synchronize hosts when the client and server are separate).
Older results without `benchmark_end_time` cannot locate the window; the report
keeps the whole sample file and warns. Use a sampler already scoped to the measured
run in that case, or rerun with benchmark-window metadata.

Timestamped samples are restricted to the measured window, trimming two seconds
from each end only when the run lasts more than four seconds. Shorter runs keep
their full measured span. No usable samples means unavailable utilisation (`null`
in JSON, `-` in the report), with a warning; a measured zero remains zero. If no
stage has a positive measured device time, the report cannot infer a finite
throughput bound from `--stage-util`.

Build the ledger:

```bash
python .claude/skills/omni-perf-ceiling/scripts/ceiling_report.py \
  --cell c1=$OUT/c1.json:$OUT/c1_bench.log:$OUT/c1_gpu.csv \
  --cell c64=$OUT/c64.json:$OUT/c64_bench.log:$OUT/c64_gpu.csv \
  --json $OUT/ledger.json
```

Add calibrated estimates once peaks and per-stage work are known. Work counts come
from the checkpoint's reference implementation, the transformers modeling code
the checkpoint ships (`trust_remote_code`), not from the vLLM-Omni code path
under test. `scripts/count_reference_work.py` builds the reference model on the
meta device (no weights, no GPU) and counts with `FlopCounterMode`; audio
decoders that are not `nn.Module` children of the reference model are counted
by a recipe that runs the reference streaming path on CPU. The workload
(tokens per audio second, mean context, uncached prompt tokens) comes from the
result JSON:

```bash
python .claude/skills/omni-perf-ceiling/scripts/count_reference_work.py \
  --model $CKPT --config-override init_vision=false --config-override init_audio=false \
  --decoder thinker@0=llm --decoder talker@1=tts.model+tts.head_code \
  --prompt-tokens <mean input tokens> --context thinker=<mean context> --context talker=<mean context> \
  --code2wav minicpmo-token2wav@2=$CKPT/assets/token2wav --prompt-audio-s <reference audio s> \
  --prompt-setup-per-request code2wav=<effective setup executions per request> \
  --tokens-per-audio-s thinker=<text tokens per audio-s> --tokens-per-audio-s talker=<codec tokens per audio-s> \
  --mean-context thinker=<tokens> --mean-context talker=<tokens> \
  --uncached-tokens thinker=<uncached prompt tokens> \
  --fill-steps thinker@1=<Thinker steps before the Talker> --fill-steps talker@2=<first codec chunk> \
  --dtype code2wav.flow=tf32 --dtype code2wav.vocoder=fp32 --json $OUT/reference_work.json
```

It prints the JSON and the `ceiling_report.py` flags:

- `--decode-bound NAME@STAGE=STREAMED_GIB,TOKENS_PER_AUDIO_S,KV_GIB_PER_SEQ,GFLOP_PER_TOKEN,DTYPE`
  per decoder. Streamed GiB is every parameter a step reads, leaving out
  embedding tables, which are gathered by row. KV GiB per sequence is the
  reference `DynamicCache` bytes per token times the mean context. GFLOP per
  token is one decode step at that context, with logits for the last position
  only, as `generate()` does.
- `--fixed-bound NAME_prefill@STAGE`: prefill FLOPs per token, averaged over
  `--prompt-tokens` and attention included, times the uncached tokens.
- `--fill-bound NAME_first_K@GATE`: `K` decode steps, which read the weights
  `K` times.
- `--compute-bound code2wav@STAGE` rows per audio second and
  `--fill-bound code2wav_<row>_first@out` for the first chunk, from the
  recipe's steady and first chunks.
- `--fixed-bound code2wav_prompt_setup@STAGE`: reference prompt setup FLOPs
  times `--prompt-setup-per-request code2wav=RATE`, using the flow compute dtype.
  `RATE` is effective setup executions divided by requests, including cache hits
  and sharing one setup among requests with the same prompt in a batch. It is
  in `[0,1]`: `0` explicitly assumes no setup work, `1` assumes one setup per
  request. For example, one setup shared by four requests gives `0.25`.
  The JSON records `prompt_setup_per_request`; omission leaves it `null` and
  prints an **unmodeled setup** warning, never silently assuming cache hits.
  If cache behavior is unknown, compare explicit `0` and `1` sensitivity
  scenarios and label the assumptions. Count at the runtime's clipped reference
  audio length and sampling steps. This average work does not model miss-tail
  latency, speaker frontend work or additional serial execution constraints.

Inputs that are not work counts:

- **Numerics.** Decoders default to the checkpoint dtype and Token2wav to the
  reference's fp32. `--dtype` sets what the measured deployment runs (TF32 flow
  model above). For decoders it also selects the default weight and KV storage
  precision, so their byte counts change with deployment precision. TF32/HF32
  use FP32 storage. Override storage separately with `--weight-dtype thinker=bf16`
  or `--kv-dtype thinker=fp16` when it differs from computation; supported storage
  types are `bf16`, `fp16` and `fp32`. Counts record reference, compute, weight and
  KV dtypes. Quantized storage requires a separate counting recipe. Report the
  reference dtype's estimate beside the deployment's.
- **Algorithm constants.** Chunk size, sampling steps, look-ahead, attention
  cache window and vocoder overlap come from the reference (MiniCPM-o:
  `s3_stream_chunk_size: 25`, `s3_stream_n_timesteps: 10`, prelook 3, cache
  prompt + 100 frames, 8 mel frames of HiFT overlap). When the deployment
  differs, that difference is an algorithm lever: bound both settings and say
  which the deployment uses.
- **Workload.** Prefix caching: `mean(input_lens) - floor(min(input_lens) /
  block_size) * block_size` from the result JSON is a safe floor for uncached
  tokens. Encoders count only when the prompt carries audio or images; add
  them as `--decoder` or recipe rows when it does.

Other checkpoints:

- Name the reference model's own attribute paths (`--decoder NAME=ATTR[+HEADS]`;
  ATTR must take `inputs_embeds` and `past_key_values`).
- Add a recipe to `CODE2WAV_RECIPES` for an audio decoder outside the model
  tree.
- `--block-import` skips an optional dependency that fails to import (a
  torchaudio built for another CUDA).

Hand counts (`2 * params * tokens`, safetensors prefix sums) are a fallback
only when no reference implementation can be built; say so in the report.

`@STAGE` is the stage id used by `--stage-util` or `--pmon`; with it the
report prints each stage's measured `g_i` beside its calibrated estimate and
the estimate/measured ratio.
Each flag is one row, a group of kernels; the stage's device floor is the sum
of each row's `max(compute, memory)` (flow model and vocoder are two rows), so
rows of one stage must not cover the same kernels. Add launch rows when kernel
counts are known (below): `graph` latency for steps that replay a captured
graph at the cell's batch, `eager` for steps that fall back (check the startup
log for the graph batch limit):

```bash
  --launch-bound talker@1=<kernels_per_step>,<codec_tokens_per_audio_s>,graph \
  --launch-bound code2wav@2=<kernels_per_chunk>,<chunks_per_audio_s>,eager
```

The supplied fixed/fill terms give the aggregate end-to-end estimate
([bound-models.md](bound-models.md#fixed-work-fill-and-aggregate-estimates)),
the TTFT estimate (terms gating `@0`) and the TTFP estimate (all terms); `@out` marks
Code2Wav's work on the first chunk, which only the first packet waits for.
Append `,<kernels>,eager|graph` to a per-request term when the kernel count of
that path is known.

MiniCPM-o 4.5 turn on L20X, as a worked example:

- **Workload.** Mean input 137.9 tokens, Thinker output 14.1 and Talker output
  111.3 tokens, so mean contexts are 145 and 70. At least 9.9 tokens per
  request miss the prefix cache. The 6 s reference audio gives a 400-frame
  flow cache.
- **Thinker.** The reference gives 14.1 GiB streamed, 15.2 GFLOP per token
  and 147456 KV bytes per token.
- **Talker.** It streams 0.361 GiB per step: the Llama backbone plus
  `head_code`, without the 0.22 GiB `emb_text` table that a parameter sum
  counts.
- **Code2Wav.** The flow model does 0.159 TFLOP per audio second (0.158 on
  the first chunk). The vocoder does 0.0355, including its 8-frame overlap
  (0.0306 on the first chunk). Setting up the prompt cache costs 0.93 TFLOP
  once per reference audio.
- **TTFP bound.** `3.5 + 3.5 + 2.3 + 0.4 + 0.6 = 10.3` ms (prefill, one
  Thinker step, 25 Talker steps, flow model and vocoder on the first chunk)
  against 186 ms measured at C1.

Audit the sweep and admission alongside:

```bash
  --series c1,c16,c64 \
  --kv-capacity 0=<thinker GPU KV cache size tokens>,<max prompt + output tokens> \
  --kv-capacity 1=<talker GPU KV cache size tokens>,<max talker prompt + codec tokens>
```

`--series` names cells of one configuration; the report warns when throughput,
aggregate RTF, TTFT or TTFP falls as C rises. KV cache tokens come from each
stage's `GPU KV cache size` startup line; output tokens per stage from
`num_tokens_out` in `request_stage_metrics`.

An analytical per-op planner for LLMs (for example
[HuggingArch](https://github.com/shenh10/HuggingArch)) is a cross-check for the
Thinker's counts. Take its counts, not its times: it prices them with
data-sheet peaks and a fixed launch constant, while this ledger uses the
measured peaks. It does not cover audio stages.

[mindie-sim](https://github.com/mindie-agent/sim) contributes two readings,
both printed beside the bound
([bound-models.md](bound-models.md#efficiency-targets-and-profiled-measurements)):

```bash
  --efficiency-target compute=0.8,memory=0.8 \
  --profiled code2wav=<profiled seconds per audio-s at this batch>
```

The efficiency target projects the shared-device floor with every streaming
row at that fraction of the measured peak. The profiled value is one row's
measured duration; the report prints `max(roofline, profiled)` as that
implementation's estimate and warns when the profile beats calibration. Its
event-engine schedule is not used. Shared demand is accumulated per resource
before taking the maximum; independent stage durations must not be summed.
An efficiency target does not measure achieved efficiency or attribute the
residual to host/scheduling. Mean RTF estimates are unavailable without
per-request work counts. These are empirical estimates, not physical limits.

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

For a compute stage without a recipe, count FLOPs on CPU with
`torch.utils.flop_counter.FlopCounterMode` around the reference module's
streaming call at the reference chunk shape, including its caches and
overlaps. Remove weight-norm parametrizations first
(`torch.nn.utils.parametrize`); the counter cannot see through them.

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
host. These microbenchmarks are calibration, not universal throughput maxima
or launch minima; other shapes and implementations can beat them.

## Pitfalls

- Too few prompts at high concurrency: ramp-up and ramp-down dominate and
  Little's law fails. Use at least `10 x C`.
- Warmup shorter than `C` requests leaves first-shape graph capture inside
  the measured window.
- Mixed output modes: a single-file engine and a streaming engine differ in
  work per request.
- Profilers on: never report profiler-enabled latency or throughput.
- Attributing latency residuals without traces: at any C, inefficient kernels,
  extra work, host, transfer and queueing can contribute. Use C1 as a baseline,
  then profile the critical path before assigning the gap.
- Bounds without arithmetic: a bound column that cannot be recomputed from the
  report's substituted formulas is not evidence.
- Shared hosts: record other processes on the device; GPU utilisation from
  another tenant invalidates the attribution.
