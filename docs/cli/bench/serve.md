# vLLM-Omni Benchmark CLI Guide

The vllm bench command launches the vLLM-Omni benchmark to evaluate the performance of multimodal models.

## Notes

vLLM-Omni registers the `openai-chat-omni`, `openai-audio-speech`, `openai-image-edits-omni`, `daily-omni`, `openai-realtime-duplex`, `openai-realtime-tts`, and `openai-realtime-chat` serving benchmark backends. It also adds the `omniinteract` dataset.

`openai-chat-omni` retries explicit `ServiceUnavailableError` responses with
code 503 up to three times, provided no text, reasoning, tool, audio, or image
output has arrived. This includes retryable multimodal cache misses reported
inside an HTTP 200 SSE stream. Other server errors and partial responses remain
failed requests. Request latency and time to first output include failed
attempts and retry delays.

## Basic Parameter Description

You can use `vllm bench serve --omni --help=all` to get descriptions of all parameters. The commonly used parameters are described below:

- `--omni`
  Enable Omni (multimodal) mode, supporting multimodal inputs and outputs such as images, videos, and audio.

- `--backend`
  Specify the backend adapter. vLLM-Omni adds `openai-chat-omni`, `openai-audio-speech`, `openai-image-edits-omni`, `daily-omni`, `openai-realtime-duplex`, `openai-realtime-tts`, and `openai-realtime-chat` to the upstream vLLM backend choices.

  The three WebSocket backends all talk to `/v1/realtime`, but they drive different turn shapes:

  | Backend | Turn shape |
  | --- | --- |
  | `openai-realtime-duplex` / `openai-realtime-tts` | One session per utterance; the target text rides the session context (`duplex_initial_user_text`) and the client streams silence so a model-native duplex session has audio units to speak on. Reports server-side `ttft_ms`/`ttfp_ms` when the server returns them, and Stage-0 engine TPOT. |
  | `openai-realtime-chat` | One session per utterance; the target text is a `conversation.item.create` message and real reference speech is streamed as an ordinary audio turn. Reports client-observed timings measured from the end of that speech. Requires `--seed-tts-reference-as-input`. |

  Because those two report from different origins, their TTFT/TTFP/RTF numbers are not comparable with each other.

- `--model`
  The model identifier to load, filled according to the models supported by vLLM-Omni.

- `--endpoint`
  The API endpoint exposed externally, to which clients send their requests.

- `--dataset-name`
  The name of the dataset used; `random-mm` generates random multimodal inputs (images, videos, audio),
  `omniinteract` replays official OmniInteract videos, and `videomme` loads Video-MME MCQ videos
  (`lmms-eval/Video-MME`, MiniCPM OmniEvalKit packing).

- `--num-prompts`
  The total number of requests to send, an integer.

- `--max-concurrency`
  "Maximum number of concurrent requests. This can be used "
        "to help simulate an environment where a higher level component "
        "is enforcing a maximum number of concurrent requests. While the "
        "--request-rate argument controls the rate at which requests are "
        "initiated, this argument will control how many are actually allowed "
        "to execute at a time. This means that when used in combination, the "
        "actual request rate may be lower than specified with --request-rate, "
        "if the server is not processing requests fast enough to keep up."

- `--request-rate`
  "Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process or gamma distribution "
        "to synthesize the request arrival times."

- `--ignore-eos`
  "Set ignore_eos flag when sending the benchmark request."

- `--metric-percentiles`
  Comma-separated list of percentiles for selected metrics. "
        "To report 25-th, 50-th, and 75-th percentiles, use \"25,50,75\". "
        "Default value is \"99\"."
        "Use \"--percentile-metrics\" to select metrics.

- `--percentile-metrics`
        "Comma-separated list of selected metrics to report percentiles."
                    "This argument specifies the metrics to report percentiles."
                    'Allowed metric names are "ttft", "tpot", "itl", "ttfc", "tpoc", "icl", '
                    '"tpop", "e2el", "audio_ttfp", "audio_rtf", "audio_duration". '

- `--print-stage`
Print per-stage benchmark metrics for --omni serving when stage metrics are returned by the server. Disabled by default.

- `--save-result`
Specify to save benchmark results to a json file

- `--save-detailed`
"When saving the results, whether to include per request "
        "information such as response, error, ttfs, tpots, etc."

- `--result-dir`
 "Specify directory to save benchmark json results."
        "If not specified, results are saved in the current directory."

- `--result-filename`
"Specify the filename to save benchmark json results."
        "If not specified, results will be saved in "
        "{label}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"

- `--random-prefix-len`
  Number of fixed prefix tokens before the random context in a request.
  The total input length is the sum of random-prefix-len and a random
  context length sampled from [input_len *(1 - range_ratio),
  input_len* (1 + range_ratio)].Only the random and random-mm modes
  support this parameter.

- `--random-input-len`
  Number of input tokens per request.Only the random and random-mm modes support this parameter.

- `--random-output-len`
  Number of output tokens per request.Only the random and random-mm modes support this parameter.

- `--random-range-ratio`
  Range ratio for sampling input/output length,
  used only for random sampling. Must be in the range [0, 1) to define
  a symmetric sampling range
  [length *(1 - range_ratio), length* (1 + range_ratio)].
  Only the random and random-mm modes support this parameter.

- `--random-mm-base-items-per-request`
  Base number of multimodal items per request for random-mm.
  Actual per-request count is sampled around this base using
  --random-mm-num-mm-items-range-ratio.
  Only the random-mm mode supports this parameter.

- `--random-mm-limit-mm-per-prompt`
  Per-modality hard caps for items attached per request, e.g.
  '{"image": 3, "video": 1, "audio": 1}'. The sampled per-request item
  count is clamped to the sum of these limits. When a modality
  reaches its cap, its buckets are excluded and probabilities are
  renormalized.
  Only the random-mm mode supports this parameter.

- `--random-mm-num-mm-items-range-ratio`
  Range ratio r in [0, 1] for sampling items per request.
  We sample uniformly from the closed integer range
  [floor(n*(1-r)), ceil(n*(1+r))]
  where n is the base items per request.
  r=0 keeps it fixed; r=1 allows 0 items. The maximum is clamped
  to the sum of per-modality limits from
  --random-mm-limit-mm-per-prompt.
  An error is raised if the computed min exceeds the max.
  Only the random-mm mode supports this parameter.

- `--random-mm-bucket-config`
  The bucket config is a dictionary mapping a multimodal item
  sampling configuration to a probability.
  Currently allows for 3 modalities: audio, images and videos.
  A bucket key is a tuple of (height, width, num_frames)
  The value is the probability of sampling that specific item.
  Example:
  --random-mm-bucket-config
  "{(256, 256, 1): 0.5, (720, 1280, 16): 0.4, (0, 1, 5): 0.10}"
  First item: images with resolution 256x256 w.p. 0.5
  Second item: videos with resolution 720x1280 and 16 frames
  Third item: audios with 1s duration and 5 channels w.p. 0.1
  OBS.: If the probabilities do not sum to 1, they are normalized.
  Only the random-mm mode supports this parameter

## Usage Examples

### Online Benchmark

<details class="admonition abstract" markdown="1">
<summary>Show more</summary>

First start serving your model:

```bash
vllm serve Qwen/Qwen2.5-Omni-7B --omni
```

Then run the benchmarking for sharegpt:

```bash
# download dataset
# wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
vllm bench serve \
  --omni \
  --port 43845 \
  --model /home/models/Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --endpoint /v1/chat/completions \
  --backend openai-chat-omni \
  --num-prompts 2 \
  --dataset-name sharegpt \
  --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
  --percentile-metrics ttft,tpot,itl,e2el
```

If successful, you will see the following output:

```text
============ Serving Benchmark Result ============
Successful requests:                     2
Failed requests:                         0
Benchmark duration (s):                  81.63
Request throughput (req/s):              0.02
Peak concurrent requests:                2.00
----------------End-to-end Latency----------------
Mean E2EL (ms):                          56966.13
Median E2EL (ms):                        56966.13
P99 E2EL (ms):                           81016.80
================== Text Result ===================
Total input tokens:                      36
Total generated tokens:                  5926
Output token throughput (tok/s):         72.60
Peak output token throughput (tok/s):    103.00
Peak concurrent requests:                2.00
Total Token throughput (tok/s):          73.04
---------------Time to First Token----------------
Mean TTFT (ms):                          124.76
Median TTFT (ms):                        124.76
P99 TTFT (ms):                           156.10
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          481.30
Median TPOT (ms):                        481.30
P99 TPOT (ms):                           947.55
---------------Inter-token Latency----------------
Mean ITL (ms):                           25.11
Median ITL (ms):                         0.33
P99 ITL (ms):                            25.17
================== Audio Result ==================
Total audio duration generated(s):       3.95
Total audio frames generated:            94890
Audio throughput(audio duration/s):      0.05
==================================================
```

Or run the benchmarking for random:

```bash
vllm bench serve \
  --omni \
  --port 43845 \
  --endpoint /v1/chat/completions \
  --backend openai-chat-omni \
  --model /home/models/Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --dataset-name random \
  --num-prompts 2 \
  --random-prefix-len 5 \
  --random-input-len 10 \
  --random-output-len 100 \
  --percentile-metrics ttft,tpot,itl,e2el,audio_ttfp,audio_rtf \
  --ignore-eos
```

If successful, you will see the following output:

```text
============ Serving Benchmark Result ============
Successful requests:                     2
Failed requests:                         0
Benchmark duration (s):                  3.89
Request throughput (req/s):              0.51
Peak concurrent requests:                2.00
----------------End-to-end Latency----------------
Mean E2EL (ms):                          3824.76
Median E2EL (ms):                        3824.76
P99 E2EL (ms):                           3888.54
================== Text Result ===================
Total input tokens:                      30
Total generated tokens:                  10101
Output token throughput (tok/s):         2595.57
Peak output token throughput (tok/s):    111.00
Peak concurrent requests:                2.00
Total Token throughput (tok/s):          2603.28
---------------Time to First Token----------------
Mean TTFT (ms):                          117.15
Median TTFT (ms):                        117.15
P99 TTFT (ms):                           142.69
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          0.73
Median TPOT (ms):                        0.73
P99 TPOT (ms):                           0.74
---------------Inter-token Latency----------------
Mean ITL (ms):                           16.47
Median ITL (ms):                         16.19
P99 ITL (ms):                            52.55
================== Audio Result ==================
Total audio duration generated(s):       15.79
Total audio frames generated:            379050
Audio throughput(audio duration/s):      4.06
---------------Time to First Packet---------------
Mean AUDIO_TTFP (ms):                    3701.37
Median AUDIO_TTFP (ms):                  3701.37
P99 AUDIO_TTFP (ms):                     3762.25
-----------------Real Time Factor-----------------
Mean AUDIO_RTF:                          0.47
Median AUDIO_RTF:                        0.47
P99 AUDIO_RTF:                           0.48
==================================================
```

Notes:
We use audio generation time / audio duration to calculate RTF.

</details>

### Omni-DuplexEval Realtime Benchmark

Omni-DuplexEval uses the same serving-benchmark lifecycle as OmniInteract. Each
selected video is one session/request, not one request per model response:

```bash
vllm bench serve --omni \
  --backend openai-realtime-duplex \
  --dataset-name omni-duplex-eval \
  --dataset-path Hothan/Omni-DuplexEval \
  --model openbmb/MiniCPM-o-4_5 \
  --base-url http://127.0.0.1:8000 --endpoint /v1/realtime \
  --duplex-eval-ref-audio /path/to/reference.wav \
  --duplex-eval-response-root ./duplex-eval-runs \
  --duplex-eval-split PR_correction \
  --num-prompts 3 --disable-shuffle --max-concurrency 1 --num-warmups 0 \
  --ready-check-timeout-sec 0 --save-result
```

`--dataset-path` accepts the existing loader's HF dataset ID, local dataset mirror,
Parquet file or JSON/JSONL manifest. It defaults to `Hothan/Omni-DuplexEval`.
Use `--duplex-eval-family`, `--duplex-eval-ids` and `--duplex-eval-media-root` for
filtering and local media. The default is three samples and single concurrency;
`--num-prompts 0` selects all, and oversized counts use every available sample
without repeating it. `--seed` controls shuffling; `--disable-shuffle` preserves
dataset order.

For a Qwen3-Omni server started with `vllm_omni/deploy/qwen3_omni_duplex.yaml`,
use the same entrypoint with server-detected turns and no reference audio:

```bash
vllm bench serve --omni \
  --backend openai-realtime-duplex --endpoint /v1/realtime \
  --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --dataset-name omni-duplex-eval --dataset-path Hothan/Omni-DuplexEval \
  --duplex-eval-split PR_correction --duplex-eval-turn-detection server_vad \
  --duplex-eval-instructions "Reply briefly to the user's latest request." \
  --duplex-eval-response-root ./duplex-eval-qwen3 \
  --num-prompts 3 --disable-shuffle --max-concurrency 1 --num-warmups 0 \
  --ready-check-timeout-sec 0 --save-result
```

`--duplex-eval-turn-detection` defaults to `none`, preserving the native duplex
runner's explicit commit. With `server_vad`, the runner sends audio in 32 ms appends
(one 512-sample detector frame at 16 kHz), appends trailing silence, and waits for
observed speech/responses to settle. A larger append can span the end of one turn
and the start of the next, hiding the earlier endpoint. Video sampling still uses
`--duplex-eval-fps`; native duplex retains its 1000 ms default audio units. The
sample metadata records the effective `unit_ms`. Audio pacing uses a cumulative
monotonic deadline so sending and frame processing do not add a full extra sleep
to every append. The standalone `generate` command rejects `server_vad` combined
with `--pace as-fast-as-possible`, which compresses turn and playback timing.
Settling is not an acknowledgement
that all final input has been processed. Frames use the transport advertised by
the server: native video frames or bounded image conversation items for Qwen3.
`--duplex-eval-instructions` defaults to the native duplex prompt; passing an empty
string uses the server's default instructions. `--duplex-eval-ref-audio` is optional
for Qwen3 and remains available for models that require a reference voice.

The CUDA nightly configuration
`tests/dfx/perf/tests/test_qwen3_omni_duplex_omni_duplex_eval.json` uses the common
performance runner on three GPUs (two for Qwen3-Omni, one for the text judge).
It measures three `PR_correction` sessions with post-run judging and two
`RTD_counting` sessions without judging, preserving dataset order. The checks require
complete artifacts, completed observed responses and measured text/audio latency
for every session; judging must cover every selected PR sample. Scores are reported
without a quality threshold. RTD scoring requires a vision-capable judge and is
outside this configuration. No hardware latency baseline is configured.
These nightly checks do not establish that every input utterance produced a
response. The dataset provides no expected VAD turn count, and a `PR_correction`
sample can contain a single user statement, so a blanket two-response minimum is
not valid. Deterministic CPU tests cover adjacent speech boundaries separately.

Inputs retain the standalone runner's real-time pacing, question audio, media
clock and 1 FPS default (`--duplex-eval-fps`). `--duplex-eval-mix` selects the
session audio: `question` (default) streams only the instruction audio, while
`soundtrack` sums it with the clip's own audio track, sample by sample with int16
clipping. The dataset pads each instruction to the clip length. Clip audio is placed
at its presentation time relative to the video start, including a leading silence
when the track starts later, so the overlay keeps the original timestamps. Use
`soundtrack` for samples that depend on what is said
or heard in the clip, such as the word-repetition prompts in
`PR_post_event_reminder` and `RTD_Omni`; a clip without an audio track keeps the
instruction alone. The standalone `generate` command accepts the same choice as `--mix`.
The MiniCPM-o Omni-DuplexEval nightly (`tests/dfx/perf/tests/test_minicpmo_4_5_omni_duplex_eval.json`)
runs with `soundtrack`; the Qwen3-Omni `server_vad` nightly keeps `question`.

Each session decodes its media in a
worker thread after acquiring a concurrency slot, and releases the decoded input
when the request finishes. `--max-concurrency` therefore bounds the number of
decoded clips held in memory; the reference audio is encoded once and shared.
Preparation is excluded from individual session latency, but any preparation
stalls are included in the total benchmark duration and throughput.
`--omni-request-timeout-s` bounds each complete session after preparation (default
900 seconds). Readiness and warmups replay complete
sessions but do not contribute metrics or judge artifacts; the example disables
both to avoid that extra runtime.

The terminal and `--save-result` output report standard serving metrics plus the
shared `duplex_request_metrics`, `duplex_session_metrics` and `duplex_stream_*`
fields. TTFT/Audio TTFP/RTF follow the same per-response definitions as
OmniInteract; stream metrics include concurrent real-time input. TPOT/ITL require
engine token timings and are unavailable when those timings are absent. E2EL
includes the whole session, including input streaming and drain, so it is not a
model-only generation latency.

A video can produce several responses. Normal barge-in cancellation ends one
response without failing the sample; its collected text remains available to the
judge. The runner drains active responses before closing the session. Responses
created during close are recorded in `unfinished_response_ids` in the metadata;
server errors, failed responses, and drain/close timeouts still fail the sample.

`--duplex-eval-response-root` is a **parent directory** and may already exist.
Each invocation creates an exclusive `run-*` subdirectory, so repeated runs and
concurrency/QPS sweeps do not overwrite or reuse earlier responses. The actual
run path is logged and returned in `omni_duplex_eval.response_root` in the saved
benchmark result. Only successful measured sessions publish `<split>/<id>.json`
and `<id>.meta.json`, using the upstream judge format: RTD scoring reads its
`sentences`, and PR scoring places its per-output `chunks` (`text`,
`current_time`) in the 10-second reminder window. A response that is a bare
sentence list is scored by sentence start. The standalone `generate` command
refuses to reuse a proactive-reminder response that has no `chunks`; RTD
responses are reused from their sentences. Publication occurs after benchmark timing, alongside
`duplex_metrics.json`; artifact errors are recorded in the saved result's
`omni_duplex_eval` summary without discarding service measurements.

Accuracy scoring is optional. To score immediately after the performance run,
add these flags to the `vllm bench serve` command above:

```bash
  --duplex-eval-evaluate \
  --duplex-eval-judge-base-url http://127.0.0.1:8001/v1 \
  --duplex-eval-judge-model your-judge-model
```

The judge must already be running; the benchmark does not launch it. Proactive
reminder (PR) tasks use a text judge, while real-time description (RTD) tasks
require a vision-capable judge. The default `--duplex-eval-judge-video-mode video_url`
requires the judge to support video inputs and access the same local video paths.
Use `--duplex-eval-judge-video-mode frame-sample` for a remote image-capable judge;
this changes content evaluation to the existing sampled-frame variant, not full-video judging.

Scoring reuses the standalone evaluator and runs **after performance timing**.
The terminal and saved benchmark result (`omni_duplex_eval.accuracy`) report scores
and coverage: total measured sessions, evaluated sessions, skipped failed/unpublished
sessions, and judge errors. Partial coverage is marked `partial`; no evaluated
sessions is `failed`, not a zero-error accuracy result. Scores are conditional on
the evaluated subset, not an accuracy score for all attempted sessions. Warmups
are never scored. Judge failures do not discard completed performance measurements.
The saved `omni_duplex_eval.published_with_unfinished_responses` counts published
samples with non-empty `unfinished_response_ids` in their metadata. A nonzero count
also emits a warning. These samples remain eligible for scoring using their collected
text, so interpret accuracy alongside this count. It only covers observed responses;
zero does not establish that every final input triggered a completed reply.
Per-sample scores and `evaluation_summary.json` are saved under the run's
`evaluation/` directory. `--duplex-eval-eval-workers` controls scoring concurrency
(default 1), and `--duplex-eval-judge-timeout-s` bounds each judge request (default 600).

The independent scoring commands remain available to retry scoring or change
judges without repeating inference. The `--limit 3` example below matches the
serving command above when `--disable-shuffle` is used and all three measured
samples succeed and publish their artifacts. Replace `run-XXXXXXXX` below with
the reported run directory:

```bash
vllm bench omni-duplex-eval --omni evaluate \
  --dataset Hothan/Omni-DuplexEval --split PR_correction --limit 3 \
  --response-root ./duplex-eval-runs/run-XXXXXXXX --score-root ./duplex-eval-scores-1 \
  --judge-base-url http://127.0.0.1:8001/v1 --judge-model your-judge-model
vllm bench omni-duplex-eval --omni summarize --score-root ./duplex-eval-scores-1
```

`evaluate --limit N` selects the first N matching samples in dataset order. It
does not reproduce the shuffle from `bench serve` or filter out failed or
unpublished sessions. To automatically score the actual successfully published
samples, including shuffled or partially successful runs, use
`--duplex-eval-evaluate` during `bench serve`.

The standalone `generate` command remains available. It still skips an existing
sample unless `--overwrite` is set, and it reuses that sample only when the
recorded mix matches (metadata without a mix counts as `question`). A different
mix is rejected instead of returning the other artifact. This integration does
not add nightly jobs. The MiniCPM-o job above selects `soundtrack`.

For shuffled runs, use `evaluate --ids` with the successful selected sample IDs
instead of `--limit`. Failed sessions have no judge input; do not silently drop
them when reporting coverage or accuracy.

This adapter retains the standalone runner's completion policy. Without
`server_vad`, a session finishes once every created response has its
`response.done`, no commit, listen, or response event has arrived for 2 seconds,
and the residual unit flushed by the commit has decided. That decision is a
`response.done` for a response created after `input_audio_buffer.committed`, or a
`response.listen` with `model_listen` after every reply that was still open at
that commit has finished. A reply already open when the clip ends does not
decide the session: completing it normally is what starts the residual unit, and
that unit can still open a response. A cancelled reply does not decide the
session either: barge-in cancels that turn, and the residual response can still
open afterwards. A failed reply does end the wait. The server does not promote
another unit after a failure, and a failed response created after the commit is
itself the residual. The listening outcome is
a session in which the model never spoke; it publishes empty judge input and
scores 0 instead of failing on the drain timeout. The runner withholds the last
PCM sample when the clip ends on a model-unit boundary, so the commit always
flushes a residual unit and that decision covers the end of the clip. A
successful transport/session result is therefore not proof that every final
input produced a response, nor an accuracy score.

### OmniInteract Realtime Benchmark

OmniInteract runs each video as one native-duplex WebSocket sample in the standard serving-benchmark lifecycle:

```bash
vllm bench serve --omni \
  --backend openai-realtime-duplex \
  --dataset-name omniinteract \
  --dataset-path /path/to/OmniInteract \
  --model openbmb/MiniCPM-o-4_5 \
  --base-url http://127.0.0.1:8000 \
  --endpoint /v1/realtime \
  --omniinteract-ref-audio /path/to/reference.wav \
  --omniinteract-output-dir ./omniinteract-artifacts \
  --num-prompts 3 \
  --num-warmups 0
```

`--dataset-path` accepts an extracted directory, `data.tar[.gz]`, or a Hugging Face dataset ID; omitting it uses
`lucky-lance/OmniInteract`. `--num-prompts` is the total across subsets and defaults to 3 for OmniInteract; explicit `0`
selects all and oversize values use all available cases. OmniInteract uses the `/v1/realtime` endpoint. Pass
`--omniinteract-ref-audio` for models that clone a reference voice (MiniCPM-o); models that refuse one (Qwen3-Omni) run
without it. `--omniinteract-instructions` sets the session system prompt; by default none is sent.

For comparable MiniCPM-o scores, explicitly pass the same reference WAV in every run, even though ordinary sessions
can use the model-bundled default when `ref_audio` is omitted. Pass the model's `assets/HT_ref_audio.wav` to benchmark
that default voice. The Qwen3-Omni examples omit reference audio because that model does not accept it.

#### Turn-based duplex models (Qwen3-Omni)

A model-native duplex model decides on its own when to speak, so each clip is streamed and committed once at the end.
A turn-based model only answers committed turns. `--omniinteract-turn-detection server_vad` lets the server's VAD cut the
soundtrack into turns. It sends 32 ms audio appends so each detector frame's endpoint is processed before the next onset;
native duplex keeps its 200 ms appends, and video sampling is unchanged. The client never commits in `server_vad` mode,
appends 1.5 s of trailing silence so a question at the very end still
closes, and waits until no speech or response is open. Frames follow the session capabilities: a model that does not
accept `video_frames` on audio appends but accepts images gets each frame as an `input_image` conversation item, with the
oldest deleted so at most 8 stay in the session (frames too large for the server's 4 MiB image budget are downscaled).
Serve with the duplex deployment, which needs the Silero VAD artifact in the Hugging Face cache (see
[Qwen3-Omni duplex](https://github.com/vllm-project/vllm-omni/blob/main/examples/online_serving/qwen3_omni/README.md)):

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091 \
  --deploy-config vllm_omni/deploy/qwen3_omni_duplex.yaml

vllm bench serve --omni \
  --backend openai-realtime-duplex \
  --endpoint /v1/realtime \
  --base-url http://127.0.0.1:8091 \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --dataset-name omniinteract \
  --omniinteract-turn-detection server_vad \
  --omniinteract-instructions "You are a real-time video assistant. Reply to the user's latest request briefly, in one or two spoken sentences, in the language the user speaks." \
  --omniinteract-output-dir ./omniinteract-qwen3 \
  --num-prompts 3
```

The deployment serves one session at a time unless `duplex_session.max_sessions` is raised, so keep
`--max-concurrency 1`.

Latency is reported per response in the `Duplex Per-Response Latency` block and as flat result keys
(`num_/mean_/median_/p99_` + name), where every response counts once however many a session had. The standard
`Mean TTFT` / `audio TTFP` rows instead average each session first, and E2EL and request throughput follow the video
length because input is paced in real time.

| Result key suffix | Starts at | Ends at |
| --- | --- | --- |
| `duplex_response_ttft_ms` / `duplex_response_ttfp_ms` | server: request accepted by the engine | first text / audio output |
| `duplex_client_ttft_ms` | client: `response.created` received | first non-empty text delta |
| `duplex_client_ttfp_ms` | client: last `input_audio_buffer.append` send before this `response.created` (after the previous `response.created` or stream start), when that send was recorded | first audio delta |

The server timer starts only after the model plugin prepared the prompt and the engine accepted it, so it leaves out
prompt preparation; for Qwen3-Omni that is decoding the retained images, the chat template, and multimodal
preprocessing. `duplex_client_ttft_ms` starts when `response.created` is received, after the server detects the end of
the turn, so Qwen3-Omni's prompt preparation after that event is included. OmniInteract and Omni-DuplexEval record
append sends, so `duplex_client_ttfp_ms` starts at the last append in that turn window rather than at `response.created`.
On a continuous stream with server VAD, the turn closes after `silence_duration_ms` of trailing silence (500 ms by
default), so that last append is typically the chunk in which server VAD closed the turn, and
`duplex_client_ttfp_ms` is about `response_created_to_first_audio_ms` plus a phase in `[0, chunk_ms)`. The phase
scales with the adapter's chunk size: native OmniInteract uses 200 ms, native Omni-DuplexEval uses ~1 s units, and
`server_vad` sends 32 ms detector frames. Proactive or silence-driven turns still use the last append in the window.
Created-origin audio stays on `response_created_to_first_audio_ms` (and
`audio_output.response_created_to_first_audio_ms`) and is not copied into `client_ttfp_ms`. Both client timers exclude
VAD silence detection and any delay before the last recorded append / `response.created` arrives, so they do not
measure the full user-speech-end to first-output latency. Per-response `client_rtf` uses the same append-send start
through the last audio packet, divided by emitted audio duration, when an append send was recorded. The server start
point differs between model plugins, so compare server keys only within one model; client append keys are only
comparable within one adapter / chunk size.
Qwen3-Omni duplex reports no per-token timing, so TPOT and ITL are unavailable. A turn-based model acknowledges a proactive request right away but does not speak again when the
event later happens, so proactive and 1QnA scenarios score low by construction.

To replay an existing sample set (for example `sampled_cases.jsonl` from a prior run), pass
`--omniinteract-video-list` instead of discovering cases. List order is preserved; `--num-prompts` takes the prefix of
the list (`0` or a value larger than the list length runs every row). Do not combine video-list mode with `--seed`,
`--dataset-path`, `--omniinteract-scenario-tags`, or `--omniinteract-scenario-focus`.

Use `--omniinteract-scenario-tags` to steer small runs toward paper-table scenarios (`realtime`, `proactive`,
`nested`, `interrupted`, `1qna`). By default the sampler covers each requested tag with at least one matching case
(when available under `--omniinteract-subsets`), then fills the remaining `--num-prompts` budget from other cases.
Add `--omniinteract-scenario-focus` to run only cases that match those tags. Example smoke coverage:

```bash
  --num-prompts 5 \
  --omniinteract-scenario-tags realtime proactive nested interrupted 1qna
```

Audio is replayed as 16 kHz PCM16 in 200 ms chunks and video at 1 FPS with real-time pacing. All selected media is decoded
before timing and remains in client memory for the run, so `--max-concurrency` does not limit media preparation memory; use
explicit `--num-prompts 0` only when the client has enough RAM for the full dataset. Media commands are bounded by
`--omniinteract-media-timeout-s`, and concurrency defaults to 1. Standard request-rate, warmup, result-saving, and summary
options apply. Use `--omniinteract-require-response` only for functional E2E cases; LISTEN is a valid benchmark result.

Each completed case writes `output.wav`, `wav_transcript.json`, `events.json`, `result.json`, and a final `.done` marker under
`--omniinteract-output-dir`. The root also contains `batch_summary.json`, `official_eval_manifest.jsonl` (eligible cases only),
and `sampled_cases.jsonl` (every sampled case as absolute `video_path` / official `output_name` / `subset` rows for Lucky-Lance
OmniInteract `batch_inference_minicpmo.py --video_list`); failed cases write
`.failed.json`. Runs sharing one output root are serialized. Completion validates transport, response lifecycle, and artifacts,
not answer accuracy. Transcript timestamps are serialized playback-queue times. Playback ACKs report cumulative progress
incrementally along that serialized clock, like a live listener; the first ack for a response goes out as soon as its audio
arrives, checkpointing the response's history position so a later committed user input updates it in place. A residual
`playback_ack_too_late` rejection is recorded as an artifact warning rather than failing the case. When a barge-in model
sends `output_audio_buffer.cleared`, the client flushes that response's unplayed audio from the serialized clock, as a live
speaker would. Clipped, cancelled, or cleared-before-played outputs are ineligible and omitted from the official manifest
(`audio_clipped`, `cancelled_response`, `cleared_response`), because their transcript holds more than the listener heard;
`audio_clipped_bytes` records output beyond the rounded video horizon.

Accuracy evaluation is opt-in and requires an already-running text judge with an OpenAI-compatible Chat Completions API. The
benchmark does not launch or stop the judge server. Add the following options to the command above:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --served-model-name Qwen2.5-7B-Instruct \
  --port 8001
```

Then add to the benchmark command:

```bash
  --omniinteract-evaluate \
  --omniinteract-judge-base-url http://127.0.0.1:8001 \
  --omniinteract-judge-model Qwen2.5-7B-Instruct
```

`--omniinteract-judge-model` must match that judge process's `--served-model-name` (or its `--model`
string if `--served-model-name` is omitted). It is not a checkpoint path to load, and it is not the
Omni DUT `--served-model-name`. Point `--omniinteract-judge-base-url` at the judge, not at the Omni
realtime endpoint.

The early / core / interrupted-partial judge prompts follow the official English
templates in [Lucky-Lance/OmniInteract](https://github.com/Lucky-Lance/OmniInteract)
`eval/evaluation/llm_judge.py` (commit `de304cef35fd9a50a5caadb5090c34cfbf0dd868`).
They correspond to the OmniInteract paper appendix
([arXiv:2605.26485](https://arxiv.org/abs/2605.26485)) Listing A.1 (early-stage),
Listing A.2 (interrupted partial quality), and Listing A.3 (core-stage).
A local OpenAI-compatible judge is not the paper's GPT-4o judge, so reported
IA-QTF1 numbers are protocol-compatible rather than official paper-table scores.

After artifact publication, the evaluator builds `[start, t_a, end)` slots, judges early and core response text, writes
per-case details plus `evaluation/unified_eval_summary.json`, and prints IA-QTF1 columns
(1Q1A realtime / proactive / nested / Global, 1QnA, All Global), interruption diagnostics (NOR / PAQ / CSM), and nested
metrics (NCCS, inner / outer IA-QTF1, missed outer). Slices and sections with no slots in the evaluated sample are
omitted from the terminal report. The header lists how many evaluated cases carry each scenario tag
(`realtime`, `proactive`, `nested`, `interrupted`, `1qna`). Realtime and proactive
exclude nested inner/outer slots; 1Q1A Global recomputes F1 from those three TP/FP/FN aggregates; All Global includes every
scored slot plus unmatched-chunk false positives. Plain `wav_transcript.json` timestamps provide chunk-level approximate
timing. If transcript chunks include `aligned_words`, the evaluator splits boundary-crossing chunks and derives trigger
timing from word alignment.

Per-response TTFT and TTFP start when the server begins executing the native model-turn request that owns the response. RTF
continues to use client receipt of `response.created` through the last audio packet, divided by emitted audio duration.
When the client recorded `input_audio_buffer.append` sends, `client_ttfp_ms` / `client_rtf` start at the last such send
before this `response.created` (after the previous `response.created` or stream start). `client_ttft_ms` still starts at
`response.created`. Created-origin audio is `response_created_to_first_audio_ms` and is not copied into `client_ttfp_ms`.
Global TTFT, TTFP, and RTF cover the complete input-stream window. TPOT/ITL use engine stage-0 timing; ITL is emitted only when every
token interval is present within a continuous generation segment. Model-unit pacing and gaps between generation segments are
excluded from TPOT/ITL. Raw request metrics retain `response_created_to_first_text_ms` and
`response_created_to_first_audio_ms` as client-envelope diagnostics.

The checked-in local performance configuration measures four deterministic cases from `1q1a`, `1q1a_math`, and
`1qna` (12 videos total), with no benchmark warmups and a maximum concurrency of two. Each subset also sends one readiness request before its
measured cases. The runner starts MiniCPM-o 4.5 on the first visible GPU and a text judge
(`Qwen/Qwen2.5-7B-Instruct`) on the second (`CUDA_VISIBLE_DEVICES=1` relative to the process). After artifacts land it
scores All Global IA-QTF1 through that local judge. This is protocol-compatible with the paper metric, not a GPT-4o
paper-table score. Run it from the repository root with two visible GPUs:

```bash
export HF_HOME=/path/to/persistent/huggingface-cache
export BENCHMARK_DIR=tests/dfx/perf/results
bash tools/nightly/run_nightly_jobs.sh \
  --test-type local \
  --model-type omni \
  --label-substr minicpmo_4_5_omniinteract
```

The first run downloads the pinned OmniInteract archive and the judge weights into `HF_HOME`; later runs reuse that cache.

It requires every case to commit its input, complete any emitted response lifecycles, and publish the expected WAV,
transcript, event, and result artifacts without errors. A valid LISTEN-only case may have no response audio or transcript
chunks. Official-manifest eligibility is reported separately because clipped or cancelled output is a benchmark-quality
signal, not a transport failure. Accuracy must finish with `status=ok` on every subset. After all three subsets finish,
All Global IA-QTF1 is recomputed from pooled `Global_TP` / `Global_FP` / `Global_FN` and must be at or above
`omniinteract_aggregate_min_ia_qtf1` (checked in as `0.2`).

`test_qwen3_omni_duplex_omniinteract.json` runs Qwen3-Omni on `vllm_omni/deploy/qwen3_omni_duplex.yaml` with server VAD
turns (see [Turn-based duplex models](#turn-based-duplex-models-qwen3-omni)). It is an L4 CUDA nightly performance job
on H100 and the B200 mirror. It needs three visible GPUs: the Qwen3-Omni stages take the first two and the text judge
the third (`cuda_visible_devices: "2"`). Run the same job locally with:

```bash
bash tools/nightly/run_nightly_jobs.sh \
  --test-type perf \
  --model-type omni \
  --label-substr "Duplex OmniInteract"
```

The job uploads benchmark JSONs and per-case audio, transcripts, events, and evaluation artifacts. It records performance
metrics without a hardware latency baseline; response validity and the math accuracy floor still gate the result.
It differs from the MiniCPM-o configuration in three ways:

- **Cold start.** The first request after the server starts compiles Triton kernels (rotary embedding, fused MoE) and
  answers seconds late. The first subset (`1q1a_math`) therefore replays its first case once, unmeasured
  (`num_warmups: 1`); later subsets reuse the warmed server and do not warm up again.
- **Interrupted responses are not scored.** The official eligibility rule is kept: a case with a cancelled, cleared, or
  clipped response is left out of accuracy, because its transcript holds text the listener never heard, and scoring it
  would make the result incomparable with MiniCPM-o and the paper. Qwen3-Omni stops speaking when the next question
  starts, so overlapping questions can make a case ineligible. A scored subset must keep at least one eligible case.
  This configuration measures latency only for `1q1a` and `1qna` (no `omniinteract_evaluate`); individual cases may
  still be eligible. Both subsets require at least one completed response with audio and transcript per case
  (`omniinteract_require_response: true`). Accuracy covers `1q1a_math` only, so it is not comparable with the
  three-subset MiniCPM-o score. Its
  IA-QTF1 must be at or above `omniinteract_aggregate_min_ia_qtf1` (checked in as `0.3`); a single run on four cases
  measured `0.50`, so the floor catches a lost correct answer or more, not small judge noise.
- **Latency.** Compare Qwen3-Omni runs on the per-response keys, primarily `mean_duplex_client_ttfp_ms`
  (last append send in the turn window to first audio; on a continuous VAD stream this is about
  `response_created_to_first_audio_ms` plus a `[0, chunk_ms)` phase from the chunk in which server VAD closed the
  turn, and includes prompt preparation after `response.created`). Compare those client append keys only within one
  adapter / chunk size. The standard TTFT rows average sessions first, and TPOT is unavailable.

### Seed-TTS reference speech as real audio input

By default the Seed-TTS dataset sends its reference clip as voice-cloning metadata: `ref_audio` / `ref_text` ride the
request body and the model is asked to synthesize the target text in that voice.

`--seed-tts-reference-as-input` changes the request shape instead of the request contents. The reference clip becomes an
ordinary **user audio turn**, `ref_audio` / `ref_text` are dropped entirely, and the system prompt asks the model to read
the accompanying text rather than transcribe or answer the audio. This is what makes the workload exercise a real duplex
conversation turn rather than a TTS call. It requires `--backend openai-realtime-chat`.

For Qwen3-Omni, use a server running the
[`qwen3_omni_duplex.yaml` deployment](gh-file:vllm_omni/deploy/qwen3_omni_duplex.yaml) with async chunking enabled.
Each dataset entry uses an independent WebSocket session at `/v1/realtime?duplex=1` with one audio turn and one response.

```bash
vllm bench serve --omni \
  --backend openai-realtime-chat \
  --endpoint /v1/realtime \
  --dataset-name seed-tts \
  --dataset-path zhaochenyang20/seed-tts-eval \
  --seed-tts-reference-as-input \
  --seed-tts-locale en \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --base-url http://127.0.0.1:8000 \
  --num-prompts 4 --max-concurrency 1 --output-len 256 \
  --percentile-metrics ttft,e2el,audio_ttfp,audio_rtf,audio_duration \
  --extra-body '{"realtime_trigger": "vad"}'
```

Each clip is normalized once to mono 24 kHz PCM16 and a one-second silent tail is appended. Both cases send the same
speech samples in chunks of at most 200 ms. A chunk is sent at the end of its simulated capture interval, using absolute
deadlines to avoid accumulated pacing drift. A partial chunk is split at the speech/silence boundary so the speech-end
timestamp stays exact even when the clip length is not a multiple of 200 ms. `realtime_trigger` selects how the turn ends:

- `explicit` — the client streams the speech at real-time speed and then sends `input_audio_buffer.commit` plus
  `response.create`. The silent tail is **not** sent: an explicit client ends the turn when the speaker stops, so
  streaming silence first would charge it for latency no real caller pays.
- `vad` — the client streams the speech *and* the tail, and server VAD ends the turn on its own, without a client commit
  or `response.create`. Its 800 ms silence threshold must stay shorter than the one-second tail or the turn never ends.

#### Run the matched nightly workload

The [performance configuration](gh-file:tests/dfx/perf/tests/test_qwen3_omni_seed_tts.json) runs both triggers using the
first 32 English Seed-TTS entries in fixed order, concurrency 1, temperature 0, and a 256-token output limit. It asks
the model to read the target text rather than answer or transcribe the accompanying audio. Each trigger runs two
unmeasured warmup requests before its 32 measured requests. The CUDA nightly job runs this configuration on two H100/B200 GPUs.

Run from the repository environment after reserving two GPUs according to your host's scheduling rules:

```bash
uv run --no-sync pytest -s -v tests/dfx/perf/scripts/run_benchmark.py \
  --test-config-file tests/dfx/perf/tests/test_qwen3_omni_seed_tts.json \
  --run-level full_model
```

`BENCHMARK_DIR` selects the result directory (default `tests/dfx/perf/results`). For cached local weights or data, copy the
JSON and replace `server_params.model` and each `benchmark_params[].dataset_path` with local paths. Preserve sample
count, order, system prompt, and generation settings across the two cases.

#### Read the reported metrics

TTFT, audio TTFP, E2EL, and audio RTF share one client-side origin: the end of reference-speech capture, immediately before
the last speech chunk is sent. They exclude session setup and the capture/upload intervals preceding that boundary.
Timing from session start would fold the client's real-time input upload into the reported wait.

- **TTFT / audio TTFP:** origin to the first non-empty text delta / first audio packet received. They include transport
  and response processing after the speech-end boundary and, for VAD, endpoint detection during the silent tail.
- **E2EL:** origin to `response.done` reception.
- **Audio RTF:** origin to the last audio packet divided by generated audio duration. It includes post-speech waiting
  and endpoint detection; it is not model-only compute RTF.
- **Audio duration:** generated audio length, received as PCM16.

The per-request rows in `duplex_request_metrics` retain `measurement_origin`, utterance identity, trigger, and these
diagnostic fields:

- `input_audio_ms`: prepared clip duration, including the silent tail.
- `input_content_ms`: reference-speech duration, excluding the tail.
- `input_uploaded_ms`: audio duration actually sent; only VAD includes the tail.
- `input_upload_ms` / `session_setup_ms`: client wall-clock upload / setup time.
- `session_start_to_first_audio_ms` / `session_start_to_response_done_ms`: timings from before text submission and audio
  upload, retained for diagnosis.
- `explicit_commit_to_first_audio_ms` (explicit only): sending the explicit trigger to receiving the first audio packet.
- `vad_stop_received_ms` (VAD only): speech-end origin to the client's receipt of `speech_stopped`.
- `vad_stop_to_first_audio_ms` (VAD only): that event's receipt to the first audio packet. Both VAD fields use client
  event timestamps, not server GPU timestamps.

The MiniCPM-o Seed-TTS Realtime backend prefers server-provided request timings from a different origin. Its
TTFT/TTFP/RTF values are not directly comparable with this Qwen3-Omni workload. This Qwen3 path provides no Stage-0 engine
token timing, so TPOT/ITL are unavailable. The backend explicitly disables derived TPOT even if a tokenizer counts
transcript tokens; `num_tpot_samples` remains zero and a configured TPOT baseline rejects the missing measurement.

#### Scope and limits

The test requires every request to complete with exactly one response containing text and audio. VAD cases must emit
speech-start and speech-stop events; a response before the reference speech ends is rejected. Although the server-VAD
contract requires interruption flags, this workload sends no overlapping speech. It does not measure interruptions,
multi-turn context, or concurrency scaling.

The job currently collects latency measurements and checks request completion; no performance regression baseline is set.
Even with 32 requests, p99 is a diagnostic rather than a stable regression threshold. After collecting repeated successful
H100 nightly runs with this configuration, a follow-up can set a separate baseline for each trigger. B200 requires its own
measurements. Results from before the capture-pacing fix could understate VAD latency by up to one 200 ms chunk and must
not be used as the corrected workload's performance reference.

### Video-MME Benchmark

Video-MME (`--dataset-name videomme`) scores multiple-choice video QA. Default packing is
OmniEvalKit MiniCPM `minicpm-frames` (up to 96 sampled frames as `image_url`). Pass a local
mirror with `--dataset-path` / `--videomme-parquet` + `--videomme-video-dir`, or a Hugging
Face dataset id (`lmms-eval/Video-MME` by default; any `org/name` is accepted when
`--dataset-name videomme` is explicit). Relative `--videomme-video-dir` values are resolved
to absolute `file://` URLs. For those URLs, start the server with
`--allowed-local-media-path` covering the video root; otherwise use `--videomme-inline-local-video`.

For MiniCPM-o 4.5, the default deployment allows 64 images per request. To run the
96-frame recipe, start the server with a stage 0 override (replace `/path/to/Video-MME`
with the local video root shared by the benchmark client and server):

```bash
vllm serve openbmb/MiniCPM-o-4_5 --omni \
  --allowed-local-media-path /path/to/Video-MME \
  --stage-overrides '{"0":{"limit_mm_per_prompt":{"image":96,"audio":64,"video":1}}}'
```

The override applies to this server invocation only. Alternatively, use
`--videomme-max-frames 64` in the benchmark to stay within the default image limit;
this evaluates a different frame-sampling setting from the 96-frame recipe.

Run the benchmark against that server:

```bash
vllm bench serve --omni \
  --backend openai-chat-omni \
  --dataset-name videomme \
  --dataset-path /path/to/Video-MME \
  --videomme-pack-mode minicpm-frames \
  --videomme-max-frames 96 \
  --model openbmb/MiniCPM-o-4_5 \
  --endpoint /v1/chat/completions \
  --num-prompts 8 \
  --max-concurrency 1 \
  --save-result
```

Accuracy keys (`videomme_accuracy`, per-duration / domain / task breakdowns) are written into
the saved JSON. `videomme_accuracy` excludes HTTP failures; `videomme_accuracy_incl_http_fail`
counts them as wrong. `videomme_submitted` / `videomme_unique_question_ids` show when
oversampling made the request count differ from unique-question coverage.
`videomme_skipped_rows` counts rows skipped during sampling because of missing or unreadable
media or invalid question fields. The opt-in accuracy runner rejects skipped rows, HTTP
failures, duplicate question IDs, and missing gold answers. This checks the sampled subset;
it does not establish that a supplied local mirror contains the entire official dataset.
Use `--videomme-save-eval-items` (or `VIDEOMME_SAVE_EVAL_ITEMS=1`) for per-request
`videomme_eval_items` rows.

### Multi-Modal Benchmark

<details class="admonition abstract" markdown="1">
<summary>Show more</summary>

Benchmark the performance of multi-modal requests in vLLM-Omni.

Generate synthetic image、video、audio inputs alongside random text prompts to stress-test vision models without external datasets.

Notes:

- Works only with online benchmark via:
    - the OpenAI chat backend (`--backend openai-chat-omni`) and endpoint `/v1/chat/completions`.
    - the OpenAI edit image backend (`--backend openai-image-edits-omni`) and endpoint `/v1/images/edits`.

Start the server (example):

```bash
vllm serve Qwen/Qwen2.5-Omni-7B --omni
```

It is recommended to use the flag `--ignore-eos` to simulate real responses. You can set the size of the output via the arg `random-output-len`.

Then run the benchmarking script:

```bash
vllm bench serve \
    --omni \
  --dataset-name random-mm \
  --port 40849 \
  --model /home/models/Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --endpoint /v1/chat/completions \
  --backend openai-chat-omni \
  --request-rate 1 \
  --num-prompts 1 \
  --random-input-len 10 \
  --random-range-ratio 0.0 \
  --random-mm-base-items-per-request 2 \
  --random-mm-num-mm-items-range-ratio 0 \
  --random-mm-limit-mm-per-prompt '{"image":1,"video":1, "audio": 1}' \
  --random-mm-bucket-config '{"(32, 32, 1)": 0.5, "(0, 1, 1)": 0.1, "(32, 32, 2)":0.4}' \
  --ignore-eos \
  --percentile-metrics ttft,tpot,itl \
  --random-output-len 2 \
  --extra-body '{"modalities": ["text"]}'
```

If successful, you will see the following output:

```text
============ Serving Benchmark Result ============
Successful requests:                     1
Failed requests:                         0
Request rate configured (RPS):           1.00
Benchmark duration (s):                  1.21
Request throughput (req/s):              0.83
Peak concurrent requests:                1.00
================== Text Result ===================
Total input tokens:                      10
Total generated tokens:                  3
Output token throughput (tok/s):         2.49
Peak output token throughput (tok/s):    3.00
Peak concurrent requests:                1.00
Total Token throughput (tok/s):          10.77
---------------Time to First Token----------------
Mean TTFT (ms):                          179.74
Median TTFT (ms):                        179.74
P99 TTFT (ms):                           179.74
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          12.76
Median TPOT (ms):                        12.76
P99 TPOT (ms):                           12.76
---------------Inter-token Latency----------------
Mean ITL (ms):                           12.76
Median ITL (ms):                         12.76
P99 ITL (ms):                            25.24
================== Audio Result ==================
Total audio duration generated(s):       0.00
Total audio frames generated:            0
Audio throughput(audio duration/s):      0.00
==================================================
```

Behavioral notes:

- If the requested base item count cannot be satisfied under the provided per-prompt limits, the tool raises an error rather than silently clamping.

How sampling works:

- Determine per-request item count k by sampling uniformly from the integer range defined by `--random-mm-base-items-per-request` and `--random-mm-num-mm-items-range-ratio`, then clamp k to at most the sum of per-modality limits.
- For each of the k items, sample a bucket (H, W, T) according to the normalized probabilities in `--random-mm-bucket-config`, while tracking how many items of each modality have been added.
- If a modality (e.g., image) reaches its limit from `--random-mm-limit-mm-per-prompt`, all buckets of that modality are excluded and the remaining bucket probabilities are renormalized before continuing.
This should be seen as an edge case, and if this behavior can be avoided by setting `--random-mm-limit-mm-per-prompt` to a large number. Note that this might result in errors due to engine config `--limit-mm-per-prompt`.
- The resulting request contains synthetic image data in `multi_modal_data` (OpenAI Chat format). When `random-mm` is used with the OpenAI Chat backend, prompts remain text and MM content is attached via `multi_modal_data`.

</details>

### Image / video generation

For `/v1/images/generations`, `/v1/images/edits`, and `/v1/videos`, set `--endpoint` to that path and omit `--backend`. The endpoint path is registered as the bench request adapter, so a separate named backend is not required.

Example (`/v1/images/generations`):

```bash
vllm bench serve --omni \
  --endpoint /v1/images/generations \
  --dataset-name random \
  --model ~/models/Qwen/Qwen-Image \
  --tokenizer ~/models/Qwen/Qwen-Image/tokenizer \
  --max-concurrency 1 \
  --num-warmups 2 \
  --num-prompts 10 \
  --random-input-len 64 \
  --random-output-len 1 \
  --ignore-eos \
  --percentile-metrics e2el \
  --extra-body '{
    "num_inference_steps": 20,
    "seed": 42,
    "true_cfg_scale": 4.0
  }'
```

If successful, following output is like:

```text
============ Serving Benchmark Result ============
Successful requests:                     3  
Failed requests:                         0  
Maximum request concurrency:             1  
Benchmark duration (s):                  8.42  
Request throughput (req/s):              0.36  
Peak concurrent requests:                2.00  
-------------------Peak Memory--------------------
Mean PEAK_MEMORY_MB (MB):                58832.00  
Median PEAK_MEMORY_MB (MB):              58832.00  
P99 PEAK_MEMORY_MB (MB):                 58832.00  
----------------End-to-end Latency----------------
Mean E2EL (ms):                          2798.16  
Median E2EL (ms):                        2799.83  
P99 E2EL (ms):                           2800.03  
================== Image Result ==================
Total images generated:                  3  
Image throughput (img/s):                0.36  
Average pixels per image:                1048576.00
Mean denoise step latency (ms):          136.59  
---------------- Image Generation ----------------
Mean IMAGE_GENERATION (ms):              2731.84  
Median IMAGE_GENERATION (ms):            2732.02  
P99 IMAGE_GENERATION (ms):               2734.65  
==================================================
```

Use the same pattern with `--endpoint /v1/images/edits` or `--endpoint /v1/videos` (and model-specific `--extra-body` as needed). Pure image/video runs omit the Text Result section when there is no generated text.

`/v1/images/edits` defaults to **non-streaming JSON** (`stream=false`) so single-stage edit models are not rejected by the server. For multi-stage pipelines that need SSE (AR TTFT / image chunks), pass `"stream": true` in `--extra-body`.

`/v1/videos` is an async job API: the client creates a job, then polls until `completed`/`failed`. Measured **e2el** therefore includes client poll sleep and any overshoot after the job actually finishes. Tune polling via `--extra-body`:

- `poll_interval_s` (default `2.0`): sleep between status polls
- `poll_timeout_s` (default `21600`, i.e. 6 hours): give up waiting for the job

**VIDEO_RTF** prefers server-reported generation time when available:

- `video_rtf = video_generation_time_ms / 1000 / video_duration`, where `video_generation_time_ms` comes from response `stage_durations` (or `inference_time_s` as fallback)
- If generation time is missing, falls back to `e2el / video_duration` (this fallback *does* include poll overhead)

### Multi-Stage Benchmark

<details class="admonition abstract" markdown="1">
<summary>Show more</summary>

Start the server (change the port and model):

```bash
vllm serve --omni --port 29999 --model ~/Qwen3-Omni-30B-A3B-Instruct
```

Then run the benchmarking script (remember to add the flag `--print-stage`):

```bash
vllm bench serve --omni \
  --dataset-name random \
  --port 29999 \
  --max-concurrency 1 \
  --num-warmups 2 \
  --model ~/Qwen3-Omni-30B-A3B-Instruct \
  --endpoint /v1/chat/completions \
  --backend openai-chat-omni \
  --num-prompts 3 \
  --random-input-len 2500 \
  --ignore-eos \
  --percentile-metrics ttft,tpot,itl,e2el,audio_ttfp,audio_rtf,ttfc,tpoc,icl \
  --random-output-len 900 \
  --extra-body '{"modalities": ["text","audio"]}' \
  --print-stage
```

Besides the "Serving Benchmark Result", there will be "Stage Benchmark Result" below:

```text
============= Stage Benchmark Result =============
=============== Stage 0 (thinker) ================
-------------------Stage Timing-------------------
Mean stage_gen_time (ms):                18243.16
Median stage_gen_time (ms):              18220.83
P99 stage_gen_time (ms):                 18296.75
================== Text Result ===================
Stage generated tokens:                  2700
-----------Serving Time to First Token------------
Mean Serving TTFT (ms):                  816.31
Median Serving TTFT (ms):                818.94
P99 Serving TTFT (ms):                   829.76
-----Time per Output Token (excl. 1st token)------
Mean TPOP (ms):                          19.41
Median TPOP (ms):                        19.40
P99 TPOP (ms):                           19.47
---------------Inter-token Latency----------------
Mean ITL (ms):                           19.41
Median ITL (ms):                         19.12
P99 ITL (ms):                            23.58
================ Stage 1 (talker) ================
-------------------Stage Timing-------------------
Mean stage_gen_time (ms):                18849.64
Median stage_gen_time (ms):              13641.15
P99 stage_gen_time (ms):                 33655.58
============= Internal Stream Result =============
-----------Serving Time to First Chunk------------
Mean Serving TTFC (ms):                  1029.14
Median Serving TTFC (ms):                1027.98
P99 Serving TTFC (ms):                   1032.59
-----Time per Output Chunk (excl. 1st chunk)------
Mean TPOP (ms):                          17.53
Median TPOP (ms):                        18.70
P99 TPOP (ms):                           18.88
---------------Inter-chunk Latency----------------
Mean ICL (ms):                           16.27
Median ICL (ms):                         18.19
P99 ICL (ms):                            25.86
=============== Stage 2 (code2wav) ===============
-------------------Stage Timing-------------------
Mean stage_gen_time (ms):                18860.74
Median stage_gen_time (ms):              13655.84
P99 stage_gen_time (ms):                 33659.79
================== Audio Result ==================
Stage audio duration generated(s):       260.11
Stage audio frames generated:            6242730
-----------Serving Time to First Packet-----------
Mean Serving AUDIO_TTFP (ms):            1366.16
Median Serving AUDIO_TTFP (ms):          1367.20
P99 Serving AUDIO_TTFP (ms):             1368.26
-----------------Real Time Factor-----------------
Mean AUDIO_RTF:                          0.24
Median AUDIO_RTF:                        0.26
P99 AUDIO_RTF:                           0.27
```

Explanation:

- stage_gen_time: Time from submitting a request to a specific stage to that stage finishing generation.

- Serving TTFC (Time to First Chunk): Time from the HTTP request being accepted by the serving frontend to
  the stage producing its first non-empty output chunk. Keep the name TTFT in text stage, TTFP in audio stage
  for easier mapping.

- TPOP (Time per Output Chunk): Average time from the first output chunk to stage completion, divided by the
  number of remaining output chunks. The TPOP abbreviation follows Qwen3.5-Omni Technical Report.

- ICL (Inter-Chunk Latency): Time between two consecutive output chunks produced by the same stage. Keep the
  name ITL in text stage for easier mapping.

</details>
