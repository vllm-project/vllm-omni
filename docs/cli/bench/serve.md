# vLLM-Omni Benchmark CLI Guide
The vllm bench command launches the vLLM-Omni benchmark to evaluate the performance of multimodal models.

## Notes
We currently only support using the "openai-chat-omni" backend.

## Basic Parameter Description
You can use `vllm bench serve --omni --help=all` to get descriptions of all parameters. The commonly used parameters are described below:
- `--omni`  
  Enable Omni (multimodal) mode, supporting multimodal inputs and outputs such as images, videos, and audio.

- `--backend`  
  Specify the backend adapter as openai-chat-omni, using OpenAI Chat compatible API behavior as the protocol. Currently only openai-chat-omni is supported.

- `--model`  
  The model identifier to load, filled according to the models supported by vLLM-Omni.

- `--endpoint`  
  The API endpoint exposed externally, to which clients send their requests.

- `--dataset-name`  
  The name of the dataset used; random-mm indicates generating random multimodal inputs (images, videos, audio).

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
                    'Allowed metric names are "ttft", "tpot", "itl", "e2el", "audio_ttfp", "audio_rtf", "audio_duration". '

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
  context length sampled from [input_len * (1 - range_ratio),
  input_len * (1 + range_ratio)].Only the random and random-mm modes
  support this parameter.

- `--random-input-len`  
  Number of input tokens per request.Only the random and random-mm modes support this parameter.

- `--random-output-len`  
  Number of output tokens per request.Only the random and random-mm modes support this parameter.

- `--random-range-ratio`  
  Range ratio for sampling input/output length,
  used only for random sampling. Must be in the range [0, 1) to define
  a symmetric sampling range
  [length * (1 - range_ratio), length * (1 + range_ratio)].
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

### Multi-Modal Benchmark

<details class="admonition abstract" markdown="1">
<summary>Show more</summary>

Benchmark the performance of multi-modal requests in vLLM-Omni.

Generate synthetic image、video、audio inputs alongside random text prompts to stress-test vision models without external datasets.

Notes:

- Works only with online benchmark via the OpenAI backend (`--backend openai-chat-omni`) and endpoint `/v1/chat/completions`.

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
  --extra_body '{"modalities": ["text"]}'
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
