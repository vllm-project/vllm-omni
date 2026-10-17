# Measurement Protocol

Full bench flags: `docs/cli/bench/serve.md` (`vllm bench serve --omni --help=all`).
Metric names: `vllm_omni/metrics/definitions.py`; aggregation:
`vllm_omni/benchmarks/metrics/metrics.py`.

## Contents
- Command templates
- Metrics that matter
- Low-overhead attribution
- Diagnostic tools
- A/B rules
- CI performance rows

## Command templates

```bash
# serve.sh
MODEL=${MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}
DEPLOY=${DEPLOY:-vllm_omni/deploy/qwen3_omni_moe.yaml}
PORT=${PORT:-8091}
vllm serve "$MODEL" --omni --port "$PORT" \
  --deploy-config "$DEPLOY" \
  ${STAGE_OVERRIDES:+--stage-overrides "$STAGE_OVERRIDES"} \
  ${EXTRA_SERVE_ARGS}
```

```bash
# request.sh: one fixed request for correctness and per-request stage timing
curl -sS http://127.0.0.1:${PORT:-8091}/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "'"$MODEL"'", "modalities": ["text", "audio"],
       "messages": [{"role": "user", "content": "<fixed prompt>"}],
       "seed": 0}' -o out/request.json
```

```bash
# bench.sh
vllm bench serve --omni --port ${PORT:-8091} --model "$MODEL" \
  --endpoint /v1/chat/completions --backend openai-chat-omni \
  --dataset-name random-mm --num-prompts ${N:-200} \
  --max-concurrency ${C:-16} \
  --extra-body '{"modalities": ["text", "audio"]}' \
  --percentile-metrics ttft,tpot,itl,e2el,audio_ttfp,audio_rtf,audio_duration \
  --metric-percentiles 50,90,99 --print-stage \
  --save-result --result-dir runs/${RUN_ID:?}
```

Use `--ignore-eos` with length-controlled random datasets. Use a turn-mode
deploy for HTTP chat benchmarks (MiniCPM-o's default YAML is duplex). Duplex
benchmarks use `--backend openai-realtime-duplex --endpoint /v1/realtime
--dataset-name omniinteract`.

## Metrics that matter

| Metric | Meaning | Watch for |
|--------|---------|-----------|
| TTFT | Client time to first text token | Thinker prefill, encoders, queueing |
| AUDIO_TTFP | Time to first non-empty audio packet | Primary latency metric for spoken output |
| AUDIO_RTF | Generation latency / audio duration | Must stay below 1 for real-time streaming |
| E2EL | End-to-end request latency | Can rise with `async_chunk` even when TTFP falls |
| audio_throughput | Audio seconds per wall second | Capacity metric |
| `audio_continuity_ok_rate` | Share of requests without audible playback gaps | Required for any streaming-quality claim |
| TTFC / TPOC / ICL / TPOP | Stage-level chunk metrics | Attribute streaming latency per stage |

## Low-overhead attribution

- `--log-stats` on the server: per-request stage wall time,
  `stage_gen_time_ms`, connector `tx_time_ms`, `rx_decode_time_ms`,
  `in_flight_time_ms`.
- `--print-stage` on the bench client: per-stage metrics from server
  snapshots.
- `--enable-ar-profiler`: AR stage generation time in per-request
  `stage_durations`.
- Prometheus `vllm_omni:*` families on `/metrics` for continuous runs.

These are cheap enough for baseline runs; still compare a run with and
without them once to confirm.

## Diagnostic tools

Use only after attribution narrows the target, and never for final numbers.

```bash
PROFILE='{"profiler":"torch","torch_profiler_dir":"/tmp/omni_profile",'\
'"torch_profiler_with_stack":false,"torch_profiler_record_shapes":true,'\
'"torch_profiler_use_gzip":true}'
STAGE_OVERRIDES="{\"1\":{\"profiler_config\":$PROFILE}}" ./serve.sh
# warm up with request.sh, then:
curl -sS -X POST http://127.0.0.1:8091/start_profile -H 'Content-Type: application/json' -d '{"stages":[1]}'
./request.sh
curl -sS -X POST http://127.0.0.1:8091/stop_profile -H 'Content-Type: application/json' -d '{"stages":[1]}'
python3 .claude/skills/diffusion-perf-opt/scripts/trace_analyzer.py /tmp/omni_profile/<trace>.json.gz --min-gap-ms 1 --topn 20
```

- Run one shape trace (`record_shapes` on, stack off) and one host-stack
  trace (stack on, shapes off) separately.
- Nsight Systems: `{"profiler":"cuda"}` in the stage override and wrap the
  server with `nsys profile --capture-range=cudaProfilerApi
  --cuda-graph-trace=node ...` (see `docs/contributing/profiling.md`).
- Orchestrator: `--enable-orch-monitor`, output at shutdown or
  `VLLM_OMNI_ORCH_MONITOR_PATH`. Read `loop_active_pct`,
  `dispatch_queue_high_water`, and per-replica `outputs_queue_size`.

## A/B rules

1. Record both commits, hardware, deploy YAML plus overrides, and exact
   commands for every column.
2. Change one variable per column. For runtime switches, run base, head
   with the switch off, and head with the switch on.
3. Warm up; exclude cold-start samples from every reported number.
4. Use hundreds of requests for P99; repeat noisy runs and report spread.
5. Hold sanity rows constant across columns: generated text tokens, audio
   duration, failure count.
6. End with an isolation statement naming the metric that stayed unchanged.
7. When graphs, caches, or recompilation can occur at runtime, also report
   the event count and the largest observed stall.

## CI performance rows

Nightly omni perf cases live in `tests/dfx/perf/tests/` (for example
`test_qwen3_omni_async_chunk.json`, `test_minicpmo_4_5.json`). Each case has
`mark` (hardware, `full_model`, `omni`), `server_params`, and
`benchmark_params` with per-hardware `baseline` thresholds. Update the
baseline only with A/B evidence on the named hardware; wiring and markers
belong to the `vllm-omni-test` skill.
