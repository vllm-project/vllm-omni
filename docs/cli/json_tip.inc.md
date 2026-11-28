Pass JSON-valued options as one shell argument. In a POSIX shell, surround the
JSON with single quotes and use double quotes for JSON keys and strings:

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni \
    --stage-overrides '{"0": {"gpu_memory_utilization": 0.8, "enforce_eager": true}}'
```

`--stage-overrides` expects an object mapping stage IDs to objects of engine
argument overrides. JSON booleans are lowercase `true` and `false`; Python
dictionary syntax such as `{'0': {'enforce_eager': True}}` is not valid JSON.
Shell quoting differs on Windows. Each JSON option has its own schema; the
stage-ID mapping above is specific to `--stage-overrides`.
