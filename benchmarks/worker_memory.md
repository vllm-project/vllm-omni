# Worker allocator diagnostics

[WorkerMemoryExtension](worker_memory.py) reads counters in an actual AR
worker through the existing collective RPC interface. Load it on a separate
diagnostic service, with the model's deployment config:

~~~bash
PYTHONPATH="$PWD" VLLM_SERVER_DEV_MODE=1 \
  vllm serve /path/to/model --omni --host 127.0.0.1 --port 18082 \
  --worker-extension-cls benchmarks.worker_memory.WorkerMemoryExtension
~~~

Keep the privileged development RPC surface on localhost. After readiness,
prewarm the intended shapes and drain requests. Reset before a diagnostic
batch, run the requests, drain again, then read:

~~~bash
curl --fail-with-body http://127.0.0.1:18082/collective_rpc \
  -H 'Content-Type: application/json' \
  -d '{"method":"benchmark_memory_reset","args":["prewarmed-batch"],"timeout":30}'
# Run and finish the diagnostic batch.
curl --fail-with-body http://127.0.0.1:18082/collective_rpc \
  -H 'Content-Type: application/json' \
  -d '{"method":"benchmark_memory_read","timeout":30}'
~~~

Both calls synchronize the current accelerator; use separate latency runs.
Responses retain worker PID, scope, reset/observation timestamps and current /
peak allocated/reserved bytes. Check expected workers and wrapped RPC errors;
HTTP 200 alone does not establish a valid measurement. Reads before reset and
missing backend counters fail. Diffusion workers are not implicitly supported.

Peaks cover the explicit reset interval, not full startup. Reserved includes
allocated. Device residency and weights/KV/buffer attribution need separate
measurements. Preserve repeated reads: a decreasing peak invalidates the
interval, even if PID/reset IDs match. This occurred across sleep/wake; start
a new serving interval after wake and keep the lost cross-cycle peak missing.

CPU checks use the matching Torch and repository test dependencies:

~~~bash
python -m pytest tests/benchmarks/test_worker_memory.py \
  -m 'core_model and cpu' --run-level=core_model
~~~

The ready-pipeline CPU route collects these tests. Serving integration evidence
and scoped results are in the [VoxCPM2 study](../recipes/OpenBMB/VoxCPM2-RTX5060Ti-16GB.md).
