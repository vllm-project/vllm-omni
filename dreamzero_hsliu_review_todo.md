# DreamZero PR #2162 hsliuustc0106 Review TODO

Scope: review comments by `hsliuustc0106` starting from:
`assert is stripped under python -O. This check (and the 14 others in this file)...`

Current status: H01, H02, and H04 implemented; remaining items pending discussion and confirmation.

## H01. Runtime `assert` in `causal_wan_model.py`

- Review: `assert` is stripped under `python -O`; `assert kv_cache is not None` and the other runtime checks in this file should be explicit `if ...: raise ...`.
- Link: https://github.com/vllm-project/vllm-omni/pull/2162#discussion_r3294053083
- File: `vllm_omni/diffusion/models/dreamzero/causal_wan_model.py`
- Current code status: implemented. Runtime asserts in `causal_wan_model.py` were replaced with explicit exceptions.
- Risk: production runs with optimization flags can silently remove these checks, turning intended validation failures into later `NoneType`, shape, or tensor operation errors.
- Proposed fix:
  - Convert runtime validation asserts to explicit exceptions.
  - Use `RuntimeError` for violated inference state invariants, such as missing `kv_cache`.
  - Use `ValueError` for invalid config, unsupported model type, invalid tensor shapes, and invalid caller inputs.
  - Keep test-file asserts unchanged.
- Test plan:
  - Done: `python -O -m py_compile vllm_omni/diffusion/models/dreamzero/causal_wan_model.py`.
  - Done: `rg` confirms no remaining `assert` or `if True` in `vllm_omni/diffusion/models/dreamzero`.
- Confirmation: confirmed and implemented.

## H02. Runtime `assert` in `state_dreamzero.py`

- Review: same issue for three `assert cache is not None` guards in `state_dreamzero.py`.
- Link: https://github.com/vllm-project/vllm-omni/pull/2162#discussion_r3294053104
- File: `vllm_omni/diffusion/models/dreamzero/state_dreamzero.py`
- Current code status: implemented. The three cache initialization asserts were replaced with `RuntimeError`.
- Risk: under `python -O`, uninitialized cache access may propagate `None` and fail later with less actionable errors.
- Proposed fix:
  - Replace the three asserts with explicit `if cache is None: raise RuntimeError(...)`.
  - Keep messages specific to KV cache vs cross-attention cache.
- Test plan:
  - Done: added `DreamZeroState` tests that call cache methods before `create_kv_caches()` and verify `RuntimeError`.
  - Done: `python -O -m py_compile vllm_omni/diffusion/models/dreamzero/state_dreamzero.py`.
- Confirmation: confirmed and implemented.

## H03. Duplicate embodiment ID for `mecka_hands` and `lapa`

- Review: `mecka_hands` and `lapa` both map to embodiment ID `27`; if they should have distinct action-head weights, one prediction path is wrong.
- Link: https://github.com/vllm-project/vllm-omni/pull/2162#discussion_r3294053140
- File: `vllm_omni/diffusion/models/dreamzero/utils.py`
- Current code status: both names map to `27`.
- Risk: if upstream expects different IDs, action conditioning selects the wrong embodiment embedding/domain for one of the two names.
- Proposed fix options:
  - Option A: verify upstream DreamZero config/checkpoint mapping and update one ID if it is a typo.
  - Option B: if the duplicate is intentional aliasing, keep the mapping and add a short comment explaining that `lapa` is an alias of the same embodiment ID.
  - Option C: if unsupported/unknown, remove one alias and require explicit `embodiment_id`.
- Test plan:
  - Update `tests/dreamzero/test_utils.py` to assert the intended mapping or alias behavior.
- Confirmation: pending.

## H04. No-op `if True:` wrapper in `causal_wan_model.py`

- Review: `if True:` is a no-op wrapper and should be removed with the body dedented.
- Link: https://github.com/vllm-project/vllm-omni/pull/2162#discussion_r3294060321
- File: `vllm_omni/diffusion/models/dreamzero/causal_wan_model.py`
- Current code status: implemented. The no-op `if True:` wrapper was removed and the body was dedented.
- Risk: no runtime behavior issue, but it is dead/refactor artifact code and reduces readability.
- Proposed fix:
  - Remove `if True:`.
  - Dedent the body.
  - Combine with H01 because both touch the same block.
- Test plan:
  - Done: `python -O -m py_compile vllm_omni/diffusion/models/dreamzero/causal_wan_model.py`.
  - Done: `tests/dreamzero/test_pipeline_state.py tests/dreamzero/test_utils.py`.
- Confirmation: confirmed and implemented.

## H05. OpenPI serving reuses `session_id` as engine `request_id`

- Review: the server uses long-lived `session_id` as per-inference engine `request_id`, causing duplicate active request IDs for concurrent clients or repeated calls.
- Link: https://github.com/vllm-project/vllm-omni/pull/2162#discussion_r3294078365
- File: `vllm_omni/entrypoints/openai/realtime/robot/openpi_serving.py`
- Current code status: `_build_request()` still sets `request_ids=[f"robot-{session_id}"]`.
- Risk:
  - Two websocket clients without explicit `session_id` both use `robot-default`.
  - Two clients sharing a logical session reuse the same engine request ID.
  - One client can reuse the same active ID across sequential calls if a previous generation has not fully drained.
  - Diffusion scheduler and `AsyncOmni.request_states` expect request IDs to be unique per inference.
- Proposed fix:
  - Keep `session_id` only in `sampling_params.extra_args` for DreamZero state lookup.
  - Generate a unique engine request ID per inference, for example `robot-{session_id}-{counter}` or `robot-{session_id}-{uuid}`.
  - Prefer a per-serving-instance monotonic counter or UUID to avoid cross-connection collisions.
  - Ensure logs still include session information for debugging.
- Test plan:
  - Unit test `_build_request()` twice with the same `session_id` and verify `request_ids[0]` differs while `extra_args["session_id"]` stays the same.
  - Add a concurrency-oriented serving test if feasible, or at least a regression test for duplicate ID avoidance.
- Confirmation: pending.

## H06. OpenPI clients do not surface msgpack structured errors

- Review: server errors are msgpack dicts like `{"type": "error", "message": ...}`, but clients only treat text frames as errors and then try to convert decoded dicts to action arrays.
- Link: https://github.com/vllm-project/vllm-omni/pull/2162#discussion_r3294078370
- Files:
  - `examples/online_serving/dreamzero/openpi_client.py`
  - `tests/dreamzero/openpi_client_helper.py`
  - `examples/online_serving/dreamzero/droid_sim_eval_client.py`
- Current code status: all three call `msgpack_numpy.unpackb(response)` and immediately convert to `np.asarray(..., dtype=np.float32)` in `infer()`.
- Risk: server-side inference errors are reported to users as confusing NumPy conversion `TypeError` instead of the real server error message.
- Proposed fix:
  - Decode binary responses first.
  - If decoded payload is a dict with `type == "error"`, raise `RuntimeError(decoded["message"])`.
  - Otherwise convert decoded action payload to `np.float32`.
  - Apply the same helper logic to all three clients to avoid drift.
- Test plan:
  - Add or update client unit tests for msgpack structured error payloads.
  - Verify normal array payloads still convert to `np.float32`.
- Confirmation: pending.
