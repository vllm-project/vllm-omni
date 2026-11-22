# Reference replay-visibility fixtures

Generated inside the imaginaire4 Cosmos3 environment with

```bash
PYTHONPATH="$PWD/packages/cosmos3:$PWD" python -m projects.cosmos3.interactive.utils.sim_auto_visibility_fixtures \
    --output <vllm-omni>/tests/diffusion/models/cosmos3_nano_sim_auto/fixtures/reference_visibility
```

Each file holds the token-level `allowed` matrices of the reference teacher-forcing replay
predicate (production policy: causal controls, strictly-past clean history for controls,
chunk-causal clean pass, decomposed scope with a 0.4 s past-only window, deduplicated
cross-view keys) for chunks 0..3 of a two-camera rollout with a two-slot KV ring, in noisy
and clean passes, with and without the first-frame RGB condition and the LiDAR condition
sweep. Tokens are identified by `(view, frame, role, occurrence)` so the vLLM-Omni planner can
be compared independently of packing order. `test_reference_visibility.py` skips when the
directory holds no fixtures.
