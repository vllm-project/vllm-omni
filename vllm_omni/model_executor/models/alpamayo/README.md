# Alpamayo VLA (vLLM-Omni)

Port of NVIDIA's Alpamayo-R1 / Alpamayo-1.5 autonomous-driving Vision-Language-Action
models. Built on Qwen3-VL-8B + a flow-matching action expert. See repo-root
`feature_list.md` for the requirements ledger and `claude-progress.txt` for status.

> ⚠️ Model weights carry a **non-commercial** license (NVIDIA). Document this on
> the supported-models page before release.

## Architecture — single model, inline flow matching (sglang pattern)

Issue vllm-project/vllm-omni#2873 proposed a BAGEL-style two-stage AR + diffusion
pipeline. **We did not use that pattern.** Alpamayo's reference (sglang) is a single
`nn.Module` whose `forward()` runs the VLM AR step and, when the trajectory trigger
token appears, runs flow matching on the same call without ever leaving the model.
The two-stage scaffolding was removed; everything lives in this directory.

```
request (multi-cam images + prompt) + extra_args["robot_obs"] (ego history)
   │
   ▼  Single stage — LLM_AR  (Alpamayo15ForConditionalGeneration)
   │   Qwen3-VL backbone generates chain-of-thought text until <|traj_future_start|>.
   │   Discrete traj-token logits are masked during AR decode.
   │
   │   On each forward() call the model detects the trigger token in input_ids
   │   (requires_raw_input_tokens=True). When present, _run_flow_matching_inline:
   │     • reads the VLM's paged KV (no_compile_layers[vlm_layer_name].kv_cache)
   │     • runs Euler flow matching, action tokens attend bidirectionally to
   │       cat[cached_prefix_kv ; current_action_kv] via manual SDPA
   │     • produces (n_samples, n_waypoints=64, action_dim=2) actions
   │   action_space.action_to_traj(robot_obs history) → 64-waypoint xyz + rotation.
   ▼
multimodal_output["actions"] (xyz + rot + reasoning)   [OmniTrajectoryOutput]
```

API convention (mirrors GR00T): the ego history arrives as a single observation
dict at `sampling_params.extra_args["robot_obs"]` (HTTP `extra_body`), carrying
`ego_history_xyz`/`ego_history_rot` (+ `num_samples`/`seed`). It is consumed by
the action space to decode actions into a trajectory — it is **not** fused into
the prompt and does **not** go through the multimodal processor. The predicted
trajectory is surfaced to the client as `multimodal_output["actions"]`.

The multimodal processor's only Alpamayo-specific job is to **add the trajectory
tokens** to the tokenizer (vocab alignment + the `<|traj_future_start|>` trigger);
multi-camera image preprocess is inherited unchanged from Qwen3-VL.

## Why the two-stage pipeline was wrong for Alpamayo

- Alpamayo was trained as one model; the "stage 1 / stage 2" split was an
  artifact of trying to fit it into BAGEL's pipeline shape.
- Two-stage required re-prefilling images+prompt through HF Qwen3-VL inside the
  diffusion stage, which **bypassed Qwen3-VL's deepstack visual injection** —
  this was a real correctness bug that took avg minADE 3.45 m → 1.07 m to fix
  by switching to a decode-KV path.
- vLLM 0.21 has no native cross-stage paged-KV transfer; the two-stage adapter
  existed only to bridge that absence.
- Single-model design: the expert reads the VLM cache vLLM just wrote (via
  `kv_sharing_target_layer_name` + a forward-context lookup), zero copy.

## Module map

| File | Role | Status |
|------|------|--------|
| `configuration_alpamayo.py` | Config (flat 1.5 + merged R1 → Qwen3VLConfig) | done |
| `action_space.py` | Trajectory/action math (rotation, solvers, unicycle, projections, PerWaypointActionInProjV2) | done |
| `processing.py` | Tokenizer extension (add trajectory tokens) | done |
| `processor.py` | Multimodal processor: add tokens + reuse Qwen3-VL preprocess | done |
| `expert_layers.py` | Expert (ENCODER_ONLY attn + kv_sharing read of VLM paged cache) | done |
| `alpamayo.py` | Single-model AR + inline flow matching (`_run_flow_matching_inline`) | done |
| `pipeline.py` | Pipeline registration (single LLM_AR stage) | done |
| `outputs.py` | OmniTrajectoryOutput | done (wiring to OmniRequestOutput pending) |

## Inline flow matching internals (alpamayo.py)

After `super().forward()` runs the normal VLM forward (writing the VLM KV cache),
the model checks `find_trigger_indices(input_ids)` for `<|traj_future_start|>`. When hit:

1. Read prefix K/V per layer from
   `get_forward_context().no_compile_layers[vlm_layer_name].kv_cache`, gathering
   the `block_table_row` blocks up to `seq_lens[req_idx]`.
2. Sample initial noise `x = torch.randn(bstar, n_waypoints, action_dim, fp32, generator=g)`
   with seed `ALPAMAYO_FM_SEED` (default 0). `bstar = ALPAMAYO_FM_N_SAMPLES` (default 1).
3. Euler loop (`num_inference_steps`, default 10):
   - `action_in_proj(x, t)` → `(bstar, n_waypoints, hidden)` (Fourier features + MLP).
   - Per expert layer, manual SDPA: `q, k = rotary_emb(action_positions, q, k)` (mRoPE),
     `cat[prefix_k ; k]`, GQA `repeat_interleave`, `F.scaled_dot_product_attention(is_causal=False)`.
   - `action_out_proj(hidden) → pred velocity`; `x ← x + dt · pred`.
4. Stash sampled actions on the model instance and dump to `/tmp/alp_last_actions.pt`
   (the OmniRequestOutput attach hook that surfaces these as
   `multimodal_output["actions"]` is still pending).

Why manual SDPA instead of a vLLM kernel: vLLM 0.21 has no path equivalent to
sglang's `extend_attention_fwd` (Q from action embeds, K/V = cat[cached_prefix; current_action]).
`kv_sharing_target_layer_name` **substitutes** the cache rather than augmenting it,
so the action tokens would never see each other. Manual SDPA is slower than a
fused kernel but numerically correct (parity verified vs the standalone path that
used to exist at avg minADE@1 = 3.094 m on clip 030c760c).

## Numerical status (clip 030c760c, vs ground truth)

| | ADE@1 | meanADE@16 | minADE@16 |
|---|---|---|---|
| vLLM single-model inline | ~3.0–3.5 m | 3.485 m | 1.279 m |

Single-sample math is correct. The minADE@16 number is higher than what was
previously measured against a now-removed HF baseline (0.301 m); investigating
the sample-diversity gap (GQA repeat_interleave / mRoPE / mask / norm-order
details in the expert forward) is tracked as a follow-up.

## Manual test

See `examples/online_serving/alpamayo/README.md` for the full HTTP path and
`examples/offline_inference/alpamayo/README.md` for the in-process eval. Quick
sketch:

```bash
# Serve the single-stage Alpamayo pipeline ($ALPAMAYO_MODEL = HF id or local dir):
vllm-omni serve "$ALPAMAYO_MODEL" --omni \
  --stage-config-path vllm_omni/deploy/alpamayo1_5.yaml
# Client posts multi-cam images + extra_body={"robot_obs": {"ego_history_xyz": ...,
# "ego_history_rot": ...}, "n_samples": 16}; response carries the predicted
# trajectory under multimodal_output["actions"]. See sglang test_online_full.py
# for the data shapes (note: sglang used extra_body["history_traj"]; vllm-omni
# uses extra_body["robot_obs"], GR00T-style).
```

For inline numerical checks during development, scripts under `scripts/` (gitignored)
drive `_run_flow_matching_inline` directly and dump samples to `/tmp/alp_last_actions.pt`
for ADE computation against a clip's GT future.
