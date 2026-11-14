# Praxis-1

Praxis-1 is Runway's closed-loop robot policy. From two or three camera views,
the arm's recent poses and a task string it jointly predicts future video and
Cartesian pose/gripper targets at 15 Hz. vLLM-Omni runs it natively as the
`Praxis1Pipeline` diffusion pipeline with two embodiments:

| Checkpoint | Rig | Views | Targets per row |
| --- | --- | --- | --- |
| `runwayml/praxis-1-droid` | DROID single arm (Franka) | exterior, wrist | 13: xyz, row-major 3x3 rotation, gripper |
| `runwayml/praxis-1-yam` | YAM two arms | exterior, left wrist, right wrist | 26: left then right |

The architecture is adapted from [runwayml/praxis-1-oss](https://github.com/runwayml/praxis-1-oss):
a chunk-causal video backbone (two MoE transformer towers over Wan2.2 latents),
an action expert that reads backbone taps, a joint Euler window sampler, the
Wan2.2 TI2V VAE and a Qwen3.5 text encoder.

## Usage

Download a checkpoint directory and keep the companion weights beside it. The
pipeline defaults to `Wan2.2_VAE.pth` and `text_encoder/` inside the checkpoint
directory; `model_config.vae_weights` and `model_config.text_encoder` override
the paths.

```text
/path/to/praxis-1-droid/
  config.json
  model.safetensors.index.json
  model-*.safetensors
  Wan2.2_VAE.pth     # Wan-AI/Wan2.2-TI2V-5B
  text_encoder/      # Qwen/Qwen3.5-2B snapshot
```

Pass the local directory to Omni with a deployment that selects the
pipeline. A minimal deployment runs one stage on one GPU; the same file works
for the DROID and the YAM checkpoint:

```yaml
# praxis1-deploy.yaml
async_chunk: false
dtype: bfloat16
stages:
  - stage_id: 0
    devices: "0"
    max_num_seqs: 1
    enforce_eager: true
    model_class_name: Praxis1Pipeline
    model_config:
      max_context_chunks: 1
```

```bash
vllm serve /path/to/praxis-1-droid --omni --served-model-name praxis-1 \
  --deploy-config praxis1-deploy.yaml --stage-init-timeout 1800
```

The
[recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/Runway/Praxis-1-H100.md)
has a runnable `Omni.generate` example. Offline requests pass `robot_obs` and a nonempty `session_id` in
`OmniDiffusionSamplingParams.extra_args`; reuse the id for continuation,
`reset` starts an episode and `end_session` releases it.

Closed-loop clients use [OpenPI](../serving/openpi_api.md) at
`/v1/realtime/robot/openpi`, which drives the same worker sessions.

### Observations and commands

DROID observations:

| Key | Contract |
| --- | --- |
| `image.exterior`, `image.wrist` | RGB uint8 `(T, H, W, 3)` |
| `robot_state.ee_pose` | `(T, 7)` xyz metres, quaternion **wxyz** |
| `grip` | `(T, 1)` or `(T,)`; 0 open, 1 closed |
| `task` or `prompt` | Task string; `task` takes precedence |
| `descriptor` | Pose frame of the client (see below); omitted means FR3 hand |

| Descriptor | Input and returned target frame |
| --- | --- |
| omitted, `""`, `remote.robolab.droid` | FR3 hand: 0.1034 m along flange z and Rz(-45°) |
| `praxis.sim` | Simulator flange: zero tool offset and Rz(-45°) |
| `remote.molmo_spaces.droid` | 0.155 m tool offset, 0.58 m base-z offset, no turn |

Other descriptors are rejected. Send `robot_state.q` when available so the
flange forward-kinematics check can catch pose-frame mismatches.

Views are centre-cropped to the native 720x1280 aspect, antialiased to 234x416
(unless already that size), centre-trimmed to 224x416 and tiled to 224x832.

YAM observations use `image.exterior`, `image.wrist_left`, `image.wrist_right`,
`robot_state.left.ee_pose`, `robot_state.right.ee_pose`, `grip.left`,
`grip.right` and `task`. Each pose is `[xyz3, quat_wxyz4]` in the rig world
frame. Views keep their native 16:10 aspect; 260x416 frames are trimmed to three
256x416 tiles. The checkpoint was trained on window captions such as
"Pick up the pen from the center of the table, then place the pen in the bag"
(no closing period); send the task in that form.

Continuation needs the executed frames and poses since the previous call plus
the current row; insufficient history restarts the context. Cold starts return
21 targets (1.4 s), later calls 24 (1.6 s). The episode seed advances with the
chunk index, so a reset reproduces the first call.

`multimodal_output["actions"]` is float32 `(T, 13)` or `(T, 26)`: absolute
client-frame `[xyz3, R_flat9, gripper]` per arm. `return_diagnostics` adds the
command dictionaries, normalized/relative actions, video latent, depth and seed
under `trajectory`; `decode_video` adds command-aligned uint8 `(T, 3, H, W)`
predicted frames.

### Deployment settings

Set these under `model_config` in a copy of the deployment:

| Setting | Default |
| --- | --- |
| `num_steps`, `cfg_scale`, `cfg_video_scale`, `t_shift` | DROID: 2 Euler steps, action/video guidance 1/4, shift 12.5. YAM: 2 steps, guidance 1/1, shift 1 |
| `attention_impl` | `flex` (block-sparse FlexAttention) or `backend` (dense masks through the Omni-selected attention backend; backends without dense-mask support fall back to SDPA) |
| `max_prompt_tokens` | 512 including special tokens; task text is capped at 64 KiB before tokenization |
| `max_num_steps` | 16; worker-enforced cap on every API |
| `max_sessions`, `max_context_chunks`, `exec_horizon` | 4 sessions, 1 context chunk (0 to 3), 24-row horizon (1 to 24); `context_mode=stateless` disables history |
| `control_mode` | DROID `droid_impedance` or `none`; YAM requires `none` |
| `session_idle_timeout` | 300 s for offline and OpenPI sessions; a transport that reserves a session keeps it alive |
| `warmup` | true; runs valid observations at the configured context depths before the server is ready |
| `text_cache_max_entries`, `text_cache_max_bytes` | 128 entries, 64 MiB; zero disables the prompt-embedding cache |

Failed inference discards the episode. The worker requires one GPU, bfloat16
and `max_num_seqs: 1`, and rejects parallelism, quantization, CPU offload,
sleep mode, LoRA and step execution at startup. Cache acceleration is not
applied: the pipeline owns its Euler loop.

## Attention and kernels

Video tokens attend under a chunk-causal visibility rule and the action expert
under a windowed rule, so the model owns a FlexAttention kernel behind Omni's
shared `Attention` layer. Block masks use 256-token blocks, which every
FlexAttention tile configuration divides, so one code path serves Hopper and
Blackwell GPUs. Routed experts run through vLLM's fused MoE kernels with the
checkpoint's sigmoid-plus-bias routing. No GPU-specific numerical branches are
used; see the validation notes below for what that implies.

The text encoder uses the Transformers Qwen3.5 implementation with vLLM
linear layers. Installing `flash-linear-attention` and `causal-conv1d` speeds up
its gated delta-rule layers; without them Transformers falls back to PyTorch
kernels. Prompt embeddings are cached per task string.

## Validation

The port is checked at three levels. Commands are in the
[recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/Runway/Praxis-1-H100.md#verification).

1. **Oracle parity on random weights.** A tiny transformer built by both the
   reference implementation and this port from the same random weights agrees
   to about 1e-4 relative in float32 at every context depth, including the
   block-sparse and dense attention paths.
2. **Real weights, same inputs.** With the DROID checkpoint, text embeddings
   are bitwise identical to the reference; the VAE latent and each transformer
   block differ at bfloat16 rounding level (about 3e-3 relative). The sampler
   is chaotic: two Euler steps with video guidance 4 amplify those rounding
   differences to tens of millimetres on individual targets, and the reference
   itself moves by a similar amount between two of its own builds (compiled
   versus eager FlexAttention, with or without the FLA text kernels). Bitwise
   parity with a specific reference build is therefore not a goal of this
   integration.
3. **Task success in simulation.** The reference repository's MuJoCo task
   suites, driven unchanged against this server over its robot-policy
   WebSocket API, are the qualification that matters. Results are recorded in
   the recipe.

Weight-free CPU and GPU tests cover the config contract, masks,
rotary tables, the sampler, weight mapping, observation processing and the
worker's session lifecycle.

Validated locally on one H100 80 GB. B200, GB300 and RTX PRO 6000 run the same
code path but have not been exercised for this port.
