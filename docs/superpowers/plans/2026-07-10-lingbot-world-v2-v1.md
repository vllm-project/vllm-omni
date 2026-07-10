# LingBot-World 2.0 v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to execute this plan task-by-task.

**Goal:** Add a draft-quality v1 integration for the official `robbyant/lingbot-world-v2-14b-causal-fast-diffusers` checkpoint, supporting one prompt, one first-frame image, one camera action directory, and request-local causal chunk generation.

**Architecture:** Implement LingBot-World as an independent Wan-family variant under `wan2_2`. Keep camera geometry pure and testable, implement a checkpoint-compatible causal transformer with request-local KV caches and TP-aware projections, then wrap it in a vLLM-Omni diffusion pipeline that performs the fixed four-step DMD schedule and registers the Hugging Face pipeline class name.

**Tech Stack:** Python 3.12, PyTorch, NumPy, SciPy, vLLM distributed linear/attention primitives, vLLM-Omni diffusion pipeline APIs, pytest.

## Global Constraints

- Work only in `/Users/bytedance/dev/vllm-omni/.worktrees/lingbot-world-v2-v1` on branch `codex/lingbot-world-v2-v1`.
- Reimplement behavior from public checkpoint contracts and Apache-licensed vLLM/SGLang patterns; do not copy source from the CC BY-NC-SA upstream repository.
- V1 supports only the official 14B causal fast Diffusers checkpoint and the DMD timesteps `[1000, 750, 500, 250]` with scheduler `flow_shift=5.0`.
- Input contract: first frame from `multi_modal_data["image"]`; camera directory from `sampling_params.extra_args["action_path"]` or prompt `additional_information["action_path"]`; reject ambiguous dual specification.
- Camera directory must contain `poses.npy` shaped `[N, 4, 4]` and `intrinsics.npy` shaped `[N, 4]`.
- Cache lifetime is one request only. Do not add cross-request sessions, WebSocket state, runtime prompt events, or runtime camera events.
- Use TP-aware linear and attention primitives. SP/Ulysses, CFG parallel, Cache-DiT, TeaCache, VAE parallelism, the causal-pretrain checkpoint, and the 1.3B checkpoint are out of scope.
- Use explicit `ValueError`/`RuntimeError` validation in production code; do not use `assert` for user inputs.
- Local macOS verification can prove pure CPU code and syntax. Tests importing vLLM may be CI-only because vLLM has no supported macOS arm64 wheel; document this boundary instead of weakening tests.
- Every task follows red-green-refactor, ends in one focused commit, and records the exact verification output in its report.

---

### Task 1: Camera trajectory loading and checkpoint ray conditioning

**Files:**

- Create: `vllm_omni/diffusion/models/wan2_2/lingbot_world_camera.py`
- Test: `tests/diffusion/models/wan2_2/test_lingbot_world_camera.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class CameraTrajectory:
    poses: torch.Tensor       # [frames, 4, 4]
    intrinsics: torch.Tensor  # [frames, 4], ordered fx, fy, cx, cy

def load_camera_trajectory(action_path: str | os.PathLike[str]) -> CameraTrajectory: ...

def interpolate_camera_trajectory(
    trajectory: CameraTrajectory,
    num_frames: int,
) -> CameraTrajectory: ...

def build_plucker_embedding(
    trajectory: CameraTrajectory,
    *,
    height: int,
    width: int,
    target_height: int,
    target_width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:  # [frames, 6, target_height, target_width]
    ...
```

- `load_camera_trajectory` resolves both files, rejects missing/non-finite/invalid shapes, converts NumPy arrays to `float32` tensors, and preserves the raw camera-to-world poses.
- Interpolate raw translations linearly and raw rotations with SciPy quaternion SLERP when the requested frame count differs; interpolation must happen before relative/framewise conversion.
- After interpolation, normalize poses into the first camera frame, compute framewise deltas as `inv(relative[i-1]) @ relative[i]` for frames `1..N-1`, set frame 0 to identity, and divide all translations by the maximum translation norm when nonzero.
- Scale `fx, fy, cx, cy` from the checkpoint reference resolution `480 x 832` to the requested pixel resolution.
- Use half-pixel centers and build six checkpoint channels in exact `[ray_origin_xyz, normalized_ray_direction_xyz]` order.
- Resize/sample the ray grid directly at the requested latent/camera-conditioning spatial resolution without materializing an unnecessarily large repeated tensor.

This camera section corrects the early plan convention to match the Apache SGLang checkpoint adapter;
the implementation reimplements the compact math and does not copy the CC BY-NC-SA upstream source.

**Steps:**

- [ ] Write tests for missing files, invalid shapes/non-finite values, raw-load semantics, interpolation-before-framewise conversion, nontrivial framewise rotations/translations, max-norm translation normalization, half-pixel ray direction, channel order, folded placement, output shape/dtype/device, and deterministic results.
- [ ] Run the focused test and capture the expected RED failure because the module does not exist.
- [ ] Implement the smallest pure camera module satisfying those tests.
- [ ] Run `../../.venv/bin/python -m pytest --noconftest -q tests/diffusion/models/wan2_2/test_lingbot_world_camera.py`; expected result: all tests pass on CPU.
- [ ] Run `../../.venv/bin/python -m compileall -q vllm_omni/diffusion/models/wan2_2/lingbot_world_camera.py`.
- [ ] Self-review for coordinate conventions, dtype/device transfers, validation messages, and accidental upstream-source copying.
- [ ] Commit with message `feat: add LingBot camera conditioning`.

---

### Task 2: Causal attention and request-local KV cache primitives

**Files:**

- Create: `vllm_omni/diffusion/models/wan2_2/lingbot_world_transformer.py`
- Test: `tests/diffusion/models/wan2_2/test_lingbot_world_attention.py`

**Interfaces:**

```python
@dataclass
class LingBotAttentionCache:
    key: torch.Tensor
    value: torch.Tensor
    end: int = 0

@dataclass
class LingBotTransformerCache:
    self_attention: list[LingBotAttentionCache]
    cross_attention: list[LingBotAttentionCache | None]

def allocate_lingbot_cache(
    *, batch_size: int, num_layers: int, max_tokens: int,
    num_local_heads: int, head_dim: int, device: torch.device,
    dtype: torch.dtype,
) -> LingBotTransformerCache: ...
```

- Add checkpoint-named attention modules `self_attn` and `cross_attn`, with separate checkpoint-compatible `q`, `k`, `v`, `o`, `norm_q`, and `norm_k` parameters.
- Use vLLM `ColumnParallelLinear`, `RowParallelLinear`, and registered `Attention` layers; preserve the checkpoint parameter names even if runtime execution uses fused views.
- Self-attention accepts a token offset and updates a preallocated per-layer request cache. It attends causally to the configured sink tokens plus sliding/local history and the current chunk.
- Cross-attention computes encoder K/V only on its first call per request and reuses the cached values on later chunks.
- Cache objects must be supplied by the caller; no module-global or pipeline-global mutable cache.

**Steps:**

- [ ] Write tiny-shape tests for cache allocation, append offsets, causal visibility, sink retention after sliding-window eviction, cross-attention K/V reuse, cache isolation between requests, and TP-world-size-1 output shapes.
- [ ] Run the focused test in a vLLM-capable environment and capture the expected RED failure. On local macOS, use collection/compile evidence and record the environment blocker precisely.
- [ ] Implement cache dataclasses, allocation, rotary-position plumbing, self-attention, and cross-attention using existing Wan/DreamZero conventions without importing robot-action behavior.
- [ ] Run the focused test in CI/Linux; expected result: all tests pass. Locally run `../../.venv/bin/python -m compileall -q vllm_omni/diffusion/models/wan2_2/lingbot_world_transformer.py tests/diffusion/models/wan2_2/test_lingbot_world_attention.py`.
- [ ] Self-review masking math, cache offsets, checkpoint names, TP head sharding, and request isolation.
- [ ] Commit with message `feat: add LingBot causal attention cache`.

---

### Task 3: Checkpoint-compatible LingBot transformer

**Files:**

- Modify: `vllm_omni/diffusion/models/wan2_2/lingbot_world_transformer.py`
- Test: `tests/diffusion/models/wan2_2/test_lingbot_world_transformer.py`

**Public class:**

```python
class CausalLingBotWorldTransformer3DModel(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        camera_hidden_states: torch.Tensor,
        *,
        cache: LingBotTransformerCache,
        start_frame: int,
        update_cache: bool,
    ) -> torch.Tensor: ...

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
```

- Match the official checkpoint topology and names: `patch_embedding`, `patch_embedding_wancamctrl`, `c2ws_hidden_states_layer1/2`, `text_embedding`, `time_embedding`, `time_projection`, `blocks`, and `head`.
- Default config contract: `in_channels=36`, `out_channels=16`, `num_layers=40`, `num_attention_heads=40`, `attention_head_dim=128`, `patch_size=(1,2,2)`, `text_dim=4096`, `sink_size=9`, `num_frames_per_block=3`, `sliding_window_num_frames=18`, `local_attn_size=-1`.
- Each block contains modulation, self-attention, cross-attention, FFN, `cam_injector_layer1`, `cam_injector_layer2`, `cam_scale_layer`, and `cam_shift_layer` with checkpoint-compatible names.
- Camera features enter through the camera patch embed and per-block injection path; do not concatenate camera channels onto the 36-channel video tensor.
- `load_weights` uses parameter-specific vLLM weight loaders and reports loaded names. It ignores only explicitly documented non-model metadata; unexpected or missing model parameters remain visible.

**Steps:**

- [ ] Write a tiny-config forward test covering video patching/unpatching, timestep/text conditioning, camera injection, four sequential chunk calls, `update_cache=False` versus `True`, and output shape.
- [ ] Add a synthetic named-weight test proving all model parameters load and unknown model keys are rejected/reported.
- [ ] Add a Hugging Face index audit test that compares expected top-level/module naming against a checked-in compact fixture derived from the public weight index, without downloading tensors.
- [ ] Run the focused tests and capture RED before adding the full model.
- [ ] Implement the block stack, embeddings, head, forward path, config parsing, and weight loading.
- [ ] Run focused tests in CI/Linux and compile locally with `../../.venv/bin/python -m compileall -q vllm_omni/diffusion/models/wan2_2/lingbot_world_transformer.py tests/diffusion/models/wan2_2/test_lingbot_world_transformer.py`.
- [ ] Self-review checkpoint name parity, TP shard axes, camera tensor layout, cache update timing, and absence of robot-action/state code.
- [ ] Commit with message `feat: add LingBot world transformer`.

---

### Task 4: Pipeline, fixed DMD chunk loop, and registry wiring

**Files:**

- Create: `vllm_omni/diffusion/models/wan2_2/pipeline_lingbot_world.py`
- Modify: `vllm_omni/diffusion/models/wan2_2/__init__.py`
- Modify: `vllm_omni/diffusion/registry.py`
- Test: `tests/diffusion/models/wan2_2/test_pipeline_lingbot_world.py`

**Public class:**

```python
class LingBotWorldCausalDMDPipeline(
    nn.Module,
    SupportImageInput,
    SupportsComponentDiscovery,
    ProgressBarMixin,
):
    ...
```

- Discover/load the scheduler, UMT5 text encoder/tokenizer, Wan VAE, and `CausalLingBotWorldTransformer3DModel` through normal vLLM-Omni component loading.
- Parse one prompt, one image, and one camera action directory. Reject absent/multiple images, absent/ambiguous action paths, invalid dimensions, unsupported batching, unsupported `num_inference_steps`, and insufficient camera frames with actionable errors.
- Encode the first image with the Wan VAE. Construct the 20-channel condition as `[mask4, image_latent16]`, then concatenate it after 16 noise channels to form the exact transformer input `[noise16, mask4, image_latent16]`.
- Use scheduler `flow_shift=5.0` and exactly four DMD timesteps `[1000, 750, 500, 250]` per generated chunk.
- After the fourth denoising step, perform the required `t=0` cache-update forward, append the generated latent chunk, and continue until the requested frame count is reached.
- Allocate transformer cache at request entry and discard it on return or error.
- Decode accumulated latents through the Wan VAE and return normal vLLM-Omni diffusion pipeline output.
- Register `_class_name == "LingBotWorldCausalDMDPipeline"` and export the pipeline/transformer classes from the Wan package.

**Steps:**

- [ ] Write stub-component tests for input parsing, action-path source precedence/ambiguity, exact sentinel channel positions, first-frame condition channels, fixed timestep ordering, one transformer-owned cache allocation per request, `t=0` update call, multi-chunk concatenation, cache teardown, and registry lookup.
- [ ] Run the focused test and capture RED before creating the pipeline.
- [ ] Implement component discovery/loading, prompt/image/camera preparation, latent creation, the chunk loop, decode/postprocess, exports, and registry entry.
- [ ] Run the focused tests in CI/Linux; locally compile all changed files and run any pure helpers that do not import vLLM native extensions.
- [ ] Run `../../.venv/bin/python -m compileall -q vllm_omni/diffusion/models/wan2_2 vllm_omni/diffusion/registry.py tests/diffusion/models/wan2_2`.
- [ ] Self-review component prefixes, device/dtype placement, generator determinism, cancellation/error cleanup, and registry class-name spelling.
- [ ] Commit with message `feat: add LingBot world causal pipeline`.

---

### Task 5: Offline example, documentation, and integration validation entry

**Files:**

- Create: `examples/offline_inference/diffusion/lingbot_world_v2.py`
- Create: `tests/e2e/offline_inference/test_lingbot_world_v2.py`
- Modify: the nearest diffusion model-support documentation/table identified in the repository
- Modify: `docs/superpowers/specs/2026-07-10-lingbot-world-v2-design.md` only if implementation decisions legitimately differ, and explain any change

**Example contract:**

```python
model = "robbyant/lingbot-world-v2-14b-causal-fast-diffusers"
# image is the first-frame input
# action_path contains poses.npy and intrinsics.npy
# extra_args includes action_path; num_inference_steps is 4
```

- The example exposes prompt, image path, action directory, height, width, frame count, seed, tensor parallel size, and output path via CLI flags.
- The E2E test is GPU-marked/slow and can be opt-in, but must exercise real component discovery and at least one generated chunk when hardware/model access is available.
- Documentation lists v1 limits honestly: 14B causal fast checkpoint only, fixed four DMD steps, image plus camera input, request-local history, TP-aware but no SP/Cache-DiT/streaming session support.

**Steps:**

- [ ] Write/extend tests for example argument construction and add the opt-in GPU E2E entry before the example implementation.
- [ ] Implement the runnable offline example using current repository conventions rather than a bespoke engine path.
- [ ] Update model support documentation and v1 limitation notes.
- [ ] Run `../../.venv/bin/python -m compileall -q examples/offline_inference/diffusion/lingbot_world_v2.py tests/e2e/offline_inference/test_lingbot_world_v2.py`.
- [ ] Run all LingBot CPU/unit tests available locally; record vLLM/macOS and GPU limitations separately from test failures.
- [ ] On Linux/CUDA, run the focused unit suite and the opt-in E2E smoke command. Capture peak memory, output shape/frame count, and generated artifact path in the task report.
- [ ] Self-review CLI usability, public checkpoint spelling, docs accuracy, and validation claims.
- [ ] Commit with message `docs: add LingBot world v2 usage`.

---

## Final Branch Verification

- [ ] Generate a whole-branch review package from `9a44e7e0` to `HEAD` and obtain a clean final code review.
- [ ] Run formatting/lint commands required by repository pre-commit configuration on all changed Python files.
- [ ] Run all locally supported LingBot tests and compile checks; state exactly which CUDA/vLLM checks were not executable locally.
- [ ] Inspect `git diff --check`, `git status --short`, commit history, and staged scope.
- [ ] Push `codex/lingbot-world-v2-v1` to the contributor fork.
- [ ] Open a draft PR against `vllm-project/vllm-omni:main` with `Closes #4990`, the supported v1 contract, implementation summary, verification table, and explicit runtime-validation limitations.
