# World Action Model: Impact Analysis for vLLM-Omni

> Discussion document — synthesized from vllm-omni RFC #1987, PR #2162, upstream WAM research, and Thinking Machines interaction models.
> Date: 2026-05-18

---

## 1. Background

Two concepts have emerged in 2025–2026, both shifting AI from one-shot generation to continuous interactive loops.

**World Action Models (WAM) — NVIDIA / Embodied AI.** Jim Fan (NVIDIA GEAR Lab) declared VLA dead at Sequoia AI Ascent 2026. WAMs jointly predict how the world evolves AND what action to take — grounding decisions in physics rather than pattern matching. Flagship: **DreamZero** (14B, Wan 2.1 backbone). Consumes multi-view images + language + proprioceptive state; outputs future video frames + action sequences. DreamZero-Flash achieves **150ms/7Hz** on GB200 via 38× optimization. Key properties: autoregressive chunk diffusion with blockwise causal attention, KV cache over past observations, flow matching across video+action modalities, single-step action decoding.

**Interaction Models — Thinking Machines Lab.** Mira Murati's lab shipped **TML-Interaction-Small** (May 2026): 276B MoE (12B active), full-duplex 200ms micro-turns. Encoder-free early fusion (dMel for audio, patch hMLP for images), persistent GPU KV cache, dual-model system (foreground interaction + background agent). Focused on human-AI conversation, but overlaps with WAM on streaming sessions, full-duplex APIs, and persistent KV cache.

---

## 2. The Multi-Front Convergence

Turn-based inference is becoming legacy. A structural shift toward **continuous interactive sessions** is unfolding across every model class simultaneously.

| Domain | Representative Systems | Pattern | Why It Matters |
|--------|----------------------|---------|----------------|
| **Conversational AI** | TML-Interaction, GPT-Realtime 2.0, Gemini 3.1 Flash Live, Moshi | 200ms micro-turns, full-duplex, persistent KV | Break "you talk, I respond" — model listens continuously, accepts interruption |
| **Embodied AI** | DreamZero (7Hz), Dream Dojo (>10 FPS), AgiBot GO-1, Pi0/OpenVLA | Closed-loop observe→act→observe | Actions change the world; model must account for consequences of its own output |
| **Video Generation** | Helios (16–19 fps), Sana-WM (720p/1min), Matrix Game 3.0, StreamDiffusionV2 | Chunked streaming output, mid-stream steering input | Generation speed crossing into real-time territory |
| **Infrastructure** | SGLang (streaming sessions), vLLM Realtime API, OpenPI, Mooncake | Persistent KV sessions, bidirectional WebSocket | Frameworks racing to support the pattern generically |

The convergence is driven by three forces:
1. **KV cache becomes session state.** Stop freeing the KV cache between turns and you unlock all continuous interaction patterns — TTS streaming, world model control loops, conversational AI.
2. **Latency unlocks new modalities.** Sub-200ms turns video diffusion into a game engine, speech into a conversational partner, action prediction into a robot controller.
3. **Full-duplex is the end state.** Turn-based was a workaround for models too slow to be interactive. As speed improves, everything converges on bidirectional streaming.

### 2.1 The Deeper Shift: Agentic AI → Physical AI

The convergence is the leading edge of a paradigm shift from **digital agentic AI** to **physical AI**.

| Dimension | Agentic AI (digital) | Physical AI (embodied) |
|-----------|---------------------|------------------------|
| **Action space** | Discrete, reversible (API calls, tool invocations) | Continuous, irreversible (joint torques, locomotion) |
| **Consequences** | Known and deterministic | Must be observed — no API schema for physics |
| **State** | Explicit text/JSON | Implicit latent state — the KV cache *is* the world model |
| **Observation** | Arrives when user sends it | Arrives continuously whether model is ready or not (30Hz cameras) |
| **Latency** | UX preference: 1–3s acceptable | Safety constraint: >150ms = control instability |
| **Causality** | Open-loop capable | Closed-loop by necessity: every action changes the world |
| **Data** | Text-dominant | 99.9% egocentric video, <0.1% action labels |
| **Deployment** | Datacenter: 8×H100, throughput batching | Edge + cloud split: onboard GPU (Jetson) + optional cloud offload |
| **Evaluation** | Deterministic benchmarks | Sim-to-real gap — success in sim ≠ success in reality |
| **Failure** | Retryable | Irreversible: collision, damage |

Three structural shifts for inference infrastructure:

1. **KV cache as world state.** In digital AI, KV cache is a performance optimization. In physical AI, it *is* the model's memory — what it observed, what it did, how the world responded. Freeing it erases causal understanding.

2. **Inference loop becomes a control loop.** Physical AI runs observe→predict→act→observe indefinitely. Inference is continuous at fixed frequency (7Hz+), not "as fast as possible." Jitter matters as much as mean latency. The scheduler needs hard deadlines, not just throughput.

3. **Simulation becomes a first-class serving mode.** Policies are trained and evaluated in simulation (Isaac Sim / MuJoCo) because real-world testing is expensive and dangerous. The API, pipeline, and performance must be identical across sim and real modes — any divergence creates a sim-to-real gap that breaks the policy.

The repo was built for **open-loop generative media** (text→image/video/audio). Physical AI demands **closed-loop** serving. The WAM infrastructure being designed today — session framework, persistent KV, OpenPI serving, real-time control loop — is the foundation for this new workload class.

---

## 3. Gap Analysis vs Current vLLM-Omni

### 3.1 New Modality: Actions

| | Current | WAM Requirement |
|---|---|---|
| I/O types | text, audio, image, video | + actions (poses, joint angles, controls), + proprioceptive state, + multi-view images |
| Encoding | T5, CLIP, VAE | + state encoder, + action decoder (flow-matching head) |
| Data layer | `DiffusionData` with image/video/audio | + `actions` field, + `robot_obs` dict |

**Status**: PR #2162 adds `robot_obs` ingestion and action transforms, but actions are model-specific, not a first-class modality in the type system or API schema.

### 3.2 Multi-Turn Stateful Sessions

| | Current | WAM Requirement |
|---|---|---|
| Session model | Single-turn: text→audio/video | Multi-turn loop: observations→actions→... indefinitely |
| Context | Stateless per request | Accumulated across turns |
| KV cache | Allocated per request, freed after | Persistent per session, append-only with sliding window |
| API | REST + WS for TTS | Bidirectional WebSocket with session lifecycle |

**Status**: PR #2162 builds OpenPI WebSocket serving, but the session framework, KV buffer management, and protocol are DreamZero-specific. Needs generic `SessionStore` abstraction.

### 3.3 Autoregressive Diffusion Engine

| | Current | WAM Requirement |
|---|---|---|
| Diffusion mode | Full-sequence denoising | Autoregressive chunk diffusion |
| Attention | Bidirectional (full) | Blockwise causal (causal across chunks, bidirectional within) |
| KV management | No persistence across steps | Persist across both denoising steps AND chunks |

**Status**: Not yet built. RFC #1987 proposes `CausalWanModel` with blockwise causal attention and preallocated KV buffers. SGLang treats dLLM as chunked prefill — same abstraction.

### 3.4 Real-Time Control Loop (7Hz Target)

| | Current | WAM Requirement |
|---|---|---|
| Latency | 1–30s (batch generation) | <150ms per step |
| Pipeline | Synchronous encode→denoise→decode | Async overlapped stages |
| Concurrency | 1–2 diffusion requests | Many concurrent sessions, each at 7Hz |

**Status**: DreamZero measures ~7.3s/step on RTX PRO 6000 — 50× off target. Optimization path: quantization (FP8/INT8), async pipeline, DiT caching, torch.compile, pipeline parallelism.

### 3.5 Ecosystem Integration

| Ecosystem | Purpose | Status |
|-----------|---------|--------|
| OpenPI | Robot policy serving standard | ✅ In PR #2162 |
| molmospace | Multi-sim benchmark harness | ✅ In PR #2162 |
| LeRobot | Open-source robot eval | Not started |
| ROS 2 | Standard robot middleware | Not started |
| Isaac Sim / Lab | NVIDIA simulation | Partial (DROID eval in #2162) |

### 3.6 RL Training

WAMs improve via RL in simulation. Issue #3435 proposes deterministic rollouts. Needs: deterministic generation (batch invariance, bitwise alignment), RL framework integration (RLinf, VERL), reward signal plumbing, and Dream Dojo-style neural simulation loop.

---

## 4. Requirements Summary

### P0 — Blocking for first WAM support (DreamZero)

| # | Requirement | Where | Status |
|---|------------|-------|--------|
| R1 | Action modality in request/response types | `DiffusionData`, API schema | 🔶 PR #2162 |
| R2 | Multi-turn WebSocket with session lifecycle | `entrypoints/.../realtime/robot/` | 🔶 PR #2162 |
| R3 | Blockwise causal attention | `diffusion/attention/backends/` | ❌ |
| R4 | Preallocated KV buffer for AR diffusion | `diffusion/sched/`, engine | ❌ |
| R5 | Robot observation ingestion + transform | `stage_input_processors/` | 🔶 PR #2162 |
| R6 | Action decoding head (flow-matching) | `diffusion/models/dreamzero/` | 🔶 PR #2162 |
| R7 | CFG parallel | `diffusion/distributed/` | ✅ PR #2162 |

### P1 — Production WAM serving

| # | Requirement | Where | Status |
|---|------------|-------|--------|
| R8 | Generic session framework (TTS + WAM) | `engine/session_store.py` | ❌ |
| R9 | Async pipeline (overlapped stages) | `diffusion/diffusion_engine.py` | ❌ |
| R10 | DiT cache for WAM (TeaCache / Flash) | `diffusion/cache/` | ❌ |
| R11 | FP8/INT8 quantization for DiT | `quantization/` | ❌ |
| R12 | torch.compile for WAM | `diffusion/compile.py` | ❌ |
| R13 | LeRobot protocol adapter | `entrypoints/.../realtime/robot/` | ❌ |
| R14 | Multi-embodiment config | `diffusion/models/*/config.py` | ❌ |

### P2 — Platform & ecosystem

| # | Requirement | Where | Status |
|---|------------|-------|--------|
| R15 | PageAttention for AR diffusion KV | `diffusion/attention/` | ❌ |
| R16 | RL training integration | `diffusion/`, RLinf bridge | ❌ RFC #3435 |
| R17 | ROS 2 bridge | `entrypoints/` or plugin | ❌ |
| R18 | Interactive video streaming | `serving_video_stream.py` | ❌ RFC #3632 |
| R19 | 3D/depth/keyframe stage support | `stage_configs/` | ❌ |
| R20 | Neural simulation loop | New subsystem | ❌ Research |

---

## 5. Models in Scope

| Model | Type | Size | Backbone | Status |
|-------|------|------|----------|--------|
| **DreamZero** | Robotics WAM | 14B | Wan 2.1 DiT | 🔶 PR #2162 |
| **LingBot-VA** | Robotics WAM | — | Video DiT | 🔶 #2885 |
| **InternVLA-A1** | VLA | 7B | InternVL | ✅ #1948 |
| **LingBot-World** | Interactive video | — | Wan 2.2 I2V | ✅ #2073 |
| **GR00T-N1.7** | VLA/WAM | 3B | — | 🔶 RFC #3553 |
| **Sana-WM** | Interactive video WAM | 2.6B | DiT + Gemma | ❌ #3656 (weights pending) |
| **Matrix Game 3.0** | Interactive game WAM | — | — | ❌ Discussed in #1987 |
| **Cosmos WFM** | Simulation WAM | — | — | Not requested |
| **Pi0 / OpenVLA** | VLA | — | — | #1948, #3471 |

---

## 6. Architectural Implications

### Stays the same
Stage pipeline (orchestrator → stage pool → diffusion engine), model registry, attention backends (extended, not replaced), CFG parallel / TP / SP primitives.

### New subsystems needed

```
vllm_omni/
├── session/                    # Generic multi-turn session framework
│   ├── session_store.py        #   Lifecycle, KV buffer management
│   └── protocols/              #   openpi.py, lerobot.py, ros2.py
├── diffusion/
│   ├── attention/backends/blockwise_causal.py
│   ├── cache/kv_buffer.py
│   └── models/{dreamzero,sana_wm,gr00t,matrix_game}/
├── entrypoints/openai/realtime/robot/
│   └── robot_session.py        # Extracted from dreamzero-specific code
└── rl/                         # Deterministic rollout + RL framework bridge
```

### Cross-cutting: Action as first-class modality
Currently model-specific dicts in `extra_args`. Must become typed, validated, serializable — touching `DiffusionData`, API schema, stage input processors, and the ZMQ serialization layer. Streaming output infrastructure from RFC #3632 can serve both video frames and action predictions incrementally.

---

## 7. Open Questions

1. **Session framework scope**: Generic (TTS + WAM) upfront, or start WAM-specific and generalize later?

2. **PageAttention migration**: When does fixed-buffer KV become a bottleneck — how many concurrent sessions?

3. **API standard**: Commit to OpenPI as primary protocol, or keep protocol-agnostic abstractions?

4. **Latency budget**: DreamZero at ~7.3s/step → optimizations to ~500ms. 150ms/7Hz may require Blackwell-class hardware. v0 latency target?

5. **RL ownership**: Full training loop, or just the deterministic inference backend?

6. **3D/depth preprocessing**: vllm-omni pipeline stages or client-side?

7. **One engine or two?** The DiT forward pass is identical. The fork is in the scheduler:

    | | Generative Media | Physical AI |
    |---|---|---|
    | Goal | Throughput (images/sec) | Deadlines (7Hz, <150ms) |
    | Batching | Amortize across requests | One session per forward |
    | KV lifecycle | Allocate → use → free | Append → slide → persist |
    | I/O | One request, one response | Bidirectional stream |
    | Error | Regenerate | Control instability |
    | Deployment | 8×H100 datacenter | Edge GPU or edge+cloud split |

    **Scheduler mode** on the existing engine (`scheduler_mode: "throughput" | "realtime"`) maximizes reuse. **Separate lightweight executor** makes edge deployment, sim stepping, and hard-deadline scheduling cleaner.

8. **Edge deployment**: Does vllm-omni target edge inference (Jetson AGX, embedded power/thermal budgets), or is it cloud-only? If edge is in scope, memory planning, KV sizing, and quantization backends need embedded constraints.

---

## 8. References

- [vLLM-Omni RFC #1987: World Model Support](https://github.com/vllm-project/vllm-omni/issues/1987)
- [PR #2162: DreamZero integration](https://github.com/vllm-project/vllm-omni/pull/2162)
- [Issue #3656: Sana-WM](https://github.com/vllm-project/vllm-omni/issues/3656) | [#3554: Robotics Integrations](https://github.com/vllm-project/vllm-omni/issues/3554) | [#3435: Deterministic Rollout for RL](https://github.com/vllm-project/vllm-omni/issues/3435)
- [Issue #3632: Streaming Diffusion Video](https://github.com/vllm-project/vllm-omni/issues/3632) | [#3553: GR00T Integration](https://github.com/vllm-project/vllm-omni/issues/3553) | [#2073: LingBot-World](https://github.com/vllm-project/vllm-omni/issues/2073)
- [DreamZero: World Action Models are Zero-shot Policies](https://dreamzero0.github.io/)
- [Thinking Machines Lab: Interaction Models](https://thinkingmachines.ai/blog/interaction-models/)
- [NVIDIA WAM / Jim Fan Sequoia AI Ascent 2026](https://www.humanoidsdaily.com/news/the-great-parallel-nvidia-s-jim-fan-outlines-the-robotics-end-game-strategy)
- [StreamDiffusionV2: Pipeline-parallel Stream-Batch](https://arxiv.org/pdf/2511.07399)
