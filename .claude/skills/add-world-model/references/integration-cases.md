# World Model Integration Cases

Sources inspected on 2026-10-26: PR descriptions, relevant discussions, and
selected implementation files. The Hy-Nomas case also uses the supplied
integration conversation. Reported model results belong to these historical
revisions; they were not rerun while writing this skill.

## Source index

| Case | PR | Inspected head SHA | Relevant decisions |
| --- | --- | --- | --- |
| Hy-Nomas | [#9338](https://github.com/vllm-project/vllm-omni/pull/9338) | `257856d3afce694803f83fd60674f292b7a34583` | Modular loading, adapter isolation, parity, continuation, SP/offload. |
| Initial LingBot-World 2.0 draft | [#5022](https://github.com/vllm-project/vllm-omni/pull/5022) | `be7770f011b05ccd400335870c8ca7f9e8bd35ef` | Checkpoint, camera, condition, CPU versus device qualification. |
| Landed LingBot-World 2.0 integration | [#5491](https://github.com/vllm-project/vllm-omni/pull/5491) | `2ea66d1e4b8cc78845ee5f9aefe775a0da14d8d4` | Typed sessions, runner-owned KV, commit/failure, independent oracle. |
| LingBot stepwise | [#6844](https://github.com/vllm-project/vllm-omni/pull/6844) | `fbd2e1099ec52a1dacbdab02960988c6f9c8387c` | One request/many chunks, shared math, decoder frame-count failure. |
| LingBot long-rollout state | [#6838](https://github.com/vllm-project/vllm-omni/pull/6838) | `981dec06f5fb825bb884fdbaf585a1fe52a4924b` | Causal condition history, temporal RoPE, independent capacity limits. |
| ABot-World | [#6392](https://github.com/vllm-project/vllm-omni/pull/6392) | `b8438927806e9e4d24e93e58064d15382594f6c4` | Native discovery, shared runtime, VAE state, canonical-quality limits. |
| TaoMate-H3 streaming | [#8268](https://github.com/vllm-project/vllm-omni/pull/8268) | `609479c6d589350b49a6209cae73b6323bc0c554` | Segmented KV, LoRA reuse, teacher/student, hold/abort. |
| TaoMate-H3 full request and audio | [#8676](https://github.com/vllm-project/vllm-omni/pull/8676) | `79e3896a81a2dfbe1aeb415cdef87d1f35f9ca92` | Complete A/V, codec buffering, content and timestamp contracts. |
| TaoMate-H3 performance | [#8919](https://github.com/vllm-project/vllm-omni/pull/8919) | `9c8bd6e2648cb2e544ea57bd454adbc001d3cb53` | Clean timing, replay proof, async cleanup, revision-scoped evidence. |

Use the SHAs to recover historical behavior. Check current source before
treating any case as the implementation pattern for a new integration.

## Hy-Nomas: component ownership and execution modes

PR #9338 uses the BF16 distilled `tencent/Hy-Nomas-Diffusers` checkpoint,
component JSON, shards, and existing modular discovery. Its pinned Hy-Nomas
package still supplies model computation, text handling, and Euler scheduling.
This is an external-adapter case; native ports follow the sibling diffusion
skill's oracle-only reference contract.

The integration moved checkpoint configuration and loading into the model
package, with public request fields in model extras. A permissive training
loader could miss camera weights, and some constructed submodules did not
consume factory device/dtype arguments. The final
[component construction and mapper](https://github.com/vllm-project/vllm-omni/blob/257856d3afce694803f83fd60674f292b7a34583/vllm_omni/diffusion/models/hy_nomas/hy_nomas_transformer.py)
checks duplicate, unexpected, and missing weights and normalizes placement.

Reference SDPA/distributed globals require worker isolation; SP reuses the
worker group. Ordinary generation, typed continuation, and realtime stepwise
output use distinct execution profiles. The three-turn run emitted 12/49/86
stable frames, totaling 147 after close: decoder lookahead spans turns.

The final performance baseline shifted by 16.91% relative to the opening run,
so the reported SP speedup remained preliminary. Match cache-clear and timing
lifetimes, recheck baselines, and report GPU allocated/reserved per rank,
separately from host PSS. See [#9338](https://github.com/vllm-project/vllm-omni/pull/9338).

## LingBot-World: independent quality and temporal boundaries

PR #5022 is an early checkpoint/camera/condition draft; #5491 is the landed
integration. Typed metadata is validated before committing chunk/event state.
The shared runner owns physical KV and session failure cleanup.

In #5491, direct/paged internal equality did not establish parity with the
unmodified official full-video run. Matching noise draw order reduced later
AR divergence in a diagnostic control, while a residual remained. Keep the
unmodified oracle and modified-noise diagnosis separate; see
[#5491's reference comparisons](https://github.com/vllm-project/vllm-omni/pull/5491).

PR #6844 shares DMD math between tick and stepwise paths, with clean-KV commit
at post-decode. Its historical stateless chunk decode returned 63 frames for
latents that decode to 81 frames in one shot. This demonstrates the need for
session-owned VAE history, rather than a current-version bug claim. See
[#6844](https://github.com/vllm-project/vllm-omni/pull/6844).

PR #6838 advances causal condition-encoder history and extends only the
current block's temporal RoPE values. Its long RoPE-crossing experiment also
raised the shared KV cap in an unmerged diagnostic patch. Qualify each bound
separately; that experiment does not prove unlimited default rollouts. See
[#6838](https://github.com/vllm-project/vllm-omni/pull/6838).

## ABot-World: native loading and qualification limits

PR #6392 reuses typed ticks, consumer, session, runner, and paged KV while
keeping native `ti2v` discovery, UMT5/Wan conversion, action conditioning, and
decoder selection model-local. A native checkpoint needs no synthetic index.

Review requested per-session VAE convolution-cache memory accounting and
multiple-live-session residency checks, plus continuous cadence and internal
offline/session comparisons beyond the original three-tick smoke. Sources:
[VAE admission](https://github.com/vllm-project/vllm-omni/pull/6392#discussion_r3949980655),
[long-session cadence](https://github.com/vllm-project/vllm-omni/pull/6392#discussion_r3889532589),
[internal parity](https://github.com/vllm-project/vllm-omni/pull/6392#discussion_r3889679297).
Choose a new model's test length from its own state boundaries.

Full Wan and approximate TAEW2.2 decoding have separate quality contracts.
At the inspected head, non-eager execution warns but continues through a path
associated with quality failures; inspect the
[configuration and generation code](https://github.com/vllm-project/vllm-omni/blob/b8438927806e9e4d24e93e58064d15382594f6c4/vllm_omni/diffusion/models/abot_world/pipeline_abot_world.py).
A future integration should enforce known correctness constraints.

The PR explicitly retains historical 35-tick serving-consistency evidence
separately from current smoke. Its official reference run was blocked by a
missing FlashAttention API, leaving no paired canonical PSNR/SSIM. Internal
finite RGB and serving consistency do not qualify official weight/DMD/action
equivalence. See [#6392's qualification limit](https://github.com/vllm-project/vllm-omni/pull/6392).

## TaoMate-H3: shared facilities and complete A/V

PR #8268 delegates clean-KV storage and scheduling to AR-Diffusion. Native
low-rank factors are validated and normalized for the shared LoRA manager;
teacher/student policy, packed layout, history, and decoding remain in the
[model pipeline](https://github.com/vllm-project/vllm-omni/blob/609479c6d589350b49a6209cae73b6323bc0c554/vllm_omni/diffusion/models/taomate_h3/pipeline.py).
Verify alpha/r, projection order, separate banks, captured addresses, and
exception-safe switching rather than copying another model's QKV remap.

Hold uses scheduler waiting and cancellation retires held sessions. Validation
covers prompt lengths, audio resets, replay, switching, and recovery. This
PR's MP4 transport is video-only despite model audio generation; see
[#8268](https://github.com/vllm-project/vllm-omni/pull/8268).

PR #8676 adds ordinary full requests and audio delivery. Real-codec regressions
exposed short-fragment buffering, AAC priming/leading content, empty-binary
metadata correspondence, rate changes, and multi-video ambiguity. Its current
CPU/codec evidence is separate from earlier GPU evidence; see
[#8676](https://github.com/vllm-project/vllm-omni/pull/8676).

PR #8919 keeps clean-KV commit eager, proves actual teacher/VAE graph replay,
propagates noise-prefetch Future exceptions, and closes its executor per
session. Instrumented dispatch is separate from clean interleaved AB/BA
timing through complete MP4 reception. Its performance results are tied to an
earlier measured parent/candidate, not the inspected review-fix head; see
[#8919](https://github.com/vllm-project/vllm-omni/pull/8919).
