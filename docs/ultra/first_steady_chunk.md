# MiniCPM-o 4.5 first and steady codec chunks

Status: configurable first-chunk state machine implemented with a
semantics-preserving default. Ascend A3 grid search, continuity checks,
official performance, and complete quality gates remain pending.

## Source and scope

This change manually ports the implementation boundary from upstream
[PR #5938](https://github.com/vllm-project/vllm-omni/pull/5938): only the
first audio-producing chunk may use a smaller codec threshold; steady chunks
continue to use `codec_chunk_frames`. Early `init_only` setup remains a control
packet and does not consume codec offset or `chunk_seq=0`.

Upstream's reported A3 result is hypothesis evidence, not a result for this
stack. It used English Seed-TTS and an explicit YAML value of 8; this
competition stack must be judged on the frozen Chinese c=1 workload and the
RTF-first promotion rule.

## Configuration precedence

1. Explicit connector `extra.initial_codec_chunk_frames`.
2. `VLLM_OMNI_MINICPMO45_INITIAL_CODEC_CHUNK_FRAMES` when the connector does
   not specify the key.
3. Model-level default 25 frames.

The effective first threshold is capped by the steady threshold. Environment
value `0` disables a candidate and restores the steady threshold. An explicit
connector value of zero is invalid and fails visibly. No deploy YAML edit is
required for the environment-driven A/B grid.

The default remains 25 because this machine has no A3 evidence proving that
8/12/16 satisfy the RTF non-inferiority and continuity gates. The experiment
grid is 8, 12, 16, and 25; a smaller automatic default may be promoted only
after the formal A3 gate passes.

## State invariants

- `state.codec_end == 0` selects the first threshold.
- The first real audio payload retains `chunk_seq=0` even when an `init_only`
  control payload was sent earlier.
- After the first real payload, every non-terminal emission returns to the
  steady threshold.
- Left context remains three codes and is derived from the actual prior chunk.
- Flush, abort, internal-request replacement, duplex epoch reset, and tail
  handling retain existing ownership and cleanup.
- The public API, SSE, codec values, Code2Wav math, CFM steps, and dtype do not
  change.

## Promotion gate

- Run 8/12/16/25 with clean restart and `B-C-C-B-B-C` on Chinese Seed-TTS c=1.
- Prefer RTF improvement; otherwise require RTF non-inferiority and TTFP at
  least 5% better. TTFT may not regress more than 1%.
- Check per-chunk RTF, first/steady Graph hit rate, audio gaps/overlaps,
  underrun, non-empty decoding, success, peak HBM, and c=4/8 guardrails.
- Complete Daily-Omni, Video-MME, ASV, and WER gates before merge.
