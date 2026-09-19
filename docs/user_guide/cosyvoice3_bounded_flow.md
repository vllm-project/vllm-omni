# Experimental bounded CosyVoice3 flow

Set `flow_context_tokens` in the model configuration to a positive token count
to solve only the new streaming suffix, conditioned on a bounded tail of
previously emitted tokens and generated mel. The default is zero and preserves
cumulative flow inference. The initial voice prompt is also limited to this
window.

This changes long-range conditioning; it is not numerically equivalent to
cumulative inference. The caller must preserve the returned per-request cache
between chunks. Finalization releases that cache. HiFT retains its existing
incremental streaming state.

The initial implementation processes a bounded-flow batch serially. It does
not claim cross-request batching throughput improvements. Official flow/HiFT
checkpoints pass a synthetic-token streaming probe, and regression tests cover
offsets, lookahead, finalization and request isolation. Real utterance WER,
speaker similarity and perceptual quality remain unvalidated. Keep this
configuration disabled in production until those checks are complete.
