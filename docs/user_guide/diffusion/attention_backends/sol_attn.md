# Sol-Attn on Ascend

Wan2.2-TI2V-5B can use MindIE-SD diagonal-threshold Sol-Attn for video
self-attention on a single Ascend 910B. Install a MindIE-SD wheel exporting
`mindiesd.sol_attn` from [MindIE-SD PR 929](https://gitcode.com/Ascend/MindIE-SD/pull/929)
(tested at `6985241966212ee36cc1f10a222e03e8154330e9`).
The integration uses unmasked BF16 attention with a head dimension of 128.

From the repository root, set `WAN_MODEL` to your local model path and `WAN_PROMPT`
to the prompt you want to compare. Run both commands on the same NPU.

```bash
wan=(python examples/offline_inference/text_to_video/text_to_video.py
     --model "$WAN_MODEL" --deploy-config vllm_omni/deploy/wan2_2_ti2v.yaml
     --prompt "$WAN_PROMPT" --seed 1024 --width 1280 --height 704 --num-frames 121 --fps 24
     --num-inference-steps 50 --guidance-scale 5 --flow-shift 5 --enforce-eager --vae-use-tiling)
"${wan[@]}" --output dense.mp4 --diffusion-attention-config '{"default":{"backend":"FLASH_ATTN"}}'
"${wan[@]}" --output diag.mp4 --diffusion-attention-config \
  '{"default":{"backend":"FLASH_ATTN"},"per_role":{"self":{"backend":"SOL_ATTN","sol_attn":{"tau":1.0,"dense_steps":4,"dense_layers":1}}}}'
```

For a two-step end-to-end smoke test, run
`pytest -q tests/e2e/offline_inference/test_wan22_t2v.py::test_ti2v_5b_sol_attn`.

The example shows the defaults. `tau` scales the diagonal routing threshold.
The first `dense_steps` denoising steps and the first `dense_layers` transformer
blocks use FlashAttention. On sparse steps Wan reorders video tokens and their
positional conditioning into Morton order around the block stack. Cross-attention
keeps its independently selected backend. Use eager inference with DiT caching
and denoise graph capture disabled. Approximate blocks use pooled keys and values,
so generated frames can differ from the dense baseline.

## Automatic fallback to full attention

Before the first sparse step, the single-NPU worker checks for a callable
`mindiesd.sol_attn`, a supported SoC (910B or 950 in MindIE-SD PR 929),
the Sol PyTorch operators with NPU dispatch, and the compiled objects referenced
by the CANN kernel manifests for that SoC in the custom OPP paths. A stale object
from a different build does not satisfy this check. If any capability is missing, the worker
logs one warning naming the reason and `FLASH_ATTN`. All Wan video
self-attention layers continue in full attention, in their original token
order, for the lifetime of that worker. Cross-attention keeps its own backend.
With a fully dense schedule there is no sparse step to probe.

Invalid configurations, driver errors and errors from executing a registered
kernel (including out-of-memory errors) propagate to the caller. On a worker
without the entire MindIE-SD package, unmasked BSND attention uses the native
NPU Flash kernel. To compare fallback output to explicit Flash on an NPU with
an older MindIE-SD wheel, run
`pytest -q tests/e2e/offline_inference/test_wan22_t2v.py::test_ti2v_5b_sol_fallback_matches_flash`.
