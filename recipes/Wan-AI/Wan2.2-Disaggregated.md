# Wan2.2 encode / generation / decode disaggregation

These opt-in configurations extend the existing `Wan22Pipeline` T2V and
image-conditioned TI2V path. They do not add disaggregation to the separate
`WanImageToVideoPipeline`, S2V or VACE implementations.

Select `vllm_omni/deploy/wan2_2_eg.yaml` for two stages, or
`vllm_omni/deploy/wan2_2_egd.yaml` for three stages, using `--deploy-config`.
Adjust device placement and connector settings for the deployment. Both configs
use NIXL; their device defaults are local examples, not a validated cross-node
deployment recipe. The existing single-stage topology remains available.

| Mode | E | G | D |
| --- | --- | --- | --- |
| EG | UMT5 + optional TI2V VAE encoder | DiT + VAE decoder | Fused with G |
| EGD | UMT5 + optional TI2V VAE encoder | DiT only | VAE decoder |

Pure T2V E does not load a VAE. TI2V E retains its encoder and `quant_conv`;
GD/D retain the decoder and `post_quant_conv`. The checkpoint loader initially
loads both VAE halves, then removes unused components before device placement.
This reduces resident weights, not peak host loading memory. FULL retains both
halves. Wan's high/low-noise transformer scheduling stays within G.

## Conditioning contract

E emits per-request `prompt_embeds` and CFG `negative_prompt_embeds`. For TI2V
models it additionally emits:

- `wan_image_condition`: normalized float32 BCTHW first-frame latents, B=T=1,
  before replication for `num_outputs_per_prompt`; absent without an image.
- `wan_conditioning_metadata`: version 1, normalization `wan_latents_mean_std`,
  layout `BCTHW`, effective height/width/frame count, spatial/temporal scale,
  and `has_image`. Metadata remains mandatory for TI2V's no-image path.

Image encoding uses posterior mode and does not advance request RNGs. G checks
the contract before preparing noise, repeats per-request conditioning for output
fan-out, and reconstructs the first-frame mask. G never falls back to an absent
text/VAE encoder. FULL uses the same image normalization helper while preserving
its noise-before-image-encode ordering.

Declared payload keys use the existing generic stage connector. Successful
transfers replace inline payloads with a handle; failed sends retain the inline
copy. The generic handoff drops original image data after E and forwards only
the current producer's payload. G -> D carries denoised `latents`, not encoder
conditioning. D emits typed video media; non-owner VAE ranks retain empty legacy
output placeholders. Explicit latent output bypasses decoding.

The scheduler groups compatible requests. Stage entry validation also rejects
incompatible manually constructed batches rather than applying request 0's
dimensions, CFG or generation settings silently.

## Validation scope

CPU helper tests cover a real tiny Diffusers Wan VAE, FULL-vs-E conditioning,
fan-out, batching, RNG preservation and malformed contracts. Runtime tests cover
stage ownership, CFG, dummy inputs, typed media and non-owner results. These are
not pretrained-model numerical equivalence or performance measurements.

Before production use, run fixed-seed single-stage/EG/EGD comparisons with the
target checkpoints and validate NIXL across the intended ranks/nodes, offload,
VAE tiling and decoder ownership. Do not infer support for other Wan variants
from these tests.
