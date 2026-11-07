# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""S3Gen's flow, vendored from resemble-ai/chatterbox.

Source: https://github.com/resemble-ai/chatterbox at commit
5de7a54aa4e5e2baadb0182dde554908b48b85c2 (package version 0.1.7), MIT, with
CosyVoice, ESPnet and Matcha-TTS files under Apache-2.0. Import paths were
rewritten and training-only code removed; the code follows this
repository's conventions (absolute imports, lint, Google-style docstrings
and plain torch operations in place of einops). Only what the checkpoint
reaches is kept: one encoder configuration, a causal single-level U-Net
and the Euler solvers, with no layer registries and no reference
``inference`` path.

Where the tree already carries a module upstream copied, it is imported:
the Conformer layer, feed-forward and relative-position attention from
IndexTTS2, with the checkpoint's LayerNorm epsilon and ESPnet's position
shift as two small subclasses here; the pre-lookahead and causal
convolutions and the padding mask from CosyVoice3; the U-Net's transformer
block and timestep embedding from diffusers. The reference x-vector model
(3D-Speaker's CAMPPlus) is IndexTTS2's, used by ``conditioning``. Numeric
parity against the source package, not textual closeness, is what is
checked.
"""
