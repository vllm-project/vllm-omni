# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""S3Gen inference modules vendored from resemble-ai/chatterbox.

Source: https://github.com/resemble-ai/chatterbox at commit
5de7a54aa4e5e2baadb0182dde554908b48b85c2 (package version 0.1.7), MIT, with
CosyVoice, ESPnet and Matcha-TTS files under Apache-2.0. Import
paths were rewritten and training-only code removed (plus a demo block, a
duplicate logger line and a missing `import logging`); the code then follows
this repository's conventions (absolute imports, lint, Google-style
docstrings and plain torch operations in place of einops). Unused flow
configuration fields and overridden base flow methods were removed, and so
was everything the checkpoint never reaches: the layer registries and the
variants they would select, the non-causal and multi-level decoder blocks,
and upstream's reference ``inference`` path. The U-Net's transformer block
and timestep embedding are diffusers' own, which Matcha-TTS had copied, and
the reference x-vector model (3D-Speaker's CAMPPlus) is IndexTTS2's,
imported by ``conditioning``.
Numeric parity against the source package, not textual closeness, is what
is checked.
"""
