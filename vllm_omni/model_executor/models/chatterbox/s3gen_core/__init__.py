# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""S3Gen inference modules vendored from resemble-ai/chatterbox.

Source: https://github.com/resemble-ai/chatterbox at commit
5de7a54aa4e5e2baadb0182dde554908b48b85c2 (package version 0.1.7), MIT, with
CosyVoice, ESPnet, Matcha-TTS and 3D-Speaker files under Apache-2.0. Import
paths were rewritten and training-only code removed (plus a demo block, a
duplicate logger line and a missing `import logging`); the code then follows
this repository's conventions (absolute imports, lint, Google-style
docstrings and plain torch operations in place of einops). Unused flow
configuration fields and overridden base flow methods were removed. The
reference ``CausalMaskedDiffWithXvec.inference`` path remains for numerical
comparison tests. Registry entries in ``class_utils`` that this checkpoint
does not select are retained from upstream. Numeric parity against the source
package, not textual closeness, is what is checked.
"""
