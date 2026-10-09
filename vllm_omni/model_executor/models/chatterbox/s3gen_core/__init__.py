# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""S3Gen inference modules vendored from resemble-ai/chatterbox.

Source: https://github.com/resemble-ai/chatterbox at commit
5de7a54aa4e5e2baadb0182dde554908b48b85c2 (package version 0.1.7), MIT, with
CosyVoice, ESPnet, Matcha-TTS and 3D-Speaker files under Apache-2.0. Import
paths were rewritten and training-only code removed (plus a demo block, a
duplicate logger line and a missing `import logging`); the code then follows
this repository's conventions (absolute imports, lint, Google-style
docstrings, plain torch operations in place of einops, and the statements
after the unconditional raise in ``ConditionalCFM.forward`` dropped) without
changing any computation. Code this checkpoint does not reach was otherwise
kept as upstream has it: ``BASECFM.forward``, which both subclasses
override, the raise-only ``ConditionalCFM.forward`` itself, and the
registry entries in ``class_utils`` the encoder's configuration does not
select. Numeric parity against the source package, not textual closeness,
is what is checked.
"""
