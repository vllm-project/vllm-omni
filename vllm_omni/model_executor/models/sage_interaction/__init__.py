# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""InternLM Sage-Interaction full-duplex model package.

Sage-Interaction shares the MiniCPM-o 4.5 three-stage weight layout
(Thinker + Talker + Token2Wav) and extends the native duplex protocol with:

- ``<|audio_start|>`` / ``<|audio_end|>`` audio-unit markers inside ``<unit>``;
- 3-way per-unit control actions (``<|listen|>``, ``<|speak|>``, ``<|delegate|>``);
- asynchronous backend delegation via ``<|im_start|>tool\\n...<|im_end|>``
  results injected before the next ``<unit>`` once the foreground turn is idle.
"""
