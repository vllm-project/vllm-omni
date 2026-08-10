# Omni-Diffusion Third-Party Sources

This directory contains the small portions of external projects that are
required to run Omni-Diffusion side components. They are vendored because the
original projects do not expose these implementations as stable, installable
Python APIs. Model checkpoints are not included here; vLLM-Omni resolves them
from the configured local directories or their official Hugging Face
repositories at runtime.

## MagVITv2

- **Directory:** `magvit/`
- **Source:** [Gen-Verse/MMaDA](https://github.com/Gen-Verse/MMaDA)
- **Source revision:**
  [`b384b4d5339d250f3f3f65e0e6ee024ecbbad08d`](https://github.com/Gen-Verse/MMaDA/commit/b384b4d5339d250f3f3f65e0e6ee024ecbbad08d)
- **License:** MIT; see [`magvit/LICENSE`](magvit/LICENSE).

The MagVIT implementation was introduced upstream in revision
`da8b33b60928f6f28b5ccf2261873e9f7126e524` and is unchanged at the source
revision recorded above, which also contains the upstream license. The
vendored files are based on MMaDA's `models/common_modules.py`,
`models/modeling_magvitv2.py`, and `models/modeling_utils.py`.

MMaDA's `common_modules.py` is itself derived from
[CompVis/taming-transformers](https://github.com/CompVis/taming-transformers),
and `modeling_utils.py` retains its original Hugging Face and NVIDIA
Apache-2.0 notices.

vLLM-Omni modifications are limited to:

- relocating the files under the vLLM-Omni package and adjusting imports;
- applying project formatting, typing, and lint conventions;
- using the installed Diffusers `ConfigMixin` and `register_to_config`;
- matching decoded codebook tensors to the decoder parameter dtype.

The MagVITv2 checkpoint is downloaded separately from `showlab/magvitv2` by
default.

## GLM-4-Voice Decoder Runtime

- **Directory:** `glm4voice/`
- **Source:** [THUDM/GLM-4-Voice](https://github.com/THUDM/GLM-4-Voice)
- **Source revision:**
  [`54d667262bc1b92a62211aca44c7fcfe290e5df9`](https://github.com/THUDM/GLM-4-Voice/commit/54d667262bc1b92a62211aca44c7fcfe290e5df9)
- **License:** Apache-2.0. The complete license text is available in the
  repository root [`LICENSE`](../../../../../LICENSE).
- **Upstream copyright:** Copyright 2024 GLM-4-Voice Model Team @ Zhipu AI.

Only the token-to-waveform runtime used by Omni-Diffusion is included:
`flow_inference.py`, the required CosyVoice flow and HiFT modules, and the
required Matcha-TTS modules. The Matcha-TTS code corresponds to GLM-4-Voice's
submodule revision
[`dd9105b34bf2be2230f4aa1e4769fb586a3c824e`](https://github.com/shivammehta25/Matcha-TTS/commit/dd9105b34bf2be2230f4aa1e4769fb586a3c824e)
and retains its MIT license in
[`glm4voice/matcha/LICENSE`](glm4voice/matcha/LICENSE).

vLLM-Omni modifications are limited to:

- relocating the runtime under the vLLM-Omni package and registering aliases
  for the official module names referenced by the decoder YAML;
- excluding GLM-4-Voice's unrelated language model from HyperPyYAML object
  construction while preserving the upstream flow and HiFT decoder;
- applying compatibility and style changes needed by the supported
  vLLM-Omni dependency set.

The GLM-4-Voice decoder checkpoint is downloaded separately from
`THUDM/glm-4-voice-decoder` by default.
