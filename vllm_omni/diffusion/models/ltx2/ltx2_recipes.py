# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""User-visible one-stage defaults for the LTX model family."""

from dataclasses import dataclass

from .ltx2_guidance import LTXGuidanceSpec, LTXModalityGuidance

LTX_DEFAULT_NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
    "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
    "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
    "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
    "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
    "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
    "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
    "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
    "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
    "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
    "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
)


@dataclass(frozen=True)
class LTXOneStageRecipe:
    height: int = 512
    width: int = 768
    num_frames: int = 121
    frame_rate: float = 24.0
    num_inference_steps: int = 40
    negative_prompt: str = LTX_DEFAULT_NEGATIVE_PROMPT
    guidance: LTXGuidanceSpec = LTXGuidanceSpec()
    use_official_sigma_schedule: bool = True


def _official_guidance(stg_block: int) -> LTXGuidanceSpec:
    return LTXGuidanceSpec(
        video=LTXModalityGuidance(
            cfg_scale=3.0,
            stg_scale=1.0,
            modality_scale=3.0,
            rescale_scale=0.7,
            stg_blocks=(stg_block,),
        ),
        audio=LTXModalityGuidance(
            cfg_scale=7.0,
            stg_scale=1.0,
            modality_scale=3.0,
            rescale_scale=0.7,
            stg_blocks=(stg_block,),
        ),
    )


LTX2_ONE_STAGE_RECIPE = LTXOneStageRecipe(guidance=_official_guidance(29))
LTX23_ONE_STAGE_RECIPE = LTXOneStageRecipe(
    num_inference_steps=30,
    guidance=_official_guidance(28),
)
LTX_POSITIVE_ONLY_RECIPE = LTXOneStageRecipe(
    guidance=LTXGuidanceSpec.positive_only(),
    use_official_sigma_schedule=False,
)
