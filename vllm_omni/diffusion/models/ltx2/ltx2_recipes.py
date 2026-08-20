# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Declarative execution recipes for the LTX model family."""

from dataclasses import dataclass, replace
from typing import Literal

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

LTX25_DEFAULT_NEGATIVE_PROMPT = (
    "has_subtitles, has_blurbox, transition from black, transition to black, speech_ending_short, "
    + LTX_DEFAULT_NEGATIVE_PROMPT
)

LTX_DISTILLED_SIGMAS = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0)
LTX_STAGE_2_DISTILLED_SIGMAS = (0.909375, 0.725, 0.421875, 0.0)
LTX_DISTILLED_ADAPTER_SLOT = "ltx_distilled"


@dataclass(frozen=True)
class LTXPhaseRecipe:
    """One denoise phase and the transition used to construct its input."""

    name: str
    guidance: LTXGuidanceSpec
    spatial_downscale: int = 1
    sigmas: tuple[float, ...] | None = None
    noise_scale: float = 0.0
    input_transform: Literal["initial", "spatial_upsample"] = "initial"
    adapter_slot: str | None = None
    sampler: Literal["euler", "euler_ancestral"] = "euler"
    allow_guidance_override: bool = True
    use_official_sigma_schedule: bool = True

    def __post_init__(self) -> None:
        if self.spatial_downscale < 1:
            raise ValueError("LTX phase spatial_downscale must be positive.")
        if self.sigmas is not None and len(self.sigmas) < 2:
            raise ValueError("An explicit LTX sigma schedule must contain at least two denoise sigmas.")
        if self.sampler not in ("euler", "euler_ancestral"):
            raise ValueError(f"Unsupported LTX sampler: {self.sampler!r}.")

    @property
    def num_inference_steps(self) -> int | None:
        return None if self.sigmas is None else len(self.sigmas) - 1


@dataclass(frozen=True)
class LTXPipelineRecipe:
    """Request defaults, ordered phases, output routing, and request capabilities."""

    phases: tuple[LTXPhaseRecipe, ...]
    height: int = 512
    width: int = 768
    num_frames: int = 121
    frame_rate: float = 24.0
    num_inference_steps: int = 40
    negative_prompt: str = LTX_DEFAULT_NEGATIVE_PROMPT
    video_output_phase: int = -1
    audio_output_phase: int = -1
    allow_request_sigmas: bool = True
    allow_request_phase_sigmas: bool = False
    allow_request_latents: bool = True
    allow_negative_prompt: bool = True
    fixed_num_inference_steps: bool = False
    supports_cache_dit: bool = False

    def __post_init__(self) -> None:
        if not self.phases:
            raise ValueError("An LTX pipeline recipe must contain at least one phase.")
        if self.phases[0].input_transform != "initial":
            raise ValueError("The first LTX phase must use the initial request input.")
        phase_count = len(self.phases)
        for output_phase in (self.video_output_phase, self.audio_output_phase):
            if not -phase_count <= output_phase < phase_count:
                raise ValueError(f"LTX output phase {output_phase} is outside a {phase_count}-phase recipe.")

    @property
    def request_guidance(self) -> LTXGuidanceSpec:
        return self.phases[0].guidance

    @property
    def max_spatial_downscale(self) -> int:
        return max(phase.spatial_downscale for phase in self.phases)


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


LTX2_ONE_STAGE_RECIPE = LTXPipelineRecipe(
    supports_cache_dit=True,
    phases=(LTXPhaseRecipe(name="generate", guidance=_official_guidance(29)),),
)
LTX23_ONE_STAGE_RECIPE = LTXPipelineRecipe(
    supports_cache_dit=True,
    num_inference_steps=30,
    phases=(LTXPhaseRecipe(name="generate", guidance=_official_guidance(28)),),
)
LTX25_FULL_RECIPE = LTXPipelineRecipe(
    height=544,
    width=960,
    num_inference_steps=30,
    negative_prompt=LTX25_DEFAULT_NEGATIVE_PROMPT,
    phases=(
        LTXPhaseRecipe(
            name="generate",
            guidance=_official_guidance(28),
            noise_scale=1.0,
        ),
    ),
)
LTX_POSITIVE_ONLY_RECIPE = LTXPipelineRecipe(
    supports_cache_dit=True,
    phases=(
        LTXPhaseRecipe(
            name="generate",
            guidance=LTXGuidanceSpec.positive_only(),
            use_official_sigma_schedule=False,
        ),
    ),
)
LTX2_DISTILLED_TWO_STAGE_RECIPE = LTXPipelineRecipe(
    # Official distilled arguments describe the final output. Stage 1 applies
    # spatial_downscale=2 and therefore runs at 512x768.
    height=1024,
    width=1536,
    num_inference_steps=len(LTX_DISTILLED_SIGMAS) - 1,
    negative_prompt="",
    phases=(
        LTXPhaseRecipe(
            name="generate_lowres",
            guidance=LTXGuidanceSpec.positive_only(),
            spatial_downscale=2,
            sigmas=LTX_DISTILLED_SIGMAS,
            noise_scale=1.0,
            allow_guidance_override=False,
            use_official_sigma_schedule=False,
        ),
        LTXPhaseRecipe(
            name="refine",
            guidance=LTXGuidanceSpec.positive_only(),
            sigmas=LTX_STAGE_2_DISTILLED_SIGMAS,
            noise_scale=LTX_STAGE_2_DISTILLED_SIGMAS[0],
            input_transform="spatial_upsample",
            allow_guidance_override=False,
            use_official_sigma_schedule=False,
        ),
    ),
    video_output_phase=1,
    audio_output_phase=1,
    allow_request_sigmas=False,
    allow_request_latents=False,
    allow_negative_prompt=False,
    fixed_num_inference_steps=True,
)

LTX25_DISTILLED_TWO_STAGE_RECIPE = LTXPipelineRecipe(
    height=1088,
    width=1920,
    num_inference_steps=len(LTX_DISTILLED_SIGMAS) - 1,
    negative_prompt="",
    phases=(
        LTXPhaseRecipe(
            name="generate_lowres",
            guidance=LTXGuidanceSpec.positive_only(),
            spatial_downscale=2,
            sigmas=LTX_DISTILLED_SIGMAS,
            noise_scale=1.0,
            allow_guidance_override=False,
            use_official_sigma_schedule=False,
            sampler="euler_ancestral",
        ),
        LTXPhaseRecipe(
            name="refine",
            guidance=LTXGuidanceSpec.positive_only(),
            sigmas=LTX_STAGE_2_DISTILLED_SIGMAS,
            noise_scale=LTX_STAGE_2_DISTILLED_SIGMAS[0],
            input_transform="spatial_upsample",
            allow_guidance_override=False,
            use_official_sigma_schedule=False,
        ),
    ),
    video_output_phase=1,
    audio_output_phase=1,
    allow_request_sigmas=False,
    allow_request_phase_sigmas=True,
    allow_request_latents=False,
    allow_negative_prompt=False,
    fixed_num_inference_steps=True,
)

# LTX-2.3 full-distilled uses the same official fixed 8 + 3 sigma schedules
# and request contract. Its distinct component profile selects the 22B merged
# distilled Transformer, BWE vocoder, and matching spatial upsampler; no
# distilled LoRA is loaded for either phase.
LTX23_DISTILLED_TWO_STAGE_RECIPE = LTX2_DISTILLED_TWO_STAGE_RECIPE


def _distilled_one_stage_recipe(
    two_stage_recipe: LTXPipelineRecipe,
    *,
    supports_cache_dit: bool = True,
) -> LTXPipelineRecipe:
    """Run the merged-distilled checkpoint at its native Stage-1 resolution."""
    generate_phase = replace(
        two_stage_recipe.phases[0],
        name="generate",
        spatial_downscale=1,
    )
    return replace(
        two_stage_recipe,
        height=two_stage_recipe.height // 2,
        width=two_stage_recipe.width // 2,
        phases=(generate_phase,),
        video_output_phase=0,
        audio_output_phase=0,
        allow_request_phase_sigmas=False,
        supports_cache_dit=supports_cache_dit,
    )


LTX2_DISTILLED_ONE_STAGE_RECIPE = _distilled_one_stage_recipe(LTX2_DISTILLED_TWO_STAGE_RECIPE)
LTX23_DISTILLED_ONE_STAGE_RECIPE = _distilled_one_stage_recipe(LTX23_DISTILLED_TWO_STAGE_RECIPE)
LTX25_DISTILLED_ONE_STAGE_RECIPE = _distilled_one_stage_recipe(
    LTX25_DISTILLED_TWO_STAGE_RECIPE,
    supports_cache_dit=False,
)


def _official_two_stage_recipe(one_stage_recipe: LTXPipelineRecipe) -> LTXPipelineRecipe:
    return LTXPipelineRecipe(
        height=one_stage_recipe.height * 2,
        width=one_stage_recipe.width * 2,
        num_frames=one_stage_recipe.num_frames,
        frame_rate=one_stage_recipe.frame_rate,
        num_inference_steps=one_stage_recipe.num_inference_steps,
        negative_prompt=one_stage_recipe.negative_prompt,
        phases=(
            LTXPhaseRecipe(
                name="generate_lowres",
                guidance=one_stage_recipe.request_guidance,
                spatial_downscale=2,
                noise_scale=1.0,
            ),
            LTXPhaseRecipe(
                name="refine",
                guidance=LTXGuidanceSpec.positive_only(),
                sigmas=LTX_STAGE_2_DISTILLED_SIGMAS,
                noise_scale=LTX_STAGE_2_DISTILLED_SIGMAS[0],
                input_transform="spatial_upsample",
                adapter_slot=LTX_DISTILLED_ADAPTER_SLOT,
                allow_guidance_override=False,
                use_official_sigma_schedule=False,
            ),
        ),
        video_output_phase=1,
        # The official second stage refines video only and deliberately
        # discards its audio result. Decode the full-context Stage-1 audio.
        audio_output_phase=0,
        allow_request_sigmas=False,
        allow_request_phase_sigmas=True,
        allow_request_latents=False,
    )


LTX2_TWO_STAGE_RECIPE = _official_two_stage_recipe(LTX2_ONE_STAGE_RECIPE)
LTX23_TWO_STAGE_RECIPE = _official_two_stage_recipe(LTX23_ONE_STAGE_RECIPE)
LTX25_TWO_STAGE_RECIPE = _official_two_stage_recipe(LTX25_FULL_RECIPE)


_PIPELINE_RECIPES: dict[tuple[str, str], LTXPipelineRecipe] = {
    ("one_stage", "2"): LTX2_ONE_STAGE_RECIPE,
    ("one_stage", "2.3"): LTX23_ONE_STAGE_RECIPE,
    ("one_stage", "2.5"): LTX25_FULL_RECIPE,
    ("two_stage", "2"): LTX2_TWO_STAGE_RECIPE,
    ("two_stage", "2.3"): LTX23_TWO_STAGE_RECIPE,
    ("two_stage", "2.5"): LTX25_TWO_STAGE_RECIPE,
    ("distilled_one_stage", "2"): LTX2_DISTILLED_ONE_STAGE_RECIPE,
    ("distilled_one_stage", "2.3"): LTX23_DISTILLED_ONE_STAGE_RECIPE,
    ("distilled_one_stage", "2.5"): LTX25_DISTILLED_ONE_STAGE_RECIPE,
    ("distilled_two_stage", "2"): LTX2_DISTILLED_TWO_STAGE_RECIPE,
    ("distilled_two_stage", "2.3"): LTX23_DISTILLED_TWO_STAGE_RECIPE,
    ("distilled_two_stage", "2.5"): LTX25_DISTILLED_TWO_STAGE_RECIPE,
    ("dmd2", "2"): LTX_POSITIVE_ONLY_RECIPE,
    ("dmd2", "2.3"): LTX_POSITIVE_ONLY_RECIPE,
}


def resolve_ltx_pipeline_recipe(pipeline_kind: str, model_version: str) -> LTXPipelineRecipe:
    """Resolve execution independently from component loading."""
    try:
        return _PIPELINE_RECIPES[(pipeline_kind, model_version)]
    except KeyError as exc:
        raise ValueError(f"Unsupported LTX pipeline kind/version: {pipeline_kind!r}/{model_version!r}.") from exc
