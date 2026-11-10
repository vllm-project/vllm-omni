# Portable H3 production workflow

This is the runtime contract for the nine H3 skills in this repository. Read it
alongside the selected creative skill. The bundle uses local files and available
tools; no proprietary canvas, choice-card widget, planner, or executor is required.

## Scope and decisions

- Resolve the requested deliverable first: prompt, storyboard, assets, or finished
  video. A prompt-only request ends with the prompt; it does not start generation.
- Reuse the user's duration, language, style, model, audio policy, supplied assets,
  and existing authorization. State reasonable defaults for optional gaps. Ask
  only about missing inputs or choices that materially affect the requested work.
- Creative checkpoints are reviewable artifacts, not mandatory repeated approvals.
  Honor an explicit request to approve a preview before rendering. When the user
  has authorized the complete workflow or delegated creative choices, show the
  plan and continue within that scope without requiring a special UI control.
- A skill never grants permission to publish, send messages, spend outside an
  authorized budget, or change models against the user's choice. User constraints
  override a preset's style, language, shot count, timing, or audio defaults.

## Tools and durable artifacts

1. Inspect the actual available tools, existing scripts, endpoint documentation,
   and supplied media. Do not invent a tool name, model parameter, or returned path.
2. Write plans and prompts to the user-selected project folder. If none is given,
   use a new named folder under `outputs/`, outside the skill package. Keep useful
   artifacts such as `brief.md`, `storyboard.md`, `prompts/`, `references/`,
   `clips/`, and `final/`; create only those needed for this deliverable.
3. Record each reference's role and path: identity, environment, style, typography,
   first/last frame, or audio. Tie generated clips to the prompt revision, input
   assets, model, settings, and output paths. Update downstream references when an
   asset changes; retain previous outputs instead of overwriting user files.
4. Use available image generation/editing tools for requested raster assets. Use
   an available video endpoint or local service for clips, and available audio
   tooling for requested standalone speech/music. For assembly and inspection,
   use an installed editor or tools such as FFmpeg, ffprobe, and OpenCV. Follow the
   host's tool policies, including its preferred image generation mechanism.
5. Independent tool jobs may run concurrently when resources permit; this skill
   does not require separate agents or delegated generation.
6. If a required capability is missing, finish the useful in-scope preparation:
   complete prompts, asset-role manifest, timeline, and verified request example
   where possible. State the missing capability and unfinished media step. Do not
   silently substitute another model, claim a render happened, or call a written
   plan a completed video. Resume generation when the capability becomes available.

## Compile the creative plan into H3 prompts

Read [base-format.md](base-format.md) for T2VA / I2VA / FL2VA / L2VA, or
[ref-format.md](ref-format.md) for Ref2VA. These are writing modes, not guaranteed API
task names. Check the selected backend's input mapping before sending a request.

- Base prompts use `integrated_multimodal_description`, `overall_soundscape`,
  `non_diegetic_music`, in that order, with the appropriate keyframe preamble.
- Ref2VA uses `subject_definitions`, `summary`, `retention_analysis`,
  `detailed_description`, `overall_soundscape`, `non_diegetic_music`, in that order.
- Every reference label must resolve to a supplied asset, in the same order used
  by the backend. Omit nonexistent optional references and their dependencies.
- Keep the creative brief in the user's language. Structured H3 rendering prompts
  use English prose and exact field labels, while dialogue, lyrics, and visible
  text retain their requested language and wording. If the user explicitly wants
  only a same-language creative prompt, deliver that without silently replacing it
  with English. Compile a separate rendering prompt when generation is requested.
- `[Shot 1]` has no initial timestamp. Later shot labels mark actual cuts. For a
  continuous morph, describe timed actions inside one shot rather than turning
  each form into a new shot. Express camera direction, speed, and useful amplitude.
- Separate image style from character identity. A style reference must not inject
  its people, scenery, layout, typography, texture, or lighting into unrelated
  shots. For literal text fidelity, inspect the rendered result; a prompt is not
  proof that the exact lettering or beat synchronization was achieved.

## Longer videos and audio

Use the actual backend's supported duration, dimensions, frame grid, reference
count, audio modes, and model partition. Inspect this checkout's H3 implementation
or serving documentation before building requests. Do not assume that a feature
branch's long-video extension exists on main or on a hosted service.

For a verified vLLM-Omni Ref2VA continuation backend, `long_video=true` with
`long_video_mode=continuation` can accept one `continuation_prompts` entry per
planned window only when the Ref2VA request is executed with its local text
encoder. Step execution and stages that use an external text encoder reject
`continuation_prompts`; omit the field and use the supported per-step inputs in
those configurations. Compute the window count from the aligned total frames,
window size, and overlap. Each prompt uses local window time; track global
shot/music time separately. Repeated overlap is guidance, not additional output
duration. Do not use this mode on a task/partition that does not support it.

Otherwise split into supported clips and assemble them. Same-scene clips may use
tail/head keyframe continuity when supported. Scene changes should have explicit
new scene references; a previous scene's latent tail can preserve the wrong
setting. Choose cuts, morphing, and transitions from the brief rather than imposing
hard cuts whenever a work exceeds 15 seconds. Generation windows and visual shots
are different: one window may contain cuts, and one shot may span multiple windows.

For supplied music, use the selected excerpt as the master timeline unless the
user requests replacement. For native audio, write the musical/SFX direction in
the prompt and verify that the backend actually returns audio. Do not invent a
generic `generate_audio` parameter. If available, locked-source audio latents guide
generation but the audio-VAE reconstruction is not bit-identical to the source.
Use explicit final audio stream-copy when an exact original track is required.
Prompted BPM is a target, not measured timing. Avoid automatically adding lyrics,
voiceover, subtitles, or multiple competing scores to an instrumental request.

## Verification and bounded recovery

Inspect reference images before using them. After rendering, inspect representative
frames, transitions, opening/ending states, text, identities, and scene continuity.
Check audio by listening when possible; report any limit on audio review. Check
duration, dimensions, fps, frame count, audio presence, and decode validity with
media tools. If preserving audio exactly, compare extracted audio stream hashes.

Correct a demonstrated failure with a targeted change to the affected prompt,
reference, or edit. Keep raw generation alongside edited/upscaled versions and
label the difference. Use an agreed retry/resource budget; absent one, allow one
targeted rerender, then report persistent defects and concrete options rather than
entering an unbounded generation loop. Do not switch providers automatically.

Deliver actual artifact paths, the prompts/settings needed to repeat the work,
and material visual/audio limitations. Distinguish requested behavior, observed
model output, and post-production repairs.
