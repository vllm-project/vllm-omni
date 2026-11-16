---
name: 3d-animation-short-generator
description: Develop stylized 3D narrative shorts through story, character and environment references, timed shot tables, storyboards, generation, and assembly. Use for coherent animated storytelling or its planning stages, not isolated image edits or photorealistic live action.
metadata:
  compatibility: File-based agent workflow with sibling h3-prompt-writing. Media generation and editing require available tools or a configured backend.
---

# 3D Animation Short Generator

Read the [portable workflow](../h3-prompt-writing/references/portable-workflow.md).
For a complete film, progress from story to references, shot table, text storyboard,
clips, and assembly. Keep these as local artifacts, using existing user decisions
and authorization. If only a planning stage is requested, deliver that stage.

## Story and visual contract

Capture premise, desired emotional outcome, duration, ratio, model, and audio mode.
Use warm stylized 3D as a default only when style is unspecified: readable geometric
silhouettes, designed hair clumps, tactile materials, soft skin shading where
appropriate, expressive brows/eyes, and elastic posing. User-selected 2D or other
styles override this preset rather than receiving accidental 3D texture.

Create an active protagonist with a want, obstacle, meaningful choice, and payoff.
Connect the ending to an earlier visual/emotional anchor. Use as many beats as the
duration needs; do not pad a tiny story into an eight-beat formula. Dialogue should
change a relationship or move the action, and use the user's requested language.

## Reference design

- Character references lock name, silhouette, proportions, face, hair, costume,
  signature props, and color. Label identity in the production document; clean
  rendering anchors should not carry labels that could leak into the video.
- Scene references contain environments only unless the user requested a composite.
  Record landmarks, lighting direction/time of day, entrances, exits, and prop state.
- A reference board with multiple views is for design review. Prefer appropriate
  clean views for model input so grids and pose sheets do not become final frames.
- When an asset changes, update all affected prompts and clips, with a manifest
  identifying the current revision.

## Shot planning

Read [shot-table-spec.md](references/shot-table-spec.md) for the six-column table
and continuity checks. Then read
[storyboard-guidelines.md](references/storyboard-guidelines.md) for the matching
text storyboard and optional pencil previews. The text storyboard is the source
of shot timing; preview art does not silently replace it.

Vary shot size and elastic performance to serve the story. Camera tilts, close-ups,
or exaggeration are choices for specific beats, not quotas. Track off-screen
characters and spatial landmarks so they do not teleport back into later shots.

## Generation and assembly

Before rendering, read [model-selection.md](references/model-selection.md) and
[fallback-policy.md](references/fallback-policy.md). Use the chosen model through
available tools. Compile prompts using the H3 base or reference guide as appropriate.
Remove storyboard-only labels, arrows, panel borders, and sketch styling from
rendering inputs. Tie each clip to the current scene and character references.

Read [qc-checklist.md](references/qc-checklist.md) for assembly and final review.
Keep the planned shot order and audio continuity. Use a continuous score when
requested, duck it under dialogue/SFX, and do not add subtitles unasked. Deliver
the actual film or the requested planning package, with reproducible prompts and
any unresolved identity, scene, or timing defects.
