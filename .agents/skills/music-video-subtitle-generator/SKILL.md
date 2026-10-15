---
name: music-video-subtitle-generator
description: Plan, write, audit, or produce music videos with coordinated music, imagery, camera motion, optional performance, and lyric typography. Use for beat-aware MVs and instrumental visual narratives, not ordinary subtitle cleanup or unrelated video edits.
metadata:
  compatibility: File-based workflow with sibling h3-prompt-writing; audio analysis, generation, and final assembly use available tools and backends.
---

# Music Video and Lyric Typography

Read the [portable workflow](../h3-prompt-writing/references/portable-workflow.md)
and, when building a timeline, [music timeline](references/music-timeline.md).
Deliver the requested scope: audit, prompt, storyboard, or complete video.

## Lock the musical and visual intent

Identify duration, ratio, output resolution, music source/excerpt, visual style,
character/scene references, performance mode, and whether text is wanted. Treat a
supplied song as the master track unless the user requests replacement. Preserve
provided lyrics exactly; do not translate or rewrite them unasked. For instrumental
work, do not invent singing, lyrics, a performer, or typography. Generate original
lyrics only when requested or clearly within an authorized songwriting brief.

Measure supplied audio duration and inspect/listen for beat, phrase, breath, drop,
and energy changes using available tools. Label estimated or planned timing when
measurement is unavailable. If the requested excerpt exceeds the audio, resolve
the discrepancy without silently stretching or looping it.

## Reference roles

- Character references control identity, silhouette, wardrobe, and presence.
- Scene references control environment, time of day, spatial anchors, and lighting.
- Typography references control letter treatment, layout, and motion, not people
  or scenery visible in those references.
- Music references control the requested track or sound style; distinguish copying
  a supplied recording from generating similar instrumentation.

Do not apply a preset's grain, darkness, hard cuts, or constant close-ups to an
incompatible brief. A bright nighttime scene remains night. A continuous flat
vector morph keeps smooth color fields and uses elastic shape motion for accents.
Use hard cuts, slow holds, or dissolves only when appropriate to the user's intent.

## Compose the timeline and H3 prompts

Map visual actions, camera moves, scene changes, and optional text to the music's
global time. Build a causal visual progression instead of defaulting to one person
singing throughout. Allow room to read a shape, gesture, or lyric before changing it.

For longer work, choose a supported continuation or clip-assembly approach through
the portable workflow. Keep visual shot boundaries distinct from generation
windows. Local window prompts must account for overlap without replaying the full
previous action. A new scene may require a new reference rather than inheritance
of the old scene. Record entry/exit states for each planned segment.

Compile the timeline into H3's base or Ref2VA fields, preserving dialogue, lyrics,
and visible text verbatim. Include concrete instrumentation and rhythmic accents,
but treat requested BPM and beat alignment as targets until checked in the output.

## Typography and finishing

When requested, give each shot one main readable text event. Keep words away from
eyes and critical lip motion. Match sung text to the active phrase, and distinguish
spatial typography from subtitle bars. Reduce text complexity if the model cannot
render it, or use a disclosed overlay consistent with the requested deliverable.

For supplied audio, assemble clips to one global master track, maintaining phrase
and lip continuity; avoid cuts inside a vowel. For native audio, inspect actual
continuity across generation windows. Preserve useful SFX and avoid layered duplicate
music. Retiming, grain, grading, interpolation, and upscaling are deliberate editing
choices, not automatic finishing requirements.

Verify transformation/story order, identities, scene changes, requested text,
musical continuity, timing, and full media decode. Label raw and processed versions.
Deliver the final file and current prompt/timeline files, with measured properties
and remaining limitations rather than claims of perfect synchronization.
