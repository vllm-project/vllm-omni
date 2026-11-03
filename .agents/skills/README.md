# Portable MiniMax H3 Skills

This directory contains one general prompt-writing skill and eight creative
skills adapted from
[MiniMax-AI/MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills).
See [H3-SOURCES.md](H3-SOURCES.md) for the source revision, attribution, and
adaptation scope.

## Skill catalog

| Skill | Intended use |
| --- | --- |
| [h3-prompt-writing](h3-prompt-writing/SKILL.md) | Structured prompts for T2VA, I2VA, FL2VA, L2VA, and Ref2VA |
| [minimalist-product-ad-generator](minimalist-product-ad-generator/SKILL.md) | Minimal product ads, isolated product references, copy, and pacing |
| [3d-animation-short-generator](3d-animation-short-generator/SKILL.md) | 3D narrative shorts, character and environment design, shot tables, storyboards, and editing |
| [papercraft-stop-motion-explainer](papercraft-stop-motion-explainer/SKILL.md) | Layered paper art, puppets, mechanisms, and miniature educational scenes |
| [brand-promo-video-generator](brand-promo-video-generator/SKILL.md) | Brand promos, verified assets and claims, product interaction, and calls to action |
| [music-video-subtitle-generator](music-video-subtitle-generator/SKILL.md) | Music videos, instrumental visual narratives, beat planning, and optional lyric typography |
| [co-op-game-intro-generator](co-op-game-intro-generator/SKILL.md) | Two-player game menus, confirmation images, and animated world entry |
| [paper-collage-explainer-generator](paper-collage-explainer-generator/SKILL.md) | Halftone photo collage, incremental paper assembly, and visual metaphors |
| [handdrawn-live-video-generator](handdrawn-live-video-generator/SKILL.md) | Live action with rough luminous drawings, physical contact, continuous morphing, and delayed pursuit |

## Usage

Codex discovers repository skills from `.agents/skills/`. Clients that scan
skills only when a session starts may need a new session. Claude Code reads the
same files through the same-named relative symlinks in `.claude/skills/`. Other
agents that support Markdown skills can load the relevant `SKILL.md` directly
and follow its links to supporting references.

Examples:

```text
Use $music-video-subtitle-generator to design a 30-second two-dimensional
fluid-motion graphic for this instrumental track. Do not add lyrics or
subtitles. Preserve leftward motion and initially return only a storyboard and
three H3 prompts.
```

```text
Use $co-op-game-intro-generator to create a two-player game intro. The players
are Lin and Mei, the style is layered paper sculpture, and the title is
Together. Create the confirmation image first and wait for my approval before
generating video.
```

## MV smoke test

Test `music-video-subtitle-generator` together with `h3-prompt-writing` using a
user-owned song with timestamped lyrics and an original adult-character
reference image:

```text
Use $music-video-subtitle-generator and $h3-prompt-writing to turn this song
into a historical-romance music video about a woman and a man. Intercut
narrative acting with the woman's singing performance, do not display lyric
text, split the work according to the backend's verified clip duration, and
preserve the complete original song in the final master.
```

Verify that the global timeline matches the source audio duration, lyrics remain
verbatim, and each generation window records its incoming state, outgoing state,
and local timing. Ref2VA prompts must preserve the six-section field order and
resolve every reference label to a real input. The assembled video must decode
completely, and its final audio must come from the supplied source rather than a
model reconstruction.

When moving the collection to another repository or a user-level skill folder,
copy all nine skill directories and preserve their sibling layout. The eight
creative skills share `h3-prompt-writing/references/portable-workflow.md` and the
H3 format guides, so copying only one `SKILL.md` is insufficient. Keep each
skill's `references/` and `agents/` directories with it. `agents/openai.yaml`
provides optional Codex UI metadata and does not restrict the workflow to an
OpenAI-based agent or require a dedicated plugin.

## Execution boundaries

Prompt writing, storyboarding, and planning require only an agent that can read
and write files. Actual image, video, speech, music, and editing work uses tools
or services available in the current environment. When a capability is missing,
deliver the complete usable production package and identify the unfinished media
step.

The user's style, language, assets, audio, model selection, and existing
authorization take priority. An MV template does not add lyrics to an
instrumental track, and a clean motion-graphics request does not inherit film
grain. A prompt-only request does not start generation. When the user requests
approval of a confirmation image before rendering, preserve that checkpoint.

Treat vLLM-Omni long-video extensions as capabilities of the checked-out branch
and active backend. Do not present a development branch's continuation parameters
as a universal H3 API. Preserve real requests and raw outputs, distinguish model
results from post-production and upscaling, and inspect actual media properties.

These skills contain no model weights and do not automatically start a service
or invoke a paid generation endpoint.
