---
name: co-op-game-intro-generator
description: Design or generate a two-player co-op game menu and opening animation from player names, title, style, and optional identity references. Use for character-led game intros, not playable game implementation or complex multi-page interfaces.
metadata:
  compatibility: File-based workflow with sibling h3-prompt-writing and bundled image/video templates. Rendering uses available generation tools.
---

# Co-op Game Intro Generator

Read the [portable workflow](../h3-prompt-writing/references/portable-workflow.md).
Before writing a confirmation-image prompt, read
[h3-confirmation-image-template.md](references/h3-confirmation-image-template.md).
Before writing the video prompt, read
[h3-video-prompt-template.md](references/h3-video-prompt-template.md).
Both templates are required package resources, not remote tools.

## Resolve player and style bindings

Reuse or collect the two player names, game title, visual style, desired duration,
ratio, and optional identity references. The template defaults to a 15-second,
16:9 menu-to-world sequence; adapt it to explicit user requirements. Do not assume
the players' gender, body shape, outfits, or equipment from a generic example.

Use each portrait only for recognizable identity anchors: face silhouette, hair,
glasses, proportions, and distinguishing traits. Reinterpret rendering, skin
texture, lighting, and costume in the selected visual style. If a portrait or body
comparison reference is absent, write a text design rather than inventing an asset
label. Maintain player-to-name mapping across all references and frames.

## Confirmation image

Fill the template's style, palette, characters, title, and name variables. Keep
the two characters central, player cards toward the upper left, menu on the right,
and the continue action as the main interaction focus. The title establishes
identity without competing with that focus. Buttons, icons, typography, and all
player/equipment panels share one palette and style. Keep menu labels on one line.

The style changes material, shapes, fonts, and decoration; it should not randomly
rearrange the menu hierarchy. User-requested layout or ratio changes take priority
over template defaults. Preserve the exact supplied title/names and requested copy.

For image-only/prompt-only work, stop at that deliverable. For video production,
generate and inspect the confirmation image if within scope. Honor a request to
approve the image first; otherwise use the established end-to-end authorization
and continue after reporting the selected image and fixing material defects.

## Video event plan

Fill the video template from the current image and player bindings. Its baseline
sequence is menu → first player's light equipment → second player's heavy equipment
→ shared confirmation → world loading → both enter the world. Equipment, world,
body contrast, and timing are editable design choices; preserve requested identity
and style instead of forcing a cyberpunk example onto another setting.

Bind UI reference, each actual portrait, and any real body-comparison image to
distinct roles. Compile this creative template into the H3 mode guide's fields;
the mixed-language template is not itself an API request. Preserve literal visible
UI text in its required language, and remove all unfilled variables before dispatch.
Do not treat a composition reference as an exact first-frame anchor unless the
selected backend actually supports that role.

## Review and repair

Check both names, exact menu strings, layout, player identity, side assignment,
equipment distinction, and the transition into the requested world. Match UI
hover/click, mechanical locks, loading, and footsteps to the visible events.
For identity swaps, strengthen name/position/reference bindings; for weak style,
revise palette, material, character, and typography together. Reduce excessive
UI text if necessary without silently changing required player names.

Deliver the requested prompts/assets or the inspected intro video, with reference
roles and any text/timing limitations. Missing templates block template execution;
missing generation capability still permits a complete prompt handoff.
