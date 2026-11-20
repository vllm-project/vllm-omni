# MiniMax H3 skill provenance

Upstream: https://github.com/MiniMax-AI/MiniMax-H3

Source snapshot: `d21241f0a4b3acbb34c97dae47fa417b7065e438`, retrieved 2026-09-19.

Source directory:
https://github.com/MiniMax-AI/MiniMax-H3/tree/d21241f0a4b3acbb34c97dae47fa417b7065e438/skills

The nine upstream skill names are retained. Each local skill corresponds to the
same-named upstream `skills/<name>/` directory. Credit for the original creative
workflows and templates belongs to MiniMax-AI and the upstream contributors.

## Files and adaptation

- `h3-prompt-writing/references/base-format.md` and `ref-format.md` are new,
  condensed descriptions of the public prompt interface. They preserve necessary
  field names and notation but do not copy the upstream guides or their examples.
  The entrypoint adds a portable execution reference and clarifies clip/window
  duration for verified long-video backends.
- The eight style-specific `SKILL.md` files are rewritten portable adaptations of
  the upstream workflows. Their reference documents retain the relevant creative
  methods while removing host-specific execution assumptions.
- The two `co-op-game-intro-generator/references/` templates retain the upstream
  creative structure with an added portable contract and explicit user overrides.
  Their instructions are normalized to English, and the video template expresses
  visible UI copy as localized variables with English defaults. It also removes
  an unrequested gender restriction and a fixed yellow/black environment
  transformation.
- The five 3D animation references are condensed adaptations of their upstream
  counterparts: shot table, storyboards, model selection, fallback, and review.
- Product, papercraft, collage, and music references extract reusable direction
  from their corresponding upstream skill. The shared portable workflow, catalog,
  and UI metadata are additions for this repository.

No upstream `LICENSE`, `COPYING`, or `NOTICE` file was present in the Git tree at
the pinned snapshot. Its README links to the MiniMax H3 Community License on the
model release. Because that license has redistribution conditions, this package
does not include the upstream prompt guides verbatim. This provenance record
preserves attribution and does not assert that this repository's code license
relicenses upstream material.

## Runtime changes

Canvas documents become local files and asset manifests; choice-card UI becomes
the host's normal clarification mechanism when a decision is actually missing.
Available generation/editing tools replace proprietary dispatch calls. Previous
user choices and authorization persist across steps. Providers, prices, resolution
support, and long-video features are checked rather than hardcoded as guarantees.

User intent overrides style presets, including instrumental/no-text work, selected
language, final morph shape, no-cut sequences, and custom player designs. Native
generation and external processing remain separately identified in delivery.
