# H3 full-reference prompt format

Use this reference for Ref2VA prompt writing. It is a compact,
repository-authored description of the output contract, not a copy of an
upstream prompt guide.

## Emit six fields in order

```text
subject_definitions: ...

summary: ...

retention_analysis: ...

detailed_description: [Shot 1] ...

overall_soundscape: ...

non_diegetic_music: ...
```

Write the structural prose in English. Preserve the requested language and exact
wording for dialogue, lyrics, and visible text.

## Reference labels

Assign labels only to supplied content and keep every label stable across all six
fields:

- `<Subject N>` identifies reusable visible content such as a person, object,
  environment, costume, interface, action, or visual treatment.
- `<Picture N>` identifies an image used as a concrete frame, composition, or
  storyboard anchor. If an image only defines a subject, cite it in that subject's
  definition instead of inventing a separate target-frame role.
- `<Video N>` identifies a source video being edited or continued, or a video
  whose temporal structure is referenced.
- `<Audio N>` identifies a supplied audio signal that is copied or referenced.
  Audio and video label numbers are independent even when they came from one file.

In `subject_definitions`, give each tracked item a separate line. State its source,
role, and the features that later sections must preserve or transform. Do not
create labels for assets or roles that do not exist.

## Summary and retention

`summary` is one short paragraph describing the target and the principal reference
relationships. Use only labels already defined. Distinguish these operations:

- a concrete image frame anchor;
- generation guided by identity, style, action, camera, or audio characteristics;
- direct editing of a source video;
- continuation from a source video;
- copying an audio signal;
- referencing audio properties without copying the signal.

`retention_analysis` has one line per label and states where it is used and what
changes. Use consistent machine-readable relationship values:

- Visual: `fully_preserved`, `partially_preserved`, `attribute_transfer`, or
  `weak_reference`.
- Audio: `fully_copy`, `partially_copy`, `reference`, or `weak_reference`.

Explain the concrete retained and changed traits after the relationship value.
Do not claim full preservation when identity, geometry, timing, or signal content
is intentionally altered. Do not count newly requested target action as an
unintended loss of reference fidelity.

## Detailed description

`detailed_description` is the playback-ordered rendering plan. For each shot,
cover the current composition, subject appearance and position, environment,
lighting, actions and state changes, camera behavior, diegetic sound, and the
point at which each reference becomes relevant.

- Start with `[Shot 1]` and no timestamp.
- Give later cuts sequential labels with strictly increasing times, such as
  `[Shot 2] At 00:03.500, ...`.
- Place relevant reference labels next to the described subject, frame, motion,
  or sound; do not collect detached labels at the end.
- Use stable `(S1)`, `(S2)`, and subsequent IDs for vocal sources. Put literal
  speech or lyrics inside `<d>[Language] ...</d>`.
- Preserve supplied visible text verbatim and place it in double quotes.
- Describe continuous transformations inside one shot unless an actual cut is
  intended. State camera direction, speed, and amplitude when useful.

For video editing, explicitly identify what stays from the source timeline and
what is replaced. For continuation, distinguish the inherited opening state from
new content. For style reference, transfer visual properties without accidentally
copying people, scenery, layout, or typography.

## Audio fields

`overall_soundscape` summarizes ambience, physical action sounds, and non-verbal
human sounds. `non_diegetic_music` describes score audible only to the audience.
Keep dialogue and diegetic music in `detailed_description`. Use `N/A` only when
the corresponding layer is intentionally absent.

If audio is copied, state whether the whole signal or only a range/layer is used.
If it is referenced, describe the specific property—such as voice timbre, tempo,
rhythm, or texture—without claiming signal reuse.

Before delivery, verify the six-field order, one-to-one label definitions,
retention coverage, shot timing, dialogue fidelity, requested language, and
absence of unresolved template variables.
