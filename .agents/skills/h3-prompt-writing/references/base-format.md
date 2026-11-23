# H3 base-mode prompt format

Use this reference for T2VA, I2VA, FL2VA, and L2VA prompt writing. It is a
compact, repository-authored description of the required output contract, not a
copy of an upstream prompt guide.

## Select the anchor mode

- T2VA has no image anchor. Build the audiovisual sequence from the brief.
- I2VA binds `<Picture 1>` to the first frame at 0.00 seconds.
- FL2VA binds `<Picture 1>` to the first frame and `<Picture 2>` to the final
  frame. Describe the observable path between them.
- L2VA binds `<Picture 1>` to the final frame. Infer a compatible opening and
  converge to the supplied image.

For an image-anchored mode, put a plain-English alignment statement before the
structured fields. State each picture label, its shot, and its exact target time.
The ending time must match the effective clip duration, formatted to two decimal
places. Do not invent a picture label when no image was supplied.

## Emit three fields in order

```text
integrated_multimodal_description: [Shot 1] ...

overall_soundscape: ...

non_diegetic_music: ...
```

`integrated_multimodal_description` is the playback-ordered visual and diegetic
audio plan. Establish the initial style, framing, subjects, environment, and
spatial relationships, then describe actions and state changes. For keyframe
modes, explicitly preserve or approach the relevant composition, identity,
clothing, colors, objects, and geometry.

`overall_soundscape` summarizes ambience, physical effects, and non-verbal human
sound. Do not repeat dialogue or music already described elsewhere. Use `N/A`
only for explicitly silent output.

`non_diegetic_music` describes music heard by the audience but not the characters.
Specify audible properties such as instrumentation, tempo, rhythm, dynamics, and
ending behavior. Put source-visible music, singing, radio, or instrument playing
in the main description. Use `N/A` when no score is wanted.

## Timeline notation

- Start the opening with `[Shot 1]` and no timestamp.
- Mark each later cut with a sequential label and a strictly increasing time,
  for example `[Shot 2] At 00:03.500, ...`.
- Use a new shot for an actual edit or meaningful viewpoint/time change. Keep a
  continuous action or morph in one shot and describe its internal timing.
- Describe camera motion as an action: type, direction, useful amplitude, and
  speed. Do not append a disconnected list of camera keywords.
- Keep every cut and event within the requested clip duration.

## Speech, text, and sound

Give each speaking or singing source a stable ID such as `(S1)` or `(S2)` and
reuse it across shots. Put only the language tag and literal words inside the
dialogue wrapper:

```text
The courier (S1) says softly: <d>[English] The gate is open.</d>
```

Preserve supplied dialogue, lyrics, punctuation, and visible text verbatim. State
when a voice is off-screen and keep the visible character's lips closed when that
is required. If a line spans a cut, describe the audio continuity on both sides
of the cut. Place visible signs, labels, and interface copy in double quotes.

Tie diegetic sounds to the event that produces them. Avoid describing the same
sound as both diegetic and non-diegetic.

## Anchor-specific checks

- I2VA: begin from the supplied frame rather than merely borrowing its style.
- FL2VA: describe intermediate motion; do not provide two unrelated still-image
  descriptions. Reach Picture 2 at the actual end time.
- L2VA: keep the final picture assigned to the last shot, not automatically to
  Shot 1, and make the preceding action physically plausible.
- All modes: reference labels must map to supplied assets and remain stable.

Before delivery, verify the field order, shot numbering, timestamps, reference
labels, dialogue fidelity, requested language, duration, and absence of unresolved
template variables.
