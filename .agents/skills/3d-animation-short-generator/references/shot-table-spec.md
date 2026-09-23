# Six-column shot table

| Shot ID & Duration | Continuity Handoff | Reference Anchors (Spatial + Identity) | Hook Type | Shot Description (Per-Second Directives) | Audio & Dialogue Track |
| --- | --- | --- | --- | --- | --- |

Use stable IDs such as S01. Record incoming and outgoing image state, prop
position, eyeline, motion, and emotional state in Continuity Handoff.

Reference Anchors contains named landmarks and screen-relative positions; each
character's screen position, facing and pose; recently exited characters' locations;
lighting baseline and modifiers; and exact character/scene asset paths or IDs.
An explicit scene/time change can replace these anchors; an accidental flip cannot.

Hook Type names the shot's purpose: setup, reveal, reversal, joke, suspense,
tenderness, chase, or callback. Do not force a joke or cut into a quiet scene.

For each second or meaningful sub-second beat, cover action/pose/expression,
camera, spatial/prop state, sound, and the handoff to the next interval. Use compact
phrases rather than inventing movement in intentionally still moments. Include
anticipation, squash/stretch, overshoot, and follow-through where appropriate.

Keep the complete dialogue/narration script in Audio & Dialogue Track, with
speaker, timing, delivery, and SFX. Distinguish off-screen narration from speaking
characters; a visible listener should not lip-sync the narrator's words.

Before rendering, check full time coverage, supported generation durations, a
readable number of active characters, matching scene/identity references, and
coherent row-to-row handoffs. Mark intentional cuts/time jumps. Fix contradictory
states rather than using a generic continuity assurance. A visual shot may span
multiple generation windows when the verified backend supports continuation.
