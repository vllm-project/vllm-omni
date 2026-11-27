# Prompt Writing

Produce an explicit appearance-to-source-person mapping. Keep the user's images, source video and prompt as the input boundary. Prompt-only work ends with the prompt or requested workflow edit; it does not start generation or restart a service.

Prompt-format dependency: read the official [MiniMax H3 prompt-writing skill](https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills/h3-prompt-writing) and follow its Ref2VA instructions and referenced files. Local installation is optional; read the repository files directly when the skill is not installed. The upstream reference is currently `references/ref-en.txt`; `ref-format.md` belongs to the local adaptation, not the upstream skill. Use English for the rendering prompt and the user's language for the explanation.

## Planning prerequisite

For new footage or a new multi-segment workflow, complete the vision-based shot/person timelines and segment plan in [video-segment.md](video-segment.md) before writing prompts. Use each planned range's actual visible people and actions, including later entrants. For an existing segment correction, inspect its footage and preserve the plan unless a boundary or mapping needs revision.

## Establish the mapping

Inspect the supplied images, source video and actual reference connections before assigning people. If the media cannot be inspected, state that limitation and use only the user's explicit mapping; ask when the selected source person is ambiguous.

- Number `<Picture N>` by actual request input order, not canvas position or filename.
- Give each target appearance and each selected source-person track separate `<Subject N>` definitions. For two replacements, targets 1/2 and source tracks 3/4 are sufficient. Define only supplied references and selected source people.
- Identify source people with visible features, roles and actions. Use initial position as an aid, then preserve the same person's track as they move. Avoid celebrity or fictional-character names as identity cues.
- Write an explicit one-to-one mapping. A replacement count means selected source people, not the total people visible in the scene. Preserve unselected people.
- Replace example-specific descriptions when the source changes. “Central performer wearing a pale suit” and “right foreground person” describe one example, not a universal person-selection rule. First-appearance order alone is often ambiguous.

## Compile the prompt

Keep the six sections in order:

1. `subject_definitions`: targets from pictures, source-person tracks from video, and the source video's role.
2. `summary`: edit the source video, state the replacement count and each mapping.
3. `retention_analysis`: target identity/appearance `fully_preserved`; source tracks `attribute_transfer` for motion, pose, expression and visibility; video `partially_preserved` for scene, camera, interactions and timing.
4. `detailed_description`: `[Shot 1]` followed by playback-ordered actions and identity constraints. Describe real source cuts only when known.
5. `overall_soundscape`: retain or reference the requested source sound properties.
6. `non_diegetic_music`: follow the user's music policy.

Use a compact shared constraint paragraph rather than repeating the same rule in all sections:

> Each target appearance belongs to exactly one assigned source-person track. Keep each person's own face, hair, clothing, materials and worn accessories throughout turns, crossings and occlusion. Preserve both the presence and absence of these features in the assigned picture. When a person leaves the view, their replacement leaves with them; on re-entry, restore the same assigned identity. Every unassigned person retains their original appearance. Render only people visible in the source at each moment. Pictures supply appearance; the source video supplies actions, handheld props, scene, camera, lighting and timing.

Adapt this paragraph to the observed issue. Avoid accumulating redundant negative instructions or naming an unwanted accessory that is absent from the inputs. Preserve source interactions and natural movement rather than freezing a reference pose. Exact soundtrack copying requires actual audio handling, not prompt wording.

## Single and multiple people

Single-person mapping skeleton (replace the bracketed description before delivery):

```text
<Subject 1>: The target person in <Picture 1>, defining their appearance.
<Subject 2>: [Unambiguous visible description of the selected source person] in <Video 1>.
Mapping: replace <Subject 2> with <Subject 1>; preserve all other people.
```

For two people, define two target appearances and two source tracks, then state `<Subject 3> → <Subject 1>` and `<Subject 4> → <Subject 2>`. Scale to the actual reference count and backend limit, not a fixed eight- or ten-subject template. Keep faces, outfits and accessories separate even during close interaction. Multi-view images depict one target person; they do not add people to the scene.

## Segment prompts

For segmented work, use the plan from [video-segment.md](video-segment.md). Inspect each segment's actual inputs independently: define only its supplied references and selected source tracks. Appearance references define identity; continuation guides establish inherited composition and pose. State what remains visible and what exits when that affects identity.

## Diagnose and verify

- **Mixed or duplicated identity:** check image order and effective request inputs first; clarify the two source tracks and exclusive mappings before adding prose.
- **Accessory leakage:** distinguish target appearance from source motion and props; preserve the assigned reference's features, including their absence. A bad frame suggests leakage but does not by itself establish its cause.
- **Late identity drift:** inspect departures, occlusion, cuts and continuation boundaries; bind identity to the person rather than their current screen position.
- **Identity retained in occluded close-ups:** if generated close-ups retain the source identity under hair, hand or lighting occlusion, the segment-level mapping may need shot-specific appearance cues. In `detailed_description`, name the assigned target at that shot and state which picture features must be visible there (the target's own hair length and color, face, glasses or their absence), e.g. "the extreme close-up shows <Subject 1> with the short light-brown hair and face from <Picture 1>". Keep it a positive description of the assigned appearance; do not enumerate the source features being removed.
- **Scope:** background removal, face anonymization and automatic appearance description are separate features. Do not introduce them to repair a prompt unless the user requests them.

Check all labels resolve, mappings are one-to-one, and no placeholders or unused references remain. When editing a workflow, keep positional and named prompt values consistent where both exist; preserve unrelated fields and user layout. Usage Guide notes should only say which media, mappings and controls to change.

For an authorized comparison, keep inputs, model, seed, LoRA and sampling fixed while changing the prompt. Inspect both identities together, occlusion and the reported failure interval; inspect frame-by-frame around the defect when needed. Report prompt edits separately from observed generation results. Health checks, valid JSON and successful decoding do not establish character-replacement quality.
