# Video Segment

Use this guide when dividing source video into character-replacement clips or revising a continuation workflow. Derive the plan from the actual video and current backend; the existing Case03 is a scene-specific example, not a general-purpose five-segment recipe.

## Visual inspection comes first

Use a vision-capable model to examine the source footage before choosing segments. Inspect the reference images too, then distinguish their target appearances from the source people. If direct video viewing is unavailable, extract timestamped frames or contact sheets with a local media tool and inspect the images visually. Metadata and scene-change scores help navigation; they do not establish person identity or action continuity.

Survey the full video, not only the first frame. Inspect more densely around shot changes, entrances, exits, crossings, heavy occlusion and interactions involving multiple people. Play or inspect consecutive frames around a proposed boundary; sparse thumbnails alone can miss a person entering briefly or exchanging positions. Record uncertain observations and inspect additional frames instead of guessing. If visual inspection is unavailable, report the blocked planning step rather than presenting a visually verified plan.

Create two linked records before selecting boundaries:

- Shot timeline: source time ranges, actual cuts, camera/composition changes and the action within each shot.
- Person visibility timeline: stable source-track IDs, distinguishing visual cues, first appearance, visible intervals, occlusion, re-entry, departure and interactions with other tracks. Track IDs remain stable across position changes and across cuts only when identity is supported by the footage; record ambiguity explicitly.

Inspect duration, FPS and frame count. For variable-frame-rate media, use timestamps or explicitly normalize the working clip; do not assume frame index equals time multiplied by nominal FPS. Reference numbering follows actual input order; source-track IDs are independent of reference numbering and screen position.

## Choose segments for multiple replacements

Choose boundaries using the shot timeline and the changing set of selected people. Prefer the fewest segments that preserve coherent action, distinguishable identities and backend duration limits. Equal-duration slicing and splitting at every entrance or exit are not defaults.

| Observed source event | Planning decision |
|---|---|
| Several selected people remain identifiable during one continuous action | Keep them in one segment when supported; give each a separate mapping. |
| A selected person enters later | Include enough source context to identify the entrant. Use an entry-focused segment or a later anchor if needed; do not require every target in the first frame. |
| Selected people leave and the scene continues with fewer people | Consider a boundary after the departure stabilizes. The next segment maps the people visible anywhere within that segment; departed identities do not migrate to the remaining person. |
| People cross, overlap or exchange props | Keep the interaction and identity cues together where possible. Avoid a boundary at maximal occlusion or during contact; if a split is unavoidable, carry context on both sides and inspect the join. |
| A cut changes composition within the same scene | Check person identity on both sides. Split if the new composition needs a different mapping or the backend cannot retain it reliably; otherwise preserve the existing cut in the segment. |
| A cut introduces a new scene or a discontinuous cast | Prefer an independent segment and anchor; avoid imposing the previous scene through continuation guides. |
| A person is temporarily occluded and later reappears | Keep the same track if supported by the footage. Occlusion alone does not define a new person or require a split. |

For each candidate segment, list every selected person visible at any point in the range, their local appearance/exit times and the reason for the chosen boundary. A later entrant still needs a reference and mapping. Keep unselected people unchanged. If the selected cast exceeds the running service's reference limit, revise the plan or report the limitation; do not silently omit a target or assume separate edits can be composited without affecting other people.

An identity anchor should make the assigned people identifiable. A later multi-person view may be a better anchor than an opening where people have not entered. Generate outward from the chosen anchor when the workflow supports it. Backward continuation is an optional strategy: deliberately reverse the source, choose guides in that direction and reverse the generated frames back for assembly. Keep generation dependency order separate from chronological assembly order. A later anchor is not proof that reversal will improve quality.

## Produce a concrete plan

Record one row per segment:

| Segment | Source range / shot | Visible selected tracks → pictures | Boundary reason | Direction / dependency | Guide source/range | Generated duration/frames | Keep range | Assembly order |
|---|---|---|---|---|---|---|---|---|

Use half-open ranges `[start, end)` in the plan; translate them to each node/API's actual inclusive or length-based convention. Distinguish source frames, generated frames, previous-output guide indices and retained assembly frames. Identify the coordinate system for every frame number.

- Compute source duration from the selected source timestamps. For confirmed constant FPS, frame count divided by FPS also gives duration.
- Calculate required output frames from duration and output FPS, then apply the running model's frame-alignment and minimum-duration constraints. The current `_align_frame_count` helper aligns positive counts to `5 + 17n`; it does not impose a 107-frame minimum or choose FPS. Treat the example's 107-frame minimum and 22-frame guide window (positions 0 and 21) as configuration to verify, not universal model constraints.
- Mark overlap frames as guidance and retain the intended source action once. Derive trim/keep ranges from output FPS and the chosen overlap; do not copy frame constants from another clip.
- Select guide indices from the actual preceding frame batch, distinguishing raw generated frames from already trimmed assembly frames. Verify indices stay in bounds and correspond to the intended poses.
- Account for generated alignment tails explicitly. FPS changes require recalculating frame-based durations, overlaps, guide indices and trims; exposing a shared FPS widget alone does not perform that conversion.
- Keep the plan's user controls in seconds or source ranges where possible. Derive internal frame arithmetic rather than asking users to guess it.

Case03's B/A/C/D/E order, reverse generation, five-person mappings and 24-FPS frame settings belong to its supplied 27-second scene. New footage needs a new plan; do not reuse its cigarette-lighting, pointing or departure actions as defaults.

## Write and connect segments

Only after the visual timelines and segment plan are complete, write each segment's prompt using [prompt-writing.md](prompt-writing.md). Include segment-local entrance/exit timing only when supported by inspection; map it from global source time and account for reversed generation where applicable. Define only references connected for that segment and maintain the same real-person assignment across segments even if local reference indices change. Preserve the same reference appearance through continuation; guide frames supply inherited pose and composition.

## Handoff to execution

The plan is ready when source ranges cover the requested edit, each selected person has a supported reference mapping, guide indices are in bounds, and trim/keep ranges remove overlap exactly once. Record frame coordinate systems and separate generation dependencies from assembly order. Follow [workflow.md](workflow.md) to prepare inputs, configure/export the ComfyUI graph, execute and review results.
