# Text storyboard and optional visualization

Maintain one `storyboard.md` with a section per shot. Each section contains:

1. Shot ID, title, duration, and narrative purpose.
2. Exact scene and character reference IDs.
3. Spatial anchors: landmarks, character positions/facing, off-screen characters,
   key/fill/rim lighting, and important props.
4. Incoming and outgoing continuity state.
5. Time-ordered beats: pose/expression, camera, audio, and anchor changes.

Use per-second or finer intervals only where meaningful; cover the complete shot.
For narrated moments, specify whether a visible character speaks or only reacts.
Optional ASCII staging diagrams can clarify screen direction but are not images
to supply as rendering references.

For heavy revision of a shot, extract its section to a named file and link it from
the main document. Choose one authoritative version; reintegrate or update the
link after revision so later generation cannot read stale content.

Only generate pencil storyboards when requested or already authorized as previews.
Use equal-sized time-ordered panels, with timecode, pose/expression, camera arrow,
and audio/anchor notes. Suggested layouts: three panels in a strip, four in a 2×2
grid, six in 2×3. More complex scenes may need fewer representative key poses.
Keep shot/character/scene labels readable for human review, but never feed these
annotated grids as clean final frames. The text timeline remains authoritative.

If preview panels fail, simplify annotation or rely on text within the agreed
scope. An optional visualization failure need not block a usable text storyboard.
