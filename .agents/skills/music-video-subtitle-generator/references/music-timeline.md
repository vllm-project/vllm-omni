# Global music and local video timing

Use this planning skeleton, omitting vocal/text fields when not requested:

```text
Global: music/excerpt, duration, ratio, character/style/scene locks, cut policy.

Segment S01 — global start/end; local generation start/end
Incoming state:
Music cue: measured beat/phrase timestamp, or clearly labeled planned accent
Vocal line and speaker (if present):
Visible typography (if requested):
Visual action and camera:
Outgoing state and transition:
References with exact roles:
```

Keep one global music clock. For measured tempo B, the nominal quarter-note spacing
is 60/B seconds, but real tracks may have pickups, swing, tempo changes, or inaccurate
automatic detections. Validate important cut points against the waveform/listening.
An onset detector also finds SFX and melodic attacks; it does not prove drum beats.

Choose a phrase-complete excerpt when possible. Track breaths, lyric pauses, and
vowel continuity for a performance MV. For an instrumental morph, accent the
physical preparation, transformation impact, and release rather than introducing
unrequested cuts. The final shape or story payoff needs a readable ending.

For continuation, maintain a table of absolute output frames and local window
times. Use the verified backend's native alignment and overlap behavior. Opening
overlap continues the inherited state; new actions belong after it. Do not sum
overlap twice or describe every window as a fresh opening shot.

For separately generated clips, use a previous tail frame only when the next shot
continues the same scene and the backend supports that keyframe role. For an
intentional cut, retain appropriate identity references and select the new scene.
Preserve screen direction or a useful match element when it helps the transition.

When text is requested, track exact string, language, entry/exit time, screen
region, and motion. Typography style cards must not donate their depicted characters
or settings. Preserve user text; do not silently shorten lyrics to fit a template.
