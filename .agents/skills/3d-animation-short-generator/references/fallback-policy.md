# Targeted generation recovery

Compare the failed result against the shot's actual identity/spatial anchors.
Classify the failure: wrong input or API mapping, reference drift, overloaded
action, timing, audio, or a render/export failure. Fix transport/configuration
errors before spending another generation attempt.

For drift, quote the relevant anchors and remove unrelated reference roles. For
overloaded action, simplify or split the shot while preserving the requested total
duration and handoffs. For a scene change, use the new scene reference and avoid
conditioning on a previous scene when that would contradict the new setting.

Use the retry/resource limit from the shared portable workflow. A persistent
failure should lead to a concrete choice (simpler action, a supplied reference,
an authorized alternate model, or an explicitly incomplete shot), not automatic
provider switching or an endless loop. Never present a placeholder as a finished
clip or hide an incorrect scene in final assembly.

For optional pencil previews, reduce annotation or panel count. Use the text
storyboard if visualization is not essential; do not rerender correct video
solely because a human-review sketch changed.
