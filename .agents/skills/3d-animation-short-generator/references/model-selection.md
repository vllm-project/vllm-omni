# Model and format selection

Honor an existing model choice. Otherwise prefer MiniMax H3 when available, and
inspect the running backend for partitions, reference modes, audio, resolutions,
duration limits, and long-video options. Do not assume hosted and local services
have identical limits, quality, prices, or parameter names.

If another model is requested, use its real documented input contract. A provider
switch needs the user's authorization; never silently use a different service
because a preset names it as a fallback. For an authorized mixed-model workflow,
record model and settings per shot and check cross-shot style differences.

Keep delivery resolution separate from native generation resolution. Record actual
width/height, fps, frame count, and any crop, retiming, or upscaling. Do not claim
native 2K when the video was generated at 768p and restored later.

Compile shot action, camera, identity, scene, and audio into the chosen model's
prompt format. For H3 use the bundled writing guides. Strip production labels and
preview grids. Prompt for designed silhouettes, elastic action, material and
lighting details when these belong to the user's 3D style; do not add text/UI just
because a model can generate them.
