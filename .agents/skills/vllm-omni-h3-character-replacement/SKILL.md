---
name: vllm-omni-h3-character-replacement
description: Plan MiniMax-H3 character replacement, write per-segment Ref2VA prompts and configure reusable H3-04 ComfyUI workflows for generation and review.
---

# vLLM-Omni H3 Character Replacement

Make character replacement easy to use: the user supplies source media and a replacement request; the agent derives the segment plan, prompts and workflow parameters. Keep the input boundary as original images, source video and explicit prompts. Work only on the requested deliverable; planning or prompt edits do not start generation, restart services or introduce preprocessing.

For a simple one-person edit without segmented continuation, reuse the [H3-04 Character Replacement workflow](<../../../apps/ComfyUI-vLLM-Omni/example_workflows/vLLM-Omni MiniMax-H3 Character Replacement.json>). For planned multi-segment work, use the Agentic Continuation template linked below; its example-specific settings must be recalculated for new footage.

1. Read [video-segment.md](video-segment.md) first. Use a vision-capable model to inspect the source video, establish shot boundaries and track when each selected person appears, interacts, becomes occluded and leaves. Produce the segment plan before writing new segment prompts.
2. Read [prompt-writing.md](prompt-writing.md), then write each segment's prompt from its inspected action and explicit source-person-to-picture mappings.
3. For a single-person, single-segment edit, configure the simple workflow linked above. For segmented or multi-person work, configure a task-specific copy of the [H3-04 Agentic Continuation workflow](<../../../apps/ComfyUI-vLLM-Omni/example_workflows/vLLM-Omni MiniMax-H3 Character Replacement Continuation (Agentic).json>), reusing Global Control, existing nodes and one shared segment-subgraph definition. Follow [workflow.md](workflow.md) for configuration, execution and review.
4. When generation is requested, submit the executable graph through ComfyUI, monitor results, inspect segment previews and deliver the assembled video with remaining quality defects.

Prompt-format dependency: read the official [MiniMax H3 prompt-writing skill](https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills/h3-prompt-writing). It can be read directly from the repository; local installation is optional. Before execution, establish the ComfyUI URL, installed node availability, effective Omni deployment settings and task output directory as described in [workflow.md](workflow.md).

Resolve repository-relative references from this skill directory after resolving any global skill symlink, not from the agent's current working directory.

Execution boundary: agent → ComfyUI execution API → existing vLLM-Omni nodes → vLLM-Omni video API. The nodes own model requests; the agent does not bypass ComfyUI with a separate generation client.

For a prompt-only correction to an existing segment, inspect its relevant footage and existing plan; revise the plan only if the observed defect exposes an incorrect boundary or mapping. Do not invent a new segmentation for every wording edit.

Preserve the user's model, resolution, FPS and fixed seed unless asked to change them. Report written plans and prompts separately from generation and visual acceptance. Keep Markdown paragraphs on single source lines and usage notes brief.
