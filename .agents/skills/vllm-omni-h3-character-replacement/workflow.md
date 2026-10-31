# Workflow

Execute one character-replacement task through ComfyUI. Use [video-segment.md](video-segment.md) for visual analysis and frame planning, and [prompt-writing.md](prompt-writing.md) for prompts. A plan-only or prompt-only request stops before submission.

## 1. Discover the environment

Resolve service addresses in order: explicit user/task configuration, the selected workflow and Global Control for the Omni endpoint, then the ComfyUI launch configuration or active application connection for the ComfyUI endpoint. Ask for unresolved required values instead of scanning arbitrary ports. Do not treat localhost addresses from another task as defaults. Resolve the ComfyUI base URL from task configuration and check `GET /system_stats`. Read `GET /object_info` for the nodes required by the selected template; guide and concatenation nodes are needed only when used. Read effective Global Control settings, including mode, service URL/model, LoRA configuration, resolution, FPS, steps and seed. Preserve the user's choices; server-local LoRA paths are not agent-local paths. Confirm the selected deployment supports the requested combination. For FastH3 + Swap, record the effective model and server revision, inspect whether the serialized request contains a nonzero LoRA, and check adapter activation evidence. A past 400 or later successful job alone cannot establish current support or prove that the adapter took effect; never assume it was silently ignored.

Create a task output directory. Keep agent-local files, ComfyUI input/output paths and Omni server paths distinct. If checking the Omni model listing, resolve `/v1/models` from the configured API base without appending `/v1` twice; generation still goes through ComfyUI nodes.

## 2. Reuse relevant history when available

Look for earlier runs of the same task. Check their inputs, settings and outputs before reuse; successful execution does not establish visual acceptance. Record observed defects as the next run's fix list. A missing history entry does not block a first run.

A previous API prompt can be a baseline when its topology matches the new plan. On versions exposing the standard history tuple, it is stored at `history[prompt_id]["prompt"][2]`; inspect the actual response before using that field. Retrieve reported output files with `GET /view`, passing their filename, subfolder and type. Avoid downloading or reviewing unrelated task outputs.

## 3. Inspect and plan

Probe source resolution, duration, frame count and nominal/average FPS. Start with timestamped contact sheets (for example, 1 FPS), then inspect more densely around entrances, occlusion, interactions and cuts. Scene-change detection can suggest candidates; thresholds such as 0.3 or 0.15 are adjustable starting points, not required settings.

Complete the shot/person timelines and segment plan in [video-segment.md](video-segment.md). Derive generated and retained frame ranges from actual FPS, model alignment, overlap and generation direction. Check complete source coverage, including the tail. Use independent anchors across real cuts when appropriate; derive continuation guide indices from the specific raw or trimmed frame batch connected as input, excluding unwanted alignment tails.

## 4. Prepare inputs and configure the graph

Create a task-specific copy of the simple or Agentic template linked in `SKILL.md`. Preserve the original and reuse Global Control, existing nodes and one shared segment-subgraph definition. Read actual node/subgraph input definitions and widget values rather than hardcoding their positions.

Prepare frame-exact source clips at the chosen FPS and a reference size accepted by the deployment. Keep aspect ratio and verify any scaling or padding. Re-probe clips and inspect first/last frames against the plan. Keep the source soundtrack separately if the final edit must retain it. Check reference-duration limits before cutting; extend undersized clips within the same shot where possible and account for added context in trimming.

For remote ComfyUI, confirm the installed media-upload contract, including MP4/M4A acceptance and input/output retrieval through `/view`. Record the tested server version and minimal upload/retrieval example in task artifacts so later runs can reuse verified behavior. The standard `POST /upload/image` accepts a multipart `image` field and returns `name`, `subfolder` and `type`; use the returned input-relative path in loader widgets. For shared local storage, resolve the configured input directory before copying. Version filenames rather than overwriting inputs or results.

Write each segment's prompt using [prompt-writing.md](prompt-writing.md), following the user's replacement scope. Apply the user's explicit appearance scope. The bundled template replaces full appearance; retaining source costumes while changing face/hair is a different policy. If the request leaves this choice materially ambiguous, ask one focused question rather than choosing from the scene or copying an old prompt without checking intent. Difficult close-ups or small figures may benefit from explicit positive appearance cues, which must be checked in generated results.

Set references, duration, guides, direction and trim/keep ranges per instance. Connect continuation dependencies explicitly and keep generated previews before concatenation. Reuse media nodes for source-order assembly and requested audio. Check socket types, including INT generation FPS versus FLOAT Create Video FPS. Keep named and positional widget values consistent where both exist; determine their actual export precedence from the installed frontend or exported API graph rather than guessing. Read subgraph instance/inner-node mappings from export; IDs such as `10078:98` are examples, not stable identifiers. `raw_frames` feeds a batch-length selection: a value such as 4096 is a finite upper bound, not a special “all frames” sentinel. Ensure it covers the planned frame count.

## 5. Export, submit and monitor

Save editable workflow JSON and export an executable API prompt through a verified facility supported by the installed ComfyUI frontend. Check native subgraphs are expanded and all required output dependencies are present. A matching history prompt is an optional baseline, not a substitute for validating a changed topology. If no verified export is accessible, provide the prepared workflow and request a one-time API-format export; do not invent a conversion API or bypass ComfyUI with direct Omni generation requests.

Submit `POST /prompt` with `{"prompt": <api graph>}`. Store the returned `prompt_id` and inspect HTTP errors and `node_errors`. Follow `/history/{prompt_id}`, `/queue` and execution events as available. Missing history alone cannot distinguish queued, running, unaccepted or cleared work. Confirm acceptance before retrying a lost submission response, and confirm terminal status before treating outputs as complete. Existing Omni nodes own downstream request serialization and polling.

When changing an open workflow, preserve current edits and verify the loaded canvas as well as the saved file; an unsaved canvas can overwrite a disk-only change.

## 6. Check and deliver

Retrieve output files reported by history. Check frame count, duration, FPS, expected audio and full decoding. Visually review segment starts/ends, joins, selected and unselected people, close-ups, wide shots, clothing/accessories and soundtrack alignment. Inspect reported defect intervals frame-by-frame where needed.

Deliver the playable result and remaining defects. Save the input mapping, segment plan, prompts, editable workflow, executable graph, execution records and raw outputs for inspection and resume. Successful execution and decoding are separate from visual acceptance.

## 7. Iterate on the affected dependency chain

Change one variable per comparison and retain previous outputs. Editing a segment can invalidate that segment and every downstream segment using its guides, plus assembly. Independent unchanged branches may reuse cache only if it is available and their effective inputs are unchanged. Inspect the actual rerun scope; do not promise that only one segment will regenerate. When necessary, reconnect saved successful outputs explicitly.

Use versioned output prefixes only on affected outputs where possible; changing every prefix can invalidate otherwise reusable save nodes. Retry within the agreed budget and report persistent defects rather than silently changing the model, replacement scope or reference images.

## First-run qualification and history backup

Validate the execution path once with empty history and new inputs. A workflow-to-API exporter/helper must handle native subgraphs, changed instance counts and current widget values without a matching prior run. A documented manual export handoff is a remaining automation limitation, not a completed one-sentence flow.

Before an authorized history clear, wait for an idle queue, save the complete `GET /history` response and verify it can be parsed with the same prompt IDs. Preserve referenced media separately if it must survive file cleanup; history JSON contains metadata and graphs, not media bytes. On versions supporting it, `POST /history` with `{"clear": true}` clears execution history; verify `GET /history` is empty afterward. This does not clear execution cache or delete source/output media. There is no assumed public history-import API: archived graphs may be inspected or resubmitted, but resubmission is a new execution, not restoration of history.
