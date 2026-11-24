# vLLM-Omni Skills

This is the canonical catalog and source directory for project skills. Codex
discovers them here. Claude Code uses same-named relative directory symlinks
under [`.claude/skills/`](../../.claude/skills/readme.md). For other agents, open
the relevant `SKILL.md` directly. If a client scans skills only at startup,
start a new session to pick up changes.

## Contributor and maintainer skills

Invoke the relevant skill by name. Combine a domain skill with `vllm-omni-test`
when tests or CI coverage change, then use `precheck-pr` for the contributor's
final self-check.

- [`add-diffusion-model`](add-diffusion-model/SKILL.md): guides integration of
  a new diffusion model into `vllm-omni`
- [`add-fullduplex-model`](add-fullduplex-model/SKILL.md): guides integration of
  new full-duplex speech and multimodal models on the unified engine-resident
  duplex framework (`DuplexOmni`, `DuplexModelPlugin`, Model Runner V2 hooks,
  `WS /v1/realtime?duplex=1`, and `omni_duplex_eval`)
- [`add-omni-model`](add-omni-model/SKILL.md): integrates a new multi-stage
  thinker/talker/code2wav omni model, from the HF reference through pipeline,
  stage hand-off processors, deploy YAML, parity ladder, tests, and recipe
- [`add-recipe`](add-recipe/SKILL.md): creates or updates in-repository model
  recipes with verified task/input/output contracts, concrete hardware,
  shared feature-guide links, exact commands, and qualification evidence
- [`add-tts-model`](add-tts-model/SKILL.md): covers integration of new TTS
  models and related serving workflows
- [`diffusion-perf-opt`](diffusion-perf-opt/SKILL.md): guides diffusion model
  performance optimization, including profiling traces, parallel strategies,
  stage timing analysis, and benchmark-driven tuning
- [`diffusion-accuracy-test`](diffusion-accuracy-test/SKILL.md): adds or extends
  image/video numerical accuracy tests with aligned references, scoped
  tolerances, and explicit Buildkite Accuracy coverage
- [`diffusion-perf-test`](diffusion-perf-test/SKILL.md): adds or extends
  image/video performance-benchmark cases and their nightly/weekly Buildkite
  Perf coverage, using the `run_benchmark.py` config schema and a fixed
  measurement contract
- [`dit-npu-fusion-analysis`](dit-npu-fusion-analysis/SKILL.md):
  routes Ascend NPU diffusion model and operator optimization to the applicable
  MindIE-SD skill, supplying Omni context and retaining repository-local tests;
  the existing name remains a discovery entry, with optimization methods and
  acceptance guidance maintained in MindIE-SD
- [`find-simplifications`](find-simplifications/SKILL.md): finds evidence-backed
  opportunities to remove or merge dead, duplicated, speculative,
  over-generalized, or unnecessarily defensive vLLM-Omni code
- [`omni-perf-opt`](omni-perf-opt/SKILL.md): guides multi-stage omni serving
  optimization, including stage attribution, TTFT / AUDIO_TTFP / AUDIO_RTF
  tuning, async_chunk codec chunking, CUDA graphs, MRv2, replicas, and
  quality-gated A/B evidence
- [`omni-perf-ceiling`](omni-perf-ceiling/SKILL.md): estimates the best
  reachable omni serving performance (for example AUDIO_RTF at a target
  concurrency) from measured cells and per-stage calibrated roofline estimates, and reports the remaining unattributed gap, on CUDA GPUs
  or Ascend NPUs
- [`precheck-pr`](precheck-pr/SKILL.md): self-checks a branch before creating a
  PR by validating title format, dead code, simplification opportunities,
  accuracy and performance claims, and merge readiness
- [`production-add-diffusion-model`](production-add-diffusion-model/SKILL.md):
  takes a working Day-0 diffusion integration through API/limit parity,
  feature-combination evidence, quantization/offload, performance, production
  serving, hardware recipes, and CI readiness
- [`quantization`](quantization/SKILL.md): guides quantization method selection,
  model integration, checkpoint loading, and quality/performance validation
  for vLLM-Omni
- [`review-pr`](review-pr/SKILL.md): provides a frozen-snapshot, contract-aware
  workflow for maintainers and reviewers; contributors should use
  `precheck-pr` for self-review
- [`vllm-omni`](vllm-omni/SKILL.md): entry skill for multi-stage omni models;
  covers deploy YAML, serving, debugging, benchmarking, and the code map, and
  routes model integration and performance work to the dedicated skills
- [`vllm-omni-npu-model-runner-upgrade`](vllm-omni-npu-upgrade/SKILL.md):
  upgrades NPU model runners to align with the latest vllm-ascend
  `NPUModelRunner`
- [`vllm-omni-test`](vllm-omni-test/SKILL.md): guides generation and execution
  of CI-aligned tests (L1–L4), pytest marker selection (`core_model` /
  `advanced_model` / `full_model`, `omni` / `tts` / `diffusion`), Buildkite
  wiring (`test-ready.yml`, `test-merge.yml`, `test-nightly.yml`,
  `test-weekly.yml`), and copy-paste local plus CI-like `pytest` commands; see
  `references/test-routing.md` for level-to-command mapping

### Ascend Optimization Routing

For Ascend NPU diffusion tasks, `diffusion-perf-opt` consults the
[MindIE-SD router](dit-npu-fusion-analysis/SKILL.md)
before hardware-specific analysis. The router locates an existing checkout or
the official dev skill index and reads the selected instructions. It does not
copy SD's operator catalogue, benefit model or acceptance protocol into Omni.

| Task | Guidance owner | Omni responsibility |
| --- | --- | --- |
| Model optimization, bottleneck unknown | MindIE-SD model optimization entry | Supply active source, workload/configuration and existing evidence |
| Fusion, kernel, pattern or focused profiling task | Relevant MindIE-SD capability | Preserve repository ownership and authorized execution scope |
| Changes to Omni | Omni test/accuracy skills plus SD NPU acceptance guidance | Implement/test the Omni changes and return actual artifacts |
| Production qualification | production-add-diffusion-model | Retain independent API, lifecycle and feature-combination checks |

Read the external index at execution time; selected names and support can vary
by revision. If guidance is inaccessible, disclose the gap rather than claiming
that the external workflow ran.

## MiniMax H3 skills

The collection contains general H3 prompt-writing guidance, eight creative
skills adapted from MiniMax H3, and the repository-specific character replacement
skill. See [H3-SOURCES.md](H3-SOURCES.md) for the upstream revision, attribution,
and scope.

| Skill | Intended use |
| --- | --- |
| [h3-prompt-writing](h3-prompt-writing/SKILL.md) | Structured prompts for T2VA, I2VA, FL2VA, L2VA, and Ref2VA |
| [minimalist-product-ad-generator](minimalist-product-ad-generator/SKILL.md) | Minimal product ads, isolated product references, copy, and pacing |
| [3d-animation-short-generator](3d-animation-short-generator/SKILL.md) | 3D narrative shorts, character and environment design, shot tables, storyboards, and editing |
| [papercraft-stop-motion-explainer](papercraft-stop-motion-explainer/SKILL.md) | Layered paper art, puppets, mechanisms, and miniature educational scenes |
| [brand-promo-video-generator](brand-promo-video-generator/SKILL.md) | Brand promos, verified assets and claims, product interaction, and calls to action |
| [music-video-subtitle-generator](music-video-subtitle-generator/SKILL.md) | Music videos, instrumental visual narratives, beat planning, and optional lyric typography |
| [co-op-game-intro-generator](co-op-game-intro-generator/SKILL.md) | Two-player game menus, confirmation images, and animated world entry |
| [paper-collage-explainer-generator](paper-collage-explainer-generator/SKILL.md) | Halftone photo collage, incremental paper assembly, and visual metaphors |
| [handdrawn-live-video-generator](handdrawn-live-video-generator/SKILL.md) | Live action with rough luminous drawings, physical contact, continuous morphing, and delayed pursuit |
| [vllm-omni-h3-character-replacement](vllm-omni-h3-character-replacement/SKILL.md) | Visual segment planning, per-segment replacement prompts and execution through reusable H3-04 ComfyUI workflows |

### Example prompts

```text
Use $music-video-subtitle-generator to design a 30-second two-dimensional
fluid-motion graphic for this instrumental track. Do not add lyrics or
subtitles. Preserve leftward motion and initially return only a storyboard and
three H3 prompts.
```

```text
Use $co-op-game-intro-generator to create a two-player game intro. The players
are Lin and Mei, the style is layered paper sculpture, and the title is
Together. Create the confirmation image first and wait for my approval before
generating video.
```

### MV smoke test

Test `music-video-subtitle-generator` together with `h3-prompt-writing` using a
user-owned song with timestamped lyrics and an original adult-character
reference image:

```text
Use $music-video-subtitle-generator and $h3-prompt-writing to turn this song
into a historical-romance music video about a woman and a man. Intercut
narrative acting with the woman's singing performance, do not display lyric
text, split the work according to the backend's verified clip duration, and
preserve the complete original song in the final master.
```

Verify that the global timeline matches the source audio duration, lyrics remain
verbatim, and each generation window records its incoming state, outgoing state,
and local timing. Ref2VA prompts must preserve the six-section field order and
resolve every reference label to a real input. The assembled video must decode
completely, and its final audio must come from the supplied source rather than a
model reconstruction.

When moving the collection to another repository or a user-level skill folder,
copy the nine upstream-adapted skill directories and preserve their sibling layout. The eight
creative skills share `h3-prompt-writing/references/portable-workflow.md` and the
H3 format guides, so copying only one `SKILL.md` is insufficient. Keep each
skill's `references/` and `agents/` directories with it. `agents/openai.yaml`
provides optional Codex UI metadata and does not restrict the workflow to an
OpenAI-based agent or require a dedicated plugin.

The character-replacement skill also depends on repository workflow templates linked from its entrypoint. Preserve those relative paths when relocating it, or update the links for the destination.

### Execution boundaries

Prompt writing, storyboarding, and planning require only an agent that can read
and write files. Actual image, video, speech, music, and editing work uses tools
or services available in the current environment. When a capability is missing,
deliver the complete usable production package and identify the unfinished media
step.

The user's style, language, assets, audio, model selection, and existing
authorization take priority. An MV template does not add lyrics to an
instrumental track, and a clean motion-graphics request does not inherit film
grain. A prompt-only request does not start generation. When the user requests
approval of a confirmation image before rendering, preserve that checkpoint.

Treat vLLM-Omni long-video extensions as capabilities of the checked-out branch
and active backend. Do not present a development branch's continuation parameters
as a universal H3 API. Preserve real requests and raw outputs, distinguish model
results from post-production and upscaling, and inspect actual media properties.

These skills contain no model weights and do not automatically start a service
or invoke a paid generation endpoint.

## Adding or updating skills

Keep each skill's `SKILL.md` and supporting files in a directory directly
under `.agents/skills/`. For a new skill, add its catalog entry above and
create the matching Claude Code discovery link from the repository root.

```bash
skill_name=your-skill-name
ln -s "../../.agents/skills/${skill_name}" ".claude/skills/${skill_name}"
```

Use the `.agents/skills/` paths in GitHub links and tools that read Git trees.

- Keep skill names short and task-oriented.
- Prefer repository-local paths, commands, and examples.
- Avoid hardcoding fast-changing support matrices unless the skill is actively
  maintained alongside those changes.
- Treat skills as contributor tooling: optimize for clarity, actionability, and
  low maintenance overhead.
