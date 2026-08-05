# Build / setup log — LongCat-Next on 4x A100 SXM4 80GB

Reproducible steps to go from a fresh GPU pod to a working
`thinker(0) -> multi_decoder(1)` LongCat-Next pipeline. Captures every
workaround that was actually needed, in order, including the dead ends —
skip a step at your own risk, several of these exist because the naive
approach silently breaks something two steps later.

Target hardware: 4x NVIDIA A100-SXM4-80GB (works the same on H100-80GB).
Base image used: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
(only Python 3.11 + system CUDA 12.4 toolchain matter from this image —
everything else gets replaced below).

## 0. Repos

```bash
mkdir -p /workspace && cd /workspace
git clone -b feat/longcat-next-integration https://github.com/gangula-karthik/vllm-omni.git
git clone -b feat/longcat-next https://github.com/gangula-karthik/vllm.git
```

Put these on a **local (non-network-mounted) disk** if you have the choice.
On Runpod, `/workspace` is a network volume (MooseFS/FUSE) — cloning here is
fine (small, one-shot), but the Python venv and pip/uv caches must NOT live
here (see step 2) or you will hit intermittent I/O stalls and outright
`RuntimeError: Task error ... Background writer channel closed` /
`Disk quota exceeded` failures under load.

## 1. uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

`uv` itself installs to `/root/.local/bin` — also local disk, so it survives
independently of whatever happens to the venv.

## 2. Venv — must be on local disk

```bash
mkdir -p /root/venvs
uv venv --python 3.11 /root/venvs/dev
source /root/venvs/dev/bin/activate
export UV_CACHE_DIR=/root/.cache/uv
```

**Do not** put the venv or `UV_CACHE_DIR` under `/workspace`. Two failure
modes observed there: (a) large multi-file installs (torch, vllm) randomly
stall in uninterruptible disk-wait (`D` state) for minutes; (b) any
`--volume-in-gb` resize on the pod (see step 6) triggers a full container
recreation, which wipes `/root` but preserves `/workspace` — so a venv on
`/workspace` would survive a resize, but one on `/root` won't. Either way,
budget for rebuilding the venv from scratch at least once per session; it's
fast once uv's package cache is warm (~2 min for vllm+vllm-omni).

## 3. vllm — editable, precompiled-wheel fast path

```bash
cd /workspace/vllm
VLLM_USE_PRECOMPILED=1 uv pip install --editable . --torch-backend=auto
```

This downloads a prebuilt vllm wheel (matched to the nearest upstream
`main` commit) for the compiled extensions, and only builds the Python
layer locally — full from-source compilation is not needed since none of
our changes touch CUDA kernels. Pulls in `torch==2.11.0+cu130` and friends
as a side effect.

## 4. vllm-omni — editable, no-deps + separate requirements install

```bash
cd /workspace/vllm-omni
uv pip install --editable . --no-deps
uv pip install -r requirements/cuda.txt
```

`--no-deps` on the first call matters: vllm-omni's own dependency
resolution does not pin vllm (it expects you to bring your own, per
setup.py), but resolving the full extras list in one shot can otherwise
drag in a conflicting vllm/torch version. Installing `requirements/cuda.txt`
separately avoids that.

## 5. flash-attn (hard requirement, not optional)

LongCat-Next's HF `modeling_longcat_next*.py` imports `flash_attn` directly
at module load time — without it the thinker/decoder workers fail with
`ImportError: This modeling file requires ... flash_attn` as soon as they
try to construct the model, not at inference time. There is no way to skip
this and still run the pipeline.

Building it from source needs a CUDA toolchain, and this is the fiddliest
part of the whole setup:

```bash
# nvcc frontend: 13.0.88 matches the headers torch's pinned
# nvidia-cuda-runtime==13.0.96 ships (anything much newer here fails an
# internal CCCL header/compiler-version compatibility check)
uv pip install nvidia-cuda-nvcc==13.0.88
cp /root/venvs/dev/lib/python3.11/site-packages/nvidia/cu13/bin/ptxas /root/ptxas_13.0.88.bak

# ptxas: 13.0.88's own bundled ptxas has an internal version skew bug —
# its cicc frontend emits PTX ISA .version 9.3 but its own ptxas only
# accepts up to 9.0 ("Unsupported .version 9.3; current version is '9.0'").
# Swap in the newer package's ptxas binary (newer ptxas accepts older/newer
# PTX ISA fine; only the frontend/headers need to match torch's runtime).
uv pip install nvidia-cuda-nvcc==13.3.73
cp /root/venvs/dev/lib/python3.11/site-packages/nvidia/cu13/bin/ptxas /root/ptxas_13.3.73.bak
uv pip install nvidia-cuda-nvcc==13.0.88   # reinstall to restore matching headers
cp /root/ptxas_13.3.73.bak /root/venvs/dev/lib/python3.11/site-packages/nvidia/cu13/bin/ptxas

# lib64 doesn't exist in this pip package layout (everything is under lib/)
# but flash-attn's / flashinfer's build scripts hardcode -L.../lib64 and
# link -lcudart / -lcuda unversioned, so:
ln -sfn /root/venvs/dev/lib/python3.11/site-packages/nvidia/cu13/lib \
        /root/venvs/dev/lib/python3.11/site-packages/nvidia/cu13/lib64
ln -sf /root/venvs/dev/lib/python3.11/site-packages/nvidia/cu13/lib/libcudart.so.13 \
       /root/venvs/dev/lib/python3.11/site-packages/nvidia/cu13/lib/libcudart.so
# libcuda.so is provided by the driver at /usr/lib/x86_64-linux-gnu/ already.

export CUDA_HOME=/root/venvs/dev/lib/python3.11/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export FLASH_ATTN_CUDA_ARCHS=80   # A100 = sm_80 only; NOT torch's
                                   # TORCH_CUDA_ARCH_LIST, which flash-attn's
                                   # setup.py ignores. Restricting away from
                                   # the default "80;90;100;120" also avoids
                                   # rebuilding kernels for archs you don't
                                   # have (much faster + sidesteps the same
                                   # ptxas PTX-version issue for sm_100/120).
export MAX_JOBS=32
uv pip install flash-attn --no-build-isolation
```

Without the `lib64` symlink, `flash-attn` itself actually links fine (its
own `-L.../lib` is correct), but `flashinfer`'s JIT-compiled sampling
kernel (built lazily at first inference call, not at install time) fails
the same way — so this bites you later, mid-generation, if skipped.

## 6. Model download

```bash
mkdir -p /workspace/models
export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_DOWNLOAD_MAX_WORKERS=4
uvx --with hf_xet hf download meituan-longcat/LongCat-Next \
    --local-dir /workspace/models/LongCat-Next --max-workers 4
```

~160GB. Two things that will bite you:

- **Pod volume quota**: `df -h /workspace` reports the underlying cluster's
  free space (hundreds of TB), which is *not* your pod's actual quota — that
  quota is set at pod-creation time via `--volume-in-gb`. A download that
  exceeds it fails with `IO Error: Disk quota exceeded (os error 122)` even
  though `df` shows plenty of room. Resize with
  `runpodctl pod update <id> --volume-in-gb 300` (or whatever you need) —
  **this recreates the container** (new hostname, `/root` wiped, `/workspace`
  preserved), so expect to redo steps 1-5's venv/tool installs afterward.
- **hf_xet transient failures**: the Xet downloader can hang indefinitely
  (CLOSE_WAIT sockets, one thread pinned at 100% CPU, no progress) or crash
  outright (`RuntimeError: Task error: File reconstruction error: Internal
  Writer Error: Background writer channel closed`). Both are recoverable —
  kill and rerun the same `hf download` command; it resumes from the
  `.cache/huggingface/download/*.incomplete` files already on disk rather
  than restarting. Reducing `--max-workers` from the default reduces (but
  doesn't eliminate) how often this happens.

## 7. Deploy config

Several other deploy variants (5-GPU A40, additional 4-GPU 80GB configs) were
tried during setup and later pruned from the repo — the only one kept is
`longcat_next_4gpu_80gb_multi_decoder.yaml` (used throughout this doc), for
the **2-stage** `multi_decoder` pipeline, because the alternative 3-stage
`longcat_next` pipeline has a real bug: the
orchestrator's `_forward_to_next_stage` always forwards `src_stage_id + 1`'s
own output, never consulting a stage's declared `input_sources`, so the
audio decoder (stage 2) always receives the image decoder's output (stage
1), never the thinker's (stage 0) — it can't ever see real audio codes.
`vllm_omni/deploy/longcat_next_4gpu_80gb_multi_decoder.yaml` in this repo
fills that gap: thinker TP=4 across all 4 GPUs, `LongcatNextMultiDecoder`
colocated with thinker rank 3 on GPU 3. On 80GB cards there's enough
headroom to skip the 5th "decoder-only" GPU the a40 variant needs — set
thinker's `gpu_memory_utilization` low enough (0.65 here) to leave ~25-28GB
free on GPU 3 for the decoder's own weights.

## 8. Code fixes required (already committed on this branch)

Getting the yaml to actually run end-to-end needed two real code fixes in
`vllm-omni`, not just infra work — see git log on
`feat/longcat-next-integration` for the full diffs:

- `vllm_omni/model_executor/models/registry.py`: `LongcatNextMultiDecoder`
  was never registered in `_OMNI_MODELS`, so vllm's `ModelRegistry` rejected
  it as an unsupported architecture.
- `vllm_omni/core/sched/omni_generation_scheduler.py`: this vllm fork
  changed `Scheduler._free_request()` to return a
  `(kv_transfer_params, ec_transfer_params)` tuple (for a new EC-connector
  feature); `omni_generation_scheduler.py` was still treating it as a
  single value, so the tuple itself got wire-serialized into the
  `kv_transfer_params` slot and blew up on decode
  (`msgspec.ValidationError: Expected object | null, got array`). Fixed the
  three call sites to unpack both values.
  (`omni_ar_scheduler.py` has its own fully-custom `_free_request` override
  that never returns a tuple — do not "fix" that one the same way.)

## 9. Run

```bash
cd /workspace/vllm-omni
python pbs/scripts/longcat_next_wired_e2e.py \
    /workspace/models/LongCat-Next \
    vllm_omni/deploy/longcat_next_4gpu_80gb_multi_decoder.yaml \
    /workspace/results --modality audio   # or --modality image
```

(`pbs/` was a scratch directory of one-off HPC job/debug scripts, removed
from the repo before this PR — the command above is preserved here as a
historical record of what was run, not a script you can invoke today.)

Both modalities reach `verdict: PASS` (correct output *shape*/*file
written*) with this setup, but that only validates pipeline wiring — see
`pbs/scripts/longcat_next_debug_quality.py` (also removed, see above) for
evidence that generation *quality* has separate, unresolved bugs (garbage
text on a plain non-multimodal prompt, and an audio-codes accumulation bug
that drops all but the last generated frame before it reaches the client).

## 10. Quality debugging log (post-wiring)

Once the pipeline ran end-to-end (§9), a second, much longer debugging pass
went after *generation quality* — text reading as generic filler instead of
answering the prompt, audio/image running forever or collapsing. This
section is the log of everything tried, in order, what it ruled out, what it
fixed, and what's still open. Read top to bottom; each subsection assumes
the previous ones' fixes are already applied (they're committed on this
branch — check `git log` for the commit named in each item).

### Text

**Fixed:**
- `LONGCAT_NGRAM_DISABLE`/oe_ids audit added (debug tooling, not a behavior
  change) to test whether n-gram fusion was the source of divergence.
- Deploy YAML stage-0 `default_sampling_params` were `0.4/–/0.9/1.05`
  (temperature/top_k/top_p/repetition_penalty); the reference's own
  `test_cases.yaml` uses `0.4/20/0.85/1.1` for this exact text-generation
  profile. `pbs/scripts/longcat_next_debug_quality.py`'s (removed, see §9)
  `run_text` had the same gap — it used `0.4/20/0.85` but omitted `repetition_penalty`
  entirely, silently falling back to vLLM's default of `1.0` (no penalty).
  This alone was the fix for the *original* symptom (`finish_reason=length`,
  output degenerating into a repeated-digit loop): with `repetition_penalty=1.1`
  restored, generation now terminates cleanly (`finish_reason=stop`) after a
  short, grammatical response.
- **Remaining symptom after that fix**: the response is coherent Chinese but
  *generic* — asked "请简单介绍一下你自己" (please introduce yourself), the
  port answers "Of course, I'll help you with whatever you need" instead of
  actually introducing itself. The reference model, given the identical
  prompt and identical sampling params, produces a full, on-topic
  self-introduction (name, developer, capabilities). This is the open bug
  this whole log chases.

**Ruled out, in order, each via a real on-pod comparison (not guessed):**
1. **Tokenization** — dumped `input_ids` from both the port
   (`[longcat-text]` log line, `LONGCAT_AUDIO_DEBUG=1`) and the reference
   (`tok(prompt, add_special_tokens=False).input_ids` on the raw HF
   tokenizer). Byte-for-byte identical 16-token sequence on both sides:
   `[46, 3320, 815, 483, 20110, 18498, 237, 444, 47, 2252, 5552, 41212,
   42446, 525, 444, 48]`. Not a prompt-construction or BOS-handling bug.
2. **N-gram embedding fusion** — `LONGCAT_NGRAM_DISABLE=1` (bypasses
   `_span_oe_ids`/`ngram_embeddings.embed_batched`, falls back to pure word
   embeddings). Result: *worse*, not better — output degenerated into
   incoherent English/Chinese word salad (`'It is beepsile岗atory的...'`).
   This rules the n-gram path OUT: a genuinely broken fusion wouldn't
   produce *more* coherent text when correctly applied, and disabling a
   subtly-wrong one shouldn't make things catastrophically worse. The
   mechanism is necessary and working as intended.
3. **MLA/RoPE config** (`LONGCAT_CONFIG_AUDIT=1`, logs `FlashConfig`'s
   actual loaded values from `vllm/model_executor/models/longcat_flash.py`) —
   compared against the checkpoint's raw `config.json`. Every value matches
   exactly, including the two most likely to silently fall back to a wrong
   Python default: `rope_theta=10000000` (not `FlashConfig`'s default
   `1000000.0`) and `max_position_embeddings=131072` (not the default
   `8192`). Config loading is not the bug.
4. **Weight loading** — reran with `VLLM_LOGGING_LEVEL=DEBUG` (the level
   `AutoWeightsLoader` actually logs missing/unexpected keys at) and grepped
   for `missing|unexpected key|not initialized|randomly init` — zero
   matches across a substantial (1046-line) log. No submodule is silently
   falling back to random init from a name/shape mismatch.

**Current lead — logit-level divergence (not yet root-caused):**
Dumped the top-10 pre-sampling logits of the first decode step
(`[longcat-logits]` log line) and compared against the reference's raw
`model.generate(inputs, max_new_tokens=1, do_sample=False, output_logits=True,
return_dict_in_generate=True)` (note: the reference's `forward()` cannot be
called directly — it needs internal `multimodal_generation_status` state that
only `generate()` sets up; calling `model(input_ids)` raises
`AttributeError: 'NoneType' object has no attribute 'mode'`). Also note the
reference needs `device_map="auto"` to load across all 4 GPUs — a plain
`.to("cuda")` OOMs on a single 80GB card.

Reference top-10: `ids=[23958, 7600, 9400, 44408, 49488, 2909, 25044, 56992,
31553, 78636]`, `vals=[25.5, 23.5, 19.5, 18.875, 18.5, 18.125, 17.75, 17.75,
17.0, 16.75]`, range `[-19.875, 25.5]` (span ~45.4).
Port top-10: `ids=[47454, 7600, 31553, 2909, 16707, 2620, 60059, 90310, 765,
23220]`, `vals=[14.875, 14.375, 14.312, 14.188, 13.75, 13.625, 13.312,
13.312, 13.125, 13.0]`, range `[-12.438, 14.875]` (span ~27.3).

Only 3/10 ids overlap (`7600`, `2909`, `31553`), ranked differently, and the
reference's dominant top-1 (`23958`, clear +2.0 margin) isn't in the port's
top-10 at all. **This re-ranking is the key detail**: a single missing head
scalar (a `logits_scale`/softcapping factor) would compress magnitude but
preserve rank order — it can't explain 7/10 ids changing which class they
even are. Divergence must accrue across layers, not live in one place at the
head. So the bisect target is the per-layer residual-stream RMS curve, not
a head-level scalar.

**ROOT-CAUSED AND FIXED — see §11.** The per-layer RMS bisect (forward hooks
on every `FlashDecoderLayer`, `[longcat-layers]`) found the port's curve
already ~10x too large at layer 0 and shaped completely differently from the
reference's (reference climbs smoothly 0.54→10.2 across layers, port starts
at 5.6 and crashes to a 0.5-0.9 plateau) — pointing well upstream of any
per-layer attention/MoE bug. Root cause: the MLA LoRA input scale
(`mla_scale_q_lora`/`mla_scale_kv_lora`) was never actually reaching the real
checkpoint weights (§11). Fixed in commit (this session, vllm-omni only, no
vLLM changes) — post-fix top-10 is a 9/10 exact match against the reference
with matching rank order (§11).

### Audio

**Fixed** (all committed, see git log on this branch for exact diffs):
- Prefill-step `decode_eligible` desync: `<longcat_audiogen_start>`/
  `<longcat_img_start>` landing as the last token of a multi-token prefill
  chunk (true whenever it's written directly into the prompt) advanced the
  state machine's `gen_step` without the runner actually invoking
  `talker_mtp` that step, silently losing the first frame's code forever.
  Fixed by gating the advance on `decode_eligible = (not is_prefill) and
  span_len == 1`, matching the runner's own talker_mtp dispatch condition.
- Audio termination: `compute_logits` used to `continue` entirely once
  `audio_state["terminal"]` was set (chunk-end or `max_gen` safety cap),
  fully unbanning EOS with no forced replacement — the model was free to
  (and did) end the *whole request* via real EOS within a few tokens,
  instead of just closing that one audio segment. Fixed by forcing
  `<longcat_audiogen_end>` on the terminal step (mirroring how the image
  side forces `IMAGE_END`), so the model only regains full freedom after a
  clean close.
- Chunk-boundary marker: once the above fix let a single request produce
  *multiple* audio segments (re-entering `<longcat_audiogen_start>` after a
  clean close), those segments' codes concatenated with no surviving
  boundary between them — `LongcatNextAudioDecoder._split_chunks` saw one
  giant merged chunk instead of several bounded ones and overflowed the
  checkpoint's fixed-size positional embedding table
  (`positions_2d 5731 != L 3000`). Fixed by emitting an explicit chunk-end
  marker row (`level0 = codebook_sizes[0]`) on every terminal step instead
  of a `-1` discard row, so `_split_chunks` can actually see the boundary.

**Current state**: mechanically clean — `verdict: PASS`, 89 rounds, 1433
kept frames, decodes without crashing. The 108.6s-for-a-5s-sentence
repetition-loop symptom, and a separate `RPC call to sample_tokens timed
out` / `EngineDeadError` hang seen later in this session, both stopped
reproducing once the MLA scale bug (§11) was fixed — a full English-prompt
re-run post-fix produced a correctly-timed 5.58s clip for a comparable
sentence with no hang. Consistent with the same shared upstream cause as
Text/Image: garbage backbone hidden states driving the autoregressive
audio-code sampling into a self-reinforcing loop (or, in the timeout case,
some downstream numerical state that never resolved). Not independently
root-caused beyond that — if it resurfaces, re-check with `LONGCAT_AUDIO_DEBUG=1`
before assuming it's a new bug.

### Image

**Fixed** (committed):
- Same prefill-step `decode_eligible` desync as audio (see above) — was
  losing the image's first pixel frame, causing the image decoder to
  receive one fewer code than the grid expected
  (`positions_2d`/frame-count mismatch).
- Grid-bound termination: image generation had no forced-termination
  transition (unlike audio's `max_gen`), so a thinker call with a loose
  `max_tokens` kept sampling past the intended grid (observed: 1994 codes
  for a 1369-code 37x37 grid) until the token budget ran out. Fixed by
  forcing `<longcat_img_end>` once `gen_step` reaches `token_h*(token_w+1)`.
- Image decoder hardened symmetrically: truncates extra codes (already
  existed) *and* now fails cleanly instead of crashing the reference decoder
  when it receives fewer codes than the grid expects.

**Implemented**: Visual CFG (classifier-free guidance) — an unconditional
"twin" request (`<longcat_img_start>` prompt with the user's instruction
string-blanked) is spawned via `prompt_expand_func`/`expand_longcat_cfg_prompts`,
paired with the parent via `__cfg_visual` request-id suffix and
`max_num_seqs=2` so both land in the same decode batch. Combined per-step:
`combined = cfg_scale * (cond_logits - uncond_logits) + uncond_logits`,
matching the reference's own formula in `output_processor.py`. Verified via
`[longcat-cfg-logits]`/`CFG codes0=` audit log (`LONGCAT_AUDIO_DEBUG=1`):
parent+twin combine every single step with zero desync warnings for the
runs tested — the CFG *mechanism* itself is wired correctly.

**Bug found and fixed**: `cfg_scale=1.0` (mathematically forces
`combined = cond`, i.e. no CFG effect — a control case) produced
"recognizable but imperfect" content (described as "some lions"); any
`cfg_scale > 1.0` (tried `1.5` and `3.0`) collapsed to a uniform grey image,
**not gradually** — 1.5 was just as broken as 3.0. Diagnostic
(`[longcat-cfg-logits]`, dumps `cond`/`uncond`/`combined` logit stats)
showed the `cond`/`uncond` delta was *small* (~0.2–0.6 out of a ~40-wide
logit range) — ruling out "amplification blowup" as the mechanism. The real
signature: at `cfg_scale=1.5`, the level-0 argmax was **frozen on a single
class for every logged step**, vs. genuine per-position variation at
`cfg_scale=1.0`. Interpretation: CFG's small perturbation doesn't flip the
top logit, but reshapes probability mass over near-tied candidates enough to
shift what `do_sample`/top-k/top-p actually draws; since visual sampling had
`repetition_penalty=1.0` (no penalty) by default, an early draw that nudges
off-distribution has nothing pulling it back and the autoregressive loop
self-reinforces into a repeated/flat code — a discrete-autoregressive CFG
failure mode, distinct from the continuous-diffusion CFG intuition. Fix:
`LONGCAT_VISUAL_REP_PENALTY` env override (default stays the reference's
`1.0`); testing with `1.5` broke the frozen-argmax pattern (verified: argmax
now varies step to step) and produced a structured, non-grey image.

**FIXED — was the same shared upstream cause.** With the MLA scale bug (§11)
fixed, a re-run of "please generate a picture of a cat sitting in a garden"
produced correct, on-prompt image content (previously: unrelated content —
one run described as "a bed and some fingers and some other random stuff").
Confirms the hypothesis below: prompt adherence was never an image-decoder
or CFG-specific bug, it was the thinker backbone producing near-garbage
conditioning for every downstream decoder.

(Original note, kept for context): same *shape* of symptom as the text
finding above (coherent but generic/off-target), and the `cond`/`uncond`
logit delta being small in the first place (weak differentiation from the
prompt) was consistent with a shared upstream cause rather than anything
CFG- or image-decoder-specific.

### Environment notes learned the hard way

- **Persistent shell state bites twice**: this pod's tmux session is one
  continuous bash shell across the whole debugging pass — `export
  LONGCAT_NGRAM_DISABLE=1` (or `LONGCAT_CFG_SCALE`, etc.) set for one A/B
  test silently persists into the *next* command if not explicitly
  `unset`. Caused at least one contaminated run (a logits-dump test
  accidentally ran with ngram fusion still disabled from an earlier test).
  Always `unset` every `LONGCAT_*` debug env var at the start of a script,
  or `echo` them back before trusting a run's output.
- **The reference model needs `device_map="auto"`** for any raw
  `transformers` script — the full model does not fit on a single 80GB GPU.
  A plain `.to("cuda")` OOMs after checkpoint loading, mid-`.to()` call.
- **The reference's `forward()` is not directly callable** — LongCat-Next's
  own remote code reads `multimodal_generation_status` (an object only
  `generate()`'s internal loop constructs) even for a single forward pass.
  Use `model.generate(..., max_new_tokens=1, output_logits=True,
  return_dict_in_generate=True)` to get pre-sampling logits instead of
  calling the model directly.
- **`transformers` version conflict between vllm-omni and the raw
  reference**: vllm/vllm-omni's own resolution pins `transformers==5.14.1`;
  the checkpoint's remote code (`Qwen2RMSNorm` import in
  `modular_longcat_next_visual.py`) only resolves against an older
  `transformers`. `4.57.1` works. Two options: a separate venv just for
  reference comparisons (safest, but needs its own `flash_attn` rebuilt —
  slow), or temporarily swap `transformers` inside the *existing* venv
  (fast, since `flash_attn` is already built there and isn't
  version-sensitive to the pure-Python `transformers` package) and swap
  back to `5.14.1` immediately after — **do not forget the swap-back**, or
  the next vllm-omni run fails with `ImportError: Support for Transformers
  v4 is deprecated and was removed in vLLM v0.24.0`.
- **RunPod's SSH proxy (`ssh.runpod.io`) requires a PTY for everything** —
  a non-interactive `ssh host 'command'` (needed for scp-style file pulls or
  piping a heredoc via stdin) silently fails or drops to an interactive
  shell instead of running the command; `ssh -tt` doesn't fix it either for
  piped/redirected use. The only reliable channel found was the existing
  interactive tmux session (`send-keys` to type commands, `capture-pane` to
  read output). To pull a binary file (e.g. a generated PNG) back to a
  local machine without direct scp access: `base64 -w0 file` inside the
  tmux pane between unique start/end markers, `tmux capture-pane -p -J`
  (the `-J` join-wrapped-lines flag is required — without it, `base64`'s
  long single line gets fragmented across the pane's physical terminal
  width and corrupts on decode), then locally `base64 -d`. Also increase
  tmux's `history-limit` (`tmux set-option -g history-limit 200000`) before
  dumping anything long — the default (found to be `50000` on this setup)
  can silently truncate the *front* of a long base64 blob out of scrollback
  before it's captured.
- **A "connection to ssh.runpod.io closed" followed by a different
  container hostname on reconnect means the pod's container was recreated**
  (community-cloud preemption, an OOM-kill at the host level, or similar) —
  `/workspace` (the persistent volume) survives, but `/root` (the venv,
  `uv`'s own install, the flash-attn build) does not. Check
  `hostname`/prompt hostname after any reconnect before assuming the venv
  from before is still there; `uv pip list` returning nothing is the
  fastest confirmation. Community cloud pods can also fully stop
  (`status: EXITED` via the RunPod API) without any prompt-visible warning
  mid-session — this is the tradeoff of choosing community over secure
  cloud pricing.

## 11. Root cause found and fixed: MLA LoRA scale never reached real weights

The §10 investigation resumed on a fresh pod with a custom pre-baked Docker
image (see §12). Three more instrumentation bugs had to be fixed before the
per-layer RMS/logit audit could be trusted at all:

- The audit's read side (`compute_logits`) disarmed itself on the *first*
  qualifying call, assuming that meant the first real decode step. It
  didn't: vLLM's own startup memory-profiling run also drives a full
  forward pass (including `compute_logits`, to size the logits tensor)
  through an all-zero hidden-state batch, and can present the same
  single-token logits shape. Every prior capture in §10 was silently this
  dummy pass, not real generation. Fix: a genuine forward pass can't land on
  an *exact* 0.0 RMS at every layer (RMSNorm, bias-free linears, and
  attention over an all-zero V all map zero to zero exactly) — use that as
  the discriminator, and don't disarm the audit on a degenerate capture.
- Ngram speculative decoding (`FlashNgramModel`) runs its own separate
  small-batch dummy warm-up pass that the layer-RMS hooks (reset only on
  `idx==0` of each forward call) couldn't distinguish from a real one either.
  `LONGCAT_NGRAM_DISABLE=1` sidesteps this for isolated captures.
- The RunPod base image's `CMD` must stay `["/start.sh"]` — overriding it
  (even to `["/bin/bash"]`) silently disables the sshd bootstrap and the pod
  never accepts SSH connections. A `PUBLIC_KEY` env var must also be set at
  pod-creation time or `/start.sh` never seeds `authorized_keys`.

With those fixed, a real per-layer RMS curve finally came through:

```
Reference: [0.54, 3.78, 6.38, 7.21, 8.13, 8.75, 9.24, 9.62, 9.99, 10.03, 10.10, 10.18, 5.95, 1.50]
Port:      [5.63, 2.40, 0.66, 0.89, 0.92, 0.68, 0.59, 0.60, 0.56, 0.52, 0.45, 0.68, 4.62, 6.68]
```

Not just wrong magnitude — inverted shape. The reference climbs smoothly
(normal residual-stream growth); the port starts ~10x too high, crashes to a
low plateau, then spikes at the end. Diverging already at layer 0 ruled out
"the bug compounds slowly across layers" and pointed upstream — either the
embedding step or layer 0 itself.

**Root cause**: `FlashModel.load_weights` (`vllm/model_executor/models/
longcat_flash.py`) applies the LongCat-specific MLA LoRA input scale
(`mla_scale_q_lora`/`mla_scale_kv_lora`) to `q_a_layernorm`/
`kv_a_layernorm.weight` in place, guarded by a permanent per-attn-module
"already scaled" flag (`_mla_q_lora_scaled`/`_mla_kv_lora_scaled`) meant to
prevent double-scaling across repeated `load_weights` calls. Auditing every
checkpoint key actually streamed into that function (a temporary
`[longcat-stream-keys]` log) showed it fires more than once per model
instance: an earlier pass whose incoming weights contain **zero**
`self_attn` keys at all, followed by a later pass carrying the real
checkpoint data. The guard trips on that first, data-less pass — multiplying
whatever default-init value (`1.0`) was sitting in the layernorm weights at
that point — and the later pass's real data then overwrites them via a
plain `copy_` (not additive), silently wiping out that scaling without ever
re-triggering it, since the guard is already permanently tripped. Net
effect: the real checkpoint's `q_a_layernorm`/`kv_a_layernorm` weights ended
up completely unscaled, every layer, every run.

(This also explains the earlier, confusing §10-era back-and-forth: the
originally-committed `_apply_mla_scale_fold()` in vllm-omni was removed
mid-session on the theory that it double-applied on top of the vLLM fork's
own scaling — reasonable, since a checksum showed the fork's logic *does*
fire and *does* set its guard. But "fires" isn't "fires on the right data".
Removing the fold left the real weights with **zero** scaling instead of
**double** scaling — worse in a different way, which is why that removal
made the port's logits regress rather than improve.)

**Fix** (commit this session, `vllm_omni/model_executor/models/longcat_next/
modeling_longcat_next.py`, no vLLM changes): reapply the scale in
vllm-omni's own `load_weights`, gated by vllm-omni's own guard, immediately
after `AutoWeightsLoader.load_weights()` returns — guaranteed to run after
every one of the fork's internal `load_weights` invocations has completed
and the real weights are in their final place, regardless of how many
internal passes happened upstream. Deliberately kept out of the vLLM fork
entirely: the bug is in a generic, shared-across-many-DeepSeek-family-models
class that has no reason to know about a LongCat-specific quirk, and fixing
it downstream in vllm-omni means no vLLM patch to maintain.

**Verified fix, layer by layer**:
```
Reference top-10: ids=[23958, 7600, 9400, 44408, 49488, 2909, 25044, 56992, 31553, 78636]
                  span ~45.4
Port (post-fix):  ids=[23958, 7600, 44408, 49488, 9400, 56992, 25044, 2909, 31553, 60059]
                  span ~45.75
```
9/10 top tokens match exactly, values within ~1%, span within 1% of the
reference. The one prior "confirmed" side-track — that n-gram embedding
fusion might itself be buggy, based on tracing `embed_input_ids` down to a
plain `self.embed_tokens()` call with no fusion — turned out to be a red
herring: `embed_input_ids` is an unused `SupportsMultiModal` interface
fallback. The actual path (`preprocess()` → `_span_oe_ids` →
`ngram_embeddings.embed_batched`) was correct the whole time; a debug audit
already built into that function (`[longcat-ngram]`) confirmed its computed
`oe_ids` once a genuinely clean run captured them.

**End-to-end confirmation**: reran text/image/audio with simple English
prompts post-fix (`scripts/run_english_demo.py`, not checked in — repo's
`scripts/` is gitignored). All three now pass on content, not just pipeline
shape:
- Text: coherent, on-topic, factually correct multi-paragraph answer
  (previously: grammatically broken filler).
- Image: correct prompt-matching content (previously: unrelated scenes, e.g.
  "a bed and some fingers").
- Audio: correct-length clip (5.58s for a comparable sentence), no
  repetition loop, no `EngineDeadError` timeout (previously: either
  ~100s+ of repeated segments, or an outright hang/crash).

## 12. Custom Docker image for fast pod respin

Building the whole toolchain from scratch (§1-§5) costs ~15-20 min of
GPU-billed idle time on every fresh pod, worse on every community-cloud
preemption. A prebuilt image collapses that to the checkpoint download only.

`Dockerfile` (repo root) bakes in: `uv`, both repos at their pinned
branches, the vllm editable install (`--torch-backend=cu130` pinned
explicitly — `auto` relies on detecting a live GPU/driver, which fails
silently under the cross-arch emulation needed to build this on a Mac, and
silently falls back to CPU-only torch), vllm-omni editable + requirements,
and flash-attn. The checkpoint itself is deliberately **not** baked in —
stays on the runtime `/workspace` volume.

Build/push (from a Mac, cross-compiling for the pod's amd64):
```bash
docker buildx build --platform linux/amd64 --push \
    -t <dockerhub-user>/longcat-next-dev:latest -f Dockerfile .
```

Then create the pod straight from that image instead of the base
`runpod/pytorch` one, with a `PUBLIC_KEY` env var set (see §11's note on
why) so SSH works immediately:
```
imageName: <dockerhub-user>/longcat-next-dev:latest
env: {"PUBLIC_KEY": "<contents of your SSH pubkey>"}
```

Things that broke building this, in order found:
- **flash-attn from source OOMs under cross-arch (QEMU) emulation** on a
  Mac, even at `MAX_JOBS=2` (cutlass/CUTLASS template instantiation is
  memory-hungry even natively; emulation multiplies it further). Fixed by
  using a prebuilt wheel from `https://wheels.astral.sh/simple/cu130/`
  instead (exact match available: `flash-attn==2.8.3.post1`, `cu13.0`,
  `torch.2.11`, `cp311`) — skips compilation entirely.
- `nvidia-cuda-nvcc` + the `lib64`/`libcudart` symlink dance (§1's
  BUILD.md, unchanged) is still needed in the image even without building
  flash-attn from source: flashinfer JIT-compiles its sampling kernel
  lazily on the *first real inference call on the GPU pod*, not at image
  build time, and needs that toolchain present then.
- Overriding the base image's `CMD` (to `["/bin/bash"]`, seemingly harmless)
  silently disables the base image's `/start.sh`, which is what launches
  sshd — the pod comes up but never accepts SSH. Keep `CMD ["/start.sh"]`.
- BuildKit cache mounts (`RUN --mount=type=cache,target=/root/.cache/uv`,
  needs `# syntax=docker/dockerfile:1.7` at the top of the Dockerfile) keep
  `uv`'s package cache out of the pushed image layers entirely — cut push
  size/time by ~30% (928s → 642s) on this image. Ordering layers with the
  most-frequently-changed one (vllm-omni's own clone+install, iterated on
  throughout this session) *last* means changing only that file reuses
  every earlier cached layer (uv, vllm, flash-attn, nvcc) on rebuild.
