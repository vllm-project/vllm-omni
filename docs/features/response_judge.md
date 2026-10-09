# Response Judge Stage

In a realtime voice pipeline, VAD commits a turn at every pause. A
backchannel ("uh-huh", "okay"), a cough, noise or speech addressed to someone
else then runs the whole downstream pipeline: the main model generates a
reply and TTS speaks it.

A **response judge** is an optional stage placed right after the ASR stage. A
small judge model reads the ASR transcript and decides whether the turn needs
a reply. A turn that does not ends at the judge, and the main model and TTS
are not run. Any judge model can be integrated through the stage's model and
decision interfaces, including chat models that answer with one token and
pooling (decision / scoring) models. Already integrated judges are selected
through deployment configuration; new models require adapter code as
described under "Adding a judge".

```text
Audio -> VAD commit -> ASR -> response judge -+-> reply needed:    main model -> TTS
                                              +-> no reply needed: turn ends
```

## Behavior

- **Opt-in.** Only pipelines that declare a `response_judge` stage run it.
  The judged pipeline is a separate deploy config; the original one is
  unchanged and loads no judge model.
- **Reuses the existing no-reply paths.** In a duplex session a rejected turn
  takes the model's existing listen decision: the client receives
  `response.listen`, prewarmed downstream requests are aborted and released,
  and the turn ends. The event carries
  `response.metadata.vllm_omni.listen_source: "response_judge"`, so a client
  can tell a judge rejection from the model's own decision to stay silent
  (AURA reports `"aura_silent"`). The field is optional and its value is an
  open string set by the plugin; clients should ignore values they do not
  know. In turn-based serving the judge's bridge yields no input
  and the request finishes through the orchestrator's existing empty-output
  path, which also aborts any downstream stage that async-chunk already
  prewarmed.
- **Lets the turn through when unsure.** An empty transcript or an answer
  that cannot be read lets the turn through with the original transcript. A
  failing judge stage uses the existing stage failure handling.
- **The main model sees the ASR output unchanged.** The judge's own output is
  never passed downstream.
- **The judge reads only the current transcript**, not the conversation
  history. A rejected turn is not added to the model's history.

## Judge models

The judge is configured on its own stage in the deploy config, under
`hf_overrides.response_judge`:

| `format` | Model | Runner | Decision |
| --- | --- | --- | --- |
| `chat_yes_no` | Any chat model; `ResponseJudgeQwen3ForCausalLM` for Qwen3 | generate (`max_tokens: 1`) | Rejects only when the answer is exactly `reject_label` (default `NO`) |
| `laya` | LAYA decision models, `LayaDecisionModel` | pooling (`task: classify`) | Rejects when P(`reply_option`) < `threshold` |
| `clm` | Contrastive-LM (CLM) encoder + heads, `ClmDecisionModel` | pooling (`task: classify`) | Rejects when P(`reply_option`) < `threshold` |

Options per format:

- `chat_yes_no`: `system_prompt`, `user_template` (with `{transcript}`), and
  `reject_label`. The chat template is rendered with thinking disabled, so
  the first generated token is the answer.
- `laya`: `question_type`, `instructions`, `options` (key -> description),
  `reply_option`, `threshold` and `state_template`. `question_type` defaults
  to the model's `laya_question_type`; if both are set they must agree.
- `clm`: the option projections are fixed when the model directory is
  prepared, and that directory's `config.json` carries the matching
  `response_judge` options (`option_keys`, `reply_option`, `threshold`,
  `instructions` and `state_template`).

`reply_option` may be a list; the probabilities of the listed options are
added.

For `laya` and `clm`, the server refuses to start if `options` is empty,
`reply_option` names an option that is not configured, or `threshold` is not
a number in [0, 1]. Loading also fails if a judge head weight does not have
exactly the shape the model expects.

The pooling judges load from a directory prepared for vLLM:

- **LAYA:** the checkpoint's weights, `config.json` with
  `architectures: [LayaDecisionModel]` and `laya_question_type`, and the
  tokenizer files at the top level of the directory.
- **CLM:** the Qwen3 encoder weights, `clm_head.safetensors` with the state
  head, the option projections (computed once from the option texts) and the
  logit scale, and `config.json` with `architectures: [ClmDecisionModel]`,
  `clm_head`, `clm_num_options` and the `response_judge` options.

## Example: AURA

`vllm_omni/deploy/aura_omni_judged.yaml` serves the `aura_omni_judged`
pipeline:

```text
Qwen3-ASR -> response judge -> AURA -> Qwen3-TTS Talker -> Code2Wav
```

It uses Qwen3-1.7B as the judge:

```yaml
  - stage_id: 1
    model: Qwen/Qwen3-1.7B
    hf_overrides:
      response_judge:
        format: chat_yes_no
    default_sampling_params:
      temperature: 0.0
      max_tokens: 1
```

`aura_omni_judged_laya.yaml` and `aura_omni_judged_clm.yaml` inherit it with
`base_config` and replace only stage 1 with a pooling judge. Serve
`aura_omni.yaml` to run AURA without a judge.

## Adding a judge

- **Another chat model:** the `aura_omni_judged` pipeline defaults stage 1
  to `model_arch: ResponseJudgeQwen3ForCausalLM`, so a non-Qwen3 judge must
  also set `model_arch`. The omni runner passes extra keyword arguments to
  `forward` and `compute_logits`; a model class that does not accept them
  needs a thin subclass like `ResponseJudgeQwen3ForCausalLM`, registered in
  `vllm_omni/model_executor/models/registry.py`:

    ```yaml
      - stage_id: 1
        model: <chat model>
        model_arch: <registered judge class>
        hf_overrides:
          response_judge:
            format: chat_yes_no
            system_prompt: "..."
        default_sampling_params:
          temperature: 0.0
          max_tokens: 1
    ```

- **Another pooling model:** add a pooling model class that returns one logit
  per option, and a format entry in
  `vllm_omni/model_executor/stage_input_processors/response_judge.py` that
  builds its prompt and reads its decision.
- **Another pipeline:** add a stage with `model_stage="response_judge"` right
  after the ASR stage. Build its input with `judge_input(...)`, and wrap the
  pipeline's original ASR -> main-model bridge with `after_judge(...)`. A
  duplex plugin that addresses stages by number must account for the extra
  stage (AURA uses a role-based stage layout).

## Latency

The judge runs once per committed turn, on the critical path before the main
model. Measured on AURA with the LAYA judge (one RTX 4080 SUPER, compact
profile, one session at a time), most of its time was the model call, so:

- **Give the judge CUDA graphs that cover its prompt.** vLLM's default capture
  sizes stop at `2 * max_num_seqs`, while a judge prompt without a prefix cache
  is one prefill of tens of tokens. `aura_omni_judged_laya.yaml` and
  `aura_omni_judged_clm.yaml` therefore set
  `compilation_config.cudagraph_capture_sizes`. With it the LAYA forward pass
  dropped from about 18 ms to about 8 ms in the pipeline; a standalone CLM
  forward pass after an idle gap dropped from about 39 ms to about 31 ms. For the
  Qwen3 judge, larger capture sizes showed no gain in a standalone test, so
  `aura_omni_judged.yaml` keeps the defaults.
- **Poll stage outputs event-driven.** `VLLM_OMNI_EVENT_DRIVEN_ORCH=1` lets
  the orchestrator pick up each stage's output as it arrives instead of on a
  1 ms poll of every stage (this applies to all stages, not only the judge).
  It cut the judge stage's output pickup from about 4 ms to about 1 ms.
- **Expect host effects after idle gaps.** On a host with the `powersave` CPU
  governor, a judge call after a pause in the conversation was slower than
  back to back: about 1.4x for CLM, 3x for Qwen3 and up to 4x for LAYA. This
  is consistent with the CPU clocking back up; it was not compared against
  another governor.

With the first two settings, the added latency (from forwarding the ASR
output to submitting the main model, judge on vs. off, 16 questions per arm)
went from 43.0 and 44.5 ms to 26.1 and 28.5 ms p50 in two repeats.

## Limitations

- The judge only sees the transcript, so it cannot use acoustic cues such as
  laughter or who is speaking.
- Judge quality depends on the model and the prompt. Zero-shot decision
  models (LAYA, CLM) are weak at "does this need a reply"; fine-tuning their
  heads is future work.
- Short answers to a question from the previous turn ("seven", "tomorrow")
  and interruptions ("wait") look like backchannels without the previous
  turn. The LAYA example prompt still rejects some of them.
- A request id carries one judged turn at a time; a request must not send a
  new transcript to the judge before the previous one has been decided.
- In an auto-response duplex session, a turn that follows a terminal listen
  (a judge rejection, or the model's own silence) may only get
  `response.listen`. This is a session issue outside the judge; a fix is
  tracked separately.
- The judge adds its own latency and GPU memory to every committed turn.
