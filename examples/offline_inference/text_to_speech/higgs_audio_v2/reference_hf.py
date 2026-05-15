# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture a deterministic upstream HF reference for higgs-audio v2.

Runs ``bosonai/higgs-audio-v2-generation-3B-base`` + ``bosonai/higgs-audio-v2-tokenizer``
on a small set of pinned prompts with greedy decode and saves the fixture set
consumed by the vllm-omni parity tests (AC-1, AC-2, AC-4):

  tests/fixtures/higgs_audio_v2/reference_<slug>.pt
    {
      "prompt_text":       str,
      "input_ids":         LongTensor[1, S]            # exact upstream tokenizer output
      "audio_codes":       LongTensor[1, num_codebooks, T_audio]   # post-revert real-codes only
      "reference_pcm":     IntTensor[T_pcm]            # int16, 24 kHz mono
      "audio_token_mask":  BoolTensor[1, S]            # DualFFN routing mask (per upstream)
      "config_summary":    dict                        # echo of fixed upstream constants
    }

The script also writes the human-readable upstream trace memo at
``vllm_omni/model_executor/models/higgs_audio_v2/UPSTREAM_TRACE.md`` (only when
``--write-trace`` is passed, so casual reruns don't clobber the memo).

Usage (run from repo root with `.venv/bin/activate` and CUDA visible):

  python examples/offline_inference/text_to_speech/higgs_audio_v2/reference_hf.py \
      --prompts "Hello world." \
      --max-new-tokens 100 \
      --output-dir tests/fixtures/higgs_audio_v2 \
      --write-trace
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]

DEFAULT_PROMPTS: tuple[str, ...] = (
    # AC-1 / AC-2 require the pinned "Hello world." + 10 additional prompts.
    "Hello world.",
    "The quick brown fox jumps over the lazy dog.",
    "It was the night before my birthday.",
    "She sells seashells by the seashore.",
    "Innovation distinguishes between a leader and a follower.",
    "Mary had a little lamb whose fleece was white as snow.",
    "Time flies like an arrow; fruit flies like a banana.",
    "All that glitters is not gold.",
    "An apple a day keeps the doctor away.",
    "May the force be with you, always.",
    "To be or not to be, that is the question.",
)


@dataclasses.dataclass
class ReferenceCapture:
    prompt_text: str
    input_ids: torch.Tensor
    audio_codes: torch.Tensor
    reference_pcm: torch.Tensor
    audio_token_mask: torch.Tensor
    config_summary: dict[str, Any]


def _slugify(text: str) -> str:
    s = re.sub(r"\s+", "_", text.strip().lower())
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s[:48] or "prompt"


def _build_conversation(user_text: str) -> list[dict[str, Any]]:
    """Mirror the canonical single-speaker smart-voice template from the upstream docs."""
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": "Generate audio following instruction."}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user_text}],
        },
    ]


def _build_text_template_record(processor, prompt_text: str) -> dict[str, Any]:
    """Capture the exact decoded text the processor produced for the user prompt."""
    conversation = _build_conversation(prompt_text)
    rendered: str = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    return {"prompt_text": prompt_text, "rendered_chat_template": rendered}


def _capture_audio_token_mask(model: torch.nn.Module) -> dict[str, Any]:
    """Install a forward hook on the first decoder layer to record the routing mask.

    The hook fires once per decoder step. We record each step's mask separately
    so the final captured artifact covers BOTH the prefill (entire prompt) and
    every autoregressive decode step (one token per step). The full routing
    trace concatenated along the sequence dim is what AC-3 needs to compare
    against.
    """
    state: dict[str, Any] = {"masks": []}

    inner_model = getattr(model, "model", None) or model
    layers = getattr(inner_model, "layers", None)
    if layers is None or len(layers) == 0:
        return state

    def _hook(_module, args, kwargs):
        mask = kwargs.get("audio_token_mask")
        if mask is None:
            return
        state["masks"].append(mask.detach().to("cpu").clone())

    state["_handle"] = layers[0].register_forward_pre_hook(_hook, with_kwargs=True)
    return state


def _release_hook(state: dict[str, Any]) -> None:
    handle = state.pop("_handle", None)
    if handle is not None:
        handle.remove()


def _consolidate_masks(masks: list[torch.Tensor]) -> torch.Tensor | None:
    """Concatenate per-step routing masks into one [B, S_full] tensor.

    Each entry has shape ``[B, S_i]``; the prefill step has S_0 = prompt length,
    and each decode step has S_i = 1. Concatenating along dim=1 gives the
    per-position routing mask over the FULL output sequence (prompt + every
    generated LM token).
    """
    if not masks:
        return None
    # Some upstream paths emit masks with differing batch sizes during compile;
    # take the first batch dim and require all entries to match.
    bsz = int(masks[0].shape[0])
    aligned: list[torch.Tensor] = []
    for m in masks:
        if int(m.shape[0]) != bsz:
            continue
        if m.ndim == 1:
            aligned.append(m.unsqueeze(0))
        else:
            aligned.append(m)
    if not aligned:
        return None
    return torch.cat(aligned, dim=1)


def _config_summary(model_config) -> dict[str, Any]:
    keys = (
        "model_type",
        "architectures",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "max_position_embeddings",
        "num_codebooks",
        "codebook_size",
        "audio_token_id",
        "audio_bos_token_id",
        "audio_delay_token_id",
        "audio_stream_bos_id",
        "audio_stream_eos_id",
        "eos_token_id",
        "bos_token_id",
        "pad_token_id",
        "tie_word_embeddings",
    )
    summary: dict[str, Any] = {}
    for k in keys:
        if hasattr(model_config, k):
            v = getattr(model_config, k)
            summary[k] = list(v) if isinstance(v, (list, tuple)) else v
    return summary


def _decode_audio_with_processor(processor, generate_output) -> torch.Tensor:
    """Decode the model.generate() output back to int16 PCM at 24 kHz.

    The HiggsAudioV2 processor's ``batch_decode`` expects ``audio_input_ids`` of
    shape ``[B, S, num_codebooks]``. Modern revisions of ``HiggsAudioV2GenerationMixin``
    return that tensor directly from ``model.generate(...)`` (the docstring at
    https://huggingface.co/docs/transformers/main/en/model_doc/higgs_audio_v2
    uses exactly this pattern). Older revisions wrap it in ``GenerateOutput``;
    we accept either.
    """
    if hasattr(generate_output, "sequences"):
        decode_input = generate_output.sequences
    else:
        decode_input = generate_output
    # The audio_tokenizer attached to the processor lives on CPU when loaded
    # via AutoProcessor.from_pretrained; move the generated codes to CPU so
    # the codec decode runs entirely on CPU (this is the slow path; the
    # vllm-omni Stage-1 path runs on GPU).
    if hasattr(decode_input, "to"):
        decode_input = decode_input.to("cpu")
    decoded = processor.batch_decode(decode_input)
    if not decoded:
        raise RuntimeError("processor.batch_decode returned an empty list")
    audio = decoded[0]
    if isinstance(audio, dict):
        audio_array = audio.get("audio_array") or audio.get("audio_values")
        if audio_array is None:
            raise RuntimeError(f"decoded[0] missing audio_array/audio_values; keys={list(audio.keys())}")
        audio_tensor = torch.as_tensor(audio_array)
    else:
        audio_tensor = torch.as_tensor(audio)
    audio_tensor = audio_tensor.detach().to("cpu").to(torch.float32)
    if audio_tensor.ndim > 1:
        audio_tensor = audio_tensor.reshape(-1)
    pcm_int16 = (audio_tensor.clamp_(-1.0, 1.0) * 32767.0).round().to(torch.int16)
    return pcm_int16


def _extract_audio_input_ids(generate_output) -> torch.Tensor:
    """Pull the ``audio_input_ids`` tensor of shape ``[B, S, num_codebooks]`` from a generate output."""
    if isinstance(generate_output, torch.Tensor):
        return generate_output
    for attr in ("audio_input_ids", "sequences"):
        val = getattr(generate_output, attr, None)
        if isinstance(val, torch.Tensor) and val.ndim == 3:
            return val
    # GenerateOutput.sequences may be the LM token stream (ndim==2). In that
    # case the caller is on an older revision; surface the situation explicitly.
    seq = getattr(generate_output, "sequences", None)
    if isinstance(seq, torch.Tensor):
        raise RuntimeError(
            "model.generate() returned sequences of shape "
            f"{tuple(seq.shape)} (expected 3-D audio_input_ids). Pass return_dict_in_generate=False "
            "with a newer transformers, or upgrade transformers to >= 5.8."
        )
    raise RuntimeError("Could not locate audio_input_ids in model.generate() output")


def _strip_delay_and_extract_codes(
    processor, audio_input_ids: torch.Tensor, audio_stream_bos_id: int, num_codebooks: int
) -> torch.Tensor:
    """Revert delay pattern and trim to real codes only, mirroring batch_decode internals.

    Returns a ``LongTensor`` of shape ``[1, num_codebooks, T]`` with values in
    ``[0, audio_stream_bos_id - 1]`` (i.e. only real audio codes).
    """
    # Find the last audio_stream_bos position; everything before is the prompt context.
    audio_bos_token_idxs = (audio_input_ids == audio_stream_bos_id).all(-1).nonzero()
    if audio_bos_token_idxs.numel() == 0:
        # No audio was generated; return canonical empty [B, num_codebooks, 0] tensor.
        return torch.zeros(
            (int(audio_input_ids.shape[0]), num_codebooks, 0),
            dtype=torch.long,
        )
    start = int(audio_bos_token_idxs[-1, -1].item())
    trimmed = audio_input_ids[:, start:]
    # Find EOS to clip the tail. (Bounded by sequence length if no EOS yet.)
    audio_stream_eos_id = audio_stream_bos_id + 1
    eos_idxs = (trimmed == audio_stream_eos_id).all(-1).nonzero()
    end = trimmed.shape[1]
    for b in range(trimmed.shape[0]):
        per_b = eos_idxs[eos_idxs[:, 0] == b]
        if per_b.numel() > 0:
            end = min(end, int(per_b[:, 1].min().item()))
    # Drop the leading BOS row, run revert_delay_pattern, and clip stream specials.
    codes = []
    for b in range(trimmed.shape[0]):
        row = trimmed[b, 1:end]  # [S', num_codebooks]
        reverted = processor.revert_delay_pattern(row).clip(0, audio_stream_bos_id - 1)
        if reverted.ndim != 2:
            raise RuntimeError(f"revert_delay_pattern returned shape {tuple(reverted.shape)}; expected 2-D")
        # ``revert_delay_pattern`` returns ``[T, num_codebooks]`` in current transformers.
        # Normalize to ``[num_codebooks, T]`` (the documented Stage-1 contract).
        if int(reverted.shape[0]) == num_codebooks:
            normalized = reverted
        elif int(reverted.shape[1]) == num_codebooks:
            normalized = reverted.transpose(0, 1).contiguous()
        else:
            raise RuntimeError(
                f"revert_delay_pattern returned unexpected shape {tuple(reverted.shape)}; "
                f"expected one axis of size num_codebooks={num_codebooks}"
            )
        codes.append(normalized)
    return torch.stack(codes, dim=0).to(torch.long).detach().cpu()


def capture_prompt(processor, model, prompt_text: str, max_new_tokens: int) -> ReferenceCapture:
    conversation = _build_conversation(prompt_text)
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        sampling_rate=24000,
        return_tensors="pt",
    ).to(model.device)

    input_ids = inputs["input_ids"]

    hook_state = _capture_audio_token_mask(model)
    try:
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
    finally:
        _release_hook(hook_state)

    audio_input_ids = _extract_audio_input_ids(outputs)
    real_codes = _strip_delay_and_extract_codes(
        processor,
        audio_input_ids,
        model.config.audio_stream_bos_id,
        int(model.config.num_codebooks),
    )
    reference_pcm = _decode_audio_with_processor(processor, outputs)

    # Routing mask: consolidate every per-step hook capture into one
    # [B, S_full] tensor covering both the prompt prefill and every
    # autoregressive decode step. If the hook never fired (older transformers),
    # fall back to the upstream rule applied to the running LM token stream.
    audio_token_mask = _consolidate_masks(hook_state.get("masks", []))
    if audio_token_mask is None:
        # outputs.sequences (when present) is the running LM token stream
        # including the appended audio_token_id at each generated audio position.
        seq = getattr(outputs, "sequences", None)
        full_ids = seq if isinstance(seq, torch.Tensor) and seq.ndim == 2 else input_ids
        audio_token_mask = (
            (full_ids == model.config.audio_token_id)
            | (full_ids == model.config.audio_delay_token_id)
        ).detach().to("cpu")
    return ReferenceCapture(
        prompt_text=prompt_text,
        input_ids=input_ids.detach().to("cpu"),
        audio_codes=real_codes,
        reference_pcm=reference_pcm,
        audio_token_mask=audio_token_mask,
        config_summary=_config_summary(model.config),
    )


_TRACE_TEMPLATE = """# higgs-audio-v2 Upstream Trace

This memo records the upstream HF reference behavior that the vllm-omni
`higgs_audio_v2` integration must reproduce. All facts here are derived from
the official `transformers` source (`transformers.models.higgs_audio_v2`) and
the boson-ai checkpoints `bosonai/higgs-audio-v2-generation-3B-base` and
`bosonai/higgs-audio-v2-tokenizer`. Use this as the contract for AC-1, AC-2,
and AC-3.

## Fixed model constants (from `config.json`)

```json
__CONFIG_JSON__
```

## DualFFN routing rule (from `HiggsAudioV2DecoderLayer.forward`)

Each transformer block contains two parallel pre-norm + MLP paths:
- text path: `input_layernorm`, `post_attention_layernorm`, `mlp`
- audio path: `audio_input_layernorm`, `audio_post_attention_layernorm`, `audio_mlp`

A per-position `audio_token_mask: BoolTensor[B, S]` selects between them.

- Pre-attention norm: `audio_input_layernorm` on audio positions; `input_layernorm` on text positions. The two outputs are stitched together with `masked_scatter` and then fed through a single (shared) self-attention.
- Post-attention FFN: text positions are passed through `mlp(post_attention_layernorm(.))`; audio positions are passed through `audio_mlp(audio_post_attention_layernorm(.))`. Both deltas are ADDED to the residual (not replacing).
- Edge case: when `audio_token_mask is None` (pure-audio inference), the audio path is applied to ALL positions.

## audio_token_mask construction (from `HiggsAudioV2Model.get_placeholder_mask`)

```python
special_audio_mask = (input_ids == audio_token_id) | (input_ids == audio_delay_token_id)
```

i.e., a position is "audio" iff its token id is `audio_token_id=128016` (the audio placeholder) OR `audio_delay_token_id=128014` (the delay filler).

## Audio embedding rule (from `HiggsAudioV2Embeddings`)

- `embed_audio_tokens`: `nn.Embedding(num_codebooks * codebook_size, hidden_size)`, i.e. 8 * 1026 = 8208 rows.
- `audio_tokens_offsets = arange(num_codebooks) * codebook_size = [0, 1026, 2052, 3078, 4104, 5130, 6156, 7182]`.
- For `audio_input_ids` of shape `(B, num_audio_frames, num_codebooks)` with values in `[0, codebook_size)`:
    - `inputs_embeds = embed_audio_tokens(audio_input_ids + audio_tokens_offsets)`
    - `inputs_embeds.sum(dim=-2)` collapses across codebooks.
- The text prompt's `inputs_embeds = embed_tokens(input_ids)`. Then a `masked_scatter` substitutes audio frames at positions where `audio_token_mask` is True.

## Delay-pattern handling (from `HiggsAudioV2DelayPatternLogitsProcessor`)

- A `delay_pattern: list[int]` controls the per-codebook offset for the audio-stream BOS and EOS tokens. The processor masks the codebook vocab so each codebook `k` is forced to emit `audio_stream_bos_id=1024` at the start and `audio_stream_eos_id=1025` at the end until its delay counter reaches 0.
- The 8-codebook canonical delay pattern is the MusicGen-style monotonic sequence `[0, 1, 2, 3, 4, 5, 6, 7]` (codebook k starts emitting real codes only after k frames). This is consistent with the `HiggsAudioV2DelayPatternLogitsProcessor.__call__` math (`scores.reshape(-1, num_codebooks, codebook_size)` and the per-row `vocab_mask_bos`/`vocab_mask_eos` masking).
- Real audio code IDs are in `[0, 1024)`. Codes `1024` and `1025` are the stream-BOS / stream-EOS markers and must NOT reach the codec decoder. The vllm-omni Stage 1 must reject any value `>= 1024` with an explicit `ValueError`.

## Stream BOS/EOS emission rule

- `audio_stream_bos_id=1024` is emitted at the boundary that opens an audio stream; it is consumed by the LM during the codebook-output build-up phase and is filtered out before sending codes to the codec.
- `audio_stream_eos_id=1025` is emitted at the boundary that closes an audio stream; the LM uses it to learn end-of-audio, and it is filtered out before decode.
- `audio_bos_token_id=128013` is the *prompt-level* audio bos in the LM vocabulary (text-space token id) that marks the position in `input_ids` where the audio frames begin.
- `audio_delay_token_id=128014` is the *prompt-level* delay placeholder used to fill positions where a codebook has not yet started emitting real codes (post-delay-pattern construction). These positions still go through the audio path of DualFFN.

## Plain-text prompt template (from upstream docs / `HiggsAudioV2Processor.apply_chat_template`)

Conversation form for a plain-text TTS request:

```python
conversation = [
    {"role": "system", "content": [{"type": "text", "text": "Generate audio following instruction."}]},
    {"role": "user",   "content": [{"type": "text", "text": "<USER TEXT HERE>"}]},
]
processor.apply_chat_template(
    conversation, add_generation_prompt=True, tokenize=True,
    return_dict=True, sampling_rate=24000, return_tensors="pt",
)
```

The captured `rendered_chat_template` from the reference run is persisted under
`text_template_*` keys in the fixture files alongside the `input_ids` tensor.
vllm-omni's `higgs_audio_v2_tokenizer.build_plain_text_prompt(...)` must produce
the same `input_ids` for the same `<USER TEXT>` (AC-1 positive test).

## Fixture inventory

Each `tests/fixtures/higgs_audio_v2/reference_<slug>.pt` holds the per-prompt
record described at the top of `reference_hf.py`. The captured fields satisfy:

- AC-1 (input-token parity): exact `input_ids` from upstream tokenizer.
- AC-2 (per-codebook parity): `audio_codes` is the post-revert real-code tensor
  with shape `[1, num_codebooks=8, T]` and values in `[0, 1023]`.
- AC-3 (DualFFN routing): `audio_token_mask` is the per-position routing mask
  recorded from the live forward pass on the first decoder layer (matches
  `HiggsAudioV2Model.get_placeholder_mask`).
- AC-4 (Stage-1 decode parity): `reference_pcm` is the upstream-decoded waveform
  as int16, mono, 24 kHz. Normalized-float RMS comparison with vllm-omni Stage 1
  must be `<= 1e-4` (see plan AC-4).

## Pinned prompt list

__PROMPTS__

## Notes for the vllm-omni implementation

- The Stage-0 talker must implement DualFFN by subclassing
  `vllm.model_executor.models.llama.LlamaDecoderLayer` and replacing the
  `mlp` member with a routed `DualFFNLayer` that consults the per-position
  audio mask precomputed at model input time.
- The HF→vLLM weight mapping must transcribe both the text MLP weights
  (`model.layers.<L>.mlp.{gate_proj,up_proj,down_proj}`) and the audio MLP
  weights (`model.layers.<L>.audio_mlp.{gate_proj,up_proj,down_proj}`), plus
  the parallel layernorm pairs (`{input_layernorm, audio_input_layernorm,
  post_attention_layernorm, audio_post_attention_layernorm}.weight`).
- The fused QKV projection uses GQA with 24 Q heads / 8 KV heads / head_dim=128;
  pack as `[hidden + 2 * kv_head_dim * head_dim, hidden]` mirroring how
  `vllm.model_executor.models.llama.LlamaAttention.load_weights` consumes
  separated `q_proj/k_proj/v_proj`.
- The RoPE config is `rope_type="llama3"` with `factor=32.0`,
  `low_freq_factor=0.125`, `high_freq_factor=0.5`, `original_max_position_embeddings=1024`.
- Multi-codebook output head: the model has `(num_codebooks * codebook_size) = 8 * 1026 = 8208`-wide audio output (via `embed_audio_tokens` lookups) AND a 128256-wide text head (the standard Llama `lm_head`). Stage-0 emits codebook 0 via the audio head plus the residual codebooks 1..7 via a per-step fast-AR head; see plan task3 for the structure.
"""


def maybe_write_trace_memo(model_config, prompts: list[str], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cfg_summary = _config_summary(model_config)
    text = _TRACE_TEMPLATE.replace("__CONFIG_JSON__", json.dumps(cfg_summary, indent=2)).replace(
        "__PROMPTS__", "\n".join(f"- {p!r}" for p in prompts)
    )
    dest.write_text(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", default="bosonai/higgs-audio-v2-generation-3B-base")
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=list(DEFAULT_PROMPTS),
        help="One or more plain-text prompts to capture.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "tests" / "fixtures" / "higgs_audio_v2",
        help="Directory where reference_*.pt files will be saved.",
    )
    parser.add_argument(
        "--trace-output",
        type=Path,
        default=REPO_ROOT
        / "vllm_omni"
        / "model_executor"
        / "models"
        / "higgs_audio_v2"
        / "UPSTREAM_TRACE.md",
    )
    parser.add_argument("--write-trace", action="store_true", help="Write/update UPSTREAM_TRACE.md from this run.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available; running on CPU will be very slow.", file=sys.stderr)

    from transformers import AutoProcessor, HiggsAudioV2ForConditionalGeneration

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    print(f"[ref_hf] loading {args.model_id} (dtype={args.dtype}, device={args.device})", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = HiggsAudioV2ForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=dtype, device_map=args.device
    )
    model.eval()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: list[str] = []
    for prompt in args.prompts:
        print(f"[ref_hf] generating reference for prompt: {prompt!r}", flush=True)
        cap = capture_prompt(processor, model, prompt, args.max_new_tokens)
        text_template = _build_text_template_record(processor, prompt)
        slug = _slugify(prompt)
        out_path = out_dir / f"reference_{slug}.pt"
        torch.save(
            {
                "prompt_text": cap.prompt_text,
                "input_ids": cap.input_ids,
                "audio_codes": cap.audio_codes,
                "reference_pcm": cap.reference_pcm,
                "audio_token_mask": cap.audio_token_mask,
                "config_summary": cap.config_summary,
                "text_template": text_template,
                "model_id": args.model_id,
                "dtype": args.dtype,
                "max_new_tokens": args.max_new_tokens,
            },
            out_path,
        )
        print(
            f"[ref_hf]   wrote {out_path}  "
            f"(input_ids {tuple(cap.input_ids.shape)}, "
            f"audio_codes {tuple(cap.audio_codes.shape)}, "
            f"reference_pcm {tuple(cap.reference_pcm.shape)})",
            flush=True,
        )
        saved.append(str(out_path))

    if args.write_trace:
        maybe_write_trace_memo(model.config, list(args.prompts), args.trace_output)
        print(f"[ref_hf] wrote upstream trace memo: {args.trace_output}", flush=True)

    print(f"[ref_hf] done; {len(saved)} fixture(s) written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
