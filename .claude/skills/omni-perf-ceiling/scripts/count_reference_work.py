#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Count per-stage work on a checkpoint's reference (transformers) implementation.

The calibrated roofline needs reference work counts, separate from the
serving engine under test: FLOPs, weight bytes streamed per step and KV bytes
per token. This script builds the model from the checkpoint's own modeling code
(``trust_remote_code``) on the meta device and counts with ``FlopCounterMode``,
so no weights are loaded and no GPU is used. Audio decoders that are not
``nn.Module`` children of the reference model are counted by a recipe that runs
the reference streaming path on CPU with the real weights.

Decoders, ``--decoder NAME[@STAGE]=ATTR[+HEAD_ATTR...]``:

- ATTR is the backbone's attribute path on the reference model (``llm``,
  ``tts.model``); it must accept ``inputs_embeds`` and ``past_key_values``.
- HEAD_ATTR modules run on the last hidden state every step (``tts.head_code``).
- Prefill FLOPs per token are averaged over ``--prompt-tokens``; decode FLOPs
  per token are one step at ``--context NAME=TOKENS`` (default prompt tokens).
- Streamed bytes are the parameters a step reads: everything except embedding
  tables, which are gathered by row (a table tied to an output head is kept).
- KV bytes per token are read from the reference ``DynamicCache``.

Code2Wav recipes, ``--code2wav RECIPE[@STAGE]=ASSET_DIR``:

- ``minicpmo-token2wav``: MiniCPM-o 4.5 ``stepaudio2`` Token2wav, streamed as
  the reference chat path does: chunks of ``--chunk-tokens`` codec tokens plus
  the flow's look-ahead, flow attention cache trimmed to prompt + 100 frames,
  and an 8-frame HiFT mel overlap.

With ``--tokens-per-audio-s`` (and optionally ``--mean-context``,
``--uncached-tokens``, ``--fill-steps``) the script prints the matching
``ceiling_report.py`` flags. Peaks are not used here; dtypes default to the
reference's (checkpoint dtype for decoders, fp32 for Token2wav) and
``--dtype NAME=DTYPE`` sets the deployment's compute mode and default decoder
storage dtype (TF32/HF32 use FP32 storage). ``--weight-dtype`` and ``--kv-dtype``
override decoder storage independently when it differs from computation.
"""

import argparse
import importlib.util
import json
import sys
import time
import types
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

GIB = 2**30
# Keys of measure_peaks.py "matmul_tflops".
PEAK_DTYPES = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}
STORAGE_DTYPES = {name: dtype for dtype, name in PEAK_DTYPES.items()}
COMPUTE_STORAGE = {**STORAGE_DTYPES, "tf32": torch.float32, "hf32": torch.float32}


def resolve(root: torch.nn.Module, path: str) -> torch.nn.Module:
    module = root
    for part in path.split("."):
        module = getattr(module, part)
    return module


def head_modules(module: torch.nn.Module) -> list[torch.nn.Module]:
    return list(module) if isinstance(module, torch.nn.ModuleList) else [module]


def streamed_bytes(modules: list[torch.nn.Module]) -> tuple[int, int]:
    """Parameter bytes read per step, and all parameter bytes, without double counting shared tensors."""
    params = {id(p): p for m in modules for p in m.parameters()}
    linear = {
        id(p)
        for m in modules
        for sub in m.modules()
        if not isinstance(sub, torch.nn.Embedding)
        for p in sub.parameters(recurse=False)
    }
    gathered = {
        id(p)
        for m in modules
        for sub in m.modules()
        if isinstance(sub, torch.nn.Embedding)
        for p in sub.parameters(recurse=False)
    }
    total = sum(p.numel() * p.element_size() for p in params.values())
    skipped = sum(params[i].numel() * params[i].element_size() for i in gathered - linear)
    return total - skipped, total


def forward_kwargs(module: torch.nn.Module) -> dict:
    # transformers generate() computes logits for the last position only.
    names = module.forward.__code__.co_varnames
    return {"logits_to_keep": 1} if "logits_to_keep" in names else {}


def count_decoder(
    model: torch.nn.Module,
    spec: str,
    prompt_tokens: int,
    context: int,
    *,
    compute_dtype: str | None = None,
    weight_dtype: str | None = None,
    kv_dtype: str | None = None,
) -> dict:
    """Count decoder work at deployment precision on meta tensors.

    Cast the selected backbone and heads before counting their streamed bytes.
    KV storage overrides price cache tensor shapes without changing the reference
    attention implementation. Unknown compute/storage types are rejected.
    """
    from transformers import DynamicCache

    if compute_dtype is not None and compute_dtype not in COMPUTE_STORAGE:
        raise SystemExit(f"unsupported decoder compute dtype {compute_dtype!r}; use {sorted(COMPUTE_STORAGE)}")
    for label, value in (("weight", weight_dtype), ("kv", kv_dtype)):
        if value is not None and value not in STORAGE_DTYPES:
            raise SystemExit(f"unsupported {label} storage dtype {value!r}; use {sorted(STORAGE_DTYPES)}")
    backbone_path, *head_paths = spec.split("+")
    backbone = resolve(model, backbone_path)
    heads = [h for path in head_paths for h in head_modules(resolve(model, path))]
    reference_dtype = next(backbone.parameters()).dtype
    storage = STORAGE_DTYPES[weight_dtype] if weight_dtype else COMPUTE_STORAGE.get(compute_dtype)
    if storage is not None:
        for module in [backbone, *heads]:
            module.to(dtype=storage)
    hidden = backbone.config.hidden_size
    dtype = next(backbone.parameters()).dtype
    kwargs = forward_kwargs(backbone)

    def step(embeds: torch.Tensor, cache) -> None:
        out = backbone(inputs_embeds=embeds, past_key_values=cache, use_cache=True, **kwargs)
        for head in heads:
            head(out[0][:, -1:])

    with torch.no_grad(), torch.device("meta"):
        cache = DynamicCache()
        with FlopCounterMode(display=False) as counter:
            step(torch.empty(1, prompt_tokens, hidden, dtype=dtype), cache)
        prefill = counter.get_total_flops()
        kv_element_size = torch.finfo(STORAGE_DTYPES[kv_dtype]).bits // 8 if kv_dtype else None
        kv_bytes = sum(
            t.numel() * (kv_element_size or t.element_size())
            for layer in cache.layers
            for t in (layer.keys, layer.values)
        )
        cache = DynamicCache()
        step(torch.empty(1, context, hidden, dtype=dtype), cache)
        with FlopCounterMode(display=False) as counter:
            step(torch.empty(1, 1, hidden, dtype=dtype), cache)
        decode = counter.get_total_flops()
    streamed, total = streamed_bytes([backbone, *heads])
    return {
        "attr": spec,
        "reference_dtype": PEAK_DTYPES.get(reference_dtype, str(reference_dtype).removeprefix("torch.")),
        "dtype": compute_dtype or PEAK_DTYPES.get(dtype, str(dtype).removeprefix("torch.")),
        "weight_dtype": PEAK_DTYPES.get(dtype, str(dtype).removeprefix("torch.")),
        "kv_dtype": kv_dtype or PEAK_DTYPES.get(dtype, str(dtype).removeprefix("torch.")),
        "param_gib": total / GIB,
        "streamed_gib_per_step": streamed / GIB,
        "prefill_gflop_per_token": prefill / prompt_tokens / 1e9,
        "prefill_tokens": prompt_tokens,
        "decode_gflop_per_token": decode / 1e9,
        "decode_context": context,
        "kv_bytes_per_token": kv_bytes / prompt_tokens,
    }


def _import_stepaudio2_modules() -> None:
    """Import Token2wav's flow and HiFT without ``stepaudio2/__init__``, which needs torchaudio and onnx models."""
    spec = importlib.util.find_spec("stepaudio2")
    if spec is None:
        raise SystemExit("minicpmo-token2wav needs the stepaudio2 package (pip install minicpmo-utils)")
    if "stepaudio2" not in sys.modules or not hasattr(sys.modules["stepaudio2"], "Token2wav"):
        package = types.ModuleType("stepaudio2")
        package.__path__ = list(spec.submodule_search_locations)
        sys.modules["stepaudio2"] = package
    import stepaudio2.cosyvoice2.flow.decoder_dit as decoder_dit
    import stepaudio2.cosyvoice2.flow.flow as flow
    import stepaudio2.cosyvoice2.flow.flow_matching as flow_matching
    import stepaudio2.cosyvoice2.transformer.upsample_encoder_v2 as upsample

    # flow.yaml names classes as cosyvoice2.*, as Token2wav aliases them.
    aliases = {
        "cosyvoice2": types.ModuleType("cosyvoice2"),
        "cosyvoice2.flow": types.ModuleType("cosyvoice2.flow"),
        "cosyvoice2.transformer": types.ModuleType("cosyvoice2.transformer"),
        "cosyvoice2.flow.flow": flow,
        "cosyvoice2.flow.flow_matching": flow_matching,
        "cosyvoice2.flow.decoder_dit": decoder_dit,
        "cosyvoice2.transformer.upsample_encoder_v2": upsample,
    }
    for name, module in aliases.items():
        sys.modules.setdefault(name, module)


def count_minicpmo_token2wav(asset_dir: str, args: argparse.Namespace) -> dict:
    """FLOPs per chunk of the reference ``Token2wav.stream`` path, on CPU with real weights."""
    _import_stepaudio2_modules()
    from hyperpyyaml import load_hyperpyyaml
    from stepaudio2.flashcosyvoice.modules.hifigan import HiFTGenerator

    assets = Path(asset_dir)
    with open(assets / "flow.yaml") as f:
        flow = load_hyperpyyaml(f)["flow"]
    flow.load_state_dict(torch.load(assets / "flow.pt", map_location="cpu", weights_only=True), strict=True)
    flow.eval()
    hift = HiFTGenerator()
    state = torch.load(assets / "hift.pt", map_location="cpu", weights_only=True)
    hift.load_state_dict({k.replace("generator.", ""): v for k, v in state.items()}, strict=True)
    hift.eval()
    # FlopCounterMode cannot read weights through weight-norm parametrizations.
    for module in hift.modules():
        if torch.nn.utils.parametrize.is_parametrized(module):
            for name in list(module.parametrizations):
                torch.nn.utils.parametrize.remove_parametrizations(module, name, leave_parametrized=True)

    token_hz = args.codec_hz
    lookahead, up_rate = flow.pre_lookahead_len, flow.up_rate
    prompt_tokens = round(args.prompt_audio_s * token_hz)
    prompt_mels = prompt_tokens * up_rate
    # Reference Token2wav.stream constants.
    att_tail, mel_overlap, samples_per_mel = 100, 8, 480
    vocab = flow.vocab_size

    chunks = []
    with torch.inference_mode():
        tokens = torch.randint(0, vocab, (1, prompt_tokens + lookahead))
        spk = torch.randn(1, 192)
        with FlopCounterMode(display=False) as counter:
            cache = flow.setup_cache(tokens, torch.randn(1, prompt_mels, 80), spk, n_timesteps=args.n_timesteps)
        setup = counter.get_total_flops()
        mel_cache, source_cache = torch.zeros(1, 80, 0), torch.zeros(1, 1, 0)
        for _ in range(args.chunks):
            chunk = torch.randint(0, vocab, (1, args.chunk_tokens + lookahead))
            with FlopCounterMode(display=False) as counter:
                chunk_mel, cache = flow.inference_chunk(
                    chunk, spk, cache, last_chunk=False, n_timesteps=args.n_timesteps
                )
            flow_flops = counter.get_total_flops()
            for key, dim in (("estimator_att_cache", 4), ("conformer_att_cache", 3)):
                att = cache[key]
                if att.shape[dim] > prompt_mels + att_tail:
                    cache[key] = torch.cat(
                        [att.narrow(dim, 0, prompt_mels), att.narrow(dim, att.shape[dim] - att_tail, att_tail)], dim=dim
                    )
            mel = torch.cat([mel_cache, chunk_mel], dim=2)
            with FlopCounterMode(display=False) as counter:
                _, source = hift(mel, source_cache)
            chunks.append({"flow": flow_flops, "vocoder": counter.get_total_flops(), "mel_frames": chunk_mel.shape[2]})
            mel_cache, source_cache = mel[..., -mel_overlap:], source[:, :, -mel_overlap * samples_per_mel :]

    chunk_audio_s = args.chunk_tokens / token_hz
    steady = chunks[-1]
    return {
        "recipe": "minicpmo-token2wav",
        "dtype": "fp32",
        "chunk_tokens": args.chunk_tokens,
        "lookahead_tokens": lookahead,
        "chunk_audio_s": chunk_audio_s,
        "n_timesteps": args.n_timesteps,
        "prompt_audio_s": args.prompt_audio_s,
        "attention_cache_frames": prompt_mels + att_tail,
        "rows": {
            "flow": {
                "tflop_per_audio_s": steady["flow"] / chunk_audio_s / 1e12,
                "first_chunk_tflop": chunks[0]["flow"] / 1e12,
            },
            "vocoder": {
                "tflop_per_audio_s": steady["vocoder"] / chunk_audio_s / 1e12,
                "first_chunk_tflop": chunks[0]["vocoder"] / 1e12,
            },
        },
        "prompt_setup_tflop": setup / 1e12,
        "chunks_tflop": [{k: (v / 1e12 if k != "mel_frames" else v) for k, v in c.items()} for c in chunks],
    }


CODE2WAV_RECIPES = {"minicpmo-token2wav": count_minicpmo_token2wav}


def keyed(raws: list[str], cast=str) -> dict:
    return {key: cast(value) for key, _, value in (raw.partition("=") for raw in raws)}


def name_stage(key: str) -> tuple[str, str | None]:
    name, _, stage = key.partition("@")
    return name, stage or None


def emit_flags(result: dict, args: argparse.Namespace) -> list[str]:
    rates = keyed(args.tokens_per_audio_s, float)
    contexts = keyed(args.mean_context, float)
    uncached = keyed(args.uncached_tokens, float)
    dtypes = keyed(args.dtype)
    flags = []
    for name, d in result["decoders"].items():
        tag = name if d["stage"] is None else f"{name}@{d['stage']}"
        dtype = d["dtype"]
        if name in rates:
            kv_gib = d["kv_bytes_per_token"] * contexts.get(name, d["decode_context"]) / GIB
            flags.append(
                f"--decode-bound {tag}={d['streamed_gib_per_step']:.4g},{rates[name]:.4g},{kv_gib:.4g},"
                f"{d['decode_gflop_per_token']:.4g},{dtype}"
            )
        if name in uncached:
            tflop = d["prefill_gflop_per_token"] * uncached[name] / 1e3
            flags.append(
                f"--fixed-bound {name}_prefill@{d['stage'] or 0}={tflop:.4g},{dtype},{d['streamed_gib_per_step']:.4g}"
            )
    for raw in args.fill_steps:
        key, _, steps = raw.partition("=")
        name, gate = name_stage(key)
        d = result["decoders"][name]
        k = float(steps)
        flags.append(
            f"--fill-bound {name}_first_{steps}@{gate}={k * d['decode_gflop_per_token'] / 1e3:.4g},"
            f"{d['dtype']},{k * d['streamed_gib_per_step']:.4g}"
        )
    for name, c in result["code2wav"].items():
        for row, r in c["rows"].items():
            dtype = dtypes.get(f"{name}.{row}", dtypes.get(name, c["dtype"]))
            tag = f"{name}" if c["stage"] is None else f"{name}@{c['stage']}"
            flags.append(f"--compute-bound {tag}={r['tflop_per_audio_s']:.4g},{dtype}")
            flags.append(f"--fill-bound {name}_{row}_first@out={r['first_chunk_tflop']:.4g},{dtype}")
    return flags


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="checkpoint directory with the reference modeling code")
    parser.add_argument("--decoder", action="append", default=[], help="NAME[@STAGE]=ATTR[+HEAD_ATTR...]")
    parser.add_argument("--code2wav", action="append", default=[], help="RECIPE[@STAGE]=ASSET_DIR")
    parser.add_argument(
        "--config-override",
        action="append",
        default=[],
        help="ATTR=JSON on the top-level config, e.g. init_vision=false to skip encoders that are not counted",
    )
    parser.add_argument(
        "--block-import",
        action="append",
        default=[],
        help="treat a module as not installed (an optional dependency that fails to import, such as torchaudio)",
    )
    parser.add_argument(
        "--prompt-tokens", type=int, default=128, help="prefill length the per-token average is taken at"
    )
    parser.add_argument("--context", action="append", default=[], help="NAME=TOKENS decode context (default prompt)")
    parser.add_argument("--chunk-tokens", type=int, default=25, help="codec tokens per Code2Wav chunk")
    parser.add_argument("--codec-hz", type=float, default=25.0, help="codec tokens per audio second of the codec")
    parser.add_argument("--n-timesteps", type=int, default=10, help="flow-matching steps per chunk")
    parser.add_argument("--prompt-audio-s", type=float, default=6.0, help="reference (speaker prompt) audio seconds")
    parser.add_argument("--chunks", type=int, default=4, help="chunks streamed; the last is the steady state")
    parser.add_argument("--tokens-per-audio-s", action="append", default=[], help="NAME=X, from the result JSON")
    parser.add_argument("--mean-context", action="append", default=[], help="NAME=TOKENS for KV GiB per sequence")
    parser.add_argument("--uncached-tokens", action="append", default=[], help="NAME=TOKENS prefilled per request")
    parser.add_argument("--fill-steps", action="append", default=[], help="NAME@GATE=STEPS decode steps before GATE")
    parser.add_argument("--dtype", action="append", default=[], help="NAME[.ROW]=DTYPE the deployment runs")
    parser.add_argument(
        "--weight-dtype",
        action="append",
        default=[],
        help="NAME=bf16|fp16|fp32 decoder weight storage; defaults to --dtype's storage or the reference",
    )
    parser.add_argument(
        "--kv-dtype",
        action="append",
        default=[],
        help="NAME=bf16|fp16|fp32 decoder KV storage; defaults to the decoder weight dtype",
    )
    parser.add_argument("--json", help="write the counts to this path")
    args = parser.parse_args()

    for name in args.block_import:
        sys.modules[name] = None
    result = {"model": args.model, "decoders": {}, "code2wav": {}}
    if args.decoder:
        if not args.model:
            parser.error("--decoder needs --model")
        from transformers import AutoConfig, AutoModel

        config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        for key, value in keyed(args.config_override).items():
            setattr(config, key, json.loads(value))
        dtype = getattr(config, "dtype", None) or getattr(config, "torch_dtype", None) or torch.bfloat16
        start = time.time()
        with torch.device("meta"):
            model = AutoModel.from_config(config, trust_remote_code=True, dtype=dtype)
        model.eval()
        contexts = keyed(args.context, int)
        dtypes = keyed(args.dtype)
        weight_dtypes = keyed(args.weight_dtype)
        kv_dtypes = keyed(args.kv_dtype)
        for raw in args.decoder:
            key, _, spec = raw.partition("=")
            name, stage = name_stage(key)
            counts = count_decoder(
                model,
                spec,
                args.prompt_tokens,
                contexts.get(name, args.prompt_tokens),
                compute_dtype=dtypes.get(name),
                weight_dtype=weight_dtypes.get(name),
                kv_dtype=kv_dtypes.get(name),
            )
            result["decoders"][name] = {"stage": stage, **counts}
        result["build_s"] = time.time() - start
    for raw in args.code2wav:
        key, _, asset_dir = raw.partition("=")
        recipe, stage = name_stage(key)
        if recipe not in CODE2WAV_RECIPES:
            parser.error(f"unknown recipe {recipe!r}; known: {sorted(CODE2WAV_RECIPES)}")
        result["code2wav"]["code2wav"] = {"stage": stage, **CODE2WAV_RECIPES[recipe](asset_dir, args)}

    print(json.dumps(result, indent=2))
    if flags := emit_flags(result, args):
        print("\n# ceiling_report.py flags")
        print(" \\\n".join(flags))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
