"""Offline benchmark script for Qwen3 TTS via vLLM Omni.

Supports warmup + test rounds with a single Omni instance to avoid
repeated model loading. Prints per-round wall time summary at the end.
"""

import asyncio
import logging
import os
import time
from typing import Any, NamedTuple

import soundfile as sf
import torch

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni import AsyncOmni, Omni

logger = logging.getLogger(__name__)


class QueryResult(NamedTuple):
    """Container for a prepared Omni request."""

    inputs: dict
    model_name: str


_model_cache: dict[str, Any] = {}


def _estimate_prompt_len(
    additional_information: dict[str, Any],
    model_name: str,
) -> int:
    """Estimate prompt_token_ids placeholder length for the Talker stage."""
    try:
        from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import Qwen3TTSConfig
        from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
            Qwen3TTSTalkerForConditionalGeneration,
        )

        if model_name not in _model_cache:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True, padding_side="left", fix_mistral_regex=True
            )
            cfg = Qwen3TTSConfig.from_pretrained(model_name, trust_remote_code=True)

            speech_tok = None
            try:
                from transformers.utils import cached_file

                from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_tokenizer import Qwen3TTSTokenizer

                st_cfg_path = cached_file(model_name, "speech_tokenizer/config.json")
                if st_cfg_path:
                    speech_tok = Qwen3TTSTokenizer.from_pretrained(
                        # os.path.dirname(st_cfg_path), torch_dtype=torch.bfloat16
                        os.path.dirname(st_cfg_path),
                        torch_dtype=torch.bfloat16,
                        fix_mistral_regex=True,
                    )
                    logger.info("Loaded speech tokenizer for exact ref_code_len estimation")
            except Exception as e:
                logger.debug("Could not load speech tokenizer: %s", e)

            _model_cache[model_name] = (tok, getattr(cfg, "talker_config", None), speech_tok)

        tok, tcfg, speech_tok = _model_cache[model_name]
        task_type = (additional_information.get("task_type") or ["CustomVoice"])[0]

        def _estimate_ref_code_len(ref_audio: object) -> int | None:
            if not isinstance(ref_audio, (str, list)):
                return None
            audio_path = ref_audio[0] if isinstance(ref_audio, list) else ref_audio
            if not isinstance(audio_path, str) or not audio_path.strip():
                return None
            try:
                from vllm.multimodal.media import MediaConnector

                connector = MediaConnector(allowed_local_media_path="/")
                audio, sr = connector.fetch_audio(audio_path)
                import numpy as np

                wav_np = np.asarray(audio, dtype=np.float32)

                if speech_tok is not None:
                    enc = speech_tok.encode(wav_np, sr=int(sr), return_dict=True)
                    ref_code = getattr(enc, "audio_codes", None)
                    if isinstance(ref_code, list):
                        ref_code = ref_code[0] if ref_code else None
                    if ref_code is not None and hasattr(ref_code, "shape"):
                        shape = ref_code.shape
                        return int(shape[0]) if len(shape) == 2 else int(shape[1]) if len(shape) == 3 else None

                codec_hz = getattr(tcfg, "codec_frame_rate", None) or 12
                return int(len(audio) / sr * codec_hz)
            except Exception:
                return None

        return Qwen3TTSTalkerForConditionalGeneration.estimate_prompt_len_from_additional_information(
            additional_information=additional_information,
            task_type=task_type,
            tokenize_prompt=lambda t: tok(t, padding=False)["input_ids"],
            codec_language_id=getattr(tcfg, "codec_language_id", None),
            spk_is_dialect=getattr(tcfg, "spk_is_dialect", None),
            estimate_ref_code_len=_estimate_ref_code_len,
        )
    except Exception as exc:
        logger.warning("Failed to estimate prompt length, using fallback 2048: %s", exc)
        return 2048


def get_custom_voice_query(use_batch_sample: bool = False) -> QueryResult:
    task_type = "CustomVoice"
    model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    if use_batch_sample:
        texts = [
            "其实我真的有发现，我是一个特别善于观察别人情绪的人。",
            "She said she would be here by noon.",
            "I like you very much.",
            "Really, you do?",
            "Yes, absolutely.",
        ]
        instructs = ["", "Very happy.", "Very happy.", "Very happy.", "Very happy."]
        languages = ["Chinese", "English", "English", "English", "English"]
        speakers = ["Vivian", "Ryan", "Ryan", "Ryan", "Ryan"]
        inputs = []
        for text, instruct, language, speaker in zip(texts, instructs, languages, speakers):
            additional_information = {
                "task_type": [task_type],
                "text": [text],
                "instruct": [instruct],
                "language": [language],
                "speaker": [speaker],
                "max_new_tokens": [2048],
            }
            inputs.append(
                {
                    "prompt_token_ids": [0] * _estimate_prompt_len(additional_information, model_name),
                    "additional_information": additional_information,
                }
            )
    else:
        text = "其实我真的有发现，我是一个特别善于观察别人情绪的人。"
        language = "Chinese"
        speaker = "Vivian"
        instruct = "用特别愤怒的语气说"
        additional_information = {
            "task_type": [task_type],
            "text": [text],
            "language": [language],
            "speaker": [speaker],
            "instruct": [instruct],
            "max_new_tokens": [2048],
        }
        inputs = {
            "prompt_token_ids": [0] * _estimate_prompt_len(additional_information, model_name),
            "additional_information": additional_information,
        }
    return QueryResult(inputs=inputs, model_name=model_name)


def get_voice_design_query(use_batch_sample: bool = False) -> QueryResult:
    task_type = "VoiceDesign"
    model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    if use_batch_sample:
        texts = [
            "哥哥，你回来啦，人家等了你好久好久了，要抱抱！",
            "It's in the top drawer... wait, it's empty? No way, that's impossible! I'm sure I put it there!",
        ]
        instructs = [
            "体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。",
            "Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice.",
        ]
        languages = ["Chinese", "English"]
        inputs = []
        for text, instruct, language in zip(texts, instructs, languages):
            additional_information = {
                "task_type": [task_type],
                "text": [text],
                "language": [language],
                "instruct": [instruct],
                "max_new_tokens": [2048],
                "non_streaming_mode": [True],
            }
            inputs.append(
                {
                    "prompt_token_ids": [0] * _estimate_prompt_len(additional_information, model_name),
                    "additional_information": additional_information,
                }
            )
    else:
        text = "哥哥，你回来啦，人家等了你好久好久了，要抱抱！"
        instruct = "体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。"
        language = "Chinese"
        additional_information = {
            "task_type": [task_type],
            "text": [text],
            "language": [language],
            "instruct": [instruct],
            "max_new_tokens": [2048],
            "non_streaming_mode": [True],
        }
        inputs = {
            "prompt_token_ids": [0] * _estimate_prompt_len(additional_information, model_name),
            "additional_information": additional_information,
        }
    return QueryResult(inputs=inputs, model_name=model_name)


def get_base_query(use_batch_sample: bool = False, mode_tag: str = "icl") -> QueryResult:
    task_type = "Base"
    model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    ref_audio_path_1 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav"
    ref_audio_single = ref_audio_path_1
    ref_text_single = (
        "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
    )
    syn_text_single = "Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye."
    syn_lang_single = "Auto"
    x_vector_only_mode = mode_tag == "xvec_only"
    if use_batch_sample:
        syn_text_batch = [
            "Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye.",
            "其实我真的有发现，我是一个特别善于观察别人情绪的人。",
        ]
        syn_lang_batch = ["Chinese", "English"]
        inputs = []
        for text, language in zip(syn_text_batch, syn_lang_batch):
            additional_information = {
                "task_type": [task_type],
                "ref_audio": [ref_audio_single],
                "ref_text": [ref_text_single],
                "text": [text],
                "language": [language],
                "x_vector_only_mode": [x_vector_only_mode],
                "max_new_tokens": [2048],
            }
            inputs.append(
                {
                    "prompt_token_ids": [0] * _estimate_prompt_len(additional_information, model_name),
                    "additional_information": additional_information,
                }
            )
    else:
        additional_information = {
            "task_type": [task_type],
            "ref_audio": [ref_audio_single],
            "ref_text": [ref_text_single],
            "text": [syn_text_single],
            "language": [syn_lang_single],
            "x_vector_only_mode": [x_vector_only_mode],
            "max_new_tokens": [2048],
        }
        inputs = {
            "prompt_token_ids": [0] * _estimate_prompt_len(additional_information, model_name),
            "additional_information": additional_information,
        }
    return QueryResult(inputs=inputs, model_name=model_name)


query_map = {
    "CustomVoice": get_custom_voice_query,
    "VoiceDesign": get_voice_design_query,
    "Base": get_base_query,
}


def _build_inputs(args) -> tuple[str, list]:
    """Resolve model name and inputs list from CLI args."""
    if args.batch_size < 1 or (args.batch_size & (args.batch_size - 1)) != 0:
        raise ValueError(
            f"--batch-size must be a power of two (got {args.batch_size}); "
            "non-power-of-two values do not align with CUDA graph capture sizes "
            "of Code2Wav."
        )

    query_func = query_map[args.query_type]
    if args.query_type in {"CustomVoice", "VoiceDesign"}:
        query_result = query_func(use_batch_sample=args.use_batch_sample)
    elif args.query_type == "Base":
        query_result = query_func(use_batch_sample=args.use_batch_sample, mode_tag=args.mode_tag)
    else:
        query_result = query_func()

    model_name = query_result.model_name

    if args.txt_prompts:
        with open(args.txt_prompts) as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            raise ValueError(f"No valid prompts found in {args.txt_prompts}")
        template = query_result.inputs if not isinstance(query_result.inputs, list) else query_result.inputs[0]
        template_info = template["additional_information"]
        inputs = [
            {
                "prompt_token_ids": [0] * _estimate_prompt_len({**template_info, "text": [t]}, model_name),
                "additional_information": {**template_info, "text": [t]},
            }
            for t in lines
        ]
    else:
        inputs = query_result.inputs if isinstance(query_result.inputs, list) else [query_result.inputs]

    return model_name, inputs


def _save_wav(output_dir: str, request_id: str, mm: dict) -> None:
    """Concatenate audio chunks and write to a wav file."""
    audio_data = mm["audio"]
    sr_raw = mm["sr"]
    sr_val = sr_raw[-1] if isinstance(sr_raw, list) and sr_raw else sr_raw
    sr = sr_val.item() if hasattr(sr_val, "item") else int(sr_val)
    audio_tensor = torch.cat(audio_data, dim=-1) if isinstance(audio_data, list) else audio_data
    out_wav = os.path.join(output_dir, f"output_{request_id}.wav")
    sf.write(out_wav, audio_tensor.float().cpu().numpy().flatten(), samplerate=sr, format="WAV")
    logger.info("Saved audio to %s", out_wav)


class _SummaryCollector:
    """Monkey-patches omni._log_summary_and_cleanup to capture summary dicts.

    NOTE: This relies on a private method ``_log_summary_and_cleanup`` which may
    break if the upstream Omni implementation is refactored.  If Omni exposes a
    public hook for summary collection in the future, prefer that instead.
    """

    def __init__(self, omni):
        self.omni = omni
        self.summaries: list[dict[str, Any]] = []
        self.collecting = False
        self._orig = omni._log_summary_and_cleanup

    def install(self):
        from pprint import pformat as _pformat

        collector = self

        def _patched(request_id: str) -> None:
            req_state = collector.omni.request_states.get(request_id)
            try:
                if req_state is None or req_state.metrics is None:
                    return
                summary = req_state.metrics.build_and_log_summary()
                logger.info("[Summary] %s", _pformat(summary, sort_dicts=False))
                if collector.collecting and summary:
                    collector.summaries.append(summary)
            except Exception:
                logger.exception("Failed to build/log summary for req=%s", request_id)
            finally:
                collector.omni.request_states.pop(request_id, None)

        collector.omni._log_summary_and_cleanup = _patched


def _print_summary(round_times: list[float], summaries: list[dict[str, Any]] | None = None) -> None:
    """Print per-round wall time and optional detailed pipeline summary."""
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    for i, t in enumerate(round_times):
        print(f"  Test round {i}: {t:.3f}s")
    print("-" * 60)
    avg = sum(round_times) / len(round_times)
    print(f"  Rounds:  {len(round_times)}")
    print(f"  Total:   {sum(round_times):.3f}s")
    print(f"  Average: {avg:.3f}s")
    print(f"  Min:     {min(round_times):.3f}s")
    print(f"  Max:     {max(round_times):.3f}s")
    print("=" * 60)

    if not summaries:
        print()
        return

    # --- Aggregate from captured summaries ---
    overall_list = [s["overall_summary"] for s in summaries if "overall_summary" in s]
    if not overall_list:
        print()
        return

    n = len(overall_list)

    # Average Wall time
    avg_wall = sum(d["e2e_wall_time_ms"] for d in overall_list) / n
    print(f"\n{'=' * 60}")
    print("DETAILED PIPELINE SUMMARY (test rounds average)")
    print("=" * 60)
    print(f"  Avg Wall Time:           {avg_wall:.3f} ms")

    # Per-stage timing
    stage_keys = sorted(k for k in overall_list[0] if k.startswith("e2e_stage_") and k.endswith("_wall_time_ms"))
    if stage_keys:
        print("-" * 60)
        print("  Per-stage timing (avg):")
        for sk in stage_keys:
            stage_avg = sum(d.get(sk, 0.0) for d in overall_list) / n
            # extract stage index from key like e2e_stage_0_wall_time_ms
            stage_idx = sk.replace("e2e_stage_", "").replace("_wall_time_ms", "")
            print(f"    Stage {stage_idx} wall time:   {stage_avg:.3f} ms")

    # Per-stage gen time from stage_table
    all_stage_gen: dict[int, list[float]] = {}
    for s in summaries:
        for req_entry in s.get("stage_table", []):
            for stage_info in req_entry.get("stages", []):
                sid = stage_info.get("stage_id", -1)
                gen_ms = stage_info.get("stage_gen_time_ms", 0.0)
                all_stage_gen.setdefault(sid, []).append(gen_ms)
    if all_stage_gen:
        print("  Per-stage gen time (avg):")
        for sid in sorted(all_stage_gen):
            vals = all_stage_gen[sid]
            print(f"    Stage {sid} gen time:     {sum(vals) / len(vals):.3f} ms")

    # Per-prompt throughput
    avg_tokens_per_s = sum(d.get("e2e_avg_tokens_per_s", 0.0) for d in overall_list) / n
    avg_time_per_req = sum(d.get("e2e_avg_time_per_request_ms", 0.0) for d in overall_list) / n
    total_tokens_list = [d.get("e2e_total_tokens", 0) for d in overall_list]
    avg_total_tokens = sum(total_tokens_list) / n
    print("-" * 60)
    print("  Per-prompt throughput (avg):")
    print(f"    Tokens per request:    {avg_total_tokens:.1f}")
    print(f"    Time per request:      {avg_time_per_req:.3f} ms")
    print(f"    Tokens/s:              {avg_tokens_per_s:.3f}")
    print("=" * 60 + "\n")


def main(args):
    """Run sync benchmark: warmup rounds then test rounds."""
    model_name, inputs = _build_inputs(args)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    omni = Omni(
        model=model_name,
        stage_configs_path=args.stage_configs_path,
        log_stats=args.log_stats,
        stage_init_timeout=args.stage_init_timeout,
        enable_diffusion_pipeline_profiler=args.enable_diffusion_pipeline_profiler,
    )

    collector = _SummaryCollector(omni)
    collector.install()

    batch_size = args.batch_size

    # --- Warmup ---
    for r in range(args.warmup_rounds):
        print(f"[Warmup {r}/{args.warmup_rounds}] running...")
        for batch_start in range(0, len(inputs), batch_size):
            batch = inputs[batch_start : batch_start + batch_size]
            for _ in omni.generate(batch):
                pass
        print(f"[Warmup {r}/{args.warmup_rounds}] done")

    # --- Test ---
    collector.collecting = True
    round_times = []
    for r in range(args.test_rounds):
        print(f"[Test {r}/{args.test_rounds}] running...")
        t_start = time.perf_counter()
        for batch_start in range(0, len(inputs), batch_size):
            batch = inputs[batch_start : batch_start + batch_size]
            for stage_outputs in omni.generate(batch):
                output = stage_outputs.request_output
                _save_wav(
                    output_dir,
                    f"round{r}_{output.request_id}",
                    output.outputs[0].multimodal_output,
                )
        elapsed = time.perf_counter() - t_start
        round_times.append(elapsed)
        print(f"[Test {r}/{args.test_rounds}] done in {elapsed:.3f}s")
    collector.collecting = False

    _print_summary(round_times, collector.summaries if args.log_stats else None)


async def main_streaming(args):
    """Run async streaming benchmark: warmup rounds then test rounds."""
    model_name, inputs = _build_inputs(args)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    omni = AsyncOmni(
        model=model_name,
        stage_configs_path=args.stage_configs_path,
        log_stats=args.log_stats,
        stage_init_timeout=args.stage_init_timeout,
        enable_diffusion_pipeline_profiler=args.enable_diffusion_pipeline_profiler,
    )

    collector = _SummaryCollector(omni)
    collector.install()

    # --- Warmup ---
    for r in range(args.warmup_rounds):
        print(f"[Warmup {r}/{args.warmup_rounds}] running...")
        for i, prompt in enumerate(inputs):
            async for _ in omni.generate(prompt, request_id=f"warmup_{r}_{i}"):
                pass
        print(f"[Warmup {r}/{args.warmup_rounds}] done")

    # --- Test ---
    collector.collecting = True
    round_times = []
    for r in range(args.test_rounds):
        print(f"[Test {r}/{args.test_rounds}] running...")
        t_start = time.perf_counter()
        for i, prompt in enumerate(inputs):
            request_id = f"round{r}_{i}"
            t_req_start = time.perf_counter()
            t_prev = t_req_start
            chunk_idx = 0
            async for stage_output in omni.generate(prompt, request_id=request_id):
                mm = stage_output.request_output.outputs[0].multimodal_output
                if not stage_output.finished:
                    t_now = time.perf_counter()
                    audio = mm.get("audio")
                    n = len(audio) if isinstance(audio, list) else (0 if audio is None else 1)
                    if chunk_idx == 0:
                        ttfa_ms = (t_now - t_req_start) * 1000
                        logger.info("Request %s: chunk %d samples=%d TTFA=%.1fms", request_id, chunk_idx, n, ttfa_ms)
                    else:
                        dt_ms = (t_now - t_prev) * 1000
                        logger.info(
                            "Request %s: chunk %d samples=%d inter_chunk=%.1fms", request_id, chunk_idx, n, dt_ms
                        )
                    t_prev = t_now
                    chunk_idx += 1
                else:
                    t_end = time.perf_counter()
                    total_ms = (t_end - t_req_start) * 1000
                    logger.info("Request %s: done total=%.1fms chunks=%d", request_id, total_ms, chunk_idx)
                    _save_wav(output_dir, request_id, mm)
        elapsed = time.perf_counter() - t_start
        round_times.append(elapsed)
        print(f"[Test {r}/{args.test_rounds}] done in {elapsed:.3f}s")
    collector.collecting = False

    _print_summary(round_times, collector.summaries if args.log_stats else None)


def parse_args():
    parser = FlexibleArgumentParser(description="Benchmark script for Qwen3 TTS via vLLM Omni")
    parser.add_argument(
        "--query-type",
        "-q",
        type=str,
        default="CustomVoice",
        choices=query_map.keys(),
        help="Query type.",
    )
    parser.add_argument(
        "--warmup-rounds",
        type=int,
        default=3,
        help="Number of warmup rounds (results discarded, default: 3).",
    )
    parser.add_argument(
        "--test-rounds",
        type=int,
        default=7,
        help="Number of test rounds (results saved and timed, default: 7).",
    )
    parser.add_argument(
        "--log-stats",
        action="store_true",
        default=False,
        help="Enable writing detailed statistics (default: disabled).",
    )
    parser.add_argument(
        "--stage-init-timeout",
        type=int,
        default=300,
        help="Timeout for initializing a single stage in seconds (default: 300).",
    )
    parser.add_argument(
        "--output-dir",
        default="output_audio",
        help="Output directory for generated wav files (default: output_audio).",
    )
    parser.add_argument(
        "--txt-prompts",
        type=str,
        default=None,
        help="Path to a .txt file with one prompt per line.",
    )
    parser.add_argument(
        "--stage-configs-path",
        type=str,
        default=None,
        help="Path to a stage configs file.",
    )
    parser.add_argument(
        "--use-batch-sample",
        action="store_true",
        default=False,
        help="Use batch input sample for CustomVoice/VoiceDesign/Base query.",
    )
    parser.add_argument(
        "--mode-tag",
        type=str,
        default="icl",
        choices=["icl", "xvec_only"],
        help="Mode tag for Base query x_vector_only_mode (default: icl).",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=False,
        help="Stream audio chunks via AsyncOmni.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of prompts per batch (default: 1).",
    )
    parser.add_argument(
        "--enable-diffusion-pipeline-profiler",
        action="store_true",
        help="Enable diffusion pipeline profiler to display stage durations.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.streaming:
        asyncio.run(main_streaming(args))
    else:
        main(args)
