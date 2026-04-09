import argparse
import json
import os

import soundfile as sf
from qwen3_omni_moe_model import Qwen3OmniMoeForConditionalGenerationWithLogging
from qwen_omni_utils import process_mm_info
from tqdm import tqdm
from transformers import Qwen3OmniMoeProcessor

MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
# MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Thinking"


def load_prompts(prompts_file: str) -> list[str]:
    """Load prompts from a text file, one prompt per line."""
    prompts = []
    with open(prompts_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)
    return prompts


def run_benchmark(
    model,
    processor,
    prompts: list[str],
    output_dir: str = "benchmark_results",
    speaker: str = "Ethan",
    use_audio_in_video: bool = True,
):
    """
    Run benchmark on a list of prompts and collect performance stats.

    Args:
        model: The Qwen3OmniMoe model
        processor: The Qwen3OmniMoe processor
        prompts: List of text prompts to process
        output_dir: Directory to save results
        speaker: Speaker voice for audio output
        use_audio_in_video: Whether to use audio in video

    Returns:
        tuple: (aggregated_stats, results, audio_outputs)
            - aggregated_stats: dict with aggregated performance statistics
            - results: list of dicts with per-prompt results
            - audio_outputs: list of audio tensors/arrays (or None if no audio)
    """
    os.makedirs(output_dir, exist_ok=True)
    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    all_stats = []
    results = []
    audio_outputs = []

    for idx, prompt in enumerate(tqdm(prompts, desc="Processing prompts")):
        conversation = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            },
        ]

        # Preparation for inference
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)
        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=use_audio_in_video,
        )
        inputs = inputs.to(model.device).to(model.dtype)

        # Inference: Generation of the output text and audio
        text_ids, audio = model.generate(
            **inputs, speaker=speaker, thinker_return_dict_in_generate=True, use_audio_in_video=use_audio_in_video
        )

        # Decode output text
        output_text = processor.batch_decode(
            text_ids.sequences[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # Collect performance stats
        perf_stats = None
        if hasattr(model, "_perf_stats_last"):
            perf_stats = model._perf_stats_last.copy()
            perf_stats["prompt_idx"] = idx
            perf_stats["prompt"] = prompt
            all_stats.append(perf_stats)

        # Save audio and collect audio output
        audio_path = None
        audio_data = None
        if audio is not None:
            audio_data = audio.reshape(-1).detach().cpu().numpy()
            audio_path = os.path.join(audio_dir, f"output_{idx:04d}.wav")
            sf.write(
                audio_path,
                audio_data,
                samplerate=24000,
            )
            audio_outputs.append(audio_data)
        else:
            audio_outputs.append(None)

        # Save result
        result = {
            "idx": idx,
            "prompt": prompt,
            "output": output_text,
            "audio_path": audio_path,
            "perf_stats": perf_stats,
        }
        results.append(result)

    # Aggregate statistics
    aggregated_stats = aggregate_stats(all_stats)

    # Save all results
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Save aggregated stats
    stats_path = os.path.join(output_dir, "perf_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({"aggregated": aggregated_stats, "per_prompt": all_stats}, f, ensure_ascii=False, indent=2)

    # Count saved audio files
    num_audio_saved = sum(1 for a in audio_outputs if a is not None)
    print(f"\nSaved {num_audio_saved} audio files to {audio_dir}/")

    return aggregated_stats, results, audio_outputs


def aggregate_stats(all_stats: list[dict]) -> dict:
    """Aggregate performance statistics from multiple runs."""
    if not all_stats:
        return {}

    keys = [
        "thinker_tokens",
        "thinker_time_s",
        "thinker_tps",
        "talker_tokens",
        "talker_time_s",
        "talker_tps",
        "code2wav_tokens",
        "code2wav_time_s",
        "code2wav_tps",
        "total_tokens",
        "total_time_s",
        "total_tps",
    ]

    aggregated = {
        "num_samples": len(all_stats),
    }

    for key in keys:
        values = [s.get(key, 0) for s in all_stats if key in s]
        if values:
            aggregated[f"{key}_sum"] = sum(values)
            aggregated[f"{key}_avg"] = sum(values) / len(values)
            aggregated[f"{key}_min"] = min(values)
            aggregated[f"{key}_max"] = max(values)

    # Calculate overall throughput
    total_tokens = aggregated.get("total_tokens_sum", 0)
    total_time = aggregated.get("total_time_s_sum", 0)
    if total_time > 0:
        aggregated["overall_tps"] = total_tokens / total_time

    return aggregated


def print_stats(stats: dict):
    """Print performance statistics in a formatted way."""
    print("\n" + "=" * 60)
    print("Performance Statistics Summary")
    print("=" * 60)

    print(f"\nNumber of samples: {stats.get('num_samples', 0)}")

    print("\n--- Thinker ---")
    print(f"  Total tokens:  {stats.get('thinker_tokens_sum', 0):.0f}")
    print(f"  Total time:    {stats.get('thinker_time_s_sum', 0):.2f}s")
    print(f"  Avg TPS:       {stats.get('thinker_tps_avg', 0):.2f}")
    print(f"  Min TPS:       {stats.get('thinker_tps_min', 0):.2f}")
    print(f"  Max TPS:       {stats.get('thinker_tps_max', 0):.2f}")

    print("\n--- Talker ---")
    print(f"  Total tokens:  {stats.get('talker_tokens_sum', 0):.0f}")
    print(f"  Total time:    {stats.get('talker_time_s_sum', 0):.2f}s")
    print(f"  Avg TPS:       {stats.get('talker_tps_avg', 0):.2f}")
    print(f"  Min TPS:       {stats.get('talker_tps_min', 0):.2f}")
    print(f"  Max TPS:       {stats.get('talker_tps_max', 0):.2f}")

    print("\n--- Code2Wav ---")
    print(f"  Total tokens:  {stats.get('code2wav_tokens_sum', 0):.0f}")
    print(f"  Total time:    {stats.get('code2wav_time_s_sum', 0):.2f}s")
    print(f"  Avg TPS:       {stats.get('code2wav_tps_avg', 0):.2f}")
    print(f"  Min TPS:       {stats.get('code2wav_tps_min', 0):.2f}")
    print(f"  Max TPS:       {stats.get('code2wav_tps_max', 0):.2f}")

    print("\n--- Overall ---")
    print(f"  Total tokens:  {stats.get('total_tokens_sum', 0):.0f}")
    print(f"  Total time:    {stats.get('total_time_s_sum', 0):.2f}s")
    print(f"  Overall TPS:   {stats.get('overall_tps', 0):.2f}")
    print(f"  Avg TPS:       {stats.get('total_tps_avg', 0):.2f}")
    print(f"  Min TPS:       {stats.get('total_tps_min', 0):.2f}")
    print(f"  Max TPS:       {stats.get('total_tps_max', 0):.2f}")

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Qwen3-Omni Benchmark Script")
    parser.add_argument(
        "--prompts-file",
        type=str,
        default="benchmark/build_dataset/top100.txt",
        help="Path to the prompts file (one prompt per line)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="benchmark_results", help="Directory to save benchmark results"
    )
    parser.add_argument("--model-path", type=str, default=MODEL_PATH, help="Path to the model")
    parser.add_argument("--speaker", type=str, default="Ethan", help="Speaker voice for audio output")
    parser.add_argument("--num-prompts", type=int, default=None, help="Number of prompts to process (default: all)")
    args = parser.parse_args()

    # Load model and processor
    print(f"Loading model from {args.model_path}...")
    model = Qwen3OmniMoeForConditionalGenerationWithLogging.from_pretrained(
        args.model_path,
        dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model_path)

    # Benchmark mode
    print(f"Loading prompts from {args.prompts_file}...")
    prompts = load_prompts(args.prompts_file)

    if args.num_prompts:
        prompts = prompts[: args.num_prompts]

    print(f"Running benchmark on {len(prompts)} prompts...")

    aggregated_stats, results, audio_outputs = run_benchmark(
        model=model,
        processor=processor,
        prompts=prompts,
        output_dir=args.output_dir,
        speaker=args.speaker,
    )

    print_stats(aggregated_stats)
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
