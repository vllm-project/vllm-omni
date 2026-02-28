Here's a cleaned-up version of your code with improved structure, error handling, and best practices:

```python
import argparse
import time
import logging
from typing import Any, Dict, List
from dataclasses import dataclass

import numpy as np
from datasets import load_dataset
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class VideoProcessingError(Exception):
    """Custom exception for video processing errors"""
    pass


def sample_frames_from_video(video_path: str, num_frames: int) -> np.ndarray:
    """
    Extracts evenly spaced frames from a video file using decord.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        
    Returns:
        np.ndarray: Array of extracted frames
        
    Raises:
        VideoProcessingError: If video processing fails
    """
    try:
        import decord
    except ImportError as e:
        raise ImportError(
            "The decord library is required for video benchmarking. "
            "Please install it using: pip install decord"
        ) from e

    try:
        decord.bridge.set_bridge("numpy")
        vr = decord.VideoReader(video_path)
        total_frames = len(vr)

        if total_frames < num_frames:
            raise VideoProcessingError(
                f"Video has only {total_frames} frames, requested {num_frames}"
            )

        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = vr.get_batch(frame_indices).asnumpy()

        return frames
    except Exception as e:
        raise VideoProcessingError(f"Failed to process video {video_path}: {str(e)}") from e


@dataclass
class BenchmarkConfig:
    """Configuration for the video inference benchmark"""
    model: str
    num_requests: int
    num_frames: int
    tp_size: int
    max_model_len: int
    max_tokens: int
    enforce_eager: bool
    show_samples: bool


def get_ucf101_dataset(num_requests: int, num_frames: int = 16) -> List[Dict[str, Any]]:
    """
    Prepares the UCF101 dataset for offline video inference benchmarking.
    
    Args:
        num_requests: Number of video requests to prepare
        num_frames: Number of frames to extract per video
        
    Returns:
        List[Dict[str, Any]]: List of formatted requests
    """
    try:
        dataset = load_dataset("sayakpaul/ucf101-subset", split="val")
    except Exception as e:
        logger.error(f"Failed to load UCF101 dataset: {str(e)}")
        raise

    if len(dataset) > num_requests:
        dataset = dataset.shuffle(seed=42).select(range(num_requests))

    requests = []
    for data in dataset:
        try:
            video_path = data["video"]
            label = data["label"]

            frames = sample_frames_from_video(video_path, num_frames)

            prompt = (
                "<|user|>\n"
                "<|video|>\n"
                "Identify and describe the main action taking place in this video.\n"
                "<|assistant|>\n"
            )

            requests.append({
                "prompt": prompt,
                "multi_modal_data": {
                    "video": frames
                },
                "expected_output": label
            })
        except VideoProcessingError as e:
            logger.warning(f"Skipping video due to processing error: {str(e)}")
            continue

    if len(requests) < num_requests:
        logger.warning(
            f"Only found {len(requests)} valid videos, requested {num_requests}"
        )

    return requests


def run_benchmark(config: BenchmarkConfig):
    """Executes the video inference benchmark"""
    logger.info(f"Fetching and preprocessing UCF101 dataset: {config.num_requests} requests...")
    requests = get_ucf101_dataset(config.num_requests, config.num_frames)

    if not requests:
        raise RuntimeError("No valid videos found for benchmarking")

    logger.info(f"Initializing vLLM Engine with model: {config.model}")
    llm = LLM(
        model=config.model,
        tensor_parallel_size=config.tp_size,
        trust_remote_code=True,
        max_model_len=config.max_model_len,
        enforce_eager=config.enforce_eager,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=config.max_tokens,
    )

    inputs = [
        {
            "prompt": req["prompt"],
            "multi_modal_data": req["multi_modal_data"],
        }
        for req in requests
    ]

    logger.info("Starting high-throughput offline inference benchmarking...")
    start_time = time.perf_counter()

    outputs = llm.generate(inputs, sampling_params=sampling_params)

    elapsed_time = time.perf_counter() - start_time
    throughput = len(outputs) / elapsed_time

    logger.info("=" * 40)
    logger.info("         BENCHMARK RESULTS")
    logger.info("=" * 40)
    logger.info(f"Model:                {config.model}")
    logger.info(f"Dataset:              UCF101")
    logger.info(f"Total Requests:       {len(outputs)}")
    logger.info(f"Frames per Video:     {config.num_frames}")
    logger.info(f"Total Elapsed Time:   {elapsed_time:.2f} seconds")
    logger.info(f"Throughput:           {throughput:.2f} requests/second")
    logger.info("=" * 40)

    if config.show_samples:
        logger.info("--- Sample Model Outputs ---")
        for i in range(min(3, len(outputs))):
            logger.info(f"Sample {i+1}:")
            logger.info(f"  Expected Label: {requests[i]['expected_output']}")
            logger.info(f"  Generated Text: {outputs[i].outputs[0].text.strip()}\n")


def main():
    parser = argparse.ArgumentParser(description="vLLM UCF101 Offline Video Inference Benchmark")

    parser.add_argument("--model", type=str, default="Qwen/Qwen2-VL-7B-Instruct",
                        help="HuggingFace model ID for video inference")
    parser.add_argument("--num-requests", type=int, default=20,
                        help="Number of video requests to process during benchmark")
    parser.add_argument("--num-frames", type=int, default=16,
                        help="Number of frames to extract uniformly per video")
    parser.add_argument("--tp-size", type=int, default=1,
                        help="Tensor Parallel degree for model sharding")
    parser.add_argument("--max-model-len", type=int, default=8192,
                        help="Maximum model context length to allocate")
    parser.add_argument("--max-tokens", type=int, default=64,
                        help="Maximum tokens to generate per request")
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Enforce eager execution (disable CUDA graphs for memory savings)")
    parser.add_argument("--show-samples", action="store_true", default=True,
                        help="Print sample generations after benchmarking")

    args = parser.parse_args()

    config = BenchmarkConfig(
        model=args.model,
        num_requests=args.num_requests,
        num_frames=args.num_frames,
        tp_size=args.tp_size,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        enforce_eager=args.enforce_eager,
        show_samples=args.show_samples
    )

    try:
        run_benchmark(config)
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
```

Key improvements:

1. **Structured error handling** with custom `VideoProcessingError` exception
2. **Logging** instead of print statements for better output control
3. **Dataclass** for configuration management
4. **Type hints** throughout for better code clarity
5. **Docstrings** for all functions
6. **Graceful handling** of video processing failures
7. **Better separation of concerns** with dedicated functions
8. **Validation** of video frame counts
9. **Warnings** when fewer videos are available than requested
10. **Cleaner main function** with proper exception handling

The code is now more maintainable, debuggable, and production-ready.