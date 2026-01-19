import base64
import datetime
import io
import math
import os
import random
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import pytest
import torch
import yaml
from vllm.logger import init_logger
from vllm.utils import get_open_port

logger = init_logger(__name__)


@pytest.fixture(autouse=True)
def clean_gpu_memory_between_tests():
    if os.getenv("VLLM_TEST_CLEAN_GPU_MEMORY", "0") != "1":
        yield
        return

    # Wait for GPU memory to be cleared before starting the test
    import gc

    from tests.utils import wait_for_gpu_memory_to_clear

    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        try:
            wait_for_gpu_memory_to_clear(
                devices=list(range(num_gpus)),
                threshold_ratio=0.1,
            )
        except ValueError as e:
            logger.info("Failed to clean GPU memory: %s", e)

    yield

    # Clean up GPU memory after the test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def dummy_messages_from_mix_data(
    system_prompt: dict[str, Any] = None,
    video_data_url: Any = None,
    audio_data_url: Any = None,
    image_data_url: Any = None,
    content_text: str = None,
):
    """Create messages with video、image、audio data URL for OpenAI API."""

    if content_text is not None:
        content = [{"type": "text", "text": content_text}]
    else:
        content = []

    media_items = []
    if isinstance(video_data_url, list):
        for video_url in video_data_url:
            media_items.append((video_url, "video"))
    else:
        media_items.append((video_data_url, "video"))

    if isinstance(image_data_url, list):
        for url in image_data_url:
            media_items.append((url, "image"))
    else:
        media_items.append((image_data_url, "image"))

    if isinstance(audio_data_url, list):
        for url in audio_data_url:
            media_items.append((url, "audio"))
    else:
        media_items.append((audio_data_url, "audio"))

    content.extend(
        {"type": f"{media_type}_url", f"{media_type}_url": {"url": url}}
        for url, media_type in media_items
        if url is not None
    )
    messages = [{"role": "user", "content": content}]
    if system_prompt is not None:
        messages = [system_prompt] + messages
    return messages


def generate_synthetic_audio(
    duration: int,  # seconds
    num_channels: int,  # 1：Mono，2：Stereo 5：5.1 surround sound
    sample_rate: int = 48000,  # Default use 48000Hz.
    save_to_file: bool = False,
) -> dict[str, Any]:
    """ "Generate synthetic audio with musical scale (do re mi fa so la ti do)."""
    import soundfile as sf

    # Initialize audio data
    num_samples = int(sample_rate * duration)
    audio_data = np.zeros((num_samples, num_channels), dtype=np.float32)

    # Major scale frequencies (C major scale starting from C4)
    # do re mi fa so la ti do
    scale_notes = [
        ("do", 261.63),  # C4
        ("re", 293.66),  # D4
        ("mi", 329.63),  # E4
        ("fa", 349.23),  # F4
        ("so", 392.00),  # G4
        ("la", 440.00),  # A4
        ("ti", 493.88),  # B4
        ("do", 523.25),  # C5 (octave higher)
    ]

    # Timing for clear notes
    note_duration = 0.5  # Each note 0.5 seconds
    pause_duration = 0.1  # Short pause between notes
    total_note_time = note_duration + pause_duration

    # Calculate how many scales we can fit
    scale_length = len(scale_notes) * total_note_time
    num_scales = max(1, int(duration / scale_length))

    current_time = 0

    for scale_num in range(num_scales):
        for note_name, freq in scale_notes:
            if current_time >= duration:
                break

            # Note sound
            start_time = current_time
            end_time = min(current_time + note_duration, duration)

            if start_time < end_time:
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)
                end_sample = min(end_sample, num_samples)

                if start_sample < end_sample:
                    # Generate clean sine wave for the note
                    t = np.arange(end_sample - start_sample) / sample_rate

                    # Main frequency
                    sound = 0.3 * np.sin(2 * math.pi * freq * t)

                    # Add harmonics for richer tone (musical)
                    # 2nd harmonic (octave)
                    sound += 0.1 * np.sin(2 * math.pi * freq * 2 * t)
                    # 3rd harmonic (fifth)
                    sound += 0.05 * np.sin(2 * math.pi * freq * 3 * t)

                    # Piano-like envelope: quick attack, longer decay
                    envelope = np.ones_like(t)
                    total_samples = len(t)

                    # Piano envelope: 5% attack, 60% decay, 35% release
                    attack_samples = int(total_samples * 0.05)  # Quick attack
                    decay_samples = int(total_samples * 0.6)  # Longer decay
                    sustain_level = 0.3  # Sustain level
                    release_samples = int(total_samples * 0.35)  # Release

                    if attack_samples > 0:
                        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)

                    if decay_samples > 0 and attack_samples + decay_samples <= total_samples:
                        envelope[attack_samples : attack_samples + decay_samples] = np.linspace(
                            1, sustain_level, decay_samples
                        )

                    if release_samples > 0:
                        envelope[-release_samples:] = np.linspace(sustain_level, 0, release_samples)

                    sound *= envelope

                    # Apply to channels with stereo positioning
                    if num_channels == 1:
                        # Mono
                        audio_data[start_sample:end_sample, 0] += sound

                    elif num_channels == 2:
                        # Stereo - different positions for different notes
                        # do: center, re: slight left, mi: center, fa: slight right, etc.
                        note_positions = {
                            "do": 0.0,  # Center
                            "re": -0.2,  # Slight left
                            "mi": 0.0,  # Center
                            "fa": 0.2,  # Slight right
                            "so": -0.3,  # More left
                            "la": 0.3,  # More right
                            "ti": 0.0,  # Center
                        }

                        pan = note_positions.get(note_name, 0.0)
                        left_gain = 0.5 - pan / 2
                        right_gain = 0.5 + pan / 2

                        audio_data[start_sample:end_sample, 0] += sound * left_gain
                        audio_data[start_sample:end_sample, 1] += sound * right_gain

                    elif num_channels == 5:
                        # 5.1 surround - notes spread around
                        # Front channels get full sound
                        audio_data[start_sample:end_sample, 0] += sound * 0.7  # Front Left
                        audio_data[start_sample:end_sample, 1] += sound * 0.7  # Front Right
                        audio_data[start_sample:end_sample, 2] += sound * 0.8  # Center

                        # Rear channels get quieter version
                        delay = int(0.02 * sample_rate)  # 20ms delay
                        rear_start = min(start_sample + delay, num_samples - 1)
                        rear_end = min(end_sample + delay, num_samples)

                        if rear_start < rear_end:
                            rear_len = rear_end - rear_start
                            sound_for_rear = sound[:rear_len] * 0.25
                            audio_data[rear_start:rear_end, 3] += sound_for_rear  # Rear Left
                            audio_data[rear_start:rear_end, 4] += sound_for_rear  # Rear Right

            # Move to next note
            current_time += total_note_time

            # Add a small transition between scales
            if note_name == "do" and scale_num < num_scales - 1 and current_time < duration:
                transition_duration = 0.3
                transition_start = current_time
                transition_end = min(transition_start + transition_duration, duration)

                if transition_start < transition_end:
                    start_sample = int(transition_start * sample_rate)
                    end_sample = int(transition_end * sample_rate)
                    end_sample = min(end_sample, num_samples)

                    if start_sample < end_sample:
                        transition_t = np.arange(end_sample - start_sample) / sample_rate

                        # Glissando effect between scales
                        start_freq = 523.25  # High do
                        end_freq = 261.63  # Low do (next scale)
                        freq_sweep = np.linspace(start_freq, end_freq, len(transition_t))

                        transition_sound = 0.15 * np.sin(2 * math.pi * freq_sweep * transition_t)

                        # Fade in/out envelope
                        transition_env = np.ones_like(transition_t)
                        env_len = len(transition_env)
                        if env_len > 0:
                            transition_env[: int(env_len * 0.3)] = np.linspace(0, 1, int(env_len * 0.3))
                            transition_env[-int(env_len * 0.3) :] = np.linspace(1, 0, int(env_len * 0.3))

                        transition_sound *= transition_env

                        for ch in range(min(num_channels, 3)):
                            audio_data[start_sample:end_sample, ch] += transition_sound

                        current_time += transition_duration

    # Add very subtle reverb effect for musical feel
    if duration > 2:
        # Simple reverb: add delayed copies
        delay_times = [0.08, 0.15, 0.25]  # Different delay times in seconds
        delay_gains = [0.3, 0.2, 0.1]  # Decreasing gains

        for delay_sec, gain in zip(delay_times, delay_gains):
            delay_samples = int(delay_sec * sample_rate)
            if delay_samples < num_samples:
                for ch in range(num_channels):
                    delayed = np.zeros(num_samples)
                    delayed[delay_samples:] = audio_data[:-delay_samples, ch] * gain
                    audio_data[:, ch] += delayed

    # Normalize to avoid clipping
    max_amplitude = np.max(np.abs(audio_data))
    if max_amplitude > 0:
        # Normalize to 85% volume
        audio_data = audio_data / max_amplitude * 0.85

    # Optional: apply gentle low-pass filter for smoother sound
    if duration > 1:
        # Simple averaging filter
        filter_size = 3
        if filter_size > 0:
            for ch in range(num_channels):
                filtered = np.convolve(audio_data[:, ch], np.ones(filter_size) / filter_size, mode="same")
                audio_data[:, ch] = filtered

    # Handle file saving
    audio_bytes = None

    if save_to_file:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"audio_{num_channels}ch_{timestamp}.wav"

        try:
            sf.write(output_path, audio_data, sample_rate, format="WAV", subtype="PCM_16")
            print(f"Audio saved: {output_path}")

            with open(output_path, "rb") as f:
                audio_bytes = f.read()
        except Exception as e:
            print(f"Save failed: {e}")
            save_to_file = False

    # If not saving or save failed, create in memory
    if not save_to_file or audio_bytes is None:
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sample_rate, format="WAV", subtype="PCM_16")
        buffer.seek(0)
        audio_bytes = buffer.read()

    # Return result
    base64_audio = base64.b64encode(audio_bytes).decode("utf-8")
    result = {
        "base64": base64_audio,
    }
    if save_to_file and output_path:
        result["file_path"] = output_path

    return result


def generate_synthetic_video(width: int, height: int, num_frames: int, save_to_file: bool = False) -> str:
    """Generate synthetic video with bouncing balls and return base64 string."""

    import cv2
    import imageio

    # Create random balls
    num_balls = random.randint(3, 8)
    balls = []

    for _ in range(num_balls):
        radius = min(width, height) // 8
        if radius < 1:
            raise ValueError(f"Video dimensions ({width}x{height}) are too small for synthetic video generation")
        x = random.randint(radius, width - radius)
        y = random.randint(radius, height - radius)

        speed = random.uniform(3.0, 8.0)
        angle = random.uniform(0, 2 * math.pi)
        vx = speed * math.cos(angle)
        vy = speed * math.sin(angle)

        # OpenCV uses BGR format, but imageio expects RGB
        # We'll create in BGR first, then convert to RGB later
        color_bgr = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

        balls.append({"x": x, "y": y, "vx": vx, "vy": vy, "radius": radius, "color_bgr": color_bgr})

    # Generate video frames
    video_frames = []

    for frame_idx in range(num_frames):
        # Create black background (BGR format)
        frame_bgr = np.zeros((height, width, 3), dtype=np.uint8)

        for ball in balls:
            # Update position
            ball["x"] += ball["vx"]
            ball["y"] += ball["vy"]

            # Boundary collision detection
            if ball["x"] - ball["radius"] <= 0 or ball["x"] + ball["radius"] >= width:
                ball["vx"] = -ball["vx"]
                ball["x"] = max(ball["radius"], min(width - ball["radius"], ball["x"]))

            if ball["y"] - ball["radius"] <= 0 or ball["y"] + ball["radius"] >= height:
                ball["vy"] = -ball["vy"]
                ball["y"] = max(ball["radius"], min(height - ball["radius"], ball["y"]))

            # Use cv2 to draw circle
            x, y = int(ball["x"]), int(ball["y"])
            radius = ball["radius"]

            # Draw solid circle (main circle)
            cv2.circle(frame_bgr, (x, y), radius, ball["color_bgr"], -1)

            # Add simple 3D effect: draw a brighter center
            if radius > 3:  # Only add highlight when radius is large enough
                highlight_radius = max(1, radius // 2)
                highlight_x = max(highlight_radius, min(x - radius // 4, width - highlight_radius))
                highlight_y = max(highlight_radius, min(y - radius // 4, height - highlight_radius))

                # Create highlight color (brighter)
                highlight_color = tuple(min(c + 40, 255) for c in ball["color_bgr"])
                cv2.circle(frame_bgr, (highlight_x, highlight_y), highlight_radius, highlight_color, -1)

        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        video_frames.append(frame_rgb)

    video_bytes = None
    saved_file_path = None

    buffer = io.BytesIO()
    writer_kwargs = {
        "format": "mp4",
        "fps": 30,
        "codec": "libx264",
        "quality": 7,
        "pixelformat": "yuv420p",
        "macro_block_size": 16,
        "ffmpeg_params": [
            "-preset",
            "medium",
            "-crf",
            "23",
            "-movflags",
            "+faststart",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            f"scale={width}:{height}",
        ],
    }

    if save_to_file:
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"video_{width}x{height}_{timestamp}.mp4"
        try:
            with imageio.get_writer(output_path, **writer_kwargs) as writer:
                for frame in video_frames:
                    writer.append_data(frame)

            saved_file_path = output_path
            print(f"Video saved to: {saved_file_path}")
            with open(output_path, "rb") as f:
                video_bytes = f.read()

        except Exception as e:
            print(f"Warning: Failed to save video to file {output_path}: {e}")
            save_to_file = False

    if not save_to_file or video_bytes is None:
        with imageio.get_writer(buffer, **writer_kwargs) as writer:
            for frame in video_frames:
                writer.append_data(frame)

        buffer.seek(0)
        video_bytes = buffer.read()

    base64_video = base64.b64encode(video_bytes).decode("utf-8")

    result = {
        "base64": base64_video,
    }
    if save_to_file and saved_file_path:
        result["file_path"] = saved_file_path

    return result


def generate_synthetic_image(width: int, height: int, save_to_file: bool = False) -> Any:
    """Generate synthetic image with randomly colored squares and return base64 string."""
    from PIL import Image, ImageDraw

    # Create white background
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    # Generate random number of squares
    num_squares = random.randint(3, 8)

    for _ in range(num_squares):
        # Random square size
        square_size = random.randint(min(width, height) // 8, min(width, height) // 4)

        # Random position
        x = random.randint(0, width - square_size - 1)
        y = random.randint(0, height - square_size - 1)

        # Random color
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        # Random border width
        border_width = random.randint(1, 5)

        # Draw square
        draw.rectangle([x, y, x + square_size, y + square_size], fill=color, outline=(0, 0, 0), width=border_width)

    # Handle file saving
    image_bytes = None
    saved_file_path = None

    if save_to_file:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"image_{width}x{height}_{timestamp}.jpg"

        try:
            # Save image to file
            image.save(output_path, format="JPEG", quality=85, optimize=True)
            saved_file_path = output_path
            print(f"Image saved to: {saved_file_path}")

            # Read file for base64 encoding
            with open(output_path, "rb") as f:
                image_bytes = f.read()

        except Exception as e:
            print(f"Warning: Failed to save image to file {output_path}: {e}")
            save_to_file = False

    # If not saving or save failed, create in memory
    if not save_to_file or image_bytes is None:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85, optimize=True)
        buffer.seek(0)
        image_bytes = buffer.read()

    # Generate base64
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    # Return result
    result = {
        "base64": base64_image,
    }
    if save_to_file and saved_file_path:
        result["file_path"] = saved_file_path

    return result


def preprocess_text(text):
    import re

    word_to_num = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }

    for word, num in word_to_num.items():
        pattern = r"\b" + re.escape(word) + r"\b"
        text = re.sub(pattern, num, text, flags=re.IGNORECASE)

    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def cosine_similarity_text(s1, s2):
    """
        Calculate cosine similarity between two text strings.
        Notes:
    ------
    - Higher score means more similar texts
    - Score of 1.0 means identical word composition (bag-of-words)
    - Score of 0.0 means completely different vocabulary
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = CountVectorizer().fit_transform([preprocess_text(s1), preprocess_text(s2)])
    vectors = vectorizer.toarray()
    return cosine_similarity([vectors[0]], [vectors[1]])[0][0]


def convert_audio_to_text(audio_data):
    """
    Convert base64 encoded audio data to text using speech recognition.
    """
    import whisper

    audio_data = base64.b64decode(audio_data)
    output_path = f"./test_{int(time.time())}"
    with open(output_path, "wb") as audio_file:
        audio_file.write(audio_data)

    print(f"audio data is saved: {output_path}")

    model = whisper.load_model("base")
    text = model.transcribe(
        output_path,
        temperature=0.0,
        word_timestamps=True,
        condition_on_previous_text=False,
        initial_prompt="Please transcribe with proper word spacing.",
    )["text"]
    if text:
        return text
    else:
        return ""


def modify_stage_config(
    yaml_path: str,
    stage_updates: dict[int, dict[str, Any]],
) -> str:
    """
    Batch modify configurations for multiple stages in a YAML file.

    Args:
        yaml_path: Path to the YAML configuration file.
        stage_updates: Dictionary where keys are stage IDs and values are dictionaries of
                      modifications for that stage. Each modification dictionary uses
                      dot-separated paths as keys and new configuration values as values.
                      Example: {
                          0: {'engine_args.max_model_len': 5800},
                          1: {'runtime.max_batch_size': 2}
                      }

    Returns:
        str: Path to the newly created modified YAML file with timestamp suffix.

    Example:
        >>> output_file = modify_stage_config(
        ...     'config.yaml',
        ...     {
        ...         0: {'engine_args.max_model_len': 5800},
        ...         1: {'runtime.max_batch_size': 2}
        ...     }
        ... )
        >>> print(f"Modified configuration saved to: {output_file}")
        Modified configuration saved to: config_1698765432.yaml
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"yaml does not exist: {path}")
    try:
        with open(yaml_path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except Exception as e:
        raise ValueError(f"Cannot parse YAML file: {e}")

    stage_args = config.get("stage_args", [])
    if not stage_args:
        raise ValueError("the stage_args does not exist")

    for stage_id, config_dict in stage_updates.items():
        target_stage = None
        for stage in stage_args:
            if stage.get("stage_id") == stage_id:
                target_stage = stage
                break

        if target_stage is None:
            available_ids = [s.get("stage_id") for s in stage_args if "stage_id" in s]
            raise KeyError(f"Stage ID {stage_id} is not exist, available IDs: {available_ids}")

        for key_path, value in config_dict.items():
            current = target_stage
            keys = key_path.split(".")
            for i in range(len(keys) - 1):
                key = keys[i]
                if key not in current:
                    raise KeyError(f"the {'.'.join(keys[: i + 1])} does not exist")

                elif not isinstance(current[key], dict) and i < len(keys) - 2:
                    raise ValueError(f"{'.'.join(keys[: i + 1])}' cannot continue deeper because it's not a dict")
                current = current[key]
            current[keys[-1]] = value

    output_path = f"{yaml_path.split('.')[0]}_{int(time.time())}.yaml"
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True, indent=2)

    return output_path


class OmniServer:
    """Omniserver for vLLM-Omni tests."""

    def __init__(
        self,
        model: str,
        serve_args: list[str],
        *,
        env_dict: dict[str, str] | None = None,
    ) -> None:
        self.model = model
        self.serve_args = serve_args
        self.env_dict = env_dict
        self.proc: subprocess.Popen | None = None
        self.host = "127.0.0.1"
        self.port = get_open_port()

    def _start_server(self) -> None:
        """Start the vLLM-Omni server subprocess."""
        env = os.environ.copy()
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        if self.env_dict is not None:
            env.update(self.env_dict)

        cmd = [
            sys.executable,
            "-m",
            "vllm_omni.entrypoints.cli.main",
            "serve",
            self.model,
            "--omni",
            "--host",
            self.host,
            "--port",
            str(self.port),
        ] + self.serve_args

        print(f"Launching OmniServer with: {' '.join(cmd)}")
        self.proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # Set working directory to vllm-omni root
        )

        # Wait for server to be ready
        max_wait = 600  # 10 minutes
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(1)
                    result = sock.connect_ex((self.host, self.port))
                    if result == 0:
                        print(f"Server ready on {self.host}:{self.port}")
                        return
            except Exception:
                pass
            time.sleep(2)

        raise RuntimeError(f"Server failed to start within {max_wait} seconds")

    def _kill_process_tree(self, pid):
        """kill process and its children"""
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass

            gone, still_alive = psutil.wait_procs(children, timeout=10)

            for child in still_alive:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass

            try:
                parent.terminate()
                parent.wait(timeout=10)
            except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                try:
                    parent.kill()
                except psutil.NoSuchProcess:
                    pass

        except psutil.NoSuchProcess:
            pass

    def __enter__(self):
        self._start_server()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.proc:
            try:
                parent = psutil.Process(self.proc.pid)
                children = parent.children(recursive=True)
                for child in children:
                    try:
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass

                gone, still_alive = psutil.wait_procs(children, timeout=10)

                for child in still_alive:
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass

                try:
                    parent.terminate()
                    parent.wait(timeout=10)
                except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                    try:
                        parent.kill()
                    except psutil.NoSuchProcess:
                        pass

            except psutil.NoSuchProcess:
                pass
