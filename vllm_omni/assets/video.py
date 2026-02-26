import functools

import librosa
import numpy as np
from vllm.assets.video import VideoAsset


@functools.lru_cache(maxsize=32)
def _extract_video_audio_cached(path: str, sampling_rate: int = 16000) -> np.ndarray:
    audio_signal, sr = librosa.load(path, sr=sampling_rate)
    return audio_signal


def extract_video_audio(path: str = None, sampling_rate: int = 16000) -> np.ndarray:
    """This function extracts the audio from a video file path and returns the audio as a numpy array.
    Args:
        path: The path to the video file.
    Returns:
        The audio as a numpy array.
    """
    if not path:
        path = VideoAsset(name="baby_reading").video_path
    return _extract_video_audio_cached(path, sampling_rate).copy()
