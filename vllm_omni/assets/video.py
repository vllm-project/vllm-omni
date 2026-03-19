import librosa
import numpy as np
from vllm.assets.video import VideoAsset


def extract_video_audio(path: str = None, sampling_rate: int = 16000) -> np.ndarray:
    """This function extracts the audio from a video file path and returns the audio as a numpy array.
    Args:
        path: The path to the video file.
    Returns:
        The audio as a numpy array.
    """
    if not path:
        path = VideoAsset(name="baby_reading").video_path
    audio_signal, sr = librosa.load(path, sr=sampling_rate)
    return audio_signal
