import numpy as np


async def extract_audio_from_video_async(video_url: str) -> tuple[np.ndarray, int | float]:
    """Extract audio from a video URL using librosa.

    Returns a (audio_array, sample_rate) tuple compatible with audio format.
    All blocking I/O operations are run in a thread pool.
    """
    import asyncio
    import os
    import tempfile
    from urllib.parse import urlparse

    parsed_url = urlparse(video_url)
    temp_video_file_path = None

    def _download_video_sync(url: str) -> bytes:
        from urllib.request import urlopen

        return urlopen(url).read()

    def _write_temp_file_sync(data: bytes, suffix: str) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(data)
            return temp_file.name

    def _load_audio_sync(file_path: str) -> tuple[np.ndarray, int | float]:
        import librosa

        return librosa.load(file_path, sr=16000)

    def _cleanup_file_sync(file_path: str) -> None:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except OSError:
            pass

    try:
        if parsed_url.scheme in ("http", "https"):
            video_data = await asyncio.to_thread(_download_video_sync, video_url)
            temp_video_file_path = await asyncio.to_thread(_write_temp_file_sync, video_data, ".mp4")
        elif parsed_url.scheme == "file":
            from urllib.request import url2pathname

            temp_video_file_path = url2pathname(parsed_url.path)
        elif parsed_url.scheme == "data":
            import base64

            header, data = video_url.split(",", 1)
            video_data = base64.b64decode(data)
            temp_video_file_path = await asyncio.to_thread(_write_temp_file_sync, video_data, ".mp4")
        else:
            # Assume local file path
            temp_video_file_path = video_url

        audio_array, sample_rate = await asyncio.to_thread(_load_audio_sync, temp_video_file_path)
        return audio_array, sample_rate
    finally:
        if temp_video_file_path and parsed_url.scheme in ("http", "https", "data"):
            await asyncio.to_thread(_cleanup_file_sync, temp_video_file_path)
