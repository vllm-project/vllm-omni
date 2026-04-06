SPEAKERS_LIST = [
    "fkms",
    "msij",
]

FORMAT_MIME_MAP = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
    "aac": "audio/aac",
    "pcm": "audio/pcm",
}

DEFAULT_FORMAT = "wav"

AUDIO_FORMAT_MAP = [
    (b"RIFF", "wav"),
    (b"\x1a\x45\xdf\xa3", "webm"),
    (b"OggS", "ogg"),
    (b"fLaC", "flac"),
    (b"ID3", "mp3"),
    (b"\xff\xfb", "mp3"),
    (b"\x00\x00\x00\x1c", "mp4"),
    (b"\x00\x00\x00\x20", "mp4"),
]

VOLUME_LEVEL_DB = -26

VOLUME_LEVEL = 10 ** (VOLUME_LEVEL_DB / 20)
