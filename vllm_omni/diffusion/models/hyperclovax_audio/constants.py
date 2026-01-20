SPEAKERS_LIST = [
    "fkjy",
    "fkms",
    "fphj",
    "mhwj",
    "mphw",
    "msij",
    "fbhg",
    "fcj",
    "fcsy",
    "fkjh",
    "fkms2",
    "fljh",
    "fljh2",
    "mcjm",
    "mhjw",
    "mjch",
    "mjjh",
    "mkdk",
    "mkmy",
    "mkyw",
    "mlk",
    "mltm",
    "mmhb",
    "mngm",
    "msnh",
    "mjhh",
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
    (b"RIFF", "wav"),  # WAV (RIFF container)
    (b"\x1a\x45\xdf\xa3", "webm"),  # WebM / MKV (EBML header)
    (b"OggS", "ogg"),  # OGG
    (b"fLaC", "flac"),  # FLAC
    (b"ID3", "mp3"),  # MP3 with ID3 tag
    (b"\xff\xfb", "mp3"),  # MP3 without ID3
    (b"\x00\x00\x00\x1c", "mp4"),  # MP4 / M4A
    (b"\x00\x00\x00\x20", "mp4"),  # MP4 / M4A
]

VOLUME_LEVEL_DB = -26
VOLUME_LEVEL = 10 ** (VOLUME_LEVEL_DB / 20)
