"""GLM-4-Voice audio tokenizer wrapper for Kimi-Audio's input audio path.

Upstream Moonshot represents input audio as BOTH discrete codec IDs (from
``THUDM/glm-4-voice-tokenizer``) AND continuous Whisper features. The HF
repo ships only weights + config — the ``WhisperVQEncoder`` modeling code
is vendored under ``./glm/``."""

from __future__ import annotations

from functools import lru_cache

import torch

_KIMIA_TOKEN_OFFSET = 152064


@lru_cache(maxsize=1)
def _load_tokenizer():
    from transformers import WhisperFeatureExtractor

    from .modeling_whisper_vq import WhisperVQEncoder

    repo = "THUDM/glm-4-voice-tokenizer"
    model = WhisperVQEncoder.from_pretrained(repo).eval()
    if torch.cuda.is_available():
        model = model.to(torch.accelerator.current_device_index())
    extractor = WhisperFeatureExtractor.from_pretrained(repo)
    return model, extractor


def tokenize_audio(audio_array, sample_rate: int) -> list[int]:
    """Encode a single waveform into Kimi-Audio absolute codec IDs
    (already offset by ``KIMIA_TOKEN_OFFSET``). Inlines upstream's
    ``extract_speech_token``."""
    import torchaudio

    model, feature_extractor = _load_tokenizer()
    device = next(model.parameters()).device
    dtype = model.conv1.weight.dtype

    audio_tensor = torch.as_tensor(audio_array).float()
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    if sample_rate != 16000:
        audio_tensor = torchaudio.functional.resample(audio_tensor, orig_freq=sample_rate, new_freq=16000)
    audio = audio_tensor[0].cpu().numpy()

    pooling_kernel_size = model.config.pooling_kernel_size or 1
    stride = model.conv1.stride[0] * model.conv2.stride[0] * pooling_kernel_size * feature_extractor.hop_length
    segments = []
    t = 0
    while t * 16000 < audio.shape[0]:
        segments.append(audio[t * 16000 : (t + 30) * 16000])
        t += 30

    all_tokens: list[int] = []
    with torch.no_grad():
        features = feature_extractor(
            segments,
            sampling_rate=16000,
            return_attention_mask=True,
            return_tensors="pt",
            padding="longest",
            pad_to_multiple_of=stride,
        )
        features["input_features"] = features["input_features"].to(device).to(dtype)
        features["attention_mask"] = features["attention_mask"].to(device)
        outputs = model(**features)
        speech_tokens = outputs.quantized_token_ids
        attn_mask = features["attention_mask"][:, :: model.conv1.stride[0] * model.conv2.stride[0]]
        attn_mask = attn_mask[:, ::pooling_kernel_size]
        for i in range(len(speech_tokens)):
            tokens = speech_tokens[i][attn_mask[i].bool()].tolist()
            all_tokens.extend(tokens)

    return [int(t) + _KIMIA_TOKEN_OFFSET for t in all_tokens]
