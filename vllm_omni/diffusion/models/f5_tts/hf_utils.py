from __future__ import annotations

from dataclasses import dataclass

from huggingface_hub import hf_hub_download

F5_PIPELINE_CLASS = "F5TTSPipeline"
F5_REPO_ID = "SWivid/F5-TTS"


@dataclass(frozen=True)
class F5ModelSpec:
    repo_id: str
    subfolder: str
    ckpt_filename: str
    tokenizer_filename: str
    tokenizer_type: str
    dim: int
    depth: int
    heads: int
    ff_mult: int
    text_dim: int
    text_conv_layers: int
    mel_dim: int
    sample_rate: int
    hop_length: int
    win_length: int
    n_fft: int
    mel_spec_type: str
    tokenizer_subfolder: str | None = None
    wav_norm: bool = False
    dim_head: int = 64
    dropout: float = 0.0
    text_mask_padding: bool = True
    pe_attn_head: int | None = None
    attn_mask_enabled: bool = False
    long_skip_connection: bool = False
    conv_pos_embed_groups: int = 16

    def hf_filename(self, filename: str) -> str:
        return f"{self.subfolder}/{filename}"


_F5_MODEL_SPECS: dict[str, F5ModelSpec] = {
    "F5TTS_v1_Base": F5ModelSpec(
        repo_id=F5_REPO_ID,
        subfolder="F5TTS_v1_Base",
        ckpt_filename="model_1250000.safetensors",
        tokenizer_filename="vocab.txt",
        tokenizer_type="pinyin",
        dim=1024,
        depth=22,
        heads=16,
        ff_mult=2,
        text_dim=512,
        text_conv_layers=4,
        mel_dim=100,
        sample_rate=24000,
        hop_length=256,
        win_length=1024,
        n_fft=1024,
        mel_spec_type="vocos",
    ),
    "F5TTS_v1_Base_no_zero_init": F5ModelSpec(
        repo_id=F5_REPO_ID,
        subfolder="F5TTS_v1_Base_no_zero_init",
        ckpt_filename="model_1250000.safetensors",
        tokenizer_filename="vocab.txt",
        tokenizer_subfolder="F5TTS_v1_Base",
        tokenizer_type="pinyin",
        dim=1024,
        depth=22,
        heads=16,
        ff_mult=2,
        text_dim=512,
        text_conv_layers=4,
        mel_dim=100,
        sample_rate=24000,
        hop_length=256,
        win_length=1024,
        n_fft=1024,
        mel_spec_type="vocos",
    ),
    "F5TTS_Base": F5ModelSpec(
        repo_id=F5_REPO_ID,
        subfolder="F5TTS_Base",
        ckpt_filename="model_1200000.safetensors",
        tokenizer_filename="vocab.txt",
        tokenizer_type="pinyin",
        dim=1024,
        depth=22,
        heads=16,
        ff_mult=2,
        text_dim=512,
        text_conv_layers=4,
        mel_dim=100,
        sample_rate=24000,
        hop_length=256,
        win_length=1024,
        n_fft=1024,
        mel_spec_type="vocos",
        text_mask_padding=False,
        pe_attn_head=1,
    ),
    "F5TTS_Base_bigvgan": F5ModelSpec(
        repo_id=F5_REPO_ID,
        subfolder="F5TTS_Base_bigvgan",
        ckpt_filename="model_1250000.pt",
        tokenizer_filename="vocab.txt",
        tokenizer_subfolder="F5TTS_Base",
        tokenizer_type="pinyin",
        dim=1024,
        depth=22,
        heads=16,
        ff_mult=2,
        text_dim=512,
        text_conv_layers=4,
        mel_dim=100,
        sample_rate=24000,
        hop_length=256,
        win_length=1024,
        n_fft=1024,
        mel_spec_type="bigvgan",
        text_mask_padding=False,
        pe_attn_head=1,
    ),
}


def _strip_hf_prefix(model: str) -> str:
    normalized = model[5:] if model.startswith("hf://") else model
    return normalized.rstrip("/")


def get_f5_model_spec(model: str) -> F5ModelSpec | None:
    normalized = _strip_hf_prefix(model)
    parts = normalized.split("/")
    if len(parts) < 3:
        return None

    repo_id = "/".join(parts[:2])
    if repo_id != F5_REPO_ID:
        return None

    subfolder = "/".join(parts[2:])
    return _F5_MODEL_SPECS.get(subfolder)


def is_f5_model(model: str) -> bool:
    return get_f5_model_spec(model) is not None


def download_f5_file(
    model: str,
    filename: str,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
) -> str:
    spec = get_f5_model_spec(model)
    if spec is None:
        raise ValueError(f"Unsupported F5 model reference: {model}")

    return hf_hub_download(
        repo_id=spec.repo_id,
        filename=spec.hf_filename(filename),
        revision=revision,
        cache_dir=cache_dir,
    )


def _count_vocab_entries(vocab_path: str) -> int:
    with open(vocab_path, encoding="utf-8") as f:
        return sum(1 for _ in f)


def build_f5_transformer_config(
    model: str,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
) -> dict[str, object]:
    spec = get_f5_model_spec(model)
    if spec is None:
        raise ValueError(f"Unsupported F5 model reference: {model}")

    tokenizer_model = model
    if spec.tokenizer_subfolder is not None:
        tokenizer_model = f"{spec.repo_id}/{spec.tokenizer_subfolder}"

    vocab_path = download_f5_file(
        tokenizer_model,
        spec.tokenizer_filename,
        revision=revision,
        cache_dir=cache_dir,
    )
    vocab_size = _count_vocab_entries(vocab_path)

    return {
        "dim": spec.dim,
        "depth": spec.depth,
        "heads": spec.heads,
        "dim_head": spec.dim_head,
        "ff_mult": spec.ff_mult,
        "dropout": spec.dropout,
        "mel_dim": spec.mel_dim,
        "text_num_embeds": vocab_size,
        "text_dim": spec.text_dim,
        "text_mask_padding": spec.text_mask_padding,
        "pe_attn_head": spec.pe_attn_head,
        "text_conv_layers": spec.text_conv_layers,
        "attn_mask_enabled": spec.attn_mask_enabled,
        "long_skip_connection": spec.long_skip_connection,
        "conv_pos_embed_groups": spec.conv_pos_embed_groups,
        "tokenizer_type": spec.tokenizer_type,
        "tokenizer_path": vocab_path,
        "vocoder_name": spec.mel_spec_type,
        "hf_repo_id": spec.repo_id,
        "hf_subfolder": spec.subfolder,
        "hf_weight_filename": spec.ckpt_filename,
        "mel_spec": {
            "sample_rate": spec.sample_rate,
            "hop_length": spec.hop_length,
            "win_length": spec.win_length,
            "n_mel_channels": spec.mel_dim,
            "n_fft": spec.n_fft,
            "mel_spec_type": spec.mel_spec_type,
            "wav_norm": spec.wav_norm,
        },
    }
