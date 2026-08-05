"""Frozen Whisper encoder wrapper (wav -> encoder embeddings)."""

import torch
import torchaudio
from transformers import WhisperFeatureExtractor, WhisperModel

WHISPER_MEL_FRAMES = 3000  # 3000 frames at 16000 Hz


class WhisperEncoder:
    """Auxiliary content encoder for SVC.

    Kept in FP32 for stable HF Whisper inference under vLLM-Omni's default
    BF16 diffusion trunk (same idea as Stable Audio's FP32 VAE). Outputs are
    cast to the trunk dtype before fusion with the CFM conditioner.
    """

    def __init__(self, device: str | torch.device | None = None) -> None:
        self.fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
        self.model = WhisperModel.from_pretrained("openai/whisper-base")
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"),
        )
        self.model = self.model.to(device=self.device, dtype=torch.float32)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    def float(self) -> "WhisperEncoder":
        self.model = self.model.float()
        return self

    def encode(
        self,
        wav: torch.Tensor,
        sr: int,
        *,
        output_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        wav = (
            torchaudio.functional.resample(wav, orig_freq=sr, new_freq=self.fe.sampling_rate)
            if sr != self.fe.sampling_rate
            else wav
        )
        wav_np = wav.detach().float().cpu().numpy().astype("float32", copy=False)

        inputs = self.fe(
            wav_np,
            sampling_rate=self.fe.sampling_rate,
            return_tensors="pt",
            padding=False,
            truncation=False,
            return_attention_mask=True,
        )

        input_features = inputs.input_features
        num_frames = input_features.shape[-1]
        if num_frames < WHISPER_MEL_FRAMES:
            pad = WHISPER_MEL_FRAMES - num_frames
            input_features = torch.nn.functional.pad(input_features, (0, pad))
        else:
            input_features = input_features[..., :WHISPER_MEL_FRAMES]

        input_features = input_features.to(device=wav.device, dtype=self.dtype)
        if self.model.device != wav.device:
            self.model = self.model.to(wav.device)

        attention_mask = inputs.attention_mask.to(wav.device) if inputs.attention_mask is not None else None

        with torch.inference_mode():
            encoder_out = self.model.encoder(input_features).last_hidden_state

        if attention_mask is not None:
            valid_mel_frames = attention_mask.sum(dim=1)
            valid_enc_frames = (valid_mel_frames + 1) // 2
            max_valid_enc_frames = min(int(valid_enc_frames.max().item()), encoder_out.shape[1])
            encoder_out = encoder_out[:, :max_valid_enc_frames, :]
            valid_len = min(int(valid_enc_frames[0].item()), max_valid_enc_frames)
            if valid_len < max_valid_enc_frames:
                encoder_out[0, valid_len:, :] = 0

        if output_dtype is not None and encoder_out.dtype != output_dtype:
            encoder_out = encoder_out.to(dtype=output_dtype)
        return encoder_out


if __name__ == "__main__":
    torch.manual_seed(0)
    audio = torch.randn(1, 24000 * 25).float().to("cuda")
    encoder = WhisperEncoder()
    whisper_encoder_out = encoder.encode(audio, sr=24000)
    print(whisper_encoder_out.shape)
