"""ROSVOT note transcription."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn as nn
from vllm.logger import init_logger

from vllm_omni.diffusion.models.interface import SupportAudioInput, SupportsComponentDiscovery
from vllm_omni.diffusion.models.soulx_singer.modules.mel_transform import spectral_normalize_torch
from vllm_omni.diffusion.models.soulx_singer.modules.note_transcription import (
    BackboneNet,
    ConvBlocks,
    Embedding,
)
from vllm_omni.diffusion.models.soulx_singer.modules.preprocess.utils import (
    boundary2Interval,
    denorm_f0,
    f0_to_coarse,
    load_model_ckpt,
    load_mono_audio,
    load_rosvot_config,
    norm_interp_f0,
    pad_or_cut_xd,
    resample_mono,
)
from vllm_omni.utils.audio import mel_filter_bank as librosa_mel_fn

logger = init_logger(__name__)


def regulate_real_note_itv(
    note_itv: np.ndarray,
    note_bd: np.ndarray,
    word_bd: np.ndarray,
    word_durs: np.ndarray,
    hop_size: int,
    audio_sample_rate: int,
) -> tuple[np.ndarray, np.ndarray]:
    assert note_itv.shape[0] == np.sum(note_bd) + 1
    assert np.sum(word_bd) <= np.sum(note_bd)
    assert word_durs.shape[0] == np.sum(word_bd) + 1, f"{word_durs.shape[0]} {np.sum(word_bd) + 1}"
    word_bd = np.cumsum(word_bd) * word_bd
    word_itv = np.zeros((word_durs.shape[0], 2))
    word_offsets = np.cumsum(word_durs)
    note2words = np.zeros(note_itv.shape[0], dtype=int)
    for idx in range(len(word_offsets) - 1):
        word_itv[idx, 1] = word_itv[idx + 1, 0] = word_offsets[idx]
    word_itv[-1, 1] = word_offsets[-1]
    note_itv_secs = note_itv * hop_size / audio_sample_rate
    for idx, itv in enumerate(note_itv):
        start_idx, end_idx = itv
        if word_bd[start_idx] > 0:
            word_dur_idx = word_bd[start_idx]
            note_itv_secs[idx, 0] = word_itv[word_dur_idx, 0]
            note2words[idx] = word_dur_idx
        if word_bd[end_idx] > 0:
            word_dur_idx = word_bd[end_idx] - 1
            note_itv_secs[idx, 1] = word_itv[word_dur_idx, 1]
            note2words[idx] = word_dur_idx
    note2words += 1
    return note_itv_secs, note2words


def regulate_ill_slur(
    notes: np.ndarray, note_itv: np.ndarray, note2words: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    res_note2words: list[int] = []
    res_note_itv: list[list[float]] = []
    res_notes: list[int] = []
    note_idx = 0
    note_idx_end = 0
    while note_idx <= len(notes) - 1:
        while note_idx <= note_idx_end < len(notes) and note2words[note_idx] == note2words[note_idx_end]:
            note_idx_end += 1
        res_note2words.append(note2words[note_idx])
        res_note_itv.append(note_itv[note_idx].tolist())
        res_notes.append(notes[note_idx])
        for idx in range(note_idx + 1, note_idx_end):
            if notes[idx] == notes[idx - 1]:
                res_note_itv[-1][1] = note_itv[idx][1]
            else:
                res_note_itv.append(note_itv[idx].tolist())
                res_note2words.append(note2words[idx])
                res_notes.append(notes[idx])
        note_idx = note_idx_end
    return (
        np.array(res_notes, dtype=notes.dtype),
        np.array(res_note_itv, dtype=note_itv.dtype),
        np.array(res_note2words, dtype=note2words.dtype),
    )


def regulate_boundary(bd_logits, threshold, min_gap=18, ref_bd=None, ref_bd_min_gap=8, non_padding=None):
    # this doesn't preserve gradient
    device = bd_logits.device
    bd_logits = torch.sigmoid(bd_logits).data.cpu()
    # bd_logits[0] = bd_logits[-1] = 1e-5     # avoid itv invalid problem
    bd = (bd_logits > threshold).long()
    bd_res = torch.zeros_like(bd).long()
    for i in range(bd.shape[0]):
        bd_i = bd[i]
        last_bd_idx = -1
        start = -1
        for j in range(bd_i.shape[0]):
            if bd_i[j] == 1:
                if 0 <= start < j:
                    continue
                elif start < 0:
                    start = j
            else:
                if 0 <= start < j:
                    if j - 1 > start:
                        bd_idx = start + int(torch.argmax(bd_logits[i, start:j]).item())
                    else:
                        bd_idx = start
                    if bd_idx - last_bd_idx < min_gap and last_bd_idx > 0:
                        bd_idx = round((bd_idx + last_bd_idx) / 2)
                        bd_res[i, last_bd_idx] = 0
                    bd_res[i, bd_idx] = 1
                    last_bd_idx = bd_idx
                    start = -1

    # assert ref_bd_min_gap <= min_gap // 2
    if ref_bd is not None and ref_bd_min_gap > 0:
        ref = ref_bd.data.cpu()
        for i in range(bd_res.shape[0]):
            ref_bd_i = ref[i]
            ref_bd_i_js = []
            for j in range(ref_bd_i.shape[0]):
                if ref_bd_i[j] == 1:
                    ref_bd_i_js.append(j)
                    seg_sum = torch.sum(bd_res[i, max(0, j - ref_bd_min_gap) : j + ref_bd_min_gap])
                    if seg_sum == 0:
                        bd_res[i, j] = 1
                    elif seg_sum == 1 and bd_res[i, j] != 1:
                        bd_res[i, max(0, j - ref_bd_min_gap) : j + ref_bd_min_gap] = ref_bd_i[
                            max(0, j - ref_bd_min_gap) : j + ref_bd_min_gap
                        ]
                    elif seg_sum > 1:
                        for k in range(1, ref_bd_min_gap + 1):
                            if bd_res[i, max(0, j - k)] == 1 and ref_bd_i[max(0, j - k)] != 1:
                                bd_res[i, max(0, j - k)] = 0
                                break
                            if (
                                bd_res[i, min(bd_res.shape[1] - 1, j + k)] == 1
                                and ref_bd_i[min(bd_res.shape[1] - 1, j + k)] != 1
                            ):
                                bd_res[i, min(bd_res.shape[1] - 1, j + k)] = 0
                                break
                        bd_res[i, j] = 1
            # final check
            assert torch.sum(bd_res[i, ref_bd_i_js]) == len(ref_bd_i_js), (
                f"{torch.sum(bd_res[i, ref_bd_i_js])} {len(ref_bd_i_js)}"
            )

    bd_res = bd_res.to(device)

    # force valid begin and end
    bd_res[:, 0] = 0
    if non_padding is not None:
        for i in range(bd_res.shape[0]):
            bd_res[i, sum(non_padding[i]) - 1 :] = 0
    else:
        bd_res[:, -1] = 0

    return bd_res


def _align_word(word_durs: list[float], mel_len: int, hop_size: int, audio_sample_rate: int) -> np.ndarray:
    mel2word = np.zeros([mel_len], int)
    start_time = 0.0
    for i_word, wd in enumerate(word_durs):
        start_frame = int(start_time * audio_sample_rate / hop_size + 0.5)
        end_frame = int((start_time + wd) * audio_sample_rate / hop_size + 0.5)
        mel2word[start_frame:end_frame] = i_word + 1
        start_time += wd
    return mel2word


class PitchDecoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hidden_size = hidden_size = hparams["hidden_size"]
        self.dropout = hparams.get("dropout", 0.0)
        self.note_bd_out = nn.Linear(hidden_size, 1)
        self.note_bd_temperature = max(1e-7, hparams.get("note_bd_temperature", 1.0))

        # note prediction
        self.pitch_attn_num_head = hparams.get("pitch_attn_num_head", 1)
        self.multihead_dot_attn = nn.Linear(hidden_size, self.pitch_attn_num_head)
        self.post = ConvBlocks(
            hidden_size,
            out_dims=hidden_size,
            dilations=None,
            kernel_size=3,
            layers_in_block=1,
            c_multiple=1,
            post_net_kernel=3,
        )
        self.pitch_out = nn.Linear(hidden_size, hparams.get("note_num", 100) + 4)
        self.note_num = hparams.get("note_num", 100)
        self.note_start = hparams.get("note_start", 30)
        self.pitch_temperature = max(1e-7, hparams.get("note_pitch_temperature", 1.0))

    def forward(self, feat, note_bd, train=True):
        bsz, T, _ = feat.shape

        attn = torch.sigmoid(self.multihead_dot_attn(feat))  # [B, T, C] -> [B, T, num_head]
        attn_feat = feat.unsqueeze(3) * attn.unsqueeze(2)  # [B, T, C, 1] x [B, T, 1, num_head] -> [B, T, C, num_head]
        attn_feat = torch.mean(attn_feat, dim=-1)  # [B, T, C, num_head] -> [B, T, C]
        mel2note = torch.cumsum(note_bd, 1)
        note_length = torch.max(torch.sum(note_bd, dim=1)).item() + 1  # max length
        note_lengths = torch.sum(note_bd, dim=1) + 1  # [B]

        attn = torch.mean(attn, dim=-1, keepdim=True)  # [B, T, num_head] -> [B, T, 1]
        denom = mel2note.new_zeros(bsz, note_length, dtype=attn.dtype).scatter_add_(
            dim=1, index=mel2note, src=attn.squeeze(-1)
        )  # [B, T] -> [B, note_length] count the note frames of each note (with padding excluded)
        frame2note = mel2note.unsqueeze(-1).repeat(1, 1, self.hidden_size)  # [B, T] -> [B, T, C], with padding included
        note_aggregate = frame2note.new_zeros(bsz, note_length, self.hidden_size, dtype=attn_feat.dtype).scatter_add_(
            dim=1, index=frame2note, src=attn_feat
        )  # [B, T, C] -> [B, note_length, C]
        note_aggregate = note_aggregate / (denom.unsqueeze(-1) + 1e-5)
        note_logits = self.post(note_aggregate)
        note_logits = self.pitch_out(note_logits) / self.pitch_temperature
        # note_logits = torch.clamp(note_logits, min=-16., max=16.)     # don't know need it or not

        note_pred = torch.softmax(note_logits, dim=-1)  # [B, note_length, note_num]
        note_pred = torch.argmax(note_pred, dim=-1)  # [B, note_length]
        # for some reason, note idx maybe 130 (why?)
        note_pred[note_pred > self.note_num] = 0
        note_pred[note_pred < self.note_start] = 0

        return note_lengths, note_logits, note_pred


class MidiExtractor(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams.copy()
        self.hidden_size = hidden_size = hparams["hidden_size"]
        self.note_bd_threshold = hparams.get("note_bd_threshold", 0.5)
        self.note_bd_min_gap = round(
            hparams.get("note_bd_min_gap", 100) * hparams["audio_sample_rate"] / 1000 / hparams["hop_size"]
        )
        self.note_bd_ref_min_gap = round(
            hparams.get("note_bd_ref_min_gap", 50) * hparams["audio_sample_rate"] / 1000 / hparams["hop_size"]
        )

        self.mel_proj = nn.Conv1d(hparams["use_mel_bins"], hidden_size, kernel_size=3, padding=1)
        self.mel_encoder = ConvBlocks(
            hidden_size,
            out_dims=hidden_size,
            dilations=None,
            kernel_size=3,
            layers_in_block=2,
            c_multiple=1,
            post_net_kernel=3,
        )
        self.use_pitch = hparams.get("use_pitch_embed", True)
        if self.use_pitch:
            self.pitch_embed = Embedding(300, hidden_size, 0, "kaiming")
            self.uv_embed = Embedding(3, hidden_size, 0, "kaiming")
        self.use_wbd = hparams.get("use_wbd", True)
        if self.use_wbd:
            self.word_bd_embed = Embedding(3, hidden_size, 0, "kaiming")
        self.cond_encoder = ConvBlocks(
            hidden_size,
            out_dims=hidden_size,
            dilations=None,
            kernel_size=3,
            layers_in_block=1,
            c_multiple=1,
            post_net_kernel=3,
        )

        # backbone
        self.net = BackboneNet(hparams)

        # note bd prediction
        self.note_bd_out = nn.Linear(hidden_size, 1)
        self.note_bd_temperature = max(1e-7, hparams.get("note_bd_temperature", 1.0))

        # note prediction
        self.pitch_decoder = PitchDecoder(hparams)

    def run_encoder(self, mel, word_bd=None, pitch=None, uv=None):
        mel_embed = self.mel_proj(mel.transpose(1, 2)).transpose(1, 2)
        mel_embed = self.mel_encoder(mel_embed)
        pitch_embed = word_bd_embed = 0
        if self.use_pitch and pitch is not None and uv is not None:
            pitch_embed = self.pitch_embed(pitch) + self.uv_embed(uv)  # [B, T, C]
        if self.use_wbd and word_bd is not None:
            word_bd_embed = self.word_bd_embed(word_bd)
        feat = self.cond_encoder(mel_embed + pitch_embed + word_bd_embed)
        return feat

    def forward(self, mel, word_bd=None, note_bd=None, pitch=None, uv=None, non_padding=None, train=True):
        ret = {}
        bsz, T, _ = mel.shape

        feat = self.run_encoder(mel, word_bd, pitch, uv)
        feat = self.net(feat)  # [B, T, C]

        # note bd prediction
        # dropout has been dropped (inference mode)
        note_bd_logits = self.note_bd_out(feat).squeeze(-1) / self.note_bd_temperature
        note_bd_logits = torch.clamp(note_bd_logits, min=-16.0, max=16.0)
        ret["note_bd_logits"] = note_bd_logits  # [B, T]
        if note_bd is None or not train:
            note_bd = regulate_boundary(
                note_bd_logits,
                self.note_bd_threshold,
                self.note_bd_min_gap,
                word_bd,
                self.note_bd_ref_min_gap,
                non_padding,
            )
            ret["note_bd_pred"] = note_bd  # [B, T]

        # note pitch prediction
        note_lengths, note_logits, note_pred = self.pitch_decoder(feat, note_bd, train)
        ret["note_lengths"], ret["note_logits"], ret["note_pred"] = note_lengths, note_logits, note_pred

        return ret


class MelNet(nn.Module):
    def __init__(self, hparams, device="cpu") -> None:
        super().__init__()
        self.n_fft = hparams["fft_size"]
        self.num_mels = hparams["audio_num_mel_bins"]
        self.sampling_rate = hparams["audio_sample_rate"]
        self.hop_size = hparams["hop_size"]
        self.win_size = hparams["win_size"]
        self.fmin = hparams["fmin"]
        self.fmax = hparams["fmax"]
        self.device = device

        mel = librosa_mel_fn(
            sr=self.sampling_rate,
            n_fft=self.n_fft,
            n_mels=self.num_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        )
        self.mel_basis = mel.float().to(self.device)
        self.hann_window = torch.hann_window(self.win_size).to(self.device)

    def to(self, device, **kwagrs):
        super().to(device=device, **kwagrs)
        self.mel_basis = self.mel_basis.to(device)
        self.hann_window = self.hann_window.to(device)
        self.device = device

    def forward(self, y, center=False, complex=False):
        if isinstance(y, np.ndarray):
            y = torch.FloatTensor(y)
        if len(y.shape) == 1:
            y = y.unsqueeze(0)
        y = y.clamp(min=-1.0, max=1.0).to(self.device)

        pad_length = math.ceil(y.shape[1] / self.hop_size) * self.hop_size - y.shape[1]
        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            [int((self.n_fft - self.hop_size) / 2), int((self.n_fft - self.hop_size) / 2 + pad_length)],
            mode="reflect",
        )
        y = y.squeeze(1)

        spec = torch.stft(
            y,
            self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.hann_window,
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        if not complex:
            spec = torch.view_as_real(spec)
            spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))  # [B, n_fft, T]
            spec = torch.matmul(self.mel_basis, spec)
            spec = spectral_normalize_torch(spec)
            spec = spec.transpose(1, 2)  # [B, T, n_fft]
        else:
            B, C, T, _ = spec.shape
            spec = spec.transpose(1, 2)  # [B, T, n_fft, 2]
        return spec


class RosvotModel(nn.Module, SupportAudioInput, SupportsComponentDiscovery):
    support_audio_input: ClassVar[bool] = True
    _dit_modules: ClassVar[list[str]] = []
    _encoder_modules: ClassVar[list[str]] = ["."]
    _vae_modules: ClassVar[list[str]] = []
    _resident_modules: ClassVar[list[str]] = []
    _layerwise_offload_blocks_attrs: ClassVar[list[str]] = ["midi.net", "midi.pitch_decoder", "midi.cond_encoder"]

    def __init__(
        self,
        rosvot_ckpt: str | Path,
        *,
        config_path: str | Path = "",
        pe: nn.Module | None = None,
        the: float = 0.85,
        verbose: bool = False,
        rosvot_source_dir: str | Path | None = None,
    ):
        super().__init__()
        self.verbose = verbose
        ckpt = Path(rosvot_ckpt)
        resolved_config = Path(config_path) if config_path else ckpt.with_name("config.yaml")
        self.hparams = load_rosvot_config(resolved_config, hparams_str=f"note_bd_threshold={the}")
        if verbose:
            logger.info("ROSVOT config: %s", resolved_config)

        self.midi = MidiExtractor(self.hparams)
        self.mel_net = MelNet(self.hparams)
        self.pe = pe if pe is not None and self.hparams.get("use_pitch_embed", False) else None
        self._checkpoint_path = str(ckpt)
        self.load_checkpoint(str(ckpt), verbose=verbose)
        self.eval()

    def load_checkpoint(self, checkpoint_path: str | None = None, *, verbose: bool = False) -> None:
        load_model_ckpt(self.midi, checkpoint_path or self._checkpoint_path, verbose=verbose)

    @torch.no_grad()
    def forward(self, wav: torch.Tensor, word_durs: list[float]) -> dict[str, Any]:
        hparams = self.hparams
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)

        mel_len = (wav.shape[-1] + hparams["hop_size"] - 1) // hparams["hop_size"]
        min_word_dur = hparams.get("min_word_dur", 20) / 1000
        wd_raw = list(word_durs)
        word_durs_filtered: list[float] = []
        for i, wd in enumerate(wd_raw):
            if wd < min_word_dur:
                if i == 0 and len(wd_raw) > 1:
                    wd_raw[i + 1] += wd
                elif word_durs_filtered:
                    word_durs_filtered[-1] += wd
            else:
                word_durs_filtered.append(wd)

        mel2word = _align_word(word_durs_filtered, mel_len, hparams["hop_size"], hparams["audio_sample_rate"])
        if mel2word.size > 0 and mel2word[0] == 0:
            mel2word = mel2word + 1

        real_len = min(mel_len, int(np.sum(mel2word > 0)))
        T = math.ceil(min(real_len, hparams["max_frames"]) / hparams["frames_multiple"]) * hparams["frames_multiple"]
        device = wav.device
        self.mel_net.to(device)
        wav_t = pad_or_cut_xd(wav.float(), T * hparams["hop_size"], 1)

        pitch_coarse = uv_t = None
        if self.pe is not None:
            f0s, uvs = self.pe.get_pitch_batch(
                wav_t,
                sample_rate=hparams["audio_sample_rate"],
                hop_size=hparams["hop_size"],
                lengths=[real_len],
                fmax=hparams["f0_max"],
                fmin=hparams["f0_min"],
            )
            f0_1d, uv_1d = norm_interp_f0(f0s[0][:T])
            f0_t = pad_or_cut_xd(torch.as_tensor(f0_1d, device=device, dtype=torch.float32), T, 0).unsqueeze(0)
            uv_t = pad_or_cut_xd(torch.as_tensor(uv_1d, device=device, dtype=torch.float32), T, 0).long().unsqueeze(0)
            pitch_coarse = f0_to_coarse(denorm_f0(f0_t, uv_t)).to(device)

        mel = pad_or_cut_xd(self.mel_net(wav_t)[0], T, dim=0).unsqueeze(0)
        mel_nonpadding_mask = torch.zeros(1, T, device=device)
        mel_nonpadding_mask[:, :real_len] = 1.0
        mel = (mel.transpose(1, 2) * mel_nonpadding_mask.unsqueeze(1)).transpose(1, 2)
        mel_nonpadding = mel.abs().sum(-1) > 0

        mel2word_t = pad_or_cut_xd(torch.as_tensor(mel2word, device=device, dtype=torch.long), T, 0)
        word_bd = torch.zeros_like(mel2word_t)
        word_bd[1:] = (mel2word_t[1:] != mel2word_t[:-1]).long()
        word_bd[real_len:] = 0
        word_bd = word_bd.unsqueeze(0)

        outputs = self.midi(
            mel=mel[:, :, : hparams.get("use_mel_bins", 80)],
            word_bd=word_bd,
            pitch=pitch_coarse,
            uv=uv_t,
            non_padding=mel_nonpadding,
        )
        outputs["word_durs_filtered"] = word_durs_filtered
        outputs["real_len"] = real_len
        outputs["word_bd"] = word_bd
        return outputs

    @staticmethod
    def _load_wav(wav_src: str | np.ndarray, sample_rate: int, *, src_sample_rate: int | None = None) -> np.ndarray:
        if isinstance(wav_src, str):
            wav, _ = load_mono_audio(wav_src, target_sr=sample_rate)
            return wav
        wav = np.asarray(wav_src, dtype=np.float32)
        if src_sample_rate is not None and src_sample_rate != sample_rate:
            wav = resample_mono(wav, orig_sr=src_sample_rate, target_sr=sample_rate)
        return wav

    @staticmethod
    def _normalize_note2words(note2words: list[int]) -> list[int]:
        if not note2words:
            return []
        out = [note2words[0]]
        for idx in range(1, len(note2words)):
            out.append(max(note2words[idx], out[-1]))
        return out

    @staticmethod
    def _build_ep_types(note2words: list[int], align_words: list[str]) -> list[int]:
        ep_types: list[int] = []
        prev = -1
        for i, w in zip(note2words, align_words):
            ep_types.append(1 if w == "<SP>" else (2 if i != prev else 3))
            prev = i
        return ep_types

    @torch.no_grad()
    def transcribe(
        self,
        item: dict[str, Any],
        *,
        segment_info: dict[str, Any] | None = None,
        verbose: bool = False,
    ) -> dict[str, Any]:
        if "word_durs" not in item:
            raise ValueError('item must contain "word_durs" from lyric transcription')

        if item.get("wav_fn"):
            wav = self._load_wav(item["wav_fn"], self.hparams["audio_sample_rate"])
        elif item.get("wav") is not None:
            wav = self._load_wav(
                item["wav"],
                self.hparams["audio_sample_rate"],
                src_sample_rate=item.get("sample_rate"),
            )
        else:
            raise ValueError('item must contain "wav_fn" or "wav"')

        device = next(self.parameters()).device
        outputs = self(torch.from_numpy(wav).float().to(device), list(item["word_durs"]))
        real_len = int(outputs["real_len"])
        word_durs_filtered = outputs["word_durs_filtered"]
        word_bd = outputs["word_bd"]
        item_name = item.get("item_name", "")

        note_lengths = outputs["note_lengths"].detach().cpu().numpy()
        note_bd_pred = outputs["note_bd_pred"][0].detach().cpu().numpy()[:real_len]
        note_pred = outputs["note_pred"][0].detach().cpu().numpy()[: note_lengths[0]]

        if note_pred.shape == (0,):
            rosvot_out = {"item_name": item_name, "pitches": [], "note_durs": [], "note2words": None}
        else:
            note_itv_pred = boundary2Interval(note_bd_pred)
            word_bd_for_reg = word_bd[0].detach().cpu().numpy()[:real_len]
            hop = self.hparams["hop_size"]
            sr = self.hparams["audio_sample_rate"]

            if self.hparams.get("infer_regulate_real_note_itv", True):
                try:
                    note_itv_pred_secs, note2words = regulate_real_note_itv(
                        note_itv_pred,
                        note_bd_pred,
                        word_bd_for_reg,
                        np.array(word_durs_filtered),
                        hop,
                        sr,
                    )
                    note_pred, note_itv_pred_secs, note2words = regulate_ill_slur(
                        note_pred, note_itv_pred_secs, note2words
                    )
                except Exception:
                    if verbose:
                        logger.exception("ROSVOT postprocess failed")
                    note_itv_pred_secs = note_itv_pred * hop / sr
                    note2words = None
            else:
                note_itv_pred_secs = note_itv_pred * hop / sr
                note2words = None

            rosvot_out = {
                "item_name": item_name,
                "pitches": note_pred.tolist(),
                "note_durs": [float(itv[1] - itv[0]) for itv in note_itv_pred_secs],
                "note2words": note2words.tolist() if note2words is not None else None,
            }

        note2words_raw = rosvot_out.get("note2words") or []
        align_words = [item["words"][idx - 1] for idx in note2words_raw if 0 < idx <= len(item["words"])]
        ep_types = self._build_ep_types(self._normalize_note2words(note2words_raw), align_words) if align_words else []
        seg = segment_info or item

        return {
            "item_name": seg.get("item_name", item_name),
            "wav_fn": seg.get("wav_fn", item.get("wav_fn", "")),
            "origin_wav_fn": seg.get("origin_wav_fn", item.get("origin_wav_fn", "")),
            "start_time_ms": seg.get("start_time_ms", ""),
            "end_time_ms": seg.get("end_time_ms", ""),
            "language": item.get("language", ""),
            "note_text": align_words,
            "note_dur": rosvot_out.get("note_durs", []),
            "note_type": ep_types,
            "note_pitch": rosvot_out.get("pitches", []),
        }
