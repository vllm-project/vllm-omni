"""
IndexTTS Model Implementation for vLLM-Omni
Github URL: https://github.com/index-tts/index-tts
License: https://github.com/index-tts/index-tts/blob/main/LICENSE


DESCRIPTION:
- Speaker conditioning: W2V-BERT features + CAMPPlus style + semantic codec
- Emotion conditioning: Similar W2V-BERT extraction, blendable with vectors
- GPT stage: Generates discrete mel codes from text + conditioning
- S2Mel stage: Flow matching to generate mel spectrogram from codes
- Vocoder: BigVGAN converts mel to waveform
"""

import random
from collections.abc import Iterable

import librosa
import torch
import torch.nn.functional as F
import torchaudio
from huggingface_hub import hf_hub_download
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
from vllm.config import VllmConfig  # type: ignore
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper

from vllm_omni.model_executor.models.index_tts.index_tts_config import IndexTTSConfig
from vllm_omni.model_executor.models.index_tts.index_tts_vocoder import IndexTTSVocoderForConditionalGeneration
from vllm_omni.model_executor.models.index_tts.utils.qwen_emotion import QwenEmotion

from .s2mel.modules.audio import mel_spectrogram
from .s2mel.modules.campplus.DTDNN import CAMPPlus
from .utils.front import TextNormalizer, TextTokenizer

logger = init_logger(__name__)


def find_most_similar_cosine(query_vector, matrix):
    query_vector = query_vector.float()
    matrix = matrix.float()

    similarities = F.cosine_similarity(query_vector, matrix, dim=1)
    most_similar_index = torch.argmax(similarities)
    return most_similar_index


class IndexTTSForConditionalGeneration(torch.nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config: IndexTTSConfig = vllm_config.model_config.hf_config
        self.prefix = prefix
        self.model_stage = vllm_config.model_config.model_stage

        if self.model_stage == "gpt":
            from vllm_omni.model_executor.models.index_tts.index_tts_gpt import IndexTTSGPTForConditionalGeneration

            # Load ONLY GPT stage
            self.gpt_stage = IndexTTSGPTForConditionalGeneration(vllm_config=vllm_config, prefix=prefix)
            self.model = self.gpt_stage

        elif self.model_stage == "s2mel":
            from vllm_omni.model_executor.models.index_tts.index_tts_s2mel import IndexTTSS2MelForConditionalGeneration

            self.s2mel_stage = IndexTTSS2MelForConditionalGeneration(vllm_config=vllm_config, prefix=prefix)
            self.model = self.s2mel_stage
        elif self.model_stage == "vocoder":
            self.vocoder_stage = IndexTTSVocoderForConditionalGeneration(vllm_config=vllm_config, prefix=prefix)
            self.model = self.vocoder_stage
        else:
            raise ValueError(f"Unknown model_stage: {self.model_stage}")

        self._init_code2wav()
        self.emo_num = list(self.config.emo_num)
        # self.stop_mel_token = self.config.gpt["stop_mel_token"]

    # def _init_talker(self):
    #     self.gpt = UnifiedVoice(**self.config.gpt)
    #     self.gpt_path = self.config.gpt_checkpoint
    #     # Defer building the inference wrapper until after weights are loaded

    def _init_code2wav(self):
        # Defer vocoder instantiation to load_weights to avoid heavy IO in __init__
        self.bigvgan = None  # will be set in load_weights via from_pretrained

        # Semantic Model (deferred to load_weights)
        self.semantic_model = None  # will be set in load_weights
        self.semantic_mean = None  # will be set in load_weights if provided
        self.semantic_std = None  # will be set in load_weights if provided

        # CAMPPlus
        # campplus_ckpt_path = hf_hub_download("funasr/campplus", filename="campplus_cn_common.bin")
        self.campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        # campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))

        # Mel Spectrogram Function
        mel_fn_args = {
            "n_fft": self.config.s2mel["preprocess_params"]["spect_params"]["n_fft"],
            "win_size": self.config.s2mel["preprocess_params"]["spect_params"]["win_length"],
            "hop_size": self.config.s2mel["preprocess_params"]["spect_params"]["hop_length"],
            "num_mels": self.config.s2mel["preprocess_params"]["spect_params"]["n_mels"],
            "sampling_rate": self.config.s2mel["preprocess_params"]["sr"],
            "fmin": self.config.s2mel["preprocess_params"]["spect_params"].get("fmin", 0),
            "fmax": None
            if self.config.s2mel["preprocess_params"]["spect_params"].get("fmax", "None") == "None"
            else 8000,
            "center": False,
        }
        self.mel_fn = lambda x: mel_spectrogram(x, **mel_fn_args)
        # Text frontend placeholders (initialized in load_weights)
        self.normalizer = None
        self.tokenizer = None

    # helper methods
    @torch.no_grad()
    def get_emb(self, input_features, attention_mask):
        vq_emb = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[17]  # (B, T, C)
        feat = (feat - self.semantic_mean) / self.semantic_std
        return feat

    def _load_and_cut_audio(self, audio_path, max_audio_length_seconds, verbose=False, sr=None):
        if not sr:
            audio, sr = librosa.load(audio_path)
        else:
            audio, _ = librosa.load(audio_path, sr=sr)
        audio = torch.tensor(audio).unsqueeze(0)
        max_audio_samples = int(max_audio_length_seconds * sr)

        if audio.shape[1] > max_audio_samples:
            # if verbose:
            # print(f"Audio too long ({audio.shape[1]} samples), truncating to {max_audio_samples} samples")
            audio = audio[:, :max_audio_samples]
        return audio, sr

    def _prepare_speaker_conditioning(self, spk_audio_prompt, device):
        audio, sr = self._load_and_cut_audio(spk_audio_prompt, 15)
        #   1. Load audio at 22.05kHz and 16kHz
        audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
        audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)

        #   2. Extract W2V-BERT features from 16kHz audio:
        inputs = self.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"]
        attention_mask = inputs["attention_mask"]
        input_features = input_features.to(device)
        attention_mask = attention_mask.to(device)
        spk_cond_emb = self.get_emb(input_features, attention_mask)

        #   5. Extract CAMPPlus style from 16kHz:
        feat = torchaudio.compliance.kaldi.fbank(
            audio_16k.to(spk_cond_emb.device), num_mel_bins=80, dither=0, sample_frequency=16000
        )
        feat = feat - feat.mean(dim=0, keepdim=True)  # feat2另外一个滤波器能量组特征[922, 80]
        style = self.campplus_model(feat.unsqueeze(0))  # 参考音频的全局style2[1,192]

        #   4. Generate reference mel from 22kHz audio
        ref_mel = self.mel_fn(audio_22k.to(spk_cond_emb.device).float())
        ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(ref_mel.device)

        return spk_cond_emb, style, ref_mel, ref_target_lengths

    def _prepare_emotion_conditioning(self, emo_audio_prompt, device):
        verbose = False
        #   1. Load audio at 16kHz
        emo_audio, _ = self._load_and_cut_audio(emo_audio_prompt, 15, verbose, sr=16000)
        #   2. Extract features
        emo_inputs = self.extract_features(emo_audio, sampling_rate=16000, return_tensors="pt")
        emo_input_features = emo_inputs["input_features"]
        emo_attention_mask = emo_inputs["attention_mask"]
        emo_input_features = emo_input_features.to(device)
        emo_attention_mask = emo_attention_mask.to(device)
        #   3. Get emo_cond_emb
        emo_cond_emb = self.get_emb(emo_input_features, emo_attention_mask)

        return emo_cond_emb

    # Utility methods
    def _interval_silence(self, wavs, sampling_rate=22050, interval_silence=200):
        """
        Silences to be insert between generated segments.
        """

        if not wavs or interval_silence <= 0:
            return wavs

        # get channel_size
        channel_size = wavs[0].size(0)
        # get silence tensor
        sil_dur = int(sampling_rate * interval_silence / 1000.0)
        return torch.zeros(channel_size, sil_dur)

    def _insert_interval_silence(self, wavs, sampling_rate=22050, interval_silence=200):
        """Insert silence between audio segments."""
        if not wavs or interval_silence <= 0:
            return wavs

        channel_size = wavs[0].size(0)
        sil_dur = int(sampling_rate * interval_silence / 1000.0)
        sil_tensor = torch.zeros(channel_size, sil_dur)

        wavs_list = []
        for i, wav in enumerate(wavs):
            wavs_list.append(wav)
            if i < len(wavs) - 1:
                wavs_list.append(sil_tensor)
        return wavs_list

    def _get_emo_vector(
        self,
        use_emo_text,
        emo_vector,
        emo_audio_prompt,
        emo_alpha,
        emo_text,
        spk_audio_prompt,
        text,
        use_random,
        style,
        device,
    ):
        if use_emo_text or emo_vector is not None:
            # we're using a text or emotion vector guidance; so we must remove
            # "emotion reference voice", to ensure we use correct emotion mixing!
            emo_audio_prompt = None

        if use_emo_text:
            # automatically generate emotion vectors from text prompt
            if emo_text is None:
                emo_text = text  # use main text prompt
            emo_dict = self.qwen_emo.inference(emo_text)
            # print(f"detected emotion vectors from text: {emo_dict}")
            # convert ordered dict to list of vectors; the order is VERY important!
            emo_vector = list(emo_dict.values())

        if emo_vector is not None:
            # we have emotion vectors; they can't be blended via alpha mixing
            # in the main inference process later, so we must pre-calculate
            # their new strengths here based on the alpha instead!
            emo_vector_scale = max(0.0, min(1.0, emo_alpha))
            if emo_vector_scale != 1.0:
                # scale each vector and truncate to 4 decimals (for nicer printing)
                emo_vector = [int(x * emo_vector_scale * 10000) / 10000 for x in emo_vector]
                # print(f"scaled emotion vectors to {emo_vector_scale}x: {emo_vector}")

        if emo_audio_prompt is None:
            # we are not using any external "emotion reference voice"; use
            # speaker's voice as the main emotion reference audio.
            emo_audio_prompt = spk_audio_prompt
            # must always use alpha=1.0 when we don't have an external reference voice
            emo_alpha = 1.0

        emovec_mat = None
        weight_vector = None
        ### Prepare emotion vector blending if provided
        if emo_vector is not None:
            weight_vector = torch.tensor(emo_vector, device=device)
            # 1. Find most similar speaker if not random
            if use_random:
                random_index = [random.randint(0, x - 1) for x in self.emo_num]
            else:
                random_index = [find_most_similar_cosine(style, tmp) for tmp in self.spk_matrix]

            # 2. Get emotion prototypes
            emo_matrix = [tmp[index].unsqueeze(0) for index, tmp in zip(random_index, self.emo_matrix)]
            emo_matrix = torch.cat(emo_matrix, 0)

            # 3. Blend emotion vectors
            emovec_mat = weight_vector.unsqueeze(1) * emo_matrix
            emovec_mat = torch.sum(emovec_mat, 0)
            emovec_mat = emovec_mat.unsqueeze(0)

        return emo_vector, emo_audio_prompt, emo_alpha, emovec_mat, weight_vector

    # ============================================================================
    # MAIN FORWARD METHOD - Complete TTS Pipeline
    # ============================================================================

    def forward(
        self,
        spk_audio_prompt,  # Speaker reference audio (path or tensor)
        text: str,  # Input text to synthesize
        output_path: str | None = None,  # Optional: save output to file
        emo_audio_prompt=None,  # Optional: emotion reference audio
        emo_alpha: float = 1.0,  # Emotion blending strength (0.0-1.0)
        emo_vector=None,  # Optional: explicit emotion vector [8 floats]
        use_emo_text: bool = False,  # Use QwenEmotion to extract from text
        emo_text: str | None = None,  # Text for emotion extraction
        max_text_tokens_per_segment: int = 120,  # Split long text into segments
        max_mel_tokens: int = 1500,  # Max tokens for GPT generation
        verbose: bool = False,
        quick_streaming_tokens=0,
        use_random: bool = False,  # Use random speaker prototypes for emo_vector blending
        interval_silence: int = 200,  # ADD THIS
        **generation_kwargs,  # Additional args: top_p, top_k, temperature, etc.
    ) -> torch.Tensor:
        """
        Complete IndexTTS inference pipeline.

        Reference: infer_v2.py lines 382-708 (infer_generator method)

        Returns: waveform tensor [1, T] at 22.05kHz
        """
        # 1. Implementing preprocessing and setup
        device = next(self.parameters()).device

        if self.model_stage == "gpt":
            ### Prepare speaker conditioning
            spk_cond_emb, style, _, _ = self._prepare_speaker_conditioning(spk_audio_prompt, device)

            emo_vector, emo_audio_prompt, emo_alpha, emovec_mat = self._get_emo_vector(
                use_emo_text,
                emo_vector,
                emo_audio_prompt,
                emo_alpha,
                emo_text,
                spk_audio_prompt,
                text,
                use_random,
                style,
                device,
            )
            ### Prepare emotion conditioning
            emo_cond_emb = self._prepare_emotion_conditioning(emo_audio_prompt, device)
            ### Tokenize and segment text
            text_tokens_list = self.tokenizer.tokenize(text)
            # segments = self.tokenizer.split_segments(
            #     text_tokens_list, max_text_tokens_per_segment, quick_streaming_tokens=quick_streaming_tokens
            # )
            output = self.model(
                text_tokens=text_tokens_list,
                spk_audio_prompt=spk_audio_prompt,
                spk_cond_emb=spk_cond_emb,
                emo_cond_emb=emo_cond_emb,
                emovec_mat=emovec_mat,
                emo_vector=emo_vector,
                emo_alpha=emo_alpha,
                max_mel_tokens=max_mel_tokens,
                **generation_kwargs,
            )
            return output

        elif self.model_stage == "s2mel":
            _, _, ref_mel, ref_target_lengths = self._prepare_speaker_conditioning(spk_audio_prompt, device)
            # output = self.model(
            #     codes=codes,
            #     latent=latent,
            #     code_lens=code_lens,
            #     spk_cond_emb=spk_cond_emb,
            #     ref_target_lengths=ref_target_lengths,
            #     ref_mel=ref_mel,
            #     style=style,
            #     **generation_kwargs,
            # )
            # return output
            pass
        elif self.model_stage == "vocoder":
            pass

        sampling_rate = 22050

        ### Main Synthesis loop
        # for seg_idx, sent in enumerate(segments):
        #     # 11.1 Prepare text tokens for segment
        #     text_tokens = self.tokenizer.convert_tokens_to_ids(sent)
        #     text_tokens = torch.tensor(text_tokens, dtype=torch.int32, device=device).unsqueeze(0)

        # Post processing
        wavs = []
        wavs = self._insert_interval_silence(wavs, sampling_rate=sampling_rate, interval_silence=interval_silence)
        wav = torch.cat(wavs, dim=1)
        # wav_length = wav.shape[-1] / sampling_rate

        wav = wav.cpu()  # to cpu
        return wav
        # return self.model(*args, **kwargs)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Strict weight loading from a HF repo + engine stream.

        - Requires repo_id and all configured file paths to exist.
        - No silent fallbacks; any failure raises.
        - Uses AutoWeightsLoader for device/dtype placement.
        """
        mapper = WeightsMapper(
            orig_to_new_prefix={
                # "gpt.": "gpt.",
                # "s2mel.": "s2mel.",
                # "semantic_codec.": "semantic_codec.",
                # "bigvgan.": "bigvgan.",
                "campplus_model.": "campplus_model.",
                "index_tts.": "",
            }
        )

        merged: list[tuple[str, torch.Tensor]] = list(weights)

        # Require repo and essential paths
        repo_id = getattr(self.config, "repo_id", None) or "IndexTeam/IndexTTS-2"
        if not isinstance(repo_id, str) or not repo_id:
            raise RuntimeError("IndexTTS: config.repo_id must be a non-empty string")

        # Text tokenizer / normalizer from HF repo (bpe.model)
        bpe_rel = getattr(self.config, "bpe_model", None) or "bpe.model"
        bpe_path = hf_hub_download(repo_id, filename=bpe_rel)
        normalizer = TextNormalizer(enable_glossary=True)
        normalizer.load()
        self.normalizer = normalizer
        self.tokenizer = TextTokenizer(bpe_path, normalizer)

        def _load_pt_with_prefix(path: str, prefix: str) -> list[tuple[str, torch.Tensor]]:
            sd = torch.load(path, map_location="cpu")
            # Handle different checkpoint formats (order matters!)
            if isinstance(sd, dict) and "model" in sd:
                sd = sd["model"]
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            if isinstance(sd, dict) and "generator" in sd and prefix.startswith("bigvgan"):
                sd = sd["generator"]
            pairs: list[tuple[str, torch.Tensor]] = []
            for name, tensor in sd.items():
                if isinstance(tensor, torch.Tensor):
                    pairs.append((f"{prefix}{name}", tensor))
            if not pairs:
                raise RuntimeError(f"IndexTTS: No tensors found in checkpoint: {path}")
            return pairs

        # with safe_open(sc_path, framework="pt") as f:
        #     keys = list(f.keys())
        #     if not keys:
        #         raise RuntimeError(f"IndexTTS: No tensors in safetensors: {sc_rel}")
        #     for name in keys:
        #         merged.append((f"semantic_codec.{name}", f.get_tensor(name)))

        # Optional strict components if specified in config
        bg_rel = self.config.vocoder.get("generator_path", None)
        if bg_rel:
            bg_path = hf_hub_download(repo_id, filename=bg_rel)
            merged.extend(_load_pt_with_prefix(bg_path, "bigvgan."))

        cp_rel = self.config.vocoder.get("campplus_path", None) or getattr(self.config, "campplus_path", None)
        if cp_rel:
            cp_path = hf_hub_download(repo_id, filename=cp_rel)
            merged.extend(_load_pt_with_prefix(cp_path, "campplus_model."))

        # Load MaskGCT W2V-BERT + extractor using HF subfolders, but load stats from main IndexTTS repo
        stats_rel = getattr(self.config, "w2v_stat", None)
        if not isinstance(stats_rel, str) or not stats_rel:
            raise RuntimeError("IndexTTS: config.w2v_stat must be provided (e.g., 'wav2vec2bert_stats.pt')")
        stats_path = hf_hub_download(repo_id, filename=stats_rel)
        stats = torch.load(stats_path, map_location="cpu")
        if not (isinstance(stats, dict) and "mean" in stats and "var" in stats):
            raise RuntimeError("IndexTTS: stats must be a dict with 'mean' and 'var'")
        self.semantic_mean = stats["mean"]
        self.semantic_std = torch.sqrt(stats["var"])  # align with build_semantic_model

        # Load via AutoWeightsLoader
        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(merged, mapper=mapper)

        # Load CAMPPlus from its HF repo (defaults to funasr/campplus + campplus_cn_common.bin)
        camp_repo = self.config.vocoder.get("campplus_repo", "funasr/campplus")
        camp_file = self.config.vocoder.get("campplus_filename", "campplus_cn_common.bin")
        camp_path = hf_hub_download(camp_repo, filename=camp_file)
        camp_sd = torch.load(camp_path, map_location="cpu")
        self.campplus_model.load_state_dict(camp_sd)
        self.campplus_model.eval()
        # Load W2V-BERT from official HF repo with matching extractor
        self.semantic_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0").eval()
        self.extract_features = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

        # Move from_pretrained-loaded modules and s2mel to the same device as model
        device = next(self.parameters()).device
        self.campplus_model.to(device)
        self.semantic_model.to(device)

        # Emotion/Speaker prototype matrices (for emotion blending)
        emo_rel = self.config.emo_matrix
        spk_rel = self.config.spk_matrix
        emo_path = hf_hub_download(repo_id, filename=emo_rel)
        spk_path = hf_hub_download(repo_id, filename=spk_rel)
        emo_full = torch.load(emo_path, map_location="cpu").to(device)
        spk_full = torch.load(spk_path, map_location="cpu").to(device)
        self.emo_num = list(self.config.emo_num)
        self.emo_matrix = torch.split(emo_full, self.emo_num)
        self.spk_matrix = torch.split(spk_full, self.emo_num)

        # QwenEmotion (optional but strict when configured): load from local path or HF subfolder
        qwen_emo_path = getattr(self.config, "qwen_emo_path", None)
        if not isinstance(qwen_emo_path, str) or not qwen_emo_path:
            raise RuntimeError("IndexTTS: config.qwen_emo_path must be provided (HF subfolder name)")
        # Load QwenEmotion strictly from HF repo subfolder (no local model_dir usage)
        # try:
        qwen_emo_repo_id = "dsinghvi/qwen0.6bemo4-merge"
        self.qwen_emo = QwenEmotion(repo_id=qwen_emo_repo_id)  # , subfolder=qwen_emo_path)
        # except Exception as e:
        # raise RuntimeError(
        # f"IndexTTS: Failed to load QwenEmotion from repo={repo_id} subfolder={qwen_emo_path}: {e}"
        # )

        # Single summary log
        total_bytes = sum(
            p.data.numel() * p.data.element_size()
            for _, p in self.named_parameters()
            if p is not None and p.data is not None
        )
        device = next(self.parameters()).device
        logger.debug(
            "[Model Loaded] name=%s, size=%.2f MB, device=%s",
            self.__class__.__name__,
            total_bytes / (1024**2),
            str(device),
        )
        print(loaded)

        return self.model.load_weights(weights)
        # return loaded
