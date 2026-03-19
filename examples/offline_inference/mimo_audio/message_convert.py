import base64
import io
import os
import random
import re
from collections.abc import Callable

import librosa
import numpy as np
import torch
import torchaudio
from process_speechdata import InputSegment, StreamingInputSegment
from torchaudio.transforms import MelSpectrogram

speech_zeroemb_idx = 151667
empty_token = "<|empty|>"
mimo_audio_tokenizer = None
device = "cpu"

asr_zh_templates = [
    "请将这段语音转换为文字",
    "帮我识别这个音频文件中的内容",
    "把这段录音转成文本",
    "请转录这段语音",
    "将音频内容转换成文字格式",
    "识别并转写这段语音",
    "把语音内容写成文字",
    "转录这个音频片段",
    "将这段对话转换为文本",
    "麻烦帮我把这段录音整理成详细的文字记录",
]

asr_en_templates = [
    "Please transcribe this audio file",
    "Convert this speech recording to text",
    "Transcribe the following voice message",
    "Turn this audio into readable text",
    "Please convert the recording to written format",
    "Transcribe what you hear in this audio",
    "Convert this spoken content to text",
    "Please write down what is said in this recording",
    "Transcribe this voice recording",
    "Could you please help me transcribe this important recording?",
    "Would you mind converting this voice message into a readable text format?",
    "I'd really appreciate it if you could turn this audio file into a written document",
]

tts_zh_templates = [
    "请将这段文字转换为语音",
    "帮我把这个文本读出来",
    "将这些文字生成音频",
    "请朗读这段内容",
    "把这段话转换成语音文件",
    "生成这段文字的语音版本",
    "请用语音播报这些内容",
    "将文本转换为可听的音频",
    "帮我朗读这段文字",
    "把这些内容念出来",
]

tts_en_templates = [
    "Please convert this text to speech",
    "Turn this writing into audio",
    "Generate speech from this text",
    "Read this content out loud",
    "Convert these words to voice",
    "Create an audio version of this text",
    "Please vocalize this content",
    "Turn this text into audible format",
    "Help me convert this writing to speech",
    "Make this text into spoken audio",
]


def detect_language(text):
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    else:
        return "en"


# ============================================
# Common Helper Functions - InputSegment Creation
# ============================================


def create_segment(text: str = "", audio=None) -> InputSegment:
    """Create a standard InputSegment with default zeroemb parameters"""
    return InputSegment(
        text=text,
        audio=audio,
        speech_zeroemb_idx=speech_zeroemb_idx,
        text_zeroemb_idx=empty_token,
    )


def create_streaming_segment(
    text: str, audio, tokenizer, group_size: int, audio_channels: int
) -> StreamingInputSegment:
    """Create a StreamingInputSegment"""
    return StreamingInputSegment(
        text=text,
        audio=audio,
        tokenizer=tokenizer,
        group_size=group_size,
        audio_channels=audio_channels,
        speech_zeroemb_idx=speech_zeroemb_idx,
        text_zeroemb_idx=empty_token,
    )


# ============================================
# Common Helper Functions - Common Token Segment Creation
# ============================================


def create_user_start() -> InputSegment:
    """Create user role start token"""
    return create_segment(text="<|im_start|>user\n")


def create_user_end() -> InputSegment:
    """Create role end token"""
    return create_segment(text="<|im_end|>\n")


def create_assistant_start() -> InputSegment:
    """Create assistant role start token"""
    return create_segment(text="<|im_start|>assistant\n")


def create_system_start() -> InputSegment:
    """Create system role start token"""
    return create_segment(text="<|im_start|>system\n")


def create_thinking_segment(thinking: bool = False) -> InputSegment:
    """Create thinking token, closed or open based on thinking parameter"""
    if thinking:
        return create_segment(text="<think>\n")
    else:
        return create_segment(text="<think>\n\n</think>\n")


def create_sostm_segment() -> InputSegment:
    """Create streaming output start token"""
    return create_segment(text="<|sostm|>")


def create_assistant_start_with_sostm() -> InputSegment:
    """Create assistant start token with sostm"""
    return create_segment(text="<|im_start|>assistant\n<|sostm|>")


def create_assistant_start_with_think() -> InputSegment:
    """Create assistant start token with think opening"""
    return create_segment(text="<|im_start|>assistant\n<think>\n")


# ============================================
# Common Helper Functions - Composite Segment Creation
# ============================================


def create_user_turn_with_audio(audio_tokenized, extra_text: str = None) -> list[InputSegment]:
    """Create a user turn containing audio"""
    segments = [
        create_user_start(),
        create_segment(audio=audio_tokenized),
    ]
    if extra_text:
        segments.append(create_segment(text=extra_text))
    segments.append(create_user_end())
    return segments


def create_user_turn_with_text(text: str) -> list[InputSegment]:
    """Create a text-only user turn"""
    return [
        create_user_start(),
        create_segment(text=text),
        create_user_end(),
    ]


def create_system_turn_with_voice_prompt(prompt_text: str, audio_token) -> list[InputSegment]:
    """Create a system turn with voice prompt"""
    return [
        create_system_start(),
        create_segment(text=prompt_text),
        create_segment(text="", audio=audio_token),
        create_user_end(),
    ]


def create_system_turn_text_only(system_text: str) -> list[InputSegment]:
    """Create a text-only system turn"""
    return [
        create_system_start(),
        create_segment(text=system_text),
        create_user_end(),
    ]


# ============================================
# Common Helper Functions - Multi-turn Dialogue Processing
# ============================================


def process_multiturn_messages(
    message_list: list[dict],
    user_processor: Callable[[dict], list[InputSegment]],
    assistant_processor: Callable[[dict], list[InputSegment]],
) -> list[InputSegment]:
    """
    Generic multi-turn dialogue message processing function

    Args:
        message_list: List of messages, each containing 'role' and 'content'
        user_processor: Function to process user messages
        assistant_processor: Function to process assistant messages

    Returns:
        Processed list of InputSegments
    """
    lm_prompt = []
    for message in message_list:
        role = message["role"]
        if role == "user":
            lm_prompt.extend(user_processor(message))
        elif role == "assistant":
            lm_prompt.extend(assistant_processor(message))
        else:
            raise ValueError(f"Invalid role: {role}")
    return lm_prompt


def create_text_user_message(message: dict) -> list[InputSegment]:
    """Process a text-only user message"""
    return [
        create_user_start(),
        create_segment(text=message["content"]),
        create_user_end(),
    ]


def create_text_assistant_message(message: dict) -> list[InputSegment]:
    """Process a text-only assistant message"""
    return [
        create_assistant_start(),
        create_segment(text=message["content"]),
        create_user_end(),
    ]


def create_audio_user_message(message: dict) -> list[InputSegment]:
    """Process an audio user message"""
    return [
        create_user_start(),
        create_segment(audio=preprocess_input(message["content"])),
        create_user_end(),
    ]


def append_assistant_ending(
    lm_prompt: list[InputSegment], thinking: bool = False, use_sostm: bool = False
) -> list[InputSegment]:
    """
    Append assistant ending to the prompt

    Args:
        lm_prompt: Existing prompt list
        thinking: Whether to use open thinking token
        use_sostm: Whether to use sostm token (for speech output)
    """
    if use_sostm:
        lm_prompt.append(create_assistant_start_with_sostm())
    else:
        lm_prompt.append(create_assistant_start())
        lm_prompt.append(create_thinking_segment(thinking))
    return lm_prompt


def get_asr_sft_prompt(
    input: None | str = None,
):
    """Build prompt for ASR (Automatic Speech Recognition) task"""
    audio_tokenized = preprocess_input(input)
    template = random.choice(asr_zh_templates + asr_en_templates)

    lm_prompt = create_user_turn_with_audio(audio_tokenized, extra_text=template)
    lm_prompt = append_assistant_ending(lm_prompt, thinking=False)
    return lm_prompt


def resample_audio_if_needed(wav_tensor: torch.Tensor, original_sr: int):
    target_sr = 24000
    if original_sr != target_sr:
        wav_tensor = torchaudio.functional.resample(wav_tensor, original_sr, target_sr)
    return wav_tensor


def wav2mel(wav, device="cpu"):
    mel_transform = MelSpectrogram(
        sample_rate=mimo_audio_tokenizer.config.sampling_rate,
        n_fft=mimo_audio_tokenizer.config.nfft,
        hop_length=mimo_audio_tokenizer.config.hop_length,
        win_length=mimo_audio_tokenizer.config.window_size,
        f_min=mimo_audio_tokenizer.config.fmin,
        f_max=mimo_audio_tokenizer.config.fmax,
        n_mels=mimo_audio_tokenizer.config.n_mels,
        power=1.0,
        center=True,
    ).to(device)
    spec = mel_transform(wav[None, :])
    return torch.log(torch.clip(spec, min=1e-7)).squeeze()


def group_by_length(features: torch.Tensor, lengths: torch.Tensor, max_length: int):
    if features.size(0) != lengths.sum().item():
        raise ValueError(f"Feature size mismatch: {features.size(0)} vs {lengths.sum().item()}")

    split_points = []
    current_sum = 0

    for i, seq_len in enumerate(lengths):
        if current_sum + seq_len > max_length and current_sum > 0:
            split_points.append(i)
            current_sum = seq_len.item()
        else:
            current_sum += seq_len.item()

    # Convert split points to group sizes
    group_sizes = []
    prev = 0
    for point in split_points:
        group_sizes.append(point - prev)
        prev = point
    if prev < len(lengths):
        group_sizes.append(len(lengths) - prev)

    len_groups = torch.split(lengths, group_sizes)
    feature_sizes = [group.sum().item() for group in len_groups]
    feature_groups = torch.split(features, feature_sizes)

    return feature_groups, len_groups


def encode_batch(input_features: torch.Tensor, input_lens: torch.Tensor, max_length: int = 256000):
    feature_groups, len_groups = group_by_length(input_features, input_lens, max_length)

    encoded_parts = []
    for features, lengths in zip(feature_groups, len_groups):
        with torch.no_grad():
            codes, _ = mimo_audio_tokenizer.encoder.encode(
                input_features=features.to(device), input_lens=lengths.to(device), return_codes_only=True
            )
            encoded_parts.append(codes)

    return torch.cat(encoded_parts, dim=-1)


def preprocess_input(input: None | str | torch.Tensor = None, device="cpu", audio_channels=4, group_size=8):
    if isinstance(input, torch.Tensor) or (isinstance(input, str) and os.path.isfile(input)):
        return "<|sosp|><|empty|><|eosp|>"

    else:
        text = input
        if (
            text.isupper() or text.islower()
        ):  # If the text only contains upper-case or lower-case letters, capitalize it.
            text = text.capitalize()
        return text


def _build_tts_system_prompt(has_voice_prompt: bool, voice_audio_token=None) -> list[InputSegment]:
    """Build system prompt for TTS task"""
    if has_voice_prompt and voice_audio_token is not None:
        return [
            create_system_start(),
            create_segment(
                # text="You need to generate a speech with the same timbre as the speech prompt, based on the specified style instructions and text content. Your timbre should be: "
                text="你需要根据指定的风格指令和文本内容来生成和语音prompt具有相同音色的语音。你的音色应该是："
            ),
            create_segment(text="", audio=voice_audio_token),
            create_user_end(),
        ]
    else:
        return create_system_turn_text_only("你需要根据指定的风格指令和文本内容来生成语音。")
        # return create_system_turn_text_only(
        #     "You need to generate speech based on the specified style instructions and text content."
        # )


def _build_tts_system_prompt_no_instruct(has_voice_prompt: bool, voice_audio_token=None) -> list[InputSegment]:
    """Build system prompt for TTS task"""
    if has_voice_prompt and voice_audio_token is not None:
        return [
            create_system_start(),
            create_segment(
                # text="You need to generate a speech with the same timbre as the speech prompt, based on the specified style instructions and text content. Your timbre should be:"
                text="你需要根据指定的风格指令和文本内容来生成和语音prompt具有相同音色的语音。你的音色应该是："
            ),
            create_segment(text="", audio=voice_audio_token),
            create_user_end(),
        ]
    else:
        return []


def get_tts_sft_prompt(
    input: None | str = None,
    instruct=None,
    read_text_only=True,
    prompt_speech=None,
):
    """
    Build prompt for TTS (Text-to-Speech) task

    Args:
        input: Input text
        instruct: Style instruction (e.g., "speak happily in a child's voice")
        read_text_only: Whether to read only plain text (False means text contains template)
        prompt_speech: Reference audio (for voice cloning)
    """
    assistant_prompt_audio_token = preprocess_input(prompt_speech) if prompt_speech is not None else None

    if not read_text_only:
        # Not just reading text, text contains template (template:text format)
        text = preprocess_input(input)
        lm_prompt = _build_tts_system_prompt(
            has_voice_prompt=assistant_prompt_audio_token is not None,
            voice_audio_token=assistant_prompt_audio_token,
        )
        lm_prompt.append(create_segment(text=f"<|im_start|>user\n{text}<|im_end|>\n"))
        lm_prompt.append(create_assistant_start_with_think())
    else:
        # Plain text (no instruction inside)
        language = detect_language(input)
        template = random.choice(tts_zh_templates if language == "zh" else tts_en_templates)
        text = preprocess_input(input)

        if instruct is None:
            # No instruct instruction
            lm_prompt = _build_tts_system_prompt_no_instruct(
                has_voice_prompt=assistant_prompt_audio_token is not None,
                voice_audio_token=assistant_prompt_audio_token,
            )
            lm_prompt.extend(
                [
                    create_segment(text=f"<|im_start|>user\n{template}: {text}<|im_end|>\n"),
                    create_assistant_start_with_sostm(),
                ]
            )
        else:
            # Has instruct instruction
            lm_prompt = _build_tts_system_prompt(
                has_voice_prompt=assistant_prompt_audio_token is not None,
                voice_audio_token=assistant_prompt_audio_token,
            )
            lm_prompt.append(create_segment(text=f"<|im_start|>user\n{template}: {text}({instruct})<|im_end|>\n"))
            lm_prompt.append(create_assistant_start_with_think())

    return lm_prompt


def get_audio_understanding_sft_prompt(
    input_speech,
    input_text,
    thinking=False,
    use_sostm=False,
):
    """Build prompt for audio understanding task"""
    audio_tokenized = preprocess_input(input_speech)

    lm_prompt = create_user_turn_with_audio(audio_tokenized, extra_text=input_text)
    lm_prompt = append_assistant_ending(lm_prompt, thinking=thinking, use_sostm=use_sostm)
    return lm_prompt


def _build_voice_prompt_system(prompt_speech) -> list[InputSegment]:
    """Build system prompt with voice prompt"""
    return create_system_turn_with_voice_prompt(
        prompt_text="Your voice should be：", audio_token=preprocess_input(prompt_speech)
    )


def get_spoken_dialogue_sft_prompt(
    input_speech,
    system_prompt=None,
    prompt_speech=None,
    add_history=False,
):
    """
    Build prompt for spoken dialogue task

    Args:
        input_speech: Input speech
        system_prompt: System prompt text
        prompt_speech: Reference audio (for voice cloning)
        add_history: Whether to add history (Note: history variable is undefined in original code)
    """
    audio_tokenized = preprocess_input(input_speech)
    lm_prompt = []

    # Note: history variable is undefined in original code, this branch may never execute
    # To use history feature, history should be passed as a parameter
    if add_history:
        # Simplified form of adding history
        lm_prompt = create_user_turn_with_audio(audio_tokenized)
        lm_prompt.append(create_assistant_start_with_sostm())
    else:
        # Add voice prompt (if available)
        if prompt_speech:
            lm_prompt.extend(_build_voice_prompt_system(prompt_speech))

        # Add user turn
        lm_prompt.append(create_user_start())
        if system_prompt:
            lm_prompt.append(create_segment(text=system_prompt))
        lm_prompt.append(create_segment(audio=audio_tokenized))
        lm_prompt.append(create_user_end())
        lm_prompt.append(create_assistant_start_with_sostm())

    return lm_prompt


def get_spoken_dialogue_sft_multiturn_prompt(
    message_list,
    system_prompt=None,
    prompt_speech=None,
    tokenizer=None,
    group_size=8,
    audio_channels=4,
):
    """
    Build prompt for multi-turn spoken dialogue task

    Args:
        message_list: List of messages containing role and content
        system_prompt: System prompt text
        prompt_speech: Reference audio (for voice cloning)
        tokenizer: Tokenizer
        group_size: Group size
        audio_channels: Number of audio channels
    """
    lm_prompt = []

    # Add voice prompt (if available)
    if prompt_speech:
        lm_prompt.extend(
            create_system_turn_with_voice_prompt(
                prompt_text="Your voice should be:", audio_token=preprocess_input(prompt_speech)
            )
        )

    # Define message processors
    def user_processor(msg):
        segments = [create_user_start()]
        if system_prompt:
            segments.append(create_segment(text=system_prompt))
        segments.append(create_segment(audio=preprocess_input(msg["content"])))
        segments.append(create_user_end())
        return segments

    def assistant_processor(msg):
        return [
            create_assistant_start(),
            create_streaming_segment(
                text=msg["content"]["text"],
                audio=preprocess_input(msg["content"]["audio"]),
                tokenizer=tokenizer,
                group_size=group_size,
                audio_channels=audio_channels,
            ),
            create_user_end(),
        ]

    # Process message list
    lm_prompt.extend(process_multiturn_messages(message_list, user_processor, assistant_processor))
    lm_prompt.append(create_assistant_start_with_sostm())

    return lm_prompt


def get_s2t_dialogue_sft_prompt(
    input_speech,
    thinking=False,
):
    """Build prompt for speech-to-text dialogue task"""
    audio_tokenized = preprocess_input(input_speech)

    lm_prompt = create_user_turn_with_audio(audio_tokenized)
    lm_prompt = append_assistant_ending(lm_prompt, thinking=thinking)
    return lm_prompt


def get_s2t_dialogue_sft_multiturn_prompt(message_list, thinking=False):
    """Build prompt for multi-turn speech-to-text dialogue task"""
    lm_prompt = process_multiturn_messages(
        message_list, user_processor=create_audio_user_message, assistant_processor=create_text_assistant_message
    )
    lm_prompt = append_assistant_ending(lm_prompt, thinking=thinking)
    return lm_prompt


def get_text_dialogue_sft_prompt(
    input_text,
    thinking=False,
):
    """Build prompt for text-only dialogue task"""
    lm_prompt = create_user_turn_with_text(input_text)
    lm_prompt = append_assistant_ending(lm_prompt, thinking=thinking)
    return lm_prompt


def get_text_dialogue_sft_multiturn_prompt(
    message_list,
    thinking=False,
):
    """Build prompt for multi-turn text-only dialogue task"""
    lm_prompt = process_multiturn_messages(
        message_list, user_processor=create_text_user_message, assistant_processor=create_text_assistant_message
    )
    lm_prompt = append_assistant_ending(lm_prompt, thinking=thinking)
    return lm_prompt


def get_in_context_learning_s2s_prompt(
    instruction,
    prompt_examples,
    audio,
    tokenizer=None,
    group_size=8,
    audio_channels=4,
):
    """
    Build prompt for In-Context Learning speech-to-speech task

    Args:
        instruction: Instruction text
        prompt_examples: List of examples, each containing input_audio, output_transcription, output_audio
        audio: Input audio to be processed
        tokenizer: Tokenizer
        group_size: Group size
        audio_channels: Number of audio channels
    """
    prompt = [create_segment(text=f"[Int]:{instruction}\n")]

    # Add examples
    for example in prompt_examples:
        prompt.extend(
            [
                create_segment(audio=preprocess_input(example["input_audio"])),
                create_segment(text="\n"),
                create_streaming_segment(
                    text=example["output_transcription"],
                    audio=preprocess_input(example["output_audio"]),
                    tokenizer=tokenizer,
                    group_size=group_size,
                    audio_channels=audio_channels,
                ),
                create_segment(text=" \n\n"),
            ]
        )

    # Add input audio to be processed
    prompt.extend(
        [
            create_segment(audio=preprocess_input(audio)),
            create_segment(text="\n"),
            create_sostm_segment(),
        ]
    )

    return prompt


def get_audio_data(audio_url):
    if audio_url.startswith("data:"):
        header, b64_data = audio_url.split(",", 1)
        audio_bytes = base64.b64decode(b64_data.strip())
        audio_file = io.BytesIO(audio_bytes)
    else:
        # File path
        audio_file = audio_url

    audio_signal, sr = librosa.load(audio_file, sr=24000)
    audio_data = (audio_signal.astype(np.float32), sr)
    return audio_data


def to_prompt(input_segs):
    out_put = []

    for input_seg in input_segs:
        if isinstance(input_seg, StreamingInputSegment) and input_seg.text:
            out_put.append("<|sostm|>")
            if input_seg.audio is not None and isinstance(input_seg.audio, str):
                out_put.append(input_seg.text)
                out_put.append("<|eot|>")
                out_put.append("<|empty|>")
            else:
                out_put.append(input_seg.text)
                out_put.append("<|eot|>")
            out_put.append("<|eostm|>")

        else:
            out_put.append(input_seg.text)
            if input_seg.audio is not None:
                out_put.append(input_seg.audio)

    prompt = "".join(out_put)
    return prompt
