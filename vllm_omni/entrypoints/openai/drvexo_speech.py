# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Request-local entry into the resident native Talker/Codec; no second LLM."""

import uuid
from pathlib import Path

import anyio


def load_speech_tokenizer(model_directory):
    """Load the local Talker text tokenizer once, before mounting its routes.

    The complete DRVEXO pipeline selects the bundle's ``talker`` subdirectory.
    Match the native prompt builder's tokenizer settings, without loading weights.
    """
    import transformers
    from transformers import AutoTokenizer

    kwargs = dict(trust_remote_code=True, use_fast=True, local_files_only=True)
    if transformers.__version__ < "5":
        kwargs["fix_mistral_regex"] = True
    tokenizer = AutoTokenizer.from_pretrained(str(Path(model_directory) / "talker"), **kwargs)
    tokenizer.padding_side = "left"
    return tokenizer


def speech_prompt(text, language="Auto"):
    if not isinstance(text, str) or not text.strip() or len(text) > 512:
        raise ValueError("Speech requires 1–512 nonempty characters")
    if language not in {"Auto", "Chinese", "English"}:
        raise ValueError("Speech language must be Auto, Chinese or English")
    # Native Base, x-vector only, streaming prefill:3 role tokens,3/4 codec
    # language tags,1 speaker,2 codec tags minus1,1 first text token.
    # Auto has no language-ID token. No dialect speaker is selected.
    size = 9 if language == "Auto" else 10
    return {
        "prompt_token_ids": [0] * size,
        "model_intermediate_buffer": {
            "text": [text],
            "task_type": ["Base"],
            "language": [language],
            "x_vector_only_mode": [True],
            "non_streaming_mode": [False],
            # Existing lexical router's caller-text entry, not generated-token provenance.
            "drvexo_fixed_reply": True,
        },
    }


async def collect_sentence(engine, text, language, final_output_kind):
    if engine.num_stages != 3:
        raise ValueError("Sentence synthesis requires the complete three-stage model")
    prompt = speech_prompt(text, language)
    params = [p.clone() for p in engine.default_sampling_params_list]
    for param in params:
        param.output_kind = final_output_kind
    identifier = "drvexo-sentence-" + uuid.uuid4().hex
    generator, final, completed = None, None, False
    try:
        generator = engine.generate(
            prompt=prompt,
            request_id=identifier,
            sampling_params_list=params,
            output_modalities=["audio"],
            entry_stage_id=1,
        )
        async for output in generator:
            if output.stage_id != 2 or output.error:
                raise RuntimeError("Sentence speech stage failed")
            final = output
        if final is None:
            raise RuntimeError("No waveform returned")
        completed = True
        return final
    finally:
        with anyio.CancelScope(shield=True):
            try:
                if generator is not None:
                    await generator.aclose()
            finally:
                if not completed:
                    await engine.abort(identifier)


async def stream_sentence(engine, text, language="Auto"):
    """Yield native Code2Wav chunks, never slice an already finished waveform."""
    import io

    import numpy as np
    import soundfile as sf
    from vllm.sampling_params import RequestOutputKind

    from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech

    if engine.num_stages != 3:
        raise ValueError("Streaming speech requires all three model stages")
    params = [p.clone() for p in engine.default_sampling_params_list]
    for param in params:
        param.output_kind = RequestOutputKind.DELTA
    identifier = "drvexo-stream-sentence-" + uuid.uuid4().hex
    generator = engine.generate(
        prompt=speech_prompt(text, language),
        request_id=identifier,
        sampling_params_list=params,
        output_modalities=["audio"],
        entry_stage_id=1,
    )
    completed, previous, sample_rate, emitted = False, 0, None, 0
    try:
        async for output in generator:
            if output.error:
                raise RuntimeError("Native streaming speech failed")
            if output.stage_id != 2:
                continue
            audio, key = OmniOpenAIServingSpeech._extract_audio_output(output)
            if audio is None or key is None:
                continue
            rate = audio.get("sr", 24000)
            if isinstance(rate, list):
                rate = rate[-1]
            rate = int(rate.item() if hasattr(rate, "item") else rate)
            if sample_rate is not None and sample_rate != rate:
                raise ValueError("Audio sample rate changed")
            sample_rate = rate
            values = audio[key]
            if isinstance(values, list):
                chunks = values[previous:]
                if len(values) < previous:
                    raise ValueError("Cumulative audio shrank")
                previous = len(values)
            else:
                chunks = [values]
            for chunk in chunks:
                if chunk is None:
                    continue
                wave = chunk.float().detach().cpu().numpy() if hasattr(chunk, "detach") else np.asarray(chunk)
                wave = np.asarray(wave, dtype=np.float32).reshape(-1)
                if not wave.size:
                    continue
                if not np.isfinite(wave).all():
                    raise ValueError("Nonfinite audio")
                data = io.BytesIO()
                sf.write(data, wave, sample_rate, format="WAV", subtype="PCM_16")
                yield {
                    "wav": data.getvalue(),
                    "sample_rate": sample_rate,
                    "samples": len(wave),
                    "chunk_index": emitted,
                    "request_id": identifier,
                }
                emitted += 1
        if not emitted:
            raise RuntimeError("Streaming speech produced no audio")
        completed = True
    finally:
        with anyio.CancelScope(shield=True):
            try:
                await generator.aclose()
            finally:
                if not completed:
                    await engine.abort(identifier)
