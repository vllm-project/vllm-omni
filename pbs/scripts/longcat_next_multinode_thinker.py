"""Node A (thinker) half of the 2-node LongCat-Next thinker+audio e2e test.

Runs the proven thinker-only pipeline (longcat_next_4gpu.yaml, TP=4, one
node's 4 GPUs) offline via vllm_omni.entrypoints.omni.Omni — not the HTTP
server. With talker_mtp wired up (modeling_longcat_next.py), the real
per-frame audio codes accumulate in
RequestOutput.outputs[0].multimodal_output["codes"]["audio"] (a [T, 8]
offset-carrying tensor) rather than needing to be parsed back out of the
visible token stream — see stage_input_processors/longcat_next.py's
thinker2audio_decoder_token_only(), which reads the same field in the
normal (single-node, in-process) 2-stage pipeline. Splitting across nodes
means writing that same field to a shared scratch file instead of handing
it to the orchestrator directly.

Writes <out_json> = {"audio_codes": [[c0..c7], ...]} for node B to pick up.

Run with: python longcat_next_multinode_thinker.py <model_path> <deploy_yaml> <out_json>
"""

import json
import os
import sys
from collections.abc import Mapping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vllm import SamplingParams
from vllm.multimodal.media.audio import load_audio

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniTextPrompt


def main() -> None:
    model_path = sys.argv[1]
    deploy_yaml = sys.argv[2]
    out_json = sys.argv[3]

    llm = Omni(
        model=model_path,
        deploy_config=deploy_yaml,
        trust_remote_code=True,
    )

    # Speech-synthesis prompt copied VERBATIM from the checkpoint's own
    # README "Audio - Speech Synthesis Example" (the previous English "Please
    # say: ..." phrasing was NOT from the README and never produced audio in
    # any run -- the model just emitted a short generic text reply + EOS). The
    # README's working example uses a Chinese synthesis *instruction* and
    # Chinese *content*, matching the Chinese reference voice vc_zh3.wav:
    #   system: "Replicate the voice in the audio clip to formulate an answer:
    #            <longcat_audio_start>./assets/vc_zh3.wav<longcat_audio_end>"
    #   user:   "用这个声音合成以下内容：明天的meeting在三楼的Conference Room举行。
    #            <longcat_audiogen_start>"
    # Built as raw chat-template text + multi_modal_data (offline OmniTextPrompt
    # skips chat-template rendering). Two things reproduced from the template:
    #   1. get_placeholder_str expands <longcat_audio_start> into the full
    #      <longcat_audio_start><longcat_audio_pad><longcat_audio_end> triple
    #      that LongcatNextMultiModalProcessor's PromptReplacement needs to
    #      attach the audio (the README's raw path string is replaced by the
    #      real waveform via multi_modal_data here).
    #   2. add_generation_prompt moves a trailing <longcat_audiogen_start> to
    #      *after* <longcat_assistant> (template's ns.suffix_to_move logic).
    ref_voice_path = os.path.join(model_path, "assets", "vc_zh3.wav")
    audio_signal, sr = load_audio(ref_voice_path, sr=16000)
    audio_placeholder = "<longcat_audio_start><longcat_audio_pad><longcat_audio_end>"

    # Byte-for-byte match to LongCat-Next-inference example/test_cases.yaml
    # spk_syn (period after "answer.", spaces between segments from the folded
    # YAML block) so the prompt is not a variable while debugging.
    prompt_text = (
        "<longcat_system>Replicate the voice in the audio clip to formulate an answer. "
        f"{audio_placeholder} "
        "<longcat_user>用这个声音合成以下内容：明天的meeting在三楼的Conference Room举行。 "
        "<longcat_assistant><longcat_audiogen_start>"
    )

    prompt = OmniTextPrompt(
        prompt=prompt_text,
        multi_modal_data={"audio": (audio_signal, sr)},
    )

    # README-recommended text/thinker sampling params for Audio-to-Audio /
    # Speech Synthesis. The audio_generation_config (temperature 0.5 /
    # top_k 5 / top_p 0.85) is applied inside talker_mtp's _sample_audio_code.
    #
    # NOTE (run 7): ignore_eos=True was tried and DISPROVED as a fix. The model
    # emitted its natural EOS at token 7, and forcing generation to continue
    # just produced 2041 more tokens of unrelated text that degenerated into
    # repetition -- still 0 audio frames. So EOS-termination was never the
    # blocker: the model simply does not enter speech-synthesis mode from this
    # prompt. Reverted to letting EOS stop generation (shorter, cleaner runs)
    # while the real conditioning problem is investigated.
    sampling_params = SamplingParams(
        max_tokens=2048,
        temperature=0.2,
        top_k=20,
        top_p=0.85,
        repetition_penalty=1.1,
        detokenize=True,
    )

    # Prompt-side sanity check BEFORE generation: audio-gen mode is entered
    # only if the last prefill token is <longcat_audiogen_start> (131123),
    # which is what preprocess()/_advance_audio_gen keys off. If this line
    # does not say True, nothing downstream can produce audio and the rest of
    # the run's audio diagnostics are moot.
    try:
        from transformers import AutoTokenizer

        _tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        _pids = _tok(prompt_text, add_special_tokens=False).input_ids
        print(f"[thinker] prompt tokens={len(_pids)} last8={_pids[-8:]}")
        print(f"[thinker] prompt ends with audiogen_start(131123): {_pids[-1] == 131123}")
    except Exception as e:  # diagnostics only
        print(f"[thinker] (prompt tokenization check skipped: {e})")

    outputs = llm.generate([prompt], sampling_params)
    out = outputs[0]
    completion = out.outputs[0]
    token_ids = list(completion.token_ids)
    print(f"[thinker] generated {len(token_ids)} visible tokens")
    # Full visible token id stream + decoded text, so we can see exactly what
    # the model produced and whether it emitted the audio markers
    # (<longcat_audiogen_start>=131123 -> transcript -> <longcat_audiotext_start>
    # =131120 -> codec). Without this we're debugging generation behavior blind.
    print(f"[thinker] visible token_ids: {token_ids}")
    marker_names = {
        131103: "audio_start", 131104: "audio_end", 131105: "audio_pad",
        131120: "audiotext_start", 131121: "audiotext_end", 131122: "audiotext_pad",
        131123: "audiogen_start", 131124: "audiogen_end", 2: "EOS",
    }
    seen = [(i, marker_names[t]) for i, t in enumerate(token_ids) if t in marker_names]
    print(f"[thinker] audio markers in visible stream: {seen or 'NONE'}")
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"[thinker] decoded visible text: {tok.decode(token_ids)!r}")
    except Exception as e:  # decoding is best-effort diagnostics only
        print(f"[thinker] (could not decode tokens: {e})")

    mm_output = getattr(completion, "multimodal_output", None)
    print(f"[thinker] multimodal_output type={type(mm_output).__name__} val={mm_output!r:.200}")
    if isinstance(mm_output, Mapping):
        print(f"[thinker] multimodal_output keys: {list(mm_output.keys())}")
        codes_val = mm_output.get("codes")
        print(f"[thinker] mm_output['codes'] type={type(codes_val).__name__} val={codes_val!r:.200}")
        if isinstance(codes_val, dict):
            audio_val = codes_val.get("audio")
            print(f"[thinker] mm_output['codes']['audio'] type={type(audio_val).__name__}", end="")
            if hasattr(audio_val, "shape"):
                print(f" shape={list(audio_val.shape)}")
            elif isinstance(audio_val, list):
                print(f" len={len(audio_val)}")
            else:
                print()
    mm_output = mm_output or {}
    audio_codes = mm_output.get("codes", {}).get("audio") if isinstance(mm_output, Mapping) else None
    if hasattr(audio_codes, "tolist"):
        audio_codes = audio_codes.tolist()
    elif isinstance(audio_codes, list):
        audio_codes = [r.tolist() if hasattr(r, "tolist") else r for r in audio_codes]
    else:
        audio_codes = audio_codes or []
    print(f"[thinker] talker_mtp produced {len(audio_codes)} audio code frames")
    if audio_codes:
        print(f"[thinker] first audio frame: {audio_codes[0]}")
        print(f"[thinker] last audio frame:  {audio_codes[-1]}")
        widths = {len(r) for r in audio_codes}
        print(f"[thinker] frame widths present: {sorted(widths)} (expect [8])")
        neg = sum(1 for r in audio_codes if any(c < 0 for c in r))
        print(f"[thinker] frames containing a negative code: {neg} (expect 0)")

    # One unambiguous verdict line, so the outcome is greppable without
    # reading the worker logs. Distinguishes the two failure modes that have
    # bitten this pipeline: never entering audio-gen at all, vs. generating
    # frames that get dropped before reaching multimodal_output.
    if audio_codes:
        verdict = f"PASS: {len(audio_codes)} audio frames reached multimodal_output"
    elif mm_output:
        verdict = "FAIL: multimodal_output present but carries no audio codes"
    else:
        verdict = (
            "FAIL: no multimodal_output at all -- either audio-gen never "
            "engaged (check the audiogen_start line above and the "
            "[longcat-audio] worker logs) or make_omni_output did not emit"
        )
    print(f"[thinker] VERDICT {verdict}")

    with open(out_json, "w") as f:
        json.dump({"audio_codes": audio_codes}, f)
    print(f"[thinker] wrote audio_codes to {out_json}")


if __name__ == "__main__":
    main()
