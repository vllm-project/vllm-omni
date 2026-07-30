"""First-ever GPU exercise of LongCat-Next image *generation* (not image
understanding, which is already GPU-verified). Thinker-only, single node,
proven fp8 config (longcat_next_4gpu.yaml) -- mirrors the very first audio-gen
validation run: prove the mechanism (visual_head sampling, state machine,
codes reaching multimodal_output) before ever touching the image decoder
stage, same discipline used for audio.

Prompt format and PROMPT UNCERTAINTY (read this before trusting a FAIL):
the reference's own test_cases.yaml (img_gen case) uses
    <longcat_img_token_size>{w} {h}</longcat_img_token_size><longcat_img_start>
appended after a description, with token_w passed separately to the harness
(the reference's per-request `input_extra_infos[0]["token_w"]`, since
generation_config.json's image_generation_config is None -- the checkpoint
declares no default grid size at all). The user/assistant wrapper here is
built the SAME way the audio driver script's already-proven prompt was
(<longcat_user>...<longcat_assistant><trigger_token>) since offline
OmniTextPrompt skips chat-template rendering -- but this exact placement is
inferred by analogy to the working audio prompt, NOT confirmed against a
known-good image-gen example the way the audio prompt eventually was. If
this run shows the model never entering image-gen mode (no [longcat-image]
advance lines, VERDICT FAIL: never entered image-gen), the prompt format is
the first thing to revisit -- exactly the class of failure the audio arc
hit on its very first prompt attempts.

Grid is small (4x4 -> 20 total steps incl. newlines) deliberately: this is
about proving the mechanism works at all, not generating a full image.

Run with: python longcat_next_image_gen_thinker.py <model_path> <deploy_yaml> <out_json>
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vllm import SamplingParams

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniTextPrompt

TOKEN_W = 4
TOKEN_H = 4


def main() -> None:
    model_path = sys.argv[1]
    deploy_yaml = sys.argv[2]
    out_json = sys.argv[3]

    llm = Omni(
        model=model_path,
        deploy_config=deploy_yaml,
        trust_remote_code=True,
    )

    prompt_text = (
        "<longcat_user>A single red apple on a white background."
        f"<longcat_img_token_size>{TOKEN_W} {TOKEN_H}</longcat_img_token_size>"
        "<longcat_assistant><longcat_img_start>"
    )
    prompt = OmniTextPrompt(
        prompt=prompt_text,
        additional_information={"token_w": TOKEN_W, "token_h": TOKEN_H},
    )

    # Small bounded budget: TOKEN_W*(TOKEN_H+1) = 20 real generation steps for
    # a 4x4 grid (4 pixels + 1 newline per row), no max_gen-style safety cap
    # exists for images (the reference's GenImageStageStage has no analogous
    # force-end transition -- termination is either the end sentinel or this
    # outer token budget), so keep it small and let this budget be the
    # bound for a first test.
    sampling_params = SamplingParams(
        max_tokens=64,
        temperature=0.5,
        top_k=1024,
        top_p=0.75,
        repetition_penalty=1.0,
        detokenize=True,
    )

    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        pids = tok(prompt_text, add_special_tokens=False).input_ids
        print(f"[image-gen] prompt tokens={len(pids)} last8={pids[-8:]}")
        print(f"[image-gen] prompt ends with img_start(131106): {pids[-1] == 131106}")
    except Exception as e:  # diagnostics only
        print(f"[image-gen] (prompt tokenization check skipped: {e})")

    outputs = llm.generate([prompt], sampling_params)
    out = outputs[0]
    completion = out.outputs[0]
    token_ids = list(completion.token_ids)
    print(f"[image-gen] generated {len(token_ids)} visible tokens")
    print(f"[image-gen] visible token_ids: {token_ids}")
    marker_names = {
        131090: "img_token_size_start", 131091: "img_token_size_end",
        131106: "img_start", 131107: "img_end", 131108: "img_pad",
        131109: "img_newline", 2: "EOS",
    }
    seen = [(i, marker_names[t]) for i, t in enumerate(token_ids) if t in marker_names]
    print(f"[image-gen] markers in visible stream: {seen or 'NONE'}")
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"[image-gen] decoded visible text: {tok.decode(token_ids)!r}")
    except Exception as e:  # decoding is best-effort diagnostics only
        print(f"[image-gen] (could not decode tokens: {e})")

    mm_output = getattr(completion, "multimodal_output", None)
    print(f"[image-gen] multimodal_output type={type(mm_output).__name__} val={mm_output!r:.200}")
    from collections.abc import Mapping

    visual_codes = None
    if isinstance(mm_output, Mapping):
        print(f"[image-gen] multimodal_output keys: {list(mm_output.keys())}")
        codes_val = mm_output.get("codes")
        print(f"[image-gen] mm_output['codes'] type={type(codes_val).__name__} val={codes_val!r:.200}")
        if isinstance(codes_val, Mapping):
            visual_codes = codes_val.get("visual")
    if hasattr(visual_codes, "tolist"):
        visual_codes = visual_codes.tolist()
    elif not isinstance(visual_codes, list):
        visual_codes = []
    print(f"[image-gen] talker_mtp produced {len(visual_codes)} visual code frames")
    if visual_codes:
        print(f"[image-gen] first frame: {visual_codes[0]}")
        print(f"[image-gen] last frame:  {visual_codes[-1]}")
        widths = {len(r) for r in visual_codes}
        print(f"[image-gen] frame widths present: {sorted(widths)} (expect [8])")
        neg = sum(1 for r in visual_codes if any(c < 0 for c in r))
        print(f"[image-gen] frames containing a negative code: {neg} (expect 0)")

    if visual_codes:
        verdict = f"PASS: {len(visual_codes)} visual frames reached multimodal_output"
    elif mm_output:
        verdict = "FAIL: multimodal_output present but carries no visual codes"
    else:
        verdict = (
            "FAIL: no multimodal_output at all -- either image-gen never engaged "
            "(check the img_start line above and [longcat-image] worker logs) "
            "or make_omni_output did not emit"
        )
    print(f"[image-gen] VERDICT {verdict}")

    with open(out_json, "w") as f:
        json.dump({"visual_codes": visual_codes}, f)
    print(f"[image-gen] wrote visual_codes to {out_json}")


if __name__ == "__main__":
    main()
