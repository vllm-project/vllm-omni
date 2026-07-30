"""Node A (thinker) half of the 2-node LongCat-Next thinker+image e2e test.

Same pattern as longcat_next_multinode_thinker.py (audio's proven node A),
but for image generation: runs the thinker-only pipeline (longcat_next_4gpu.yaml,
TP=4) offline via Omni(), and with talker_mtp's visual branch wired up
(modeling_longcat_next.py), the real per-position visual codes accumulate in
RequestOutput.outputs[0].multimodal_output["codes"]["visual"] (a [T, 8]
tensor of raw codebook indices, no offsets -- see
_code_embeddings/make_omni_output). Already proven to reach multimodal_output
on GPU (job 15030222, thinker-only smoke test); this script is the same
mechanism, just also writing the handoff file + token_h/token_w for node B's
image decoder to consume (see LongcatNextImageDecoder.forward(), which
needs token_h/token_w to reshape the flat code stream back into a grid).

Writes <out_json> = {"visual_codes": [[c0..c7], ...], "token_h": H, "token_w": W}
for node B to pick up.

Run with: python longcat_next_multinode_image_thinker.py <model_path> <deploy_yaml> <out_json>
"""

import json
import os
import sys
from collections.abc import Mapping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vllm import SamplingParams

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniTextPrompt

# Same small bounded grid already proven on GPU (job 15030222) -- this run's
# job is proving the 2-node handoff + decoder wiring, not a bigger image.
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
        print(f"[thinker] prompt tokens={len(pids)} last8={pids[-8:]}")
        print(f"[thinker] prompt ends with img_start(131106): {pids[-1] == 131106}")
    except Exception as e:  # diagnostics only
        print(f"[thinker] (prompt tokenization check skipped: {e})")

    outputs = llm.generate([prompt], sampling_params)
    out = outputs[0]
    completion = out.outputs[0]
    token_ids = list(completion.token_ids)
    print(f"[thinker] generated {len(token_ids)} visible tokens")

    mm_output = getattr(completion, "multimodal_output", None)
    print(f"[thinker] multimodal_output type={type(mm_output).__name__} val={mm_output!r:.200}")

    visual_codes = None
    if isinstance(mm_output, Mapping):
        codes_val = mm_output.get("codes")
        if isinstance(codes_val, Mapping):
            visual_codes = codes_val.get("visual")
    if hasattr(visual_codes, "tolist"):
        visual_codes = visual_codes.tolist()
    elif not isinstance(visual_codes, list):
        visual_codes = []
    print(f"[thinker] talker_mtp produced {len(visual_codes)} visual code frames")
    if visual_codes:
        widths = {len(r) for r in visual_codes}
        print(f"[thinker] frame widths present: {sorted(widths)} (expect [8])")
        neg = sum(1 for r in visual_codes if any(c < 0 for c in r))
        print(f"[thinker] frames containing a negative code: {neg} (expect 0)")

    if visual_codes:
        verdict = f"PASS: {len(visual_codes)} visual frames reached multimodal_output"
    elif mm_output:
        verdict = "FAIL: multimodal_output present but carries no visual codes"
    else:
        verdict = (
            "FAIL: no multimodal_output at all -- either image-gen never "
            "engaged or make_omni_output did not emit"
        )
    print(f"[thinker] VERDICT {verdict}")

    with open(out_json, "w") as f:
        json.dump(
            {"visual_codes": visual_codes, "token_h": TOKEN_H, "token_w": TOKEN_W},
            f,
        )
    print(f"[thinker] wrote visual_codes to {out_json}")


if __name__ == "__main__":
    main()
