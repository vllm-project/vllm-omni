"""Node B (audio decoder) half of the 2-node LongCat-Next thinker+audio e2e
test. Reads the thinker's real per-frame audio codes -- produced by
talker_mtp (the audio_head depth-transformer loop, modeling_longcat_next.py)
and written by node A's longcat_next_multinode_thinker.py to a shared
scratch file -- and decodes them with LongcatNextAudioDecoder directly -- no
vLLM engine, no server, same pattern as the proven
pbs/scripts/longcat_next_audio_decoder_standalone.py, just with real thinker
codes instead of synthetic ones.

Run with: python longcat_next_multinode_audio_decoder.py <model_path> <in_json> <out_wav>
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vllm.config import VllmConfig

from vllm_omni.config.model import OmniModelConfig
from vllm_omni.model_executor.models.longcat_next.modeling_longcat_next_audio_decoder import (
    LongcatNextAudioDecoder,
)


def main() -> None:
    model_path = sys.argv[1]
    in_json = sys.argv[2]
    out_wav = sys.argv[3]

    with open(in_json) as f:
        payload = json.load(f)
    codes = payload["audio_codes"]
    print(f"[audio_decoder] read {len(codes)} audio code frames from {in_json}")
    if not codes:
        verdict = (
            "FAIL: no audio codes in handoff file -- check that node A's talker_mtp "
            "actually ran (see '[thinker] talker_mtp produced N audio code frames' in its log)"
        )
        print(f"[audio_decoder] VERDICT {verdict}")
        sys.exit(1)
    audio_token_ids = [c for row in codes for c in row]

    free_before, total = torch.cuda.mem_get_info()
    print(f"[audio_decoder] before load: {(total - free_before) / 1e9:.2f} GB used / {total / 1e9:.2f} GB total")

    model_config = OmniModelConfig(
        model=model_path,
        model_arch="LongcatNextAudioDecoder",
        trust_remote_code=True,
        dtype="bfloat16",
        seed=0,
    )
    vllm_config = VllmConfig(model_config=model_config)
    decoder = LongcatNextAudioDecoder(vllm_config=vllm_config, prefix="")
    decoder._ensure_weights()

    free_after_load, _ = torch.cuda.mem_get_info()
    print(f"[audio_decoder] after weight load: {(total - free_after_load) / 1e9:.2f} GB used")

    out = decoder.forward(
        input_ids=None,
        positions=None,
        additional_information={"audio_token_ids": audio_token_ids},
    )

    waveform = out.multimodal_outputs.get("model_outputs")
    sr_tensor = out.multimodal_outputs.get("sr")
    if waveform is None:
        print("[audio_decoder] VERDICT FAIL: decoder ran but produced no waveform")
        sys.exit(1)

    # sr_tensor is a torch.tensor([24000], dtype=torch.int32) (see
    # modeling_longcat_next_audio_decoder.py's forward()), not a plain int --
    # must be unwrapped before use as a sample rate.
    sample_rate = int(sr_tensor.item()) if hasattr(sr_tensor, "item") else 24000
    wf = waveform.reshape(-1).detach().to(torch.float32).cpu()
    duration_s = wf.shape[0] / sample_rate if sample_rate else 0.0
    print(
        f"[audio_decoder] OK: waveform shape={tuple(waveform.shape)} dtype={waveform.dtype} "
        f"sample_rate={sample_rate} duration={duration_s:.3f}s"
    )

    # soundfile, not torchaudio.save: torchaudio's default backend on this
    # venv requires torchcodec, which is not installed (confirmed: import
    # raises ImportError). soundfile is installed and already proven to work
    # (pbs/scripts/longcat_next_audio_decoder_standalone.py, job 15026737).
    import soundfile as sf

    sf.write(out_wav, wf.numpy(), sample_rate)
    print(f"[audio_decoder] wrote {out_wav}")
    print(
        f"[audio_decoder] VERDICT PASS: decoded {len(codes)} frames into a "
        f"{duration_s:.3f}s waveform at {sample_rate}Hz"
    )


if __name__ == "__main__":
    main()
