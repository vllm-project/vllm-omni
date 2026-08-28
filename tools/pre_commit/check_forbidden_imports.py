#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ban dangerous / policy-violating imports unless a file is allowlisted.

Ported from vllm/tools/pre_commit/check_forbidden_imports.py. Existing
vllm_omni/ usages are grandfathered; add a path to the matching
allowed_files set only after a security/policy review.
"""

import sys
from dataclasses import dataclass, field

import regex as re

# Hub entry points that must go through the vLLM-tagged repo_utils helpers.
_HF_NAMES = (
    r"HfApi|HfFileSystem|hf_hub_download|snapshot_download"
    r"|list_repo_files|file_exists|try_to_load_from_cache"
    r"|list_repo_refs|repo_exists"
)

# Non-library trees: examples, tests, and tooling may keep stdlib imports.
_NON_LIBRARY_DIRS = {
    "examples/",
    "tests/",
    "benchmarks/",
    "apps/",
    "docs/",
    "tools/",
    "scripts/",
    ".buildkite/",
    ".github/",
}


@dataclass
class ForbiddenImport:
    pattern: str
    tip: str
    allowed_pattern: re.Pattern = re.compile(r"^$")  # matches nothing by default
    allowed_files: set[str] = field(default_factory=set)
    allowed_dirs: set[str] = field(default_factory=set)


CHECK_IMPORTS = {
    "pickle/cloudpickle": ForbiddenImport(
        pattern=(
            r"^\s*(import\s+(?:[\w.]+\s*(?:as\s+\w+)?\s*,\s*)*(pickle|cloudpickle)\b"
            r"|from\s+(pickle|cloudpickle)\s+import\b)"
        ),
        tip=("Avoid using pickle or cloudpickle or add this file to tools/pre_commit/check_forbidden_imports.py."),
        allowed_files={
            # STOP AND READ BEFORE YOU ADD ANYTHING ELSE TO THIS LIST:
            # pickle/cloudpickle are unsafe when deserializing untrusted data.
            "tests/diffusion/attention/test_attention_sp.py",
            "tests/helpers/process.py",
            "vllm_omni/diffusion/distributed/group_coordinator.py",
        },
    ),
    "base64": ForbiddenImport(
        pattern=r"^\s*(?:import\s+base64(?:$|\s|,)|from\s+base64\s+import)",
        tip=("Replace 'import base64' with 'import pybase64' or 'import pybase64 as base64'."),
        allowed_pattern=re.compile(r"^\s*import\s+pybase64(\s*|\s+as\s+base64\s*)$"),
        allowed_dirs=_NON_LIBRARY_DIRS,
        allowed_files={
            "vllm_omni/benchmarks/data_modules/daily_omni_dataset.py",
            "vllm_omni/benchmarks/data_modules/random_multi_modal_dataset.py",
            "vllm_omni/benchmarks/data_modules/seed_tts_dataset.py",
            "vllm_omni/entrypoints/openai/api_server.py",
            "vllm_omni/entrypoints/openai/audio_utils_mixin.py",
            "vllm_omni/entrypoints/openai/image_api_utils.py",
            "vllm_omni/entrypoints/openai/protocol/images.py",
            "vllm_omni/entrypoints/openai/realtime_connection.py",
            "vllm_omni/entrypoints/openai/serving_chat.py",
            "vllm_omni/entrypoints/openai/serving_speech.py",
            "vllm_omni/entrypoints/openai/serving_speech_stream.py",
            "vllm_omni/entrypoints/openai/video_api_utils.py",
            "vllm_omni/entrypoints/openai/video_stream_base.py",
            "vllm_omni/entrypoints/openai/video_stream_session.py",
            "vllm_omni/experimental/fullduplex/client.py",
            "vllm_omni/experimental/fullduplex/engine/contracts.py",
            "vllm_omni/experimental/fullduplex/joyvl/memory/memory.py",
            "vllm_omni/experimental/fullduplex/minicpmo45/adapter.py",
            "vllm_omni/experimental/fullduplex/minicpmo45/input.py",
            "vllm_omni/experimental/fullduplex/minicpmo45/runtime.py",
            "vllm_omni/experimental/fullduplex/minicpmo45/stage0.py",
            "vllm_omni/experimental/fullduplex/openai/audio.py",
            "vllm_omni/experimental/fullduplex/openai/realtime_input.py",
            "vllm_omni/experimental/fullduplex/openai/runtime_bridge.py",
            "vllm_omni/experimental/fullduplex/openai/serving.py",
            "vllm_omni/experimental/fullduplex/personaplex/input.py",
            "vllm_omni/experimental/fullduplex/personaplex/runtime_extension.py",
            "vllm_omni/experimental/fullduplex/personaplex/stage0.py",
            "vllm_omni/model_executor/models/indextts2/tokenizer_v2_5.py",
            "vllm_omni/model_executor/models/qwen3_tts/prompt_embeds_builder.py",
            "vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_tokenizer.py",
            "vllm_omni/tokenizers/mammoth_moda2_tokenizer.py",
            "vllm_omni/transformers_utils/processors/audiox.py",
        },
    ),
    "re": ForbiddenImport(
        pattern=r"^\s*(?:import\s+re(?:$|\s|,)|from\s+re\s+import)",
        tip="Replace 'import re' with 'import regex as re' or 'import regex'.",
        allowed_pattern=re.compile(r"^\s*import\s+regex(\s*|\s+as\s+re\s*)$"),
        allowed_dirs=_NON_LIBRARY_DIRS,
        allowed_files={
            "setup.py",
            "vllm_omni/benchmarks/data_modules/daily_omni_eval.py",
            "vllm_omni/config/stage_config.py",
            "vllm_omni/diffusion/layers/mot/ops/mot_gemm.py",
            "vllm_omni/diffusion/model_loader/diffusers_loader.py",
            "vllm_omni/diffusion/models/diffusers_adapter/pipeline_diffusers_adapter.py",
            "vllm_omni/diffusion/models/dreamid_omni/fusion.py",
            "vllm_omni/diffusion/models/dreamzero/pipeline_dreamzero.py",
            "vllm_omni/diffusion/models/glm_image/pipeline_glm_image.py",
            "vllm_omni/diffusion/models/gr00t/modeling/processing_gr00t_n1d7.py",
            "vllm_omni/diffusion/models/hunyuan_video/pipeline_hunyuan_video_1_5.py",
            "vllm_omni/diffusion/models/longcat_image/pipeline_longcat_image.py",
            "vllm_omni/diffusion/models/longcat_image/pipeline_longcat_image_edit.py",
            "vllm_omni/diffusion/models/longcat_video/pipeline_longcat_video_avatar.py",
            "vllm_omni/diffusion/models/minimax_h3/encoder.py",
            "vllm_omni/diffusion/models/nextstep_1_1/pipeline_nextstep_1_1.py",
            "vllm_omni/diffusion/models/omnivoice/pipeline_omnivoice.py",
            "vllm_omni/diffusion/models/soulx_singer/modules/preprocess/asr.py",
            "vllm_omni/diffusion/models/soulx_singer/preprocess/g2p.py",
            "vllm_omni/diffusion/profiler/diffusion_pipeline_profiler.py",
            "vllm_omni/distributed/omni_connectors/utils/env.py",
            "vllm_omni/entrypoints/cli/logo.py",
            "vllm_omni/entrypoints/openai/serving_speech.py",
            "vllm_omni/experimental/fullduplex/joyvl/memory/memory.py",
            "vllm_omni/model_executor/models/common/whisper_vq.py",
            "vllm_omni/model_executor/models/fish_speech/prompt_utils.py",
            "vllm_omni/model_executor/models/glm_tts/text_frontend.py",
            "vllm_omni/model_executor/models/higgs_audio_v2/higgs_audio_v2_tokenizer.py",
            "vllm_omni/model_executor/models/indextts2/indextts2_talker.py",
            "vllm_omni/model_executor/models/indextts2/text_processing_v2_5.py",
            "vllm_omni/model_executor/models/indextts2/utils/common.py",
            "vllm_omni/model_executor/models/ming_flash_omni/text_processing.py",
            "vllm_omni/model_executor/models/ming_flash_omni/vision_encoder.py",
            "vllm_omni/model_executor/models/ming_tts/prompt_assembly.py",
            "vllm_omni/model_executor/models/nemotron_voicechat/nemo_vendored/ear_tts_commons.py",
            "vllm_omni/model_executor/stage_input_processors/ming_flash_omni.py",
            "vllm_omni/outputs/output_modality.py",
            "vllm_omni/quantization/mxfp4_config.py",
            "vllm_omni/quantization/tools/merge_mxfp4_dualscale_checkpoint.py",
            "vllm_omni/utils/custom_voice_io.py",
        },
    ),
    "triton": ForbiddenImport(
        pattern=r"^(from|import)\s+triton(\s|\.|$)",
        tip="Use 'from vllm.triton_utils import triton' instead.",
        allowed_pattern=re.compile(r"from vllm.triton_utils import (triton|tl|tl, triton)"),
    ),
    "tilelang": ForbiddenImport(
        pattern=r"^(from|import)\s+tilelang(\s|\.|$)",
        tip="Use 'from vllm.tilelang_utils import tilelang, T' instead.",
        allowed_pattern=re.compile(
            r"from\s+vllm\.tilelang_utils\s+import\s+"
            r"(tilelang|T|T, tilelang|tilelang, T)\b"
        ),
    ),
    "huggingface_hub repo API": ForbiddenImport(
        # Catch `from huggingface_hub import `, including parenthesized,
        # multi-line imports.
        pattern=(
            r"^\s*from\s+huggingface_hub\s+import\s*\([^)]*\b(?:" + _HF_NAMES + r")\b"
            r"|"
            r"^\s*from\s+huggingface_hub\s+import\b[^\n]*\b(?:" + _HF_NAMES + r")\b"
        ),
        tip=(
            "Use the shared, vLLM-tagged helpers from "
            "vllm.transformers_utils.repo_utils (e.g. hf_api(), hf_fs(), "
            "list_repo_files, file_exists) instead of calling "
            "huggingface_hub directly."
        ),
        allowed_dirs=_NON_LIBRARY_DIRS,
        allowed_files={
            "vllm_omni/benchmarks/data_modules/daily_omni_dataset.py",
            "vllm_omni/benchmarks/data_modules/seed_tts_dataset.py",
            "vllm_omni/benchmarks/data_modules/seed_tts_eval.py",
            "vllm_omni/diffusion/lora/loader.py",
            "vllm_omni/diffusion/model_loader/hub_prefetch.py",
            "vllm_omni/diffusion/models/cosmos3/sound_tokenizer.py",
            "vllm_omni/diffusion/models/dreamzero/pipeline_dreamzero.py",
            "vllm_omni/diffusion/models/helios/pipeline_helios.py",
            "vllm_omni/diffusion/models/longcat_video/pipeline_longcat_video_avatar.py",
            "vllm_omni/diffusion/models/ltx2/ltx2_components.py",
            "vllm_omni/diffusion/models/magi_human/pipeline_magi_human.py",
            "vllm_omni/diffusion/models/omnivoice/pipeline_omnivoice.py",
            "vllm_omni/diffusion/models/pi0/pipeline_pi0.py",
            "vllm_omni/diffusion/models/sensenova_u1/pipeline_sensenova_u1.py",
            "vllm_omni/diffusion/models/soulx_singer/utils.py",
            "vllm_omni/diffusion/models/utils.py",
            "vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py",
            "vllm_omni/engine/arg_utils.py",
            "vllm_omni/engine/stage_init_utils.py",
            "vllm_omni/entrypoints/openai/api_server.py",
            "vllm_omni/entrypoints/openai/serving_speech.py",
            "vllm_omni/entrypoints/openai/tts_adapters/cosyvoice3.py",
            "vllm_omni/experimental/fullduplex/personaplex/runtime.py",
            "vllm_omni/experimental/fullduplex/personaplex/serving/server.py",
            "vllm_omni/model_executor/model_loader/weight_utils.py",
            "vllm_omni/model_executor/models/audex/checkpoint.py",
            "vllm_omni/model_executor/models/bagel/bagel.py",
            "vllm_omni/model_executor/models/cosyvoice3/cosyvoice3.py",
            "vllm_omni/model_executor/models/covo_audio/covo_audio_code2wav.py",
            "vllm_omni/model_executor/models/dots_tts/dots_tts_talker.py",
            "vllm_omni/model_executor/models/dynin_omni/dynin_omni_common.py",
            "vllm_omni/model_executor/models/dynin_omni/dynin_omni_token2audio.py",
            "vllm_omni/model_executor/models/glm_tts/glm_tts.py",
            "vllm_omni/model_executor/models/higgs_audio_v2/higgs_audio_v2_code2wav.py",
            "vllm_omni/model_executor/models/higgs_audio_v2/higgs_audio_v2_tokenizer.py",
            "vllm_omni/model_executor/models/higgs_audio_v3/higgs_audio_v3_code2wav.py",
            "vllm_omni/model_executor/models/indextts2/preprocess_utils.py",
            "vllm_omni/model_executor/models/indextts2/s2mel/modules/bigvgan.py",
            "vllm_omni/model_executor/models/indextts2/tokenizer_v2_5.py",
            "vllm_omni/model_executor/models/ming_tts/speaker_extractor.py",
            "vllm_omni/model_executor/models/minicpmo_4_5/minicpmo_4_5_code2wav.py",
            "vllm_omni/model_executor/models/omnivoice/omnivoice.py",
            "vllm_omni/model_executor/models/personaplex/personaplex_mimi.py",
            "vllm_omni/model_executor/models/step_audio2/step_audio2_token2wav.py",
            "vllm_omni/model_executor/models/voxtral_tts/voxtral_tts.py",
            "vllm_omni/transformers_utils/configs/higgs_audio_v3.py",
        },
    ),
}


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/")


def check_file(path: str) -> int:
    path = _normalize_path(path)
    with open(path, encoding="utf-8") as f:
        content = f.read()
    return_code = 0
    # Check all patterns in the whole file
    for import_name, forbidden_import in CHECK_IMPORTS.items():
        # Skip files that are allowed for this import
        if path in forbidden_import.allowed_files:
            continue
        # Skip directories that are allowed for this import
        if any(path.startswith(prefix) for prefix in forbidden_import.allowed_dirs):
            continue
        # Search for forbidden imports
        for match in re.finditer(forbidden_import.pattern, content, re.MULTILINE):
            # Check if it's allowed
            if forbidden_import.allowed_pattern.match(match.group()):
                continue
            # Skip matches inside a comment
            line_start = content.rfind("\n", 0, match.start()) + 1
            if "#" in content[line_start : match.start()]:
                continue
            # Calculate line number from match position
            line_num = content[: match.start() + 1].count("\n") + 1
            print(
                f"{path}:{line_num}: "
                "\033[91merror:\033[0m "  # red color
                f"Found forbidden import: {import_name}. {forbidden_import.tip}"
            )
            return_code = 1
    return return_code


def main():
    returncode = 0
    for path in sys.argv[1:]:
        returncode |= check_file(path)
    return returncode


def test_regex():
    def matches(rule: str, content: str) -> bool:
        return bool(re.search(CHECK_IMPORTS[rule].pattern, content, re.MULTILINE))

    pickle_cases = [
        # Should match
        ("import pickle", True),
        ("import cloudpickle", True),
        ("import pickle as pkl", True),
        ("import cloudpickle as cpkl", True),
        ("from pickle import *", True),
        ("from cloudpickle import dumps", True),
        ("from pickle import dumps, loads", True),
        ("from cloudpickle import (dumps, loads)", True),
        ("    import pickle", True),
        ("\timport cloudpickle", True),
        ("from pickle import loads", True),
        ("import pickle, os", True),
        ("import os, pickle", True),
        ("import os, pickle as pkl", True),
        ("import pickle as pkl, os", True),
        ("import os.path, cloudpickle", True),
        ("import cloudpickle, json", True),
        # Should not match
        ("import somethingelse", False),
        ("from somethingelse import pickle", False),
        ("# import pickle", False),
        ("print('import pickle')", False),
        ("import pickleas as asdf", False),
    ]
    for i, (content, should_match) in enumerate(pickle_cases):
        result = matches("pickle/cloudpickle", content)
        assert result == should_match, f"pickle case {i} failed: {content!r} (expected {should_match}, got {result})"

    tilelang_cases = [
        # Should match
        ("import tilelang", True),
        ("import tilelang.language as T", True),
        ("from tilelang.jit import JITImpl", True),
        ("from tilelang.jit.kernel import JITKernel", True),
        # Should not match: indented (local) imports are allowed, mirroring
        # the "triton" rule, so mocked-module test imports are not flagged.
        ("    import tilelang", False),
        ("    from tilelang.jit import JITImpl", False),
        ("from vllm.tilelang_utils import tilelang", False),
        ("from vllm.tilelang_utils import T", False),
        ("from vllm.tilelang_utils import T, tilelang", False),
        ("from vllm.tilelang_utils import tilelang, T", False),
        ("import tilelang_kernels", False),
    ]
    for i, (content, should_match) in enumerate(tilelang_cases):
        result = matches("tilelang", content)
        assert result == should_match, f"tilelang case {i} failed: {content!r} (expected {should_match}, got {result})"

    hf_cases = [
        # Should match
        ("from huggingface_hub import snapshot_download", True),
        ("from huggingface_hub import hf_hub_download", True),
        ("from huggingface_hub import HfApi", True),
        ("from huggingface_hub import HfFileSystem", True),
        ("from huggingface_hub import list_repo_files", True),
        ("from huggingface_hub import try_to_load_from_cache", True),
        ("    from huggingface_hub import snapshot_download", True),
        ("from huggingface_hub import PyTorchModelHubMixin, hf_hub_download", True),
        ("from huggingface_hub import (snapshot_download)", True),
        # Parenthesized multi-line import must not bypass the hook
        ("from huggingface_hub import (\n    snapshot_download,\n)", True),
        (
            "from huggingface_hub import (\n    PyTorchModelHubMixin,\n    HfApi,\n)",
            True,
        ),
        # Should not match
        ("import huggingface_hub", False),
        ("import huggingface_hub as hf", False),
        ("from huggingface_hub import PyTorchModelHubMixin", False),
        ("from huggingface_hub.constants import HF_HUB_CACHE", False),
        ("from huggingface_hub.utils import EntryNotFoundError", False),
        ("from vllm.transformers_utils.repo_utils import hf_api", False),
        ("from huggingface_hub import (\n    PyTorchModelHubMixin,\n)", False),
        ("# from huggingface_hub import snapshot_download", False),
    ]
    for i, (content, should_match) in enumerate(hf_cases):
        result = matches("huggingface_hub repo API", content)
        assert result == should_match, (
            f"huggingface_hub case {i} failed: {content!r} (expected {should_match}, got {result})"
        )

    print("All regex tests passed.")


if __name__ == "__main__":
    if "--test-regex" in sys.argv:
        test_regex()
    else:
        sys.exit(main())
