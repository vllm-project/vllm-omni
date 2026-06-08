from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[4]
PIPELINE_PATH = REPO_ROOT / "vllm_omni/diffusion/models/omniweaving/pipeline_omniweaving.py"
TRANSFORMER_PATH = REPO_ROOT / "vllm_omni/diffusion/models/hunyuan_video/hunyuan_video_15_transformer.py"
COMMON_PATH = REPO_ROOT / "examples/offline_inference/omniweaving/omniweaving_common.py"
END2END_PATH = REPO_ROOT / "examples/offline_inference/omniweaving/end2end.py"


def _load_common_module():
    spec = importlib.util.spec_from_file_location("omniweaving_common_for_tests", COMMON_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_end2end_module():
    spec = importlib.util.spec_from_file_location("omniweaving_end2end_for_tests", END2END_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _find_node(tree: ast.AST, name: str) -> ast.FunctionDef | ast.ClassDef:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            return node
    raise AssertionError(f"Could not find AST node {name!r}")


def _load_resolve_single_conditioning_image():
    tree = ast.parse(PIPELINE_PATH.read_text(encoding="utf-8"))
    node = _find_node(tree, "_resolve_single_conditioning_image")
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {"Any": Any}
    exec(compile(module, str(PIPELINE_PATH), "exec"), namespace)
    return namespace["_resolve_single_conditioning_image"]


def _load_offline_snapshot_helpers():
    tree = ast.parse(PIPELINE_PATH.read_text(encoding="utf-8"))
    nodes = [
        _find_node(tree, "_offline_env_enabled"),
        _find_node(tree, "_resolve_cached_hub_snapshot_if_offline"),
    ]
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {
        "logger": SimpleNamespace(debug=lambda *args, **kwargs: None),
        "os": __import__("os"),
    }
    exec(compile(module, str(PIPELINE_PATH), "exec"), namespace)
    return namespace


def test_omniweaving_rejects_multi_image_payloads_before_generation():
    resolve_single_image = _load_resolve_single_conditioning_image()

    assert resolve_single_image([]) is None
    assert resolve_single_image(["only.png"]) == "only.png"
    with pytest.raises(ValueError, match="Multi-image OmniWeaving requests are not implemented"):
        resolve_single_image(["first.png", "second.png"])


def test_omniweaving_example_rejects_multi_image_paths(tmp_path: Path):
    common = _load_common_module()
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(first)
    Image.new("RGB", (4, 4), color=(0, 255, 0)).save(second)

    with pytest.raises(ValueError, match="Multi-image OmniWeaving requests are not implemented"):
        common.build_prompt_payload("animate", image_paths=[str(first), str(second)])


def test_omniweaving_offline_external_paths_use_cached_hub_snapshots(monkeypatch, tmp_path: Path):
    helpers = _load_offline_snapshot_helpers()
    resolve_cached_snapshot = helpers["_resolve_cached_hub_snapshot_if_offline"]
    calls = []

    def fake_snapshot_download(model_path: str, *, local_files_only: bool):
        calls.append((model_path, local_files_only))
        return f"/cached/{model_path.replace('/', '--')}"

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=fake_snapshot_download))

    assert resolve_cached_snapshot("Qwen/Qwen2.5-VL-7B-Instruct") == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert calls == []

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert resolve_cached_snapshot("Qwen/Qwen2.5-VL-7B-Instruct") == "/cached/Qwen--Qwen2.5-VL-7B-Instruct"
    assert calls == [("Qwen/Qwen2.5-VL-7B-Instruct", True)]

    local_model = tmp_path / "model"
    local_model.mkdir()
    assert resolve_cached_snapshot(str(local_model)) == str(local_model)
    assert calls == [("Qwen/Qwen2.5-VL-7B-Instruct", True)]


def test_omniweaving_end2end_rejects_multi_image_mode():
    end2end = _load_end2end_module()

    assert "mi2v" not in end2end.FLOW_SHIFT_PRESETS
    args = SimpleNamespace(image_path=None, image_paths=["first.png", "second.png"])
    with pytest.raises(ValueError, match="Multi-image OmniWeaving requests are not implemented"):
        end2end._infer_mode(args)


def test_omniweaving_empty_glyph_byt5_mask_is_inactive():
    source = PIPELINE_PATH.read_text(encoding="utf-8")
    assert "glyph_text_embeds_mask = torch.zeros" in source
    assert "glyph_text_embeds_mask = torch.ones((1, self.tokenizer_2_max_length)" not in source


def test_hunyuan_single_attention_has_loadable_output_projection():
    tree = ast.parse(TRANSFORMER_PATH.read_text(encoding="utf-8"))
    single_attention = _find_node(tree, "HunyuanVideo15SingleAttention")
    assert isinstance(single_attention, ast.ClassDef)

    source = ast.get_source_segment(TRANSFORMER_PATH.read_text(encoding="utf-8"), single_attention)
    assert source is not None
    assert "self.to_out = nn.ModuleList" in source
    assert "RowParallelLinear" in source
    assert "return self.to_out[0](hidden_states)" in source

    pipeline_source = PIPELINE_PATH.read_text(encoding="utf-8")
    assert 'map_k(f"{p}.attn_proj.weight", f"{t_p}.attn.to_out.0.weight")' in pipeline_source
    assert 'map_k(f"{p}.attn_proj.bias", f"{t_p}.attn.to_out.0.bias")' in pipeline_source
