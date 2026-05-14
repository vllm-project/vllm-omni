import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _load_html_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "tools" / "nightly" / "generate_nightly_perf_html.py"
    spec = importlib.util.spec_from_file_location("generate_nightly_perf_html", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_generate_html_report_with_perf_templates(tmp_path: Path):
    module = _load_html_module()
    repo_root = Path(__file__).resolve().parents[1]
    perf_scripts_dir = repo_root / "tests" / "dfx" / "perf" / "scripts"

    omni_template_path = perf_scripts_dir / "result_omni_template.json"
    diffusion_template_path = perf_scripts_dir / "diffusion_result_template.json"

    omni_record = json.loads(omni_template_path.read_text(encoding="utf-8"))
    diffusion_records = json.loads(diffusion_template_path.read_text(encoding="utf-8"))

    input_dir = tmp_path / "input"
    diffusion_input_dir = tmp_path / "diffusion_input"
    input_dir.mkdir()
    diffusion_input_dir.mkdir()

    omni_result_file = input_dir / "result_test_perf_random_1_4_in2500_out900_20260415-185642.json"
    diffusion_result_file = diffusion_input_dir / "diffusion_result_qwen_image_edit_20260415-193200.json"
    omni_result_file.write_text(json.dumps(omni_record, ensure_ascii=False, indent=2), encoding="utf-8")
    diffusion_result_file.write_text(json.dumps(diffusion_records, ensure_ascii=False, indent=2), encoding="utf-8")

    output_file = tmp_path / "nightly_perf_v2.html"
    module.generate_html_report(
        input_dir=str(input_dir),
        diffusion_input_dir=str(diffusion_input_dir),
        output_file=str(output_file),
    )

    assert output_file.exists()
    html = output_file.read_text(encoding="utf-8")
    assert "Nightly Performance Report" in html
    assert "Omni records <strong>1</strong>" in html
    assert f"Diffusion records <strong>{len(diffusion_records)}</strong>" in html
    assert "const DIFF_DATA =" in html
