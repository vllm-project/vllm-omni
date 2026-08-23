import json
from shutil import rmtree

import pytest
from filelock import ReadWriteLock

from tests.model_tests.diffusion.model_settings import DIFFUSION_TEST_SETTINGS


@pytest.fixture(scope="session")
def tiny_model_paths(request, run_level, tmp_path_factory):
    """Build or download the tiny models for the selected tests.

    At core_model level, builds tiny models via the builder function.
    At advanced_model / full_model level, uses the real HF model path.

    NOTE: this is session scoped to avoid churn in tiny model creation,
    but will ensure all the tiny models you need are created for the selected tests
    before it starts to execute them."""
    base_path = tmp_path_factory.getbasetemp().parent / "tiny_models"
    use_lock = ReadWriteLock(base_path.with_suffix(".use.lock"))
    build_lock = ReadWriteLock(base_path.with_suffix(".build.lock"))
    data_file = base_path.with_suffix(".json")

    model_paths = {}
    built_paths = []

    with use_lock.read_lock():
        with build_lock.write_lock():
            if data_file.is_file():
                model_paths = json.loads(data_file.read_text())
            else:
                print("Initializing tiny models...")
                for item in request.session.items:
                    if not hasattr(item, "callspec"):
                        continue
                    model_name = item.callspec.params.get("model_name")
                    if model_name is None or model_name not in DIFFUSION_TEST_SETTINGS:
                        continue
                    if model_name not in model_paths:
                        settings = DIFFUSION_TEST_SETTINGS[model_name]
                        if run_level == "core_model":
                            print(f"Calling tiny model builder for: {model_name}")
                            path = settings.builder()
                            built_paths.append(path)
                        else:
                            print(f"Run level is {run_level}; {model_name} will use the full model")
                            path = settings.model
                        model_paths[model_name] = path
                data_file.write_text(json.dumps(model_paths))

        yield model_paths

    with use_lock.write_lock():
        if built_paths:
            data_file.unlink(missing_ok=True)
        for path in built_paths:
            print(f"Removing tiny model: {path}")
            rmtree(path, ignore_errors=True)
