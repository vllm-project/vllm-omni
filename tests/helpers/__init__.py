"""Shared, importable test helper utilities.

Submodules (``assertions``, ``env``, ``media``, ``runtime``, …) are imported
explicitly by callers. Avoid star-importing everything here: that ran before
refactor only inside the old monolithic ``conftest``; a greedy ``__init__``
changes import order and can affect in-process Omni (``OmniRunner`` / offline
e2e) vs subprocess-based ``OmniServer`` tests.
"""

import pytest


def skip_if_gated_repo_inaccessible(repo_id: str) -> None:
    """Skip the test if a gated HuggingFace repo is not accessible.

    Tries to download the model's config.json via ``hf_hub_download``,
    which performs an actual file-access check (unlike ``HfApi().model_info()``
    that only checks metadata).  If the token has metadata access but not
    file-download access, ``hf_hub_download`` will raise ``GatedRepoError``
    and we skip cleanly.
    """
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    except Exception:
        return
    try:
        hf_hub_download(repo_id=repo_id, filename="config.json")
    except GatedRepoError as exc:
        pytest.skip(
            f"Skipping: gated HF repo {repo_id!r} inaccessible to the current "
            f"HF_TOKEN ({exc}). See docs/contributing/ci/hf_credentials.md."
        )
    except RepositoryNotFoundError as exc:
        pytest.skip(f"Skipping: HF repo {repo_id!r} not found ({exc}).")
    except Exception:
        return
