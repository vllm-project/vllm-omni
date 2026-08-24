# ruff: noqa: E402, I001
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.accuracy.image_to_image.gedit_bench import main


if __name__ == "__main__":
    raise SystemExit(main())
