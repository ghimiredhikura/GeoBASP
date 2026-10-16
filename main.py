"""Run the released workflows from an existing Python environment."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from nepal_cnn.cli.main import main

if __name__ == "__main__":
    main()
