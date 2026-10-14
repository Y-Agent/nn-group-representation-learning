"""Compatibility entry point; prefer python -m paper_reproduction render."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_reproduction.__main__ import main

if __name__ == "__main__":
    sys.argv[1:1] = ["render"]
    main()
