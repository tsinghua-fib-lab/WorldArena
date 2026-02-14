"""Register third-party metric dependencies on sys.path."""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def _add(path: Path):
	p = str(path)
	if p not in sys.path:
		sys.path.append(p)


# SEA-RAFT (optical flow)
_add(BASE_DIR / "SEA-RAFT")

# VFIMamba (video interpolation model)
_add(BASE_DIR / "VFIMamba")

# Fallback for historical relative lookups used by copied code
_add(BASE_DIR)
