"""Use the in-tree RSL-RL fork without modifying the active Python environment."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "rsl_rl"), str(ROOT / "source" / "omnicontact")]
