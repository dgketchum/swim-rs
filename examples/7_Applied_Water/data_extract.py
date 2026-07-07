"""Earth Engine extraction for Example 7, reusing the Example 5 pipeline.

Thin wrapper: it imports the Example 5 ``data_extract`` module and repoints its
config loader at ``7_Applied_Water.toml``. All extraction logic (NDVI, 6-model
OpenET v2.1 ETf, GridMET, SNODAS, SSURGO/IrrMapper properties, OpenET refET) is
identical to E2 — these fields are CONUS, so the same inputs apply.

    uv run python examples/7_Applied_Water/data_extract.py [--steps ...] [--sites ...]
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
E2 = HERE.parent / "5_Flux_Ensemble"
if str(E2) not in sys.path:
    sys.path.insert(0, str(E2))

import data_extract as e2  # noqa: E402  (Example 5 pipeline)

from swimrs.swim.config import ProjectConfig  # noqa: E402


def _load_config() -> ProjectConfig:
    conf = HERE / "7_Applied_Water.toml"
    cfg = ProjectConfig()
    if os.path.isdir("/data/ssd2/swim"):
        cfg.read_config(str(conf))
    else:
        cfg.read_config(str(conf), project_root_override=str(HERE.parent))
    return cfg


# Repoint the E2 pipeline's config loader at this project.
e2._load_config = _load_config

if __name__ == "__main__":
    e2.main()
