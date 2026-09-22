"""getInfo IrrMapper annual irrigation time series for the Example 7 cohort.

Writes one row per field with columns ``irr_1987``..``irr_2024`` = fraction of the
field classified irrigated by IrrMapper that year (IrrMapperComp, 30 m). This is a
cohort QC check, independent of the metered volumes: the 100 irrigated fields should
read ~1 in their metered years, and the 10 rainfed controls should read ~0 throughout.

Uses the same server-side stack + ``.getInfo()`` path as the ``properties`` extraction
step (``get_irrigation(dest="local")``). CO (SLV) and ID (ESPA) are both inside
IrrMapper's western-11 coverage, so ``lanid=False`` (pure IrrMapper).

    uv run python examples/7_Applied_Water/irrmapper_timeseries.py
"""

import os
from pathlib import Path

from swimrs.data_extraction.ee.ee_props import get_irrigation
from swimrs.data_extraction.ee.ee_utils import is_authorized
from swimrs.swim.config import ProjectConfig

HERE = Path(__file__).resolve().parent


def main() -> None:
    cfg = ProjectConfig()
    conf = HERE / "7_Applied_Water.toml"
    if os.path.isdir("/data/ssd2/swim"):
        cfg.read_config(str(conf))
    else:
        cfg.read_config(str(conf), project_root_override=str(HERE.parent))

    is_authorized()
    out_dir = os.path.join(cfg.data_dir, "properties", "getinfo")
    get_irrigation(
        cfg.fields_shapefile,
        f"{cfg.project_name}_irrmapper_timeseries",
        selector=cfg.feature_id_col,
        lanid=False,
        dest="local",
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
