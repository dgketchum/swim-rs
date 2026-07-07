"""Transfer-path scoring for Example 7 (no local calibration).

Applies the frozen Example-2 CONUS-cropland median parameter vector
(``examples/6_Flux_International/transfer/ex5_cropland_params.json``) unchanged to
every field and scores simulated annual applied water against the withheld metered
truth — the ungauged-field / transferability half of the claim. This is a thin
driver over ``evaluate_applied_water.py``: it fixes ``--params-json`` to the frozen
vector and ``--label transfer``.

    uv run python examples/7_Applied_Water/transfer_applied_water.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import evaluate_applied_water as ev  # noqa: E402

DEFAULT_VECTOR = HERE.parent / "6_Flux_International" / "transfer" / "ex5_cropland_params.json"


def main() -> None:
    argv = sys.argv[1:]
    if not any(a.startswith("--params-json") for a in argv):
        argv += ["--params-json", str(DEFAULT_VECTOR)]
    if not any(a.startswith("--label") for a in argv):
        argv += ["--label", "transfer"]
    sys.argv = [sys.argv[0]] + argv
    ev.main()


if __name__ == "__main__":
    main()
