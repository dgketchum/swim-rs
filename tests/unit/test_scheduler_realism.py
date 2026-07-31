"""WP-C1 scheduler-realism knob tests for the fast loop.

Covers the three new first-class scheduler parameters wired through
``_run_loop_jit`` / ``run_daily_loop_fast``:

- ``refill_frac`` — irrigation refill target as a fraction of depletion
  (default 1.1 = legacy refill-past-FC; < 1.0 leaves the profile below FC).
- ``min_irr_days`` — minimum return interval between irrigation events
  (default 0 = no constraint).
- ``irr_depth`` — optional fixed application depth per event (default 0 = off).

The bit-for-bit test compares the default-knob run against a golden captured
from the pre-WP-C1 committed loop (``tests/fixtures/scheduler_baseline_golden.npz``,
generated from ``git show HEAD:...loop_fast.py``), proving the new knobs at their
defaults — together with the all-NaN prescribed-irrigation override — reproduce
the historical scheduler exactly.
"""

from pathlib import Path

import numpy as np

from swimrs.process.cover_modes import COVER_MODE_KCB
from swimrs.process.kcb_modes import KCB_MODE_SIGMOID
from swimrs.process.loop_fast import _run_loop_jit
from swimrs.process.state import FieldProperties

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "scheduler_baseline_golden.npz"

N_DAYS = 40
N_FIELDS = 3  # fields 0,1 irrigated; field 2 rainfed


def _scenario(*, refill_frac=1.1, min_irr_days=0.0, irr_depth=0.0):
    """Deterministic irrigated dry-season scenario matching the golden fixture.

    Field 0 and 1 are irrigated, field 2 is rainfed. Scheduler knobs are
    broadcast across all fields.
    """
    n_days, n_fields = N_DAYS, N_FIELDS
    prcp = np.zeros((n_days, n_fields))
    prcp[5] = 8.0
    prcp[20] = 12.0
    return dict(
        n_days=n_days,
        n_fields=n_fields,
        all_ndvi=np.full((n_days, n_fields), 0.6),
        all_etr=np.full((n_days, n_fields), 7.0),
        all_prcp=prcp,
        all_tmin=np.full((n_days, n_fields), 12.0),
        all_tmax=np.full((n_days, n_fields), 30.0),
        all_srad=np.full((n_days, n_fields), 260.0),
        all_irr_flag=np.ones((n_days, n_fields)),
        all_f_sub=np.zeros((n_days, n_fields)),
        all_prescribed_irr=np.full((n_days, n_fields), np.nan),
        awc=np.full(n_fields, 180.0),
        rew=np.full(n_fields, 8.0),
        tew=np.full(n_fields, 25.0),
        cn2=np.full(n_fields, 78.0),
        zr_max=np.full(n_fields, 1.2),
        zr_min=np.full(n_fields, 0.1),
        mad=np.full(n_fields, 0.5),
        refill_frac=np.full(n_fields, refill_frac),
        min_irr_days=np.full(n_fields, min_irr_days),
        irr_depth=np.full(n_fields, irr_depth),
        irr_status=np.array([1.0, 1.0, 0.0]),
        perennial=np.zeros(n_fields),
        gw_status=np.zeros(n_fields),
        ke_max=np.full(n_fields, 1.2),
        kc_max=np.full(n_fields, 1.35),
        kc_min=np.full(n_fields, 0.15),
        ndvi_k=np.full(n_fields, 7.0),
        ndvi_0=np.full(n_fields, 0.45),
        ndvi_alpha=np.full(n_fields, 0.2),
        ndvi_beta=np.full(n_fields, 1.25),
        swe_alpha=np.full(n_fields, 0.3),
        swe_beta=np.full(n_fields, 2.0),
        kr_damp=np.full(n_fields, 0.5),
        ks_damp=np.full(n_fields, 0.5),
        max_irr_rate=np.full(n_fields, 25.0),
        # WP-C7 mad split: -1.0 sentinel reuses mad*taw for the Ks stress onset
        stress_depl_frac=-1.0,
        depl_root_init=np.full(n_fields, 60.0),
        depl_ze_init=np.full(n_fields, 10.0),
        swe_init=np.zeros(n_fields),
        albedo_init=np.full(n_fields, 0.45),
        kr_init=np.ones(n_fields),
        ks_init=np.ones(n_fields),
        zr_init=np.full(n_fields, 0.4),
        s_init=np.full(n_fields, 84.7),
        s1_init=np.full(n_fields, 84.7),
        s2_init=np.full(n_fields, 84.7),
        s3_init=np.full(n_fields, 84.7),
        s4_init=np.full(n_fields, 84.7),
        daw3_init=np.zeros(n_fields),
        taw3_init=np.zeros(n_fields),
        # Transpiration cover weight: default = the FAO-56 Eq. 76 Kcb-derived form
        cover_mode=COVER_MODE_KCB,
        cover_lin_lo=-1.0,
        cover_lin_hi=-1.0,
        # NDVI->Kcb curve: 0 = the default sigmoid on (ndvi_k, ndvi_0)
        kcb_mode=KCB_MODE_SIGMOID,
    )


_OUT_NAMES = [
    "eta",
    "etf",
    "kcb",
    "ke",
    "ks",
    "kr",
    "runoff",
    "rain",
    "melt",
    "swe",
    "depl_root",
    "dperc",
    "irr_sim",
    "gw_sim",
    "daw3",
    "zr",
]


def _run(**knobs):
    res = _run_loop_jit(**_scenario(**knobs))
    return {name: res[i] for i, name in enumerate(_OUT_NAMES)}


class TestBitForBitDefaults:
    def test_default_knobs_match_precrange_golden(self):
        """Default knobs (1.1, 0, 0) + all-NaN prescribed_irr reproduce the
        pre-WP-C1 committed loop bit-for-bit."""
        golden = np.load(FIXTURE)
        out = _run()  # defaults
        for key in ("eta", "depl_root", "irr_sim", "daw3", "zr"):
            np.testing.assert_array_equal(
                out[key], golden[key], err_msg=f"{key} diverged from baseline golden"
            )

    def test_field_properties_defaults_are_legacy(self):
        """A FieldProperties built without the knobs defaults to legacy values."""
        props = FieldProperties(n_fields=4)
        np.testing.assert_array_equal(props.refill_frac, np.full(4, 1.1))
        np.testing.assert_array_equal(props.min_irr_days, np.zeros(4))
        np.testing.assert_array_equal(props.irr_depth, np.zeros(4))


class TestRefillFracDrydown:
    def test_below_fc_refill_increases_drydown(self):
        """A below-FC refill target must leave the irrigated root zone drier —
        higher mean root-zone depletion — than the refill-past-FC default.

        Note: this is a mechanical monotonicity guarantee only. Whether a
        below-FC refill *raises the variance* of θ is scenario-dependent (a
        partial refill can instead produce a tighter, more frequent oscillation
        band) and is settled empirically in the WP-C1 forward what-if, not here.
        """
        base = _run(refill_frac=1.1)
        dry = _run(refill_frac=0.5)
        for f in (0, 1):  # irrigated fields
            assert dry["depl_root"][:, f].mean() > base["depl_root"][:, f].mean()

    def test_rainfed_unaffected_by_refill_frac(self):
        """refill_frac must not touch a rainfed field (scheduler is skipped)."""
        base = _run(refill_frac=1.1)
        dry = _run(refill_frac=0.4)
        np.testing.assert_array_equal(base["depl_root"][:, 2], dry["depl_root"][:, 2])
        assert base["irr_sim"][:, 2].sum() == 0.0


class TestMinReturnInterval:
    def test_min_interval_reduces_event_frequency(self):
        """A minimum return interval must reduce the number of irrigation events.

        Uses a small fixed application depth (< max_irr_rate) so no event ever
        spills into a carryover day: each positive-irrigation day is then exactly
        one event, and event count is unambiguous.
        """
        base = _run(irr_depth=10.0, min_irr_days=0.0)
        gated = _run(irr_depth=10.0, min_irr_days=7.0)
        for f in (0, 1):
            n0 = int(np.sum(base["irr_sim"][:, f] > 0.0))
            ng = int(np.sum(gated["irr_sim"][:, f] > 0.0))
            assert ng < n0, f"field {f}: gated events {ng} not < baseline {n0}"

    def test_min_interval_spacing_respected(self):
        """Consecutive irrigation events are >= min_irr_days apart.

        Fixed small depth keeps events to single clean days (no carryover), so
        every positive-irrigation day is an event onset.
        """
        gated = _run(irr_depth=10.0, min_irr_days=10.0)
        for f in (0, 1):
            event_days = np.where(gated["irr_sim"][:, f] > 0.0)[0]
            if len(event_days) >= 2:
                assert np.all(np.diff(event_days) >= 10)


class TestFixedApplicationDepth:
    def test_irr_depth_sets_fixed_amount(self):
        """A fixed irr_depth (below max_irr_rate) makes each applied irrigation
        equal to that depth rather than the depletion-proportional refill."""
        depth = 15.0
        out = _run(irr_depth=depth, min_irr_days=6.0)  # spacing avoids carryover overlap
        for f in (0, 1):
            applied = out["irr_sim"][:, f]
            nonzero = applied[applied > 0.0]
            assert nonzero.size > 0
            # every applied amount is the fixed depth (<= max_irr_rate, no cap)
            np.testing.assert_allclose(nonzero, depth, atol=1e-9)
