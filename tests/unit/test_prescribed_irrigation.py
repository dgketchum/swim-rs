"""Tests for the WP-B0 prescribed-irrigation override in ``_run_loop_jit``.

The override (loop_fast.py step 13b) lets a diagnostic exogenous daily-irrigation
series REPLACE the internal scheduler's ``irr_sim`` per field/day. A ``NaN``
sentinel means "no prescription, keep the scheduler". These tests pin the three
contract points required before the capability is trusted for the WP-B1
prescribed-irrigation attribution experiment:

  (a) an all-NaN prescription is a strict no-op — bit-for-bit identical to the
      scheduler-only baseline;
  (b) a finite prescribed value appears verbatim in ``out_irr_sim`` on its day,
      and NaN days fall back to the scheduler;
  (c) prescribed water actually enters the balance — ``depl_root`` drops and
      ``theta_avail`` rises relative to baseline.
"""

import numpy as np

from swimrs.process.loop_fast import _run_loop_jit

from .test_loop_fast_bounds import _make_inputs, _unpack

N_DAYS = 10
N_FIELDS = 2


def _theta_avail(out, awc, zr_max, field):
    """Replicate evaluate.theta_available for one field's time series."""
    soil_water = awc * out["zr"][:, field] - out["depl_root"][:, field] + out["daw3"][:, field]
    return soil_water / (zr_max * 1000.0)


def _irrigating_inputs(**overrides):
    """Dry, warm, irrigation-active inputs so the scheduler actually irrigates."""
    base = dict(
        n_days=N_DAYS,
        n_fields=N_FIELDS,
        all_prcp=np.zeros((N_DAYS, N_FIELDS)),  # dry: depletion grows past RAW
        irr_status=np.ones(N_FIELDS),  # scheduler eligible
        all_irr_flag=np.ones((N_DAYS, N_FIELDS)),  # every day an irrigation day
        depl_root_init=np.full(N_FIELDS, 120.0),  # start already depleted
    )
    base.update(overrides)
    return _make_inputs(**base)


def test_all_nan_is_scheduler_identity():
    """An all-NaN prescription reproduces the scheduler-only run bit-for-bit.

    Verified by re-injecting the scheduler's OWN ``out_irr_sim`` as the
    prescription: since the override only assigns ``irr_sim[i] = pv`` for finite
    ``pv``, feeding back the scheduler's exact output (or NaN) must leave every
    one of the 25 outputs untouched. A non-trivial baseline (the scheduler
    actually irrigates) makes the identity meaningful rather than vacuous.
    """
    inputs = _irrigating_inputs()
    baseline = _unpack(_run_loop_jit(**inputs))

    assert baseline["irr_sim"].sum() > 0.0, (
        "baseline must irrigate for a non-trivial identity check"
    )

    reinjected = dict(inputs)
    reinjected["all_prescribed_irr"] = baseline["irr_sim"].copy()
    replayed = _unpack(_run_loop_jit(**reinjected))

    for key in baseline:
        assert np.array_equal(baseline[key], replayed[key]), (
            f"output '{key}' changed under scheduler re-injection"
        )


def test_prescribed_constant_appears_verbatim():
    """A finite prescription lands verbatim in out_irr_sim; NaN days fall back.

    (1) A constant overrides even an ACTIVE scheduler (override beats scheduler).
    (2) With the scheduler SILENT (irr_status=0) the sentinel is unambiguous:
        finite days show the prescription verbatim, NaN days stay at the
        scheduler's value (0). Note the scheduler cannot be compared against a
        pristine baseline on NaN days when it is active, because prescribed water
        alters depletion/carryover and the trajectory legitimately diverges.
    """
    const = 5.0

    # (1) constant on every day/field, active scheduler -> out is exactly const.
    all_const = _irrigating_inputs(all_prescribed_irr=np.full((N_DAYS, N_FIELDS), const))
    out_const = _unpack(_run_loop_jit(**all_const))
    assert np.array_equal(out_const["irr_sim"], np.full((N_DAYS, N_FIELDS), const))

    # (2) scheduler silent -> NaN days are exactly 0, finite days are verbatim.
    silent = dict(
        n_days=N_DAYS,
        n_fields=N_FIELDS,
        all_prcp=np.zeros((N_DAYS, N_FIELDS)),
        irr_status=np.zeros(N_FIELDS),
    )
    mixed = np.full((N_DAYS, N_FIELDS), np.nan)
    even = np.arange(N_DAYS) % 2 == 0
    mixed[even, :] = const
    out_mixed = _unpack(_run_loop_jit(**_make_inputs(all_prescribed_irr=mixed, **silent)))

    assert np.array_equal(
        out_mixed["irr_sim"][even, :], np.full((int(even.sum()), N_FIELDS), const)
    )
    assert np.array_equal(out_mixed["irr_sim"][~even, :], np.zeros((int((~even).sum()), N_FIELDS)))


def test_depl_root_and_theta_respond():
    """Prescribed water enters the balance: depl_root drops, theta_avail rises.

    Scheduler is OFF (irr_status=0) and it is dry, so the baseline only depletes.
    A large early prescription must reduce end-of-run root-zone depletion and
    raise available soil water relative to that baseline.
    """
    awc, zr_max = 150.0, 1.2
    dry = dict(
        n_days=N_DAYS,
        n_fields=N_FIELDS,
        all_prcp=np.zeros((N_DAYS, N_FIELDS)),
        irr_status=np.zeros(N_FIELDS),  # scheduler contributes nothing
        depl_root_init=np.full(N_FIELDS, 60.0),
        awc=np.full(N_FIELDS, awc),
        zr_max=np.full(N_FIELDS, zr_max),
    )
    baseline = _unpack(_run_loop_jit(**_make_inputs(**dry)))
    assert baseline["irr_sim"].sum() == 0.0, (
        "scheduler must be silent so the response is purely prescribed"
    )

    prescribed = np.full((N_DAYS, N_FIELDS), np.nan)
    prescribed[:3, :] = 30.0  # three big early applications
    watered = _unpack(_run_loop_jit(**_make_inputs(all_prescribed_irr=prescribed, **dry)))

    # Prescribed water lands verbatim and lowers end-of-run depletion.
    assert np.array_equal(watered["irr_sim"][:3, :], np.full((3, N_FIELDS), 30.0))
    assert (watered["depl_root"][-1, :] < baseline["depl_root"][-1, :]).all()

    # theta_avail (evaluate.theta_available) rises for every field.
    for f in range(N_FIELDS):
        assert (
            _theta_avail(watered, awc, zr_max, f)[-1] > _theta_avail(baseline, awc, zr_max, f)[-1]
        )
