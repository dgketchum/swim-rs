"""Unit tests for the E3 Experiment C provenance modules.

``examples/6_Flux_International/coverage_diagnostics.py`` and
``expc_paired_deltas.py`` are the tracked code behind the manuscript's
additional-dates paragraph. Two of the numbers they publish are
convention-dependent and one of them is a paired statistic, so the tests here
pin the definitions rather than the values:

1. **Both window conventions keep the same rescued numerator.** ``resample``
   (anchored 16-day bins, partial trailing bin retained) and ``floor`` (whole
   bins only) disagree on the *denominator* -- 11,220 vs 11,154 windows on the
   real cohort, hence 28.0% vs 28.4% -- but must agree exactly on how many
   Landsat-empty windows the auxiliary record fills. A change that moved the
   numerator would mean one of the two published rates is wrong.
2. **``interior`` gaps ignore year edges.** The manuscript's 38.3 -> 33.1 d
   comes from distances *between* consecutive captures; counting Jan-1-to-first
   and last-to-Dec-31 instead gives 50.8 -> 42.6 d. The two are not small
   perturbations of each other, so the convention is asserted directly.
3. **The estimand is the median of site-level paired differences**, not the
   difference of the two marginal medians. A constant offset makes the two
   coincide, so the construction here deliberately uses nonconstant deltas: on
   the real daily KGE data the paired median is +0.0020 and the difference of
   medians is +0.0063.
4. **Absolute MBE is rebuilt from the two arms.** The archived ``bias_delta``
   is a signed difference, while Table S10 reports the change in |MBE|; a site
   whose bias flips sign is scored very differently by the two.
5. **The archive is hash-gated and the reproduction is exact.**
   ``expc_paired_deltas.py`` consumes the archived paired metrics rather than
   recomputing them, so the tests check that drift fails closed and that the
   published Table S10 comes back row for row. The RNG scope is included: a
   per-metric reseed still reproduces 5 of the 8 rows, so only the full
   comparison catches it.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
E6_DIR = REPO_ROOT / "examples" / "6_Flux_International"

# Every example directory has its own ``evaluate.py``, and this one also holds a
# ``shapefile.py`` that would shadow the ``shapefile`` package for the rest of the
# session. Same collision handling as ``test_e3_stratified_transfer_run.py``.
SIBLINGS = ("evaluate", "coverage_diagnostics")


def _load(module_name):
    saved = {name: sys.modules.pop(name, None) for name in SIBLINGS}
    sys.path.insert(0, str(E6_DIR))
    spec = importlib.util.spec_from_file_location(module_name, E6_DIR / f"{module_name}.py")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = [p for p in sys.path if p != str(E6_DIR)]
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod
    return module


@pytest.fixture(scope="module")
def cov():
    return _load("coverage_diagnostics")


@pytest.fixture(scope="module")
def deltas():
    return _load("expc_paired_deltas")


def _single_site(n_days, primary_days, aux_days, start="2018-08-01"):
    """One-column capture masks over ``n_days`` starting at ``start``."""
    time = pd.date_range(start, periods=n_days, freq="D")
    primary = np.zeros((n_days, 1), dtype=bool)
    auxiliary = np.zeros((n_days, 1), dtype=bool)
    primary[list(primary_days), 0] = True
    auxiliary[list(aux_days), 0] = True
    return time, primary, auxiliary


# --------------------------------------------------------------------------- #
# coverage_diagnostics: window conventions
# --------------------------------------------------------------------------- #


def test_window_conventions_agree_on_rescued_and_differ_on_total(cov):
    """40 days at window 16: resample keeps the 8-day stub, floor drops it.

    Capture layout puts a Landsat scene in bin 0, a JET-only scene in bin 1,
    and nothing in the trailing stub -- so the stub is an extra empty window
    that only the resample convention sees.
    """
    time, primary, auxiliary = _single_site(40, primary_days=[0], aux_days=[20])

    resample = cov.window_coverage(time, primary, auxiliary, window_days=16, convention="resample")
    floor = cov.window_coverage(time, primary, auxiliary, window_days=16, convention="floor")

    assert resample["windows_total"] == 3
    assert floor["windows_total"] == 2

    # The numerator is a property of the data, not of the binning stub.
    assert resample["windows_rescued"] == floor["windows_rescued"] == 1

    # The denominator, and therefore the published rate, is not.
    assert resample["windows_empty_primary"] == 2
    assert floor["windows_empty_primary"] == 1
    assert resample["windows_rescued_frac"] == pytest.approx(0.5)
    assert floor["windows_rescued_frac"] == pytest.approx(1.0)

    # A window the auxiliary record does not reach stays empty in both.
    assert resample["windows_empty_joint"] == 1
    assert floor["windows_empty_joint"] == 0


def test_window_coverage_overlapping_aux_rescues_nothing(cov):
    """A JET capture on a day Landsat already covers cannot fill a window."""
    time, primary, auxiliary = _single_site(32, primary_days=[0, 20], aux_days=[20])
    counts = cov.window_coverage(time, primary, auxiliary, window_days=16, convention="floor")
    assert counts["windows_empty_primary"] == 0
    assert counts["windows_rescued"] == 0


def test_window_coverage_rejects_unknown_convention(cov):
    time, primary, auxiliary = _single_site(32, primary_days=[0], aux_days=[20])
    with pytest.raises(ValueError, match="unknown window convention"):
        cov.window_coverage(time, primary, auxiliary, window_days=16, convention="rolling")


# --------------------------------------------------------------------------- #
# coverage_diagnostics: gap conventions
# --------------------------------------------------------------------------- #


def test_interior_gap_ignores_year_edges(cov):
    """Two captures 50 d apart in a 365-day year.

    ``interior`` reports the 50-day interval between them. ``edges`` reports the
    214-day tail from the last capture to Dec 31, which dominates. This is the
    whole reason the manuscript's 38.3 d and the alternative 50.8 d differ.
    """
    time, primary, auxiliary = _single_site(
        365, primary_days=[100, 150], aux_days=[], start="2019-01-01"
    )

    interior = cov.longest_gap(time, primary, auxiliary, convention="interior")
    edges = cov.longest_gap(time, primary, auxiliary, convention="edges")

    control_interior = next(r for r in interior if r["record"] == "primary_only")
    control_edges = next(r for r in edges if r["record"] == "primary_only")

    assert control_interior["mean"] == pytest.approx(50.0)
    assert control_edges["mean"] == pytest.approx(364 - 150)


def test_auxiliary_capture_shortens_the_interior_gap(cov):
    """A JET capture inside the Landsat gap halves it, as the treatment claims."""
    time, primary, auxiliary = _single_site(
        365, primary_days=[100, 150], aux_days=[125], start="2019-01-01"
    )
    rows = cov.longest_gap(time, primary, auxiliary, convention="interior")
    control = next(r for r in rows if r["record"] == "primary_only")
    treatment = next(r for r in rows if r["record"] == "primary_plus_aux")
    assert control["mean"] == pytest.approx(50.0)
    assert treatment["mean"] == pytest.approx(25.0)


def test_interior_gap_skips_site_years_with_one_capture(cov):
    """A single capture defines no interior interval; ``edges`` still scores it."""
    time, primary, auxiliary = _single_site(
        365, primary_days=[100], aux_days=[], start="2019-01-01"
    )
    interior = next(
        r
        for r in cov.longest_gap(time, primary, auxiliary, convention="interior")
        if r["record"] == "primary_only"
    )
    edges = next(
        r
        for r in cov.longest_gap(time, primary, auxiliary, convention="edges")
        if r["record"] == "primary_only"
    )
    assert interior["n_site_years"] == 0
    assert edges["n_site_years"] == 1


def test_longest_gap_rejects_unknown_convention(cov):
    time, primary, auxiliary = _single_site(
        365, primary_days=[100, 150], aux_days=[], start="2019-01-01"
    )
    with pytest.raises(ValueError, match="unknown gap convention"):
        cov.longest_gap(time, primary, auxiliary, convention="longest_run")


# --------------------------------------------------------------------------- #
# coverage_diagnostics: capture counts and the treated/control split
# --------------------------------------------------------------------------- #


def test_capture_counts_partition_activated_and_overlap(cov):
    """``aux_activated`` and ``aux_excluded_overlap`` must exhaust the captures.

    The published 10,223 of 11,868 is an "only on Landsat-free dates"
    criterion, so an auxiliary capture is either activated or excluded by
    overlap, with nothing unaccounted for.
    """
    time = pd.date_range("2018-08-01", periods=10, freq="D")
    primary = np.zeros((10, 3), dtype=bool)
    auxiliary = np.zeros((10, 3), dtype=bool)
    primary[[0, 1], 0] = True
    auxiliary[[1, 2, 3], 0] = True  # one overlap, two activated
    auxiliary[[5], 1] = True  # activated, no primary
    # column 2 has no auxiliary capture at all -> stochastic control

    counts = cov.capture_counts(primary, auxiliary)
    assert counts["aux_captures"] == 4
    assert counts["aux_activated"] == 3
    assert counts["aux_excluded_overlap"] == 1
    assert counts["aux_activated"] + counts["aux_excluded_overlap"] == counts["aux_captures"]
    assert counts["aux_activated_frac"] == pytest.approx(0.75)
    assert counts["sites_with_aux"] == 2
    assert counts["sites_without_aux"] == 1
    assert counts["sites_with_aux"] + counts["sites_without_aux"] == auxiliary.shape[1]

    table = cov.per_site_table(["a", "b", "c"], primary, auxiliary)
    assert table.set_index("fid")["group"].to_dict() == {
        "a": "treated",
        "b": "treated",
        "c": "control",
    }
    assert len(time) == 10  # time is not consulted by either helper


# --------------------------------------------------------------------------- #
# expc_paired_deltas: the estimand
# --------------------------------------------------------------------------- #


def _paired_frame(fids, ctl, trt, subset="ecostress_active"):
    """A frame in the archived paired-metric schema."""
    n = len(fids)
    return pd.DataFrame(
        {
            "fid": fids,
            "r2_ctl": ctl,
            "r2_trt": trt,
            "r2_delta": np.asarray(trt) - np.asarray(ctl),
            "kge_ctl": ctl,
            "kge_trt": trt,
            "kge_delta": np.asarray(trt) - np.asarray(ctl),
            "rmse_ctl": ctl,
            "rmse_trt": trt,
            "rmse_delta": np.asarray(trt) - np.asarray(ctl),
            "bias_ctl": ctl,
            "bias_trt": trt,
            "bias_delta": np.asarray(trt) - np.asarray(ctl),
            "n": np.full(n, 500),
            "subset": [subset] * n,
        }
    )


def test_estimand_is_the_median_of_paired_differences(deltas):
    """The median paired delta is not the difference of the two medians.

    A constant offset makes the two estimands coincide, which is why the earlier
    constant-offset test could not tell them apart. This frame is built so they
    differ: the treatment helps the low-scoring sites and hurts the high-scoring
    ones, leaving the marginal medians nearly unchanged while the paired median
    is clearly positive.
    """
    ctl = np.array([0.10, 0.20, 0.30, 0.40, 0.50])
    trt = np.array([0.60, 0.70, 0.35, 0.45, 0.55])
    frame = _paired_frame([f"S{i}" for i in range(5)], ctl, trt)

    _, d = deltas.paired_deltas(frame, "kge")
    rng = np.random.default_rng(42)
    out = deltas.bootstrap_median_delta(d, rng, reps=200)

    # Most sites gain a little; two gain a lot. The typical site gains 0.05.
    assert out["median_delta"] == pytest.approx(np.median(trt - ctl))
    assert out["median_delta"] == pytest.approx(0.05)

    # The marginal medians move by five times as much, because they are not
    # taken at the same site.
    difference_of_medians = np.median(trt) - np.median(ctl)
    assert difference_of_medians == pytest.approx(0.25)
    assert out["median_delta"] != pytest.approx(difference_of_medians, abs=1e-6), (
        "the two estimands must be distinguishable on nonconstant deltas"
    )


def test_absolute_mbe_is_rebuilt_from_the_two_arms(deltas):
    """``bias_delta`` is signed; Table S10 reports the change in |MBE|.

    A site whose bias flips from +0.30 to -0.10 improved by 0.20 in absolute
    terms while its signed delta is -0.40. Reading the recorded delta column
    would report a fourfold-larger improvement.
    """
    frame = _paired_frame(["FLIP"], np.array([0.30]), np.array([-0.10]))

    _, signed = deltas.paired_deltas(frame, "rmse")
    _, absolute = deltas.paired_deltas(frame, "absolute_mbe")

    assert signed[0] == pytest.approx(-0.40)
    assert absolute[0] == pytest.approx(-0.20)


def test_paired_deltas_filters_each_metric_independently(deltas):
    """The finite filter is per metric, so one bad metric cannot drop a site."""
    frame = _paired_frame(["A", "B"], np.array([0.4, 0.5]), np.array([0.45, 0.55]))
    frame.loc[0, "kge_delta"] = np.nan

    kge_fids, kge = deltas.paired_deltas(frame, "kge")
    nse_fids, nse = deltas.paired_deltas(frame, "nse")

    assert kge_fids == ["B"] and len(kge) == 1
    assert nse_fids == ["A", "B"] and len(nse) == 2


def test_bootstrap_median_delta_advances_the_shared_rng(deltas):
    """One stream per scale: consecutive metrics must not repeat the same draws.

    The recorded convention reinitializes the RNG once per scale and then draws
    sequentially across metrics. If a caller reseeded per metric, two metrics
    with identical deltas would return identical intervals.
    """
    d = np.sort(np.random.default_rng(0).normal(0, 1, 101))

    rng = np.random.default_rng(42)
    state_before = rng.bit_generator.state["state"]["state"]
    first = deltas.bootstrap_median_delta(d, rng, reps=2000)
    assert rng.bit_generator.state["state"]["state"] != state_before

    # A second metric drawn from the same stream sees different resamples. The
    # point estimate is unchanged (same deltas); the interval need not visibly
    # move, because a bootstrap median is always one of the site values and the
    # percentiles are correspondingly coarse -- see the archive-gated test below
    # for the case where it does matter.
    second = deltas.bootstrap_median_delta(d, rng, reps=2000)
    assert first["median_delta"] == pytest.approx(second["median_delta"])

    # Reseeding per metric would instead replay the first draw.
    restarted = deltas.bootstrap_median_delta(d, np.random.default_rng(42), reps=2000)
    assert restarted["ci_lower_95"] == pytest.approx(first["ci_lower_95"])
    assert restarted["ci_upper_95"] == pytest.approx(first["ci_upper_95"])


def test_subset_streams_do_not_perturb_the_all_sites_rows(deltas):
    """Subset rows use isolated RNGs, so computing them cannot move Table S10."""
    ctl = np.linspace(0.2, 0.8, 12)
    trt = ctl + np.linspace(-0.05, 0.05, 12)
    frame = _paired_frame([f"S{i}" for i in range(12)], ctl, trt)
    frame.loc[6:, "subset"] = "no_ecostress"
    frames = {"daily": frame}

    before = deltas.reproduce_frozen(frames, reps=200, seed=42)
    deltas.subset_rows(frames, reps=200, seed=42)
    after = deltas.reproduce_frozen(frames, reps=200, seed=42)

    pd.testing.assert_frame_equal(before, after)


# --------------------------------------------------------------------------- #
# expc_paired_deltas: the archive gate and the frozen reproduction
# --------------------------------------------------------------------------- #


def _archive(deltas):
    metadata = deltas.load_metadata()
    return metadata, deltas.resolve_archive(metadata)


def _archive_available(deltas):
    try:
        metadata, archive_dir = _archive(deltas)
    except (OSError, KeyError):
        return False
    return all(
        (archive_dir / Path(entry["path"]).name).exists()
        for entry in metadata["source_files"].values()
    )


def test_verify_sources_rejects_a_modified_archive(deltas, tmp_path):
    """The hash gate must fail closed on any drift in the paired inputs.

    A changed archive means the pairing is no longer the one behind Table S10,
    so re-bootstrapping it would produce numbers with no published counterpart.
    """
    metadata = {
        "source_files": {
            "daily_paired_site_metrics": {
                "path": "archive/6_evaluation/control_vs_treatment_daily.csv",
                "sha256": "0" * 64,
            }
        }
    }
    (tmp_path / "control_vs_treatment_daily.csv").write_text("fid,kge_delta\nA,0.1\n")
    with pytest.raises(SystemExit, match="SHA-256 mismatch"):
        deltas.verify_sources(tmp_path, metadata)


def test_verify_sources_rejects_a_missing_source(deltas, tmp_path):
    metadata = {
        "source_files": {
            "daily_paired_site_metrics": {
                "path": "archive/6_evaluation/control_vs_treatment_daily.csv",
                "sha256": "0" * 64,
            }
        }
    }
    with pytest.raises(SystemExit, match="missing archived source"):
        deltas.verify_sources(tmp_path, metadata)


def test_frozen_metadata_and_summary_are_consistent(deltas):
    """The published summary must match the conventions its metadata records."""
    metadata = deltas.load_metadata()
    frozen = pd.read_csv(deltas.FROZEN_SUMMARY)

    assert tuple(metadata["bootstrap"]["metric_order"]) == deltas.METRIC_ORDER
    assert set(frozen["metric"]) == set(deltas.METRIC_ORDER)
    assert (frozen["n_bootstrap"] == metadata["bootstrap"]["replicates"]).all()
    assert (frozen["seed"] == metadata["bootstrap"]["seed"]).all()

    # Every published interval spans zero -- the claim the supplement makes.
    assert ((frozen["ci_lower_95"] <= 0) & (frozen["ci_upper_95"] >= 0)).all()


def test_reproduces_the_frozen_table_s10_row_for_row(deltas):
    """End-to-end: hash-gated archive in, published Table S10 out.

    This is the assertion the module exists to support. It runs only where the
    archived run is mounted; the estimand and convention tests above are
    machine-independent.
    """
    if not _archive_available(deltas):
        pytest.skip("archived E3 Exp C run not mounted on this machine")

    metadata, archive_dir = _archive(deltas)
    sources = deltas.verify_sources(archive_dir, metadata)
    frames = {
        scale: pd.read_csv(sources[key])
        for scale, key in deltas.SCALE_SOURCE.items()
        if key in sources
    }
    reproduced = deltas.reproduce_frozen(
        frames, metadata["bootstrap"]["replicates"], metadata["bootstrap"]["seed"]
    )
    merged = deltas.compare_to_frozen(reproduced)

    assert len(merged) == 8
    assert merged["matches"].all(), merged.loc[~merged["matches"], ["scale", "metric"]].to_dict()
    assert set(merged.loc[merged["scale"] == "daily", "n_sites"]) == {63}
    assert set(merged.loc[merged["scale"] == "monthly", "n_sites"]) == {50}


def test_reproduction_detects_a_perturbed_frozen_summary(deltas, tmp_path):
    """The comparison must fail when the reproduction and the artifact disagree."""
    if not _archive_available(deltas):
        pytest.skip("archived E3 Exp C run not mounted on this machine")

    metadata, archive_dir = _archive(deltas)
    sources = deltas.verify_sources(archive_dir, metadata)
    frames = {
        scale: pd.read_csv(sources[key])
        for scale, key in deltas.SCALE_SOURCE.items()
        if key in sources
    }
    reproduced = deltas.reproduce_frozen(
        frames, metadata["bootstrap"]["replicates"], metadata["bootstrap"]["seed"]
    )

    tampered = pd.read_csv(deltas.FROZEN_SUMMARY)
    tampered.loc[0, "median_delta"] += 1e-6
    path = tmp_path / "tampered.csv"
    tampered.to_csv(path, index=False)

    merged = deltas.compare_to_frozen(reproduced, path)
    assert not merged["matches"].all()


def test_per_metric_reseeding_does_not_reproduce_the_frozen_table(deltas):
    """The recorded RNG scope is load-bearing, and quietly so.

    Reseeding once per metric instead of once per scale still reproduces 5 of
    the 8 published rows exactly -- so a spot check would pass -- while shifting
    three monthly bounds. Only the full row-for-row comparison catches it.
    """
    if not _archive_available(deltas):
        pytest.skip("archived E3 Exp C run not mounted on this machine")

    metadata, archive_dir = _archive(deltas)
    sources = deltas.verify_sources(archive_dir, metadata)
    frames = {
        scale: pd.read_csv(sources[key])
        for scale, key in deltas.SCALE_SOURCE.items()
        if key in sources
    }
    reps = metadata["bootstrap"]["replicates"]
    seed = metadata["bootstrap"]["seed"]

    rows = []
    for scale, frame in frames.items():
        for metric in deltas.METRIC_ORDER:
            _, d = deltas.paired_deltas(frame, metric)
            rows.append(
                {
                    "scale": scale,
                    "metric": metric,
                    **deltas.bootstrap_median_delta(d, np.random.default_rng(seed), reps),
                }
            )
    reseeded = deltas.compare_to_frozen(pd.DataFrame(rows))

    assert not reseeded["matches"].all(), "per-metric reseeding must not reproduce Table S10"
    assert reseeded["matches"].sum() == 5
    assert set(reseeded.loc[~reseeded["matches"], "scale"]) == {"monthly"}
