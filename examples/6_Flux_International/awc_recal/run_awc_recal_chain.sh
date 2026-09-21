#!/usr/bin/env bash
# HWSD AWC units recalibration (examples/6_Flux_International/notes/HANDOFF_HWSD_AWC_UNITS_RECAL.md).
# One unattended chain, run AFTER the in-session steps (container refresh, Gate G1 PASS, arm
# containers re-copied):
#   A. canonical GrassBasis E2 calibration exactly as September (batch_runner calibrate-all,
#      200 reals, noptmax 3, 20 workers, batch 50, --exclude-uncovered), with the RUN_POLICY
#      Cats 1-3 capture, the objective audit against the superseded pest_archive, and Gate G2
#      (aw_* priors follow the HWSD rule, not a constant) as a HARD STOP before launch;
#   B. Cats 4-5 archive, Gate G3 (informational: posterior aw ceiling fraction / Spearman vs HWSD);
#   C. canonical Cat 6 evaluation (daily, monthly, ETf), pooled + derived metrics with the
#      uncalibrated baseline, the by-irrigation Run 22 transfer, the phase-11 evaluation summary,
#      the closure-pool summary, and the S9.1 28-day monthly sensitivity;
#   D. both E0 formulation arms + pooled disjoint gates (e0_disjoint/run_e0_arms.sh, unchanged).
# Steps in C that only report problems (evaluation summary, closure pool) do not abort D; the
# chain exits nonzero at the end if any of them failed.
set -euo pipefail
REPO=/home/dgketchum/code/swim-rs
EX6=$REPO/examples/6_Flux_International
E2=/data/ssd1/swim/6_Flux_International
RESULTS=$E2/results
SUP=$RESULTS/superseded_awc320_20260921
QA_CANON=$E2/data/e2_etf_refooting
QA=$E2/data/awc_recal/qa_canon
RUN=6_Flux_International_LSEnsemble_GrassBasis_POR_annual2yr
CFG=$EX6/$RUN.toml
GB=$E2/data/6_Flux_International_ls_ensemble_grassbasis_por_annual2yr.swim
PESTRUN=$E2/pestrun_ls_ensemble_grassbasis_por_annual2yr
ARCHIVE=$RESULTS/$RUN/archive
LOG=$E2/nohup_calibrate_ls_ensemble_grassbasis_por_annual2yr_awcrecal.out
TRANSFER_OUT=$RESULTS/e2_run22_transfer_by_irrigation_to_grassbasis
PY="uv --directory $REPO run python -u"
LAUNCH="uv --directory $REPO run python -u -m swimrs.calibrate.batch_runner --config $CFG --action calibrate-all --reals 200 --noptmax 3 --workers 20 --batch-size 50 --exclude-uncovered"

step() { echo; echo "================ $1  $(date -Is) ================"; }

step "0 preconditions"
$PY - <<PYEOF
import json, sys
g = json.load(open("$E2/data/awc_recal/gate_g1_container.json"))
print("gate G1:", "PASS" if g.get("pass") else g.get("problems"))
sys.exit(0 if g.get("pass") else 1)
PYEOF
for d in "$RESULTS/$RUN" "$PESTRUN" "$TRANSFER_OUT" "$RESULTS/e0_disjoint"; do
  if [ -e "$d" ]; then echo "refusing: $d exists (superseded runs must be moved first)"; exit 1; fi
done
# Cat 2 sidecar QA files from the September canon (same inputs; transition CSV verified identical by G1)
for f in daily_basis_gate_summary.json daily_basis_gate_by_site.csv ssebop_conversion_summary.json \
         ssebop_native_consolidation_summary.json le07_delivery_summary.json; do
  cp "$QA_CANON/$f" "$QA/"
done
ls "$QA"

step "A1 prep + build-all (canonical)"
$PY -m swimrs.calibrate.batch_runner --config "$CFG" --action prep \
    --reals 200 --noptmax 3 --workers 20 --batch-size 50 --exclude-uncovered
$PY -m swimrs.calibrate.batch_runner --config "$CFG" --action build-all \
    --reals 200 --noptmax 3 --workers 20 --batch-size 50 --exclude-uncovered

step "A2 objective audit vs superseded pest_archive (ETf weights must reproduce; only aw priors change)"
$PY $EX6/e2_refooting/phase9_objective_audit.py --config "$CFG" \
    --baseline-pest-archive $SUP/pestrun/pestrun_ls_ensemble_grassbasis_por_annual2yr/pest_archive \
    --out-dir "$QA" --noptmax 3 --reals 200

step "A3 Cats 1-3 capture"
$PY $EX6/e2_refooting/phase9_archive_prelaunch.py capture --config "$CFG" --run-name "$RUN" \
    --qa-root "$QA" --command "$LAUNCH" --workers 20 --reals 200 --noptmax 3 --batch-size 50

step "A4 Gate G2: aw_* priors follow the HWSD rule (HARD STOP)"
$PY $EX6/awc_recal/gate_g2_priors.py --run-name "$RUN"

step "A5 hash verify + LAUNCH canonical calibration"
$PY $EX6/e2_refooting/phase9_archive_prelaunch.py verify --config "$CFG" --run-name "$RUN" --qa-root "$QA"
echo "$LAUNCH"
$LAUNCH > "$LOG" 2>&1
echo "calibration finished $(date -Is); log $LOG"

step "B1 Cats 4-5 post-calibration archive"
$PY $EX6/e2_refooting/phase11_archive_postcalibration.py --config "$CFG" --run-name "$RUN" --log "$LOG"
POST=$ARCHIVE/4_pest_outputs/merged/merged_posterior.csv
test -f "$POST"

step "B2 Gate G3 (informational): posterior aw vs the superseded run"
set +e
$PY $EX6/awc_recal/gate_g3_posterior.py --run-name "$RUN"
echo "gate G3 exit $? (informational)"
set -e

step "C1 Cat 6 evaluation: daily, monthly, ETf"
$PY $EX6/evaluate.py --config "$CFG" --par-csv "$POST"
$PY $EX6/evaluate.py --config "$CFG" --par-csv "$POST" --monthly
$PY $EX6/evaluate.py --config "$CFG" --par-csv "$POST" --etf

step "C2 pooled metrics + derived metrics with the uncalibrated (HWSD m/m) baseline"
$PY $EX6/pooled_metrics.py --results-dir "$RESULTS/$RUN"
$PY $EX6/derived_metrics.py --config "$CFG" --uncalibrated --out "$RESULTS/$RUN/derived"

step "C3 irrigation-stratified Run 22 transfer into E2"
$PY $EX6/transfer/build_e3_irrigation_mapping.py --container "$GB" \
    --out-dir "$RESULTS/$RUN/transfer_refresh" --allow-unexpected
$PY $EX6/transfer_ex5_params.py --config "$CFG" \
    --params $REPO/paper/data/final/e2_run22_transfer_vector.json \
    --params-by-site "$RESULTS/$RUN/transfer_refresh/e3_irrigation_stratified_param_mapping.json" \
    --container "$GB" --e3-results-dir "$RESULTS/$RUN" --out "$TRANSFER_OUT" --require-empty-out

FAILED=""
step "C4 phase-11 evaluation summary (baseline forward skipped: the por_annual2yr container still stores mm/m)"
set +e
$PY $EX6/e2_refooting/phase11_evaluation_summary.py --config "$CFG" --run-name "$RUN" --skip-baseline-forward
rc=$?; [ $rc -ne 0 ] && FAILED="$FAILED evaluation_summary(rc=$rc)"
step "C5 closure-pool summary (47 EBR sites; Table 5 / S8 / section 3.3)"
$PY $EX6/e2_refooting/phase11_closure_pool_summary.py --config "$CFG" --run-name "$RUN"
rc=$?; [ $rc -ne 0 ] && FAILED="$FAILED closure_pool(rc=$rc)"
step "C6 S9.1 28-day monthly sensitivity on the closure pool"
$PY $EX6/awc_recal/monthly_28day_sensitivity.py --run-name "$RUN"
rc=$?; [ $rc -ne 0 ] && FAILED="$FAILED monthly_28day(rc=$rc)"
set -e

step "D E0 formulation arms + pooled disjoint gates"
bash $EX6/e0_disjoint/run_e0_arms.sh

step "chain finished"
if [ -n "$FAILED" ]; then echo "STEPS WITH PROBLEMS:$FAILED"; exit 1; fi
echo "all steps completed"
