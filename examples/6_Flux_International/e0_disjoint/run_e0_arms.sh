#!/usr/bin/env bash
# E0 disjoint confirmation on the E2 footing: calibrate the two unscaled formulation arms
# (fao56_sig = unscaled sigmoid, fao56 = unscaled linear) with exactly the canonical
# GrassBasis method (batch_runner calibrate-all, 200 reals, noptmax 3, 20 workers,
# batch 50, --exclude-uncovered), archive per RUN_POLICY, evaluate, then run the pooled
# arm-vs-arm gates on the 37 pool sites outside the E1 calibration cohort (and on the
# full 47-site closure pool as a check). Arms run SEQUENTIALLY: run_pest.py pins the
# PEST++ master to port 5005, so two batch runners cannot coexist on this host.
# ~55 min calibration per arm (canonical: 40.5 min batch 000 + 13 min batch 001).
set -euo pipefail
REPO=/home/dgketchum/code/swim-rs
EX6=$REPO/examples/6_Flux_International
E2=/data/ssd1/swim/6_Flux_International
RESULTS=$E2/results
QA_CANON=$E2/data/e2_etf_refooting
BASE=6_Flux_International_LSEnsemble_GrassBasis_POR_annual2yr
CANON_ARCHIVE=$RESULTS/$BASE/archive
PY="uv --directory $REPO run python -u"

for ARM in ${ARMS:-fao56_sig fao56}; do  # ARMS="fao56" resumes a single arm; the pooled gates need both posteriors
  RUN=${BASE}_${ARM}
  CFG=$EX6/${RUN}.toml
  QA=$E2/data/e0_disjoint/qa_${ARM}
  LOG=$E2/nohup_calibrate_ls_ensemble_grassbasis_por_annual2yr_${ARM}.out
  LAUNCH="uv --directory $REPO run python -u -m swimrs.calibrate.batch_runner --config $CFG --action calibrate-all --reals 200 --noptmax 3 --workers 20 --batch-size 50 --resume --exclude-uncovered"
  mkdir -p "$QA"
  echo "================ ARM $ARM  $(date -Is) ================"

  # Cat 2 input audit on the arm container (hash-identical copy of the canonical inputs)
  $PY $EX6/container_build/e2_refooting/phase8_container_health.py --config "$CFG" --out-dir "$QA"
  for f in irrigation_classifier_transition.csv daily_basis_gate_summary.json daily_basis_gate_by_site.csv \
           ssebop_conversion_summary.json ssebop_native_consolidation_summary.json le07_delivery_summary.json; do
    [ -f "$QA_CANON/$f" ] && cp "$QA_CANON/$f" "$QA/"
  done

  # prep (preflight health gate + batch_manifest.csv + run_manifest.json), build the PEST++
  # problems (no run), then the independent objective audit against the canonical GrassBasis
  # pest_archive: every ETf weight must reproduce exactly (same target, same weighting; only
  # the physics/parameterization differs). build-all alone writes no manifest.
  $PY -m swimrs.calibrate.batch_runner --config "$CFG" --action prep \
      --reals 200 --noptmax 3 --workers 20 --batch-size 50 --exclude-uncovered
  $PY -m swimrs.calibrate.batch_runner --config "$CFG" --action build-all \
      --reals 200 --noptmax 3 --workers 20 --batch-size 50 --exclude-uncovered
  $PY $EX6/e2_refooting/phase9_objective_audit.py --config "$CFG" \
      --baseline-pest-archive $E2/pestrun_ls_ensemble_grassbasis_por_annual2yr/pest_archive \
      --out-dir "$QA" --noptmax 3 --reals 200

  # Cats 1-3 pre-launch, hash verify at launch, then calibrate (build reused via --resume)
  $PY $EX6/e2_refooting/phase9_archive_prelaunch.py capture --config "$CFG" --run-name "$RUN" \
      --qa-root "$QA" --command "$LAUNCH" --workers 20 --reals 200 --noptmax 3 --batch-size 50
  $PY $EX6/e2_refooting/phase9_archive_prelaunch.py verify --config "$CFG" --run-name "$RUN" --qa-root "$QA"
  $LAUNCH > "$LOG" 2>&1

  # Cats 4-5 + completion checks, then Cat 6 evaluation from the merged posterior
  $PY $EX6/e2_refooting/phase11_archive_postcalibration.py --config "$CFG" --run-name "$RUN" --log "$LOG"
  POST=$RESULTS/$RUN/archive/4_pest_outputs/merged/merged_posterior.csv
  $PY $EX6/evaluate.py --config "$CFG" --par-csv "$POST"
  $PY $EX6/evaluate.py --config "$CFG" --par-csv "$POST" --monthly
done

# Pooled gates. pooled_arm_compare.py exits 1 on GATE FAIL, so run outside set -e.
set +e
A_POST=$CANON_ARCHIVE/4_pest_outputs/merged/merged_posterior.csv
SIG_POST=$RESULTS/${BASE}_fao56_sig/archive/4_pest_outputs/merged/merged_posterior.csv
LIN_POST=$RESULTS/${BASE}_fao56/archive/4_pest_outputs/merged/merged_posterior.csv
for SET in disjoint37:disjoint_sites.txt pool47:closure_pool_sites.txt; do
  TAG=${SET%%:*}; SITES=$EX6/e0_disjoint/${SET##*:}
  OUT=$RESULTS/e0_disjoint
  $PY $EX6/pooled_arm_compare.py --a-name grassbasis --a-config $EX6/$BASE.toml --a-par "$A_POST" \
      --b-name fao56_sig --b-config $EX6/${BASE}_fao56_sig.toml --b-par "$SIG_POST" \
      --sites-file "$SITES" --out-dir "$OUT/grassbasis_vs_fao56_sig/$TAG"
  $PY $EX6/pooled_arm_compare.py --a-name grassbasis --a-config $EX6/$BASE.toml --a-par "$A_POST" \
      --b-name fao56 --b-config $EX6/${BASE}_fao56.toml --b-par "$LIN_POST" \
      --sites-file "$SITES" --out-dir "$OUT/grassbasis_vs_fao56/$TAG"
  $PY $EX6/pooled_arm_compare.py --a-name fao56_sig --a-config $EX6/${BASE}_fao56_sig.toml --a-par "$SIG_POST" \
      --b-name fao56 --b-config $EX6/${BASE}_fao56.toml --b-par "$LIN_POST" \
      --sites-file "$SITES" --out-dir "$OUT/fao56_sig_vs_fao56/$TAG"
done
echo "================ E0 disjoint chain finished $(date -Is) ================"
