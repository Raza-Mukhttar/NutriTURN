#!/usr/bin/env bash
# Regenerate every table and figure in the paper from the shipped data.
#
#   PYTHONHASHSEED=0 bash scripts/run_all.sh
#
# PYTHONHASHSEED=0 is required for the null controls to reproduce (see ENVIRONMENT.md).
# Annotation (scripts/pipeline) is NOT run here: it needs GPUs and the abstracts.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PY:-python3}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
mkdir -p results figures tables logs
FAIL=0

run() {                      # run <label> <script> [args...]
  local label="$1"; shift
  printf '%-46s ' "$label"
  if "$PY" "$@" > "logs/$label.log" 2>&1; then
    echo "ok"
  else
    echo "FAILED (see logs/$label.log)"; FAIL=$((FAIL+1))
  fi
}

E=scripts/experiments
F=scripts/figures

echo "== 0. rebuild checks =="
run verify_release              scripts/verify_release.py
run endpoint_contract           $E/endpoint_contract.py
run smoke_test                  $E/smoke_test.py

echo "== 1. core scores (later stages depend on these) =="
run stationary_baselines        $E/stationary_baselines.py
run temporal_offset             $E/temporal_offset.py
run sign_invariant_scores       $E/sign_invariant_scores.py
run origin_calendar_cohort      $E/origin_calendar_cohort.py
run full_metric_suite           $E/full_metric_suite.py
run validation_addendum         $E/validation_addendum.py
run extractor_agreement         $E/extractor_agreement.py
run dataset_card                $E/dataset_card.py

echo "== 2. audit analyses =="
run agreement_definition_audit  $E/agreement_definition_audit.py
run cumulative_endpoints        $E/cumulative_endpoints.py
run orientation_audit           $E/orientation_audit.py
run two_variable_transfer       $E/two_variable_transfer.py
run calendar_forward_evaluation $E/calendar_forward_evaluation.py
run effect_typing               $E/effect_typing.py
run recalibrated_order_features $E/recalibrated_order_features.py
run consistency_checks          $E/consistency_checks.py
run snapshot_provenance         $E/snapshot_provenance.py
run origprompt_cell_analysis    $E/origprompt_cell_analysis.py
run controlled_annotation_design $E/controlled_annotation_design.py

echo "== 3. null controls and inference (slowest; ~1h total) =="
run stationary_null_permutation $E/stationary_null_permutation.py
run stationary_null_simulation  $E/stationary_null_simulation.py
run stationary_null_summary     $E/stationary_null_summary.py
run null_controls               $E/null_controls.py
run null_c_invariance           $E/null_c_invariance.py
run null_d_optimizer_audit      $E/null_d_optimizer_audit.py
run wild_cluster_bootstrap      $E/wild_cluster_bootstrap.py
run cluster_aware_inference     $E/cluster_aware_inference.py
run clustered_roc_all_cells     $E/clustered_roc_all_cells.py --refit
run calendar_refit_bootstrap    $E/calendar_refit_bootstrap.py
run resampling_power            $E/resampling_power.py

echo "== 4. reanalyses =="
for c in rebuild coef common stratum decisive union; do
  run "core_reanalyses_$c"      $E/core_reanalyses.py "$c"
done
run agreement_definition_variants $E/agreement_definition_variants.py
run auroc_pair_decomposition    $E/auroc_pair_decomposition.py
run anchor_paired_strata        $E/anchor_paired_strata.py
run anchor_slope_differences    $E/anchor_slope_differences.py
run numeric_anchor_by_era       $E/numeric_anchor_by_era.py
run intersection_anchor_and_confusion $E/intersection_anchor_and_confusion.py
run agreement_branch_and_class_prior  $E/agreement_branch_and_class_prior.py
run accrual_and_censoring       $E/accrual_and_censoring.py
run documented_reversals        $E/documented_reversals.py
run incomplete_origin_windows   $E/incomplete_origin_windows.py
run retrieval_cap_sensitivity   $E/retrieval_cap_sensitivity.py
run retrieval_audit_reconciliation $E/retrieval_audit_reconciliation.py
run within_claim_label_drift    $E/within_claim_label_drift.py
run annotation_drift_by_decade  $E/annotation_drift_by_decade.py drift

echo "== 5. figures and tables =="
run regenerate_main_figures     $F/regenerate_main_figures.py
run build_latex_tables_main     scripts/build_latex_tables_main.py
run build_latex_tables_audit    scripts/build_latex_tables_audit.py
run build_results_report        scripts/build_results_report.py
run collect_paper_numbers       scripts/collect_paper_numbers.py
run build_file_manifest         scripts/build_file_manifest.py

echo
echo "Figure 1 (figures/nutrifig.pdf) is TikZ: compile figures/src/nutrifig.tex"
echo "with pdflatex, xelatex or tectonic."
echo
if [ "$FAIL" -eq 0 ]; then echo "run_all: all stages ok"; else echo "run_all: $FAIL stage(s) failed"; fi
exit "$FAIL"
