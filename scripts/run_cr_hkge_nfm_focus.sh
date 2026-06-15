#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export DATASET="${DATASET:-dataset-aromatique-crhkge-ready}"
export LOG_DIR="${LOG_DIR:-$ROOT_DIR/final_study/logs_crhkge_nfm_focus}"
export CR_HKGE_FINAL_EPOCHS="${CR_HKGE_FINAL_EPOCHS:-100}"

mkdir -p "$LOG_DIR"

echo "==> CR-HKGE NFM-focus tuning"
echo "dataset=$DATASET"
echo "log_dir=$LOG_DIR"
echo "epochs=$CR_HKGE_FINAL_EPOCHS"

targets=(
  cr_hkge_final_alpha_0_075
  cr_hkge_final_alpha_0_05
  cr_hkge_final_alpha_0_1_prior_0_5
  cr_hkge_final_alpha_0_1_prior_0_25
  cr_hkge_final_alpha_0_1_no_relation_message
  cr_hkge_final_alpha_0_075_prior_0_5
)

bash "$ROOT_DIR/scripts/run_cr_hkge_final_study.sh" "${targets[@]}"

echo "==> Current summary"
python "$ROOT_DIR/scripts/summarize_final_study.py" "$LOG_DIR"/*_subset_eval.log | tee "$LOG_DIR/final_summary.md"

if ls "$LOG_DIR"/*_subset_eval.log >/dev/null 2>&1; then
  echo "==> NFM-focus delta summary"
  python "$ROOT_DIR/scripts/summarize_nfm_focus.py" "$LOG_DIR"/*_subset_eval.log --reference nfm | tee "$LOG_DIR/nfm_focus_summary.md"
fi
