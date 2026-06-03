#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BASELINE_DATASET="${1:-dataset-aromatique-attribute-kgat-ready}"
CR_VARIANT="${2:-FINAL_prior_message_alpha_0_5}"

KGAT_LOG="$ROOT_DIR/corrected_baselines/logs/${BASELINE_DATASET}_subset_eval.log"
CR_LOG="$ROOT_DIR/ablation/logs/${CR_VARIANT}_subset_eval.log"

echo "==> Controlled paper comparison"
echo "KGAT baseline dataset: $BASELINE_DATASET"
echo "CR-HKGE variant: $CR_VARIANT"

bash "$ROOT_DIR/scripts/run_corrected_kgat_baselines.sh" "$BASELINE_DATASET"
bash "$ROOT_DIR/scripts/run_cr_hkge_ablation.sh" "$CR_VARIANT"

python "$ROOT_DIR/scripts/summarize_cr_hkge_comparison.py" \
  --kgat-log "$KGAT_LOG" \
  --cr-log "$CR_LOG" \
  --kgat-name "KGAT ${BASELINE_DATASET}" \
  --cr-name "CR-HKGE ${CR_VARIANT}"
