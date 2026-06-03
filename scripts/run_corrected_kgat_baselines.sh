#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$ROOT_DIR/Model"
LOG_DIR="$ROOT_DIR/corrected_baselines/logs"

mkdir -p "$LOG_DIR"

python "$ROOT_DIR/scripts/build_aromatique_dataset_variants.py"

COMMON_ARGS=(
  --model_type kgat
  --data_path ../
  --alg_type bi
  --adj_type si
  --adj_uni_type sum
  --regs '[1e-5,1e-5]'
  --layer_size '[64,32,16]'
  --embed_size 64
  --kge_size 64
  --lr 0.0001
  --batch_size 64
  --mess_dropout '[0.1,0.1,0.1]'
  --node_dropout '[0.1]'
  --Ks '[3,5,10]'
  --epoch 100
  --save_flag 1
)

run_dataset() {
  local dataset="$1"
  local weights_path="../corrected_baselines/${dataset}/"
  local train_log="$LOG_DIR/${dataset}_train.log"
  local eval_log="$LOG_DIR/${dataset}_subset_eval.log"

  echo "==> Training KGAT corrected baseline: $dataset"
  (
    cd "$MODEL_DIR"
    python Main.py \
      --weights_path "$weights_path" \
      --dataset "$dataset" \
      "${COMMON_ARGS[@]}"
  ) 2>&1 | tee "$train_log"

  echo "==> Evaluating KGAT corrected baseline: $dataset"
  (
    cd "$MODEL_DIR"
    python evaluate_item_subsets.py \
      --weights_path "$weights_path" \
      --dataset "$dataset" \
      "${COMMON_ARGS[@]}" \
      --cr_subset_data_path ../ \
      --cr_subset_dataset dataset-aromatique-kgat-ready
  ) 2>&1 | tee "$eval_log"

  echo "==> Finished $dataset"
  echo "train_log=$train_log"
  echo "eval_log=$eval_log"
}

if [ "$#" -eq 0 ]; then
  datasets=(
    dataset-aromatique-local-kgat-ready
    dataset-aromatique-attribute-kgat-ready
  )
else
  datasets=("$@")
fi

for dataset in "${datasets[@]}"; do
  run_dataset "$dataset"
done
