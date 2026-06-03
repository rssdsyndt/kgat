#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$ROOT_DIR/Model"
LOG_DIR="$ROOT_DIR/ablation/logs"

mkdir -p "$LOG_DIR"

COMMON_ARGS=(
  --model_type cr_hkge
  --data_path ../
  --dataset dataset-aromatique-kgat-ready
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
  --cr_best_metric ndcg
  --cr_best_k 3
)

variant_args() {
  case "$1" in
    A1_no_crossref)
      echo "--cr_use_relation_weight 1 --cr_use_cross_ref 0 --cr_relation_weight_mode semantic --cr_relation_aware_message 1 --cr_relation_message_scale type_count"
      ;;
    A2_no_relation_message)
      echo "--cr_use_relation_weight 1 --cr_use_cross_ref 1 --cr_relation_weight_mode semantic --cr_relation_aware_message 0 --cr_relation_message_scale type_count"
      ;;
    A3_probability_scale)
      echo "--cr_use_relation_weight 1 --cr_use_cross_ref 1 --cr_relation_weight_mode semantic --cr_relation_aware_message 1 --cr_relation_message_scale probability"
      ;;
    A4_no_novelty_modules)
      echo "--cr_use_relation_weight 0 --cr_use_cross_ref 0 --cr_relation_weight_mode semantic --cr_relation_aware_message 0 --cr_relation_message_scale type_count"
      ;;
    FINAL_strict_gated)
      echo "--cr_use_relation_weight 1 --cr_use_cross_ref 1 --cr_relation_weight_mode semantic --cr_relation_aware_message 0 --cr_relation_message_scale type_count --cr_cross_ref_bi_interaction 0 --cr_cross_ref_gate 1 --cr_cross_ref_gate_init -2.0"
      ;;
    FINAL_strict_additive)
      echo "--cr_use_relation_weight 1 --cr_use_cross_ref 1 --cr_relation_weight_mode semantic --cr_relation_aware_message 0 --cr_relation_message_scale type_count --cr_cross_ref_bi_interaction 0 --cr_cross_ref_gate 0 --cr_cross_ref_alpha 1.0"
      ;;
    FINAL_alpha_0_5)
      echo "--cr_use_relation_weight 1 --cr_use_cross_ref 1 --cr_relation_weight_mode semantic --cr_relation_aware_message 0 --cr_relation_message_scale type_count --cr_cross_ref_bi_interaction 0 --cr_cross_ref_gate 0 --cr_cross_ref_alpha 0.5"
      ;;
    FINAL_alpha_1_5)
      echo "--cr_use_relation_weight 1 --cr_use_cross_ref 1 --cr_relation_weight_mode semantic --cr_relation_aware_message 0 --cr_relation_message_scale type_count --cr_cross_ref_bi_interaction 0 --cr_cross_ref_gate 0 --cr_cross_ref_alpha 1.5"
      ;;
    FINAL_prior_message_alpha_0_5)
      echo "--cr_use_relation_weight 1 --cr_use_cross_ref 1 --cr_relation_weight_mode semantic --cr_relation_prior_mode fragrance --cr_relation_prior_strength 1.0 --cr_relation_attention_scale type_count --cr_relation_aware_message 1 --cr_relation_message_scale type_count --cr_cross_ref_bi_interaction 0 --cr_cross_ref_gate 0 --cr_cross_ref_alpha 0.5"
      ;;
    FINAL_prior_message_gated)
      echo "--cr_use_relation_weight 1 --cr_use_cross_ref 1 --cr_relation_weight_mode semantic --cr_relation_prior_mode fragrance --cr_relation_prior_strength 1.0 --cr_relation_attention_scale type_count --cr_relation_aware_message 1 --cr_relation_message_scale type_count --cr_cross_ref_bi_interaction 0 --cr_cross_ref_gate 1 --cr_cross_ref_gate_init -2.0"
      ;;
    FINAL_prior_attention_alpha_0_5)
      echo "--cr_use_relation_weight 1 --cr_use_cross_ref 1 --cr_relation_weight_mode semantic --cr_relation_prior_mode fragrance --cr_relation_prior_strength 1.0 --cr_relation_attention_scale type_count --cr_relation_aware_message 0 --cr_relation_message_scale type_count --cr_cross_ref_bi_interaction 0 --cr_cross_ref_gate 0 --cr_cross_ref_alpha 0.5"
      ;;
    *)
      echo "unknown variant: $1" >&2
      echo "available: A1_no_crossref A2_no_relation_message A3_probability_scale A4_no_novelty_modules FINAL_strict_gated FINAL_strict_additive FINAL_alpha_0_5 FINAL_alpha_1_5 FINAL_prior_message_alpha_0_5 FINAL_prior_message_gated FINAL_prior_attention_alpha_0_5" >&2
      return 1
      ;;
  esac
}

run_variant() {
  local variant="$1"
  local weights_path="../ablation/${variant}/"
  local train_log="$LOG_DIR/${variant}_train.log"
  local eval_log="$LOG_DIR/${variant}_subset_eval.log"

  read -r -a extra_args <<< "$(variant_args "$variant")"

  echo "==> Training $variant"
  (
    cd "$MODEL_DIR"
    python Main.py \
      --weights_path "$weights_path" \
      "${COMMON_ARGS[@]}" \
      "${extra_args[@]}"
  ) 2>&1 | tee "$train_log"

  echo "==> Evaluating $variant"
  (
    cd "$MODEL_DIR"
    python evaluate_item_subsets.py \
      --weights_path "$weights_path" \
      "${COMMON_ARGS[@]}" \
      "${extra_args[@]}"
  ) 2>&1 | tee "$eval_log"

  echo "==> Finished $variant"
  echo "train_log=$train_log"
  echo "eval_log=$eval_log"
}

if [ "$#" -eq 0 ]; then
  variants=(A2_no_relation_message A3_probability_scale A4_no_novelty_modules)
else
  variants=("$@")
fi

for variant in "${variants[@]}"; do
  run_variant "$variant"
done
