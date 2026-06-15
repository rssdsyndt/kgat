#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$ROOT_DIR/Model"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/final_study/logs}"
DATASET="${DATASET:-dataset-aromatique-kgat-ready}"

mkdir -p "$LOG_DIR"

COMMON_ARGS=(
  --data_path ../
  --dataset "$DATASET"
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
  --epoch "${CR_HKGE_FINAL_EPOCHS:-100}"
  --save_flag 1
)

model_args() {
  case "$1" in
    bprmf)
      echo "--model_type bprmf"
      ;;
    cke)
      echo "--model_type cke"
      ;;
    nfm)
      echo "--model_type nfm"
      ;;
    cfkg)
      echo "--model_type cfkg"
      ;;
    kgat)
      echo "--model_type kgat"
      ;;
    cr_hkge_final)
      # Model utama paper:
      # semua komponen CR-HKGE aktif, yaitu relation-type attention,
      # fragrance prior, relation-aware message, dan cross-reference.
      echo "--model_type cr_hkge --cr_use_relation_weight 1 --cr_use_cross_ref 1 --cr_relation_weight_mode semantic --cr_relation_prior_mode fragrance --cr_relation_prior_strength 1.0 --cr_relation_attention_scale type_count --cr_relation_aware_message 1 --cr_relation_message_scale type_count --cr_cross_ref_bi_interaction 0 --cr_cross_ref_gate 0 --cr_cross_ref_alpha 0.5 --cr_best_metric ndcg --cr_best_k 3 --cr_export_embeddings 1 --cr_model_version cr_hkge_final_flow"
      ;;
    cr_hkge_final_alpha_0_25)
      # Kandidat final: semua novelty tetap aktif, tetapi kekuatan
      # cross-reference diturunkan. Dipakai ketika inspired_by membantu recall
      # awal namun terlalu kuat menggeser ranking dari sinyal accord/family.
      echo "--model_type cr_hkge --cr_use_relation_weight 1 --cr_use_cross_ref 1 --cr_relation_weight_mode semantic --cr_relation_prior_mode fragrance --cr_relation_prior_strength 1.0 --cr_relation_attention_scale type_count --cr_relation_aware_message 1 --cr_relation_message_scale type_count --cr_cross_ref_bi_interaction 0 --cr_cross_ref_gate 0 --cr_cross_ref_alpha 0.25 --cr_best_metric ndcg --cr_best_k 3 --cr_export_embeddings 1 --cr_model_version cr_hkge_final_alpha_0_25"
      ;;
    cr_hkge_final_alpha_0_1)
      # Kandidat final: semua novelty tetap aktif dengan cross-reference sangat
      # konservatif. Tujuannya menjaga manfaat inspired_by tanpa mengalahkan
      # neighborhood KG utama.
      echo "--model_type cr_hkge --cr_use_relation_weight 1 --cr_use_cross_ref 1 --cr_relation_weight_mode semantic --cr_relation_prior_mode fragrance --cr_relation_prior_strength 1.0 --cr_relation_attention_scale type_count --cr_relation_aware_message 1 --cr_relation_message_scale type_count --cr_cross_ref_bi_interaction 0 --cr_cross_ref_gate 0 --cr_cross_ref_alpha 0.1 --cr_best_metric ndcg --cr_best_k 3 --cr_export_embeddings 1 --cr_model_version cr_hkge_final_alpha_0_1"
      ;;
    cr_hkge_final_alpha_0_075)
      # NFM-focus tuning: cross-reference lebih kecil dari alpha 0.1.
      # Tujuannya menaikkan ketajaman ranking awal tanpa mematikan novelty
      # inspired_by sepenuhnya.
      echo "--model_type cr_hkge --cr_use_relation_weight 1 --cr_use_cross_ref 1 --cr_relation_weight_mode semantic --cr_relation_prior_mode fragrance --cr_relation_prior_strength 1.0 --cr_relation_attention_scale type_count --cr_relation_aware_message 1 --cr_relation_message_scale type_count --cr_cross_ref_bi_interaction 0 --cr_cross_ref_gate 0 --cr_cross_ref_alpha 0.075 --cr_best_metric ndcg --cr_best_k 3 --cr_export_embeddings 1 --cr_model_version cr_hkge_final_alpha_0_075"
      ;;
    cr_hkge_final_alpha_0_05)
      # NFM-focus tuning: cross-reference dibuat sangat konservatif.
      # Ini menguji apakah NDCG@3 membaik ketika sinyal local fragrance/KG
      # kembali lebih dominan dibanding inspired_by.
      echo "--model_type cr_hkge --cr_use_relation_weight 1 --cr_use_cross_ref 1 --cr_relation_weight_mode semantic --cr_relation_prior_mode fragrance --cr_relation_prior_strength 1.0 --cr_relation_attention_scale type_count --cr_relation_aware_message 1 --cr_relation_message_scale type_count --cr_cross_ref_bi_interaction 0 --cr_cross_ref_gate 0 --cr_cross_ref_alpha 0.05 --cr_best_metric ndcg --cr_best_k 3 --cr_export_embeddings 1 --cr_model_version cr_hkge_final_alpha_0_05"
      ;;
    cr_hkge_final_alpha_0_1_prior_0_5)
      # NFM-focus tuning: alpha tetap 0.1, tetapi prior relasi fragrance
      # dilemahkan agar relation attention tidak terlalu agresif.
      echo "--model_type cr_hkge --cr_use_relation_weight 1 --cr_use_cross_ref 1 --cr_relation_weight_mode semantic --cr_relation_prior_mode fragrance --cr_relation_prior_strength 0.5 --cr_relation_attention_scale type_count --cr_relation_aware_message 1 --cr_relation_message_scale type_count --cr_cross_ref_bi_interaction 0 --cr_cross_ref_gate 0 --cr_cross_ref_alpha 0.1 --cr_best_metric ndcg --cr_best_k 3 --cr_export_embeddings 1 --cr_model_version cr_hkge_final_alpha_0_1_prior_0_5"
      ;;
    cr_hkge_final_alpha_0_1_prior_0_25)
      # NFM-focus tuning: versi prior fragrance paling konservatif.
      # Dipakai untuk mengecek apakah ranking awal lebih tajam saat prior
      # relasi hanya menjadi bias ringan, bukan pendorong utama.
      echo "--model_type cr_hkge --cr_use_relation_weight 1 --cr_use_cross_ref 1 --cr_relation_weight_mode semantic --cr_relation_prior_mode fragrance --cr_relation_prior_strength 0.25 --cr_relation_attention_scale type_count --cr_relation_aware_message 1 --cr_relation_message_scale type_count --cr_cross_ref_bi_interaction 0 --cr_cross_ref_gate 0 --cr_cross_ref_alpha 0.1 --cr_best_metric ndcg --cr_best_k 3 --cr_export_embeddings 1 --cr_model_version cr_hkge_final_alpha_0_1_prior_0_25"
      ;;
    cr_hkge_final_alpha_0_1_no_relation_message)
      # NFM-focus tuning: relation attention tetap aktif pada skor KGE/attention,
      # tetapi tidak mengalikan pesan adjacency. Ini mengurangi risiko relation
      # prior mengganggu message passing yang sudah stabil.
      echo "--model_type cr_hkge --cr_use_relation_weight 1 --cr_use_cross_ref 1 --cr_relation_weight_mode semantic --cr_relation_prior_mode fragrance --cr_relation_prior_strength 1.0 --cr_relation_attention_scale type_count --cr_relation_aware_message 0 --cr_relation_message_scale type_count --cr_cross_ref_bi_interaction 0 --cr_cross_ref_gate 0 --cr_cross_ref_alpha 0.1 --cr_best_metric ndcg --cr_best_k 3 --cr_export_embeddings 1 --cr_model_version cr_hkge_final_alpha_0_1_no_relation_message"
      ;;
    cr_hkge_final_alpha_0_075_prior_0_5)
      # Kombinasi dua kalibrasi yang paling mungkin membantu NDCG: alpha 0.075
      # dan prior fragrance 0.5.
      echo "--model_type cr_hkge --cr_use_relation_weight 1 --cr_use_cross_ref 1 --cr_relation_weight_mode semantic --cr_relation_prior_mode fragrance --cr_relation_prior_strength 0.5 --cr_relation_attention_scale type_count --cr_relation_aware_message 1 --cr_relation_message_scale type_count --cr_cross_ref_bi_interaction 0 --cr_cross_ref_gate 0 --cr_cross_ref_alpha 0.075 --cr_best_metric ndcg --cr_best_k 3 --cr_export_embeddings 1 --cr_model_version cr_hkge_final_alpha_0_075_prior_0_5"
      ;;
    cr_hkge_final_gated)
      # Kandidat final: semua novelty tetap aktif, tetapi kontribusi
      # cross-reference dikontrol oleh gate trainable. Init -2.0 berarti
      # kontribusi awal inspired_by kecil lalu dapat naik jika memang berguna.
      echo "--model_type cr_hkge --cr_use_relation_weight 1 --cr_use_cross_ref 1 --cr_relation_weight_mode semantic --cr_relation_prior_mode fragrance --cr_relation_prior_strength 1.0 --cr_relation_attention_scale type_count --cr_relation_aware_message 1 --cr_relation_message_scale type_count --cr_cross_ref_bi_interaction 0 --cr_cross_ref_gate 1 --cr_cross_ref_gate_init -2.0 --cr_best_metric ndcg --cr_best_k 3 --cr_export_embeddings 1 --cr_model_version cr_hkge_final_gated"
      ;;
    A_no_cross_reference)
      # Ablation: menghapus Novelty cross-reference via inspired_by.
      echo "--model_type cr_hkge --cr_use_relation_weight 1 --cr_use_cross_ref 0 --cr_relation_weight_mode semantic --cr_relation_prior_mode fragrance --cr_relation_prior_strength 1.0 --cr_relation_attention_scale type_count --cr_relation_aware_message 1 --cr_relation_message_scale type_count --cr_best_metric ndcg --cr_best_k 3"
      ;;
    A_no_relation_attention)
      # Ablation: menghapus relation-type attention, sehingga relasi fragrance
      # tidak lagi mendapat bobot prioritas berbeda.
      echo "--model_type cr_hkge --cr_use_relation_weight 0 --cr_use_cross_ref 1 --cr_relation_aware_message 0 --cr_cross_ref_bi_interaction 0 --cr_cross_ref_gate 0 --cr_cross_ref_alpha 0.5 --cr_best_metric ndcg --cr_best_k 3"
      ;;
    A_no_relation_message)
      # Ablation: relation attention tetap mempengaruhi skor attention,
      # tetapi tidak mengalikan pesan adjacency per relasi.
      echo "--model_type cr_hkge --cr_use_relation_weight 1 --cr_use_cross_ref 1 --cr_relation_weight_mode semantic --cr_relation_prior_mode fragrance --cr_relation_prior_strength 1.0 --cr_relation_attention_scale type_count --cr_relation_aware_message 0 --cr_relation_message_scale type_count --cr_cross_ref_bi_interaction 0 --cr_cross_ref_gate 0 --cr_cross_ref_alpha 0.5 --cr_best_metric ndcg --cr_best_k 3"
      ;;
    A_no_fragrance_prior)
      # Ablation: bobot relasi tetap learnable, tetapi inisialisasinya netral
      # tanpa prior domain fragrance.
      echo "--model_type cr_hkge --cr_use_relation_weight 1 --cr_use_cross_ref 1 --cr_relation_weight_mode semantic --cr_relation_prior_mode none --cr_relation_attention_scale type_count --cr_relation_aware_message 1 --cr_relation_message_scale type_count --cr_cross_ref_bi_interaction 0 --cr_cross_ref_gate 0 --cr_cross_ref_alpha 0.5 --cr_best_metric ndcg --cr_best_k 3"
      ;;
    A_no_novelty_modules)
      # Ablation paling dasar: semua novelty CR-HKGE dimatikan.
      # Ini mendekati KGAT dengan wrapper CR-HKGE dan dipakai sebagai kontrol.
      echo "--model_type cr_hkge --cr_use_relation_weight 0 --cr_use_cross_ref 0 --cr_relation_aware_message 0 --cr_best_metric ndcg --cr_best_k 3"
      ;;
    *)
      echo "unknown study target: $1" >&2
      return 1
      ;;
  esac
}

run_target() {
  local target="$1"
  local train_log="$LOG_DIR/${target}_train.log"
  local eval_log="$LOG_DIR/${target}_subset_eval.log"
  local weights_path="../final_study/${target}/"

  read -r -a extra_args <<< "$(model_args "$target")"

  echo "==> Training $target"
  # Setiap target dilatih dari awal dengan split dan hyperparameter yang sama
  # agar perbandingan baseline, CR-HKGE, dan ablation tetap fair.
  (
    cd "$MODEL_DIR"
    python Main.py \
      --weights_path "$weights_path" \
      "${COMMON_ARGS[@]}" \
      "${extra_args[@]}"
  ) 2>&1 | tee "$train_log"

  echo "==> Evaluating $target"
  (
    cd "$MODEL_DIR"
    python evaluate_item_subsets.py \
      --weights_path "$weights_path" \
      "${COMMON_ARGS[@]}" \
      "${extra_args[@]}"
  ) 2>&1 | tee "$eval_log"

  echo "==> Finished $target"
  echo "train_log=$train_log"
  echo "eval_log=$eval_log"
}

if [ "$#" -eq 0 ]; then
  targets=(
    bprmf
    cke
    nfm
    cfkg
    kgat
    cr_hkge_final
    A_no_cross_reference
    A_no_relation_attention
    A_no_relation_message
    A_no_fragrance_prior
    A_no_novelty_modules
  )
else
  targets=("$@")
fi

for target in "${targets[@]}"; do
  run_target "$target"
done

python "$ROOT_DIR/scripts/summarize_final_study.py" "$LOG_DIR"/*_subset_eval.log | tee "$LOG_DIR/final_summary.md"
