#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

python scripts/train_cr_hkge_pure.py \
  --dataset-path dataset-aromatique-kgat-ready \
  --artifact-path artifacts/cr_hkge_pure \
  --epochs "${CR_HKGE_PURE_EPOCHS:-100}" \
  --embed-dim "${CR_HKGE_PURE_EMBED_DIM:-64}" \
  --layers "${CR_HKGE_PURE_LAYERS:-3}" \
  --cross-ref-alpha "${CR_HKGE_PURE_CROSS_REF_ALPHA:-0.5}" \
  --prior-strength "${CR_HKGE_PURE_PRIOR_STRENGTH:-1.0}"

ARTIFACT="$(ls -td artifacts/cr_hkge_pure/dataset-aromatique-kgat-ready/cr_hkge_blueprint_pure_* | head -n 1)"
echo "ARTIFACT=$ARTIFACT"

python scripts/cr_hkge_retrieve.py \
  --artifact-path "$ARTIFACT" \
  --query-json '{"accords":["vanilla","sweet","amber"],"family":"AMBER","occasion":"evening"}' \
  --top-k 3

python scripts/evaluate_cr_hkge_pure_queries.py \
  --artifact-path "$ARTIFACT" \
  --ks '[3,5,10]'
