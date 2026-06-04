"""Evaluate pure CR-HKGE artifacts with query-to-product retrieval metrics.

This evaluation follows the blueprint serving contract:
structured query -> query vector -> cosine ranking over product embeddings.

It does not read KGAT train.txt/test.txt and does not use user-item pairs.
Queries are generated from each product's KG paths, without using product names.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cr_hkge_retrieve import CRHKGERetriever  # noqa: E402


LOCAL_RELATION_TO_FIELD = {
    "has_accord": "accords",
    "belongs_to_family": "family",
    "has_visual_note": "visual_notes",
}


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(value)] if str(value).strip() else []


def unique(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        key = value.casefold().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def add_query_value(query: dict[str, Any], field: str, value: str) -> None:
    if not value:
        return
    if field == "family":
        query.setdefault(field, value)
        return
    query.setdefault(field, [])
    query[field].append(value)


def product_query(paths: list[dict[str, Any]], mode: str, max_accords: int) -> dict[str, Any]:
    query: dict[str, Any] = {}
    reference = ""

    for path in paths:
        relation = path.get("relation_name", "")
        entity = path.get("tail_entity_name", "")
        if relation == "inspired_by" and not reference:
            reference = entity
        if relation not in LOCAL_RELATION_TO_FIELD:
            continue
        add_query_value(query, LOCAL_RELATION_TO_FIELD[relation], entity)

    if "accords" in query:
        query["accords"] = unique(as_list(query["accords"]))[:max_accords]
    if "visual_notes" in query:
        query["visual_notes"] = unique(as_list(query["visual_notes"]))[:1]

    if mode == "local":
        return query
    if mode == "cross_reference":
        return {"reference": reference} if reference else {}
    if mode == "hybrid":
        if reference:
            query["reference"] = reference
        return query
    raise ValueError("unsupported query mode: %s" % mode)


def target_rank(recommendations: list[dict[str, Any]], target_product_id: int) -> int | None:
    target = str(target_product_id)
    for row in recommendations:
        if str(row.get("product_id")) == target:
            return int(row["rank"])
    return None


def metric_summary(ranks: list[int | None], ks: list[int]) -> dict[str, Any]:
    n = len(ranks)
    if n == 0:
        return {
            "n_queries": 0,
            "mrr": 0.0,
            "hit": [0.0 for _ in ks],
            "recall": [0.0 for _ in ks],
            "ndcg": [0.0 for _ in ks],
        }

    hit = []
    recall = []
    ndcg = []
    for k in ks:
        hits = [1.0 if rank is not None and rank <= k else 0.0 for rank in ranks]
        hit.append(sum(hits) / n)
        recall.append(sum(hits) / n)
        ndcg.append(
            sum(
                (1.0 / math.log2(rank + 1.0)) if rank is not None and rank <= k else 0.0
                for rank in ranks
            ) / n
        )

    mrr = sum((1.0 / rank) if rank is not None else 0.0 for rank in ranks) / n
    return {
        "n_queries": n,
        "mrr": mrr,
        "hit": hit,
        "recall": recall,
        "ndcg": ndcg,
    }


def fmt(values: list[float]) -> str:
    return "[" + "\t".join("%.5f" % value for value in values) + "]"


def evaluate_mode(
    retriever: CRHKGERetriever,
    mode: str,
    ks: list[int],
    candidate_pool: int,
    max_accords: int,
    sample_limit: int,
):
    max_k = max(ks)
    ranks: list[int | None] = []
    rows = []

    for product_id in sorted(retriever.kg_paths.keys()):
        paths = retriever.kg_paths[product_id]
        query = product_query(paths, mode, max_accords=max_accords)
        if not query:
            continue

        try:
            result = retriever.recommend(
                query,
                top_k=max_k,
                candidate_pool=max(candidate_pool, max_k),
                min_matched_paths=1,
                match_bonus=0.05,
                query_aware_rerank=True,
            )
        except RuntimeError:
            continue

        rank = target_rank(result["recommendations"], product_id)
        ranks.append(rank)
        rows.append({
            "product_id": product_id,
            "query": query,
            "target_rank": rank,
            "top_product_ids": [row["product_id"] for row in result["recommendations"]],
        })

        if sample_limit > 0 and len(rows) >= sample_limit:
            break

    return metric_summary(ranks, ks), rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate pure CR-HKGE query retrieval artifacts.")
    parser.add_argument("--artifact-path", required=True, type=Path)
    parser.add_argument("--modes", default="local,cross_reference,hybrid")
    parser.add_argument("--ks", default="[3,5,10]")
    parser.add_argument("--candidate-pool", type=int, default=340)
    parser.add_argument("--max-accords", type=int, default=3)
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    retriever = CRHKGERetriever(args.artifact_path)
    ks = [int(value) for value in json.loads(args.ks)]
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]

    print("artifact=%s" % args.artifact_path)
    print("evaluation=query_to_product_retrieval")
    print("interaction_training=none")
    print("train_txt_used=0")
    print("test_txt_used=0")
    print("Ks=%s" % ks)

    all_rows = defaultdict(list)
    for mode in modes:
        summary, rows = evaluate_mode(
            retriever,
            mode=mode,
            ks=ks,
            candidate_pool=args.candidate_pool,
            max_accords=args.max_accords,
            sample_limit=args.sample_limit,
        )
        all_rows[mode] = rows
        print("%s: n_queries=%d, mrr=%.5f" % (mode, summary["n_queries"], summary["mrr"]))
        print(
            "  hit=%s, recall=%s, ndcg=%s" %
            (fmt(summary["hit"]), fmt(summary["recall"]), fmt(summary["ndcg"]))
        )

    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(all_rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
