"""Offline CR-HKGE retrieval smoke test.

This script consumes exported CR-HKGE artifacts and implements the serving
contract expected by the thesis blueprint:
structured query -> query embedding -> Top-3 products -> matched KG paths.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any


DEFAULT_QUERY = {
    "accords": ["vanilla", "sweet", "amber"],
    "family": "AMBER",
    "occasion": "evening",
}

FIELD_ALIASES = {
    "accords": ["accords", "main_accords"],
    "family": ["family", "olfactory_family"],
    "notes": ["notes"],
    "visual_notes": ["visual_notes", "visual_note"],
    "reference": ["reference", "global_ref", "revolutionize"],
    "inspired_by": ["inspired_by"],
}


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = re.sub(r"[_\-/;:,]+", " ", value.casefold())
    return re.sub(r"\s+", " ", value).strip()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0:
        return vector
    return [value / norm for value in vector]


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)] if str(value).strip() else []


def unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        key = normalize(value)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


class CRHKGERetriever:
    def __init__(self, artifact_path: Path):
        self.artifact_path = artifact_path
        self.config = self._load_config()
        self.relation_weights = self._load_relation_weights()
        self.entities = self._load_embeddings("entity_embeddings.tsv")
        self.products = self._load_embeddings("product_embeddings.tsv")
        self.kg_paths = self._load_kg_paths()

        if not self.products:
            raise RuntimeError("product_embeddings.tsv is empty")
        if not self.entities:
            raise RuntimeError("entity_embeddings.tsv is empty")

        self.embedding_dim = len(self.products[0]["embedding"])
        self.product_matrix = [l2_normalize(row["embedding"]) for row in self.products]
        self.entity_by_id = {row["entity_id"]: row for row in self.entities}
        self.entity_index = self._build_entity_index()

    def _load_config(self) -> dict[str, Any]:
        path = self.artifact_path / "query_encoder_config.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))

        return {
            "entity_matching": {
                "accords": ["accord", "global_accord"],
                "family": ["family", "global_family"],
                "notes": ["note"],
                "visual_notes": ["note"],
                "reference": ["global_ref"],
                "inspired_by": ["global_ref"],
            },
            "field_relation_map": {
                "accords": {"accord": "has_accord", "global_accord": "has_global_accord"},
                "family": {"family": "belongs_to_family", "global_family": "belongs_to_global_family"},
                "notes": {"note": "has_visual_note"},
                "visual_notes": {"note": "has_visual_note"},
                "reference": {"global_ref": "inspired_by"},
                "inspired_by": {"global_ref": "inspired_by"},
            },
            "top_k": 3,
        }

    def _load_relation_weights(self) -> dict[str, float]:
        path = self.artifact_path / "relation_weights.tsv"
        if not path.exists():
            return {}

        weights: dict[str, float] = {}
        for row in read_tsv(path):
            relation_name = row.get("relation_type_name", "")
            multiplier = row.get("multiplier", "")
            if not relation_name or not multiplier:
                continue
            try:
                weights[relation_name] = float(multiplier)
            except ValueError:
                continue
        return weights

    def _load_embeddings(self, filename: str) -> list[dict[str, Any]]:
        path = self.artifact_path / filename
        rows = []
        for row in read_tsv(path):
            embedding = [float(value) for value in json.loads(row["embedding_json"])]
            rows.append({
                "entity_id": int(row["entity_id"]),
                "old_entity_id": row.get("old_entity_id", ""),
                "entity_type": row.get("entity_type", ""),
                "entity_name": row.get("entity_name", ""),
                "embedding": embedding,
                "model_version": row.get("model_version", ""),
            })
        return rows

    def _load_kg_paths(self) -> dict[int, list[dict[str, Any]]]:
        path = self.artifact_path / "kg_paths.jsonl"
        if not path.exists():
            return {}

        rows: dict[int, list[dict[str, Any]]] = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                rows[int(item["product_id"])] = item.get("kg_path", [])
        return rows

    def _build_entity_index(self) -> dict[tuple[str, str], list[dict[str, Any]]]:
        index: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for entity in self.entities:
            key = (normalize(entity["entity_name"]), entity["entity_type"])
            index.setdefault(key, []).append(entity)
        return index

    def recommend(self, query: dict[str, Any], top_k: int | None = None) -> dict[str, Any]:
        matched_entities, unmatched_terms = self.match_query_entities(query)
        query_vector = self.build_query_vector(matched_entities)
        scores = [dot(product_vector, query_vector) for product_vector in self.product_matrix]

        k = int(top_k or self.config.get("top_k", 3))
        top_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:k]
        recommendations = []
        for rank, product_index in enumerate(top_indices, start=1):
            product = self.products[int(product_index)]
            product_id = int(product["entity_id"])
            recommendations.append({
                "rank": rank,
                "product_id": str(product_id),
                "product_name": product["entity_name"],
                "match_score": round(clamp((scores[product_index] + 1.0) * 50.0, 0.0, 100.0), 2),
                "raw_cosine": round(scores[product_index], 6),
                "kg_path": self.build_kg_path(product_id, matched_entities),
                "model_version": product.get("model_version", ""),
            })

        return {
            "query_vector_dim": len(query_vector),
            "matched_entities": [
                {
                    "text": item["text"],
                    "entity_id": int(item["entity_id"]),
                    "entity_type": item["entity_type"],
                    "entity_name": item["entity_name"],
                    "relation": item["relation"],
                    "weight": round(float(item["weight"]), 6),
                }
                for item in matched_entities
            ],
            "unmatched_terms": unique(unmatched_terms),
            "recommendations": recommendations,
        }

    def match_query_entities(self, query: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
        matched: list[dict[str, Any]] = []
        unmatched: list[str] = []
        consumed_keys: set[str] = set()
        seen_matches: set[tuple[int, str]] = set()

        entity_matching = self.config.get("entity_matching", {})
        relation_map = self.config.get("field_relation_map", {})

        for field, allowed_types in entity_matching.items():
            values = []
            for alias in FIELD_ALIASES.get(field, [field]):
                if alias in query:
                    consumed_keys.add(alias)
                    values.extend(as_list(query.get(alias)))

            for text in values:
                candidates = self._find_candidates(text, allowed_types)
                if not candidates:
                    unmatched.append(text)
                    continue

                for entity in candidates:
                    relation = relation_map.get(field, {}).get(entity["entity_type"], "")
                    if not relation:
                        continue
                    key = (int(entity["entity_id"]), relation)
                    if key in seen_matches:
                        continue
                    seen_matches.add(key)
                    matched.append({
                        "text": text,
                        "entity_id": int(entity["entity_id"]),
                        "entity_type": entity["entity_type"],
                        "entity_name": entity["entity_name"],
                        "relation": relation,
                        "weight": self.relation_weights.get(relation, 1.0),
                    })

        for key, value in query.items():
            if key in consumed_keys:
                continue
            unmatched.extend(as_list(value))

        return matched, unmatched

    def _find_candidates(self, text: str, allowed_types: list[str]) -> list[dict[str, Any]]:
        term = normalize(text)
        if not term:
            return []

        candidates = []
        for entity_type in allowed_types:
            candidates.extend(self.entity_index.get((term, entity_type), []))

        if candidates:
            return candidates

        # Conservative fuzzy match for minor dataset spelling variants.
        fuzzy = []
        for entity_type in allowed_types:
            for entity in self.entities:
                if entity["entity_type"] != entity_type:
                    continue
                entity_name = normalize(entity["entity_name"])
                if len(term) >= 4 and (term in entity_name or entity_name in term):
                    fuzzy.append(entity)
        return fuzzy[:3]

    def build_query_vector(self, matched_entities: list[dict[str, Any]]) -> list[float]:
        if not matched_entities:
            raise RuntimeError("No query term matched a KG entity; cannot build CR-HKGE query vector")

        weighted_sum = [0.0] * self.embedding_dim
        total_weight = 0.0
        for item in matched_entities:
            entity = self.entity_by_id[item["entity_id"]]
            weight = float(item["weight"])
            for idx, value in enumerate(entity["embedding"]):
                weighted_sum[idx] += value * weight
            total_weight += weight

        if total_weight <= 0:
            raise RuntimeError("Matched query entities have zero total weight")
        return l2_normalize([value / total_weight for value in weighted_sum])

    def build_kg_path(self, product_id: int, matched_entities: list[dict[str, Any]], max_paths: int = 6) -> list[dict[str, Any]]:
        paths = self.kg_paths.get(product_id, [])
        matched_ids = {int(item["entity_id"]) for item in matched_entities}
        matched_names = {normalize(item["entity_name"]) for item in matched_entities}

        selected: list[dict[str, Any]] = []

        def append_path(path: dict[str, Any], matched: bool) -> None:
            if len(selected) >= max_paths:
                return
            relation = path.get("relation_name", "")
            entity_name = path.get("tail_entity_name", "")
            selected.append({
                "relation": relation,
                "entity": entity_name,
                "matched": bool(matched),
                "reason": self._path_reason(relation, entity_name, matched, path.get("relation_scope", "")),
            })

        for path in paths:
            tail_id = int(path.get("tail_entity_id", -1))
            tail_name = normalize(str(path.get("tail_entity_name", "")))
            relation = path.get("relation_name", "")
            is_match = tail_id in matched_ids or tail_name in matched_names
            if is_match and relation != "inspired_by":
                append_path(path, True)

        for path in paths:
            relation = path.get("relation_name", "")
            if relation != "inspired_by":
                continue
            tail_id = int(path.get("tail_entity_id", -1))
            tail_name = normalize(str(path.get("tail_entity_name", "")))
            append_path(path, tail_id in matched_ids or tail_name in matched_names)

        if len(selected) < max_paths:
            for path in paths:
                relation = path.get("relation_name", "")
                if relation in {"has_accord", "belongs_to_family", "has_visual_note", "has_global_accord", "belongs_to_global_family"}:
                    tail_id = int(path.get("tail_entity_id", -1))
                    tail_name = normalize(str(path.get("tail_entity_name", "")))
                    is_match = tail_id in matched_ids or tail_name in matched_names
                    if not is_match:
                        append_path(path, False)
                if len(selected) >= max_paths:
                    break

        return selected

    def _path_reason(self, relation: str, entity_name: str, matched: bool, scope: str) -> str:
        if matched:
            return "Sesuai preferensi %s" % entity_name
        if relation == "inspired_by":
            return "Memberi konteks referensi global"
        if scope == "global_reference":
            return "Konteks semantic enrichment dari referensi global"
        return "Atribut KG pendukung produk"


def load_query(args: argparse.Namespace) -> dict[str, Any]:
    if args.query_file:
        return json.loads(Path(args.query_file).read_text(encoding="utf-8"))
    if args.query_json:
        return json.loads(args.query_json)
    return DEFAULT_QUERY


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CR-HKGE artifact retrieval.")
    parser.add_argument("--artifact-path", required=True, type=Path)
    parser.add_argument("--query-json", default="")
    parser.add_argument("--query-file", default="")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    retriever = CRHKGERetriever(args.artifact_path)
    result = retriever.recommend(load_query(args), top_k=args.top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
