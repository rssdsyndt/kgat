"""Build a CR-HKGE-ready Aromatique dataset.

Dataset lama `dataset-aromatique-kgat-ready` tetap dipertahankan sebagai
baseline/rollback. Script ini membuat dataset baru yang train/test positive
pairs-nya lebih selaras dengan tiga novelty CR-HKGE:

1. Fragrance-specific heterogeneous KG construction
2. Cross-reference via `inspired_by`
3. Relation-type priority/attention

Output tetap mengikuti format KGAT (`train.txt`, `test.txt`, `kg_final.txt`)
agar KGAT dan CR-HKGE bisa dibandingkan pada split yang sama.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "dataset-aromatique-kgat-ready"
DEFAULT_OUTPUT = ROOT / "dataset-aromatique-crhkge-ready"

COPY_FILES = [
    "entity2id.txt",
    "entity2id_typed.tsv",
    "old_to_new_entity_id.tsv",
    "product2id.tsv",
    "kg_final.txt",
    "relation2id.txt",
    "item_list.txt",
    "entity_list.txt",
]

SCORING_WEIGHTS = {
    "local_accord": 2.00,
    "visual_note": 1.25,
    "local_family": 1.50,
    "global_accord": 2.00,
    "global_family": 1.25,
    "cross_ref_global_to_local_accord": 1.75,
    "cross_ref_global_to_local_family": 0.80,
    "same_global_reference": 3.00,
    "sem_similar": 0.35,
    "enriched_cross_ref_bonus": 0.20,
}


@dataclass
class Entity:
    new_id: int
    old_id: int
    entity_type: str
    name: str


@dataclass
class ProductFeature:
    product_id: int
    product_name: str
    local_accords: set[str] = field(default_factory=set)
    visual_notes: set[str] = field(default_factory=set)
    local_families: set[str] = field(default_factory=set)
    global_refs: set[int] = field(default_factory=set)
    global_accords: set[str] = field(default_factory=set)
    global_families: set[str] = field(default_factory=set)
    sem_similar_items: set[int] = field(default_factory=set)

    @property
    def is_enriched(self) -> bool:
        return bool(self.global_refs)


def normalize_term(value: str) -> str:
    return " ".join(value.casefold().strip().split())


def read_relation_map(path: Path) -> dict[int, str]:
    rows: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines()[1:]:
        if not line.strip():
            continue
        parts = line.split()
        rows[int(parts[-1])] = " ".join(parts[:-1])
    return rows


def read_entities(path: Path) -> dict[int, Entity]:
    rows: dict[int, Entity] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines()[1:]:
        if not line.strip():
            continue
        new_id, old_id, entity_type, name = line.split("\t", 3)
        rows[int(new_id)] = Entity(
            new_id=int(new_id),
            old_id=int(old_id),
            entity_type=entity_type,
            name=name,
        )
    return rows


def read_products(path: Path) -> dict[int, str]:
    rows: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines()[1:]:
        if not line.strip():
            continue
        product_id, _old_id, product_name = line.split("\t", 2)
        rows[int(product_id)] = product_name
    return rows


def read_triples(path: Path) -> list[tuple[int, int, int]]:
    triples: list[tuple[int, int, int]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        head, relation, tail = (int(part) for part in line.split())
        triples.append((head, relation, tail))
    return triples


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    inter = len(left & right)
    if inter == 0:
        return 0.0
    return inter / len(left | right)


def relation_counts(triples: list[tuple[int, int, int]], relation_by_id: dict[int, str]) -> dict[str, int]:
    counts = Counter(relation_by_id[relation_id] for _head, relation_id, _tail in triples)
    return dict(sorted(counts.items()))


def build_product_features(
    products: dict[int, str],
    entities: dict[int, Entity],
    relation_by_id: dict[int, str],
    triples: list[tuple[int, int, int]],
) -> dict[int, ProductFeature]:
    features = {
        product_id: ProductFeature(product_id=product_id, product_name=name)
        for product_id, name in products.items()
    }
    global_accords: dict[int, set[str]] = defaultdict(set)
    global_families: dict[int, set[str]] = defaultdict(set)

    for head, relation_id, tail in triples:
        relation_name = relation_by_id[relation_id]
        tail_entity = entities[tail]
        tail_term = normalize_term(tail_entity.name)

        if relation_name == "has_global_accord":
            global_accords[head].add(tail_term)
        elif relation_name == "belongs_to_global_family":
            global_families[head].add(tail_term)

    for head, relation_id, tail in triples:
        if head not in features:
            continue

        relation_name = relation_by_id[relation_id]
        target_entity = entities[tail]
        target_term = normalize_term(target_entity.name)
        product = features[head]

        if relation_name == "has_accord":
            product.local_accords.add(target_term)
        elif relation_name == "has_visual_note":
            product.visual_notes.add(target_term)
        elif relation_name == "belongs_to_family":
            product.local_families.add(target_term)
        elif relation_name == "inspired_by":
            product.global_refs.add(tail)
            product.global_accords.update(global_accords.get(tail, set()))
            product.global_families.update(global_families.get(tail, set()))
        elif relation_name == "sem_similar" and target_entity.entity_type == "product":
            product.sem_similar_items.add(tail)

    return features


def score_pair(left: ProductFeature, right: ProductFeature) -> tuple[float, dict[str, float]]:
    parts: dict[str, float] = {}

    parts["local_accord"] = SCORING_WEIGHTS["local_accord"] * jaccard(left.local_accords, right.local_accords)
    parts["visual_note"] = SCORING_WEIGHTS["visual_note"] * jaccard(left.visual_notes, right.visual_notes)
    parts["local_family"] = (
        SCORING_WEIGHTS["local_family"]
        if left.local_families and right.local_families and bool(left.local_families & right.local_families)
        else 0.0
    )
    parts["global_accord"] = SCORING_WEIGHTS["global_accord"] * jaccard(left.global_accords, right.global_accords)
    parts["global_family"] = SCORING_WEIGHTS["global_family"] * jaccard(left.global_families, right.global_families)
    parts["cross_ref_global_to_local_accord"] = SCORING_WEIGHTS["cross_ref_global_to_local_accord"] * max(
        jaccard(left.global_accords, right.local_accords),
        jaccard(right.global_accords, left.local_accords),
    )
    parts["cross_ref_global_to_local_family"] = SCORING_WEIGHTS["cross_ref_global_to_local_family"] * max(
        jaccard(left.global_families, right.local_families),
        jaccard(right.global_families, left.local_families),
    )
    parts["same_global_reference"] = (
        SCORING_WEIGHTS["same_global_reference"]
        if left.global_refs and right.global_refs and bool(left.global_refs & right.global_refs)
        else 0.0
    )
    parts["sem_similar"] = (
        SCORING_WEIGHTS["sem_similar"]
        if right.product_id in left.sem_similar_items or left.product_id in right.sem_similar_items
        else 0.0
    )
    parts["enriched_cross_ref_bonus"] = (
        SCORING_WEIGHTS["enriched_cross_ref_bonus"]
        if left.is_enriched
        and right.is_enriched
        and (
            parts["global_accord"] > 0.0
            or parts["global_family"] > 0.0
            or parts["cross_ref_global_to_local_accord"] > 0.0
            or parts["same_global_reference"] > 0.0
        )
        else 0.0
    )

    score = sum(parts.values())
    return score, parts


def split_ranked_candidates(
    candidates: list[tuple[int, float, dict[str, float]]],
    train_per_profile: int,
    test_per_profile: int,
) -> tuple[list[tuple[int, float, dict[str, float], int]], list[tuple[int, float, dict[str, float], int]]]:
    total_needed = train_per_profile + test_per_profile
    selected = candidates[:total_needed]
    if len(selected) < total_needed:
        raise ValueError(
            "Not enough candidates for profile; got %d, need %d" % (len(selected), total_needed)
        )

    test_positions: set[int] = set()
    stride = max(1, math.floor(total_needed / test_per_profile))
    pos = 0
    while len(test_positions) < test_per_profile and pos < total_needed:
        test_positions.add(pos)
        pos += stride
    pos = total_needed - 1
    while len(test_positions) < test_per_profile:
        test_positions.add(pos)
        pos -= 1

    train_rows = []
    test_rows = []
    for rank, (target_id, score, parts) in enumerate(selected, start=1):
        row = (target_id, score, parts, rank)
        if rank - 1 in test_positions:
            test_rows.append(row)
        else:
            train_rows.append(row)

    if len(train_rows) != train_per_profile or len(test_rows) != test_per_profile:
        raise AssertionError("Unexpected train/test split sizes")
    return train_rows, test_rows


def write_interaction_file(path: Path, rows: dict[int, list[int]]) -> None:
    lines = []
    for profile_id in sorted(rows):
        items = " ".join(str(item_id) for item_id in rows[profile_id])
        lines.append("%d %s" % (profile_id, items))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_dataset(args: argparse.Namespace) -> dict[str, object]:
    source = Path(args.source_dataset)
    if not source.is_absolute():
        source = ROOT / source

    output = Path(args.output_dataset)
    if not output.is_absolute():
        output = ROOT / output

    if not source.exists():
        raise FileNotFoundError("Source dataset not found: %s" % source)
    if output.exists():
        if not args.overwrite:
            raise FileExistsError("Output dataset already exists: %s" % output)
        shutil.rmtree(output)
    output.mkdir(parents=True)

    for filename in COPY_FILES:
        shutil.copy2(source / filename, output / filename)

    relation_by_id = read_relation_map(source / "relation2id.txt")
    entities = read_entities(source / "entity2id_typed.tsv")
    products = read_products(source / "product2id.tsv")
    triples = read_triples(source / "kg_final.txt")
    features = build_product_features(products, entities, relation_by_id, triples)

    train_rows: dict[int, list[int]] = {}
    test_rows: dict[int, list[int]] = {}
    pair_score_lines = [
        "\t".join(
            [
                "profile_id",
                "source_product_id",
                "source_product_name",
                "target_product_id",
                "target_product_name",
                "split",
                "rank",
                "score",
                *SCORING_WEIGHTS.keys(),
            ]
        )
    ]
    profile_lines = ["profile_id\tsource_product_id\tsource_product_name"]
    score_values: list[float] = []

    for profile_id, source_product_id in enumerate(sorted(features)):
        source_feature = features[source_product_id]
        profile_lines.append(
            "%d\t%d\t%s" % (profile_id, source_product_id, source_feature.product_name)
        )

        candidates: list[tuple[int, float, dict[str, float]]] = []
        for target_product_id in sorted(features):
            if target_product_id == source_product_id:
                continue
            score, parts = score_pair(source_feature, features[target_product_id])
            if score >= args.min_score:
                candidates.append((target_product_id, score, parts))

        candidates.sort(key=lambda row: (-row[1], row[0]))
        train_split, test_split = split_ranked_candidates(
            candidates,
            train_per_profile=args.train_per_profile,
            test_per_profile=args.test_per_profile,
        )

        train_rows[profile_id] = [target_id for target_id, _score, _parts, _rank in train_split]
        test_rows[profile_id] = [target_id for target_id, _score, _parts, _rank in test_split]

        for split_name, split_rows in [("train", train_split), ("test", test_split)]:
            for target_id, score, parts, rank in split_rows:
                score_values.append(score)
                target_feature = features[target_id]
                pair_score_lines.append(
                    "\t".join(
                        [
                            str(profile_id),
                            str(source_product_id),
                            source_feature.product_name,
                            str(target_id),
                            target_feature.product_name,
                            split_name,
                            str(rank),
                            "%.6f" % score,
                            *("%.6f" % parts[key] for key in SCORING_WEIGHTS.keys()),
                        ]
                    )
                )

    write_interaction_file(output / "train.txt", train_rows)
    write_interaction_file(output / "test.txt", test_rows)
    (output / "positive_pair_scores.tsv").write_text(
        "\n".join(pair_score_lines) + "\n",
        encoding="utf-8",
    )
    (output / "profile2product.tsv").write_text(
        "\n".join(profile_lines) + "\n",
        encoding="utf-8",
    )

    all_train_items = {item for items in train_rows.values() for item in items}
    all_test_items = {item for items in test_rows.values() for item in items}
    enriched_products = [product_id for product_id, feature in features.items() if feature.is_enriched]
    summary: dict[str, object] = {
        "source_dataset": source.name,
        "output_dataset": output.name,
        "construction": "cr_hkge_aligned_content_positive_pairs",
        "profile_semantics": "one content/query profile per source product",
        "n_products": len(products),
        "n_profiles": len(train_rows),
        "train_per_profile": args.train_per_profile,
        "test_per_profile": args.test_per_profile,
        "n_train_interactions": sum(len(items) for items in train_rows.values()),
        "n_test_interactions": sum(len(items) for items in test_rows.values()),
        "n_unique_train_items": len(all_train_items),
        "n_unique_test_items": len(all_test_items),
        "n_enriched_products": len(enriched_products),
        "n_standard_products": len(products) - len(enriched_products),
        "relation_counts": relation_counts(triples, relation_by_id),
        "scoring_weights": SCORING_WEIGHTS,
        "score_min": min(score_values),
        "score_max": max(score_values),
        "score_mean": sum(score_values) / len(score_values),
        "files": {
            "train": "train.txt",
            "test": "test.txt",
            "pair_scores": "positive_pair_scores.tsv",
            "profile_mapping": "profile2product.tsv",
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    readme = [
        "# dataset-aromatique-crhkge-ready",
        "",
        "Dataset ini dibuat oleh `scripts/build_cr_hkge_ready_dataset.py`.",
        "",
        "Tujuan utamanya adalah menyelaraskan `train.txt` dan `test.txt` dengan tiga novelty CR-HKGE:",
        "",
        "1. Fragrance-specific heterogeneous KG construction.",
        "2. Cross-reference via `inspired_by`.",
        "3. Relation-type priority/attention.",
        "",
        "Format file tetap KGAT-compatible, sehingga KGAT, CR-HKGE, dan baseline lain dapat dibandingkan pada split yang sama.",
        "",
        "Setiap profile merepresentasikan satu produk sumber/query content. Item positif adalah produk lain dengan skor relevance tertinggi berdasarkan kombinasi local fragrance attributes, global reference enrichment, cross-reference bridge, dan weak `sem_similar` support.",
        "",
        "File penting:",
        "",
        "- `train.txt`: positive pairs untuk BPR training.",
        "- `test.txt`: held-out positive pairs untuk evaluasi Top-K.",
        "- `profile2product.tsv`: mapping profile ke produk sumber.",
        "- `positive_pair_scores.tsv`: audit skor setiap pasangan positif.",
        "- `summary.json`: statistik dataset dan bobot scoring.",
        "",
    ]
    (output / "README.md").write_text("\n".join(readme), encoding="utf-8")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output-dataset", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--train-per-profile", type=int, default=8)
    parser.add_argument("--test-per-profile", type=int, default=4)
    parser.add_argument("--min-score", type=float, default=0.01)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summary = build_dataset(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
