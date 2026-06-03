"""Audit overlap between Aromatique interaction proxy and KG relations.

This diagnostic checks whether test positives are already recoverable from
local product attributes used by the KGAT baseline. High overlap means the
current offline Top-K task rewards local family/accord structure more than the
cross-reference novelty targeted by CR-HKGE.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset-aromatique-kgat-ready"
N_ITEMS = 340


def read_relation_map(path: Path):
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        parts = line.split()
        result[" ".join(parts[:-1])] = int(parts[-1])
    return result


def read_interactions(path: Path):
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = [int(part) for part in line.split()]
        result[parts[0]] = parts[1:]
    return result


def main():
    rel = read_relation_map(DATASET / "relation2id.txt")
    families = defaultdict(set)
    accords = defaultdict(set)
    notes = defaultdict(set)
    global_refs = defaultdict(set)
    sem_neighbors = defaultdict(set)
    relation_counts = defaultdict(int)

    for line in (DATASET / "kg_final.txt").read_text(encoding="utf-8").splitlines():
        head, relation, tail = [int(part) for part in line.split()]
        relation_counts[relation] += 1

        if head >= N_ITEMS:
            continue

        if relation == rel["belongs_to_family"]:
            families[head].add(tail)
        elif relation == rel["has_accord"]:
            accords[head].add(tail)
        elif relation == rel["has_visual_note"]:
            notes[head].add(tail)
        elif relation == rel["inspired_by"]:
            global_refs[head].add(tail)
        elif relation == rel["sem_similar"] and tail < N_ITEMS:
            sem_neighbors[head].add(tail)

    train = read_interactions(DATASET / "train.txt")
    test = read_interactions(DATASET / "test.txt")

    stats = defaultdict(int)
    total = 0
    for user, test_items in test.items():
        train_items = train.get(user, [])
        for item in test_items:
            total += 1
            same_family = any(families[item] & families[train_item] for train_item in train_items)
            same_note = any(notes[item] & notes[train_item] for train_item in train_items)
            same_global_ref = any(global_refs[item] & global_refs[train_item] for train_item in train_items)
            sem_link = any(
                item in sem_neighbors[train_item] or train_item in sem_neighbors[item]
                for train_item in train_items
            )
            max_shared_accord = max(
                [len(accords[item] & accords[train_item]) for train_item in train_items] or [0]
            )

            stats["same_family"] += int(same_family)
            stats["same_visual_note"] += int(same_note)
            stats["same_global_ref"] += int(same_global_ref)
            stats["sem_similar_to_train"] += int(sem_link)
            stats["shared_accord_ge1"] += int(max_shared_accord >= 1)
            stats["shared_accord_ge2"] += int(max_shared_accord >= 2)
            stats["shared_accord_ge3"] += int(max_shared_accord >= 3)
            stats["family_or_accord3"] += int(same_family or max_shared_accord >= 3)

    print("dataset=%s" % DATASET.name)
    print("total_test_pairs=%d" % total)
    for key in [
        "same_family",
        "same_visual_note",
        "shared_accord_ge1",
        "shared_accord_ge2",
        "shared_accord_ge3",
        "family_or_accord3",
        "sem_similar_to_train",
        "same_global_ref",
    ]:
        print("%s=%d (%.4f)" % (key, stats[key], stats[key] / float(total)))

    print("relation_counts:")
    for name, relation_id in sorted(rel.items(), key=lambda row: row[1]):
        print("  %d\t%s\t%d" % (relation_id, name, relation_counts[relation_id]))


if __name__ == "__main__":
    main()
