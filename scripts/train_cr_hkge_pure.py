"""Train blueprint CR-HKGE without user-item interaction training.

This script is intentionally separate from Model/Main.py and KGAT's BPR
training loop. It reads only the fragrance KG and entity metadata, then exports
the same retrieval artifact format consumed by scripts/cr_hkge_retrieve.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


FRAGRANCE_PRIORS = {
    "inspired_by": 1.8,
    "has_accord": 2.0,
    "has_visual_note": 0.8,
    "belongs_to_family": 1.8,
    "sem_similar": 3.0,
    "has_global_accord": 1.2,
    "belongs_to_global_family": 1.2,
}


def read_relation_map(path: Path) -> dict[int, str]:
    rows: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        parts = line.split()
        rows[int(parts[-1])] = " ".join(parts[:-1])
    return rows


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_metadata(dataset: Path):
    entity_meta = {}
    for row in read_tsv(dataset / "entity2id_typed.tsv"):
        entity_meta[int(row["new_id"])] = {
            "old_entity_id": row.get("old_id", ""),
            "type": row.get("entity_type", ""),
            "name": row.get("entity_name", ""),
        }

    product_meta = {}
    for row in read_tsv(dataset / "product2id.tsv"):
        product_meta[int(row["new_product_id"])] = {
            "old_entity_id": row.get("old_entity_id", ""),
            "name": row.get("product_name", ""),
        }

    return entity_meta, product_meta


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    norm[norm <= 0] = 1.0
    return matrix / norm


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def relation_multipliers(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probs = softmax(logits)
    return probs, probs * float(len(probs))


def embedding_json(vector: np.ndarray) -> str:
    return json.dumps([round(float(value), 8) for value in vector.tolist()], separators=(",", ":"))


def initial_relation_logits(relation_names: list[str], strength: float) -> np.ndarray:
    priors = np.asarray(
        [max(FRAGRANCE_PRIORS.get(name, 1.0), 1e-6) for name in relation_names],
        dtype=np.float32,
    )
    return np.log(priors) * float(strength)


def train_transe_with_relation_attention(
    triples: np.ndarray,
    n_entities: int,
    relation_names: list[str],
    embed_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    relation_lr: float,
    margin: float,
    prior_strength: float,
    seed: int,
):
    rng = np.random.default_rng(seed)
    entity = rng.normal(0.0, 0.05, size=(n_entities, embed_dim)).astype(np.float32)
    relation = rng.normal(0.0, 0.05, size=(len(relation_names), embed_dim)).astype(np.float32)
    relation_logits = initial_relation_logits(relation_names, prior_strength).astype(np.float32)

    entity = l2_normalize(entity)
    relation = l2_normalize(relation)
    n_triples = int(len(triples))

    for epoch in range(epochs):
        order = rng.permutation(n_triples)
        total_loss = 0.0
        active_count = 0

        for start in range(0, n_triples, batch_size):
            batch = triples[order[start:start + batch_size]]
            h = batch[:, 0]
            r = batch[:, 1]
            t = batch[:, 2]
            neg_t = rng.integers(0, n_entities, size=len(batch), dtype=np.int32)

            probs, multipliers = relation_multipliers(relation_logits)
            m = multipliers[r][:, None]

            h_e = entity[h]
            r_e = relation[r]
            t_e = entity[t]
            n_e = entity[neg_t]

            pos_vec = h_e + m * r_e - t_e
            neg_vec = h_e + m * r_e - n_e
            pos_dist = np.sum(pos_vec * pos_vec, axis=1)
            neg_dist = np.sum(neg_vec * neg_vec, axis=1)
            losses = margin + pos_dist - neg_dist
            active = losses > 0
            if not np.any(active):
                continue

            scale = 1.0 / float(np.sum(active))
            ah = h[active]
            ar = r[active]
            at = t[active]
            an = neg_t[active]
            am = m[active]
            apos = pos_vec[active]
            aneg = neg_vec[active]
            ar_e = relation[ar]

            grad_h = (2.0 * apos - 2.0 * aneg) * scale
            grad_r = (2.0 * apos - 2.0 * aneg) * am * scale
            grad_t = (-2.0 * apos) * scale
            grad_n = (2.0 * aneg) * scale

            np.add.at(entity, ah, -lr * grad_h)
            np.add.at(relation, ar, -lr * grad_r)
            np.add.at(entity, at, -lr * grad_t)
            np.add.at(entity, an, -lr * grad_n)

            d_loss_dm = (
                2.0 * np.sum(apos * ar_e, axis=1) -
                2.0 * np.sum(aneg * ar_e, axis=1)
            ) * scale
            grad_logits = np.zeros_like(relation_logits)
            for rel_id, grad_m in zip(ar, d_loss_dm):
                grad_logits += grad_m * multipliers[rel_id] * (-probs)
                grad_logits[rel_id] += grad_m * multipliers[rel_id]
            relation_logits -= relation_lr * grad_logits

            total_loss += float(np.sum(losses[active]))
            active_count += int(np.sum(active))

        if (epoch + 1) % 10 == 0 or epoch == 0:
            entity = l2_normalize(entity)
            relation = l2_normalize(relation)
            avg_loss = total_loss / float(max(active_count, 1))
            print("Epoch %d: kg_margin_loss=%.5f, active_triples=%d" %
                  (epoch + 1, avg_loss, active_count))

    probs, multipliers = relation_multipliers(relation_logits)
    return l2_normalize(entity), l2_normalize(relation), probs, multipliers


def build_graph_context(
    entity: np.ndarray,
    triples: np.ndarray,
    relation_names: list[str],
    relation_name_to_id: dict[str, int],
    multipliers: np.ndarray,
    n_items: int,
    n_layers: int,
    cross_ref_alpha: float,
):
    current = entity.copy()
    all_layers = [l2_normalize(current)]

    edges = []
    for h, r, t in triples:
        h = int(h)
        r = int(r)
        t = int(t)
        edges.append((h, t, r))
        edges.append((t, h, r))

    product_to_global = defaultdict(list)
    global_attrs = defaultdict(list)
    inspired_id = relation_name_to_id.get("inspired_by")
    global_attr_ids = {
        relation_name_to_id[name]
        for name in ("has_global_accord", "belongs_to_global_family")
        if name in relation_name_to_id
    }

    for h, r, t in triples:
        h = int(h)
        r = int(r)
        t = int(t)
        if inspired_id is not None and r == inspired_id and h < n_items:
            product_to_global[h].append(t)
        if r in global_attr_ids:
            global_attrs[h].append((r, t))

    for _layer in range(n_layers):
        side = np.zeros_like(current)
        deg = np.zeros((current.shape[0], 1), dtype=np.float32)
        for h, t, r in edges:
            weight = float(multipliers[r])
            side[h] += weight * current[t]
            deg[h, 0] += abs(weight)
        deg[deg <= 0] = 1.0
        side = side / deg

        cross = np.zeros_like(current)
        if inspired_id is not None:
            inspired_weight = float(multipliers[inspired_id])
            for product_id, refs in product_to_global.items():
                contexts = []
                for ref_id in refs:
                    attr_contexts = []
                    for attr_relation, attr_tail in global_attrs.get(ref_id, []):
                        attr_contexts.append(float(multipliers[attr_relation]) * current[attr_tail])
                    if attr_contexts:
                        attr_context = np.mean(np.asarray(attr_contexts), axis=0)
                    else:
                        attr_context = np.zeros(current.shape[1], dtype=np.float32)
                    contexts.append(current[ref_id] + attr_context)
                if contexts:
                    cross[product_id] = (
                        float(cross_ref_alpha) * inspired_weight *
                        np.mean(np.asarray(contexts), axis=0)
                    )

        current = np.tanh(current + side + cross)
        current = l2_normalize(current)
        all_layers.append(current)

    return l2_normalize(np.concatenate(all_layers, axis=1))


def write_embeddings(path: Path, embeddings: np.ndarray, entity_meta, product_meta, model_version: str, product_only: bool):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "entity_id", "old_entity_id", "entity_type", "entity_name",
            "embedding_dim", "embedding_json", "model_version",
        ])
        for entity_id, embedding in enumerate(embeddings):
            meta = entity_meta.get(entity_id, {})
            if product_only:
                product = product_meta.get(entity_id, {})
                entity_type = "product"
                entity_name = product.get("name", meta.get("name", "entity_%d" % entity_id))
                old_entity_id = product.get("old_entity_id", meta.get("old_entity_id", ""))
            else:
                entity_type = meta.get("type", "")
                entity_name = meta.get("name", "entity_%d" % entity_id)
                old_entity_id = meta.get("old_entity_id", "")
            writer.writerow([
                entity_id, old_entity_id, entity_type, entity_name,
                int(len(embedding)), embedding_json(embedding), model_version,
            ])


def write_relation_weights(path: Path, relation_names: list[str], probs: np.ndarray, multipliers: np.ndarray, model_version: str):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "relation_type_id", "relation_type_name", "probability",
            "multiplier", "message_multiplier", "model_version",
        ])
        for relation_id, name in enumerate(relation_names):
            writer.writerow([
                relation_id, name, "%.8f" % probs[relation_id],
                "%.8f" % multipliers[relation_id],
                "%.8f" % multipliers[relation_id], model_version,
            ])


def write_query_encoder_config(path: Path, embedding_dim: int, model_version: str):
    config = {
        "model_version": model_version,
        "embedding_dim": int(embedding_dim),
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
        "kg_path_matching": {
            "policy": "relation_compatible",
            "allow_name_match": True,
            "relation_compatible_entity_types": {
                "has_accord": ["accord"],
                "has_global_accord": ["global_accord"],
                "belongs_to_family": ["family"],
                "belongs_to_global_family": ["global_family"],
                "has_visual_note": ["note"],
                "inspired_by": ["global_ref"],
            },
        },
        "retrieval_rerank": {
            "query_aware_rerank": True,
            "candidate_pool": 50,
            "min_matched_paths": 1,
            "match_bonus": 0.05,
        },
        "relation_weights_used": True,
        "relation_weight_mode": "semantic",
        "relation_weight_file": "relation_weights.tsv",
        "product_embedding_file": "product_embeddings.tsv",
        "entity_embedding_file": "entity_embeddings.tsv",
        "kg_path_file": "kg_paths.jsonl",
        "aggregation": "weighted_mean",
        "normalization": "l2",
        "score_function": "cosine",
        "top_k": 3,
    }
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def write_kg_paths(path: Path, triples: np.ndarray, relation_names: list[str], entity_meta, n_items: int):
    product_paths = defaultdict(list)
    for h, r, t in triples:
        h = int(h)
        r = int(r)
        t = int(t)
        if h >= n_items:
            continue
        head_meta = entity_meta.get(h, {})
        tail_meta = entity_meta.get(t, {})
        product_paths[h].append({
            "head_entity_id": h,
            "head_entity_type": head_meta.get("type", "product"),
            "head_entity_name": head_meta.get("name", "entity_%d" % h),
            "relation_id": r,
            "relation_name": relation_names[r],
            "relation_scope": "product",
            "tail_entity_id": t,
            "tail_entity_type": tail_meta.get("type", ""),
            "tail_entity_name": tail_meta.get("name", "entity_%d" % t),
        })

    inspired_id = relation_names.index("inspired_by") if "inspired_by" in relation_names else None
    global_attr_ids = {
        idx for idx, name in enumerate(relation_names)
        if name in ("has_global_accord", "belongs_to_global_family")
    }
    global_attrs = defaultdict(list)
    for h, r, t in triples:
        if int(r) in global_attr_ids:
            global_attrs[int(h)].append((int(r), int(t)))

    if inspired_id is not None:
        for h, r, t in triples:
            h = int(h)
            r = int(r)
            t = int(t)
            if h >= n_items or r != inspired_id:
                continue
            head_meta = entity_meta.get(t, {})
            for attr_relation, attr_tail in global_attrs.get(t, []):
                tail_meta = entity_meta.get(attr_tail, {})
                product_paths[h].append({
                    "head_entity_id": t,
                    "head_entity_type": head_meta.get("type", "global_ref"),
                    "head_entity_name": head_meta.get("name", "entity_%d" % t),
                    "relation_id": attr_relation,
                    "relation_name": relation_names[attr_relation],
                    "relation_scope": "global_reference",
                    "tail_entity_id": attr_tail,
                    "tail_entity_type": tail_meta.get("type", ""),
                    "tail_entity_name": tail_meta.get("name", "entity_%d" % attr_tail),
                })

    with path.open("w", encoding="utf-8") as f:
        for product_id in sorted(product_paths):
            meta = entity_meta.get(product_id, {})
            row = {
                "product_id": int(product_id),
                "product_name": meta.get("name", "product_%d" % product_id),
                "kg_path": product_paths[product_id],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Train pure interaction-free CR-HKGE artifacts.")
    parser.add_argument("--dataset-path", default="dataset-aromatique-kgat-ready")
    parser.add_argument("--artifact-path", default="artifacts/cr_hkge_pure")
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--relation-lr", type=float, default=0.001)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--prior-strength", type=float, default=1.0)
    parser.add_argument("--cross-ref-alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2019)
    parser.add_argument("--model-version", default="cr_hkge_blueprint_pure_v1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = Path(args.dataset_path)
    relation_id_to_name = read_relation_map(dataset / "relation2id.txt")
    relation_names = [relation_id_to_name[idx] for idx in sorted(relation_id_to_name)]
    relation_name_to_id = {name: idx for idx, name in enumerate(relation_names)}
    entity_meta, product_meta = load_metadata(dataset)

    triples = np.loadtxt(dataset / "kg_final.txt", dtype=np.int32)
    if triples.ndim == 1:
        triples = triples.reshape(1, -1)
    triples = np.unique(triples, axis=0)

    n_entities = max(max(entity_meta.keys()) + 1, int(np.max(triples[:, [0, 2]])) + 1)
    n_items = len(product_meta)

    print("CR-HKGE blueprint pure training")
    print("dataset=%s" % dataset)
    print("interaction_training=none")
    print("train_txt_used=0")
    print("test_txt_used=0")
    print("n_items=%d, n_entities=%d, n_relations=%d, n_triples=%d" %
          (n_items, n_entities, len(relation_names), len(triples)))

    entity, _relation, probs, multipliers = train_transe_with_relation_attention(
        triples=triples,
        n_entities=n_entities,
        relation_names=relation_names,
        embed_dim=args.embed_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        relation_lr=args.relation_lr,
        margin=args.margin,
        prior_strength=args.prior_strength,
        seed=args.seed,
    )

    final_embeddings = build_graph_context(
        entity=entity,
        triples=triples,
        relation_names=relation_names,
        relation_name_to_id=relation_name_to_id,
        multipliers=multipliers,
        n_items=n_items,
        n_layers=args.layers,
        cross_ref_alpha=args.cross_ref_alpha,
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    artifact_dir = Path(args.artifact_path) / dataset.name / ("cr_hkge_blueprint_pure_%s" % timestamp)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    write_embeddings(
        artifact_dir / "product_embeddings.tsv",
        final_embeddings[:n_items],
        entity_meta,
        product_meta,
        args.model_version,
        product_only=True,
    )
    write_embeddings(
        artifact_dir / "entity_embeddings.tsv",
        final_embeddings,
        entity_meta,
        product_meta,
        args.model_version,
        product_only=False,
    )
    write_relation_weights(
        artifact_dir / "relation_weights.tsv",
        relation_names,
        probs,
        multipliers,
        args.model_version,
    )
    write_kg_paths(artifact_dir / "kg_paths.jsonl", triples, relation_names, entity_meta, n_items)
    write_query_encoder_config(artifact_dir / "query_encoder_config.json", final_embeddings.shape[1], args.model_version)

    model_config = {
        "model_version": args.model_version,
        "model_type": "cr_hkge_blueprint_pure",
        "dataset": dataset.name,
        "n_items": int(n_items),
        "n_entities": int(n_entities),
        "n_relations": int(len(relation_names)),
        "n_triples": int(len(triples)),
        "embedding_dim_final": int(final_embeddings.shape[1]),
        "interaction_training": "none",
        "train_txt_used": False,
        "test_txt_used": False,
        "kg_training_objective": "TransE margin ranking with relation-type attention",
        "cr_use_cross_ref": True,
        "cr_cross_ref_relation": "inspired_by",
        "cr_cross_ref_alpha": float(args.cross_ref_alpha),
        "cr_relation_prior_mode": "fragrance",
        "cr_relation_prior_strength": float(args.prior_strength),
        "cr_relation_attention": {
            name: {
                "probability": float(probs[idx]),
                "multiplier": float(multipliers[idx]),
            }
            for idx, name in enumerate(relation_names)
        },
        "query_encoder_config": "query_encoder_config.json",
    }
    (artifact_dir / "model_config.json").write_text(
        json.dumps(model_config, indent=2) + "\n",
        encoding="utf-8",
    )

    print("export CR-HKGE blueprint pure artifacts in path: %s" % artifact_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
