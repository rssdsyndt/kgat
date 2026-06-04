"""Evaluasi checkpoint untuk overall, enriched, dan standard item.

Pembagian subset berasal dari KG Aromatique:
- enriched: produk yang memiliki edge `inspired_by`.
- standard: produk tanpa edge `inspired_by`.

Candidate item tetap seluruh katalog non-training. Yang difilter hanya item
relevan pada test set, sehingga evaluasi menjawab apakah model mampu menaikkan
ranking produk enriched/standard pada setting Top-K yang sama.
"""

from __future__ import annotations

import heapq
import os
import sys
from typing import Dict, Iterable, List, Set

import numpy as np

from utility.tf_compat import tf
from utility.parser import parse_args
from utility.loader_bprmf import BPRMF_loader
from utility.loader_cke import CKE_loader
from utility.loader_cfkg import CFKG_loader
from utility.loader_nfm import NFM_loader
from utility.loader_kgat import KGAT_loader
import utility.metrics as metrics

from BPRMF import BPRMF
from CKE import CKE
from CFKG import CFKG
from NFM import NFM
from KGAT import KGAT
from CRHKGE import CRHKGE


def relation_map(dataset_path: str) -> Dict[str, int]:
    path = os.path.join(dataset_path, "relation2id.txt")
    if not os.path.exists(path):
        raise RuntimeError("relation2id.txt not found in %s" % dataset_path)

    result: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 2:
            result[" ".join(parts[:-1])] = int(parts[-1])
    return result


def enriched_item_ids(data_generator: KGAT_loader) -> Set[int]:
    return enriched_item_ids_from_path(data_generator.path, data_generator.n_items)


def enriched_item_ids_from_path(dataset_path: str, n_items: int) -> Set[int]:
    # Fase identifikasi produk enriched:
    # produk disebut enriched jika memiliki relasi inspired_by ke global
    # reference. Ini menjadi dasar analisis dampak cross-reference.
    rel_map = relation_map(dataset_path)
    if "inspired_by" not in rel_map:
        raise RuntimeError("relation 'inspired_by' not found in relation2id.txt")

    inspired_by_id = rel_map["inspired_by"]
    enriched = set()
    kg_path = os.path.join(dataset_path, "kg_final.txt")
    if not os.path.exists(kg_path):
        raise RuntimeError("kg_final.txt not found in %s" % dataset_path)

    kg_data = np.loadtxt(kg_path, dtype=np.int32)
    if kg_data.ndim == 1:
        kg_data = np.asarray([kg_data])

    for head, relation, _tail in kg_data:
        head = int(head)
        relation = int(relation)
        if head < n_items and relation == inspired_by_id:
            enriched.add(head)
    return enriched


def build_loader(args):
    # Loader dipilih sesuai model. Dengan ini evaluator yang sama bisa dipakai
    # untuk BPRMF, CKE, NFM, CFKG, KGAT, dan CR-HKGE pada split yang sama.
    path = args.data_path + args.dataset
    if args.model_type == "bprmf":
        return BPRMF_loader(args=args, path=path)
    if args.model_type == "cke":
        return CKE_loader(args=args, path=path)
    if args.model_type == "cfkg":
        return CFKG_loader(args=args, path=path)
    if args.model_type in ["fm", "nfm"]:
        return NFM_loader(args=args, path=path)
    if args.model_type in ["kgat", "cr_hkge"]:
        return KGAT_loader(args=args, path=path)
    raise NotImplementedError("unsupported model_type for subset evaluation: %s" % args.model_type)


def build_config(args, data_generator):
    # Konfigurasi model disusun sesuai kebutuhan tiap baseline.
    # KGAT/CR-HKGE/CFKG butuh struktur KG lengkap, sedangkan BPRMF hanya butuh
    # jumlah user/profile dan item.
    config = {
        "n_users": data_generator.n_users,
        "n_items": data_generator.n_items,
    }

    if hasattr(data_generator, "n_entities"):
        config["n_entities"] = data_generator.n_entities
    if hasattr(data_generator, "n_relations"):
        config["n_relations"] = data_generator.n_relations

    if args.model_type in ["kgat", "cr_hkge", "cfkg"]:
        config["A_in"] = sum(data_generator.lap_list)
        config["all_h_list"] = data_generator.all_h_list
        config["all_r_list"] = data_generator.all_r_list
        config["all_t_list"] = data_generator.all_t_list
        config["all_v_list"] = data_generator.all_v_list

    if args.model_type == "cr_hkge":
        config["cr_hkge_config"] = data_generator.get_cr_hkge_config()
        config["lap_list"] = data_generator.lap_list
        config["adj_r_list"] = data_generator.adj_r_list

    return config


def build_model(args, config):
    if args.model_type == "bprmf":
        return BPRMF(data_config=config, pretrain_data=None, args=args)
    if args.model_type == "cke":
        return CKE(data_config=config, pretrain_data=None, args=args)
    if args.model_type == "cfkg":
        return CFKG(data_config=config, pretrain_data=None, args=args)
    if args.model_type in ["fm", "nfm"]:
        return NFM(data_config=config, pretrain_data=None, args=args)
    if args.model_type == "kgat":
        return KGAT(data_config=config, pretrain_data=None, args=args)
    if args.model_type == "cr_hkge":
        return CRHKGE(data_config=config, pretrain_data=None, args=args)
    raise NotImplementedError("unsupported model_type for subset evaluation: %s" % args.model_type)


def weights_path(args, model) -> str:
    reg_key = "-".join([str(r) for r in eval(args.regs)])
    if args.model_type in ["bprmf", "cke", "fm", "cfkg"]:
        return "%sweights/%s/%s/l%s_r%s" % (
            args.weights_path,
            args.dataset,
            model.model_type,
            str(args.lr),
            reg_key,
        )

    layer = "-".join([str(l) for l in eval(args.layer_size)])
    return "%sweights/%s/%s/%s/l%s_r%s" % (
        args.weights_path,
        args.dataset,
        model.model_type,
        layer,
        str(args.lr),
        reg_key,
    )


def rank_for_user(rating: np.ndarray, training_items: Iterable[int], item_num: int, max_k: int) -> List[int]:
    candidate_items = list(set(range(item_num)) - set(training_items))
    item_score = {item: rating[item] for item in candidate_items}
    return heapq.nlargest(max_k, item_score, key=item_score.get)


def metric_row(ranked_items: List[int], positives: List[int], ks: List[int]):
    relevance = [1 if item in positives else 0 for item in ranked_items]
    return {
        "recall": np.asarray([metrics.recall_at_k(relevance, k, len(positives)) for k in ks]),
        "precision": np.asarray([metrics.precision_at_k(relevance, k) for k in ks]),
        "hit": np.asarray([metrics.hit_at_k(relevance, k) for k in ks]),
        "ndcg": np.asarray([metrics.ndcg_at_k(relevance, k) for k in ks]),
    }


def evaluate_subset(sess, model, data_generator, subset_items: Set[int] | None, ks: List[int]):
    # Evaluasi Top-K:
    # - subset_items=None berarti overall.
    # - subset_items=enriched/standard berarti hanya item relevan pada subset
    #   tersebut yang dihitung sebagai ground truth.
    result = {
        "recall": np.zeros(len(ks), dtype=np.float64),
        "precision": np.zeros(len(ks), dtype=np.float64),
        "hit": np.zeros(len(ks), dtype=np.float64),
        "ndcg": np.zeros(len(ks), dtype=np.float64),
    }
    max_k = max(ks)
    users = []
    test_pos_count = 0

    for user, positives in data_generator.test_user_dict.items():
        if subset_items is None:
            target_positives = list(positives)
        else:
            target_positives = [item for item in positives if item in subset_items]
        if not target_positives:
            continue
        users.append((user, target_positives))
        test_pos_count += len(target_positives)

    if not users:
        return result, 0, 0

    item_batch = range(data_generator.n_items)
    for user, target_positives in users:
        # Semua item kandidat diberi skor, lalu item yang sudah ada di train
        # dikeluarkan agar evaluasi fokus pada kemampuan ranking item test.
        feed_dict = data_generator.generate_test_feed_dict(
            model=model,
            user_batch=[user],
            item_batch=item_batch,
            drop_flag=False,
        )
        rating = model.eval(sess, feed_dict=feed_dict).reshape((-1, data_generator.n_items))[0]
        training_items = data_generator.train_user_dict.get(user, [])
        ranked_items = rank_for_user(rating, training_items, data_generator.n_items, max_k)
        row = metric_row(ranked_items, target_positives, ks)
        for key in result:
            result[key] += row[key] / float(len(users))

    return result, len(users), test_pos_count


def format_metric(values: np.ndarray) -> str:
    return "[" + "\t".join(["%.5f" % value for value in values]) + "]"


def print_result(name: str, result, n_users: int, n_pos: int):
    print("%s: n_users=%d, n_test_pos=%d" % (name, n_users, n_pos))
    print(
        "  recall=%s, precision=%s, hit=%s, ndcg=%s"
        % (
            format_metric(result["recall"]),
            format_metric(result["precision"]),
            format_metric(result["hit"]),
            format_metric(result["ndcg"]),
        )
    )


def main():
    tf.set_random_seed(2019)
    np.random.seed(2019)
    args = parse_args()

    data_generator = build_loader(args)
    config = build_config(args, data_generator)
    model = build_model(args, config)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    restore_path = weights_path(args, model)
    ckpt = tf.train.get_checkpoint_state(restore_path)
    if not ckpt or not ckpt.model_checkpoint_path:
        raise RuntimeError(
            "checkpoint not found in %s; run training with --save_flag 1 first, "
            "or restore the Model/weights directory in this runtime." % restore_path
        )
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("loaded checkpoint: %s" % ckpt.model_checkpoint_path)

    ks = eval(args.Ks)
    subset_data_path = getattr(args, "cr_subset_data_path", "")
    subset_dataset = getattr(args, "cr_subset_dataset", "")
    if subset_dataset:
        subset_base = subset_data_path if subset_data_path else args.data_path
        subset_path = subset_base + subset_dataset
        enriched = enriched_item_ids_from_path(subset_path, data_generator.n_items)
        print("subset source: %s" % subset_path)
    else:
        enriched = enriched_item_ids(data_generator)

    # Standard adalah komplemen dari enriched. Ini penting untuk menjelaskan
    # apakah cross-reference membantu produk yang punya inspired_by tanpa
    # merusak performa produk biasa.
    standard = set(range(data_generator.n_items)) - enriched

    print("subset definition: enriched=inspired_by products, standard=non-inspired_by products")
    print("n_items=%d, enriched_items=%d, standard_items=%d" %
          (data_generator.n_items, len(enriched), len(standard)))
    print("Ks=%s" % ks)

    overall_result, overall_users, overall_pos = evaluate_subset(
        sess, model, data_generator, None, ks)
    enriched_result, enriched_users, enriched_pos = evaluate_subset(
        sess, model, data_generator, enriched, ks)
    standard_result, standard_users, standard_pos = evaluate_subset(
        sess, model, data_generator, standard, ks)

    print_result("overall", overall_result, overall_users, overall_pos)
    print_result("enriched", enriched_result, enriched_users, enriched_pos)
    print_result("standard", standard_result, standard_users, standard_pos)


if __name__ == "__main__":
    sys.exit(main())
