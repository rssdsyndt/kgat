'''
Created on Dec 18, 2018
Tensorflow Implementation of Knowledge Graph Attention Network (KGAT) model in:
Wang Xiang et al. KGAT: Knowledge Graph Attention Network for Recommendation. In KDD 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import numpy as np
from utility.load_data import Data
from time import time
import scipy.sparse as sp
import random as rd
import collections
import os

class KGAT_loader(Data):
    def __init__(self, args, path):
        super().__init__(args, path)

        # generate the sparse adjacency matrices for user-item interaction & relational kg data.
        self.adj_list, self.adj_r_list = self._get_relational_adj_list()

        # generate the sparse laplacian matrices.
        self.lap_list = self._get_relational_lap_list()

        # generate the triples dictionary, key is 'head', value is '(tail, relation)'.
        self.all_kg_dict = self._get_all_kg_dict()

        self.all_h_list, self.all_r_list, self.all_t_list, self.all_v_list = self._get_all_kg_data()

        # CR-HKGE uses the raw Aromatique KG semantics in addition to KGAT's
        # expanded relation IDs. Keep vanilla KGAT's loader path unchanged.
        if getattr(args, 'model_type', '') == 'cr_hkge':
            self.cr_hkge_data = self._build_cr_hkge_data()
        else:
            self.cr_hkge_data = {}


    def _get_relational_adj_list(self):
        t1 = time()
        adj_mat_list = []
        adj_r_list = []
        raw_n_relations = self.n_relations
        self.n_raw_relations = raw_n_relations

        def _np_mat2sp_adj(np_mat, row_pre, col_pre):
            n_all = self.n_users + self.n_entities
            # single-direction
            a_rows = np_mat[:, 0] + row_pre
            a_cols = np_mat[:, 1] + col_pre
            a_vals = [1.] * len(a_rows)

            b_rows = a_cols
            b_cols = a_rows
            b_vals = [1.] * len(b_rows)

            a_adj = sp.coo_matrix((a_vals, (a_rows, a_cols)), shape=(n_all, n_all))
            b_adj = sp.coo_matrix((b_vals, (b_rows, b_cols)), shape=(n_all, n_all))

            return a_adj, b_adj

        R, R_inv = _np_mat2sp_adj(self.train_data, row_pre=0, col_pre=self.n_users)
        adj_mat_list.append(R)
        adj_r_list.append(0)

        adj_mat_list.append(R_inv)
        adj_r_list.append(raw_n_relations + 1)
        print('\tconvert ratings into adj mat done.')

        for r_id in self.relation_dict.keys():
            K, K_inv = _np_mat2sp_adj(np.array(self.relation_dict[r_id]), row_pre=self.n_users, col_pre=self.n_users)
            adj_mat_list.append(K)
            adj_r_list.append(r_id + 1)

            adj_mat_list.append(K_inv)
            adj_r_list.append(r_id + 2 + raw_n_relations)
        print('\tconvert %d relational triples into adj mat done. @%.4fs' %(len(adj_mat_list), time()-t1))

        self.n_relations = len(adj_r_list)
        # print('\tadj relation list is', adj_r_list)

        return adj_mat_list, adj_r_list

    def _get_relational_lap_list(self):
        def _bi_norm_lap(adj):
            rowsum = np.array(adj.sum(1))

            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
            return bi_lap.tocoo()

        def _si_norm_lap(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()

        if self.args.adj_type == 'bi':
            lap_list = [_bi_norm_lap(adj) for adj in self.adj_list]
            print('\tgenerate bi-normalized adjacency matrix.')
        else:
            lap_list = [_si_norm_lap(adj) for adj in self.adj_list]
            print('\tgenerate si-normalized adjacency matrix.')
        return lap_list

    def _get_all_kg_dict(self):
        all_kg_dict = collections.defaultdict(list)
        for l_id, lap in enumerate(self.lap_list):

            rows = lap.row
            cols = lap.col

            for i_id in range(len(rows)):
                head = rows[i_id]
                tail = cols[i_id]
                relation = self.adj_r_list[l_id]

                all_kg_dict[head].append((tail, relation))
        return all_kg_dict

    def _get_all_kg_data(self):
        def _reorder_list(org_list, order):
            new_list = np.array(org_list)
            new_list = new_list[order]
            return new_list

        all_h_list, all_t_list, all_r_list = [], [], []
        all_v_list = []

        for l_id, lap in enumerate(self.lap_list):
            all_h_list += list(lap.row)
            all_t_list += list(lap.col)
            all_v_list += list(lap.data)
            all_r_list += [self.adj_r_list[l_id]] * len(lap.row)

        assert len(all_h_list) == sum([len(lap.data) for lap in self.lap_list])

        # resort the all_h/t/r/v_list,
        # ... since tensorflow.sparse.softmax requires indices sorted in the canonical lexicographic order
        print('\treordering indices...')
        org_h_dict = dict()

        for idx, h in enumerate(all_h_list):
            if h not in org_h_dict.keys():
                org_h_dict[h] = [[],[],[]]

            org_h_dict[h][0].append(all_t_list[idx])
            org_h_dict[h][1].append(all_r_list[idx])
            org_h_dict[h][2].append(all_v_list[idx])
        print('\treorganize all kg data done.')

        sorted_h_dict = dict()
        for h in org_h_dict.keys():
            org_t_list, org_r_list, org_v_list = org_h_dict[h]
            sort_t_list = np.array(org_t_list)
            sort_order = np.argsort(sort_t_list)

            sort_t_list = _reorder_list(org_t_list, sort_order)
            sort_r_list = _reorder_list(org_r_list, sort_order)
            sort_v_list = _reorder_list(org_v_list, sort_order)

            sorted_h_dict[h] = [sort_t_list, sort_r_list, sort_v_list]
        print('\tsort meta-data done.')

        od = collections.OrderedDict(sorted(sorted_h_dict.items()))
        new_h_list, new_t_list, new_r_list, new_v_list = [], [], [], []

        for h, vals in od.items():
            new_h_list += [h] * len(vals[0])
            new_t_list += list(vals[0])
            new_r_list += list(vals[1])
            new_v_list += list(vals[2])


        assert sum(new_h_list) == sum(all_h_list)
        assert sum(new_t_list) == sum(all_t_list)
        assert sum(new_r_list) == sum(all_r_list)
        # try:
        #     assert sum(new_v_list) == sum(all_v_list)
        # except Exception:
        #     print(sum(new_v_list), '\n')
        #     print(sum(all_v_list), '\n')
        print('\tsort all data done.')


        return new_h_list, new_r_list, new_t_list, new_v_list

    def _load_relation_id_to_name(self):
        relation_file = os.path.join(self.path, 'relation2id.txt')
        relation_id_to_name = {}

        if os.path.exists(relation_file):
            with open(relation_file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]

            for line in lines[1:]:
                parts = line.split()
                if len(parts) >= 2:
                    relation_name = ' '.join(parts[:-1])
                    relation_id = int(parts[-1])
                    relation_id_to_name[relation_id] = relation_name

        for relation_id in range(self.n_raw_relations):
            relation_id_to_name.setdefault(relation_id, 'relation_%d' % relation_id)

        return relation_id_to_name

    def _row_normalize_sparse(self, rows, cols, n_rows, n_cols):
        if len(rows) == 0:
            return sp.coo_matrix((n_rows, n_cols), dtype=np.float32)

        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        vals = np.ones(len(rows), dtype=np.float32)
        mat = sp.coo_matrix((vals, (rows, cols)), shape=(n_rows, n_cols), dtype=np.float32)
        mat.sum_duplicates()

        mat = mat.tocoo()
        row_sum = np.asarray(mat.sum(1)).flatten()
        norm_vals = mat.data / row_sum[mat.row]
        return sp.coo_matrix((norm_vals, (mat.row, mat.col)), shape=(n_rows, n_cols), dtype=np.float32)

    def _build_cr_hkge_data(self):
        # Fase input CR-HKGE:
        # fungsi ini hanya dipanggil ketika model_type=cr_hkge.
        # Tujuannya menyiapkan metadata tambahan yang tidak ada di KGAT asli,
        # terutama mapping relasi fragrance dan struktur cross-reference.
        relation_id_to_name = self._load_relation_id_to_name()
        n_nodes = self.n_users + self.n_entities

        # KGAT memperluas relasi menjadi forward dan inverse relation.
        # Bagian ini memberi nama yang mudah dibaca untuk relation id hasil
        # ekspansi tersebut, sehingga relation attention CR-HKGE bisa dijelaskan
        # dalam tipe semantik seperti inspired_by atau has_accord.
        expanded_relation_names = ['unknown_relation_%d' % i for i in range(self.n_relations)]
        expanded_relation_names[0] = 'user_interacts_item'
        inverse_interaction_id = self.n_raw_relations + 1
        if inverse_interaction_id < len(expanded_relation_names):
            expanded_relation_names[inverse_interaction_id] = 'item_interacted_by_user'

        relation_type_names = ['interaction']
        for raw_relation_id in range(self.n_raw_relations):
            relation_type_names.append(relation_id_to_name[raw_relation_id])

            forward_id = raw_relation_id + 1
            inverse_id = raw_relation_id + 2 + self.n_raw_relations
            if forward_id < len(expanded_relation_names):
                expanded_relation_names[forward_id] = relation_id_to_name[raw_relation_id]
            if inverse_id < len(expanded_relation_names):
                expanded_relation_names[inverse_id] = 'inverse_%s' % relation_id_to_name[raw_relation_id]

        # Mapping ini mengikat forward dan inverse relation ke tipe relasi
        # semantik yang sama. Contoh: has_accord dan inverse_has_accord sama-sama
        # memakai bobot relation-type attention "has_accord".
        expanded_relation_type_ids = np.zeros(self.n_relations, dtype=np.int32)
        expanded_relation_type_ids[0] = 0
        if inverse_interaction_id < len(expanded_relation_type_ids):
            expanded_relation_type_ids[inverse_interaction_id] = 0

        for raw_relation_id in range(self.n_raw_relations):
            relation_type_id = raw_relation_id + 1
            forward_id = raw_relation_id + 1
            inverse_id = raw_relation_id + 2 + self.n_raw_relations
            if forward_id < len(expanded_relation_type_ids):
                expanded_relation_type_ids[forward_id] = relation_type_id
            if inverse_id < len(expanded_relation_type_ids):
                expanded_relation_type_ids[inverse_id] = relation_type_id

        name_to_relation_id = {name: relation_id for relation_id, name in relation_id_to_name.items()}
        inspired_by_id = name_to_relation_id.get('inspired_by', 0)
        global_attr_relation_ids = [
            relation_id for relation_id, name in relation_id_to_name.items()
            if name in ['has_global_accord', 'belongs_to_global_family']
        ]

        # Struktur cross-reference yang disiapkan:
        # - product_to_global_ref: produk lokal -> parfum global reference.
        # - global_ref_to_attributes: global reference -> global accord/family.
        # - enriched_product_ids: produk yang punya inspired_by.
        product_global_rows, product_global_cols = [], []
        global_attr_edges = collections.defaultdict(lambda: ([], []))
        global_attr_attention_edges = []
        product_to_global_ref = collections.defaultdict(list)
        global_ref_to_attributes = collections.defaultdict(list)
        enriched_product_ids = set()

        for head, relation, tail in self.kg_data:
            head = int(head)
            relation = int(relation)
            tail = int(tail)

            if relation == inspired_by_id and head < self.n_items:
                # Edge inspired_by adalah inti novelty cross-reference.
                # Node produk dan node global reference digeser dengan n_users
                # karena KGAT menyimpan user node sebelum entity node.
                product_node = self.n_users + head
                global_ref_node = self.n_users + tail
                product_global_rows.append(product_node)
                product_global_cols.append(global_ref_node)
                product_to_global_ref[head].append(tail)
                enriched_product_ids.add(head)

            if relation in global_attr_relation_ids:
                # Atribut global reference ini dipakai untuk memperkaya konteks
                # parfum global sebelum dialirkan ke produk lokal.
                head_node = self.n_users + head
                tail_node = self.n_users + tail
                rows, cols = global_attr_edges[relation]
                rows.append(head_node)
                cols.append(tail_node)
                global_attr_attention_edges.append(
                    (head_node, relation + 1, tail_node))
                global_ref_to_attributes[head].append((relation, tail))

        product_global_mat = self._row_normalize_sparse(
            product_global_rows, product_global_cols, n_nodes, n_nodes)

        # Setiap global attribute relation dibuat sebagai sparse matrix
        # terpisah agar CR-HKGE dapat memberi bobot berbeda per tipe relasi.
        global_attr_relation_mats = []
        for relation_id in sorted(global_attr_edges.keys()):
            rows, cols = global_attr_edges[relation_id]
            mat = self._row_normalize_sparse(rows, cols, n_nodes, n_nodes)
            global_attr_relation_mats.append((relation_id, mat))

        product_mask = np.zeros((n_nodes, 1), dtype=np.float32)
        for product_id in enriched_product_ids:
            product_mask[self.n_users + product_id, 0] = 1.0

        # Tensor attention global attribute dipakai oleh CRHKGE.py untuk
        # menghitung atribut global reference yang paling relevan.
        global_attr_attention_edges = sorted(
            set(global_attr_attention_edges),
            key=lambda edge: (edge[0], edge[2], edge[1]))
        if global_attr_attention_edges:
            global_attr_attention_heads = np.asarray(
                [edge[0] for edge in global_attr_attention_edges],
                dtype=np.int32)
            global_attr_attention_relations = np.asarray(
                [edge[1] for edge in global_attr_attention_edges],
                dtype=np.int32)
            global_attr_attention_tails = np.asarray(
                [edge[2] for edge in global_attr_attention_edges],
                dtype=np.int32)
        else:
            global_attr_attention_heads = np.asarray([], dtype=np.int32)
            global_attr_attention_relations = np.asarray([], dtype=np.int32)
            global_attr_attention_tails = np.asarray([], dtype=np.int32)

        print('\tCR-HKGE metadata: enriched_products=%d, product_global_edges=%d, global_attr_relations=%d.' %
              (len(enriched_product_ids), len(product_global_rows), len(global_attr_relation_mats)))

        return {
            'raw_n_relations': self.n_raw_relations,
            'relation_id_to_name': relation_id_to_name,
            'expanded_relation_names': expanded_relation_names,
            'expanded_relation_type_ids': expanded_relation_type_ids,
            'relation_type_names': relation_type_names,
            'inspired_by_raw_relation_id': inspired_by_id,
            'inspired_by_expanded_relation_id': inspired_by_id + 1,
            'global_attr_relation_ids': global_attr_relation_ids,
            'product_global_mat': product_global_mat,
            'global_attr_relation_mats': global_attr_relation_mats,
            'global_attr_attention_heads': global_attr_attention_heads,
            'global_attr_attention_relations': global_attr_attention_relations,
            'global_attr_attention_tails': global_attr_attention_tails,
            'product_mask': product_mask,
            'enriched_product_ids': sorted(enriched_product_ids),
            'product_to_global_ref': {k: sorted(v) for k, v in product_to_global_ref.items()},
            'global_ref_to_attributes': {k: v for k, v in global_ref_to_attributes.items()},
        }

    def get_cr_hkge_config(self):
        return self.cr_hkge_data

    def _generate_train_A_batch(self):
        exist_heads = list(self.all_kg_dict.keys())

        if self.batch_size_kg <= len(exist_heads):
            heads = rd.sample(exist_heads, self.batch_size_kg)
        else:
            heads = [rd.choice(exist_heads) for _ in range(self.batch_size_kg)]

        def sample_pos_triples_for_h(h, num):
            pos_triples = self.all_kg_dict[h]
            n_pos_triples = len(pos_triples)

            pos_rs, pos_ts = [], []
            while True:
                if len(pos_rs) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_triples, size=1)[0]

                t = pos_triples[pos_id][0]
                r = pos_triples[pos_id][1]

                if r not in pos_rs and t not in pos_ts:
                    pos_rs.append(r)
                    pos_ts.append(t)
            return pos_rs, pos_ts

        def sample_neg_triples_for_h(h, r, num):
            neg_ts = []
            while True:
                if len(neg_ts) == num: break

                t = np.random.randint(low=0, high=self.n_users + self.n_entities, size=1)[0]
                if (t, r) not in self.all_kg_dict[h] and t not in neg_ts:
                    neg_ts.append(t)
            return neg_ts

        pos_r_batch, pos_t_batch, neg_t_batch = [], [], []

        for h in heads:
            pos_rs, pos_ts = sample_pos_triples_for_h(h, 1)
            pos_r_batch += pos_rs
            pos_t_batch += pos_ts

            neg_ts = sample_neg_triples_for_h(h, pos_rs[0], 1)
            neg_t_batch += neg_ts

        return heads, pos_r_batch, pos_t_batch, neg_t_batch

    def generate_train_batch(self):
        users, pos_items, neg_items = self._generate_train_cf_batch()

        batch_data = {}
        batch_data['users'] = users
        batch_data['pos_items'] = pos_items
        batch_data['neg_items'] = neg_items

        return batch_data

    def generate_train_feed_dict(self, model, batch_data):
        feed_dict = {
            model.users: batch_data['users'],
            model.pos_items: batch_data['pos_items'],
            model.neg_items: batch_data['neg_items'],

            model.mess_dropout: eval(self.args.mess_dropout),
            model.node_dropout: eval(self.args.node_dropout),
        }

        return feed_dict

    def generate_train_A_batch(self):
        heads, relations, pos_tails, neg_tails = self._generate_train_A_batch()

        batch_data = {}

        batch_data['heads'] = heads
        batch_data['relations'] = relations
        batch_data['pos_tails'] = pos_tails
        batch_data['neg_tails'] = neg_tails
        return batch_data

    def generate_train_A_feed_dict(self, model, batch_data):
        feed_dict = {
            model.h: batch_data['heads'],
            model.r: batch_data['relations'],
            model.pos_t: batch_data['pos_tails'],
            model.neg_t: batch_data['neg_tails'],

        }

        return feed_dict


    def generate_test_feed_dict(self, model, user_batch, item_batch, drop_flag=True):

        feed_dict ={
            model.users: user_batch,
            model.pos_items: item_batch,
            model.mess_dropout: [0.] * len(eval(self.args.layer_size)),
            model.node_dropout: [0.] * len(eval(self.args.layer_size)),

        }

        return feed_dict

