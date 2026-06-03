from utility.tf_compat import tf, xavier_initializer
from KGAT import KGAT
import json
import os
import time
import numpy as np


class CRHKGE(KGAT):
    """Cross-Reference Semantic Enrichment over KGAT.

    The class keeps KGAT's CF/KGE training pipeline and adds two thesis
    mechanisms: relation-type attention weights and cross-reference context
    propagation for products connected to a global reference through
    `inspired_by`.
    """

    def _parse_args(self, data_config, pretrain_data, args):
        super(CRHKGE, self)._parse_args(data_config, pretrain_data, args)
        self.model_type = self.model_type.replace('kgat_', 'cr_hkge_', 1)

        self.cr_use_relation_weight = bool(int(getattr(args, 'cr_use_relation_weight', 1)))
        self.cr_use_cross_ref = bool(int(getattr(args, 'cr_use_cross_ref', 1)))
        self.cr_relation_weight_mode = getattr(args, 'cr_relation_weight_mode', 'semantic')
        self.cr_relation_aware_message = bool(int(getattr(args, 'cr_relation_aware_message', 1)))
        self.cr_relation_message_scale = getattr(args, 'cr_relation_message_scale', 'type_count')
        self.cr_cross_ref_alpha = float(getattr(args, 'cr_cross_ref_alpha', 1.0))
        self.cr_model_version = getattr(args, 'cr_model_version', 'cr_hkge_v1')

        self.cr_config = data_config.get('cr_hkge_config', {})
        self.cr_lap_list = data_config.get('lap_list', None)
        self.cr_adj_r_list = data_config.get('adj_r_list', None)
        self.cr_relation_type_ids = self._resolve_relation_type_ids()
        self.cr_relation_type_names = self._resolve_relation_type_names()
        self.cr_expanded_relation_names = self.cr_config.get(
            'expanded_relation_names',
            ['relation_%d' % i for i in range(self.n_relations)]
        )

        self.cr_inspired_expanded_relation_id = int(
            self.cr_config.get('inspired_by_expanded_relation_id', 1))
        self.cr_global_attr_relation_ids = [
            int(r) for r in self.cr_config.get('global_attr_relation_ids', [])
        ]

    def _resolve_relation_type_ids(self):
        if self.cr_relation_weight_mode == 'semantic':
            relation_type_ids = self.cr_config.get('expanded_relation_type_ids')
            if relation_type_ids is not None and len(relation_type_ids) == self.n_relations:
                return np.asarray(relation_type_ids, dtype=np.int32)

        return np.arange(self.n_relations, dtype=np.int32)

    def _resolve_relation_type_names(self):
        if self.cr_relation_weight_mode == 'semantic':
            relation_type_names = self.cr_config.get('relation_type_names')
            if relation_type_names:
                return list(relation_type_names)

        return self.cr_config.get(
            'expanded_relation_names',
            ['relation_%d' % i for i in range(self.n_relations)]
        )

    def _build_weights(self):
        all_weights = super(CRHKGE, self)._build_weights()
        initializer = xavier_initializer()

        self.cr_n_relation_types = len(self.cr_relation_type_names)

        if self.cr_use_relation_weight:
            all_weights['cr_relation_type_logits'] = tf.Variable(
                tf.zeros([self.cr_n_relation_types], dtype=tf.float32),
                name='cr_relation_type_logits')
            self.cr_relation_type_probs = tf.nn.softmax(
                all_weights['cr_relation_type_logits'],
                name='cr_relation_type_probs')
            self.cr_relation_type_multipliers = self.cr_relation_type_probs
            if self.cr_relation_message_scale == 'type_count':
                self.cr_relation_type_message_multipliers = (
                    self.cr_relation_type_probs * float(self.cr_n_relation_types))
            elif self.cr_relation_message_scale == 'probability':
                self.cr_relation_type_message_multipliers = self.cr_relation_type_probs
            else:
                raise ValueError('unsupported cr_relation_message_scale: %s' %
                                 self.cr_relation_message_scale)
        else:
            self.cr_relation_type_probs = None
            self.cr_relation_type_multipliers = None
            self.cr_relation_type_message_multipliers = None

        if self.cr_use_cross_ref:
            # KGAT assigns self.weights after _build_weights returns. CR-HKGE
            # needs relation embeddings while building strict cross-reference
            # attention, so expose the in-progress dictionary early.
            self.weights = all_weights
            for k in range(self.n_layers):
                current_dim = self.weight_size_list[k]
                all_weights['W_cr_%d' % k] = tf.Variable(
                    initializer([current_dim, current_dim]), name='W_cr_%d' % k)
                all_weights['b_cr_%d' % k] = tf.Variable(
                    initializer([1, current_dim]), name='b_cr_%d' % k)

            self._build_cross_ref_tensors()

        return all_weights

    def _build_cross_ref_tensors(self):
        product_global_mat = self.cr_config.get('product_global_mat')
        product_mask = self.cr_config.get('product_mask')
        global_attr_relation_mats = self.cr_config.get('global_attr_relation_mats', [])
        global_attr_heads = np.asarray(
            self.cr_config.get('global_attr_attention_heads', []),
            dtype=np.int32)
        global_attr_relations = np.asarray(
            self.cr_config.get('global_attr_attention_relations', []),
            dtype=np.int32)
        global_attr_tails = np.asarray(
            self.cr_config.get('global_attr_attention_tails', []),
            dtype=np.int32)

        n_nodes = self.n_users + self.n_entities

        if product_global_mat is None:
            import scipy.sparse as sp
            product_global_mat = sp.coo_matrix((n_nodes, n_nodes), dtype=np.float32)

        if product_mask is None:
            product_mask = np.zeros((n_nodes, 1), dtype=np.float32)

        self.cr_product_global_tensor = self._convert_sp_mat_to_sp_tensor(product_global_mat)
        self.cr_product_mask_tensor = tf.constant(product_mask, dtype=tf.float32)
        self.cr_global_attr_relation_tensors = [
            (int(relation_id), self._convert_sp_mat_to_sp_tensor(mat))
            for relation_id, mat in global_attr_relation_mats
        ]
        self.cr_global_attr_attention_tensor = None
        self.cr_global_attr_attention_edge_count = int(len(global_attr_heads))

        if (len(global_attr_heads) > 0 and
                len(global_attr_heads) == len(global_attr_relations) == len(global_attr_tails)):
            self.cr_global_attr_attention_h = tf.constant(global_attr_heads, dtype=tf.int32)
            self.cr_global_attr_attention_r = tf.constant(global_attr_relations, dtype=tf.int32)
            self.cr_global_attr_attention_t = tf.constant(global_attr_tails, dtype=tf.int32)

            indices = np.column_stack((global_attr_heads, global_attr_tails)).astype(np.int64)
            self.cr_global_attr_attention_indices = tf.constant(indices, dtype=tf.int64)
            self.cr_global_attr_attention_shape = np.asarray([n_nodes, n_nodes], dtype=np.int64)
            self.cr_global_attr_attention_tensor = self._create_global_attr_attention_tensor()

    def _create_global_attr_attention_tensor(self):
        scores = self._generate_transE_score(
            self.cr_global_attr_attention_h,
            self.cr_global_attr_attention_t,
            self.cr_global_attr_attention_r)

        attention_input = tf.SparseTensor(
            self.cr_global_attr_attention_indices,
            scores,
            self.cr_global_attr_attention_shape)
        return tf.sparse.softmax(attention_input)

    def _relation_multiplier_for_r(self, r):
        if not self.cr_use_relation_weight:
            return tf.ones_like(tf.cast(r, tf.float32), dtype=tf.float32)

        relation_type_ids = tf.constant(self.cr_relation_type_ids, dtype=tf.int32)
        selected_type_ids = tf.gather(relation_type_ids, r)
        return tf.gather(self.cr_relation_type_multipliers, selected_type_ids)

    def _relation_multiplier_for_expanded_id(self, expanded_relation_id):
        if not self.cr_use_relation_weight:
            return tf.constant(1.0, dtype=tf.float32)

        relation_type_id = int(self.cr_relation_type_ids[int(expanded_relation_id)])
        return tf.gather(self.cr_relation_type_multipliers, relation_type_id)

    def _relation_message_multiplier_for_expanded_id(self, expanded_relation_id):
        if not self.cr_use_relation_weight:
            return tf.constant(1.0, dtype=tf.float32)

        relation_type_id = int(self.cr_relation_type_ids[int(expanded_relation_id)])
        return tf.gather(self.cr_relation_type_message_multipliers, relation_type_id)

    def _relation_multiplier_for_raw_id(self, raw_relation_id):
        expanded_relation_id = int(raw_relation_id) + 1
        return self._relation_multiplier_for_expanded_id(expanded_relation_id)

    def _generate_transE_score(self, h, t, r):
        embeddings = tf.concat([self.weights['user_embed'], self.weights['entity_embed']], axis=0)
        embeddings = tf.expand_dims(embeddings, 1)

        h_e = tf.nn.embedding_lookup(embeddings, h)
        t_e = tf.nn.embedding_lookup(embeddings, t)

        r_e = tf.nn.embedding_lookup(self.weights['relation_embed'], r)
        trans_M = tf.nn.embedding_lookup(self.weights['trans_W'], r)

        h_e = tf.reshape(tf.matmul(h_e, trans_M), [-1, self.kge_dim])
        t_e = tf.reshape(tf.matmul(t_e, trans_M), [-1, self.kge_dim])

        kg_score = tf.reduce_sum(tf.multiply(t_e, tf.tanh(h_e + r_e)), 1)
        relation_multiplier = self._relation_multiplier_for_r(r)
        return kg_score * relation_multiplier

    def _create_bi_interaction_embed(self):
        A = self.A_in
        A_fold_hat = self._split_A_hat(A)
        relation_A_fold_hat = self._build_relation_aware_A_fold_hat()

        ego_embeddings = tf.concat([self.weights['user_embed'], self.weights['entity_embed']], axis=0)
        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):
            if relation_A_fold_hat is not None:
                side_embeddings = self._relation_aware_side_embeddings(
                    ego_embeddings,
                    relation_A_fold_hat)
            else:
                temp_embed = []
                for f in range(self.n_fold):
                    temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

                side_embeddings = tf.concat(temp_embed, 0)

            if self.cr_use_cross_ref:
                side_embeddings = side_embeddings + self._create_cross_reference_context(ego_embeddings, k)

            add_embeddings = ego_embeddings + side_embeddings

            sum_embeddings = tf.nn.leaky_relu(
                tf.matmul(add_embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])

            bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
            bi_embeddings = tf.nn.leaky_relu(
                tf.matmul(bi_embeddings, self.weights['W_bi_%d' % k]) + self.weights['b_bi_%d' % k])

            ego_embeddings = bi_embeddings + sum_embeddings
            ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.mess_dropout[k])

            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)
            all_embeddings += [norm_embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)

        ua_embeddings, ea_embeddings = tf.split(all_embeddings, [self.n_users, self.n_entities], 0)
        return ua_embeddings, ea_embeddings

    def _build_relation_aware_A_fold_hat(self):
        if not self.cr_use_relation_weight:
            return None

        if not self.cr_relation_aware_message:
            return None

        if self.cr_lap_list is None or self.cr_adj_r_list is None:
            return None

        return [
            (int(relation_id), self._split_A_hat(lap.tocsr()))
            for relation_id, lap in zip(self.cr_adj_r_list, self.cr_lap_list)
        ]

    def _relation_aware_side_embeddings(self, ego_embeddings, relation_A_fold_hat):
        relation_messages = []

        for relation_id, A_fold_hat in relation_A_fold_hat:
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            relation_embedding = tf.concat(temp_embed, 0)
            relation_multiplier = self._relation_message_multiplier_for_expanded_id(relation_id)
            relation_messages.append(relation_embedding * relation_multiplier)

        return tf.add_n(relation_messages)

    def _create_cross_reference_context(self, ego_embeddings, layer_id):
        if self.cr_global_attr_attention_tensor is not None:
            attr_context = tf.sparse_tensor_dense_matmul(
                self.cr_global_attr_attention_tensor,
                ego_embeddings)
        else:
            attr_context = tf.zeros_like(ego_embeddings)
            for raw_relation_id, relation_tensor in self.cr_global_attr_relation_tensors:
                relation_context = tf.sparse_tensor_dense_matmul(relation_tensor, ego_embeddings)
                relation_multiplier = self._relation_multiplier_for_raw_id(raw_relation_id)
                attr_context = attr_context + relation_context * relation_multiplier

        global_reference_context = ego_embeddings + attr_context
        product_context = tf.sparse_tensor_dense_matmul(
            self.cr_product_global_tensor,
            global_reference_context)

        transformed_context = tf.nn.leaky_relu(
            tf.matmul(product_context, self.weights['W_cr_%d' % layer_id]) +
            self.weights['b_cr_%d' % layer_id])

        inspired_multiplier = self._relation_multiplier_for_expanded_id(
            self.cr_inspired_expanded_relation_id)

        return (self.cr_cross_ref_alpha * inspired_multiplier *
                transformed_context * self.cr_product_mask_tensor)

    def export_artifacts(self, sess, args, data_generator, final_perf):
        export_feed = {
            self.mess_dropout: [0.] * self.n_layers,
            self.node_dropout: [0.] * self.n_layers,
        }
        user_embeddings, entity_embeddings = sess.run(
            [self.ua_embeddings, self.ea_embeddings],
            feed_dict=export_feed)

        relation_weight_rows = self._relation_weight_rows(sess)

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        artifact_dir = os.path.abspath(os.path.join(
            getattr(args, 'cr_artifact_path', '../artifacts/cr_hkge'),
            args.dataset,
            '%s_%s' % (self.model_type, timestamp)
        ))
        os.makedirs(artifact_dir, exist_ok=True)

        product_meta = self._load_product_metadata(getattr(data_generator, 'path', ''))
        entity_meta = self._load_entity_metadata(getattr(data_generator, 'path', ''))

        self._write_embeddings_tsv(
            os.path.join(artifact_dir, 'product_embeddings.tsv'),
            entity_embeddings[:self.n_items],
            product_meta,
            entity_meta,
            entity_kind='product')

        self._write_embeddings_tsv(
            os.path.join(artifact_dir, 'entity_embeddings.tsv'),
            entity_embeddings,
            product_meta,
            entity_meta,
            entity_kind='entity')

        self._write_relation_weights(
            os.path.join(artifact_dir, 'relation_weights.tsv'),
            relation_weight_rows)

        self._write_kg_paths(
            os.path.join(artifact_dir, 'kg_paths.jsonl'),
            data_generator,
            entity_meta)

        self._write_query_encoder_config(
            os.path.join(artifact_dir, 'query_encoder_config.json'),
            int(entity_embeddings.shape[1]))

        model_config = {
            'model_version': self.cr_model_version,
            'model_type': self.model_type,
            'dataset': args.dataset,
            'n_users': int(self.n_users),
            'n_items': int(self.n_items),
            'n_entities': int(self.n_entities),
            'n_relations_expanded': int(self.n_relations),
            'embedding_dim_final': int(entity_embeddings.shape[1]),
            'user_embedding_dim_final': int(user_embeddings.shape[1]),
            'cr_use_relation_weight': self.cr_use_relation_weight,
            'cr_use_cross_ref': self.cr_use_cross_ref,
            'cr_relation_weight_mode': self.cr_relation_weight_mode,
            'cr_relation_aware_message': self.cr_relation_aware_message,
            'cr_relation_message_scale': self.cr_relation_message_scale,
            'cr_cross_ref_alpha': self.cr_cross_ref_alpha,
            'cr_best_metric': getattr(args, 'cr_best_metric', 'ndcg'),
            'cr_best_k': int(getattr(args, 'cr_best_k', 3)),
            'cr_export_best_checkpoint': bool(int(getattr(args, 'cr_export_best_checkpoint', 1))),
            'cr_cross_ref_attention': 'strict_neighbor_attention',
            'cr_global_attr_attention_edges': int(getattr(self, 'cr_global_attr_attention_edge_count', 0)),
            'enriched_product_count': int(len(self.cr_config.get('enriched_product_ids', []))),
            'global_attr_relation_ids': [int(r) for r in self.cr_global_attr_relation_ids],
            'query_encoder_config': 'query_encoder_config.json',
            'final_performance': final_perf,
        }

        with open(os.path.join(artifact_dir, 'model_config.json'), 'w', encoding='utf-8') as f:
            json.dump(model_config, f, indent=2)

        print('export CR-HKGE artifacts in path: ', artifact_dir)
        return artifact_dir

    def _relation_weight_rows(self, sess):
        if self.cr_use_relation_weight:
            probs, multipliers, message_multipliers = sess.run([
                self.cr_relation_type_probs,
                self.cr_relation_type_multipliers,
                self.cr_relation_type_message_multipliers
            ])
        else:
            probs = np.ones(self.cr_n_relation_types, dtype=np.float32) / float(self.cr_n_relation_types)
            multipliers = np.ones(self.cr_n_relation_types, dtype=np.float32)
            message_multipliers = np.ones(self.cr_n_relation_types, dtype=np.float32)

        rows = []
        for type_id, type_name in enumerate(self.cr_relation_type_names):
            rows.append({
                'relation_type_id': type_id,
                'relation_type_name': type_name,
                'probability': float(probs[type_id]),
                'multiplier': float(multipliers[type_id]),
                'message_multiplier': float(message_multipliers[type_id]),
            })
        return rows

    def _load_product_metadata(self, dataset_path):
        product_file = os.path.join(dataset_path, 'product2id.tsv')
        product_meta = {}
        if not os.path.exists(product_file):
            return product_meta

        with open(product_file, 'r', encoding='utf-8') as f:
            next(f, None)
            for line in f:
                parts = line.rstrip('\n').split('\t')
                if len(parts) >= 3:
                    product_meta[int(parts[0])] = {
                        'old_entity_id': parts[1],
                        'name': parts[2],
                    }
        return product_meta

    def _load_entity_metadata(self, dataset_path):
        entity_file = os.path.join(dataset_path, 'entity2id_typed.tsv')
        entity_meta = {}
        if not os.path.exists(entity_file):
            return entity_meta

        with open(entity_file, 'r', encoding='utf-8') as f:
            next(f, None)
            for line in f:
                parts = line.rstrip('\n').split('\t')
                if len(parts) >= 4:
                    entity_meta[int(parts[0])] = {
                        'old_entity_id': parts[1],
                        'type': parts[2],
                        'name': parts[3],
                    }
        return entity_meta

    def _embedding_to_json(self, embedding):
        return json.dumps(
            [round(float(value), 8) for value in embedding.tolist()],
            separators=(',', ':'))

    def _write_embeddings_tsv(self, path, embeddings, product_meta, entity_meta, entity_kind):
        with open(path, 'w', encoding='utf-8') as f:
            f.write('entity_id\told_entity_id\tentity_type\tentity_name\tembedding_dim\tembedding_json\tmodel_version\n')
            for entity_id, embedding in enumerate(embeddings):
                meta = entity_meta.get(entity_id, {})
                if entity_kind == 'product':
                    product = product_meta.get(entity_id, {})
                    entity_name = product.get('name', meta.get('name', 'entity_%d' % entity_id))
                    old_entity_id = product.get('old_entity_id', meta.get('old_entity_id', ''))
                    entity_type = 'product'
                else:
                    entity_name = meta.get('name', 'entity_%d' % entity_id)
                    old_entity_id = meta.get('old_entity_id', '')
                    entity_type = meta.get('type', '')

                f.write('%d\t%s\t%s\t%s\t%d\t%s\t%s\n' % (
                    entity_id,
                    old_entity_id,
                    entity_type,
                    entity_name,
                    len(embedding),
                    self._embedding_to_json(embedding),
                    self.cr_model_version))

    def _write_relation_weights(self, path, rows):
        with open(path, 'w', encoding='utf-8') as f:
            f.write('relation_type_id\trelation_type_name\tprobability\tmultiplier\tmessage_multiplier\tmodel_version\n')
            for row in rows:
                f.write('%d\t%s\t%.8f\t%.8f\t%.8f\t%s\n' % (
                    row['relation_type_id'],
                    row['relation_type_name'],
                    row['probability'],
                    row['multiplier'],
                    row['message_multiplier'],
                    self.cr_model_version))

    def _write_query_encoder_config(self, path, embedding_dim):
        config = {
            'model_version': self.cr_model_version,
            'embedding_dim': int(embedding_dim),
            'entity_matching': {
                'accords': ['accord', 'global_accord'],
                'family': ['family', 'global_family'],
                'notes': ['note'],
                'visual_notes': ['note'],
                'reference': ['global_ref'],
                'inspired_by': ['global_ref'],
            },
            'field_relation_map': {
                'accords': {
                    'accord': 'has_accord',
                    'global_accord': 'has_global_accord',
                },
                'family': {
                    'family': 'belongs_to_family',
                    'global_family': 'belongs_to_global_family',
                },
                'notes': {
                    'note': 'has_visual_note',
                },
                'visual_notes': {
                    'note': 'has_visual_note',
                },
                'reference': {
                    'global_ref': 'inspired_by',
                },
                'inspired_by': {
                    'global_ref': 'inspired_by',
                },
            },
            'kg_path_matching': {
                'policy': 'relation_compatible',
                'allow_name_match': True,
                'relation_compatible_entity_types': {
                    'has_accord': ['accord'],
                    'has_global_accord': ['global_accord'],
                    'belongs_to_family': ['family'],
                    'belongs_to_global_family': ['global_family'],
                    'has_visual_note': ['note'],
                    'inspired_by': ['global_ref'],
                },
            },
            'retrieval_rerank': {
                'query_aware_rerank': True,
                'candidate_pool': 50,
                'min_matched_paths': 1,
                'match_bonus': 0.05,
            },
            'relation_weights_used': bool(self.cr_use_relation_weight),
            'relation_weight_mode': self.cr_relation_weight_mode,
            'relation_weight_file': 'relation_weights.tsv',
            'product_embedding_file': 'product_embeddings.tsv',
            'entity_embedding_file': 'entity_embeddings.tsv',
            'kg_path_file': 'kg_paths.jsonl',
            'aggregation': 'weighted_mean',
            'normalization': 'l2',
            'score_function': 'cosine',
            'top_k': 3,
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

    def _write_kg_paths(self, path, data_generator, entity_meta):
        relation_id_to_name = self.cr_config.get('relation_id_to_name', {})
        product_paths = {}

        for head, relation, tail in data_generator.kg_data:
            head = int(head)
            relation = int(relation)
            tail = int(tail)
            if head >= self.n_items:
                continue

            head_meta = entity_meta.get(head, {})
            tail_meta = entity_meta.get(tail, {})
            product_paths.setdefault(head, []).append({
                'head_entity_id': head,
                'head_entity_type': head_meta.get('type', 'product'),
                'head_entity_name': head_meta.get('name', 'entity_%d' % head),
                'relation_id': relation,
                'relation_name': relation_id_to_name.get(relation, 'relation_%d' % relation),
                'relation_scope': 'product',
                'tail_entity_id': tail,
                'tail_entity_type': tail_meta.get('type', ''),
                'tail_entity_name': tail_meta.get('name', 'entity_%d' % tail),
            })

        product_to_global_ref = self.cr_config.get('product_to_global_ref', {})
        global_ref_to_attributes = self.cr_config.get('global_ref_to_attributes', {})

        for product_id, global_refs in product_to_global_ref.items():
            product_id = int(product_id)
            for global_ref_id in global_refs:
                global_ref_id = int(global_ref_id)
                attrs = global_ref_to_attributes.get(global_ref_id)
                if attrs is None:
                    attrs = global_ref_to_attributes.get(str(global_ref_id), [])

                head_meta = entity_meta.get(global_ref_id, {})
                for relation, tail in attrs:
                    relation = int(relation)
                    tail = int(tail)
                    tail_meta = entity_meta.get(tail, {})
                    product_paths.setdefault(product_id, []).append({
                        'head_entity_id': global_ref_id,
                        'head_entity_type': head_meta.get('type', 'global_ref'),
                        'head_entity_name': head_meta.get('name', 'entity_%d' % global_ref_id),
                        'relation_id': relation,
                        'relation_name': relation_id_to_name.get(relation, 'relation_%d' % relation),
                        'relation_scope': 'global_reference',
                        'tail_entity_id': tail,
                        'tail_entity_type': tail_meta.get('type', ''),
                        'tail_entity_name': tail_meta.get('name', 'entity_%d' % tail),
                    })

        with open(path, 'w', encoding='utf-8') as f:
            for product_id in sorted(product_paths.keys()):
                entity_meta_row = entity_meta.get(product_id, {})
                row = {
                    'product_id': product_id,
                    'product_name': entity_meta_row.get('name', 'product_%d' % product_id),
                    'model_version': self.cr_model_version,
                    'kg_path': product_paths[product_id],
                }
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
