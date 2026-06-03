'''
Created on Dec 18, 2018
Tensorflow Implementation of Knowledge Graph Attention Network (KGAT) model in:
Wang Xiang et al. KGAT: Knowledge Graph Attention Network for Recommendation. In KDD 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run KGAT.")
    parser.add_argument('--weights_path', nargs='?', default='',
                        help='Store model path.')
    parser.add_argument('--data_path', nargs='?', default='../Data/',
                        help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default='',
                        help='Project path.')

    parser.add_argument('--dataset', nargs='?', default='yelp2018',
                        help='Choose a dataset from {yelp2018, last-fm, amazon-book}')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of epoch.')

    parser.add_argument('--embed_size', type=int, default=64,
                        help='CF Embedding size.')
    parser.add_argument('--kge_size', type=int, default=64,
                        help='KG Embedding size.')
    parser.add_argument('--layer_size', nargs='?', default='[64]',
                        help='Output sizes of every layer')

    parser.add_argument('--batch_size', type=int, default=1024,
                        help='CF batch size.')
    parser.add_argument('--batch_size_kg', type=int, default=2048,
                        help='KG batch size.')

    parser.add_argument('--regs', nargs='?', default='[1e-5,1e-5,1e-2]',
                        help='Regularization for user and item embeddings.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')

    parser.add_argument('--model_type', nargs='?', default='kgat',
                        help='Specify a model type from {kgat, cr_hkge, bprmf, fm, nfm, cke, cfkg}.')
    parser.add_argument('--adj_type', nargs='?', default='si',
                        help='Specify the type of the adjacency (laplacian) matrix from {bi, si}.')
    parser.add_argument('--alg_type', nargs='?', default='ngcf',
                        help='Specify the type of the graph convolutional layer from {bi, gcn, graphsage}.')
    parser.add_argument('--adj_uni_type', nargs='?', default='sum',
                        help='Specify a loss type (uni, sum).')

    parser.add_argument('--gpu_id', type=int, default=0,
                        help='0 for NAIS_prod, 1 for NAIS_concat')

    parser.add_argument('--node_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]',
                        help='Output sizes of every layer')

    parser.add_argument('--save_flag', type=int, default=0,
                        help='0: Disable model saver, 1: Activate model saver')

    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    parser.add_argument('--report', type=int, default=0,
                        help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')

    parser.add_argument('--use_att', type=bool, default=True,
                        help='whether using attention mechanism')
    parser.add_argument('--use_kge', type=bool, default=True,
                        help='whether using knowledge graph embedding')
    
    parser.add_argument('--l1_flag', type=bool, default=True,
                        help='Flase: using the L2 norm, True: using the L1 norm.')

    parser.add_argument('--cr_use_relation_weight', type=int, default=1,
                        help='CR-HKGE only. 1: use relation-type specific attention weights, 0: disable.')
    parser.add_argument('--cr_use_cross_ref', type=int, default=1,
                        help='CR-HKGE only. 1: use cross-reference propagation, 0: disable.')
    parser.add_argument('--cr_relation_weight_mode', nargs='?', default='semantic',
                        help='CR-HKGE only. semantic: tie forward/inverse KG relations, expanded: one weight per expanded relation.')
    parser.add_argument('--cr_relation_prior_mode', nargs='?', default='none',
                        help='CR-HKGE only. none: zero-init relation logits; fragrance: initialize relation logits with fragrance-domain priors.')
    parser.add_argument('--cr_relation_prior_strength', type=float, default=1.0,
                        help='CR-HKGE only. Multiplier for relation prior logits when --cr_relation_prior_mode is enabled.')
    parser.add_argument('--cr_relation_attention_scale', nargs='?', default='type_count',
                        help='CR-HKGE only. probability: use softmax lambda directly in attention scores; type_count: multiply by number of relation types so uniform lambda preserves KGAT score scale.')
    parser.add_argument('--cr_relation_aware_message', type=int, default=0,
                        help='CR-HKGE only. 1: apply relation weights to neighborhood message propagation, 0: keep KGAT propagation strict.')
    parser.add_argument('--cr_relation_message_scale', nargs='?', default='type_count',
                        help='CR-HKGE only. probability: use softmax lambda directly for messages; type_count: multiply by number of relation types to preserve KGAT message scale.')
    parser.add_argument('--cr_cross_ref_alpha', type=float, default=1.0,
                        help='CR-HKGE only. Scalar multiplier for cross-reference context.')
    parser.add_argument('--cr_cross_ref_bi_interaction', type=int, default=0,
                        help='CR-HKGE only. 1: inject cross-reference context into KGAT bi-interaction branch, 0: inject only into additive branch.')
    parser.add_argument('--cr_cross_ref_gate', type=int, default=0,
                        help='CR-HKGE only. 1: use a trainable scalar gate for cross-reference context.')
    parser.add_argument('--cr_cross_ref_gate_init', type=float, default=-2.0,
                        help='CR-HKGE only. Initial logit for the trainable cross-reference gate.')
    parser.add_argument('--cr_best_metric', nargs='?', default='ndcg',
                        help='CR-HKGE only. Metric used to save/export the best checkpoint: recall, precision, hit, or ndcg.')
    parser.add_argument('--cr_best_k', type=int, default=3,
                        help='CR-HKGE only. K value used with --cr_best_metric, e.g. 3 for NDCG@3.')
    parser.add_argument('--cr_export_best_checkpoint', type=int, default=1,
                        help='CR-HKGE only. 1: restore best saved checkpoint before artifact export when save_flag=1.')
    parser.add_argument('--cr_export_embeddings', type=int, default=0,
                        help='CR-HKGE only. 1: export product/entity embeddings and relation weights after training.')
    parser.add_argument('--cr_artifact_path', nargs='?', default='../artifacts/cr_hkge',
                        help='CR-HKGE only. Directory used for exported artifacts.')
    parser.add_argument('--cr_model_version', nargs='?', default='cr_hkge_v1',
                        help='CR-HKGE only. Version label written to exported artifacts.')
    parser.add_argument('--cr_subset_data_path', nargs='?', default='',
                        help='Subset evaluation only. Optional data path used to derive enriched/standard item subsets.')
    parser.add_argument('--cr_subset_dataset', nargs='?', default='',
                        help='Subset evaluation only. Optional dataset name used to derive enriched/standard item subsets.')

    return parser.parse_args()
