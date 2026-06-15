'''
Created on Dec 18, 2018
Tensorflow Implementation of Knowledge Graph Attention Network (KGAT) model in:
Wang Xiang et al. KGAT: Knowledge Graph Attention Network for Recommendation. In KDD 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
from utility.tf_compat import tf
from utility.helper import *
from utility.batch_test import *
from time import time

from BPRMF import BPRMF
from CKE import CKE
from CFKG import CFKG
from NFM import NFM
from KGAT import KGAT
from CRHKGE import CRHKGE


import os
import sys
import random as rd
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def load_pretrained_data(args):
    pre_model = 'mf'
    if args.pretrain == -2:
        pre_model = 'kgat'
    pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, pre_model)
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the pretrained bprmf model parameters.')
    except Exception:
        pretrain_data = None
    return pretrain_data


def _scalar(x):
    return float(np.asarray(x).reshape(-1)[0])


def _metric_key(metric):
    metric = str(metric).lower()
    if metric in ['hit', 'hit_ratio', 'hr']:
        return 'hit_ratio'
    if metric in ['recall', 'precision', 'ndcg']:
        return metric
    raise ValueError('unsupported metric: %s' % metric)


def _metric_index_for_k(k_values, target_k):
    if target_k in k_values:
        return k_values.index(target_k)
    print('warning: requested K=%d is not in Ks=%s; using K=%d.' %
          (target_k, k_values, k_values[0]))
    return 0


def _metric_value(ret, metric, target_k, k_values):
    metric_key = _metric_key(metric)
    metric_idx = _metric_index_for_k(k_values, target_k)
    return float(ret[metric_key][metric_idx])


if __name__ == '__main__':
    # get argument settings.
    tf.set_random_seed(2019)
    np.random.seed(2019)
    rd.seed(2019)
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    """
    *********************************************************
    Load Data from data_generator function.
    """
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    config['n_relations'] = data_generator.n_relations
    config['n_entities'] = data_generator.n_entities

    if args.model_type in ['kgat', 'cr_hkge', 'cfkg']:
        "Load the laplacian matrix."
        config['A_in'] = sum(data_generator.lap_list)

        "Load the KG triplets."
        config['all_h_list'] = data_generator.all_h_list
        config['all_r_list'] = data_generator.all_r_list
        config['all_t_list'] = data_generator.all_t_list
        config['all_v_list'] = data_generator.all_v_list

        if args.model_type == 'cr_hkge' and hasattr(data_generator, 'get_cr_hkge_config'):
            config['cr_hkge_config'] = data_generator.get_cr_hkge_config()
            config['lap_list'] = data_generator.lap_list
            config['adj_r_list'] = data_generator.adj_r_list

    t0 = time()

    """
    *********************************************************
    Use the pretrained data to initialize the embeddings.
    """
    if args.pretrain in [-1, -2]:
        pretrain_data = load_pretrained_data(args)
    else:
        pretrain_data = None

    """
    *********************************************************
    Select one of the models.
    """
    if args.model_type == 'bprmf':
        model = BPRMF(data_config=config, pretrain_data=pretrain_data, args=args)

    elif args.model_type == 'cke':
        model = CKE(data_config=config, pretrain_data=pretrain_data, args=args)

    elif args.model_type in ['cfkg']:
        model = CFKG(data_config=config, pretrain_data=pretrain_data, args=args)

    elif args.model_type in ['nfm', 'fm']:
        model = NFM(data_config=config, pretrain_data=pretrain_data, args=args)

    elif args.model_type in ['kgat']:
        model = KGAT(data_config=config, pretrain_data=pretrain_data, args=args)

    elif args.model_type in ['cr_hkge']:
        model = CRHKGE(data_config=config, pretrain_data=pretrain_data, args=args)

    else:
        raise NotImplementedError('unsupported model_type: %s' % args.model_type)

    saver = tf.train.Saver()

    """
    *********************************************************
    Save the model parameters.
    """
    if args.save_flag == 1:
        if args.model_type in ['bprmf', 'cke', 'fm', 'cfkg']:
            weights_save_path = '%sweights/%s/%s/l%s_r%s' % (args.weights_path, args.dataset, model.model_type,
                                                             str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))

        elif args.model_type in ['ncf', 'nfm', 'kgat', 'cr_hkge']:
            layer = '-'.join([str(l) for l in eval(args.layer_size)])
            weights_save_path = '%sweights/%s/%s/%s/l%s_r%s' % (
                args.weights_path, args.dataset, model.model_type, layer, str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))

        ensureDir(weights_save_path)
        save_saver = tf.train.Saver(max_to_keep=1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    """
    *********************************************************
    Reload the model parameters to fine tune.
    """
    if args.pretrain == 1:
        if args.model_type in ['bprmf', 'cke', 'fm', 'cfkg']:
            pretrain_path = '%sweights/%s/%s/l%s_r%s' % (args.weights_path, args.dataset, model.model_type, str(args.lr),
                                                             '-'.join([str(r) for r in eval(args.regs)]))

        elif args.model_type in ['ncf', 'nfm', 'kgat', 'cr_hkge']:
            layer = '-'.join([str(l) for l in eval(args.layer_size)])
            pretrain_path = '%sweights/%s/%s/%s/l%s_r%s' % (
                args.weights_path, args.dataset, model.model_type, layer, str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))

        ckpt = tf.train.get_checkpoint_state(os.path.dirname(pretrain_path + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('load the pretrained model parameters from: ', pretrain_path)

            # *********************************************************
            # get the performance from the model to fine tune.
            if args.report != 1:
                users_to_test = list(data_generator.test_user_dict.keys())

                ret = test(sess, model, users_to_test, drop_flag=False, batch_test_flag=batch_test_flag)
                cur_best_pre_0 = ret['recall'][0]

                pretrain_ret = 'pretrained model recall=[%.5f, %.5f], precision=[%.5f, %.5f], hit=[%.5f, %.5f],' \
                               'ndcg=[%.5f, %.5f], auc=[%.5f]' % \
                               (ret['recall'][0], ret['recall'][-1],
                                ret['precision'][0], ret['precision'][-1],
                                ret['hit_ratio'][0], ret['hit_ratio'][-1],
                                ret['ndcg'][0], ret['ndcg'][-1], ret['auc'])
                print(pretrain_ret)

                # *********************************************************
                # save the pretrained model parameters of mf (i.e., only user & item embeddings) for pretraining other models.
                if args.save_flag == -1:
                    user_embed, item_embed = sess.run(
                        [model.weights['user_embedding'], model.weights['item_embedding']],
                        feed_dict={})
                    # temp_save_path = '%spretrain/%s/%s/%s_%s.npz' % (args.proj_path, args.dataset, args.model_type, str(args.lr),
                    #                                                  '-'.join([str(r) for r in eval(args.regs)]))
                    temp_save_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, model.model_type)
                    ensureDir(temp_save_path)
                    np.savez(temp_save_path, user_embed=user_embed, item_embed=item_embed)
                    print('save the weights of fm in path: ', temp_save_path)
                    exit()

                # *********************************************************
                # save the pretrained model parameters of kgat (i.e., user & item & kg embeddings) for pretraining other models.
                if args.save_flag == -2:
                    user_embed, entity_embed, relation_embed = sess.run(
                        [model.weights['user_embed'], model.weights['entity_embed'], model.weights['relation_embed']],
                        feed_dict={})

                    temp_save_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, args.model_type)
                    ensureDir(temp_save_path)
                    np.savez(temp_save_path, user_embed=user_embed, entity_embed=entity_embed, relation_embed=relation_embed)
                    print('save the weights of kgat in path: ', temp_save_path)
                    exit()

        else:
            sess.run(tf.global_variables_initializer())
            cur_best_pre_0 = 0.
            print('without pretraining.')
    else:
        sess.run(tf.global_variables_initializer())
        cur_best_pre_0 = 0.
        print('without pretraining.')

    """
    *********************************************************
    Get the final performance w.r.t. different sparsity levels.
    """
    if args.report == 1:
        assert args.test_flag == 'full'
        users_to_test_list, split_state = data_generator.get_sparsity_split()

        users_to_test_list.append(list(data_generator.test_user_dict.keys()))
        split_state.append('all')

        save_path = '%sreport/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
        ensureDir(save_path)
        f = open(save_path, 'w')
        f.write('embed_size=%d, lr=%.4f, regs=%s, loss_type=%s, \n' % (args.embed_size, args.lr, args.regs,
                                                                       args.loss_type))

        for i, users_to_test in enumerate(users_to_test_list):
            ret = test(sess, model, users_to_test, drop_flag=False, batch_test_flag=batch_test_flag)

            final_perf = "recall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                         ('\t'.join(['%.5f' % r for r in ret['recall']]),
                          '\t'.join(['%.5f' % r for r in ret['precision']]),
                          '\t'.join(['%.5f' % r for r in ret['hit_ratio']]),
                          '\t'.join(['%.5f' % r for r in ret['ndcg']]))
            print(final_perf)

            f.write('\t%s\n\t%s\n' % (split_state[i], final_perf))
        f.close()
        exit()

    """
    *********************************************************
    Train.
    """
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    stopping_step = 0
    should_stop = False

    show_step = 10
    k_values = eval(args.Ks)
    cr_best_metric = getattr(args, 'cr_best_metric', 'ndcg')
    cr_best_k = int(getattr(args, 'cr_best_k', 3))
    cr_best_score = -np.inf
    cr_best_epoch = None

    for epoch in range(args.epoch):
        t1 = time()
        loss, base_loss, kge_loss, reg_loss = 0., 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1

        """
        *********************************************************
        Alternative Training for KGAT:
        ... phase 1: to train the recommender.
        """
        # Fase I CR-HKGE/KGAT:
        # training BPR untuk ranking rekomendasi. Pada penelitian ini, pasangan
        # positif berasal dari content-based positive pairs, bukan purchase/rating
        # historis. Fase ini melatih user/profile embedding dan item embedding
        # agar positive item mendapat skor lebih tinggi daripada negative item.
        for idx in range(n_batch):
            btime= time()

            batch_data = data_generator.generate_train_batch()
            feed_dict = data_generator.generate_train_feed_dict(model, batch_data)

            _, batch_loss, batch_base_loss, batch_kge_loss, batch_reg_loss = model.train(sess, feed_dict=feed_dict)

            loss += batch_loss
            base_loss += batch_base_loss
            kge_loss += batch_kge_loss
            reg_loss += batch_reg_loss

        if np.isnan(loss) == True:
            print('ERROR: loss@phase1 is nan.')
            sys.exit()

        """
        *********************************************************
        Alternative Training for KGAT:
        ... phase 2: to train the KGE method & update the attentive Laplacian matrix.
        """
        if args.model_type in ['kgat', 'cr_hkge']:

            n_A_batch = len(data_generator.all_h_list) // args.batch_size_kg + 1

            if args.use_kge is True:
                # Fase II-A:
                # training TransR/KGE atas triples KG. Pada CR-HKGE, skor KGE
                # juga dipengaruhi relation-type attention ketika fitur tersebut
                # aktif, sehingga relasi fragrance punya prioritas berbeda.
                for idx in range(n_A_batch):
                    btime = time()

                    A_batch_data = data_generator.generate_train_A_batch()
                    feed_dict = data_generator.generate_train_A_feed_dict(model, A_batch_data)

                    _, batch_loss, batch_kge_loss, batch_reg_loss = model.train_A(sess, feed_dict=feed_dict)

                    loss += batch_loss
                    kge_loss += batch_kge_loss
                    reg_loss += batch_reg_loss

            if args.use_att is True:
                # Fase II-B:
                # update attentive adjacency A_in. Matriks inilah yang dipakai
                # pada message passing KGAT/CR-HKGE di epoch berikutnya.
                model.update_attentive_A(sess)

        if np.isnan(loss) == True:
            print('ERROR: loss@phase2 is nan.')
            sys.exit()

        if (epoch + 1) % show_step != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f]' % (
                    epoch, time() - t1, _scalar(loss), _scalar(base_loss), _scalar(kge_loss), _scalar(reg_loss))
                print(perf_str)
            continue

        """
        *********************************************************
        Test.
        """
        t2 = time()
        users_to_test = list(data_generator.test_user_dict.keys())

        ret = test(sess, model, users_to_test, drop_flag=False, batch_test_flag=batch_test_flag)

        """
        *********************************************************
        Performance logging.
        """
        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f], recall=[%.5f, %.5f], ' \
                       'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                       (epoch, t2 - t1, t3 - t2, _scalar(loss), _scalar(base_loss), _scalar(kge_loss), _scalar(reg_loss), ret['recall'][0], ret['recall'][-1],
                        ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                        ret['ndcg'][0], ret['ndcg'][-1])
            print(perf_str)

        if args.model_type == 'cr_hkge':
            # CR-HKGE disimpan berdasarkan metric yang dipilih, default NDCG@3,
            # karena target utama paper adalah kualitas ranking Top-K.
            cur_cr_score = _metric_value(ret, cr_best_metric, cr_best_k, k_values)
            if cur_cr_score > cr_best_score:
                cr_best_score = cur_cr_score
                cr_best_epoch = epoch
                if args.save_flag == 1:
                    save_saver.save(sess, weights_save_path + '/weights', global_step=epoch)
                    print('save CR-HKGE best %s@%d=%.5f checkpoint in path: %s' %
                          (cr_best_metric, cr_best_k, cr_best_score, weights_save_path))

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=10)

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if args.model_type != 'cr_hkge' and ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
            save_saver.save(sess, weights_save_path + '/weights', global_step=epoch)
            print('save the weights in path: ', weights_save_path)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break

    if len(rec_loger) == 0:
        users_to_test = list(data_generator.test_user_dict.keys())
        ret = test(sess, model, users_to_test, drop_flag=False, batch_test_flag=batch_test_flag)

        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            perf_str = 'Final Eval [%.1fs]: recall=[%.5f, %.5f], precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                       (time() - t0, ret['recall'][0], ret['recall'][-1],
                        ret['precision'][0], ret['precision'][-1],
                        ret['hit_ratio'][0], ret['hit_ratio'][-1],
                        ret['ndcg'][0], ret['ndcg'][-1])
            print(perf_str)

        # Smoke test sering memakai epoch kecil (< show_step=10). Dalam kondisi
        # itu training belum masuk evaluasi periodik, sehingga checkpoint belum
        # pernah disimpan walaupun Final Eval berhasil. Simpan checkpoint fallback
        # agar script evaluasi terpisah tetap bisa memuat bobot model.
        if args.save_flag == 1:
            final_epoch = max(0, args.epoch - 1)
            save_saver.save(sess, weights_save_path + '/weights', global_step=final_epoch)
            if args.model_type == 'cr_hkge':
                cr_best_score = _metric_value(ret, cr_best_metric, cr_best_k, k_values)
                cr_best_epoch = final_epoch
                print('save CR-HKGE final-eval checkpoint in path: ', weights_save_path)
            else:
                print('save final-eval weights in path: ', weights_save_path)

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    if args.model_type == 'cr_hkge':
        # Untuk CR-HKGE, best iteration dilaporkan mengikuti metric pilihan
        # cr_best_metric/cr_best_k, bukan selalu Recall@K pertama seperti KGAT.
        metric_key = _metric_key(cr_best_metric)
        metric_idx = _metric_index_for_k(k_values, cr_best_k)
        metric_log = {
            'recall': recs,
            'precision': pres,
            'hit_ratio': hit,
            'ndcg': ndcgs,
        }[metric_key]
        metric_scores = metric_log[:, metric_idx]
        idx = int(np.argmax(metric_scores))
        best_metric_label = '%s@%d=%.5f' % (cr_best_metric, cr_best_k, metric_scores[idx])
    else:
        best_rec_0 = max(recs[:, 0])
        idx = list(recs[:, 0]).index(best_rec_0)
        best_metric_label = 'recall@%d=%.5f' % (k_values[0], recs[idx][0])

    final_perf = "Best Iter=[%d]@[%.1f]\tbest_metric=%s, recall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, best_metric_label, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)

    save_path = '%soutput/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
    ensureDir(save_path)
    f = open(save_path, 'a')

    f.write('embed_size=%d, lr=%.4f, layer_size=%s, node_dropout=%s, mess_dropout=%s, regs=%s, adj_type=%s, use_att=%s, use_kge=%s, pretrain=%d\n\t%s\n'
            % (args.embed_size, args.lr, args.layer_size, args.node_dropout, args.mess_dropout, args.regs, args.adj_type, args.use_att, args.use_kge, args.pretrain, final_perf))
    f.close()

    if hasattr(model, 'export_artifacts') and int(getattr(args, 'cr_export_embeddings', 0)) == 1:
        # Export artifact hanya relevan untuk CR-HKGE karena model ini dipakai
        # sebagai recommendation engine di sistem conversational setelah offline
        # training selesai.
        if (args.model_type == 'cr_hkge' and args.save_flag == 1 and
                int(getattr(args, 'cr_export_best_checkpoint', 1)) == 1):
            ckpt = tf.train.get_checkpoint_state(weights_save_path)
            if ckpt and ckpt.model_checkpoint_path:
                save_saver.restore(sess, ckpt.model_checkpoint_path)
                print('restore CR-HKGE best checkpoint for artifact export: ', ckpt.model_checkpoint_path)
            else:
                print('warning: no CR-HKGE checkpoint found; exporting current session parameters.')
        model.export_artifacts(sess, args, data_generator, final_perf)
