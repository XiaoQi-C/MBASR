import tensorflow as tf
import numpy as np
import pandas as pd
import os
import argparse
from utility import *
from SASRecModules import *
import random
import datetime
from evaluation import evaluate_GBGSR
from augmentation import augmentation
from model import GBGSR
import os

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="2"

def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised GRU.")

    parser.add_argument('--epoch', type=int, default=10000,
                        help='Number of max epochs.')
    parser.add_argument('--dataset', nargs='?', default='Tmall',
                        help='dataset')
    parser.add_argument('--data', nargs='?', default='processed_data/Tmall/data',
                        help='data directory')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size.')
    parser.add_argument('--emb_size', type=int, default=64,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--random_seed', default=0, type=float)
    parser.add_argument('--l2', default=0., type=float)
    parser.add_argument('--early_stop_epoch', default=20, type=int)
    parser.add_argument('--alpha', type=float, default=0.8, help='Order Perturbation')
    parser.add_argument('--beta', type=float, default=0.4, help='Redundancy Reduction.')
    parser.add_argument('--gamma', type=float, default=0.4, help='Pairwise Swapping.')
    parser.add_argument('--lamda', type=float, default=0.8, help='swap behaivor.')
    parser.add_argument('--zeta', type=float, default=0.4, help='Behavior Transition.')
    parser.add_argument('--k', type=float, default=0.2, help='add similar user interactions.')
    parser.add_argument('--p', type=float, default=2, help='threshold')
    parser.add_argument('--tag', type=int, default=9, help='1->Order Perturbation 2->Redundancy Reduction '
                                                           '3->Behavior Transition 4->Pairwise Swapping 5->Similar Insertion 6->SI-PS' )

    parser.add_argument('--traintype', nargs='?', default='train_2.pkl', help='traintype')
    return parser.parse_args()


if __name__ == '__main__':
    # tf.compat.v1.disable_eager_execution()
    
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    # Network parameters
    args = parse_args()

    tag, alpha, beta, gamma, lamda, zeta, k, p = args.tag, args.alpha, args.beta, args.gamma, args.lamda, args.zeta, args.k, args.p

    data_directory = args.data
    data_statis = pd.read_pickle(os.path.join(data_directory, 'data_statis.df'))  # read data statistics, including state_size and item_num
    state_size = data_statis['state_size'][0]  # the length of history to define the item_seq
    item_num = data_statis['item_num'][0]  # total number of items
    topk=[5,10,20]

    tf.compat.v1.reset_default_graph()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    tf.compat.v1.set_random_seed(args.random_seed)

    max_relation_num = 5
    global_graph_df = pd.read_pickle(os.path.join(data_directory, 'global_graph_train.df'))
    global_graph = list(global_graph_df['global_graph'])

    model = GBGSR(hidden_size=args.emb_size, learning_rate=args.lr,item_num=item_num,state_size=state_size, global_graph=global_graph, max_relation_num=max_relation_num, args=args)

    saver = tf.train.Saver(max_to_keep=10000)


    nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    if tag == 0:
        label = 'origin'
    elif tag == 1 :
        label = alpha
    elif tag == 2:
        label = beta
    elif tag == 3 :
        label = zeta
    elif tag == 4:
        label = gamma
    elif tag == 5:
        label = p * 10 + k
    elif tag == 6:
        label = alpha
    else:
        label = 'test'

    save_dir = './model/Tmall/newGPG/tag_{}_param_{}_{}'.format(args.tag, label, nowTime)

    isExists = os.path.exists(save_dir)
    if not isExists:
        os.makedirs(save_dir)
        
    data_loader = pd.read_pickle(os.path.join(data_directory, args.traintype))
    print("data number of click :{} , data number of purchase :{}".format(
        data_loader[data_loader['is_buy'] == 0].shape[0],
        data_loader[data_loader['is_buy'] == 1].shape[0],
    ))

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    
    total_step=0
    with tf.compat.v1.Session(config=config) as sess:
        # Initialize variables
        sess.run(tf.compat.v1.global_variables_initializer())
        num_rows=data_loader.shape[0]
        num_batches=int(num_rows/args.batch_size)
        print(num_rows, num_batches)
        best_hit_5 = -1
        count = 0
        for epoch_i in range(args.epoch):
            print(epoch_i)
            start_time_i = datetime.datetime.now()

            for j in range(num_batches):
                batch = data_loader.sample(n=args.batch_size).to_dict()
                item_seq = list(batch['item_seq'].values())
                behavior_seq = list(batch['behavior_seq'].values())
                len_seq = list(batch['len_seq'].values())
                target=list(batch['target'].values())
                target_behavior=list(batch['is_buy'].values())

                item_seq_grouped = list(batch['item_seq_grouped'].values())
                behavior_seq_grouped = list(batch['behavior_seq_grouped'].values())
                jaccard_similarity = list(batch['similar_jaccard'].values())
                
                # item_seq = [row[-5::-5][::-1] for row in item_seq]
                # behavior_seq = [row[-5::-5][::-1] for row in behavior_seq]
              
                # len_seq = [len(row) for row in item_seq]
                len_seq = [np.sum(seq!=item_num) for seq in item_seq]
                len_seq = [ss if ss > 0 else 1 for ss in len_seq]
                # len_seq = [len(row) for row in item_seq]
                item_seq = [list(item_seq[r][:l1]) for r,l1 in enumerate(len_seq)]
                behavior_seq = [list(behavior_seq[r][:l1]) for r,l1 in enumerate(len_seq)]
                
                
                item_seq, behavior_seq, len_seq = augmentation(item_seq, behavior_seq, len_seq, item_num, state_size,
                                                               jaccard_similarity, item_seq_grouped,
                                                               behavior_seq_grouped, tag, alpha, beta, gamma, lamda,
                                                               zeta, k, p)

                local_graph_in = list(batch['local_graph_in'].values())
                local_graph_out = list(batch['local_graph_out'].values())
                
                loss, _ = sess.run([model.loss, model.opt],
                                   feed_dict={model.item_seq: item_seq,
                                              model.behavior_seq: behavior_seq,
                                              model.len_seq: len_seq,
                                              model.target: target,
                                              model.target_behavior: target_behavior,
                                              model.local_graph_in: local_graph_in, 
                                              model.local_graph_out: local_graph_out, 
                                              model.is_training:True})
                total_step+=1
                if total_step % 200 == 0:
                    print("the loss in %dth batch is: %f" % (total_step, loss))
            over_time_i = datetime.datetime.now() 
            total_time_i = (over_time_i - start_time_i).total_seconds()
            print('total times: %s' % total_time_i)

            hit5, ndcg5, hit10, ndcg10, hit20, ndcg20 = evaluate_GBGSR(sess, model, data_directory, topk,
                                                                        have_dropout=True,
                                                                        is_test=False)

            if hit5 > best_hit_5:
                best_hit_5 = hit5
                count = 0
                save_root = os.path.join(save_dir,
                                         'epoch_{}_hit@5_{:.4f}_ndcg@5_{:.4f}_hit@10_{:.4f}_ndcg@10_{:.4f}_hit@20_{:.4f}_ndcg@20_{:.4f}'.format(
                                             epoch_i, hit5, ndcg5, hit10, ndcg10, hit20, ndcg20))
                isExists = os.path.exists(save_root)
                model_name = 'gpg4hsr.ckpt'
                save_root = os.path.join(save_root, model_name)
                saver.save(sess, save_root)
                
            else:
                count += 1
            if count == args.early_stop_epoch:
                break
            
    # with tf.Session() as sess :
    #     saver.restore(sess, '/xiaojing/GPG4HSR/model/GPG4HSR/JD/tag_3_param_0.2/epoch_31_hit@5_0.3195_ndcg@5_0.2309_hit@10_0.4035_ndcg@10_0.2580_hit@20_0.4827_ndcg@20_0.2780/gpg4hsr.ckpt')
    #     # hit5, ndcg5, hit10, ndcg10, hit20, ndcg20 = evaluate_rib(sess, GRUnet, data_directory, topk, have_dropout=True,
    #     #                                                          have_user_emb=False, is_test=True)    
    #     hit5, ndcg5, hit10, ndcg10, hit20, ndcg20 = evaluate_GBGSR(sess, model, data_directory, topk,
    #                                                                 have_dropout=True,
    #                                                                 is_test=True)
