import tensorflow as tf
import numpy as np
import pandas as pd
import os
import argparse
from utility import *
from SASRecModules import *
from augmentation import augmentation
import random
import datetime
from evaluation import evaluate_origin
import math

def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised GRU.")

    parser.add_argument('--epoch', type=int, default=10000,
                        help='Number of max epochs.')
    # parser.add_argument('--dataset', nargs='?', default='datasets/Tmall/data',
    #                     help='dataset')
    parser.add_argument('--data', nargs='?', default='datasets/Tmall/data',
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
    parser.add_argument('--early_stop_epoch', default=20, type=int)
    parser.add_argument('--l2', default=0., type=float)
    parser.add_argument('--alpha', type=float, default=0.8, help='Order Perturbation')
    parser.add_argument('--beta', type=float, default=0.4, help='Redundancy Reduction.')
    parser.add_argument('--gamma', type=float, default=0.4, help='Pairwise Swapping.')
    parser.add_argument('--lamda', type=float, default=0.8, help='swap behaivor.')
    parser.add_argument('--zeta', type=float, default=0.4, help='Behavior Transition.')
    parser.add_argument('--k', type=float, default=0.2, help='add similar user interactions.')
    parser.add_argument('--p', type=float, default=2, help='threshold')
    parser.add_argument('--tag', type=int, default=9, help='1->Order Perturbation 2->Redundancy Reduction '
                                                           '3->Behavior Transition 4->Pairwise Swapping 5->Similar Insertion 6->SI-PS' )


    return parser.parse_args()


class SASRecnetwork:
    def __init__(self, hidden_size,learning_rate,item_num,state_size):
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.emb_size = hidden_size
        self.hidden_size = hidden_size
        self.item_num=int(item_num)
        self.is_training = tf.placeholder(tf.bool, shape=())

        all_embeddings=self.initialize_embeddings()

        self.item_seq = tf.placeholder(tf.int32, [None, state_size],name='item_seq')
        self.len_seq=tf.placeholder(tf.int32, [None],name='len_seq')
        self.target= tf.placeholder(tf.int32, [None],name='target') # target item, to calculate ce loss

        self.input_emb=tf.nn.embedding_lookup(all_embeddings['item_embeddings'],self.item_seq)
        # Positional Encoding
        pos_emb=tf.nn.embedding_lookup(all_embeddings['pos_embeddings'],tf.tile(tf.expand_dims(tf.range(tf.shape(self.item_seq)[1]), 0), [tf.shape(self.item_seq)[0], 1]))
        self.seq=self.input_emb+pos_emb

        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.item_seq, item_num)), -1)
        #Dropout
        self.seq = tf.layers.dropout(self.seq,
                                     rate=args.dropout_rate,
                                     seed=args.random_seed,
                                     training=tf.convert_to_tensor(self.is_training))
        self.seq *= mask

        # Build blocks

        for i in range(args.num_blocks):
            with tf.variable_scope("num_blocks_%d" % i):
                # Self-attention
                self.seq = multihead_attention(queries=normalize(self.seq),
                                               keys=self.seq,
                                               num_units=self.hidden_size,
                                               num_heads=args.num_heads,
                                               dropout_rate=args.dropout_rate,
                                               is_training=self.is_training,
                                               causality=True,
                                               scope="self_attention")

                # Feed forward
                self.seq = feedforward(normalize(self.seq), num_units=[self.hidden_size, self.hidden_size],
                                       dropout_rate=args.dropout_rate,
                                       is_training=self.is_training)
                self.seq *= mask

        self.seq = normalize(self.seq)
        self.state_hidden=extract_axis_1(self.seq, self.len_seq - 1)

        self.output = tf.contrib.layers.fully_connected(self.state_hidden,self.item_num,activation_fn=tf.nn.softmax,scope='fc')

        self.reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(args.l2), tf.trainable_variables())
        self.loss = tf.keras.losses.sparse_categorical_crossentropy(self.target,self.output)
        self.loss = tf.reduce_mean(self.loss + self.reg)

        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def initialize_embeddings(self):
        all_embeddings = dict()
        item_embeddings= tf.Variable(tf.random_normal([self.item_num, self.emb_size], 0.0, 0.01),
            name='item_embeddings')
        padding = tf.zeros([1,self.emb_size],dtype= tf.float32)
        item_embeddings = tf.concat([item_embeddings,padding],axis=0)
        pos_embeddings=tf.Variable(tf.random_normal([self.state_size, self.hidden_size], 0.0, 0.01),
            name='pos_embeddings')
        all_embeddings['item_embeddings']=item_embeddings
        all_embeddings['pos_embeddings']=pos_embeddings
        return all_embeddings

def get_target(input, target, item_num, a=0.8, tau=0.3):
    sample_target = []
    sample_seq = []
    for i in range (len(input)):
        item_seq = input[i]
        t = target[i]
        length = len(item_seq) + 1
        if item_seq[0] == item_num:
            sample_target.append(t)
            item_seq = np.pad(item_seq, (0, 50 - len(item_seq)), 'constant', constant_values=item_num)
            sample_seq.append(list(item_seq))
            continue
        
        select_num = math.ceil(length * tau)
        item_indices = np.arange(select_num)  #
        item_importance = np.power(a, select_num - item_indices)
        
        # item_importance = np.exp(item_importance)
        prob = item_importance / np.sum(item_importance)
        target_index = np.random.choice(range(select_num), size=1, replace=False, p=prob)[0]
        target_index = length - select_num + target_index
        if target_index >= length - 1:
            sample_target.append(t)
            item_seq = np.pad(item_seq, (0, 50 - len(item_seq)), 'constant', constant_values=item_num)
            sample_seq.append(list(item_seq))
        else:
            sample_target.append(item_seq[target_index])
            del item_seq[target_index]
            item_seq.append(t)
            item_seq = np.pad(item_seq, (0, 50 - len(item_seq)), 'constant', constant_values=item_num)
            sample_seq.append(list(item_seq))
            
    return sample_target, sample_seq

if __name__ == '__main__':
    # Network parameters
    args = parse_args()
    tag, alpha, beta, gamma, lamda, zeta, k, p = args.tag, args.alpha, args.beta, args.gamma, args.lamda, args.zeta, args.k, args.p

    data_directory = args.data
    data_statis = pd.read_pickle(os.path.join(data_directory, 'data_statis.df'))  # read data statistics, includeing state_size and item_num
    state_size = data_statis['state_size'][0]  # the length of history to define the state
    item_num = data_statis['item_num'][0]  # total number of items
    topk=[5,10,20]
    # save_file = 'pretrain-GRU/%d' % (hidden_size)
    tf.reset_default_graph()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)


    SASRec = SASRecnetwork(hidden_size=args.emb_size, learning_rate=args.lr,item_num=item_num,state_size=state_size)

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
        
    save_dir = './model/UB/RSSSASRec/7/tag_{}_param_{}_{}'.format(args.tag, label, nowTime)


    isExists = os.path.exists(save_dir)
    if not isExists:
        os.makedirs(save_dir)

    data_loader = pd.read_pickle(os.path.join(data_directory, 'train_2.df'))

    print("data number of click :{} , data number of purchase :{}".format(
        data_loader[data_loader['is_buy'] == 0].shape[0],
        data_loader[data_loader['is_buy'] == 1].shape[0],
    ))

    total_step=0

    with tf.Session() as sess:
        # Initialize variables
        sess.run(tf.global_variables_initializer())
        # evaluate(sess)
        num_rows=data_loader.shape[0]
        num_batches=int(num_rows/args.batch_size)
        print(num_rows,num_batches)
        best_hit_5 = -1
        count = 0
        for i in range(args.epoch):
            print(i)
            start_time_i = datetime.datetime.now()  

            for j in range(num_batches):
                batch = data_loader.sample(n=args.batch_size).to_dict()
                item_seq = list(batch['item_seq'].values())
                len_seq = list(batch['len_seq'].values())
                target=list(batch['target'].values())

                # sp_item_seq = [row[-7::-7][::-1] for row in item_seq]
                sp_item_seq = item_seq
                sp_len_seq = [np.sum(seq!=item_num) for seq in sp_item_seq]
                
                sp_len_seq = [ss if ss > 0 else 1 for ss in sp_len_seq]

                input = [list(sp_item_seq[r][:l1]) for r,l1 in enumerate(sp_len_seq)]
                
                sample_target, sample_seq = get_target(input, target, item_num)
               
    
                target=list(batch['target'].values())
              
                loss, _ = sess.run([SASRec.loss, SASRec.opt],
                                   feed_dict={SASRec.item_seq: sample_seq,
                                              SASRec.len_seq: sp_len_seq,
                                              SASRec.target: sample_target,
                                              SASRec.is_training:True})
               
                total_step+=1
                if total_step % 200 == 0:
                    print("the loss in %dth batch is: %f" % (total_step, loss))
            over_time_i = datetime.datetime.now()  
            total_time_i = (over_time_i - start_time_i).total_seconds()
            print('total times: %s' % total_time_i)

            hit5, ndcg5, hit10, ndcg10, hit20, ndcg20 = evaluate_origin(sess, SASRec, data_directory, topk,
                                                                        have_dropout=True,have_user_emb=False,
                                                                        is_test=False)

            if hit5 > best_hit_5 :
                best_hit_5 = hit5
                count = 0
                save_root = os.path.join(save_dir,
                                         'epoch_{}_hit@5_{:.4f}_ndcg@5_{:.4f}_hit@10_{:.4f}_ndcg@10_{:.4f}_hit@20_{:.4f}_ndcg@20_{:.4f}'.format(
                                             i, hit5, ndcg5, hit10, ndcg10, hit20, ndcg20))
                isExists = os.path.exists(save_root)
                if not isExists:
                    os.makedirs(save_root)
                model_name = 'sasrec.ckpt'
                save_root = os.path.join(save_root, model_name)
                saver.save(sess, save_root)
            else:
                count += 1
            if count == args.early_stop_epoch:
                break

    # with tf.Session() as sess :
    #     saver.restore(sess, '/home/temp_user/xiaoj/SHOCCF-baselines/model/UB/RSSSASBAR/5/tag_2_param_0.8_20231117_122131/epoch_63_hit@5_0.0611_ndcg@5_0.0402_hit@10_0.0887_ndcg@10_0.0491_hit@20_0.1282_ndcg@20_0.0591/sasrec.ckpt')
    #     hit5, ndcg5, hit10, ndcg10, hit20, ndcg20 = evaluate_BAR(sess, SASRec, data_directory, topk,
                                                                # have_dropout=True, is_test=True)
    # #     hit5, ndcg5, hit10, ndcg10, hit20, ndcg20 = evaluate_BAR(sess, SASRec, data_directory, topk,
    # #                                                             have_dropout=True, have_user_emb=False,
    # #                                                             is_test=True,type='clicked')
    # #     hit5, ndcg5, hit10, ndcg10, hit20, ndcg20 = evaluate_BAR(sess, SASRec, data_directory, topk,
    # #                                                             have_dropout=True, have_user_emb=False,
    # #                                                             is_test=True,type='unclicked')






