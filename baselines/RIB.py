import argparse
import datetime
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from augmentation import augmentation
from evaluation import evaluate_rib
import random


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised GRU.")

    parser.add_argument('--epoch', type=int, default=10000,
                        help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='../datasets/JD/data',
                        help='data directory')
    # parser.add_argument('--pretrain', type=int, default=1,
    #                     help='flag for pretrain. 1: initialize from pretrain; 0: randomly initialize; -1: save the model to pretrain file')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size.')
    parser.add_argument('--emb_size', type=int, default=64,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--random_seed', default=0, type=float)
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

    return parser.parse_args()


class GRUnetwork:
    def __init__(self, emb_size,learning_rate,item_num,state_size):
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.behavior_num = 2
        self.item_num=int(item_num)

        self.all_embeddings=self.initialize_embeddings()

        self.item_seq = tf.placeholder(tf.int32, [None, state_size],name='item_seq')
        self.len_seq=tf.placeholder(tf.int32, [None],name='len_seq')
        self.target= tf.placeholder(tf.int32, [None],name='target') # target item, to calculate ce loss
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.behavior_seq = tf.placeholder(tf.int32, [None, state_size])

        self.behavior_emb = tf.nn.embedding_lookup(self.all_embeddings['behavior_embeddings'], self.behavior_seq)
        self.input_emb=tf.nn.embedding_lookup(self.all_embeddings['item_embeddings'],self.item_seq)
        self.new_input_emb = tf.concat([self.input_emb,self.behavior_emb],axis=2)
        a = tf.Print(self.new_input_emb, ["OUTPUT", self.new_input_emb])
        #print(a.shape)
        self.gru_out, self.states_hidden= tf.nn.dynamic_rnn(
            tf.contrib.rnn.GRUCell(self.emb_size),
            self.new_input_emb,
            dtype=tf.float32,
            sequence_length=self.len_seq,
        )
        a = tf.Print(self.gru_out, ["OUTPUT", self.gru_out])
        #print(a.shape)
        a = tf.Print(self.states_hidden, ["OUTPUT", self.states_hidden])
        #print(a.shape)
        self.att_net = tf.contrib.layers.fully_connected(self.gru_out, self.hidden_size,
                                                         activation_fn=tf.nn.tanh, scope="att_net1")
        a = tf.Print(self.att_net, ["OUTPUT", self.att_net])
        #print(a.shape)
        self.att_net = tf.contrib.layers.fully_connected(self.att_net, 1,
                                                         activation_fn=None, scope="att_net2") # batch,state_len,1
        mask = tf.expand_dims(tf.not_equal(self.item_seq, item_num), -1)
        a = tf.Print(mask, ["OUTPUT", mask])
        #print(mask , "HHHHHHHHHHHHHHHH")
        paddings = tf.ones_like(self.att_net) * (-2 ** 32 + 1)
        self.att_net = tf.where(mask, self.att_net, paddings)  # [B, 1, T]
        self.att_net = tf.nn.softmax(self.att_net,axis=1)
        a = tf.Print(self.gru_out * self.att_net, ["OUTPUT", self.gru_out * self.att_net])
        #print(a.shape ,"TTTTTTTTTTTTTTTTTTTTTTT")
        # Add dropout
        self.final_state = tf.reduce_sum(self.gru_out * self.att_net,axis=1)
        with tf.name_scope("dropout"):
            self.final_state = tf.layers.dropout(self.final_state,
                                     rate=args.dropout_rate,
                                   seed=args.random_seed,
                                   training=tf.convert_to_tensor(self.is_training))

        self.output = tf.contrib.layers.fully_connected(self.final_state,self.item_num,activation_fn=tf.nn.softmax,scope='fc')
        self.loss = tf.keras.losses.sparse_categorical_crossentropy(self.target,self.output)
        self.loss = tf.reduce_mean(self.loss)
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


    def initialize_embeddings(self):
        all_embeddings = dict()
        item_embeddings= tf.Variable(tf.random_normal([self.item_num, self.hidden_size], 0.0, 0.01),
            name='item_embeddings')
        padding = tf.zeros([1,self.hidden_size],dtype= tf.float32)
        item_embeddings = tf.concat([item_embeddings,padding],axis=0)
        behavior_embeddings = tf.Variable(tf.random_normal([self.behavior_num, self.hidden_size], 0.0, 0.01),
                                          name='behavior_embeddings')
        padding = tf.zeros([1,self.hidden_size],dtype= tf.float32)
        behavior_embeddings = tf.concat([behavior_embeddings,padding],axis=0)
        all_embeddings['item_embeddings']=item_embeddings
        all_embeddings['behavior_embeddings'] = behavior_embeddings
        return all_embeddings

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


    GRUnet = GRUnetwork(emb_size=args.emb_size, learning_rate=args.lr,item_num=item_num,state_size=state_size)

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
        
    save_dir = './model/JD/newRIB/tag_{}_param_{}_{}'.format(args.tag, label, nowTime)

    isExists = os.path.exists(save_dir)
    if not isExists:
        os.makedirs(save_dir)


    data_loader = pd.read_pickle(os.path.join(data_directory, 'train_2.pkl'))

    single_element_count = data_loader[data_loader['item_seq_grouped'].apply(len) == 1].shape[0]

    print(f"There are {single_element_count} users that do not have the same target. ")

    print("data number of click :{} , data number of purchase :{}".format(
        data_loader[data_loader['is_buy'] == 0].shape[0],
        data_loader[data_loader['is_buy'] == 1].shape[0],
    ))

    total_step = 0

    with tf.Session() as sess:
        # Initialize variables
        sess.run(tf.global_variables_initializer())
        # evaluate(sess)
        num_rows=data_loader.shape[0]
        num_batches=int(num_rows/args.batch_size)
        print(num_rows,num_batches)
        best_hit_10 = -1
        count = 0
        for i in range(args.epoch):
            print(i)
            start_time_i = datetime.datetime.now()

            for j in range(num_batches):
                batch = data_loader.sample(n=args.batch_size).to_dict()
                item_seq = list(batch['item_seq'].values())
                behavior_seq = list(batch['behavior_seq'].values())
                len_seq = list(batch['len_seq'].values())
                target=list(batch['target'].values())
                item_seq_grouped = list(batch['item_seq_grouped'].values())
                behavior_seq_grouped = list(batch['behavior_seq_grouped'].values())
                jaccard_similarity = list(batch['similar_jaccard'].values())


                len_seq = [np.sum(seq!=item_num) for seq in item_seq]
                len_seq = [ss if ss > 0 else 1 for ss in len_seq]

                item_seq = [list(item_seq[r][:l1]) for r,l1 in enumerate(len_seq)]
                behavior_seq = [list(behavior_seq[r][:l1]) for r,l1 in enumerate(len_seq)]

                item_seq, behavior_seq, len_seq = augmentation(item_seq, behavior_seq, len_seq, item_num, state_size, jaccard_similarity, item_seq_grouped, behavior_seq_grouped, tag, alpha, beta, gamma, lamda, zeta, theta, k, p)


                loss, _ = sess.run([GRUnet.loss, GRUnet.opt],
                                   feed_dict={GRUnet.item_seq: item_seq,
                                              GRUnet.len_seq: len_seq,
                                              GRUnet.behavior_seq : behavior_seq,
                                              GRUnet.target: target,
                                              GRUnet.is_training:True
                })

                total_step+=1
                if total_step % 200 == 0:
                    print("the loss in %dth batch is: %f" % (total_step, loss))

            over_time_i = datetime.datetime.now()
            total_time_i = (over_time_i - start_time_i).total_seconds()
            print('total times: %s' % total_time_i)

            hit5, ndcg5,hit10,ndcg10,hit20,ndcg20 = evaluate_rib(sess,GRUnet,data_directory,topk,have_dropout=True,have_user_emb=False,is_test=False)

            if hit10 > best_hit_10 :
                best_hit_10 = hit10
                count = 0
                save_root = os.path.join(save_dir,
                                         'epoch_{}_hit@5_{:.4f}_ndcg@5_{:.4f}_hit@10_{:.4f}_ndcg@10_{:.4f}_hit@20_{:.4f}_ndcg@20_{:.4f}'.format(
                                             i, hit5, ndcg5, hit10, ndcg10, hit20, ndcg20))
                isExists = os.path.exists(save_root)
                if not isExists:
                    os.makedirs(save_root)
                model_name = 'rib.ckpt'
                save_root = os.path.join(save_root, model_name)
                saver.save(sess, save_root)

            else:
                count += 1
            if count == args.early_stop_epoch:
                break



    # with tf.Session() as sess :
    #     # saver.restore(sess, './model/JD/newRIB/1/tag_4_param_0.4_20240923_212017/epoch_69_hit@5_0.3066_ndcg@5_0.2121_hit@10_0.3932_ndcg@10_0.2401_hit@20_0.4715_ndcg@20_0.2600/rib.ckpt')
    #
    #     saver.restore(sess, "./epoch_54_hit@5_0.3114_ndcg@5_0.2179_hit@10_0.4092_ndcg@10_0.2496_hit@20_0.4890_ndcg@20_0.2697/rib.ckpt")
    #
    #     hit5, ndcg5, hit10, ndcg10, hit20, ndcg20 = evaluate_rib(sess, GRUnet, data_directory, topk, have_dropout=True, have_user_emb=False, is_test=True)
