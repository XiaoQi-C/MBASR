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
from evaluation import evaluate_BAR


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


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
        self.behavior_num = 2
        self.is_training = tf.placeholder(tf.bool, shape=())

        self.all_embeddings=self.initialize_embeddings()

        self.item_seq = tf.placeholder(tf.int32, [None, state_size],name='item_seq')
        self.len_seq=tf.placeholder(tf.int32, [None],name='len_seq')
        self.target= tf.placeholder(tf.int32, [None],name='target') # target item, to calculate ce loss
        self.next_behaviors_input = tf.placeholder(tf.int32, [None])
        self.behavior_seq = tf.placeholder(tf.int32, [None, state_size])
        self.position = tf.placeholder(tf.int32, [None, state_size])
        self.next_behavior_emb = tf.nn.embedding_lookup(self.all_embeddings['behavior_embeddings'], self.next_behaviors_input)
        self.behavior_emb = tf.nn.embedding_lookup(self.all_embeddings['behavior_embeddings'], self.behavior_seq)
        self.position_emb = tf.nn.embedding_lookup(self.all_embeddings['position_embeddings'], self.position)

        self.att_behavior_input = tf.tile(tf.expand_dims(self.next_behavior_emb, axis=1), (1, self.state_size, 1))

        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.item_seq, item_num)), -1)
        self.input_emb=tf.nn.embedding_lookup(self.all_embeddings['item_embeddings'],self.item_seq)

        self.mix_input = self.position_emb + self.behavior_emb + self.input_emb
        self.att_input = tf.concat([self.mix_input,self.mix_input-self.att_behavior_input,self.mix_input*self.att_behavior_input, self.att_behavior_input], axis=2)
        self.att_input*=mask

        self.att_net = tf.contrib.layers.fully_connected(self.att_input, self.hidden_size,
                                                         activation_fn=tf.nn.relu, scope="att_net1")
        self.att_net = tf.contrib.layers.fully_connected(self.att_net, 1,
                                                         activation_fn=tf.nn.tanh, scope="att_net2")
        self.att = self.att_net
        # Positional Encoding
        pos_emb=tf.nn.embedding_lookup(self.all_embeddings['pos_embeddings'],tf.tile(tf.expand_dims(tf.range(tf.shape(self.item_seq)[1]), 0), [tf.shape(self.item_seq)[0], 1]))
        self.seq=self.input_emb * (1 + self.att) + pos_emb

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

        self.state_behavior_feat = tf.contrib.layers.fully_connected(
            tf.concat([self.state_hidden, self.next_behavior_emb], axis=1), self.hidden_size,
            activation_fn=tf.nn.relu, scope="state_behavior_feat")  # all q-values

        self.final_feat = tf.concat([self.state_hidden, self.state_behavior_feat,self.next_behavior_emb], axis=1)
        # self.final_feat = tf.contrib.layers.fully_connected(self.final_feat,self.hidden_size)
        # self.output = tf.nn.softmax(tf.matmul(self.final_feat, tf.transpose(self.all_embeddings['item_embeddings'], [1, 0])))
        self.output = tf.contrib.layers.fully_connected(self.final_feat,self.item_num,activation_fn=tf.nn.softmax,scope='fc')

        self.reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(args.l2), tf.trainable_variables())
        self.loss = tf.keras.losses.sparse_categorical_crossentropy(self.target,self.output)
        self.loss = tf.reduce_mean(self.loss + self.reg)

        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


    def initialize_embeddings(self):
        all_embeddings = dict()
        pos_embeddings=tf.Variable(tf.random_normal([self.state_size, self.hidden_size], 0.0, 0.01),
            name='pos_embeddings')
        item_embeddings= tf.Variable(tf.random_normal([self.item_num, self.hidden_size], 0.0, 0.01),
            name='item_embeddings')
        padding = tf.zeros([1,self.hidden_size],dtype= tf.float32)
        item_embeddings = tf.concat([item_embeddings,padding],axis=0)
        behavior_embeddings = tf.Variable(tf.random_normal([self.behavior_num, self.hidden_size], 0.0, 0.01),
                                          name='behavior_embeddings')
        padding = tf.zeros([1,self.hidden_size],dtype= tf.float32)
        behavior_embeddings = tf.concat([behavior_embeddings,padding],axis=0)
        position_embeddings = tf.Variable(tf.random_normal([self.state_size + 1, self.hidden_size], 0.0, 0.01),
                                          name='position_embeddings')
        all_embeddings['item_embeddings']=item_embeddings
        all_embeddings['pos_embeddings']=pos_embeddings
        all_embeddings['behavior_embeddings'] = behavior_embeddings
        all_embeddings['position_embeddings'] = position_embeddings
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
        
    save_dir = './model/Tmall/newSASBAR/tag_{}_param_{}_{}'.format(args.tag, label, nowTime)


    # save_dir = './model/RIB/7/emb_{}_dropout_{}_{}'.format(args.emb_size,args.dropout_rate,nowTime)

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
        merged = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter("logs/",sess.graph)
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
                next_behavior = list(batch['is_buy'].values())
                len_seq = list(batch['len_seq'].values())
                target=list(batch['target'].values())
                item_seq_grouped = list(batch['item_seq_grouped'].values())
                behavior_seq_grouped = list(batch['behavior_seq_grouped'].values())
                jaccard_similarity = list(batch['similar_jaccard'].values())

                # len_seq = [len(row) for row in item_seq]
                len_seq = [np.sum(seq!=item_num) for seq in item_seq]
                len_seq = [ss if ss > 0 else 1 for ss in len_seq]

                item_seq = [list(item_seq[r][:l1]) for r,l1 in enumerate(len_seq)]
                behavior_seq = [list(behavior_seq[r][:l1]) for r,l1 in enumerate(len_seq)]

                item_seq, behavior_seq, len_seq = augmentation(item_seq, behavior_seq, len_seq, item_num, state_size,
                                                               jaccard_similarity, item_seq_grouped,
                                                               behavior_seq_grouped, tag, alpha, beta, gamma, lamda,
                                                               zeta, k, p)

                position_info = np.zeros((args.batch_size,state_size))
                for idx, l in enumerate(len_seq):
                    position_info[idx][:l]=range(l,0,-1)

                loss, _ = sess.run([SASRec.loss, SASRec.opt],
                                   feed_dict={SASRec.item_seq: item_seq,
                                              SASRec.len_seq: len_seq,
                                              SASRec.behavior_seq: behavior_seq,
                                              SASRec.next_behaviors_input: next_behavior,
                                              SASRec.position: position_info,
                                              SASRec.target: target,
                                              SASRec.is_training:True})

                # merged_summary = sess.run(merged)
                # writer.add_summary(merged_summary, j)

                total_step+=1
                if total_step % 200 == 0:
                    print("the loss in %dth batch is: %f" % (total_step, loss))
            over_time_i = datetime.datetime.now()
            total_time_i = (over_time_i - start_time_i).total_seconds()
            print('total times: %s' % total_time_i)

            hit5, ndcg5, hit10, ndcg10, hit20, ndcg20 = evaluate_BAR(sess, SASRec, data_directory, topk,
                                                                        have_dropout=True,
                                                                        is_test=False)
            # tf.summary.scalar('hit10', hit10)
            if hit10 > best_hit_10 :
                best_hit_10 = hit10
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
    #     saver.restore(sess, './model/JD/newSASBAR/tag_2_param_0.2_20241207_080306/epoch_32_hit@5_0.3118_ndcg@5_0.2134_hit@10_0.3982_ndcg@10_0.2415_hit@20_0.4803_ndcg@20_0.2622/sasrec.ckpt')
    #
    #     hit5, ndcg5, hit10, ndcg10, hit20, ndcg20 = evaluate_BAR(sess, SASRec, data_directory, topk,
    #                                                             have_dropout=True, is_test=True)


