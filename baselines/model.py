import tensorflow as tf
import numpy as np
import pandas as pd
import os
import argparse
from utility import *
from SASRecModules import *
import random
import datetime
import os

class GBGSR:
    def __init__(self, hidden_size,learning_rate,item_num,state_size, global_graph, max_relation_num, args):
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.emb_size = hidden_size
        self.hidden_size = hidden_size
        self.item_num=int(item_num)
        self.behavior_num = 2
        self.max_relation_num = max_relation_num
        self.is_training = tf.compat.v1.placeholder(tf.bool, shape=())

        max_local_relation_num = 1

        all_embeddings=self.initialize_embeddings()

        self.item_seq = tf.compat.v1.placeholder(tf.int32, [None, state_size],name='item_seq')
        self.len_seq=tf.compat.v1.placeholder(tf.int32, [None],name='len_seq')
        self.target= tf.compat.v1.placeholder(tf.int32, [None],name='target') # target item, to calculate ce loss
        self.target_behavior = tf.compat.v1.placeholder(tf.int32, [None],name='target_behavior')
        self.behavior_seq = tf.compat.v1.placeholder(tf.int32, [None, state_size])
        self.local_graph_in = tf.compat.v1.placeholder(tf.int32, [None, state_size, max_local_relation_num])
        self.local_graph_out = tf.compat.v1.placeholder(tf.int32, [None, state_size, max_local_relation_num])
        self.global_graph = tf.convert_to_tensor(global_graph) # [6, item_num+1, max_relation_num, 2] 2: item_id, weight
        # print(self.global_graph.shape.as_list())
        
        
        mask= tf.expand_dims(tf.cast(tf.not_equal(self.item_seq, item_num), dtype=tf.float32), -1)

        self.seq = tf.nn.embedding_lookup(params = all_embeddings['item_embeddings'], ids = self.item_seq)
        self.seq_behavior = tf.nn.embedding_lookup(params = all_embeddings['behavior_embeddings'], ids = self.behavior_seq)
        self.seq_with_b = self.seq + self.seq_behavior
        


        mask_p= tf.equal(self.behavior_seq, 1)
        mask_e= tf.equal(self.behavior_seq, 0)

        self.p = self.global_graph_neural_network(self.seq, self.item_seq, all_embeddings['item_embeddings'], mask_p=mask_p, mask_e=mask_e)
        self.q = self.local_graph_neural_network(self.seq_with_b, self.item_seq, all_embeddings['item_embeddings'])
        self.p *= mask
        self.q *= mask


        alpha = tf.compat.v1.layers.dense(tf.concat([self.p, self.q], -1),1,activation=tf.nn.sigmoid,name='weight_alpha')
        self.seq = alpha * self.p + (1-alpha) * self.q

        pos_emb=tf.nn.embedding_lookup(params=all_embeddings['pos_embeddings'],ids=tf.tile(tf.expand_dims(tf.range(tf.shape(input=self.item_seq)[1]), 0), [tf.shape(input=self.item_seq)[0], 1]))
        self.seq = self.seq + pos_emb

        self.seq *= mask
        
        #Dropout
        self.seq = tf.compat.v1.layers.dropout(self.seq,
                                     rate=args.dropout_rate,
                                     seed=args.random_seed,
                                     training=tf.convert_to_tensor(value=self.is_training))
        self.seq *= mask

        for i in range(args.num_blocks):
            with tf.compat.v1.variable_scope("num_blocks_%d" % i):
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
        
        self.target_behavior_emb = tf.nn.embedding_lookup(params = all_embeddings['behavior_embeddings'], ids = self.target_behavior)
        self.state_hidden = tf.concat([self.state_hidden, self.target_behavior_emb], -1)

        self.output = tf.compat.v1.layers.dense(self.state_hidden,self.item_num,activation=tf.nn.softmax,name='fc')
        self.reg = sum(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES))
        self.loss = tf.keras.losses.sparse_categorical_crossentropy(self.target,self.output)
        self.loss = tf.reduce_mean(input_tensor=self.loss + self.reg)

        self.opt = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


    def initialize_embeddings(self):
        all_embeddings = dict()
        item_embeddings= tf.Variable(tf.random.normal([self.item_num, self.emb_size], 0.0, 0.01),
            name='item_embeddings')
        padding = tf.zeros([1,self.emb_size],dtype= tf.float32)
        item_embeddings = tf.concat([item_embeddings,padding],axis=0)
        behavior_embeddings = tf.Variable(tf.random.normal([self.behavior_num, self.hidden_size], 0.0, 0.01),
                                          name='behavior_embeddings')
        padding = tf.zeros([1,self.hidden_size],dtype= tf.float32)
        behavior_embeddings = tf.concat([behavior_embeddings,padding],axis=0)
        pos_embeddings=tf.Variable(tf.random.normal([self.state_size, self.hidden_size], 0.0, 0.01),
            name='pos_embeddings')
        all_embeddings['item_embeddings'] = item_embeddings
        all_embeddings['behavior_embeddings'] = behavior_embeddings
        all_embeddings['pos_embeddings'] = pos_embeddings
        return all_embeddings

    def global_graph_neural_network(self, h_0, inputs, item_embeddings, mask_p, mask_e):
        # h_0 = tf.nn.embedding_lookup(params=item_embeddings,ids=self.inputs)

        mask_padding = tf.ones_like(inputs) * self.item_num

        inputs_p = tf.where(mask_p, inputs, mask_padding)
        inputs_e = tf.where(mask_e, inputs, mask_padding)

        p2p = tf.nn.embedding_lookup(params=self.global_graph[0], ids=inputs_p)
        e2e = tf.nn.embedding_lookup(params=self.global_graph[1], ids=inputs_e)
        e2p_in = tf.nn.embedding_lookup(params=self.global_graph[2], ids=inputs_p)
        e2p_out = tf.nn.embedding_lookup(params=self.global_graph[3], ids=inputs_e)
        p2e_in = tf.nn.embedding_lookup(params=self.global_graph[4], ids=inputs_e)
        p2e_out = tf.nn.embedding_lookup(params=self.global_graph[5], ids=inputs_p)

        p2p_ids = p2p[:, :, :, 0]
        p2p_weight = tf.cast(p2p[:, :, :, 1], tf.float32)
        p2p_weight = p2p_weight / tf.expand_dims(tf.reduce_sum(p2p_weight, -1) + 1e-5, -1) # add 1e-5 to prevent division by zero
        p2p_weight = tf.expand_dims(p2p_weight, -1)
        # print(p2p_weight.shape.as_list())

        e2e_ids = e2e[:, :, :, 0]
        e2e_weight = tf.cast(e2e[:, :, :, 1], tf.float32)
        e2e_weight = e2e_weight / tf.expand_dims(tf.reduce_sum(e2e_weight, -1) + 1e-5, -1)
        e2e_weight = tf.expand_dims(e2e_weight, -1)

        e2p_in_ids = e2p_in[:, :, :, 0]
        e2p_in_weight = tf.cast(e2p_in[:, :, :, 1], tf.float32)
        e2p_in_weight = e2p_in_weight / tf.expand_dims(tf.reduce_sum(e2p_in_weight, -1) + 1e-5, -1)
        e2p_in_weight = tf.expand_dims(e2p_in_weight, -1)
        
        e2p_out_ids = e2p_out[:, :, :, 0]
        e2p_out_weight = tf.cast(e2p_out[:, :, :, 1], tf.float32)
        e2p_out_weight = e2p_out_weight / tf.expand_dims(tf.reduce_sum(e2p_out_weight, -1) + 1e-5, -1)
        e2p_out_weight = tf.expand_dims(e2p_out_weight, -1)
        
        p2e_in_ids = p2e_in[:, :, :, 0]
        p2e_in_weight = tf.cast(p2e_in[:, :, :, 1], tf.float32)
        p2e_in_weight = p2e_in_weight / tf.expand_dims(tf.reduce_sum(p2e_in_weight, -1) + 1e-5, -1)
        p2e_in_weight = tf.expand_dims(p2e_in_weight, -1)

        p2e_out_ids = p2e_out[:, :, :, 0]
        p2e_out_weight = tf.cast(p2e_out[:, :, :, 1], tf.float32)
        p2e_out_weight = p2e_out_weight / tf.expand_dims(tf.reduce_sum(p2e_out_weight, -1) + 1e-5, -1)
        p2e_out_weight = tf.expand_dims(p2e_out_weight, -1)

        h_1_p2p = tf.reduce_sum(tf.nn.embedding_lookup(item_embeddings, ids=p2p_ids) * p2p_weight, axis=2) 
        h_1_e2e = tf.reduce_sum(tf.nn.embedding_lookup(item_embeddings, ids=e2e_ids) * e2e_weight, axis=2) 
        h_1_e2p_in = tf.reduce_sum(tf.nn.embedding_lookup(item_embeddings, ids=e2p_in_ids) * e2p_in_weight, axis=2) 
        h_1_e2p_out = tf.reduce_sum(tf.nn.embedding_lookup(item_embeddings, ids=e2p_out_ids) * e2p_out_weight, axis=2) 
        h_1_p2e_in = tf.reduce_sum(tf.nn.embedding_lookup(item_embeddings, ids=p2e_in_ids) * p2e_in_weight, axis=2) 
        h_1_p2e_out = tf.reduce_sum(tf.nn.embedding_lookup(item_embeddings, ids=p2e_out_ids) * p2e_out_weight, axis=2)     

        weight_edge = tf.compat.v1.layers.dense(tf.concat([h_1_p2p, h_1_e2e, h_1_e2p_in, h_1_e2p_out, h_1_p2e_in, h_1_p2e_out], -1), 4, activation=tf.nn.sigmoid,name='weight_edge')

        w_p2p = tf.expand_dims(weight_edge[:,:,0], -1)
        w_e2e = tf.expand_dims(weight_edge[:,:,1], -1)
        w_e2p = tf.expand_dims(weight_edge[:,:,2], -1)
        w_p2e = tf.expand_dims(weight_edge[:,:,3], -1)

        h_1 = h_0 + (h_1_p2p * w_p2p + h_1_e2e * w_e2e + (h_1_e2p_in + h_1_e2p_out) * 0.5 * w_e2p + (h_1_p2e_in + h_1_p2e_out) * 0.5 * w_p2e) #*0.25?
        return h_1

    def local_graph_neural_network(self, h_0, inputs, item_embeddings):
        local_in = tf.nn.embedding_lookup(item_embeddings, self.local_graph_in)
        local_out = tf.nn.embedding_lookup(item_embeddings, self.local_graph_out)

        h_1_in = tf.reduce_mean(local_in, 2)
        h_1_out = tf.reduce_mean(local_out, 2)
        h_1 = h_0 + (h_1_in + h_1_out)
        return h_1
