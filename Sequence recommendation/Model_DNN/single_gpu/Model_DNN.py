import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras import Model
import math
class Model(keras.Model):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, num_interest, seq_len=256, add_pos=False):
        super(Model, self).__init__()
        self.embedding_dim = embedding_dim
        self.reg = False
        self.batch_size = batch_size
        self.n_mid = n_mid
        self.neg_num = 10
        self.seq_len = seq_len
        self.num_interest = num_interest
        self.hidden_size = hidden_size
        self.add_pos = add_pos
        #self.mid_embeddings_var = tf.keras.layers.Embedding(input_dim=n_mid, output_dim=embedding_dim)
        self.mid_embeddings_var = tf.Variable(tf.random.uniform(shape=[n_mid,embedding_dim], minval=-1/math.sqrt(self.embedding_dim), maxval=1/math.sqrt(self.embedding_dim)))
        self.mid_embeddings_bias = tf.Variable(tf.zeros([n_mid]), trainable=False)
        self.user_emb = tf.keras.layers.Dense(self.hidden_size, activation=None) 

    def call(self, item_id, hist_item, hist_mask):
        mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, item_id)
        mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, hist_item)

        item_his_eb = mid_his_batch_embedded * tf.reshape(hist_mask, (-1, self.seq_len, 1))
        item_list_emb = tf.reshape(item_his_eb, [-1, self.seq_len, self.embedding_dim])
        masks = tf.concat([tf.expand_dims(hist_mask, -1) for _ in range(self.embedding_dim)], axis=-1)

        item_his_emb_mean = tf.reduce_sum(item_his_eb, 1) / (tf.reduce_sum(tf.cast(masks, dtype=tf.float32), 1) + 1e-9)
        user_emb = self.user_emb(item_his_emb_mean)
        loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(self.mid_embeddings_var, self.mid_embeddings_bias,
                                                              tf.reshape(item_id, [-1, 1]), user_emb,
                                                             self.neg_num * self.batch_size, self.n_mid))
        return loss

    def output_item(self):
        return self.mid_embeddings_var


    def output_user(self, hist_item, hist_mask):
        mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, tf.cast(hist_item,tf.int32))
        item_his_eb = mid_his_batch_embedded * tf.reshape(hist_mask, (-1, self.seq_len, 1))
        item_list_emb = tf.reshape(item_his_eb, [-1, self.seq_len, self.embedding_dim])
        masks = tf.concat([tf.expand_dims(hist_mask, -1) for _ in range(self.embedding_dim)], axis=-1)

        item_his_emb_mean = tf.reduce_sum(item_his_eb, 1) / (tf.reduce_sum(tf.cast(masks, dtype=tf.float32), 1) + 1e-9)
        user_emb = self.user_emb(item_his_emb_mean)
        return user_emb
