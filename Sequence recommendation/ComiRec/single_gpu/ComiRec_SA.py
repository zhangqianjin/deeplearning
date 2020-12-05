import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
import math
class ComiRec_SA(Model):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, num_interest, seq_len=256, add_pos=True):
        super(ComiRec_SA, self).__init__()
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
        if self.add_pos:
            self.position_embedding = tf.Variable(tf.random.uniform(shape=[1, self.seq_len, self.embedding_dim], minval=-1/math.sqrt(self.embedding_dim), maxval=1/math.sqrt(self.embedding_dim)))
        self.item_hidden = tf.keras.layers.Dense(self.hidden_size * 4, activation=tf.nn.tanh)
        self.item_att_w = tf.keras.layers.Dense(self.num_interest, activation=None) 

    def call(self, item_id, hist_item, hist_mask):
        mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, item_id)
        mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, hist_item)

        item_his_eb = mid_his_batch_embedded * tf.reshape(hist_mask, (-1, self.seq_len, 1))
        item_list_emb = tf.reshape(item_his_eb, [-1, self.seq_len, self.embedding_dim])
        if self.add_pos:
            item_list_add_pos = item_list_emb + tf.tile(self.position_embedding, [item_list_emb.shape[0], 1, 1])
        else:
            item_list_add_pos = item_list_emb

        num_heads = self.num_interest
        item_hidden = self.item_hidden(item_list_add_pos)
        item_att_w = self.item_att_w(item_hidden)
        item_att_w = tf.transpose(item_att_w, [0, 2, 1])

        atten_mask = tf.tile(tf.expand_dims(hist_mask, axis=1), [1, num_heads, 1])
        paddings = tf.ones_like(atten_mask) * (-2 ** 32 + 1)

        item_att_w = tf.where(tf.equal(atten_mask, 0), paddings, item_att_w)
        item_att_w = tf.nn.softmax(item_att_w)

        interest_emb = tf.matmul(item_att_w, item_list_emb)
        atten = tf.matmul(interest_emb, tf.reshape(mid_batch_embedded, [item_list_emb.shape[0], self.embedding_dim, 1]))
        atten = tf.nn.softmax(tf.pow(tf.reshape(atten, [item_list_emb.shape[0], num_heads]), 1))
        user_emb = tf.gather(tf.reshape(interest_emb, [-1, self.embedding_dim]), tf.argmax(atten, axis=1, output_type=tf.int32) + tf.range(item_list_emb.shape[0]) * num_heads)
        loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(self.mid_embeddings_var, self.mid_embeddings_bias,
                                                              tf.reshape(item_id, [-1, 1]), user_emb,
                                                             self.neg_num * self.batch_size, self.n_mid))
        return loss

    def output_item(self):
        return self.mid_embeddings_var


    def output_user(self, hist_item, hist_mask):
        mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, tf.cast(hist_item,tf.int32))
        test = tf.reshape(hist_mask, (-1, self.seq_len, 1))
        item_his_eb = mid_his_batch_embedded * tf.reshape(hist_mask, (-1, self.seq_len, 1))
        item_list_emb = tf.reshape(item_his_eb, [-1, self.seq_len, self.embedding_dim])
        if self.add_pos:
            item_list_add_pos = item_list_emb + tf.tile(self.position_embedding, [item_list_emb.shape[0], 1, 1])
        else:
            item_list_add_pos = item_list_emb

        num_heads = self.num_interest
        item_hidden = self.item_hidden(item_list_add_pos)
        item_att_w = self.item_att_w(item_hidden)
        item_att_w = tf.transpose(item_att_w, [0, 2, 1])

        atten_mask = tf.tile(tf.expand_dims(hist_mask, axis=1), [1, num_heads, 1])
        paddings = tf.ones_like(atten_mask) * (-2 ** 32 + 1)

        item_att_w = tf.where(tf.equal(atten_mask, 0), paddings, item_att_w)
        item_att_w = tf.nn.softmax(item_att_w)

        interest_emb = tf.matmul(item_att_w, item_list_emb)
        return interest_emb
