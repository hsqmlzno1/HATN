
import tensorflow as tf
from nn_utils import *
import numpy as np

class HAN(object):

    def __init__(self,
                 cfg,
                 word2vec,
                 word_pe=None,
                 sent_pe=None,
                 init=None,
                 scope=None):

        self.memory_size = cfg.memory_size
        self.sent_size   = cfg.sent_size
        self.embed_size  = cfg.embed_size
        self.hidden_size = cfg.hidden_size
        self.hops        = cfg.hops

        self.word2vec = word2vec
        self.word_pe  = word_pe
        self.sent_pe  = sent_pe
        self.init     = init
        self.scope    = scope

        self.build_vars()

    def build_vars(self):

        with tf.variable_scope(self.scope):
            self.word_query = tf.Variable(self.init([1, self.hidden_size]), name='word_query')
            self.sent_query = tf.Variable(self.init([1, self.hidden_size]), name='sent_query')

    def __call__(self, reviews, word_mask=None, sent_mask=None, reuse=False):

        with tf.variable_scope(self.scope, reuse=reuse):

            with tf.variable_scope("embedding_layer"):
                self.emb = tf.nn.embedding_lookup(self.word2vec, reviews)
                if word_mask == None and sent_mask == None:
                    word_mask, sent_mask = self.get_masks(reviews)

            if self.word_pe == None:
                word_reps = tf.reshape(self.emb,[-1, self.sent_size, self.embed_size])
            else:
                word_reps = tf.reshape(self.emb, [-1, self.sent_size, self.embed_size]) + self.word_pe

            with tf.variable_scope('word_attention') as sc:

                W_w = tf.get_variable(shape=[self.embed_size, self.hidden_size], name='W_w', dtype=tf.float32)
                b_w = tf.get_variable(shape=[self.hidden_size], name='b_w', dtype=tf.float32)
                query = self.word_query
                for i in xrange(self.hops):
                    sent_reps, query, word_attentions = self.attention_layer(word_reps,
                                                                             query, word_mask,
                                                                             W_w, b_w, scope=sc)
                    word_attentions = tf.reshape(word_attentions, [-1, self.memory_size, self.sent_size])

            if self.sent_pe == None:
                sent_reps = tf.reshape(sent_reps, [-1, self.memory_size, self.hidden_size])
            else:
                sent_reps = tf.reshape(sent_reps, [-1, self.memory_size, self.hidden_size]) + self.sent_pe

            with tf.variable_scope('sent_attention') as sc:

                W_c = tf.get_variable(shape=[self.hidden_size, self.hidden_size], name='W_c', dtype=tf.float32)
                b_c = tf.get_variable(shape=[self.hidden_size], name='b_c', dtype=tf.float32)
                query = self.sent_query
                for i in xrange(self.hops):
                    doc_reps, query, sentence_attentions = self.attention_layer(sent_reps,
                                                                                query, sent_mask,
                                                                                W_c, b_c, scope=sc)
                    sentence_attentions = tf.reshape(sentence_attentions, [-1, self.memory_size])

            reps = doc_reps

        return reps, word_attentions, sentence_attentions

    def attention_layer(self, inputs, query, mask, W, b, activation=tf.tanh, scope=None):

        with tf.variable_scope(scope):

            _, element_size, embed_dim = inputs.shape.as_list()
            mask_re = tf.reshape(mask, [-1, mask.shape.as_list()[-1]])
            mask_re = tf.cast(mask_re, dtype=tf.float32)

            inputs_flat = tf.reshape(inputs, [-1, embed_dim])
            projection  = activation(tf.matmul(inputs_flat, W) + b)
            projection  = tf.reshape(projection, [-1, element_size, projection.shape.as_list()[-1]])

            query_t = tf.squeeze(query)
            attention_weights = tf.reduce_sum(tf.multiply(projection, query_t), axis = 2)
            attention_weights = tf.expand_dims(mask_softmax(attention_weights, axis=1, mask=mask_re), -1)

            outputs = tf.reduce_sum(projection * attention_weights, 1)
            new_query = query + tf.reduce_sum(outputs, axis=0)

            return outputs, new_query, attention_weights

    def get_masks(self, reviews):

        word_zeros_mat = tf.zeros_like(reviews)
        word_mask = tf.cast(tf.not_equal(reviews, word_zeros_mat), dtype=tf.float32)
        word_mask = tf.reshape(word_mask, [-1, self.sent_size])

        sentence_reviews = tf.reshape(tf.reduce_sum(reviews, 2), [-1, self.memory_size])
        sentence_zeros_mat = tf.zeros_like(sentence_reviews)
        sentence_mask = tf.cast(tf.not_equal(sentence_reviews, sentence_zeros_mat), dtype=tf.float32)

        return word_mask, sentence_mask