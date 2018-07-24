
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import tensorflow.contrib.layers as layers
import architectures
import numpy as np
from sklearn import metrics

from flip_gradient import flip_gradient
import nn_utils

class PNet(object):

    def __init__(self,
                 config,
                 args,
                 word_vecs,
                 init = tf.random_uniform_initializer(minval=-0.01, maxval=0.01), # init = layers.xavier_initializer(),
                 name='PNet'):

        self.cfg       = config
        self.args      = args
        self.word_vecs =  word_vecs
        self.init      = init
        self.name      = name

        self.memory_size    = self.cfg.memory_size
        self.sent_size      = self.cfg.sent_size
        self.embed_size     = self.cfg.embed_size
        self.hidden_size    = self.cfg.hidden_size
        self.l2_reg_lambda  = self.cfg.l2_reg_lambda
        self.max_grad_norm  = self.cfg.max_grad_norm
        self.hops           = self.cfg.hops

        self.build_vars()
        self.build_eval_op()

    def build_vars(self):

        with tf.variable_scope(self.name):

            A = tf.convert_to_tensor(self.word_vecs)
            self.word2vec = tf.Variable(A, name="word2vec", trainable=True)
            self.nil_vars = set([self.word2vec.name])
            self.lr = tf.placeholder(tf.float32, [], name="learning_rate")
            self.adapt = tf.placeholder(tf.float32, [], name="adapt_rate")

            self.P_net  = architectures.HAN(self.cfg, self.word2vec, init=self.init, scope="P_net")

    def build_eval_op(self):

        self.reviews    = tf.placeholder(tf.int32, [None, self.memory_size, self.sent_size], name="reviews")
        self.labels     = tf.placeholder(tf.int32,   [None, 2], name="labels")

        with tf.variable_scope(self.name):

            self.p_reps,  self.p_word_attns,  self.p_sent_attns  = self.P_net(self.reviews, reuse=False)

            with tf.variable_scope('P_net'):
                self.sen_logits = nn_utils.fc_layer(self.p_reps, output_dim=2, scope="sentiment_classifier", reuse=False)
                self.sen_predictions = tf.argmax(self.sen_logits, 1, name="sen_predictions")
                sen_loss = tf.losses.softmax_cross_entropy(self.labels, self.sen_logits, weights=1.0, scope="sentiment_classifier")
                var_list = [var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if "sentiment_classifier" in var.name]
                reg_loss = self.l2_reg_lambda * sum(tf.nn.l2_loss(var) for var in var_list)
                self.loss = sen_loss + reg_loss

                self.dom_logits = nn_utils.fc_layer(self.p_reps, output_dim=2, scope="domain_classifier", reuse=False)
                self.dom_predictions = tf.argmax(self.dom_logits, 1, name="dom_predictions")

    def build_graph(self, sen_reviews, sen_labels, dom_reviews, dom_labels):

        with tf.variable_scope(self.name):

            sen_reps, _, _  = self.P_net(sen_reviews, reuse=True)
            dom_reps, _, _  = self.P_net(dom_reviews, reuse=True)

            with tf.variable_scope('P_net'):
                self.sen_loss = self.add_clf_loss(sen_reps, sen_labels, scope='sentiment_classifier')
                self.dom_loss = self.add_dann_loss(dom_reps, dom_labels, self.adapt, scope='domain_classifier')
                self.pivot_loss = self.sen_loss + self.dom_loss

    def add_clf_loss(self, cls_reps, cls_labels, num_labels=2, weight=1.0, scope=None):

        cls_logits = nn_utils.fc_layer(cls_reps, output_dim=num_labels, scope=scope, reuse=True)
        clf_loss = tf.losses.softmax_cross_entropy(cls_labels, cls_logits, weights=weight, scope=scope)

        var_list = [var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if scope in var.name]
        reg_loss = self.l2_reg_lambda * sum(tf.nn.l2_loss(var) for var in var_list)

        loss = clf_loss + reg_loss

        return loss

    def add_dann_loss(self, dom_reps, dom_labels, adapt_rate, weight=1.0, scope=None):

        dom_reps_grl = flip_gradient(dom_reps, adapt_rate)
        dom_logits = nn_utils.fc_layer(dom_reps_grl, output_dim=2, scope=scope, reuse=True)
        dom_loss = tf.losses.softmax_cross_entropy(dom_labels, dom_logits, weights=weight, scope=scope)

        var_list = [var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if scope in var.name]
        reg_loss = self.l2_reg_lambda * sum(tf.nn.l2_loss(var) for var in var_list)

        loss = dom_loss + reg_loss

        return loss

    def fc_layer(self, inputs, output_dim=None, activation=None, scope=None, reuse=False):

        with tf.variable_scope(scope, reuse=reuse):

            _, embed_dim = inputs.shape.as_list()
            W_fc = tf.get_variable(shape=[embed_dim, output_dim], name='W_fc', dtype=tf.float32)
            b_fc = tf.get_variable(shape=[output_dim], name='b_fc', dtype=tf.float32)

            if activation != None:
                outputs = activation(tf.matmul(inputs, W_fc) + b_fc)
            else:
                outputs = tf.matmul(inputs, W_fc) + b_fc

        return outputs

    def create_train_op(self):

        with tf.name_scope('train'):

            self.opt = tf.train.MomentumOptimizer(self.lr, 0.9)
            self.train_op = nn_utils.train_network(self.opt, self.pivot_loss, self.nil_vars, self.max_grad_norm, "train_op")

    def eval_sen(self, sess, reviews, sen_labels, batch_size=None):

        preds = []
        batch_generator = nn_utils.batch_generator(reviews=reviews, batch_size=batch_size, shuffle=False)
        for i in xrange(batch_generator.batch_num):
            xb = batch_generator.next_batch()
            pred = self.predict_sen(sess, xb)
            preds += list(pred)
        preds = np.array(preds)
        acc = metrics.accuracy_score(preds, np.argmax(sen_labels, axis=1))

        return acc, preds

    def eval_dom(self, sess, src_reviews, tar_reviews, src_labels, tar_labels, steps=None, batch_size=None):

        dom_preds  = []
        dom_labels = []
        src_batch_generator = nn_utils.batch_generator(reviews=src_reviews, labels=src_labels, batch_size=batch_size, shuffle=False)
        tar_batch_generator = nn_utils.batch_generator(reviews=tar_reviews, labels=tar_labels, batch_size=batch_size, shuffle=False)
        for i in xrange(steps):
            xb_s, yb_s = src_batch_generator.next_batch()
            xb_t, yb_t = tar_batch_generator.next_batch()
            xb_d       = np.vstack((xb_s, xb_t))
            dom_label  = np.vstack((yb_s, yb_t))
            dom_pred   = self.predict_dom(sess, xb_d)
            dom_preds  += list(dom_pred)
            dom_labels += list(dom_label)

        dom_acc = metrics.accuracy_score(np.array(dom_preds), np.argmax(dom_labels, axis=1))

        return dom_acc

    def vis_attention(self, sess, reviews, batch_size=None):

        p_w_atttns,  p_s_attns  = [], []
        batch_generator = nn_utils.batch_generator(reviews, batch_size=batch_size, shuffle=False)
        for i in xrange(batch_generator.batch_num):
            xb  = batch_generator.next_batch()
            w1, s1 = self.get_attention(sess, xb)
            p_w_atttns += list(w1)
            p_s_attns += list(s1)

        return np.array(p_w_atttns), np.array(p_s_attns)

    def initialize_session(self, sess):
        sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.save_path="./work/models/" + self.args.source_domain + '_' + self.args.target_domain + "_PNet.ckpt"
        print(self.save_path)

    def save_model(self, sess):
        self.saver.save(sess, self.save_path)

    def load_model(self, sess):
        try:
            self.saver.restore(sess, self.save_path)
        except Exception as e:
            raise IOError("Failed to to load model " "from save path: %s" % self.save_path)
        self.saver.restore(sess, self.save_path)
        print("Successfully load model from save path: %s" % self.save_path)

    def predict_sen(self, sess, reviews):

        feed_dict = {
            self.reviews: reviews
        }
        sen_predictions = sess.run(self.sen_predictions, feed_dict=feed_dict)

        return sen_predictions

    def predict_dom(self, sess, reviews):
        feed_dict = {
            self.reviews: reviews,
        }
        dom_predictions = sess.run(self.dom_predictions, feed_dict=feed_dict)

        return dom_predictions

    def eval_loss(self, sess, reviews, labels):

        feed_dict = {
            self.reviews: reviews,
            self.labels: labels
        }
        loss = sess.run(self.loss, feed_dict=feed_dict)

        return loss

    def get_attention(self, sess, reviews):

        feed_dict = {
            self.reviews: reviews
        }
        p_word_attns, p_sent_attns = sess.run([self.p_word_attns, self.p_sent_attns], feed_dict=feed_dict)

        return p_word_attns, p_sent_attns