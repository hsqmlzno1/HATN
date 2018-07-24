
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import tensorflow.contrib.layers as layers
import architectures
import numpy as np
from sklearn import metrics
import os

from flip_gradient import flip_gradient
import nn_utils

class HATN(object):

    def __init__(self,
                 config,
                 args,
                 word_vecs,
                 init = tf.random_uniform_initializer(minval=-0.01, maxval=0.01), # init = layers.xavier_initializer(),
                 name='HATN'):

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
            self.w_pe = tf.Variable(self.init([1, self.sent_size, self.embed_size]), name='w_pe')
            self.s_pe = tf.Variable(self.init([1, self.memory_size, self.hidden_size]), name='s_pe')
            self.nil_vars = set([self.word2vec.name])
            self.lr = tf.placeholder(tf.float32, [], name="learning_rate")
            self.adapt = tf.placeholder(tf.float32, [], name="adapt_rate")

            self.P_net  = architectures.HAN(self.cfg, self.word2vec, self.w_pe, self.s_pe, init=self.init, scope="P_net")
            self.NP_net = architectures.HAN(self.cfg, self.word2vec, self.w_pe, self.s_pe, init=self.init, scope="NP_net")

    def build_eval_op(self):

        self.reviews    = tf.placeholder(tf.int32, [None, self.memory_size, self.sent_size], name="reviews")
        self.word_mask  = tf.placeholder(tf.float32, [None, self.memory_size, self.sent_size], name="word_mask")
        self.sent_mask  = tf.placeholder(tf.float32, [None, self.memory_size], name="sent_mask")
        self.labels     = tf.placeholder(tf.int32,   [None, 2], name="labels")

        with tf.variable_scope(self.name):

            self.p_reps,  self.p_word_attns,  self.p_sent_attns  = self.P_net(self.reviews, reuse=False)
            self.np_reps, self.np_word_attns, self.np_sent_attns = self.NP_net(self.reviews, self.word_mask, self.sent_mask, reuse=False)
            self.sen_reps = tf.concat([self.p_reps, self.np_reps], -1)

            with tf.variable_scope('P_net'):
                self.dom_logits = nn_utils.fc_layer(self.p_reps, output_dim=2, scope="domain_classifier", reuse=False)
                self.dom_predictions = tf.argmax(self.dom_logits, 1, name="dom_predictions")

            with tf.variable_scope('NP_net'):
                self.pos_logits = nn_utils.fc_layer(self.np_reps, output_dim=2, scope="pos_pivot_predictor", reuse=False)
                self.pos_predictions = tf.argmax(self.pos_logits, 1, name="pos_predictions")

                self.neg_logits = nn_utils.fc_layer(self.np_reps, output_dim=2, scope="neg_pivot_predictor", reuse=False)
                self.neg_predictions = tf.argmax(self.neg_logits, 1, name="neg_predictions")

            with tf.variable_scope('joint'):
                self.sen_logits = nn_utils.fc_layer(self.sen_reps, output_dim=2, scope="sentiment_classifier", reuse=False)
                self.sen_predictions = tf.argmax(self.sen_logits, 1, name="sen_predictions")
                sen_loss = tf.losses.softmax_cross_entropy(self.labels, self.sen_logits, weights=1.0, scope="sentiment_classifier")
                var_list = [var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if "sentiment_classifier" in var.name]
                reg_loss = self.l2_reg_lambda * sum(tf.nn.l2_loss(var) for var in var_list)
                self.loss = sen_loss + reg_loss

    def build_graph(self, sen_reviews, sen_word_mask, sen_sent_mask, sen_labels, sen_u, sen_v,
                          dom_reviews, dom_word_mask, dom_sent_mask, dom_labels, dom_u, dom_v):

        with tf.variable_scope(self.name):

            p_reps,   _, _  = self.P_net(sen_reviews, reuse=True)
            dom_reps, _, _  = self.P_net(dom_reviews, reuse=True)

            np_reps, _, _  = self.NP_net(sen_reviews, sen_word_mask, sen_sent_mask, reuse=True)
            aux_reps, _, _ = self.NP_net(dom_reviews, dom_word_mask, dom_sent_mask, reuse=True)

            sen_reps = tf.concat([p_reps, np_reps], -1)

            with tf.variable_scope('P_net'):
                self.dom_loss = self.add_dann_loss(dom_reps, dom_labels, self.adapt, scope='domain_classifier')
            with tf.variable_scope('NP_net'):
                self.lab_pos_loss   = self.add_clf_loss(np_reps, sen_u, scope='pos_pivot_predictor')
                self.lab_neg_loss   = self.add_clf_loss(np_reps, sen_v, scope='neg_pivot_predictor')
                self.unlab_pos_loss = self.add_clf_loss(aux_reps, dom_u, scope='pos_pivot_predictor')
                self.unlab_neg_loss = self.add_clf_loss(aux_reps, dom_v, scope='neg_pivot_predictor')
                self.lab_aux_loss   = self.lab_pos_loss + self.lab_neg_loss
                self.unlab_aux_loss = self.unlab_pos_loss + self.unlab_neg_loss
            with tf.variable_scope('joint'):
                self.sen_loss = self.add_clf_loss(sen_reps, sen_labels, scope='sentiment_classifier')
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

    def create_train_op(self):

        with tf.name_scope('train'):

            self.opt = tf.train.MomentumOptimizer(self.lr, 0.9)
            self.train_joint_op = nn_utils.train_network(self.opt, self.pivot_loss+self.lab_aux_loss, self.nil_vars, self.max_grad_norm, "joint_train")
            self.train_aux_op   = nn_utils.train_network(self.opt, self.unlab_aux_loss, self.nil_vars, self.max_grad_norm, "aux_train")
            self.train_dom_op   = nn_utils.train_network(self.opt, self.dom_loss, self.nil_vars, self.max_grad_norm, "dom_train")

    def eval_sen(self, sess, reviews, word_masks, sent_masks, sen_labels, batch_size=None):

        preds = []
        batch_generator = nn_utils.batch_generator(reviews=reviews, word_masks=word_masks, sent_masks=sent_masks, batch_size=batch_size, shuffle=False)
        for i in xrange(batch_generator.batch_num):
            xb, wmb, smb = batch_generator.next_batch()
            pred = self.predict_sen(sess, xb, wmb, smb)
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

    def eval_pivots(self, sess, reviews, word_masks, sent_masks, u_labels, v_labels, batch_size=None):

        u_preds = []
        v_preds = []
        batch_generator = nn_utils.batch_generator(reviews=reviews, word_masks=word_masks, sent_masks=sent_masks, batch_size=batch_size, shuffle=False)
        for i in xrange(batch_generator.batch_num):
            xb, wmb, smb = batch_generator.next_batch()
            u_pred, v_pred = self.predict_pivots(sess, xb, wmb, smb)
            u_preds += list(u_pred)
            v_preds += list(v_pred)

        u_acc = metrics.accuracy_score(np.array(u_preds), np.argmax(u_labels, axis=1))
        v_acc = metrics.accuracy_score(np.array(v_preds), np.argmax(v_labels, axis=1))

        return u_acc, v_acc

    def vis_attention(self, sess, reviews, word_masks, sent_masks, batch_size=None):

        p_w_atttns,  p_s_attns  = [], []
        np_w_atttns, np_s_attns = [], []
        batch_generator = nn_utils.batch_generator(reviews=reviews, word_masks=word_masks, sent_masks=sent_masks, batch_size=batch_size, shuffle=False)
        for i in xrange(batch_generator.batch_num):
            xb, wmb, smb   = batch_generator.next_batch()
            w1, s1, w2, s2 = self.get_attention(sess, xb, wmb, smb)
            p_w_atttns += list(w1)
            p_s_attns += list(s1)
            np_w_atttns += list(w2)
            np_s_attns += list(s2)

        return np.array(p_w_atttns), np.array(p_s_attns), np.array(np_w_atttns), np.array(np_s_attns)

    def initialize_session(self, sess):
        sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        output_dir = "./work/models/"
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        self.save_path="./work/models/" + self.args.source_domain + '_' + self.args.target_domain + "_HATN.ckpt"

    def save_model(self, sess):
        self.saver.save(sess, self.save_path)

    def load_model(self, sess):
        try:
            self.saver.restore(sess, self.save_path)
        except Exception as e:
            raise IOError("Failed to to load model " "from save path: %s" % self.save_path)
        self.saver.restore(sess, self.save_path)
        print("Successfully load model from save path: %s" % self.save_path)

    def predict_sen(self, sess, reviews, word_mask, sent_mask):

        feed_dict = {
            self.reviews: reviews,
            self.word_mask: word_mask,
            self.sent_mask: sent_mask
        }
        sen_predictions = sess.run(self.sen_predictions, feed_dict=feed_dict)

        return sen_predictions

    def predict_dom(self, sess, reviews):
        feed_dict = {
            self.reviews: reviews,
        }
        dom_predictions = sess.run(self.dom_predictions, feed_dict=feed_dict)

        return dom_predictions

    def predict_pivots(self, sess, reviews, word_mask, sent_mask):
        feed_dict = {
            self.reviews: reviews,
            self.word_mask: word_mask,
            self.sent_mask: sent_mask
        }
        pos_predictions, neg_predictions = sess.run([self.pos_predictions, self.neg_predictions],
                                                    feed_dict=feed_dict)

        return pos_predictions, neg_predictions

    def eval_loss(self, sess, reviews, word_mask, sent_mask, labels):

        feed_dict = {
            self.reviews: reviews,
            self.word_mask: word_mask,
            self.sent_mask: sent_mask,
            self.labels: labels
        }
        loss = sess.run(self.loss, feed_dict=feed_dict)

        return loss

    def get_attention(self, sess, reviews, word_mask, sent_mask):

        feed_dict = {
            self.reviews: reviews,
            self.word_mask: word_mask,
            self.sent_mask: sent_mask
        }
        p_word_attns, p_sent_attns, np_word_attns, np_sent_attns \
            = sess.run([self.p_word_attns, self.p_sent_attns,
                        self.np_word_attns, self.np_sent_attns], feed_dict=feed_dict)

        return p_word_attns, p_sent_attns, np_word_attns, np_sent_attns