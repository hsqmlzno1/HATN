
import tensorflow as tf
import numpy as np
import math

def mask_softmax(target, axis, mask, epsilon=1e-12, name=None):
    with tf.name_scope(name, 'softmax',[target]):
        max_axis = tf.reduce_max(target, axis, keep_dims=True)
        target_exp = tf.exp(target - max_axis) * mask
        normalize = tf.reduce_sum(target_exp, axis, keep_dims=True)
        softmax = target_exp / (normalize + epsilon)
        return softmax

def add_gradient_noise(t, stddev=1e-3, name=None):
    """
    Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].

    The input Tensor `t` should be a gradient.

    The output will be `t` + gaussian noise.

    0.001 was said to be a good fixed value for memory networks [2].
    """
    with tf.name_scope(name, "add_gradient_noise", [t, stddev]) as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)

def zero_nil_slot(t, name=None):
    """
    Overwrites the nil_slot (first row) of the input Tensor with zeros.

    The nil_slot is a dummy slot and should not be trained and influence
    the training algorithm.
    """
    with tf.name_scope(name, "zero_nil_slot", [t]) as name:
        t = tf.convert_to_tensor(t, name="t")
        s = tf.shape(t)[1]
        z = tf.zeros([1, s])
        return tf.concat([z, tf.slice(t, [1, 0], [-1, -1])], 0, name=name)

def fc_layer(inputs, output_dim=None, activation=None, scope=None, reuse=False):

    with tf.variable_scope(scope, reuse=reuse):

        _, embed_dim = inputs.shape.as_list()
        W_fc = tf.get_variable(shape=[embed_dim, output_dim], name='W_fc', dtype=tf.float32)
        b_fc = tf.get_variable(shape=[output_dim], name='b_fc', dtype=tf.float32)

        if activation != None:
            outputs = activation(tf.matmul(inputs, W_fc) + b_fc)
        else:
            outputs = tf.matmul(inputs, W_fc) + b_fc

    return outputs

def train_network(opt, loss, nil_vars, max_grad_norm, scope):

    grads_and_vars = opt.compute_gradients(loss)
    print(scope)
    for g, v in grads_and_vars:
        if g is not None:
            print(v)
    grads_and_vars = [(tf.clip_by_norm(g, max_grad_norm), v) for g, v in grads_and_vars if g is not None]
    grads_and_vars = [(add_gradient_noise(g), v) for g, v in grads_and_vars]
    nil_grads_and_vars = []
    for g, v in grads_and_vars:
        if v.name in nil_vars:
            nil_grads_and_vars.append((zero_nil_slot(g), v))
        else:
            nil_grads_and_vars.append((g, v))
    train_op = opt.apply_gradients(nil_grads_and_vars, name=scope)

    return train_op

class batch_generator(object):

    def __init__(self,
                 reviews,
                 word_masks=None,
                 sent_masks=None,
                 labels=None,
                 batch_size=100,
                 shuffle=False):

        self.reviews = reviews
        self.word_mask = word_masks
        self.sent_mask = sent_masks
        self.labels    = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sample_num = reviews.shape[0]
        self.batch_num = int(math.ceil(self.sample_num / float(batch_size)))
        self.idx = [i for i in xrange(self.sample_num)]
        self.start_new_epoch()

    def get_reviews(self):
        return self.reviews[self.idx]

    def get_masks(self):
        return self.word_mask[self.idx], self.sent_mask[self.idx]

    def get_labels(self):
        return self.labels[self.idx]

    def get_number(self):
        return self.sample_num

    def start_new_epoch(self):
        self.batch_counter = 0
        if self.shuffle:
            self.shuffle_data()

    def start_new_epoch_without_shuffle(self):
        self.batch_counter = 0

    def no_shuffle(self):
        self.idx = [i for i in xrange(self.sample_num)]

    def shuffle_data(self):
        np.random.shuffle(self.idx)

    def next_batch(self):
        start = self.batch_counter * self.batch_size
        end = (self.batch_counter + 1) * self.batch_size
        self.batch_counter += 1
        reviews_slice = self.reviews[self.idx[start: end]]

        if self.word_mask is not None:
            word_mask_slice = self.word_mask[self.idx[start:end]]
            sent_mask_slice =self.sent_mask[self.idx[start:end]]

            return reviews_slice, word_mask_slice, sent_mask_slice

        elif self.labels is not None:
            labels_slice = self.labels[self.idx[start:end]]

            return reviews_slice, labels_slice

        else:

            return reviews_slice
