# coding:utf-8
from __future__ import print_function
import tensorflow as tf
from config import *
import numpy as np
from numpy import *
import argparse
import matplotlib
matplotlib.use('Agg')

from data_utils_hatn import *
from models import HATN
np.random.seed(FLAGS.random_seed)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--source_domain', '-s', type=str,
                        choices=['books', 'dvd', 'electronics', 'kitchen', 'video'],
                        default='electronics')
    parser.add_argument('--target_domain', '-t', type=str,
                        choices=['books', 'dvd', 'electronics', 'kitchen', 'video'],
                        default='kitchen')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    source_domain = args.source_domain
    target_domain = args.target_domain
    data_path     = FLAGS.data_path

    print("loading data...")
    train_data, val_data, test_data, source_unlabeled_data, target_unlabeled_data, vocab \
        = load_data(source_domain, target_domain, data_path)
    pos_pivot, neg_pivot = get_all_pivots(source_domain, target_domain)

    data = train_data + val_data + test_data + source_unlabeled_data + target_unlabeled_data
    source_data = train_data+val_data+source_unlabeled_data
    target_data = target_unlabeled_data

    max_story_size  = max(map(len, (pairs[0] for pairs in data)))
    mean_story_size = int(np.mean([len(pairs[0]) for pairs in data]))
    sentences = map(len, (sentence for pairs in data for sentence in pairs[0]))
    max_sentence_size = max(sentences)
    mean_sentence_size = int(mean(sentences))
    memory_size = min(FLAGS.memory_size, max_story_size)
    print("max  story size:", max_story_size)
    print("mean story size:", mean_story_size)
    print("max  sentence size:", max_sentence_size)
    print("mean sentence size:", mean_sentence_size)
    print("max memory size:", memory_size)
    max_sentence_size = FLAGS.sent_size

    word_embedding, word2idx, idx2word = get_w2vec(vocab, FLAGS)
    vocab_size = len(word_embedding)

    x_train,  _,      y_train, u_train, v_train, word_mask_train, sent_mask_train = vectorize_data(train_data,  pos_pivot, neg_pivot, word2idx, memory_size, max_sentence_size)
    x_val,    _,      y_val,   u_val,   v_val,   word_mask_val,   sent_mask_val   = vectorize_data(val_data,    pos_pivot, neg_pivot, word2idx, memory_size, max_sentence_size)
    x_test,   _,      y_test,  u_test,  v_test,  word_mask_test,  sent_mask_test  = vectorize_data(test_data,   pos_pivot, neg_pivot, word2idx, memory_size, max_sentence_size)
    x_s,      d_s,    _,       u_s,     v_s,     word_mask_s,     sent_mask_s     = vectorize_data(source_data, pos_pivot, neg_pivot, word2idx, memory_size, max_sentence_size)
    x_t,      d_t,    _,       u_t,     v_t,     word_mask_t,     sent_mask_t     = vectorize_data(target_data, pos_pivot, neg_pivot, word2idx, memory_size, max_sentence_size)

    n_train  = x_train.shape[0]
    n_test   = x_test.shape[0]
    n_val    = x_val.shape[0]
    n_source = x_s.shape[0]
    n_target = x_t.shape[0]
    n_domain = min(n_source, n_target)
    print(n_train, n_val, n_test, n_source, n_target)

    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(FLAGS.random_seed)

        model = HATN(FLAGS, args, word_embedding)

        xb_train, wmb_train, smb_train, yb_train, ub_train, vb_train = create_input(x_train, word_mask_train, sent_mask_train, y_train, u_train, v_train, FLAGS.batch_size)
        xb_s, wmb_s, smb_s, db_s, ub_s, vb_s = create_input(x_s, word_mask_s, sent_mask_s, d_s, u_s, v_s, FLAGS.batch_size)
        xb_t, wmb_t, smb_t, db_t, ub_t, vb_t = create_input(x_t, word_mask_t, sent_mask_t, d_t, u_t, v_t, FLAGS.batch_size)

        xb_d  = tf.concat([xb_s, xb_t], 0)
        wmb_d = tf.concat([wmb_s, wmb_t], 0)
        smb_d = tf.concat([smb_s, smb_t], 0)
        db_d  = tf.concat([db_s, db_t], 0)
        ub_d  = tf.concat([ub_s, ub_t], 0)
        vb_d  = tf.concat([vb_s, vb_t], 0)

        model.build_graph(xb_train, wmb_train, smb_train, yb_train, ub_train, vb_train,
                          xb_d, wmb_d, smb_d, db_d, ub_d, vb_d)
        model.create_train_op()

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(graph=graph, config=config) as sess:
        model.initialize_session(sess)

        best_epoch = 0
        best_val_acc = 0
        best_val_loss = float('inf')
        counter = 0

        if args.train:

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            # begin training
            steps = int(math.floor(1.0 * n_train / FLAGS.batch_size))
            # num_steps = steps_per_epoch * FLAGS.max_epoch

            for epoch in xrange(1, FLAGS.max_epoch + 1):

                p = float(epoch - 1) / FLAGS.max_epoch
                lr    = max(0.005 / (1. + 10 * p) ** 0.75, 0.002)
                adapt = min(2. / (1. + np.exp(-10. * p)) - 1, 0.1)
                if args.verbose:
                    print('adapt',adapt,'lr',lr)

                loss = 0.0
                sen_loss = 0.0
                dom_loss = 0.0
                sen_aux_loss = 0.0
                unlabel_aux_loss = 0.0
                for step in xrange(steps):

                    _, _, sen_cost, dom_cost, sen_aux_cost, unlabel_aux_cost = sess.run([model.train_joint_op, model.train_aux_op,
                                                                                         model.sen_loss, model.dom_loss, model.lab_aux_loss, model.unlab_aux_loss],
                                                                                        feed_dict={model.lr:lr, model.adapt:adapt})

                    sen_loss += sen_cost
                    dom_loss += dom_cost
                    sen_aux_loss += sen_aux_cost
                    unlabel_aux_loss += unlabel_aux_cost
                    loss += (sen_cost + dom_cost + sen_aux_cost + unlabel_aux_cost)

                # validation accuracy
                val_acc, _  = model.eval_sen(sess, x_val, word_mask_val, sent_mask_val, y_val, batch_size=FLAGS.batch_size)
                val_loss    = model.eval_loss(sess, x_val, word_mask_val, sent_mask_val, y_val)

                if epoch % FLAGS.evaluation_interval == 0:

                    train_acc, _ = model.eval_sen(sess, x_train, word_mask_train, sent_mask_train, y_train, batch_size=FLAGS.batch_size)
                    dom_acc      = model.eval_dom(sess, x_s, x_t, d_s, d_t, steps, batch_size=FLAGS.batch_size)

                if args.verbose:
                    print("Epoch: [%-3d] loss: %4.8f, sen-loss: %4.8f, dom-loss: %4.8f, src-aux-loss: %4.8f, tar-aux-loss: %4.8f" % (
                           epoch, loss, sen_loss, dom_loss, sen_aux_loss, unlabel_aux_loss))
                    print("Epoch: [%-3d] train-acc: %.8f, dom-acc: %.8f, val-acc: %.8f, val_loss: %.8f" % (
                           epoch, train_acc, dom_acc, val_acc, val_loss))
                    print("---------------------------------------------------\n")

                if best_val_loss > val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    counter = 0
                    model.save_model(sess)
                else:
                    counter += 1

                if counter == FLAGS.patience:
                    coord.request_stop()
                    coord.join(threads)
                    break

        if args.test:

            model.load_model(sess)
            train_acc, train_preds = model.eval_sen(sess, x_train, word_mask_train, sent_mask_train, y_train, batch_size=FLAGS.batch_size)
            test_acc,  test_preds  = model.eval_sen(sess, x_test, word_mask_test, sent_mask_test, y_test, batch_size=FLAGS.batch_size)
            print("Best Epoch: [%3d] best val accuracy: %.8f best val loss: %.8f" % (best_epoch, best_val_acc, best_val_loss))
            print("Testing accuracy: %.8f" % (test_acc))
            store_results("HATN", source_domain, target_domain, test_acc)

            p_w_atttns, p_s_attns, np_w_atttns, np_s_attns = model.vis_attention(sess, x_train, word_mask_train, sent_mask_train, batch_size=FLAGS.batch_size)
            visualization(x_train,
                          np.argmax(y_train, axis=1), train_preds,
                          np.argmax(u_train, axis=1), np.argmax(v_train, axis=1),
                          p_w_atttns, p_s_attns,
                          np_w_atttns, np_s_attns,
                          word_mask_train, sent_mask_train,
                          idx2word, source_domain+'_'+target_domain, "train")

            p_w_atttns, p_s_attns, np_w_atttns, np_s_attns = model.vis_attention(sess, x_test, word_mask_test, sent_mask_test, batch_size=FLAGS.batch_size)
            visualization(x_test,
                          np.argmax(y_test, axis=1), test_preds,
                          np.argmax(u_test, axis=1), np.argmax(v_test, axis=1),
                          p_w_atttns, p_s_attns,
                          np_w_atttns, np_s_attns,
                          word_mask_test, sent_mask_test,
                          idx2word, source_domain+'_'+target_domain, "test")
