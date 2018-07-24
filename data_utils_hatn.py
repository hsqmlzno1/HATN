import numpy as np
from numpy import *
import tensorflow as tf
import os
import re
import pickle
import nltk
from utils import *

def get_all_pivots(source_domain, target_domain):

    fname = "./work/pivots/%s_%s_pos.txt" % (source_domain, target_domain)
    pos_pivot = get_pivot_list(fname)
    print pos_pivot

    fname = "./work/pivots/%s_%s_neg.txt" % (source_domain, target_domain)
    neg_pivot = get_pivot_list(fname)
    print neg_pivot

    return pos_pivot, neg_pivot

def get_pivot_list(fname):

    pivot = []
    with open(fname) as f:
        lines = f.readlines()
        for line in lines:
            word = line.strip().split(" ")[0]
            if len(word) != 0:
                pivot.append(word)
    return pivot

def vectorize_data(data, pos_pivot, neg_pivot, word2idx, memory_size, sentence_size):

    reviews = []
    word_masks = []
    sentence_masks = []
    labels = []
    domains = []
    u_labels  = []
    v_labels  = []

    for review, domain, label in data:

        label_idx  = [0] * 2
        label_idx[1 - label] = 1
        domain_idx = [0] * 2
        domain_idx[1 - domain] = 1

        review_idx = []
        word_mask = []
        sentence_mask = []
        u_label  = [0, 1]
        v_label  = [0, 1]

        for i, sentence in enumerate(review, 1):

            mask_idx = []
            for word in sentence:
                if word in pos_pivot:
                    mask_idx.append(0)
                    u_label = [1, 0]
                elif word in neg_pivot:
                    mask_idx.append(0)
                    v_label = [1, 0]
                else:
                    mask_idx.append(1)

            pad = max(0, sentence_size - len(sentence))
            if pad == 0:
                sent_idx = [word2idx[w] for w in sentence][:sentence_size]
                mask_idx = mask_idx[:sentence_size]
            else:
                sent_idx = [word2idx[w] for w in sentence] + [0] * pad
                mask_idx = mask_idx + [0] * pad

            review_idx.append(sent_idx)
            word_mask.append(mask_idx)

            mask_sum = np.sum(mask_idx)
            if mask_sum == 0:
                sentence_mask.append(0)
            else:
                sentence_mask.append(1)

        review_idx = review_idx[:memory_size]
        word_mask = word_mask[:memory_size]
        sentence_mask = sentence_mask[:memory_size]

        # pad to memory_size
        lm = max(0, memory_size - len(review_idx))

        for _ in range(lm):
            review_idx.append([0] * sentence_size)
            word_mask.append([0] * sentence_size)
            sentence_mask.append(0)

        reviews.append(review_idx)
        word_masks.append(word_mask)
        sentence_masks.append(sentence_mask)
        labels.append(label_idx)
        domains.append(domain_idx)
        u_labels.append(u_label)
        v_labels.append(v_label)

    return np.array(reviews), np.array(domains), np.array(labels), np.array(u_labels), np.array(v_labels), np.array(word_masks), np.array(sentence_masks)

def create_input(reviews, word_masks, sent_masks, labels, u_labels, v_labels, batch_size):

    review, word_mask, sent_mask, label, z1_label, z2_label = tf.train.slice_input_producer([reviews, word_masks, sent_masks, labels, u_labels, v_labels])

    return tf.train.batch([review, word_mask, sent_mask, label, z1_label, z2_label], batch_size=batch_size)

def visualization(reviews, y_label, y_pred, u_labels, v_labels, word_attentions1, sentence_attentions1, word_attentions2, sentence_attentions2, word_mask, sentence_mask, idx2word, domain, mode):

    fname = "./work/attentions/" + domain + "_" + mode +"_HATN.txt"
    print(fname)

    with open(fname, "w") as f:

        for i in xrange(len(reviews)):

            review = reviews[i]

            f.write("review#%d:  GT:%d  Prediction:%d\n" % (i+1, 1 - y_label[i], 1 - y_pred[i]))
            sen_idx  = np.argmax(np.array(sentence_attentions1[i]))
            word_idx = np.argmax(np.array(word_attentions1[i][sen_idx]))
            most_important_word = review[sen_idx][word_idx]
            if most_important_word != 0:
                word = idx2word[most_important_word]
                f.write("most important word:%s sen:%d word:%d\n" % (word, sen_idx+1, word_idx+1))

            for j in xrange(len(sentence_mask[i])):
                if j <19 and sentence_mask[i][j] == 0 and sentence_mask[i][j+1] == 0:
                    break
                if j == 19 and sentence_mask[i][j] == 0:
                    break

                sentence = review[j]
                f.write("sen#%d %f: " % (j+1, sentence_attentions1[i][j]))

                for k in xrange(len(word_mask[i][j])):

                    w_idx = sentence[k]

                    if w_idx != 0:

                        word = idx2word[w_idx]
                        f.write("%s %f " % (word, word_attentions1[i][j][k]))

                f.write("\n")
            f.write("\n")


            f.write("review#%d: u:%d  v:%d\n" % (i+1, 1 - u_labels[i], 1 - v_labels[i]))
            sen_idx  = np.argmax(np.array(sentence_attentions2[i]))
            word_idx = np.argmax(np.array(word_attentions2[i][sen_idx]))
            most_important_word = review[sen_idx][word_idx]

            most_important_sentence = [idx2word[idx] for idx in review[sen_idx] if idx != 0]

            if most_important_word != 0:
                word = idx2word[most_important_word]
                f.write("most important word:%s sen:%d word:%d\n" % (word, sen_idx+1, word_idx+1))

            for j in xrange(len(sentence_mask[i])):
                if j <19 and sentence_mask[i][j] == 0 and sentence_mask[i][j+1] == 0:
                    break
                if j == 19 and sentence_mask[i][j] == 0:
                    break

                sentence = review[j]
                f.write("sen#%d %f: " % (j+1, sentence_attentions2[i][j]))

                for k in xrange(len(word_mask[i][j])):

                    w_idx = sentence[k]

                    if w_idx != 0 and word_mask[i][j][k] != 0:
                        word = idx2word[w_idx]
                        f.write("%s %f " % (word, word_attentions2[i][j][k]))
                    if w_idx != 0 and word_mask[i][j][k] == 0:
                        word = idx2word[w_idx]
                        f.write("UNK(%s) %f " % (word, word_attentions2[i][j][k]))

                f.write("\n")
            f.write("\n")

        f.close()
