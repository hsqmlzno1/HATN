
import numpy as np
from numpy import *
import tensorflow as tf
import os
import re
import pickle
import nltk
from utils import *

def vectorize_data(data, word2idx, memory_size, sentence_size):

    S = []
    Q = []
    domains = []
    word_len = []
    sentence_len = []

    for review, domain, label in data:

        ss = []
        mask = []
        for i, sentence in enumerate(review, 1):
            ls = max(0, sentence_size - len(sentence))
            if ls == 0:
                sent_idx = [word2idx[w] for w in sentence][:sentence_size]
            else:
                sent_idx = [word2idx[w] for w in sentence] + [0] * ls
            ss.append(sent_idx)
            mask.append(sentence_size - ls)

        ss = ss[:memory_size]
        mask = mask[:memory_size]

        # pad to memory_size
        lm = max(0, memory_size - len(ss))

        for _ in range(lm):
            ss.append([0] * sentence_size)
            mask.append(0)

        q = [0] * 2
        q[1 - label] = 1
        domain_idx = [0] * 2
        domain_idx[1 - domain] = 1

        S.append(ss)
        Q.append(q)
        domains.append(domain_idx)
        word_len.append(mask)
        sentence_len.append(memory_size-lm)

    return np.array(S), np.array(domains), np.array(Q), np.array(word_len), np.array(sentence_len)

def create_input(reviews, labels, batch_size):

    review, label = tf.train.slice_input_producer([reviews, labels])

    return tf.train.batch([review, label], batch_size=batch_size)

def visualization(reviews, y_label, y_pred, word_attentions, sentence_attentions, word_mask, sentence_mask, idx2word, domain, mode):

    pos_list = ['JJ','JJS','JJR','RB','RBS','RBR','VB','VBZ','VBD','VBN','VBG','VBP']

    fname = "./work/attentions/" + domain + "_" + mode +".txt"
    print(fname)

    pos_pivots = {}
    neg_pivots = {}

    with open(fname, "w") as f:

        for i in xrange(len(reviews)):

            review = reviews[i]
            f.write("review#%d:  GT:%d  Prediction:%d\n" % (i+1, 1 - y_label[i], 1 - y_pred[i]))

            sen_idx  = np.argmax(np.array(sentence_attentions[i]))
            word_idx = np.argmax(np.array(word_attentions[i][sen_idx]))
            most_important_word = review[sen_idx][word_idx]

            most_important_sentence = [idx2word[idx] for idx in review[sen_idx] if idx != 0]
            pos_sentence = nltk.pos_tag(most_important_sentence)
            word_pos = pos_sentence[word_idx][1]

            if most_important_word != 0 and word_pos in pos_list:

                word = idx2word[most_important_word]
                f.write("most important word:%s\n" % word)
                if (1 - y_label[i] == 1) and (y_label[i] == y_pred[i]):
                    pos_pivots[word] = pos_pivots.get(word, 0) + 1
                if (1 - y_label[i] == 0) and (y_label[i] == y_pred[i]):
                    neg_pivots[word] = neg_pivots.get(word, 0) + 1

            for j in xrange(sentence_mask[i]):
                sentence = review[j]
                f.write("sen#%d %f: " % (j+1, sentence_attentions[i][j]))

                for k in xrange(word_mask[i][j]):
                    word = idx2word[sentence[k]]
                    f.write("%s %f " % (word, word_attentions[i][j][k]))

                f.write("\n")

            f.write("\n")

        f.close()

    pos_pivots_words = set(pos_pivots.keys())
    neg_pivots_words = set(neg_pivots.keys())
    outliers = pos_pivots_words & neg_pivots_words
    for outlier in outliers:
        if pos_pivots[outlier] > neg_pivots[outlier]:
            del neg_pivots[outlier]
        elif pos_pivots[outlier] < neg_pivots[outlier]:
            del pos_pivots[outlier]
        else:
            del pos_pivots[outlier]
            del neg_pivots[outlier]

    from nltk.corpus import stopwords
    stopWords = stopwords.words('english')
    stopWord_short = ["'m", "'s", "'re", "'ve", "e", "d"]
    adverse_list = ['not', 'no', 'without', 'never', 'n\'t', 'don\'t', 'hardly']

    for pivot, fre in pos_pivots.items():
        if fre < 5 or pivot in stopWords or pivot in adverse_list or pivot in stopWord_short:
            del pos_pivots[pivot]

    for pivot, fre in neg_pivots.items():
        if fre < 5 or pivot in stopWords or pivot in adverse_list or pivot in stopWord_short:
            del neg_pivots[pivot]

    pos_pivots = sorted(pos_pivots.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
    neg_pivots = sorted(neg_pivots.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)

    return pos_pivots, neg_pivots

def store_pivots(pos_pivots, neg_pivots, domain):

    fname = "./work/pivots/" + domain +"_pos.txt"
    print(fname)

    with open(fname, "w") as f:
        for (key, val) in pos_pivots:
            f.write("%s %d\n" % (key, val))
    f.close()

    fname = "./work/pivots/" + domain +"_neg.txt"
    print(fname)

    with open(fname, "w") as f:
        for (key, val) in neg_pivots:
            f.write("%s %d\n" % (key, val))
    f.close()