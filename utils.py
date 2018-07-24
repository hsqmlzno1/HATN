
import numpy as np
from numpy import *
import tensorflow as tf
import os
import re
import pickle
import nltk

def load_data(source_domain, target_domain, root_path):

    train_data = []
    test_data = []
    val_data = []
    source_unlabeled_data = []
    target_unlabeled_data = []
    src, tar = 1, 0

    print "source domain: ", source_domain, "target domain:", target_domain

    # load training data
    for (mode, label) in [("train", "positive"), ("train", "negative")]:
        fname = root_path+"%s/tokens_%s.%s" % (source_domain, mode, label)
        train_data.extend(get_review(fname, src, label))
    print "train-size: ", len(train_data)

    # load validation data
    for (mode, label) in [("test", "positive"), ("test", "negative")]:
        fname = root_path+"/%s/tokens_%s.%s" % (source_domain, mode, label)
        val_data.extend(get_review(fname, src, label))
    print "val-size: ", len(val_data)

    # load testing data
    for (mode, label) in [("train", "positive"), ("train", "negative"), ("test", "positive"), ("test", "negative")]:
        fname = root_path+"%s/tokens_%s.%s" % (target_domain, mode, label)
        test_data.extend(get_review(fname, tar, label))
    print "test-size: ", len(test_data)

    # load unlabeled data
    for (mode, label) in [("train", "unlabeled")]:
        fname = root_path+"%s/tokens_%s.%s" % (source_domain, mode, label)
        source_unlabeled_data.extend(get_review(fname, src, label))
        fname = root_path+"%s/tokens_%s.%s" % (target_domain, mode, label)
        target_unlabeled_data.extend(get_review(fname, tar, label))
    print "unlabeled-size: ", len(source_unlabeled_data), len(target_unlabeled_data)

    vocab = getVocab(train_data + val_data + test_data + source_unlabeled_data + target_unlabeled_data)
    print "vocab-size: ", len(vocab)

    output_dir = "./work/logs/"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    return train_data, val_data, test_data, source_unlabeled_data, target_unlabeled_data, vocab

def get_review(f, domain, label):

    reviews = []

    y = 1  # sentiment label
    if label == "positive":
        y = 1
    elif label == "negative":
        y = 0

    with open(f, 'rb') as F:
        token_list = pickle.load(F)
        for tokens in token_list:
            # print tokens,"\n"
            reviews.append((tokens, domain, y))

    return reviews

def getVocab(data):
    """
    Get the frequency of each feature in the file named fname.
    """
    vocab = {}

    for review, _, _, in data:
        for sentence in review:
            for word in sentence:
                vocab[word] = vocab.get(word, 0) + 1

    return vocab

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, dim=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, dim)


def get_w2vec(vocab, FLAGS):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    word_vecs = load_bin_vec(FLAGS.w2v_path, vocab)
    add_unknown_words(word_vecs, vocab)

    dim = word_vecs.values()[0].shape[0]
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    idx_word_map = dict()

    W = np.zeros(shape=(vocab_size + 1, dim), dtype='float32')
    W[0] = np.zeros(dim, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        idx_word_map[i] = word
        i += 1

    return W, word_idx_map, idx_word_map

def store_results(fname, source_domain, target_domain, test_acc):

    output_dir = "./work/results/"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    f = open('./work/results/%s_results.txt' % fname, "a")
    f.write('{:<20} {:.4f} \n'.format(source_domain+'-'+target_domain, test_acc))
    f.close()
