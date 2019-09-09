
# -*- coding: utf8 -*-

import chardet
import re
import os
from parse import Parser
import numpy as np
import string
import nltk
import pickle
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import RegexpTokenizer

def parseRawData(domain):

    # Generate the working directory
    work_dir     = os.path.abspath(os.path.join(os.path.curdir,"work2"))
    word_dir     = os.path.abspath(os.path.join(work_dir,'word'))
    sentence_dir = os.path.abspath(os.path.join(work_dir,'sentence'))
    domain_dir1 = os.path.abspath(os.path.join(word_dir,domain))
    domain_dir2 = os.path.abspath(os.path.join(sentence_dir,domain))

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    if not os.path.exists(word_dir):
        os.makedirs(word_dir)
    if not os.path.exists(sentence_dir):
        os.makedirs(sentence_dir)
    if not os.path.exists(domain_dir1):
        os.makedirs(domain_dir1)
    if not os.path.exists(domain_dir2):
        os.makedirs(domain_dir2)

    fname = "./work/%s/review_%s" % (domain, "positive")
    h_pos_data, pos_data = get_review(fname)

    fname = "./work/%s/review_%s" % (domain, "negative")
    h_neg_data, neg_data = get_review(fname)

    pos_num, neg_num = len(h_pos_data), len(h_neg_data)

    np.random.seed(7)
    shuffle_pos_idx = np.random.permutation(np.arange(pos_num))
    h_pos_shuffle = h_pos_data[shuffle_pos_idx]
    h_pos_train   = h_pos_shuffle[:2800]
    h_pos_test    = h_pos_shuffle[2800:]
    write_h_tokensToFile(h_pos_train, domain, "train", "positive")
    write_h_tokensToFile(h_pos_test,  domain, "test",  "positive")

    pos_shuffle   = pos_data[shuffle_pos_idx]
    pos_train     = pos_shuffle[:2800]
    pos_test      = pos_shuffle[2800:]
    write_tokensToFile(pos_train, domain, "train", "positive")
    write_tokensToFile(pos_test,  domain, "test",  "positive")

    shuffle_neg_idx = np.random.permutation(np.arange(neg_num))
    h_neg_shuffle = h_neg_data[shuffle_neg_idx]
    h_neg_train   = h_neg_shuffle[:2800]
    h_neg_test    = h_neg_shuffle[2800:]
    write_h_tokensToFile(h_neg_train, domain, "train", "negative")
    write_h_tokensToFile(h_neg_test,  domain, "test",  "negative")

    neg_shuffle   = neg_data[shuffle_neg_idx]
    neg_train     = neg_shuffle[:2800]
    neg_test      = neg_shuffle[2800:]
    write_tokensToFile(neg_train, domain, "train", "negative")
    write_tokensToFile(neg_test,  domain, "test",  "negative")

    fname = "./work/%s/review_%s" % (domain, "unlabeled")
    h_unlab_data, unlab_data = get_review(fname)
    write_h_tokensToFile(h_unlab_data, domain, "train", "unlabeled")
    write_tokensToFile(unlab_data,     domain, "train", "unlabeled")

def get_review(fname):

    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
    with open("./data/stopwords") as F:
        stopWords = set(map(string.strip, F.readlines()))

    h_tokens_list  = []
    tokens_list    = []
    with open(fname) as f:

        lines = f.readlines()
        for line in lines:

            review = line.strip().lower()
            sentences = sent_tokenizer.tokenize(review)

            h_tokens = []
            tokens   = []
            for sentence in sentences:

                table = string.maketrans("", "")
                delEStr = string.punctuation + string.digits
                words = tokenizer.tokenize(str(sentence))

                symbols = list(string.punctuation + string.digits)
                symbols.remove('!')
                elements = words
                words = []
                for word in elements:
                    if word not in symbols:
                        if word != '!':
                            word = word.translate(table, delEStr)
                        if len(word) != 0:
                            words.append(word)

                if len(words) > 0:
                    if len(words) == 1 and (words[0] == '!' or words[0] in stopWords):
                        pass
                    else:
                        h_tokens.append(words)
                        tokens.extend(words)

            h_tokens_list.append(h_tokens)
            tokens_list.append(tokens)

    return np.array(h_tokens_list), np.array(tokens_list)

def write_tokensToFile(tokens_list, domain, mode, label):

    fname = "./work2/word/%s/tokens_%s.%s" % (domain, mode, label)
    print(fname, len(tokens_list))
    F = open(fname, 'w')
    for tokens in tokens_list:
        for token in tokens:
            F.write("%s " % token)
        F.write("\n")
    F.close()
    pass

def write_h_tokensToFile(h_tokens_list, domain, mode, label):

    fname = "./work2/sentence/%s/tokens_%s.%s" % (domain, mode, label)
    print(fname, len(h_tokens_list))
    with open(fname, 'wb') as F:
        pickle.dump(h_tokens_list, F)
    F.close()
    pass


if __name__ == "__main__":

    domains = ["books", "dvd", "electronics", "kitchen", "video"]
    for idx, domain in enumerate(domains):
        parseRawData(domain)
