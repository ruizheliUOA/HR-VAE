import re
import random
import numpy as np
# from PIL import Image
from torch.utils.data import Dataset
import operator
from itertools import groupby
from functools import reduce
import torch
import pickle
import csv
import os
# base_path = "./data"
# if os.getcwd() == "/content":
#     base_path = "/content/drive/My Drive/Test/"


class Corpus(Dataset):
    def __init__(self, path, word_dic=None, min_word_count=4):
        print("start to load Corpus data")
        with open(path, 'r') as f:
            corpus = f.readlines()
        corpus = [self._split(i.strip()) for i in corpus]
        # 长短排序
        corpus = [(t, len(t)) for t in corpus]
        corpus.sort(key=operator.itemgetter(1))
        self.texts = [x for x, _ in corpus][:2000000]
        print("start to build dictionary")
        if word_dic is not None:
            self.word_id = word_dic
        else:
            doc = " ".join([" ".join(i) for i in self.texts])
            self.word_id = self._make_dic(doc, min_word_count)
        self.id_word = self._make_inv_dic(self.word_id)
        self.voca_size = len(self.word_id)
        self.sos_token = torch.tensor([1])
        self.eos_token = torch.tensor([2])
        print("start to make one-hot vectors")
        self.textcodes = [self._txt_vecs(l) for l in self.texts]
        self.doc_size = len(self.texts)
        self.max_length = max((len(i) for i in self.texts))

    def _split(self, sen):
        sen = sen.lower()
        sen = re.sub(r"[.]+", ".", sen)
        sen = re.sub(r"([.?!,]|'s)", r" \1", sen)
        return re.split(r"\s+", sen)

    def _split2(self, labels):
        labels = re.sub(r"[ ]", "-", labels)
        return re.split(r",-*", labels)

    def _make_dic(self, doc, min_word_count, addflag=True):
        flag_count = 4 if addflag else 0
        doc_ = re.split(r"\s", "".join(doc))
        words = sorted(doc_)
        word_count = [(w, sum(1 for _ in c)) for w, c in groupby(words)]
        word_count = [(w, c) for w, c in word_count if c >= min_word_count]
        word_count.sort(key=operator.itemgetter(1), reverse=True)
        word_id = dict([(w, i+flag_count)
                        for i, (w, _) in enumerate(word_count)])
        if addflag:
            word_id["<pad>"] = 0
            word_id["<unk0>"] = 1
            word_id["<sos>"] = 2
            word_id["<eos>"] = 3
            self.pad_id = 0
        return word_id

    def _make_inv_dic(self, word_id_dic):
        id_word = dict([(i, w) for w, i in word_id_dic.items()])
        return id_word

    def _word_onehot(self, word):
        v = torch.zeros([self.voca_size], dtype=torch.long)
        v[self.word_id.get(word, 1)] = 1
        return v

    def _txt_vecs(self, txt):
        v = [self.word_id.get(w, 1) for w in txt]
        v = [2]+v+[3]
        v = torch.tensor(v)
        return v

    def __getitem__(self, index):
        return self.textcodes[index]

    def __len__(self):
        return self.doc_size

    def totext(self, sen):
        text = [self.id_word[i] for i in sen]
        return " ".join(text)





