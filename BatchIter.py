import random
import math
import operator
from torch.nn.utils.rnn import pad_sequence
import torch


class BatchIter():
    def __init__(self, dataset, batch_size, batch_first=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.id = 0
        self.batch_first = batch_first

    def __len__(self):
        return math.ceil(len(self.dataset)/self.batch_size)

    def __getitem__(self, index):
        if self.id >= len(self.dataset):
            self.id = 0
            raise StopIteration()
        sens = self.dataset[self.id:self.id+self.batch_size]
        sens = pad_sequence(sens, padding_value=0,
                            batch_first=self.batch_first)
        self.id += self.batch_size
        return sens

    def totext(self, sen):
        return self.dataset.totext(sen)
