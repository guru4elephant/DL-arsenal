from __future__ import print_function

import logging
import numpy as np
from dataset_generator import MultiSlotDataset
import logging
import sys
import os

class PairwiseReader(MultiSlotDataset):
    def init_reader(self, max_len, sampling_rate):
        np.random.seed(1)
        self.max_len = max_len
        self.sampling_rate = sampling_rate
        self.query_buffer = None
        self.pos_title_buffer = None
        self.neg_title_buffer = None

    def generate_sample(self, line):
        def check_empty(str):
            if str == "":
                return int(750000)
            else:
                return int(str)
        def get_rand(low = 0.0, high = 1.0):
            return ((1.0 * np.random.randint(0, 0x7fff)) / 0x7fff) * (high - low) + low

        def trunc_by_len(vec, max_l=self.max_len):
            end = len(vec) if len(vec) <= max_l else max_l
            return vec[0:end]

        def pairwise_iterator():
            items = line.strip("\t\n").split(";")
            pos_num, neg_num = [int(i) for i in items[1].split(" ")]
            assert (len(items) > 3 + 2 * (pos_num + neg_num))
            query = [check_empty(i) for i in trunc_by_len(items[2].split(" "))]
            pos_title = [[check_empty(i) for i in trunc_by_len(items[3 + index * 2].split(" "))]
                         for index in range(pos_num)]
            pos_abs = [[check_empty(i) for i in trunc_by_len(items[3 + index * 2 + 1].split(" "))]
                       for index in range(pos_num)]
            neg_title = []
            for index in range(neg_num):
                neg_title.append([check_empty(i)
                                  for i in trunc_by_len(items[3 + pos_num * 2 + index * 2].split(" "))])
                neg_abs = [[check_empty(i) for i in trunc_by_len(items[3 + pos_num * 2 + index * 2 +
                                                                       1].split(" "))]
                           for index in range(neg_num)]
            assert(len(pos_title) == len(pos_abs) == pos_num
                   and len(neg_title) == len(neg_abs) == neg_num)
            for i in range(len(neg_title)):
                prob = get_rand()
                if prob < self.sampling_rate:
                    pos_index = np.random.randint(0, len(pos_title))
                    self.query_buffer = query
                    self.pos_title_buffer = pos_title[pos_index]
                    self.neg_title_buffer = neg_title[i]
                    yield ("query", query), ("pos_title", pos_title[pos_index]), ("neg_title", neg_title[i])
                else:
                    continue
            yield None
        return pairwise_iterator


pairwise_reader = PairwiseReader()
pairwise_reader.init_reader(10000, 1.0)
pairwise_reader.run_from_stdin(is_local=True)
