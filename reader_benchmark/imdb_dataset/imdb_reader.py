#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
import paddle
import re
import numpy as np
import paddle.fluid.incubate.data_generator as dg

class IMDBDataset(dg.MultiSlotDataGenerator):
    def init_hashing(self):
        self._pattern = re.compile(r'(;|,|\.|\?|!|\s|\(|\))')
        self.vocab_size = 89527

    def load_resource(self, dictfile):
        self._vocab = {}
        wid = 0
        with open(dictfile) as f:
            for line in f:
                self._vocab[line.strip()] = wid
                wid += 1
        self._unk_id = len(self._vocab)
        self._pattern = re.compile(r'(;|,|\.|\?|!|\s|\(|\))')
        self.return_value = ("words", [1, 2, 3, 4, 5, 6]), ("label", [0])


    def get_words_and_label_by_hashing(self, line):
        send = '|'.join(line.split('|')[:-1]).lower().replace("<br />",                                                                                             " ").strip()
        label = [int(line.split('|')[-1])]
        words = [x for x in self._pattern.split(send) if x and x != " "]
        feas = [
            hash(x) % self.vocab_size for x in words
        ]
        return feas, label

    def get_words_and_label(self, line):
        send = '|'.join(line.split('|')[:-1]).lower().replace("<br />",
                                                              " ").strip()
        label = [int(line.split('|')[-1])]
        
        words = [x for x in self._pattern.split(send) if x and x != " "]
        feas = [
            self._vocab[x] if x in self._vocab else self._unk_id for x in words
        ]
        return feas, label

    def infer_reader(self, infer_filelist, batch, buf_size):
        def local_iter():
            linenum = 0
            for fname in infer_filelist:
                with open(fname, "r") as fin:
                    for line in fin:
                        linenum += 1
                        feas, label = self.get_words_and_label_by_hashing(line)
                        yield np.array(feas), np.array(label)
        batch_iter = paddle.batch(
            paddle.reader.shuffle(local_iter, buf_size=buf_size),
            batch_size=batch)
        return batch_iter

    def generate_sample(self, line):
        def memory_iter():
            for i in range(1000):
                yield self.return_value
        def data_iter():
            feas, label = self.get_words_and_label_by_hashing(line)
            yield ("words", feas), ("label", label)
        return data_iter

if __name__ == "__main__":
    imdb = IMDBDataset()
    #imdb.load_resource("imdb.vocab")
    imdb.init_hashing()
    imdb.run_from_stdin()

