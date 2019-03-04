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
from dataset_generator import MultiSlotDataset


class IMDBDataset(MultiSlotDataset):
    def load_resource(self, dictfile):
        self._vocab = {}
        wid = 0
        with open(dictfile) as f:
            for line in f:
                self._vocab[line.strip()] = wid
                wid += 1
        self._unk_id = len(self._vocab)
        self._pattern = re.compile(r'(;|,|\.|\?|!|\s|\(|\))')
    
    def generate_sample(self, line):
        def data_iter():
            send = '|'.join(line.split('|')[:-1]).lower().replace("<br />",
                                                                  " ").strip()
            label = [int(line.split('|')[-1])]
            
            words = [x for x in self._pattern.split(send) if x and x != " "]
            feas = [
                self._vocab[x] if x in self._vocab else self._unk_id for x in words
            ]
            
            yield ("words", feas), ("label", label)
        return data_iter


imdb = IMDBDataset()
imdb.load_resource("imdb.vocab")
imdb.run_from_stdin(is_local=True)
