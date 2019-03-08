# -*- coding: utf-8 -*

import numpy as np
from data_generator import MultiSlotDataGenerator
import logging
import sys

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)

def native_to_unicode(s):
    if _is_unicode(s):
        return s
    try:
        return _to_unicode(s)
    except UnicodeDecodeError:
        res = _to_unicode(s, ignore_errors=True)
        tf.logging.info("Ignoring Unicode error, outputting: %s" % res)
        return res

def load_code_format_file(filename, word_dict, code_dict):
    with open(filename, "r") as f:
        for line in f:
            line = line.decode(encoding='UTF-8')
            items = line.split("\t")
            word = items[0]
            code_items = items[1].split()
            code = [int(item) for item in code_items]
            code_dict[word_dict[word]] = code

def load_kv_format_file(filename, kv_dict):
    word_id = 0
    with open(filename, "r") as f:
        for line in f:
            line = line.decode(encoding='UTF-8')
            word, count = line.split()[0], int(line.split()[1])
            kv_dict[word] = word_id
            word_id += 1
            

class Word2VecGenerator(MultiSlotDataGenerator):
    def load_resource(self, dict_path, pcode_path=None,
                      ptable_path=None, window_size=5):
        self.window_size_ = window_size
        word_id = 0
        self.word_to_id_ = dict()
        self.id_to_code = dict()
        self.id_to_path = dict()
        self.with_hs = False
        load_kv_format_file(dict_path, self.word_to_id_)
        if pcode_path:
            load_code_format_file(
                pcode_path,
                self.word_to_id_,
                self.id_to_code)
            self.with_hs = True
        if ptable_path:
            load_code_format_file(
                ptable_path,
                self.word_to_id_,
                self.id_to_path)

    def get_context_words(self, words, idx, window_size):
        target_window = np.random.randint(1, window_size + 1)
        start_point = idx - target_window if (idx - target_window) > 0 else 0
        end_point = idx + target_window
        targets = set(words[start_point:idx] + words[idx + 1:end_point + 1])
        return list(targets)

    def generate_sample(self, line):
        word_ids = [
            self.word_to_id_[word] for word in line.split()
            if word in self.word_to_id_
        ]
        def local_iterator():
            for idx, target_id in enumerate(word_ids):
                context_word_ids = self.get_context_words(
                    word_ids, idx, self.window_size_)
                if context_word_ids == []:
                    yield None
                if self.with_hs:
                    for context_id in context_word_ids:
                        yield ("target", [target_id]), ("context", [context_id]), \
                        ("pcode", self.id_to_code[target_id]), \
                        ("ptable", self.id_to_path[target_id])
                else:
                    for context_id in context_word_ids:
                        yield ("target", [target_id]), ("context", [context_id])
            yield None
            
        return local_iterator


if __name__ == "__main__":
    window_size = 10

    word2vec = Word2VecGenerator()
    word2vec.load_resource("data/1-billion_dict",
                           pcode_path="data/1-billion_dict_pcode",
                           ptable_path="data/1-billion_dict_ptable",
                           window_size=window_size)
    '''
    filelist = ["data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/news.en-%s-of-00100" \
                % str(i).zfill(5) for i in range(1, 20, 1)]
    '''
    '''
    line_limit = 1000000
    process_num = 20
    word2vec.run_from_files(filelist=filelist,
                            line_limit=line_limit,
                            process_num=process_num, 
                            output_dir="./data_output_with_pcode")
    '''
    word2vec.run_from_stdin(is_local=True)
    '''
    hadoop_ugi="nlp,hello123", 
    hadoop_host="hdfs://yq01-heng-hdfs.dmop.baidu.com:54310",
    proto_path="/app/ssg/nlp/sc/yebaiwei/async_executor_data_generator_output")
    '''

