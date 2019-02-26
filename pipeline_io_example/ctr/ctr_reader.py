from __future__ import print_function
import gzip
import os
import sys
import logging
from data_generator import *
import model_conf
import xxhash

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)

class TiebaGenerator(MultiSlotDataGenerator):
    def load_resource(self, fea_sections):
        self.slot_name = ['slot%d' % i for i in range(31)] + ["label"]
        self.fea_sections = fea_sections

    def process(self, line):
        def data_iterator():
            fields = line.decode('gbk').encode('utf-8').strip('\r\n').split(' ')
            if len(fields) != 32:
                yield None
            label_ctr = int(fields[0])
            slots = []
            feature_fields = fields[1:]
            for i in range(0, len(fea_sections)):
                slot = []
                describe = fea_sections[i]['fea_des']
                fea_type = fea_sections[i]['fea_type']
                size = int(fea_sections[i]['max_sz'])
                value_list = feature_fields[i].split(',')
                if fea_type in ['sparse']:
                    for value in value_list:
                        # why do hashing here? hashing should incorporate slotid
                        slot.append(xxhash.xxh64_intdigest(value) % size)
                    if len(slot) == 0:
                        slot.append(0)
                    slots.append(slot)
            slots.append([label_ctr])
            yield zip(self.slot_name, slots)
        return data_iterator

fea_sz, fea_sections, model_dict = model_conf.model_conf('thirdparty/model.conf')
tieba = TiebaGenerator()
tieba.load_resource(fea_sections)
tieba.run_from_stdin(is_local=True)
