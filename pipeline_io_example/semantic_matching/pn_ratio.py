import sys
import os

pos_num = 0
neg_num = 0
for line in sys.stdin:
    line = line.strip()
    if "pos_num" in line:
        pos_num += int(line.split("\t")[-1])
    if "neg_num" in line:
        neg_num += int(line.split("\t")[-1])
print(float(pos_num) / float(neg_num))
