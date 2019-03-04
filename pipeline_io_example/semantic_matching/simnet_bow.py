"""
   docstring
"""
import os
import sys
import time
import numpy as np
import math
import paddle.fluid as fluid
base_lr = 0.2
emb_lr = base_lr * 3
dict_dim = 1451594
emb_dim = 256
hid_dim = 2048
margin = 0.1
batch_size = 128

q = fluid.layers.data(
    name="query", shape=[1], dtype="int64", lod_level=1)
pt = fluid.layers.data(
    name="pos_title", shape=[1], dtype="int64", lod_level=1)
nt = fluid.layers.data(
    name="neg_title", shape=[1], dtype="int64", lod_level=1)

## embedding
q_emb = fluid.layers.embedding(input=q,
                               size=[dict_dim, emb_dim],
                               param_attr=fluid.ParamAttr(name="__emb__", learning_rate=emb_lr),
                               is_sparse=True)
## vsum
q_sum = fluid.layers.sequence_pool(input=q_emb,
                                   pool_type='sum')
q_ss = fluid.layers.softsign(q_sum)
## fc layer after conv
q_fc = fluid.layers.fc(input=q_ss,
                       size=hid_dim,
                       param_attr=fluid.ParamAttr(name="__q_fc__", learning_rate=base_lr))

# label data
#label = fluid.layers.data(name="label", shape=[1], dtype="int64")

## embedding
pt_emb = fluid.layers.embedding(input=pt,
                                size=[dict_dim, emb_dim],
                                param_attr=fluid.ParamAttr(name="__emb__", learning_rate=emb_lr),
                                is_sparse=True)
## vsum
pt_sum = fluid.layers.sequence_pool(
    input=pt_emb,
    pool_type='sum')
pt_ss = fluid.layers.softsign(pt_sum)
## fc layer
pt_fc = fluid.layers.fc(input=pt_ss,
                        size=hid_dim,
                        param_attr=fluid.ParamAttr(name="__fc__", learning_rate=base_lr),
                        bias_attr=fluid.ParamAttr(name="__fc_b__"))

## embedding
nt_emb = fluid.layers.embedding(input=nt,
                                size=[dict_dim, emb_dim],
                                param_attr=fluid.ParamAttr(name="__emb__", learning_rate=emb_lr),
                                is_sparse=True)
## vsum
nt_sum = fluid.layers.sequence_pool(
    input=nt_emb,
    pool_type='sum')
nt_ss = fluid.layers.softsign(nt_sum)
## fc layer
nt_fc = fluid.layers.fc(input=nt_ss,
                        size=hid_dim,
                        param_attr=fluid.ParamAttr(name="__fc__", learning_rate=base_lr),
                        bias_attr=fluid.ParamAttr(name="__fc_b__"))
cos_q_pt = fluid.layers.cos_sim(q_fc, pt_fc)
cos_q_nt = fluid.layers.cos_sim(q_fc, nt_fc)
# hinge_loss
loss_op1 = fluid.layers.elementwise_sub( \
                                         fluid.layers.fill_constant_batch_size_like(input=cos_q_pt, \
                                                                                    shape=[-1, 1], value=margin, dtype='float32'), \
                                         cos_q_pt)
loss_op2 = fluid.layers.elementwise_add(loss_op1, cos_q_nt)
loss_op3 = fluid.layers.elementwise_max( \
                                         fluid.layers.fill_constant_batch_size_like(input=loss_op2, \
                                                                                    shape=[-1, 1], value=0.0, dtype='float32'), \
                                         loss_op2)
avg_cost = fluid.layers.mean(loss_op3)
'''
acc = fluid.layers.accuracy(input=cos_q_pt, \
                            label=label, k=1)
'''
#real_acc = get_acc(cos_q_nt, cos_q_pt)
# SGD optimizer
sgd_optimizer = fluid.optimizer.SGD(learning_rate=base_lr)
sgd_optimizer.minimize(avg_cost)

place = fluid.CPUPlace()
exe = fluid.Executor(place)                
exe.run(fluid.default_startup_program())
async_exe = fluid.AsyncExecutor(place)
thread_num = 10
dataset = fluid.DataFeedDesc('data_feed.proto')
dataset.set_batch_size(32)
dataset.set_use_slots([q.name, pt.name, nt.name])
dataset.set_pipe_command("/home/users/dongdaxiang/paddle_whls/new_io/paddle_release_home/python/bin/python pairwise_reader.py")
#dataset.set_pipe_command("cat")
filelist = ["ids/%s" % x for x in os.listdir("ids")]
#filelist = ["prepared.txt"]
print(filelist)
async_exe.run(fluid.default_main_program(), dataset, filelist, thread_num, [], debug=False)

                
