import os

import math

import datetime
import paddle
import paddle.fluid as fluid
import sys
import time
import numpy as np

sys.path.append('thirdparty')
import model_conf
import fluid_net
import argparse
import hdfs_utils

PROFILE_STEP = 30
PROFILE_ON = bool(int(os.getenv("PROFILE_ON", "0")))


def print_log(log_str):
    time_stamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print(str(time_stamp) + " " + log_str)


def parse_args():
    parser = argparse.ArgumentParser("Training for CTR model.")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="The path of training data.")
    parser.add_argument(
        "--test_data_dir",
        type=str,
        required=True,
        help="The path of testing data.")
    parser.add_argument(
        "--batch_size",
        type=str,
        required=True,
        help="Train batch size.")
    parser.add_argument(
        "--cpu_num",
        type=str,
        required=True,
        help="train cpu number.")
    parser.add_argument(
        "--use_parallel_exe",
        type=int,
        default=0,
        help="if use parallel_executor.")
    return parser.parse_args()


PASS_NUM = 1
EMBED_SIZE = 64
IS_SPARSE = False
CNN_DIM = 128
CNN_FILTER_SIZE = 5
is_distributed = True

DICT_SIZE = 10000 * 10
AGE_SIZE = 10
GENDER_SIZE = 4
MTID_SIZE = 4
CMATCH_SIZE = 4
WORD_SIZE = 25000


class DataConfig(object):
    def __init__(self, name, shape, dtype, lod_level=0):
        self.name = None
        self.shape = shape
        self.dtype = dtype
        self.lod_level = lod_level


def infer(is_local, fea_sections, id_dict, batch_size, model_dir, use_parallel_executor):
    print_log("start to infer @ {} ##### \n".format(time.time()))
    os.environ["IS_TRAIN"] = "0"

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    inference_scope = fluid.core.Scope()
    test_reader = paddle.batch(ctr_reader.cluster_data_reader(fea_sections),
                               batch_size=batch_size)

    with fluid.scope_guard(inference_scope):
        startup_program = fluid.framework.Program()
        test_program = fluid.framework.Program()
        with fluid.framework.program_guard(test_program, startup_program):
            with fluid.unique_name.guard():
                data_list, predict, auc_var, cur_auc_var, auc_states, avg_cost, label, reader = fluid_net.net(
                    fea_sections, batch_size)

        feeder = fluid.DataFeeder(feed_list=data_list, place=place)

        exe.run(startup_program)

        fluid.io.load_persistables(executor=exe, dirname=model_dir, main_program=test_program)
        with open("test.proto", "w") as f:
            f.write(str(test_program))

        def set_zero(var_name):
            param = inference_scope.var(var_name).get_tensor()
            param_array = np.zeros(param._get_dims()).astype("int64")
            param.set(param_array, place)

        for auc_state in auc_states:
            set_zero(auc_state.name)

        if use_parallel_executor:
            cpu_num_backup = os.environ['CPU_NUM']
            os.environ['CPU_NUM'] = "1"
            exec_strategy = fluid.ExecutionStrategy()
            exec_strategy.num_threads = int(os.getenv("CPU_NUM"))
            build_strategy = fluid.BuildStrategy()
            if int(os.getenv("CPU_NUM")) > 1:
                build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce
            pe = fluid.ParallelExecutor(
                use_cuda=False, main_program=test_program,
                build_strategy=build_strategy,
                exec_strategy=exec_strategy)
            os.environ['CPU_NUM'] = cpu_num_backup

        auc_val = None
        cur_auc_val = None
        test_start_time = time.time()

        batch_id = 0
        for data in test_reader():
            if use_parallel_executor:
                predict_val, auc_val, cur_auc_val = pe.run(
                    feed=feeder.feed(data),
                    fetch_list=[predict.name, auc_var.name, cur_auc_var.name])
            else:
                raise Exception("Only support ParallelExecutor")

            auc_val = np.mean(auc_val)
            cur_auc_val = np.mean(cur_auc_val)

            if batch_id % 100 == 0:
                print_log("test_auc:" + str(auc_val) + " test_auc_val:" + str(cur_auc_val))
            batch_id += 1

        test_end_time = time.time()

        pass_num = int(os.getenv("CUR_PASS_NUM"))
        hour = int(os.getenv("CUR_PASS_HOUR"))
        day = os.getenv("BEGIN_DAY")
        day = datetime.datetime.strptime(day, '%Y-%m-%d')
        day = day + datetime.timedelta(days=pass_num, hours=hour)

        print("final " + day.strftime('pass=%Y%m%d, hour=%H') + "test_auc:" + str(auc_val) + " test_auc_val:" + str(cur_auc_val)
                  + " test_time=" + str(test_end_time - test_start_time))
        print("### ### ## ####\n")

def train_async_local(batch_size):
    fea_sz, fea_sections, model_dict = model_conf.model_conf('thirdparty/model.conf')
    id_dict = None
    
    #data_list, predict, auc_var, cur_auc_var, auc_states, avg_cost, label = \
        #fluid_net.async_net(fea_sections)
    data_list, predict, avg_cost, label = fluid_net.async_net(fea_sections)
    
    optimizer = fluid.optimizer.Adam(learning_rate=0.0005, lazy_mode=True)
    #optimizer = fluid.optimizer.SGD(learning_rate=0.01)
    optimize_ops, params_grads = optimizer.minimize(avg_cost)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    async_exe = fluid.AsyncExecutor(place)

    def train_loop(main_program, trainer_id=None):
        dataset = fluid.DataFeedDesc('data_feed.proto')
        dataset.set_batch_size(32)
        dataset.set_use_slots([d.name for d in data_list])
        dataset.set_pipe_command('/home/users/dongdaxiang/paddle_whls/new_io/paddle_release_home/python/bin/python ctr_reader.py')
        # how to define the protocol
        thread_num = 10
        for pass_id in xrange(PASS_NUM):
            for hour in range(24):
                hour_filelist = ["./test_data_dir/%s" % x for x in os.listdir("./test_data_dir/") if "part" in x]
                print(hour_filelist)
                async_exe.run(main_program,
                              dataset,
                              hour_filelist,
                              thread_num,
                              [avg_cost],
                              debug=True)
                
    train_loop(fluid.default_main_program())
