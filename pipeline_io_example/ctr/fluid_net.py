import paddle.fluid as fluid
import math

CNN_DIM = 128
CNN_FILTER_SIZE = 5

def async_net(INPUT):
    user_emb_list = []
    video_emb_list = []
    data_list = []
    for i, inp in enumerate(INPUT):
        if inp['fea_type'] in ['sparse']:
            data = fluid.layers.data(name='slot%d' % i,
                                     shape=[1], lod_level=1,
                                     dtype='int64')
            data_list.append(data)
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    data_list.append(label)
    idx = 0
    for inp in INPUT:
        if inp['fea_type'] in ['sparse']:
            embed = fluid.layers.embedding(
                input=data_list[idx],
                size=[inp['max_sz'], inp['out_sz']],
                dtype='float32',
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Normal(scale=1 / math.sqrt(inp['max_sz']))),
                is_sparse=True)
            cnn = fluid.layers.sequence_pool(input=embed, pool_type='sum')
            if inp['fea_des'].startswith('user'):
                user_emb_list.append(cnn)
            elif inp['fea_des'].startswith('thread'):
                video_emb_list.append(cnn)
            idx += 1
    user_emb_concat = fluid.layers.concat(user_emb_list, axis=1)
    u_fc0 = fluid.layers.fc(name='u_fc0', input=user_emb_concat, size=512, act='relu',
                            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                                scale=1 / math.sqrt(user_emb_concat.shape[1]))))
    
    u_fc1 = fluid.layers.fc(name='u_fc1', input=u_fc0, size=256, act='relu',
                            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                            scale=1 / math.sqrt(u_fc0.shape[1]))))
    
    u_fc2 = fluid.layers.fc(name='u_fc2', input=u_fc1, size=128, act='relu',
                            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                                scale=1 / math.sqrt(u_fc1.shape[1]))))
    
    u_fc3 = fluid.layers.fc(name='u_fc3', input=u_fc2, size=128, act='relu',
                            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                                scale=1 / math.sqrt(u_fc2.shape[1]))))
    
    u_fc4 = fluid.layers.fc(name='u_fc4', input=u_fc3, size=128, act='relu',
                            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                                scale=1 / math.sqrt(u_fc3.shape[1]))))
    
    u_fc5 = fluid.layers.fc(name='u_fc5', input=u_fc4, size=32, act=None,
                            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                                scale=1 / math.sqrt(u_fc4.shape[1]))))

    video_emb_concat = fluid.layers.concat(video_emb_list, axis=1)
    
    t_fc0 = fluid.layers.fc(name='t_fc0', input=video_emb_concat, size=512, act='relu',
                            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                                scale=1 / math.sqrt(video_emb_concat.shape[1]))))
    
    t_fc1 = fluid.layers.fc(name='t_fc1', input=t_fc0, size=256, act='relu',
                        param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                            scale=1 / math.sqrt(video_emb_concat.shape[1]))))
    
    tc_fc2 = fluid.layers.fc(name='tc_fc2', input=t_fc1, size=128, act='relu',
                             param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                                 scale=1 / math.sqrt(t_fc1.shape[1]))))

    tc_fc3 = fluid.layers.fc(name='tc_fc3', input=tc_fc2, size=128, act='relu',
                             param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                                 scale=1 / math.sqrt(tc_fc2.shape[1]))))
    
    tc_fc4 = fluid.layers.fc(name='tc_fc4', input=tc_fc3, size=128, act='relu',
                             param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                                 scale=1 / math.sqrt(tc_fc3.shape[1]))))
    
    tc_fc5 = fluid.layers.fc(name='tc_fc5', input=tc_fc4, size=32, act=None,
                             param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                                 scale=1 / math.sqrt(tc_fc4.shape[1]))))

    sim = fluid.layers.elementwise_mul(u_fc5, tc_fc5)
    scale = fluid.layers.reduce_sum(sim, dim=1, keep_dim=True)
    predict = fluid.layers.fc(input=scale, size=2, act="softmax")
    #auc_var, cur_auc_var, auc_states = fluid.layers.auc(input=predict, label=data_list[-1], slide_steps=20)
    label = fluid.layers.cast(x=data_list[-1], dtype='float32')
    cost = fluid.layers.sigmoid_cross_entropy_with_logits(x=scale, label=label)
    avg_cost = fluid.layers.reduce_mean(cost)
    
    #return data_list, predict, auc_var, cur_auc_var, auc_states, avg_cost, data_list[-1]
    return data_list, predict, avg_cost, data_list[-1]

