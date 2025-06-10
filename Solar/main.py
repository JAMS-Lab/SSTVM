# -*- coding: utf-8 -*-



import os
import numpy as np
import scipy.sparse as sp
import tensorflow.compat.v1 as tf
import pandas as pd
from model import DVGAE

from graph import get_adjacent_matrix

import argparse
import configparser

# prepare dataset
parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configurations/solar_energy.conf', type=str,
                    help="configuration file path")

args = parser.parse_args()
config1 = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config))
config1.read(args.config)
data_config = config1['Data']
dataset = data_config['dataset']

def main():
    tf.disable_v2_behavior()
    print('gpu is running or not? ' + str(tf.test.is_gpu_available()))

    # if dataset == 'PEMS08':
    #     data_seq = np.load(data_config['graph_signals'])
    #     features = data_seq['data']
    # else:
    #     features = pd.read_csv(data_config['graph_signals'])

    features = np.load(data_config['graph_signals'], allow_pickle=True)
    features = np.array(features['data'])
    n_nodes = features.shape[1]
    features_num = data_config['features_num']
    features_num = int(features_num)
    print("nodes number {0}".format(n_nodes))
    print("features number {0}".format(features_num))

    adj_predefine = get_adjacent_matrix(data_config['adj'], num_nodes=n_nodes)

    adj_predefine = np.array(adj_predefine > 0.5, dtype=float)

    coo_adjacency = sp.coo_matrix(adj_predefine)

    features = features.swapaxes(0, 1)
    max_value = 1.0
    features = features / max_value


    # ======= 这里开始加噪声 =======
    noise_level = 0.01  # 噪声强度，可调节
    noise = noise_level * np.random.normal(loc=0.0, scale=1.0, size=features.shape)
    features = features + noise
    # 如果你确定数据是非负的，可以用下面这行保证非负
    features = np.clip(features, 0, None)
    print(f"Added Gaussian noise with std dev: {noise_level}")
    # ======= 加噪声结束 =======

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    config = tf.ConfigProto(allow_soft_placement=True)
    # gpu_options = tf.GPUOptions()
    config.gpu_options.per_process_gpu_memory_fraction=0.7
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as tf_sess:
        model = DVGAE(tf_sess, n_nodes, adj_predefine, features_num, config1)
        model.Train(coo_adjacency, features, max_value)


if __name__ == '__main__':
    main()
