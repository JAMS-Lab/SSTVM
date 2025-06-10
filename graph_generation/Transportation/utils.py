# -*- coding: utf-8 -*-


import tensorflow.compat.v1 as tf
import numpy as np




tf.disable_v2_behavior()

def UniformRandomWeights(shape, name = None):
    randoms = tf.random_uniform(shape, minval = -np.sqrt(6.0 / (shape[0] + shape[1])), maxval = np.sqrt(6.0 / (shape[0] + shape[1])), dtype=tf.float32)
    return tf.Variable(randoms, name = name)#返回Tensorflow变量




def GaussianSampleWithNP(mean1, diag_cov1 ,mean2, diag_cov2 ):
    e1 = mean1 + np.random.normal(size = diag_cov1.shape)
    e2 = mean2 + np.random.normal(size = diag_cov2.shape)
    return e1,e2


def FirstGCNLayerWithActiveFun_NoX(norm_adj_mat, W, b,x):
    return tf.nn.relu(tf.add(tf.sparse_tensor_dense_matmul(norm_adj_mat, tf.matmul(x,W)), b))


def SecondGCNLayerWithoutActiveFun(norm_adj_mat, h, W, b):
    return  tf.nn.sigmoid(tf.add(tf.matmul(tf.sparse_tensor_dense_matmul(norm_adj_mat, h), W), b))


def FirstTCNLayerWithActiveFun_NoX(norm_adj_mat, W, b, x):
    # TCN 首层的实现，使用 ReLU 作为激活函数
    # 接收规范化邻接矩阵 `norm_adj_mat`、权重矩阵 `W`、偏置向量 `b` 和输入特征向量 `x`

    # 矩阵乘法：x * W
    xW = tf.matmul(x, W)

    # 稀疏矩阵与稠密矩阵相乘：norm_adj_mat * (x * W)
    adj_xW = tf.sparse_tensor_dense_matmul(norm_adj_mat, xW)

    # 加偏置：norm_adj_mat * (x * W) + b
    adj_xW_b = tf.add(adj_xW, b)

    # 应用激活函数：ReLU(norm_adj_mat * (x * W) + b)
    return tf.nn.relu(adj_xW_b)


def SecondTCNLayerWithoutActiveFun(norm_adj_mat, h, W, b):

    adj_h = tf.sparse_tensor_dense_matmul(norm_adj_mat, h)

    # 矩阵乘法：(norm_adj_mat * h) * W
    adj_h_W = tf.matmul(adj_h, W)

    # 加偏置：(norm_adj_mat * h) * W + b
    adj_h_W_b = tf.add(adj_h_W, b)

    # 应用激活函数：sigmoid((norm_adj_mat * h) * W + b)
    return tf.nn.sigmoid(adj_h_W_b)


def sde(self, input1, input2):
    """Create the drift and diffusion functions for the reverse SDE/ODE."""
    drift, diffusion = input1, input2
    score = input1, input2
    drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5)
    # Set the diffusion function to zero for ODEs.
    diffusion = 0
    return drift, diffusion

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def preprocess_data(data, time_len, rate):
    train_size = int(time_len * rate)
    train_data = data[:, 0:train_size]
    test_data = data[:, train_size:time_len]


    return train_data, test_data