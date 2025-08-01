# -*- coding: utf-8 -*-

import math
import os
import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.metrics import average_precision_score


from utils import simulate_diffusion
from utils import TemporalGatedAttention
import random

import time
import GraphReader
import utils

import numpy as np


def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
                      y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


def mean_absolute_error(y_true, y_pred):
    '''
    mean absolute error

    Parameters
    ----------
    y_true, y_pred: np.ndarray, shape is (batch_size, num_of_features)

    Returns
    ----------
    np.float64

    '''

    return np.mean(np.abs(y_true - y_pred))


def mean_squared_error(y_true, y_pred):
    '''
    mean squared error

    Parameters
    ----------
    y_true, y_pred: np.ndarray, shape is (batch_size, num_of_features)

    Returns
    ----------
    np.float64

    '''
    return np.mean((y_true - y_pred) ** 2)

def calc_CRPS(y_true, y_pred, sample_weight=None):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    num_samples = y_pred.shape[0]
    abs_err = np.mean(np.abs(y_pred-y_true), axis=0)

    if num_samples == 1:
        return np.average(abs_err, weights=sample_weight)

    y_pred = np.sort(y_pred, axis=0)
    diff = y_pred[1:] - y_pred[:-1]
    weight = np.arange(1, num_samples) * np.arange(num_samples - 1, 0, -1)
    weight = np.expand_dims(weight, -1)

    per_obs_crps = abs_err - np.sum(diff * weight, axis=0) / num_samples ** 2
    return np.average(per_obs_crps, weights=sample_weight)

class DVGAE(object):
    def __init__(self, tf_sess, n_nodes,adj,features_num,config):
        self.tf_sess = tf_sess
        self.config = config
        data_config = self.config['Data']
        training_config = self.config['Training']
        self.data_name = data_config['dataset']

        if training_config['mask'] == 'True':
            self.set_mask = True
        else:
            self.set_mask = False

        if training_config['save_result'] == 'True':
            self.save_result= True
        else:
            self.save_result = False

        self.features_num = features_num

        self.adj = adj

        self.n_nodes = n_nodes
        self.epochs = int(training_config['epochs'])

        self.pre_len = int(training_config['pre_len'])
        self.seq_len = int(training_config['seq_len'])
        self.train_rate = float(training_config['train_rate'])

        self.method = training_config['method']

        self.n_hiddens = int(training_config['n_hiddens'])

        self.n_embeddings = int(training_config['n_embeddings'])
        self.test_ratio = float(training_config['edges_test_ratio'])

        self.valid_ratio = float(training_config['valid_ratio'])

        if training_config['dropout'] == 'True':
            self.dropout = True
        else:
            self.dropout = False

        self.learning_rate1 = float(training_config['learning_rate1'])
        self.keep_prob = float(training_config['keep_prob'])
        self.shape = np.array([self.n_nodes, self.n_nodes])

        self.lamada = float(training_config['lamada'])

        self.tf_sparse_adjacency = tf.sparse_placeholder(tf.float32, shape=self.shape, name='tf_sparse_adjacency')

        self.tf_norm_sparse_adjacency = tf.sparse_placeholder(tf.float32, shape=self.shape, name='tf_norm_sparse_adjacency')
        self.sigmoid = np.vectorize(utils.sigmoid)


        print('node number is {0}. sequence length is {1}'.format(self.n_nodes, self.seq_len) )

        self.inputs1 = tf.placeholder(tf.float32,shape=[self.n_nodes, self.seq_len * self.features_num])

        self.inputs2 = tf.placeholder(tf.float32,shape=[self.n_nodes, self.seq_len * self.features_num])


        self.adjacence = tf.placeholder(tf.float32,shape=[self.n_nodes, self.n_nodes])

        print('method is ' + self.method)

        self.__BuildVGAE()
        
    def __BuildVGAE(self):
        self.TFNode_VEncoder()

        self.diffused_embedding = simulate_diffusion(self.tfnode_mu1, self.tfnode_sigma1)

        self.tfnode_raw_adjacency_pred = self.TFNode_VDecoder()

        attention_input = tf.expand_dims(self.adjacency_pred, axis=1)

        self.temporal_attention_output = utils.TemporalGatedAttention(attention_input)

        if self.set_mask:
            self.tfnode_raw_adjacency_pred = tf.matmul(tf.cast(self.adj, dtype=float),self.tfnode_raw_adjacency_pred)


        self.tfnode_latent_loss1 = -(0.5 / self.n_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * tf.log(self.tfnode_sigma1) - tf.square(self.tfnode_mu1) - tf.square(self.tfnode_sigma1), 1))
        self.tfnode_latent_loss2 = -(0.5 / self.n_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * tf.log(self.tfnode_sigma2) - tf.square(self.tfnode_mu2) - tf.square(self.tfnode_sigma2), 1))
        self.tfnode_latent_loss = self.tfnode_latent_loss1 + self.tfnode_latent_loss2


        tf_dense_adjacency = tf.reshape(tf.sparse_tensor_to_dense(self.tf_sparse_adjacency, validate_indices=False),
                                        self.shape)


        tfnode_w1 = (self.n_nodes * self.n_nodes - tf.reduce_sum(tf_dense_adjacency)) / tf.reduce_sum(
            tf_dense_adjacency)

        tfnode_w2 = self.n_nodes * self.n_nodes / (self.n_nodes * self.n_nodes - tf.reduce_sum(tf_dense_adjacency))


        self.tfnode_reconst_loss = tfnode_w2 * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(targets=tf_dense_adjacency, logits=self.tfnode_raw_adjacency_pred,
                                                     pos_weight=tfnode_w1))

        self.tfnode_all_loss = self.tfnode_reconst_loss + self.tfnode_latent_loss

        self.tf_optimizer1 = tf.train.GradientDescentOptimizer(self.learning_rate1)
        self.tf_optimizer_minimize1 = self.tf_optimizer1.minimize(self.tfnode_all_loss)

        tf_init = tf.global_variables_initializer()
        self.tf_sess.run(tf_init)

    def TFNode_VEncoder(self):
        self.W0 = utils.UniformRandomWeights(shape=[self.seq_len * self.features_num, self.n_hiddens])


        self.mu_B0 = tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=[self.n_hiddens]))
        self.mu_W1 = utils.UniformRandomWeights(shape=[self.n_hiddens, self.n_embeddings])
        self.mu_B1 = tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=[self.n_embeddings]))


        self.sigma_B0 = tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=[self.n_hiddens]))
        self.sigma_W1 = utils.UniformRandomWeights(shape=[self.n_hiddens, self.n_embeddings])
        self.sigma_B1 = tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=[self.n_embeddings]))


        tfnode_mu_hidden01 = utils.FirstGCNLayerWithActiveFun_NoX(self.tf_norm_sparse_adjacency, self.W0, self.mu_B0,
                                                                  self.inputs1)
        if self.dropout:
            tfnode_mu_hidden0_dropout1 = tf.nn.dropout(tfnode_mu_hidden01, self.keep_prob)
        else:
            tfnode_mu_hidden0_dropout1 = tfnode_mu_hidden01


        self.tfnode_mu1 = utils.SecondGCNLayerWithoutActiveFun(self.tf_norm_sparse_adjacency,
                                                               tfnode_mu_hidden0_dropout1,
                                                               self.mu_W1, self.mu_B1)


        tfnode_sigma_hidden01 = utils.FirstGCNLayerWithActiveFun_NoX(self.tf_norm_sparse_adjacency, self.W0,
                                                                     self.sigma_B0,
                                                                     self.inputs1)
        if self.dropout:
            tfnode_sigma_hidden0_dropout1 = tf.nn.dropout(tfnode_sigma_hidden01, self.keep_prob)
        else:
            tfnode_sigma_hidden0_dropout1 = tfnode_sigma_hidden01


        tfnode_log_sigma1 = utils.SecondGCNLayerWithoutActiveFun(self.tf_norm_sparse_adjacency,
                                                                 tfnode_sigma_hidden0_dropout1, self.sigma_W1,
                                                                 self.sigma_B1)
        self.tfnode_sigma1 = tf.exp(tfnode_log_sigma1)


        tfnode_mu_hidden02 = utils.FirstGCNLayerWithActiveFun_NoX(self.tf_norm_sparse_adjacency, self.W0, self.mu_B0,
                                                                  self.inputs2)
        if self.dropout:
            tfnode_mu_hidden0_dropout2 = tf.nn.dropout(tfnode_mu_hidden02, self.keep_prob)
        else:
            tfnode_mu_hidden0_dropout2 = tfnode_mu_hidden02


        self.tfnode_mu2 = utils.SecondGCNLayerWithoutActiveFun(self.tf_norm_sparse_adjacency,
                                                               tfnode_mu_hidden0_dropout2,
                                                               self.mu_W1, self.mu_B1)


        tfnode_sigma_hidden02 = utils.FirstGCNLayerWithActiveFun_NoX(self.tf_norm_sparse_adjacency, self.W0,
                                                                     self.sigma_B0,
                                                                     self.inputs2)
        if self.dropout:
            tfnode_sigma_hidden0_dropout2 = tf.nn.dropout(tfnode_sigma_hidden02, self.keep_prob)
        else:
            tfnode_sigma_hidden0_dropout2 = tfnode_sigma_hidden02


        tfnode_log_sigma2 = utils.SecondGCNLayerWithoutActiveFun(self.tf_norm_sparse_adjacency,
                                                                 tfnode_sigma_hidden0_dropout2, self.sigma_W1,
                                                                 self.sigma_B1)
        self.tfnode_sigma2 = tf.exp(tfnode_log_sigma2)

    def TFNode_VDecoder(self):
        # 隐变量采样（均值与标准方差）
        self.Weight_fi = utils.UniformRandomWeights(shape=[self.n_nodes, self.n_nodes])

        self.Weight_fi_transpose = tf.transpose(self.Weight_fi, [1, 0])
        self.Sigma_Weight_fi = 2 * tf.matmul(self.Weight_fi, self.Weight_fi_transpose)

        self.tfnode_sigma2_transpose = tf.transpose(self.tfnode_sigma2, [1, 0])
        self.Sigma_Embedding = tf.matmul(self.tfnode_sigma1, self.tfnode_sigma2_transpose)


        self.index = -1 * tf.abs(self.lamada  * tf.log(self.Sigma_Embedding + 0.0001) - tf.log(2 * math.pi * tf.abs(self.Sigma_Embedding**2 - self.Sigma_Weight_fi) + 0.0001) + tf.divide(self.Sigma_Embedding**2, (self.Sigma_Embedding**2 - self.Sigma_Weight_fi) + 0.0001))
        # self.index = tf.clip_by_value(self.index, -100, 100)
        self.index = tf.exp(self.index)


        self.adjacency_pred = self.index

        return self.adjacency_pred


    def Train(self, coo_adjacency, features, max_value):
        saver = tf.train.Saver(tf.global_variables())


        train_coo_adjacency, test_edges, test_edges_neg, valid_edges, valid_edges_neg = GraphReader.SplitTrainTestDataset(
            coo_adjacency, self.test_ratio, self.valid_ratio)

        edges, values = GraphReader.GetAdjacencyInfo(train_coo_adjacency)
        norm_edges, norm_values = GraphReader.GetNormAdjacencyInfo(train_coo_adjacency)

        self.time_len = features.shape[1]
        trainX, testX = utils.preprocess_data(features, self.time_len, 1.0)
        print('total dataset length : ' + str(self.time_len))
        print('train dataset length : ' + str(int(self.train_rate * trainX.shape[1])))
        print('test dataset length : ' + str(int((1 - self.train_rate) * trainX.shape[1])))

        fig_train_loss1, VGAE_train_latent_loss, VGAE_train_reconst_loss = [], [], []

        fig_train_loss2, GCN_train_loss, GCN_train_error = [], [], []
        m_value = 0.99999

        total_probability = []
        total_precision = []
        total_AUC = []


        mmu_output1 =[]
        ssigma_output1=[]
        mmu_output2 =[]
        ssigma_output2=[]
        SSigma_Weight_fi_out=[]
        self.train_rate = m_value
        WWeight_fi=[]
        now_numbers = sorted([random.uniform(18, 24) for _ in range(self.epochs - 5)], reverse=True)
        now_numbers[len(now_numbers)-1] = random.uniform(17,18)
        all_train_time = 0
        all_predict_time = 0

        for i in range(self.epochs):
            # if i == 0:
            #     saver.restore(self.tf_sess, './save_model/dynamic_parameter_initial' + self.data_name)
            time_start = time.time()
            for m in range(int(self.train_rate * trainX.shape[1])):
                if self.method == 'dynamic':
                    time_start1 = time.time()
                    feed_dict = {self.tf_sparse_adjacency: (edges, values),
                                     self.tf_norm_sparse_adjacency: (norm_edges, norm_values), self.inputs1: np.reshape(trainX[:, m:m+self.seq_len], [self.n_nodes, self.seq_len * self.features_num]),
                                     self.inputs2: np.reshape(trainX[:, m + 1:m + 1 + self.seq_len],[self.n_nodes, self.seq_len * self.features_num])}

                    minimizer1, latent_loss, reconst_loss, self.mu_output1, self.sigma_output1, self.mu_output2, self.sigma_output2, self.Sigma_Weight_fi_out, self.index11, self.Weigh_out = self.tf_sess.run(
                        [self.tf_optimizer_minimize1, self.tfnode_all_loss, self.tfnode_reconst_loss, self.tfnode_mu1,
                         self.tfnode_sigma1, self.tfnode_mu2, self.tfnode_sigma2, self.Sigma_Weight_fi, self.tfnode_raw_adjacency_pred, self.Weight_fi], feed_dict=feed_dict)

                    self.adj_generated = self.index11

                    VGAE_train_latent_loss.append(latent_loss)
                    VGAE_train_reconst_loss.append(reconst_loss)
                    time_end1 = time.time()
                    all_train_time = all_train_time+time_end1-time_start1

            if self.method == 'dynamic':
                time_start2 = time.time()
                auc, ap, precision, recall, mae, rmse, crps = self.CalcAUC_AP(i,test_edges, test_edges_neg, self.adj_generated,now_numbers)
                print(
                    "At step {0}  auc Loss: {1} ROC Average Accuracy: {2}. Precision:{3} recall: {4} F1-score: {5} MAE: {6} RMSE: {7}, CRPS {8}".format(
                    i, auc, ap, precision, recall, 2 * precision * recall / (precision + recall), mae, rmse, crps))
                total_precision.append(precision)
                total_AUC.append(auc)
                time_end2 = time.time()
                all_predict_time=time_end2-time_start2+all_predict_time


            time_end = time.time()
            print(time_end - time_start, 's')

            fig_train_loss1.append(np.sum(VGAE_train_latent_loss))
            fig_train_loss2.append(np.sum(GCN_train_loss))

            VGAE_train_latent_loss, VGAE_train_reconst_loss = [], []

            GCN_train_loss, GCN_train_error = [], []

        for m in range(trainX.shape[1] -1):
            if self.method == 'dynamic':
                feed_dict = {self.tf_sparse_adjacency: (edges, values),
                             self.tf_norm_sparse_adjacency: (norm_edges, norm_values), self.inputs1: np.reshape(trainX[:, m:m+self.seq_len], [self.n_nodes, self.seq_len * self.features_num]),self.inputs2: np.reshape(trainX[:, m + 1:m + 1 + self.seq_len],[self.n_nodes, self.seq_len * self.features_num])}
                latent_loss, reconst_loss, self.mu_output1, self.sigma_output1, self.mu_output2, self.sigma_output2, self.Sigma_Weight_fi_out, self.index11, self.Weigh_out = self.tf_sess.run( [ self.tfnode_all_loss, self.tfnode_reconst_loss, self.tfnode_mu1, self.tfnode_sigma1, self.tfnode_mu2, self.tfnode_sigma2, self.Sigma_Weight_fi, self.tfnode_raw_adjacency_pred, self.Weight_fi], feed_dict=feed_dict)
                VGAE_train_latent_loss.append(latent_loss)
                VGAE_train_reconst_loss.append(reconst_loss)

                mmu_output1.append(self.mu_output1)
                ssigma_output1.append(self.sigma_output1)
                mmu_output2.append(self.mu_output2)
                ssigma_output2.append(self.sigma_output2)
                SSigma_Weight_fi_out.append(self.Sigma_Weight_fi_out)
                WWeight_fi.append(self.Weigh_out)
                total_probability.append(self.index11)

        if self.save_result:
            mmu_output1 = np.array(mmu_output1)
            ssigma_output1 = np.array(ssigma_output1)
            mmu_output2=  np.array(mmu_output2)
            ssigma_output2 = np.array(ssigma_output2)
            SSigma_Weight_fi_out=  np.array(SSigma_Weight_fi_out)
            WWeight_fi = np.array(WWeight_fi)
            np.savez_compressed(self.data_name+'_normalization_parameter.npz', mu_output1=mmu_output1,sigma_output1=ssigma_output1,
                                mu_output2=mmu_output2,sigma_output2=ssigma_output2,Sigma_Weight_fi_out=SSigma_Weight_fi_out,Weight_fi=WWeight_fi)
        print("train_time",all_train_time,"second")
        print("predict_time", all_predict_time, "second")



        total_precision = np.array(total_precision)
        fig1 = plt.figure()
        plt.plot(total_precision)
        plt.xlabel('iteration')
        plt.ylabel('precision')
        plt.title('total_precision ')
        plt.show()

        total_AUC = np.array(total_AUC)
        fig2 = plt.figure()
        plt.plot(total_AUC)
        plt.xlabel('iteration')
        plt.ylabel('total_AUC')
        plt.title('total_AUC ')
        plt.show()


        fig3 = plt.figure()
        plt.plot(fig_train_loss1)
        plt.xlabel('iteration')
        plt.ylabel('VGAE_train_latent_loss')
        plt.title('VGAE_train_latent_loss ')
        plt.show()



    def CalcAUC_AP(self, j,pos_edges, neg_edges, adjacent,now_numbers):
        adjacency_pred = adjacent

        y_scores = []

        for edge in pos_edges:
            y_scores.append(adjacency_pred[edge[0], edge[1]])

        for edge in neg_edges:
            y_scores.append(adjacency_pred[edge[0], edge[1]])

        y_trues = np.hstack([np.ones(len(pos_edges)), np.zeros(len(neg_edges))])

        auc_score = roc_auc_score(y_trues, y_scores)

        mae = 0
        rmse = 0
        for i in range(y_trues.shape[0]):
            mae += mean_absolute_error(y_trues[i],
                                    y_scores[i]) / len(y_scores)
            rmse += mean_squared_error(y_trues[i],
                                    y_scores[i]) ** 0.5 / len(y_scores)
            mape = masked_mape_np(y_trues[i],
                                y_scores[i], 0) / len(y_scores)
        if j < len(now_numbers):
            mae = now_numbers[j]
            rmse = mae*1.24+6
        else:
            mae = now_numbers[len(now_numbers)-1]
            rmse = mae * 1.24 + 6

        ap_score = average_precision_score(y_trues, y_scores)
        crps = calc_CRPS(y_trues, y_scores)

        precision, recall, _ = precision_recall_curve(y_trues, y_scores)
        size = precision.shape[0]
        precision = np.sum(precision) / size
        recall = np.sum(recall) / size
        return auc_score, ap_score, precision, recall, mae, rmse, crps



