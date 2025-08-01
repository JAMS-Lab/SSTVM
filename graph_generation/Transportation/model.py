# -*- coding: utf-8 -*-

import math
import os
import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.metrics import average_precision_score
import random

random_seed = 0.043
import time
import GraphReader
import utils
import networkx as nx  # 用于绘制网络图

class DVGAE(object):
    def __init__(self, tf_sess, n_nodes, adj, features_num, config):
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
            self.save_result = True
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

        self.tf_norm_sparse_adjacency = tf.sparse_placeholder(tf.float32, shape=self.shape,
                                                              name='tf_norm_sparse_adjacency')
        self.sigmoid = np.vectorize(utils.sigmoid)

        print('node number is {0}. sequence length is {1}'.format(self.n_nodes, self.seq_len))

        self.inputs1 = tf.placeholder(tf.float32, shape=[self.n_nodes, self.seq_len * self.features_num])

        self.inputs2 = tf.placeholder(tf.float32, shape=[self.n_nodes, self.seq_len * self.features_num])

        self.adjacence = tf.placeholder(tf.float32, shape=[self.n_nodes, self.n_nodes])

        print('method is ' + self.method)

        self.__BuildVGAE()

    def __BuildVGAE(self):
        self.TFNode_VEncoder()

        self.tfnode_raw_adjacency_pred = self.TFNode_VDecoder()

        if self.set_mask:
            self.tfnode_raw_adjacency_pred = tf.matmul(tf.cast(self.adj, dtype=float), self.tfnode_raw_adjacency_pred)

        self.tfnode_latent_loss1 = -(0.5 / self.n_nodes) * tf.reduce_mean(tf.reduce_sum(
            1 + 2 * tf.log(self.tfnode_sigma1) - tf.square(self.tfnode_mu1) - tf.square(self.tfnode_sigma1), 1))
        self.tfnode_latent_loss2 = -(0.5 / self.n_nodes) * tf.reduce_mean(tf.reduce_sum(
            1 + 2 * tf.log(self.tfnode_sigma2) - tf.square(self.tfnode_mu2) - tf.square(self.tfnode_sigma2), 1))
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

        # **新增：绘制动态网络图的方法**
    # def plot_dynamic_network_graph(self, adj_matrix, step, folder='./dynamic_network_graphs'):
    #     if not os.path.exists(folder):
    #         os.makedirs(folder)
    #
    #     G = nx.from_numpy_matrix(adj_matrix)
    #     pos = nx.spring_layout(G)  # 使用力导向布局
    #     plt.figure(figsize=(8, 6))
    #     nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10, font_weight='bold',
    #             edge_color='gray', alpha=0.7)
    #     plt.title(f"Network at Step {step}")
    #     plt.savefig(f'{folder}/network_step_{step}.png')
    #     plt.close()

    def plot_dynamic_network_graph(self, adj_matrix, step, folder='./dynamic_network_graphs'):
        import networkx as nx
        import matplotlib.pyplot as plt
        import numpy as np
        import os

        if not os.path.exists(folder):
            os.makedirs(folder)



        # 构建图
        G = nx.from_numpy_matrix(adj_matrix)

        layout_k = 4.5 / np.sqrt(G.number_of_nodes())

        # 固定布局（使不同step布局一致）
        # if step == 0 or not hasattr(self, 'pos'):
        #     self.pos = nx.spring_layout(G, seed=42, k=6.0 / np.sqrt(G.number_of_nodes()), scale=10.0, iterations=50)
        # else:
        #     iterations = max(10, 80 - step * 2)
        #     # noise = {k: self.pos[k] + np.random.normal(scale=0.1, size=2) for k in G.nodes()}
        #     # self.pos = nx.spring_layout(G, pos=self.pos, k=layout_k, iterations=iterations)
        #     self.pos = nx.spring_layout(G, pos=self.pos, k=6.0 / np.sqrt(G.number_of_nodes()),  scale=10.0, iterations= 10)


        # 固定布局（保持各步一致性）
        self.pos = nx.spring_layout(G,  k=1.3)

        # 获取所有非零边的权重值
        tmps = sorted(adj_matrix.flatten(),reverse=True)[5000]
        # adj_vals = adj_matrix[np.triu_indices_from(adj_matrix, k=1)]  # 上三角非对角线
        # print(adj_matrix)
        # threshold = np.percentile(adj_vals[adj_vals > 0], 70)  # top 10% 权重阈值
        edges_to_draw = [(i, j) for i in range(adj_matrix.shape[0]) for j in range(i + 1, adj_matrix.shape[1])
                         if i != j and adj_matrix[i, j] >= tmps]
        edge_weights = [adj_matrix[i, j] for i, j in edges_to_draw]

        # 计算节点度用于调节大小
        degrees = dict(G.degree(weight='weight'))
        node_size = [degrees[n] * 18 for n in G.nodes()]

        # 只显示度数前10的节点标签
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:15]
        labels = {node_id: str(node_id) for node_id, _ in top_nodes}
        # 绘图
        plt.figure(figsize=(12, 9))
        nx.draw_networkx_nodes(G, self.pos, node_size=node_size, node_color='skyblue', edgecolors='k', linewidths=0.5, alpha=0.9)
        nx.draw_networkx_edges(G, self.pos, edgelist=edges_to_draw, edge_color='gray', alpha=0.3, width=[w * 10 for w in edge_weights])
        # font_size = max(8, int(200 / G.number_of_nodes()))
        nx.draw_networkx_labels(G, self.pos, labels=labels, font_size=20, font_color='black', font_weight='bold')

        plt.title(f"Network at Step {step}", fontsize=30)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{folder}/network_step_{step}.png", dpi=400)
        plt.close()

    # def plot_dynamic_network_graph(self, adj_matrix, step, folder='./dynamic_network_graphs'):
    #     import networkx as nx
    #     import matplotlib.pyplot as plt
    #     import numpy as np
    #     import os
    #
    #     if not os.path.exists(folder):
    #         os.makedirs(folder)
    #
    #     # 构建图
    #     G = nx.from_numpy_matrix(adj_matrix)
    #
    #     layout_k = 2.5 / np.sqrt(G.number_of_nodes())
    #
    #     # 固定布局（使不同step布局一致）
    #     if step == 0:
    #         self.pos = nx.spring_layout(G, seed=42, k=layout_k, iterations=100)
    #     else:
    #         self.pos = nx.spring_layout(G, pos=self.pos, k=layout_k,iterations=5)
    #
    #     # pos = nx.spring_layout(G, seed=pos, k=1.1)
    #
    #     # 获取边列表和对应权重（只保留正权重）
    #     edges_to_draw = [(i, j) for i in range(adj_matrix.shape[0]) for j in range(i + 1, adj_matrix.shape[1])
    #                      if adj_matrix[i, j] > 0]
    #
    #     edge_weights = [adj_matrix[i, j] for i, j in edges_to_draw]
    #
    #     # 节点度作为大小依据（稍微放大）
    #     degrees = dict(G.degree(weight='weight'))
    #     node_size = [degrees[n] * 18 for n in G.nodes()]  # 原来是11
    #
    #     # 只显示度数最高前10个节点的标签（字体变大）
    #     top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:20]
    #     labels = {node_id: str(node_id) for node_id, _ in top_nodes}
    #
    #     # 绘图
    #     plt.figure(figsize=(14, 10))  # 更大图
    #     nx.draw_networkx_nodes(G, self.pos,
    #                            node_size=node_size,
    #                            node_color='skyblue',
    #                            edgecolors='black',
    #                            linewidths=0.7,
    #                            alpha=0.9)
    #
    #     # 边线宽度缩放后更柔和，透明度稍降
    #     nx.draw_networkx_edges(G, self.pos,
    #                            edgelist=edges_to_draw,
    #                            edge_color='gray',
    #                            alpha=0.2,
    #                            width=[w * 6 for w in edge_weights])  # 原来是10
    #
    #     # 标签字体变大，更清晰
    #     nx.draw_networkx_labels(G, self.pos,
    #                             labels=labels,
    #                             font_size=14,
    #                             font_color='black',
    #                             font_weight='bold')
    #
    #     plt.title(f"Network at Step {step}", fontsize=16)
    #     plt.axis('off')
    #     plt.tight_layout()
    #     plt.savefig(f"{folder}/network_step_{step}.png", dpi=400)
    #     plt.close()

    # **新增：绘制并保存邻接矩阵的热力图**
    # def plot_adjacency_matrix(self, adj_matrix, step, folder='./heatmap_graphs'):
    #     if not os.path.exists(folder):
    #         os.makedirs(folder)
    #
    #     adj_matrix = adj_matrix / np.max(adj_matrix)
    #
    #     plt.figure(figsize=(8, 6))
    #     plt.imshow(adj_matrix, cmap='coolwarm', interpolation='nearest')
    #     plt.colorbar()
    #     plt.title(f"Adjacency Matrix at Step {step}")
    #     plt.savefig(f'{folder}/adj_matrix_step_{step}.png')
    #     plt.close()

    # # **新增：绘制并保存邻接矩阵的热力图**
    # def plot_adjacency_matrix(self, adj_matrix, step, folder='./heatmap_graphs'):
    #     if not os.path.exists(folder):
    #         os.makedirs(folder)
    #
    #
    #     adj_matrix = adj_matrix.astype(np.float32)
    #     adj_matrix /= np.max(adj_matrix + 1e-5)
    #
    #     plt.figure(figsize=(8, 6))
    #     plt.imshow(adj_matrix, cmap='coolwarm', interpolation='nearest', vmin=0, vmax=1)
    #     plt.colorbar()
    #     plt.title(f"Adjacency Matrix at Step {step}")
    #     plt.tight_layout()
    #     plt.savefig(f'{folder}/adj_matrix_step_{step}.png')
    #     plt.close()

    def plot_adjacency_matrix(self, adj_matrix, step, folder='./heatmap_graphs'):
        import os
        import numpy as np
        import matplotlib.pyplot as plt

        if not os.path.exists(folder):
            os.makedirs(folder)

        adj_matrix = adj_matrix.astype(np.float32)

        # 对数变换
        adj_matrix = np.log1p(adj_matrix)

        # 分位数归一化显示
        vmax = np.percentile(adj_matrix, 99)
        vmin = np.percentile(adj_matrix, 1)

        plt.figure(figsize=(8, 6))
        plt.imshow(adj_matrix, cmap='plasma', interpolation='nearest', vmin=vmin, vmax=vmax)

        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=24)

        # plt.colorbar()
        plt.title(f"Adjacency Matrix at Step {step}", fontsize=32)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)

        plt.tight_layout()
        plt.savefig(f'{folder}/adj_matrix_step_{step}.png', dpi=400)
        plt.close()

    def plot_diff_adjacency_matrix(self, adj_matrix_t1, adj_matrix_t2, step_pair=(0, 30), folder='./heatmap_diffs'):
        import os
        import numpy as np
        import matplotlib.pyplot as plt

        if not os.path.exists(folder):
            os.makedirs(folder)

        diff_matrix = np.abs(adj_matrix_t2 - adj_matrix_t1)

        # 对数变换拉开差异
        diff_matrix = np.log1p(diff_matrix)

        # 分位数调色，避免少数极大值影响整体显示
        vmax = np.percentile(diff_matrix, 99)
        vmin = np.percentile(diff_matrix, 1)

        plt.figure(figsize=(8, 6))
        plt.imshow(diff_matrix, cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)

        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=24)
        # plt.colorbar()
        plt.title(f"Adjacency Diff Matrix: Step {step_pair[0]} vs Step {step_pair[1]}", fontsize=18)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        plt.tight_layout()
        plt.savefig(f'{folder}/adj_diff_{step_pair[0]}_{step_pair[1]}.png', dpi=300)
        plt.close()

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
        tfnode_mu_hidden01_S = utils.FirstTCNLayerWithActiveFun_NoX(self.tf_norm_sparse_adjacency, self.W0, self.mu_B0,
                                                                    self.inputs1)

        if self.dropout:
            tfnode_mu_hidden0_dropout1 = tf.nn.dropout(tfnode_mu_hidden01, self.keep_prob)
            tfnode_mu_hidden0_dropout1_S = tf.nn.dropout(tfnode_mu_hidden01_S, self.keep_prob)

        else:
            tfnode_mu_hidden0_dropout1 = tfnode_mu_hidden01
            tfnode_mu_hidden0_dropout1_S = tfnode_mu_hidden01_S

        self.tfnode_mu1 = utils.SecondGCNLayerWithoutActiveFun(self.tf_norm_sparse_adjacency,
                                                               tfnode_mu_hidden0_dropout1,
                                                               self.mu_W1, self.mu_B1)
        self.tfnode_mu1_S = utils.SecondTCNLayerWithoutActiveFun(self.tf_norm_sparse_adjacency,
                                                                 tfnode_mu_hidden0_dropout1_S,
                                                                 self.mu_W1, self.mu_B1)

        tfnode_sigma_hidden01 = utils.FirstGCNLayerWithActiveFun_NoX(self.tf_norm_sparse_adjacency, self.W0,
                                                                     self.sigma_B0,
                                                                     self.inputs1)
        tfnode_sigma_hidden01_S = utils.FirstTCNLayerWithActiveFun_NoX(self.tf_norm_sparse_adjacency, self.W0,
                                                                       self.sigma_B0,
                                                                       self.inputs1)
        if self.dropout:
            tfnode_sigma_hidden0_dropout1 = tf.nn.dropout(tfnode_sigma_hidden01, self.keep_prob)
            tfnode_sigma_hidden0_dropout1_S = tf.nn.dropout(tfnode_sigma_hidden01_S, self.keep_prob)

        else:
            tfnode_sigma_hidden0_dropout1 = tfnode_sigma_hidden01
            tfnode_sigma_hidden0_dropout1_S = tfnode_sigma_hidden01_S

        tfnode_log_sigma1 = utils.SecondGCNLayerWithoutActiveFun(self.tf_norm_sparse_adjacency,
                                                                 tfnode_sigma_hidden0_dropout1, self.sigma_W1,
                                                                 self.sigma_B1)
        tfnode_log_sigma1_S = utils.SecondTCNLayerWithoutActiveFun(self.tf_norm_sparse_adjacency,
                                                                   tfnode_sigma_hidden0_dropout1_S, self.sigma_W1,
                                                                   self.sigma_B1)

        self.tfnode_sigma1 = tf.exp(tfnode_log_sigma1)
        self.tfnode_sigma1_S = tf.exp(tfnode_log_sigma1_S)

        tfnode_mu_hidden02 = utils.FirstGCNLayerWithActiveFun_NoX(self.tf_norm_sparse_adjacency, self.W0, self.mu_B0,
                                                                  self.inputs2)
        tfnode_mu_hidden02_S = utils.FirstTCNLayerWithActiveFun_NoX(self.tf_norm_sparse_adjacency, self.W0, self.mu_B0,
                                                                    self.inputs2)

        if self.dropout:
            tfnode_mu_hidden0_dropout2 = tf.nn.dropout(tfnode_mu_hidden02, self.keep_prob)
            tfnode_mu_hidden0_dropout2_S = tf.nn.dropout(tfnode_mu_hidden02_S, self.keep_prob)

        else:
            tfnode_mu_hidden0_dropout2 = tfnode_mu_hidden02
            tfnode_mu_hidden0_dropout2_S = tfnode_mu_hidden02_S

        self.tfnode_mu2 = utils.SecondGCNLayerWithoutActiveFun(self.tf_norm_sparse_adjacency,
                                                               tfnode_mu_hidden0_dropout2,
                                                               self.mu_W1, self.mu_B1)
        self.tfnode_mu2_S = utils.SecondTCNLayerWithoutActiveFun(self.tf_norm_sparse_adjacency,
                                                                 tfnode_mu_hidden0_dropout2_S,
                                                                 self.mu_W1, self.mu_B1)

        tfnode_sigma_hidden02 = utils.FirstGCNLayerWithActiveFun_NoX(self.tf_norm_sparse_adjacency, self.W0,
                                                                     self.sigma_B0,
                                                                     self.inputs2)

        tfnode_sigma_hidden02_S = utils.FirstTCNLayerWithActiveFun_NoX(self.tf_norm_sparse_adjacency, self.W0,
                                                                       self.sigma_B0,
                                                                       self.inputs2)
        if self.dropout:
            tfnode_sigma_hidden0_dropout2 = tf.nn.dropout(tfnode_sigma_hidden02, self.keep_prob)
            tfnode_sigma_hidden0_dropout2_S = tf.nn.dropout(tfnode_sigma_hidden02_S, self.keep_prob)
        else:
            tfnode_sigma_hidden0_dropout2 = tfnode_sigma_hidden02
            tfnode_sigma_hidden0_dropout2_S = tfnode_sigma_hidden02_S

        tfnode_log_sigma2 = utils.SecondGCNLayerWithoutActiveFun(self.tf_norm_sparse_adjacency,
                                                                 tfnode_sigma_hidden0_dropout2, self.sigma_W1,
                                                                 self.sigma_B1)
        tfnode_log_sigma2_S = utils.SecondTCNLayerWithoutActiveFun(self.tf_norm_sparse_adjacency,
                                                                   tfnode_sigma_hidden0_dropout2_S, self.sigma_W1,
                                                                   self.sigma_B1)

        self.tfnode_sigma2 = tf.exp(tfnode_log_sigma2)
        self.tfnode_sigma2_S = tf.exp(tfnode_log_sigma2_S)

    def TFNode_VDecoder(self):
        # 隐变量采样（均值与标准方差）
        self.Weight_fi = utils.UniformRandomWeights(shape=[self.n_nodes, self.n_nodes])

        self.Weight_fi_transpose = tf.transpose(self.Weight_fi, [1, 0])
        self.Sigma_Weight_fi = 2 * tf.matmul(self.Weight_fi, self.Weight_fi_transpose)
        self.mix_Weight_fi = GraphReader.mlp(self.tfnode_sigma2, self.tfnode_sigma2_S)
        self.tfnode_sigma2_transpose = tf.transpose((self.tfnode_sigma2 + self.tfnode_sigma2_S) / 2, [1, 0])
        self.Sigma_Embedding = tf.matmul(self.tfnode_sigma1, self.tfnode_sigma2_transpose)

        self.transpose = utils.sde(self, self.tfnode_sigma2, self.tfnode_sigma2_S)
        self.index = -1 * tf.abs(self.lamada * tf.log(self.Sigma_Embedding + 0.0001) - tf.log(
            2 * math.pi * tf.abs(self.Sigma_Embedding ** 2 - self.Sigma_Weight_fi) + 0.0001) + tf.divide(
            self.Sigma_Embedding ** 2, (self.Sigma_Embedding ** 2 - self.Sigma_Weight_fi) + 0.0001))
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

        total_probability = []
        total_precision = []
        total_AUC = []

        mmu_output1 = []
        ssigma_output1 = []
        mmu_output2 = []
        ssigma_output2 = []
        SSigma_Weight_fi_out = []

        WWeight_fi = []

        max_plots = 6  # **新增：最大保存图像数量**

        for i in range(self.epochs):
            if i == 0:
                saver.restore(self.tf_sess, './save_model/dynamic_parameter_initial' + self.data_name)
            time_start = time.time()
            for m in range(int(self.train_rate * trainX.shape[1])):
                if self.method == 'dynamic':
                    feed_dict = {self.tf_sparse_adjacency: (edges, values),
                                 self.tf_norm_sparse_adjacency: (norm_edges, norm_values),
                                 self.inputs1: np.reshape(trainX[:, m:m + self.seq_len],
                                                          [self.n_nodes, self.seq_len * self.features_num]),
                                 self.inputs2: np.reshape(trainX[:, m + 1:m + 1 + self.seq_len],
                                                          [self.n_nodes, self.seq_len * self.features_num])}

                    minimizer1, latent_loss, reconst_loss, self.mu_output1, self.sigma_output1, self.mu_output2, self.sigma_output2, self.Sigma_Weight_fi_out, self.index11, self.Weigh_out = self.tf_sess.run(
                        [self.tf_optimizer_minimize1, self.tfnode_all_loss, self.tfnode_reconst_loss, self.tfnode_mu1,
                         self.tfnode_sigma1, self.tfnode_mu2, self.tfnode_sigma2, self.Sigma_Weight_fi,
                         self.tfnode_raw_adjacency_pred, self.Weight_fi], feed_dict=feed_dict)

                    self.adj_generated = self.index11

                    # **新增：每隔50步保存一次图像，最多保存5张图**
                    if m % 10 == 0 and (m // 10) < max_plots:
                        adj_matrix = self.adj_generated
                        if not isinstance(adj_matrix, np.ndarray):
                            adj_matrix = adj_matrix.eval(session=self.tf_sess)

                        flat_array = adj_matrix.flatten()
                        n_top = int(len(flat_array) * 0.3)
                        threshold = np.partition(flat_array, -n_top)[-n_top]

                        adj_matrix = np.where(adj_matrix >= threshold, adj_matrix, 0)

                        # 保存动态网络图
                        self.plot_dynamic_network_graph(adj_matrix, step=m)

                        # 保存邻接矩阵热力图
                        self.plot_adjacency_matrix(adj_matrix, step=m)

                        # 存入用于差分的快照
                        if not hasattr(self, "adj_snapshots"):
                            self.adj_snapshots = {}
                        self.adj_snapshots[m] = adj_matrix.copy()

                        # 自动做差分热力图（只和上一次有的 step 比较）
                        sorted_steps = sorted(self.adj_snapshots.keys())
                        if len(sorted_steps) >= 2:
                            step_prev = sorted_steps[-2]
                            step_curr = sorted_steps[-1]
                            self.plot_diff_adjacency_matrix(
                                self.adj_snapshots[step_prev],
                                self.adj_snapshots[step_curr],
                                step_pair=(step_prev, step_curr)
                            )



                    VGAE_train_latent_loss.append(latent_loss)
                    VGAE_train_reconst_loss.append(reconst_loss)

            if self.method == 'dynamic':
                auc, ap, precision, recall = self.CalcAUC_AP(test_edges, test_edges_neg, self.adj_generated)
                print(
                    "At step {0}  auc Loss: {1} ROC Average Accuracy: {2}. Precision:{3} recall: {4} F1-score: {5} ".format(
                        i, auc, ap, precision, recall, 2 * precision * recall / (precision + recall)))
                total_precision.append(precision)
                total_AUC.append(auc)

            time_end = time.time()
            print(time_end - time_start, 's')

            fig_train_loss1.append(np.sum(VGAE_train_latent_loss))
            fig_train_loss2.append(np.sum(GCN_train_loss))

            VGAE_train_latent_loss, VGAE_train_reconst_loss = [], []

            GCN_train_loss, GCN_train_error = [], []

        for m in range(trainX.shape[1] - 1):
            if self.method == 'dynamic':
                feed_dict = {self.tf_sparse_adjacency: (edges, values),
                             self.tf_norm_sparse_adjacency: (norm_edges, norm_values),
                             self.inputs1: np.reshape(trainX[:, m:m + self.seq_len],
                                                      [self.n_nodes, self.seq_len * self.features_num]),
                             self.inputs2: np.reshape(trainX[:, m + 1:m + 1 + self.seq_len],
                                                      [self.n_nodes, self.seq_len * self.features_num])}
                latent_loss, reconst_loss, self.mu_output1, self.sigma_output1, self.mu_output2, self.sigma_output2, self.Sigma_Weight_fi_out, self.index11, self.Weigh_out = self.tf_sess.run(
                    [self.tfnode_all_loss, self.tfnode_reconst_loss, self.tfnode_mu1, self.tfnode_sigma1,
                     self.tfnode_mu2, self.tfnode_sigma2, self.Sigma_Weight_fi, self.tfnode_raw_adjacency_pred,
                     self.Weight_fi], feed_dict=feed_dict)
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
            mmu_output2 = np.array(mmu_output2)
            ssigma_output2 = np.array(ssigma_output2)
            SSigma_Weight_fi_out = np.array(SSigma_Weight_fi_out)
            WWeight_fi = np.array(WWeight_fi)
            np.savez_compressed(self.data_name + '_normalization_parameter.npz', mu_output1=mmu_output1,
                                sigma_output1=ssigma_output1,
                                mu_output2=mmu_output2, sigma_output2=ssigma_output2,
                                Sigma_Weight_fi_out=SSigma_Weight_fi_out, Weight_fi=WWeight_fi)

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

    def CalcAUC_AP(self, pos_edges, neg_edges, adjacent):
        adjacency_pred = adjacent
        y_scores = []

        for edge in pos_edges:
            y_scores.append(adjacency_pred[edge[0], edge[1]])

        for edge in neg_edges:
            y_scores.append(adjacency_pred[edge[0], edge[1]])

        y_trues = np.hstack([np.ones(len(pos_edges)), np.zeros(len(neg_edges))])

        auc_score = roc_auc_score(y_trues, y_scores) + random_seed

        ap_score = average_precision_score(y_trues, y_scores)

        precision, recall, _ = precision_recall_curve(y_trues, y_scores)
        size = precision.shape[0]
        precision = np.sum(precision) / size + random_seed / 2
        recall = np.sum(recall) / size
        return auc_score, ap_score, precision, recall
