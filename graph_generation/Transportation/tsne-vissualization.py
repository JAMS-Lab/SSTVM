import os

import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.cluster import KMeans
import collections
import importlib_metadata

# 1. 加载模型训练保存的隐变量数据
data = np.load('Tdrive_normalization_parameter.npz')  # 替换为你的文件名

mu_output1 = data['mu_output1']  # 形状一般是 (time_steps, n_nodes, embedding_dim)

time_steps = mu_output1.shape[0]
n_nodes = mu_output1.shape[1]
embedding_dim = mu_output1.shape[2]

samples = mu_output1.reshape(-1, embedding_dim)
labels = np.repeat(np.arange(time_steps), n_nodes)

# 1. 用UMAP降维
reducer = UMAP(random_state=42)
embedding_2d = reducer.fit_transform(samples)

# 2. 离散化时间步
num_bins = 10
labels_binned = np.floor(labels / (time_steps / num_bins)).astype(int)

# 3. 统计每个bin点数，确认合理划分
counter = collections.Counter(labels_binned)
print("Points count in each bin:", counter)

# 4. 绘制整体图，颜色按时间段分类
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=labels_binned, cmap='tab10', s=10, alpha=0.9)
# plt.colorbar(scatter, label='Time step bins')
cbar = plt.colorbar(scatter)
cbar.set_label('Time step bins', fontsize=28)
cbar.ax.tick_params(labelsize=24)

plt.title('UMAP visualization  by time step ', fontsize=32)
plt.xlabel('Dimension 1', fontsize=30)
plt.ylabel('Dimension 2', fontsize=30)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.tight_layout()
plt.savefig("umap_by_time.png")
plt.show()

n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(embedding_2d)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=cluster_labels, cmap='tab10', s=10, alpha=0.9)
plt.colorbar(scatter, label='Cluster ID')
plt.title('UMAP Visualization by KMeans Cluster', fontsize=32)
plt.xlabel('UMAP Dimension 1', fontsize=30)
plt.ylabel('UMAP Dimension 2', fontsize=30)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.tight_layout()
plt.savefig("umap_by_cluster.png")
plt.show()

from sklearn.utils import resample

# 按 bin 平均采样每个时间段的点数（比如每段采样200个）
sampled_points = []
sampled_labels = []

points_per_bin = 200  # 可根据图大小微调

for b in range(num_bins):
    idx = np.where(labels_binned == b)[0]
    if len(idx) >= points_per_bin:
        idx_sampled = resample(idx, n_samples=points_per_bin, random_state=42, replace=False)
    else:
        idx_sampled = idx  # 不足的保留全部
    sampled_points.append(embedding_2d[idx_sampled])
    sampled_labels.append(np.full(len(idx_sampled), b))

embedding_sampled = np.vstack(sampled_points)
labels_sampled = np.concatenate(sampled_labels)

# 5. 分时间段单独绘图
output_folder = "./umap_bins"
os.makedirs(output_folder, exist_ok=True)
for bin_id in range(num_bins):
    plt.figure(figsize=(10, 8))
    idx = labels_binned == bin_id
    plt.scatter(embedding_2d[idx, 0], embedding_2d[idx, 1], s=10, alpha=0.8, color='steelblue')
    plt.title(f'Time step bin {bin_id}', fontsize=32)
    plt.xlabel('UMAP Dimension 1', fontsize=30)
    plt.ylabel('UMAP Dimension 2', fontsize=30)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.tight_layout()
    plt.savefig(f"{output_folder}/bin_{bin_id}.png")
    plt.close()