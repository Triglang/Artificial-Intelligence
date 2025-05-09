import numpy as np
from sklearn.cluster import KMeans
from typing import List, Set, Tuple
from sklearn import datasets 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # 用于3D图


def print_3d(n_clusters,data_np):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')  # 创建3D坐标轴
    for i in range(n_clusters):
        cluster_points = data_np[labels == i]
        ax.scatter(
            cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
            c=colors[i], label=f'Cluster {i}', alpha=0.7
        )

    # 绘制簇中心
    ax.scatter(
        centers[:, 0], centers[:, 1], centers[:, 2],
        c='black', marker='X', s=200, label='Centers'
    )

    # 添加标签和标题
    ax.set_title("K-means Clustering (K=3)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()

def print_2d(n_clusters,data_np):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # 绘制数据点
    plt.figure(figsize=(8, 6))
    for i in range(n_clusters):
        # 筛选属于当前簇的点
        cluster_points = data_np[labels == i]
        plt.scatter(
            cluster_points[:, 0], cluster_points[:, 1],
            c=colors[i], label=f'Cluster {i}', alpha=0.7
        )

    # 绘制簇中心
    plt.scatter(
        centers[:, 0], centers[:, 1],
        c='black', marker='X', s=200, label='Centers'
    )

    # 添加标签和标题
    plt.title("K-means Clustering (K=3)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__=='__main__':
    # 数据集 1
    train_set_path = 'data/data1.csv'
    df = pd.read_csv(train_set_path)
    data_np = df.to_numpy()
    data_np=data_np[:100]

    # 数据集 2
    # iris = datasets.load_iris()
    # data_np = iris.data[:, :4]    # 表示我们取特征空间中的4个维度
    
    n_clusters=4
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data_np)  # 输出每个点的簇标签 (0或1)
    centers = kmeans.cluster_centers_

    print_3d(n_clusters,data_np)
    
    
    