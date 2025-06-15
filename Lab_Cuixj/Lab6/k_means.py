import math
import random
from typing import List, Set, Tuple
from sklearn import datasets 
import numpy as np
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt
class Point:
    def __init__(self, coordinates: List[float]):
        self.coordinates = coordinates  # 存储n维属性的列表
        

def euclidean_distance(p1: Point, p2: Point) -> float:
    """计算n维空间中的欧氏距离"""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1.coordinates, p2.coordinates)))

def k_means(data: List[Point], k: int, max_iterations: int = 100) -> Tuple[List[Set[Point]], List[Point]]:
    """
    n维K-means聚类算法
    :param data: 包含n维属性的点列表
    :param k: 簇数量
    :param max_iterations: 最大迭代次数（防止不收敛）
    :return: (簇集合列表, 中心点列表)
    """
    # 1. 随机初始化k个中心点（从数据中选取）
    centers = random.sample(data, k)
    clusters = [set() for _ in range(k)]
    
    for _ in range(max_iterations):
        # 2. 清空当前簇
        for cluster in clusters:
            cluster.clear()
        
        # 3. 分配点到最近的中心点对应簇
        for point in data:
            closest_center_idx = min(
                range(k),
                key=lambda i: euclidean_distance(point, centers[i])
            )
            clusters[closest_center_idx].add(point)
        
        # 4. 计算新中心点
        new_centers = []
        for cluster in clusters:
            if not cluster:
                # 如果簇为空，保持原中心点（或重新随机初始化）
                new_centers.append(centers[clusters.index(cluster)])
                continue
            # 计算簇中所有点在每个维度上的均值
            new_coords = [
                sum(p.coordinates[dim] for p in cluster) / len(cluster)
                for dim in range(len(data[0].coordinates))
            ]
            new_centers.append(Point(new_coords))
        
        # 5. 检查中心点是否变化
        if all(new == old for new, old in zip(new_centers, centers)):
            break
            
        centers = new_centers
    
    return clusters, centers

def plot_clusters(clusters: List[Set[Point]], centers: List[Point]):
    """可视化聚类结果（自动适配2D或3D数据）"""
    dim = len(centers[0].coordinates)
    colors = ['red', 'blue', 'green']  # 颜色池（最多支持5个簇）
    markers = ['o', 's', '^']  # 标记形状池
    
    
    plt.figure(figsize=(8, 6))
    for i, cluster in enumerate(clusters):   
        x = [p.coordinates[0] for p in cluster]
        y = [p.coordinates[1] for p in cluster]
        plt.scatter(x, y, c=colors[i], marker=markers[i], label=f'Cluster {i}', alpha=0.7)
    
    # 绘制中心点
    center_x = [c.coordinates[0] for c in centers]
    center_y = [c.coordinates[1] for c in centers]
    plt.scatter(center_x, center_y, c='black', marker='X', s=200, label='Centers')
    
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    
    plt.title("K-means Clustering (Manual Implementation)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__=='__main__':
    iris = datasets.load_iris()
    X = iris.data[:, :4]    # 表示我们取特征空间中的4个维度
    points = [Point(coords.tolist()) for coords in X]
    clusters, centers = k_means(points, k=3)
    plot_clusters(clusters,centers)
