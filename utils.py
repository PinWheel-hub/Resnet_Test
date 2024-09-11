import torch
import torch.nn as nn
from torchvision import models
import torch.utils.checkpoint
from typing import Tuple
import cv2
from torchvision import transforms as T
import cv2
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, OPTICS
from sklearn.metrics import silhouette_score

# resnet模型
resnets = {
    'resnet18': models.resnet18,
    'resnet50': models.resnet50,
    'wide_resnet50_2': models.wide_resnet50_2,
}
class ResNet_Encoder(nn.Module):
    def __init__(self, model='resnet18'):
        super().__init__()
        self.resnet = resnets[model](pretrained=True)
        self.layer0 = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu, self.resnet.maxpool)
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.transform = T.Compose([
            T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = self.transform(x).unsqueeze(0)
        h = self.layer0(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        return h

# 去除图像两侧的白边
def remove_white_cols(img: np.array):
    # 将图像转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 获取图像列的平均亮度
    col_means = np.mean(gray, axis=0)
    threshold = np.mean(col_means)
    mask = col_means < threshold
    first_col = np.argmax(mask)
    last_col = len(mask) - 1 - np.argmax(np.flip(mask))

    # 从原始图像中裁剪除去白色条纹的部分
    result = img[:, first_col: last_col]
    return result

# 将图像两侧切为方块
def split_large_img(img: np.array, patch_size=576, visual_grid=0, flip_right=True):
    row_num = img.shape[0] // patch_size
    rgs_l = []
    rgs_r = []
    for i in range(0, row_num):
        if patch_size * (i + 1) < img.shape[0]:
            rg_l = img[patch_size * i: patch_size * (i + 1), 0: patch_size]
            rg_r = img[patch_size * i: patch_size * (i + 1), img.shape[1] - patch_size: img.shape[1]]
            if flip_right:
                rg_r = cv2.flip(rg_r, 1)
            if visual_grid > 1:
                block_size = patch_size // visual_grid
                # 计算水平和垂直分割的数量
                num_blocks_vertical = patch_size // block_size
                num_blocks_horizontal = patch_size // block_size
                # 绘制水平线
                # for m in range(1, num_blocks_vertical):
                #     y = m * block_size
                #     cv2.line(rg_l, (0, y), (patch_size, y), (0, 255, 0), 1)
                #     cv2.line(rg_r, (0, y), (patch_size, y), (0, 255, 0), 1)
                # 绘制垂直线
                for n in range(1, num_blocks_horizontal):
                    x = n * block_size
                    cv2.line(rg_l, (x, 0), (x, patch_size), (0, 255, 0), 1)
                    cv2.line(rg_r, (x, 0), (x, patch_size), (0, 255, 0), 1)
            rgs_l.append(rg_l)
            rgs_r.append(rg_r)
    return rgs_l, rgs_r

# 特征降维可视化，并与按列分割对比
def cluster_and_visualization(feature: np.array, dimension_reduction='PCA', cluster='KMeans', cluster_num=6, column_num=6, save_path=''):
    H, W, C = feature.shape
    feature = feature.reshape(H * W, C)

    # 聚类
    if cluster == 'OPTICS':   
        clustering = OPTICS()
    else:
        clustering = KMeans(n_clusters=cluster_num, n_init='auto')
    labels = clustering.fit_predict(feature)
    # 计算轮廓系数
    silhouette_kmeans = silhouette_score(feature, labels)
    print(f'Silhouette Coefficient: {silhouette_kmeans}')

    # 降维
    if dimension_reduction == 'TSNE':
        reduction = TSNE(n_components=2)
    else:
        reduction = PCA(n_components=2)
    features_2d = reduction.fit_transform(feature)

    # 可视化聚类结果
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis')
    plt.title('Clustering')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')

    # 分列可视化
    colors = plt.cm.jet(np.linspace(0, 1, column_num))
    color_map = np.zeros((H, W))

    # 分配颜色给的特征
    for i in range(column_num):
        color_map[:, i * W // column_num: (i + 1) * W // column_num] = i
    color_map = color_map.flatten()

    plt.subplot(1, 2, 2)
    for i in range(column_num):
        plt.scatter(features_2d[color_map == i, 0], features_2d[color_map == i, 1], 
                    color=colors[i], label=f'Column group {i+1}', marker='o')
    plt.title('Split by column')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')

    if save_path:
        plt.savefig(save_path)