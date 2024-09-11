import torch
import os
import torch.utils.checkpoint
from typing import Tuple
import cv2
from torchvision import transforms as T
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from utils import cluster_and_visualization, ResNet_Encoder

def analyse(img_path: str=''):
    model = ResNet_Encoder().eval()
    # 如果路径是目录，分析大图的一整列
    if os.path.isdir(img_path):
        imgs = [os.path.join(img_path, f) for f in os.listdir(img_path) if f.endswith('.jpg') or f.endswith('.png')]
        features = []
        for img in imgs:
            img_cv = cv2.imread(img)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_cv)
            features.append(model(img_pil))
        feature = torch.concat(features, dim=2)
    # 若路径不是目录，则分析patch
    else:        
        img_cv = cv2.imread(img_path)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_cv)
        feature = model(img_pil)

    print('feature shape:', feature.shape)
    B, C, H, W = feature.shape
    feature = feature.permute((2, 3, 0, 1)).reshape(H, W, C).detach().numpy()

    # 特征聚类、降维可视化
    # cluster_and_visualization(feature=feature, save_path='visualization_result.png', column_num=9)

    # 计算每列特征的均值和标准差
    mean = np.mean(feature, axis=0, keepdims=True)
    # std = np.std(feature, axis=0, keepdims=True)
    # std[std == 0] = 1e-8

    # 初始化存储距离和相似度的数组，存储每个位置和每一列特征的相似度
    euclidean_distances = np.zeros((H, W, W))
    cosine_similarities = np.zeros((H, W, W))

    for i in range(W):
        # 计算欧式距离
        euclidean_distances[:, :, i] = np.linalg.norm(feature - mean[:, i, :], axis=-1)

        # 计算余弦相似度
        dot_products = np.sum(feature * mean[:, i, :], axis=-1)
        norm_features = np.linalg.norm(feature, axis=-1)
        norm_feature_set = np.linalg.norm(mean[:, i, :], axis=-1)
        cosine_similarities[:, :, i] = dot_products / (norm_features * norm_feature_set)

    # 设置 NumPy 的打印选项以显示完整的数组
    np.set_printoptions(threshold=np.inf)

    # 从最高相似度索引可以看出，最大相似度索引基本是当前列或在当前列附近,符合预期
    # print("欧式距离:", np.argmin(euclidean_distances, axis=-1))
    # print("余弦相似度:", np.argmax(cosine_similarities, axis=-1))

    # 对于1、2、3区域，观察当前图像，特征列索引中，1在12左右，2在18左右，3在32左右，可以计算三列之间的相似度矩阵
    select = [10, 17, 24]
    # 欧式距离
    print(np.mean(euclidean_distances[:, :, :], axis=0)[select, :][:, select])
    # 余弦相似度
    print(np.mean(cosine_similarities[:, :, :], axis=0)[select, :][:, select])


if __name__ == '__main__':
    img_dir = 'test_imgs/I911102052/l'
    analyse(img_dir)