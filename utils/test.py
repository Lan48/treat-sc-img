import os
import glob
import json
import random
import warnings
import h5py
import argparse
import tempfile
import shutil
import gzip
import tarfile
import anndata as ad
import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np
import pandas as pd
from scipy.io import mmread
#import mygene
import re
#import rarfile
from pathlib import Path
from scipy.spatial import KDTree
from sklearn.decomposition import PCA
def count_average_cells(root_dir: str) -> float:
    root = Path(root_dir)
    h5ad_paths = sorted(root.glob("*.h5ad"))
    if not h5ad_paths:
        raise ValueError(f"目录 {root} 下未找到 *.h5ad 文件")

    cells_per_file = []
    for p in h5ad_paths:
        adata = ad.read_h5ad(p)          # 只读元数据，不读 X
        n_cells = adata.shape[0]    # obs 维度即细胞数
        print(f"样本 {p.name} 中细胞数: {n_cells}")
        cells_per_file.append(n_cells)

    return float(np.mean(cells_per_file))
'''
h5_path = 'project1/spatial_data/all_data/all_data_part2.h5'
# 打开H5文件查看内部结构
with h5py.File(h5_path, 'r') as f:
    # 查看所有样本分组（每个样本是一个group）
    #print("包含的样本分组：", list(f.keys()))
    
    # 选择第一个样本为例（可替换为实际样本名）
    sample_name = list(f.keys())[0]
    sample_group = f[sample_name]
    print(len(f.keys()))
    # 查看该样本的数据集
    print(f"\n样本 {sample_name} 包含的数据集：", list(sample_group.keys()))
    
    # 读取数据（coords_map、expr_map、vocab_index）
    coords = sample_group['coords_map'][:]  # 空间坐标
    expr = sample_group['expr_map'][:]      # 表达矩阵
    vocab_idx = sample_group['vocab_index'][:]  # 词汇表索引
    print(coords[:5][0])
'''
def compute_average_nearest_distance(adata: ad.AnnData) -> float:
    """
    计算每个细胞到其最近邻细胞的距离，并返回这些距离的平均值
    
    参数:
        adata: 包含空间转录组数据的AnnData对象，需有obsm['spatial']坐标信息
        
    返回:
        所有细胞到最近邻细胞的平均距离
    """
    # 检查是否存在空间坐标
    if 'spatial' not in adata.obsm:
        raise ValueError("AnnData对象的obsm中未找到'spatial'坐标信息")
    
    # 获取空间坐标 (n_obs × 2 数组)
    coordinates = adata.obsm['spatial']
    
    # 确保坐标是二维的
    if coordinates.shape[1] != 2:
        raise ValueError(f"空间坐标应为二维，实际为{coordinates.shape[1]}维")
    
    # 检查是否有足够的细胞进行计算
    if len(coordinates) < 2:
        raise ValueError(f"至少需要2个细胞才能计算距离，实际有{len(coordinates)}个")
    
    # 使用KDTree高效查找最近邻
    tree = KDTree(coordinates)
    
    # 查找每个点的最近邻（包括距离和索引）
    # k=2表示返回最近的2个点，因为第一个是点本身
    distances, indices = tree.query(coordinates, k=2)
    
    # 提取每个点到最近邻的距离（排除自身）
    nearest_distances = distances[:, 1]
    
    # 计算平均距离
    average_distance = np.mean(nearest_distances)
    
    # 可选：将每个细胞的最近距离存储在obs中
    adata.obs['nearest_neighbor_distance'] = nearest_distances
    
    return average_distance

from pathlib import Path
import anndata as ad

def report_spots_per_subdir(root_dir: str):
    root = Path(root_dir)
    for subdir in sorted(d for d in root.iterdir() if d.is_dir()):
        h5ad_files = list(subdir.glob('*.h5ad'))
        if not h5ad_files:
            print(f'{subdir.name}: 0 h5ad files')
            continue

        total_spots = 0
        for f in h5ad_files:
            try:
                adata = ad.read_h5ad(f, backed='r')   # 只读元数据，最快
                n_spots = adata.n_obs
                print(f'{subdir.name}/{f.name}: {n_spots} spots')
                total_spots += n_spots
            except Exception as e:
                print(f'{subdir.name}/{f.name}: 读取失败 -> {e}')

        print(f'{subdir.name} 合计: {total_spots} spots\n')

report_spots_per_subdir('project1/spatial_data/down_stream_data/Colorectal cancer histopathologyspatial transcriptomics data from Valdeolivas et al')

h5ad_path = 'project1/spatial_data/down_stream_data/raw_data_DLPFC/DLPFC/151507.h5ad'
adata = ad.read_h5ad(h5ad_path)
#avg = count_average_cells('project1/spatial_data/samples')
#print(f"各文件平均细胞数: {avg:.2f}")
print(adata)
abbr_map = {
    'medulla sinuses': 'MS',
    'medulla cords': 'MC', 
    'cortex': 'C',
    'pericapsular adipose tissue': 'PAT',
    'follicle': 'F',
    'capsule': 'CA',
    'hilum': 'H',
    'medulla vessels': 'MV',
    'subcapsular sinus': 'SCS',
    'trabeculae': 'T'
}

# 2. 将映射应用到新列
#data.obs['manual-anno-abv'] = adata.obs['manual-anno'].map(abbr_map)

#adata.uns.clear()
#print(type(adata.obsm['coords_sample'][0][0]))
#print(adata.obs['manual-anno'].unique)
for cls in adata.obs['Pathologist Annotation'].unique():
    print(cls) 
'''
sc.pl.spatial(
    adata, 
    color=['manual-anno-abv'],
    title='',
    spot_size=1,
    show=False
)

plt.savefig(
    f'k.png', 
    transparent=True,  # 关键参数
    bbox_inches='tight',
    dpi=300
)
'''
#print(compute_average_nearest_dista