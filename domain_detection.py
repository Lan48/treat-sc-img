from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import anndata
import os
import torch
import json
import scipy.sparse
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, accuracy_score


def evaluate_domain_detection(adata, gt, pred):
        """
        评估空间转录组数据的域检测结果
        
        参数:
            adata: AnnData对象，空间转录组数据
            gt: 字符串，真实标签在adata.obs中的列名
            pred: 字符串，预测标签在adata.obs中的列名
            
        返回:
            tuple: 包含两个浮点数 (调整兰德指数ARI, 准确率ACC)
        """
        # 检查输入的列名是否存在
        if gt not in adata.obs.columns:
            raise ValueError(f"真实标签列 '{gt}' 不在adata.obs中")
        if pred not in adata.obs.columns:
            raise ValueError(f"预测标签列 '{pred}' 不在adata.obs中")
        
        # 提取真实标签和预测标签
        gt_labels = adata.obs[gt]
        pred_labels = adata.obs[pred]
        
        # 处理可能的非数值标签（转换为整数编码）
        if not pd.api.types.is_numeric_dtype(gt_labels):
            gt_labels, _ = pd.factorize(gt_labels)
        if not pd.api.types.is_numeric_dtype(pred_labels):
            pred_labels, _ = pd.factorize(pred_labels)
        
        # 计算调整兰德指数
        ari_score = adjusted_rand_score(gt_labels, pred_labels)
        
        # 计算准确率
        # 注意：准确率要求预测标签和真实标签的类别编号直接对应
        # 如果类别编号不直接对应，这个指标可能不准确
        acc_score = accuracy_score(gt_labels, pred_labels)
        
        return ari_score, acc_score

def domain_detection_gmm(
    adata, n_components=7, covariance_type='full', max_iter=500,
    init_params='k-means++', use_rep='X_pca', n_pca_components=15, random_state=42
):
    # 数据预处理
    if use_rep in adata.obsm_keys():
        X = adata.obsm[use_rep]
    else:
        sc.tl.pca(adata, n_comps=n_pca_components)
        X = adata.obsm['X_pca'][:, :n_pca_components]
    
    # 高维自动降级协方差类型
    if X.shape[1] > 50 and covariance_type == 'full':
        covariance_type = 'diag'
    
    # 初始化并训练GMM
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        max_iter=max_iter,
        init_params=init_params,
        random_state=random_state
    )
    gmm.fit(X)
    labels = gmm.predict(X)
        
    # 保存结果
    adata.obs['GMM_domains'] = [f"Domain_{i+1}" for i in labels]
    return adata

def domain_detection_knn(
    adata,
    n_clusters,
    use_rep='X_pca',
    is_pca=False,  # 新增参数：控制是否执行PCA
    n_pca_components=None,
    random_state=42
):
    """
    空间域检测：基于KNN图 + K-means聚类（支持PCA预处理）
    
    参数说明
    ----------
    adata : AnnData
        空间转录组数据集
    n_clusters : int
        期望的空间域数量
    use_rep : str, default 'X_pca'
        用于聚类的表征：
        - 若为'X'，使用adata.X基因表达矩阵
        - 否则使用adata.obsm[use_rep]
    is_pca : bool, default False
        是否对输入表征进行PCA降维
    n_pca_components : int or None, default None
        PCA降维的组件数（None时自动计算）
    random_state : int, default 42
        随机种子
    """
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    
    # ========== 数据预处理 ==========
    # 获取输入数据
    if use_rep == 'X':
        X = adata.X  # 原始基因表达矩阵
    elif use_rep in adata.obsm:
        X = adata.obsm[use_rep]  # 预存表征
    else:
        raise ValueError(f"'{use_rep}' not found in adata.obsm")
    
    # 执行PCA降维（若启用）
    if is_pca:
        # 自动确定PCA组件数
        if n_pca_components is None:
            n_pca_components = min(X.shape[1] - 1, 50)
        
        # 执行PCA降维
        pca = PCA(
            n_components=n_pca_components,
            random_state=random_state
        )
        X = pca.fit_transform(X)
        print(f"Applied PCA: reduced to {n_pca_components} components")
    
    # ========== K-means聚类 ==========
    km = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init='auto'  # 避免未来版本警告
    )
    labels = km.fit_predict(X)
    
    # ========== 保存结果 ==========
    adata.obs['KNN_domains'] = [f"Domain_{i+1}" for i in labels]
    
    # 可选：存储PCA结果（若执行了降维）
    if is_pca:
        adata.obsm['X_pca_processed'] = X
    
    return adata

if __name__ == '__main__':
    
    h5ad_path = 'project1/spatial_data/raw_data_DLPFC/DLPFC/151507.h5ad'
    adata = anndata.read_h5ad(h5ad_path)
    adata = domain_detection_gmm(adata,use_rep='X_emb')
    adata = domain_detection_knn(adata,use_rep='X_emb',is_pca=False,n_clusters = 7)
    print(evaluate_domain_detection(adata, gt='sce.layer_guess', pred='GMM_domains')[0])
    print(evaluate_domain_detection(adata, gt='sce.layer_guess', pred='KNN_domains')[0])
    sc.pl.spatial(adata, color=['KNN_domains'],title= '',spot_size=1, save=f'P70_dynamic_mlp.png')
