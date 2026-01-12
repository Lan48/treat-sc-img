import random
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
from model.MAE import MAEModel
from utils.utils import get_least_used_gpu
from typing import List, Dict, Union
import anndata as ad
import warnings
from anndata._warnings import ImplicitModificationWarning
from scipy.sparse import issparse,csr_matrix
import re

warnings.filterwarnings("ignore", category=ImplicitModificationWarning)



def bin_matrix(matrix, bins):
    if issparse(matrix):
        matrix_dense = matrix.toarray()
    else:
        matrix_dense = matrix.copy()
    
    n_cells = matrix_dense.shape[0]
    for i in range(n_cells):
        # 获取当前细胞的所有基因表达值
        cell_data = matrix_dense[i, :]
        
        # 计算分位数边界
        quantiles = np.quantile(cell_data, np.linspace(0, 1, bins + 1))
        
        # 处理所有分位数相同的情况（所有基因表达值相同）
        if np.all(quantiles == quantiles[0]):
            matrix_dense[i, :] = 0
        else:
            # 添加随机噪声到分位数边界以避免完全相等的边界
            noise = np.random.uniform(-1e-10, 1e-10, size=quantiles.shape)
            quantiles = quantiles + noise
            
            # 确保分位数边界是单调递增的
            quantiles = np.sort(quantiles)
            
            # 使用分位数进行分箱
            binned_data = np.digitize(cell_data, quantiles[1:-1], right=False)
            matrix_dense[i, :] = binned_data
    
    # 返回与原始矩阵相同类型的矩阵
    if issparse(matrix):
        return csr_matrix(matrix_dense) if issparse(matrix) else matrix_dense.astype(np.float32)
    else:
        return matrix_dense.astype(np.float32)

def scale_spatial_coordinates(adata, target_range=(0, 1)):
    """
    将 adata.obsm['spatial'] 中的坐标缩放到目标范围；
    若 adata.obsm 中已存在 'ori_spatial'，则视为已缩放过，直接跳过。

    参数
    ----
    adata : AnnData
    target_range : tuple, 默认 (0, 1)

    返回
    ----
    adata : AnnData（原位修改，同时返回方便链式调用）
    """
    # 如果已经保存过原始坐标，则不再重复缩放
    if 'ori_spatial' in adata.obsm:
        print("检测到 'ori_spatial'，空间坐标已缩放过，跳过本次操作。")
        return adata

    spatial = adata.obsm['spatial'].copy()
    min_vals = spatial.min(axis=0)
    max_vals = spatial.max(axis=0)

    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1  # 防止除零

    target_min, target_max = target_range
    scaled_spatial = (spatial - min_vals) / ranges * (target_max - target_min) + target_min

    adata.obsm['ori_spatial'] = spatial        # 保存原始坐标
    adata.obsm['spatial'] = scaled_spatial     # 更新缩放后坐标

    print(f"空间坐标已从范围 {min_vals} - {max_vals} 缩放到 {target_min} - {target_max}")
    return adata

'''
def preprocess(ad, vocab, config, hv_list=None):
    """
    预处理adata对象：基因过滤并分割成小切片
    返回adata列表，每个元素为一个小切片对象,加权随机选择高变基因
    
    参数：
    ad - AnnData对象，包含空间转录组数据
    vocab - 基因字典，包含基因名到ID的映射
    config - 配置字典，包含：
        'h' - 小切片高度（网格行数）
        'w' - 小切片宽度（网格列数）
        'depth' - 每切片选择基因数
        'pad_token' - 填充标记（默认为"<pad>"）
    hv_list - 高变基因列表，如果提供则使用这些基因而非随机选择
    
    返回：
    adata_list - 分割后的小切片AnnData对象列表
    """
    ad = scale_spatial_coordinates(ad, target_range=(0, 100))
    normalize = config.get('normalize', 0)
    if normalize > 0:
        if _is_log_transformed(ad.X):
            ad.X = np.expm1(ad.X) 
        sc.pp.normalize_total(ad, target_sum=normalize)
        log1p_with_backup(ad)
    
    # 1. 初始基因过滤（保留vocab中存在的基因）
    valid_genes = [gene for gene in ad.var_names if gene in vocab]
    ad = ad[:, valid_genes].copy()
       
    spatial_coords = ad.obsm['spatial']
    
    # 3. 计算空间网格划分
    min_x, min_y = np.min(spatial_coords, axis=0)
    max_x, max_y = np.max(spatial_coords, axis=0)
    grid_x = np.arange(min_x, max_x, config['w'])
    grid_y = np.arange(min_y, max_y, config['h'])
    
    adata_list = []
    pad_id = vocab.get(config.get('pad_token', '<pad>'), 0)
    
    # 初始化切片计数器
    slice_counter = 0
    
    # 4. 遍历网格创建小切片
    for i in range(len(grid_y)):
        for j in range(len(grid_x)):
            # 计算当前切片边界
            y_start, y_end = grid_y[i], grid_y[i] + config['h']
            x_start, x_end = grid_x[j], grid_x[j] + config['w']
            
            # 选择当前切片内的细胞
            in_slice = (
                (spatial_coords[:, 0] >= x_start) & 
                (spatial_coords[:, 0] < x_end) &
                (spatial_coords[:, 1] >= y_start) & 
                (spatial_coords[:, 1] < y_end)
            )
            
            if not np.any(in_slice):
                continue  # 跳过无细胞的切片
                
            slice_ad = ad[in_slice, :].copy()
            
            # 6. 添加坐标元数据
            abs_coords = spatial_coords[in_slice]
            
            # 计算网格化相对坐标（映射到h*w网格）
            rel_x = abs_coords[:, 0] - x_start
            rel_y = abs_coords[:, 1] - y_start
            
            # 映射到离散网格（整数索引）
            grid_x_idx = np.floor(rel_x * config['w'] / config['w']).astype(int)
            grid_y_idx = np.floor(rel_y * config['h'] / config['h']).astype(int)
            
            # 转换为float64格式存储
            rel_coords = np.column_stack([
                grid_x_idx.astype(np.float64),
                grid_y_idx.astype(np.float64)
            ])
            
            # 添加切片ID信息
            slice_ad.obs = slice_ad.obs.assign(
                original_index=slice_ad.obs.index,
                x_abs=abs_coords[:, 0],
                y_abs=abs_coords[:, 1],
                x_rel=rel_coords[:, 0],
                y_rel=rel_coords[:, 1],
                slice_id=slice_counter  # 添加切片ID
            )
            
            # 7. 基因选择与填充
            X = slice_ad.X.toarray() if scipy.sparse.issparse(slice_ad.X) else slice_ad.X
            
            # 根据hv_list参数选择不同的基因选择策略[2](@ref)
            if hv_list is not None:
                # 使用指定的高变基因列表[2](@ref)
                #print(f"Using provided HVG list with {len(hv_list)} genes")
                
                # 获取当前切片中存在的hv_list基因[7](@ref)
                available_hv_genes = [gene for gene in hv_list if gene in slice_ad.var_names]
                #print(f"Available HVG in current slice: {len(available_hv_genes)} genes")
                
                if len(available_hv_genes) == 0:
                    print("Warning: No HVG genes available in current slice, using fallback strategy")
                    # 回退到原始策略：计算基因方差权重
                    gene_vars = np.var(X, axis=0)
                    weights = (gene_vars + 1e-5) / np.sum(gene_vars + 1e-5)
                    
                    n_genes = min(config['depth'], len(slice_ad.var))
                    selected_idx = np.random.choice(
                        len(slice_ad.var), 
                        size=n_genes, 
                        replace=False, 
                        p=weights
                    )
                    selected_genes = slice_ad.var_names[selected_idx].tolist()
                else:
                    # 选择可用的高变基因，最多选择config['depth']个[7](@ref)
                    n_select = min(config['depth'], len(available_hv_genes))
                    selected_genes = available_hv_genes[:n_select]
                    
                    # 获取这些基因在slice_ad中的索引[7](@ref)
                    selected_idx = [slice_ad.var_names.get_loc(gene) for gene in selected_genes 
                                  if gene in slice_ad.var_names]
            else:
                # 原始策略：加权随机选择基因[2](@ref)
                # 计算基因方差权重
                gene_vars = np.var(X, axis=0)
                weights = (gene_vars + 1e-5) / np.sum(gene_vars + 1e-5)
                
                n_genes = min(config['depth'], len(slice_ad.var))
                selected_idx = np.random.choice(
                    len(slice_ad.var), 
                    size=n_genes, 
                    replace=False, 
                    p=weights
                )
                selected_genes = slice_ad.var_names[selected_idx].tolist()
            
            # 创建新表达矩阵
            new_X = np.zeros((len(slice_ad), config['depth']))
            
            if len(selected_idx) > 0:
                # 确保不超出矩阵范围[7](@ref)
                n_genes_to_use = min(len(selected_idx), config['depth'])
                new_X[:, :n_genes_to_use] = X[:, selected_idx[:n_genes_to_use]]
            
            # 创建基因ID映射
            gene_ids = []
            for k in range(min(len(selected_idx), config['depth'])):
                gene_name = slice_ad.var_names[selected_idx[k]]
                gene_ids.append(vocab.get(gene_name, pad_id))
            
            # 填充剩余位置[7](@ref)
            if len(gene_ids) < config['depth']:
                gene_ids += [pad_id] * (config['depth'] - len(gene_ids))
            
            # 8. 构建新AnnData对象
            new_ad = sc.AnnData(
                X=new_X,
                obs=slice_ad.obs,  # 包含slice_id信息
                var=pd.DataFrame({'gene_ids': gene_ids})
            )
            
            # 添加额外的切片信息作为uns属性
            new_ad.uns['slice_info'] = {
                'slice_id': slice_counter,
                'grid_x': j,
                'grid_y': i,
                'x_range': [x_start, x_end],
                'y_range': [y_start, y_end]
            }
            
            adata_list.append(new_ad)
            
            # 增加切片计数器
            slice_counter += 1
    
    print(f"Created {slice_counter} slices in total")
    return adata_list
'''
'''
def preprocess(ad, vocab, config):
    """
    预处理adata对象：基因过滤、log1p处理并分割成小切片
    返回adata列表，每个元素为一个小切片对象，选择高变基因
    
    参数：
    ad - AnnData对象，包含空间转录组数据
    vocab - 基因字典，包含基因名到ID的映射
    config - 配置字典，包含：
        'h' - 小切片高度（网格行数）
        'w' - 小切片宽度（网格列数）
        'depth' - 每切片选择基因数
        'pad_token' - 填充标记（默认为"<pad>"）
    
    返回：
    adata_list - 分割后的小切片AnnData对象列表
    """
    # 1. 初始基因过滤（保留vocab中存在的基因）
    valid_genes = [gene for gene in ad.var_names if gene in vocab]
    ad = ad[:, valid_genes].copy()
    
    # 2. 基因表达值log1p变换
    #sc.pp.log1p(ad)
    if config.get('is_bin', False):
        bins = config.get('bins', 50)
        ad.X = bin_matrix(ad.X, bins)
    # 3. 直接使用原始空间坐标
    spatial_coords = ad.obsm['spatial']
    
    # 4. 计算空间网格划分
    min_x, min_y = np.min(spatial_coords, axis=0)
    max_x, max_y = np.max(spatial_coords, axis=0)
    grid_x = np.arange(min_x, max_x, config['w'])
    grid_y = np.arange(min_y, max_y, config['h'])
    
    adata_list = []
    pad_id = vocab.get(config.get('pad_token', '<pad>'), 0)
    
    # 5. 遍历网格创建小切片
    for i in range(len(grid_y)):
        for j in range(len(grid_x)):
            # 计算当前切片边界
            y_start, y_end = grid_y[i], grid_y[i] + config['h']
            x_start, x_end = grid_x[j], grid_x[j] + config['w']
            
            # 选择当前切片内的细胞
            in_slice = (
                (spatial_coords[:, 0] >= x_start) & 
                (spatial_coords[:, 0] < x_end) &
                (spatial_coords[:, 1] >= y_start) & 
                (spatial_coords[:, 1] < y_end)
            )
            
            if not np.any(in_slice):
                continue  # 跳过无细胞的切片
                
            slice_ad = ad[in_slice, :].copy()
            
            # 6. 添加坐标元数据
            abs_coords = spatial_coords[in_slice]
            
            # 计算网格化相对坐标（映射到h*w网格）
            rel_x = abs_coords[:, 0] - x_start
            rel_y = abs_coords[:, 1] - y_start
            
            # 映射到离散网格（整数索引）
            grid_x_idx = np.floor(rel_x * config['w'] / config['w']).astype(int)
            grid_y_idx = np.floor(rel_y * config['h'] / config['h']).astype(int)
            
            # 转换为float64格式存储
            rel_coords = np.column_stack([
                grid_x_idx.astype(np.float64),
                grid_y_idx.astype(np.float64)
            ])
            
            slice_ad.obs = slice_ad.obs.assign(
                original_index=slice_ad.obs.index,
                x_abs=abs_coords[:, 0],
                y_abs=abs_coords[:, 1],
                x_rel=rel_coords[:, 0],
                y_rel=rel_coords[:, 1]
            )
            
            # 7. 高变基因选择与填充（核心修改）
            X = slice_ad.X.toarray() if scipy.sparse.issparse(slice_ad.X) else slice_ad.X
            
            # 计算基因方差并排序
            gene_vars = np.var(X, axis=0)
            sorted_idx = np.argsort(gene_vars)[::-1]  # 方差降序排序
            
            # 选择top高变基因
            n_genes = min(config['depth'], len(slice_ad.var))
            selected_idx = sorted_idx[:n_genes]  # 取方差最大的前n_genes个基因
            
            # 创建新表达矩阵
            new_X = np.zeros((len(slice_ad), config['depth']))
            new_X[:, :n_genes] = X[:, selected_idx]
            
            # 创建基因ID映射
            gene_ids = [vocab[slice_ad.var_names[i]] for i in selected_idx]
            if len(gene_ids) < config['depth']:
                gene_ids += [pad_id] * (config['depth'] - len(gene_ids))
            
            # 8. 构建新AnnData对象
            new_ad = sc.AnnData(
                X=new_X,
                obs=slice_ad.obs,
                var=pd.DataFrame({'gene_ids': gene_ids})
            )
            adata_list.append(new_ad)
    
    return adata_list
'''

def preprocess(ad, vocab, config, hv_list=None, gene_selection_method="hvg", hvg_method="seurat"):
    """
    预处理adata对象：基因过滤并分割成小切片
    返回adata列表，每个元素为一个小切片对象，支持多种基因选择策略

    参数：
    ad - AnnData对象，包含空间转录组数据
    vocab - 基因字典，包含基因名到ID的映射
    config - 配置字典，包含：
        'h' - 小切片高度（网格行数）
        'w' - 小切片宽度（网格列数）
        'depth' - 每切片选择基因数
        'pad_token' - 填充标记（默认为"<pad>"）
    hv_list - 高变基因列表，如果提供则优先使用这些基因
    gene_selection_method - 基因选择策略，可选：
        "hvg": 使用高变基因
        "weighted_random": 加权随机选择（基于基因方差，默认）
        "uniform_random": 直接均匀随机选择
    hvg_method - 当gene_selection_method="hvg"且未提供hv_list时，用于计算高变基因的方法，可选：
        "seurat": 基于基因离散度（Scanpy默认方法）[1,3,7](@ref)
        "seurat_v3": 基于基因归一化方差（Seurat v3方法）[1,7](@ref)
        "pearson_residuals": 基于皮尔森残差[1,3,7](@ref)

    返回：
    adata_list - 分割后的小切片AnnData对象列表
    """
    ad = scale_spatial_coordinates(ad, target_range=(0, 100))
    normalize = config.get('normalize', 0)
    if normalize > 0:
        if _is_log_transformed(ad.X):
            ad.X = np.expm1(ad.X) 
        sc.pp.normalize_total(ad, target_sum=normalize)
        log1p_with_backup(ad)
    
    # 1. 初始基因过滤（保留vocab中存在的基因）
    valid_genes = [gene for gene in ad.var_names if gene in vocab]
    ad = ad[:, valid_genes].copy()
       
    spatial_coords = ad.obsm['spatial']
    
    # 2. 计算空间网格划分
    min_x, min_y = np.min(spatial_coords, axis=0)
    max_x, max_y = np.max(spatial_coords, axis=0)
    grid_x = np.arange(min_x, max_x, config['w'])
    grid_y = np.arange(min_y, max_y, config['h'])
    
    adata_list = []
    pad_id = vocab.get(config.get('pad_token', '<pad>'), 0)
    slice_counter = 0  # 切片计数器
    
    # 3. 遍历网格创建小切片
    for i in range(len(grid_y)):
        for j in range(len(grid_x)):
            # 计算当前切片边界
            y_start, y_end = grid_y[i], grid_y[i] + config['h']
            x_start, x_end = grid_x[j], grid_x[j] + config['w']
            
            # 选择当前切片内的细胞
            in_slice = (
                (spatial_coords[:, 0] >= x_start) & 
                (spatial_coords[:, 0] < x_end) &
                (spatial_coords[:, 1] >= y_start) & 
                (spatial_coords[:, 1] < y_end)
            )
            
            if not np.any(in_slice):
                continue  # 跳过无细胞的切片
                
            slice_ad = ad[in_slice, :].copy()
            X = slice_ad.X.toarray() if scipy.sparse.issparse(slice_ad.X) else slice_ad.X
            if np.any(np.isinf(X)):
                print("Warning: Found infinite values in ad.X after preprocessing.")
            # 4. 基因选择策略
            n_genes_to_select = min(config['depth'], len(slice_ad.var))
            selected_genes = []
            selected_idx = []
            
            if gene_selection_method == "hvg":
                # 策略1: 使用高变基因
                if hv_list is not None:
                    # 使用提供的高变基因列表
                    available_hv_genes = [gene for gene in hv_list if gene in slice_ad.var_names]
                else:
                    # 自动计算高变基因[1,3](@ref)
                    # 创建当前切片的临时AnnData对象用于计算高变基因
                    temp_ad = sc.AnnData(X=slice_ad.X,
                                         var=slice_ad.var, obs=slice_ad.obs)
                    
                    # 根据指定方法计算高变基因
                    if hvg_method == "seurat":
                        # 基于基因离散度的方法[1,3,7](@ref)
                        sc.pp.highly_variable_genes(
                            temp_ad, 
                            flavor="seurat", 
                            n_top_genes=min(2000, len(temp_ad.var)),
                            inplace=True
                        )
                    elif hvg_method == "seurat_v3":
                        # 基于基因归一化方差的方法[1,7](@ref)
                        sc.pp.highly_variable_genes(
                            temp_ad,
                            flavor="seurat_v3",
                            n_top_genes=min(2000, len(temp_ad.var)),
                            inplace=True
                        )
                    elif hvg_method == "pearson_residuals":
                        # 基于皮尔森残差的方法[1,3](@ref)
                        sc.experimental.pp.highly_variable_genes(
                            temp_ad,
                            flavor="pearson_residuals", 
                            n_top_genes=min(2000, len(temp_ad.var)),
                            inplace=True
                        )
                    else:
                        raise ValueError("hvg_method must be one of: 'seurat', 'seurat_v3', 'pearson_residuals'")
                    
                    # 获取计算出的高变基因
                    hv_genes = temp_ad.var_names[temp_ad.var['highly_variable']].tolist()
                    available_hv_genes = [gene for gene in hv_genes if gene in slice_ad.var_names]
                
                # 选择可用的高变基因
                if len(available_hv_genes) == 0:
                    print(f"Warning: No HVG genes available in slice {slice_counter}, using weighted random fallback")
                    # 回退到加权随机选择
                    gene_vars = np.var(X, axis=0)
                    if np.sum(gene_vars) == 0:  # 防止全零方差
                        gene_vars = np.ones_like(gene_vars)
                    weights = (gene_vars + 1e-5) / np.sum(gene_vars + 1e-5)
                    selected_idx = np.random.choice(
                        len(slice_ad.var), size=n_genes_to_select, replace=False, p=weights
                    )
                else:
                    # 从可用高变基因中选择
                    n_select = min(n_genes_to_select, len(available_hv_genes))
                    selected_genes = available_hv_genes[:n_select]
                    selected_idx = [slice_ad.var_names.get_loc(gene) for gene in selected_genes 
                                  if gene in slice_ad.var_names]
            
            elif gene_selection_method == "weighted_random":
                # 策略2: 加权随机选择（基于基因方差）
                gene_vars = np.var(X, axis=0)
                if np.sum(gene_vars) == 0:  # 防止全零方差
                    gene_vars = np.ones_like(gene_vars)
                weights = (gene_vars + 1e-5) / np.sum(gene_vars + 1e-5)
                selected_idx = np.random.choice(
                    len(slice_ad.var), size=n_genes_to_select, replace=False, p=weights
                )
            
            elif gene_selection_method == "uniform_random":
                # 策略3: 直接均匀随机选择
                selected_idx = np.random.choice(
                    len(slice_ad.var), size=n_genes_to_select, replace=False
                )
            
            else:
                raise ValueError("gene_selection_method must be one of: 'hvg', 'weighted_random', 'uniform_random'")
            
            # 如果selected_idx为空（如回退情况），确保有基因被选择
            if len(selected_idx) == 0:
                selected_idx = np.random.choice(len(slice_ad.var), size=min(1, len(slice_ad.var)), replace=False)
            
            # 获取基因名（如果未在hvg策略中设置）
            if not selected_genes:
                selected_genes = slice_ad.var_names[selected_idx].tolist()
            
            # 5. 创建新表达矩阵并填充
            new_X = np.zeros((len(slice_ad), config['depth']))
            n_genes_used = min(len(selected_idx), config['depth'])
            if n_genes_used > 0:
                new_X[:, :n_genes_used] = X[:, selected_idx[:n_genes_used]]
            
            # 映射基因ID
            gene_ids = []
            for k in range(n_genes_used):
                gene_name = selected_genes[k] if k < len(selected_genes) else slice_ad.var_names[selected_idx[k]]
                gene_ids.append(vocab.get(gene_name, pad_id))
            
            # 填充剩余位置
            if len(gene_ids) < config['depth']:
                gene_ids += [pad_id] * (config['depth'] - len(gene_ids))
            
            # 6. 添加坐标信息（与原代码相同）
            abs_coords = spatial_coords[in_slice]
            rel_x = abs_coords[:, 0] - x_start
            rel_y = abs_coords[:, 1] - y_start
            grid_x_idx = np.floor(rel_x * config['w'] / config['w']).astype(int)
            grid_y_idx = np.floor(rel_y * config['h'] / config['h']).astype(int)
            rel_coords = np.column_stack([grid_x_idx.astype(np.float64), grid_y_idx.astype(np.float64)])
            
            slice_ad.obs = slice_ad.obs.assign(
                original_index=slice_ad.obs.index,
                x_abs=abs_coords[:, 0],
                y_abs=abs_coords[:, 1],
                x_rel=rel_coords[:, 0],
                y_rel=rel_coords[:, 1],
                slice_id=slice_counter
            )
            
            # 7. 构建新AnnData对象
            new_ad = sc.AnnData(
                X=new_X,
                obs=slice_ad.obs,
                var=pd.DataFrame({'gene_ids': gene_ids})
            )
            
            # 记录使用的基因选择方法
            used_method = gene_selection_method
            if gene_selection_method == "hvg" and hv_list is None:
                used_method = f"hvg_{hvg_method}"
            
            new_ad.uns['slice_info'] = {
                'slice_id': slice_counter,
                'grid_x': j,
                'grid_y': i,
                'x_range': [x_start, x_end],
                'y_range': [y_start, y_end],
                'gene_selection_method': used_method,
                'n_genes_selected': n_genes_used
            }
            adata_list.append(new_ad)
            slice_counter += 1
    
    print(f"Created {slice_counter} slices using gene selection method: {gene_selection_method}")
    return adata_list

def map_slices_to_original_extended(original_ad, adata_list):
    """
    将预处理后切片中的新obs属性和obsm属性映射回原始AnnData对象（包括更新原始已有的obsm属性）。
    
    参数:
        original_ad: 原始的AnnData对象。
        adata_list: 预处理后返回的AnnData对象列表（每个对象是一个切片）。
        
    返回:
        original_ad: 更新后的原始AnnData对象，添加/更新了切片中的obs和obsm属性。
    """
    # 处理obs属性（逻辑不变，仅添加新属性或更新已有属性）
    new_obs_attributes = ['x_rel', 'y_rel','slice_id']  # 可根据实际扩展
    
    for attr in new_obs_attributes:
        if attr not in original_ad.obs.columns:
            original_ad.obs[attr] = np.nan  # 原始没有则初始化
    
    # 收集所有切片中存在的obsm键（包括原始数据已有的，取消过滤）
    all_obsm_keys = set()
    for slice_ad in adata_list:
        all_obsm_keys.update(slice_ad.obsm_keys())  # 直接收集所有切片中的obsm键
    
    # 初始化/检查obsm属性（原始没有则初始化，原始已有则检查形状一致性）
    for key in all_obsm_keys:
        # 找到一个包含该key的切片作为形状参考
        ref_slice = next((s for s in adata_list if key in s.obsm), None)
        if ref_slice is None:
            continue  # 理论上不会触发，因为all_obsm_keys来自切片
        
        ref_shape = ref_slice.obsm[key].shape[1:]  # 除细胞数外的维度（如坐标的2维）
        
        if key not in original_ad.obsm:
            # 原始数据没有该obsm键，初始化与切片同形状的空数组（填充NaN）
            obsm_shape = (original_ad.n_obs,) + ref_shape
            original_ad.obsm[key] = np.full(obsm_shape, np.nan)
        else:
            # 原始数据已有该obsm键，检查形状是否匹配（避免维度不兼容）
            original_shape = original_ad.obsm[key].shape[1:]
            if original_shape != ref_shape:
                raise ValueError(
                    f"obsm键'{key}'形状不匹配：原始数据为{original_shape}，切片为{ref_shape}"
                )
    
    # 遍历所有切片，更新原始数据的obs和obsm属性
    for slice_ad in adata_list:
        original_indices = slice_ad.obs['original_index']  # 切片细胞在原始数据中的索引
        #print(f"slice_adnan数量：{np.isnan(slice_ad.obsm[config['emb_name']]).sum()}")

        for idx_in_slice, original_idx in enumerate(original_indices):
            if original_idx not in original_ad.obs_names:
                print(f"Warning: 切片中的细胞{original_idx}在原始数据中不存在，跳过")
                continue
            
            # 更新obs属性
            for attr in new_obs_attributes:
                if attr in slice_ad.obs.columns:
                    original_ad.obs.loc[original_idx, attr] = slice_ad.obs.iloc[idx_in_slice][attr]
            
            # 更新obsm属性（包括原始已有的）
            for key in all_obsm_keys:
                if key in slice_ad.obsm:
                    # 找到原始数据中该细胞的位置索引
                    original_pos = original_ad.obs_names.get_loc(original_idx)
                    # 赋值（利用切片数据覆盖原始数据对应位置）
                    original_ad.obsm[key][original_pos] = slice_ad.obsm[key][idx_in_slice]
    return original_ad

def add_emb(
    model, 
    adata_list: List[ad.AnnData], 
    vocab: Dict, 
    config: Dict,
    device: Union[str, torch.device] = None,
) -> List[ad.AnnData]:
    """
    为每个空间转录组样本计算细胞嵌入并添加到样本自身的AnnData对象中
    新增功能：按cell_batch_size拆分细胞，分批处理以减少GPU内存占用
    """
    cell_batch_size = config.get('batch_size', 16)
    mode = config['mode']
    k = config['pad_size']
    
    # 参数校验
    if mode not in ['spatial', 'single', 'padding']:
        raise ValueError("mode必须为'spatial', 'single'或'k'")
    if mode == 'padding' and (k is None or not 0 <= k <= 1):
        raise ValueError("mode='k'时需提供0-1之间的k值")

    # 设备设置
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    model = model.to(device)
    model.eval()

    for sample_adata in adata_list:
        if mode == 'spatial':
            # 原始空间坐标处理逻辑（保持不变，若内存不足可类似拆分）
            h, w = config['h'], config['w']
            c = sample_adata.n_vars
            expr_matrix = np.zeros((c, h, w), dtype=np.float32)
            
            x_coords = sample_adata.obs['x_rel'].astype(int).values
            y_coords = sample_adata.obs['y_rel'].astype(int).values
            valid_idx = (x_coords >= 0) & (x_coords < w) & (y_coords >= 0) & (y_coords < h)
            
            x_valid = x_coords[valid_idx]
            y_valid = y_coords[valid_idx]
            
            for i, (x, y) in enumerate(zip(x_coords, y_coords)):
                if 0 <= x < w and 0 <= y < h:
                    expr = sample_adata.X[i, :].flatten()  
                    expr_matrix[:, y, x] = expr

            expr_tensor = torch.tensor(expr_matrix, device=device).unsqueeze(0)
            gene_ids = sample_adata.var['gene_ids'].map(lambda x: vocab.get(x, 0)).values
            gene_ids_tensor = torch.tensor(gene_ids, device=device).unsqueeze(0)
            
            with torch.no_grad():
                _, _, _, enc_output = model(expression=expr_tensor, gene_ids=gene_ids_tensor)
                emb = enc_output

            cell_emb = emb[0, :, y_valid, x_valid].cpu().numpy().T
            
            full_emb_matrix = np.full((len(sample_adata), cell_emb.shape[1]), np.nan)
            full_emb_matrix[valid_idx] = cell_emb
            sample_adata.obsm[config['emb_name']] = full_emb_matrix

        else:
            # 处理single和k模式（核心修改：按cell_batch_size拆分细胞）
            h, w = config['h'], config['w']
            c = sample_adata.n_vars
            n_cells = sample_adata.n_obs  # 总细胞数
            gene_ids = sample_adata.var['gene_ids'].map(lambda x: vocab.get(x, 0)).values
            gene_ids_tensor = torch.tensor(gene_ids, device=device).unsqueeze(0)  # [1, c]
            
            # 初始化存储所有细胞嵌入的数组（在GPU上预分配）
            all_cell_emb = torch.empty((n_cells, config['en_dim']), device=device, dtype=torch.float32)
            
            # 按批次处理细胞（核心修改点）
            for cell_start in range(0, n_cells, cell_batch_size):
                # 计算当前批次的细胞范围（避免超出总细胞数）
                cell_end = min(cell_start + cell_batch_size, n_cells)
                current_batch_size = cell_end - cell_start  # 当前批次实际细胞数
                current_cells = sample_adata[cell_start:cell_end]  # 提取当前批次细胞
                
                # 构建当前批次的表达矩阵（仅包含当前批次细胞）
                batch_expr = torch.zeros((current_batch_size, c, h, w), device=device, dtype=torch.float32)
                gene_ids_batch = gene_ids_tensor.expand(current_batch_size, -1)  # [current_batch_size, c]
                
                if mode == 'single':
                    # 每个细胞的表达值置于中心位置（仅当前批次）
                    x_center, y_center = w // 2, h // 2
                    for i in range(current_batch_size):
                        expr_val = current_cells.X[i, :].flatten()  # 当前批次第i个细胞的表达值
                        batch_expr[i, :, y_center, x_center] = torch.tensor(expr_val, device=device)
                else:  # mode == 'padding'
                    # 每个细胞的表达值随机填充至k比例位置（仅当前批次）
                    total_positions = h * w
                    num_positions = max(1, int(total_positions * k))
                    for i in range(current_batch_size):
                        indices = torch.randperm(total_positions, device=device)[:num_positions]
                        y_coords = indices // w
                        x_coords = indices % w
                        expr_val = current_cells.X[i, :].flatten()
                        for j in range(num_positions):
                            y, x = y_coords[j].item(), x_coords[j].item()
                            batch_expr[i, :, y, x] = torch.tensor(expr_val, device=device)
                
                # 模型推理（仅当前批次细胞）
                with torch.no_grad():
                    _, _, _, _, enc_output = model(expression=batch_expr, gene_ids=gene_ids_batch)
                
                # 提取当前批次的细胞嵌入
                if mode == 'single':
                    x_center, y_center = w // 2, h // 2
                    current_emb = enc_output[:, :, y_center, x_center]  # [current_batch_size, en_dim]
                else:
                    # k模式：计算随机位置嵌入的平均值（仅当前批次）
                    emb_list = []
                    for i in range(current_batch_size):
                        non_zero_mask = batch_expr[i].sum(dim=0) != 0  # [h, w]
                        if non_zero_mask.any():
                            cell_emb = enc_output[i, :, non_zero_mask].mean(dim=1)  # [en_dim]
                        else:
                            cell_emb = torch.zeros(config['en_dim'], device=device)
                        emb_list.append(cell_emb)
                    current_emb = torch.stack(emb_list)  # [current_batch_size, en_dim]
                
                # 将当前批次嵌入存入总结果数组
                all_cell_emb[cell_start:cell_end] = current_emb
                
                # 清理当前批次未使用的中间变量，减少内存碎片
                del batch_expr, gene_ids_batch, enc_output, current_emb
                torch.cuda.empty_cache()
            
            # 所有批次处理完成后，将嵌入转移到CPU并存储
            sample_adata.obsm[config['emb_name']] = all_cell_emb.cpu().numpy()
            del all_cell_emb  # 释放GPU内存
            torch.cuda.empty_cache()

    return adata_list
def _is_log_transformed(X, threshold=10):
    """
    判断数据是否已经过对数化处理
    
    参数
    ----
    X : numpy.ndarray, scipy.sparse.spmatrix
        输入数据矩阵
    threshold : float
        判断阈值，最大值超过此值则认为未对数化
    
    返回
    ----
    bool
        是否已经过对数化处理
    """
    # 处理稀疏矩阵
    if scipy.sparse.issparse(X):
        X_data = X.data
    else:
        X_data = X
        
    # 检查数据中最大值
    max_val = np.max(X_data) if len(X_data) > 0 else 0
    
    # 如果最大值大于阈值，则认为数据未对数化
    if max_val > threshold:
        return False
    
    # 进一步检查数据特征：对数化后的数据通常包含小数
    if scipy.sparse.issparse(X):
        sample_values = X_data[:min(1000, len(X_data))]
    else:
        sample_values = X.flatten()[::max(1, X.size // 1000)]
    
    # 检查小数部分占比
    has_fraction = np.any(np.modf(sample_values)[0] != 0)
    
    # 如果最大值较小且包含小数，则认为已对数化
    return has_fraction or max_val < 5


def log1p_with_backup(adata, layer_raw='counts', layer_log='log1p'):
    """
    对 adata.X 做 log1p（如果尚未对数化），同时完整保留原始计数。
    
    参数
    ----
    adata : AnnData
        输入对象
    layer_raw : str
        用来存放原始计数的 layer 名，默认 'counts'
    layer_log : str
        用来存放 log1p 后值的 layer 名，默认 'log1p'
    """
    # 检查数据是否已经对数化
    if _is_log_transformed(adata.X):
        print("✅ 数据已经过对数化处理，跳过 log1p 转换")
        # 确保原始数据已备份
        if adata.raw is None:
            adata.raw = adata.copy()
        if layer_raw not in adata.layers:
            adata.layers[layer_raw] = adata.raw.X.copy()
        if layer_log not in adata.layers:
            adata.layers[layer_log] = adata.X.copy()
        return
    
    # 1) 拷贝原始计数到 adata.raw（scanpy 推荐做法）
    adata.raw = adata.copy()
    
    # 2) 再存一份到指定 layer，方便矩阵操作
    adata.layers[layer_raw] = adata.X.copy()
    
    # 3) 真正执行 log1p（只在 adata.X 上操作）
    sc.pp.log1p(adata)          # 原位修改 adata.X
    
    # 4) 把 log1p 后的结果也存一份 layer，方便调用
    adata.layers[layer_log] = adata.X.copy()
    
    print("✅ log1p 完成：")
    print(f"   原始计数 -> adata.raw, adata.layers['{layer_raw}']")
    print(f"   log1p 值 -> adata.X, adata.layers['{layer_log}']")

def get_hv_list_variance_weighted(adata_train, n_genes=2000, random_seed=42):
    """
    基于方差加权随机选择高变基因（hv_list）
    
    参数:
        adata_train (AnnData): 训练集的AnnData对象
        n_genes (int): 要选择的高变基因数量，默认为2000
        random_seed (int): 随机种子确保可复现性
    
    返回:
        hv_list (list): 高变基因名称列表
    """
    # 设置随机种子确保可复现性
    np.random.seed(random_seed)
    
    # 1. 获取表达矩阵并确保是稠密格式[2,5](@ref)
    if issparse(adata_train.X):
        X = adata_train.X.toarray()  # 稀疏矩阵转稠密
    else:
        X = adata_train.X  # 已经是稠密矩阵
    
    # 2. 计算每个基因的方差（沿轴=0计算基因维度）[7](@ref)
    gene_vars = np.var(X, axis=0)
    
    # 3. 处理方差为零的情况（避免除零错误）
    gene_vars = gene_vars + 1e-5  # 添加微小平滑项
    
    # 4. 将方差转换为选择权重（归一化概率）[7](@ref)
    weights = gene_vars / np.sum(gene_vars)
    
    # 5. 基于权重随机选择基因[7](@ref)
    n_genes = min(n_genes, len(adata_train.var_names))  # 确保不超出范围
    selected_indices = np.random.choice(
        len(adata_train.var_names), 
        size=n_genes, 
        replace=False,  # 无放回抽样
        p=weights       # 方差加权概率
    )
    
    # 6. 获取基因名称列表
    hv_list = adata_train.var_names[selected_indices].tolist()
    
    print(f"Selected {len(hv_list)} highly variable genes based on variance-weighted random sampling")
    print(f"Gene variance range: {np.min(gene_vars):.4f} - {np.max(gene_vars):.4f}")
    
    return hv_list
    
if __name__ == '__main__':
    seed = 42  

    # 1. 设置Python内置随机数生成器
    random.seed(seed)

    # 2. 设置NumPy的随机数生成器
    np.random.seed(seed)

    # 3. 设置PyTorch的随机数生成器
    torch.manual_seed(seed)
    vocab_path = "project1/spatial_data/spatial_data/new_vocab.json"
    if os.path.exists(vocab_path):
        print(f"从 {vocab_path} 加载基因索引...")
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
    else:
        vocab = None
        print(f"警告: 未找到词汇表文件 {vocab_path}")

    # 配置参数
    config = {
        'normalize':10000,
        "encoder_layers": 6,
        "decoder_layers": 2,
        'is_bin': False,   # 是否进行bin处理
        'bins': 50,
        'is_mask':False,    # 模型内部是否掩码
        'c': 512,           # 最大基因长度
        'depth':512,
        'h': 14,            # 高度
        'w': 14,            # 宽度
        'patch_size': 1,     # 块大小
        'emb_dim': 256,      # 嵌入维度
        'en_dim': 256,       # 编码器维度
        'de_dim': 256,       # 解码器维度
        'mlp1_depth': 2,     # MLP1深度
        'mlp2_depth': 4,     # MLP2深度
        'mask_ratio': 0.0,   # 掩码比例
        'mask_ratio_list': [0.0], # 掩码比例列表
        'lr': 2e-5,           # 学习率
        'weight_decay': 0.05, # 权重衰减
        'batch_size': 32,     # 批次大小
        'num_workers': 4,     # 数据加载工作进程数
        'epochs': 100,        # 训练轮数
        'data_dir': "project1/spatial_data/samples", # 数据目录
        'pad_id': vocab["<pad>"] if vocab else 0,  # 填充ID
        'num_genes': max(vocab.values()) + 1 if vocab else 1000, # 基因数量 (包括pad)
        'model_output_dir': 'project1/model_outputs', # 模型输出目录
        'model_type': 'transformer', # 模型类型
        'model_path': 'project1/model_outputs/maskdual_valid512(14)/checkpoint_epoch_40.pth',
        'emb_name': 'X_emb512_model40_14',
        'mode': 'spatial', # 模式选择 spatial or single or padding
        'pad_size': 1.0
    }
    gene_selection_method = 'hvg' #"hvg": 使用高变基因"weighted_random": 加权随机选择）"uniform_random": 直接均匀随机选择
    directory_path = 'project1/spatial_data/down_stream_data/raw_data_DLPFC/DLPFC'
    if torch.cuda.is_available():
        device = get_least_used_gpu()
        print(f"使用GPU: {torch.cuda.get_device_name(device.index)}")
    else:
        device = torch.device("cpu")
        print("使用CPU进行训练")

    # 初始化模型
    model = MAEModel(config).to(device)
    model.load_state_dict(torch.load(config['model_path'])['model_state_dict'])

    
    # 获取目录下所有.h5ad文件 [1,5](@ref)
    h5ad_files = [f for f in os.listdir(directory_path) if f.endswith('.h5ad')]
    train_files = [f for f in h5ad_files]
    adata_train_list = []
    '''
    for train_file in train_files:
        file_path = os.path.join(directory_path, train_file)
        print(f"Loading training file: {file_path}")
        adata = ad.read_h5ad(file_path)
        adata.obs['source_file'] = train_file  # 标记来源文件
        adata_train_list.append(adata)
    
    # 合并训练集[7](@ref)
    if len(adata_train_list) == 1:
        adata_train = adata_train_list[0]
    else:
        adata_train = adata_train_list[0].concatenate(
            adata_train_list[1:], 
            batch_key='source_file',
            index_unique=None  # 避免索引重复
        )
    h5ad_path = 'project1/spatial_data/raw_data_DLPFC/DLPFC/151507.h5ad'
    adata = anndata.read_h5ad(h5ad_path)
    hv_list = get_hv_list_variance_weighted(adata_train,n_genes=config['depth'],random_seed=42)
    '''
    # 循环处理目录下的每个h5ad文件 [1,2](@ref)
    for file_name in h5ad_files:
        # 构建完整的文件路径
        h5ad_path = os.path.join(directory_path, file_name)
        print(f"正在处理文件: {h5ad_path}")
        
        # 读取单个h5ad文件
        adata = anndata.read_h5ad(h5ad_path)
        adata.var_names_make_unique()
        # 处理流程
        log1p_with_backup(adata)
        preprocessed_adata_list = preprocess(adata, vocab, config,gene_selection_method = gene_selection_method)
        
        preprocessed_adata_list = add_emb(model, preprocessed_adata_list, vocab, config, device)
        #print(preprocessed_adata_list[0])
        
        adata = map_slices_to_original_extended(adata, preprocessed_adata_list)
        print(adata)
        print(f"nan数量：{np.isnan(adata.obsm[config['emb_name']]).sum()}")
        
        # 将处理后的数据写回原文件（或可指定新路径）
        adata.write(h5ad_path)
        print(f"文件 {file_name} 处理完成\n")

    print("所有文件处理完毕！")
    '''
    # 单个文件添加嵌入
    h5ad_path = 'project1/spatial_data/raw_data_DLPFC/DLPFC/151507.h5ad'
    adata = anndata.read_h5ad(h5ad_path)
    log1p_with_backup(adata)
    preprocessed_adata_list = preprocess(adata, vocab,config,hv_list=None)
    preprocessed_adata_list = add_emb(model,preprocessed_adata_list,vocab,config,device)
    print(preprocessed_adata_list[0])
    adata = map_slices_to_original_extended(adata,preprocessed_adata_list)
    print(adata)
    adata.write(h5ad_path)
    '''