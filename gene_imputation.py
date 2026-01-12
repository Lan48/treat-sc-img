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
from scipy.sparse import issparse,csr_matrix

import warnings
from anndata._warnings import ImplicitModificationWarning

warnings.filterwarnings("ignore", category=ImplicitModificationWarning)

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
def preprocess(ad, vocab, config):
    """
    预处理adata对象：基因过滤、log1p处理并分割成小切片
    返回adata列表，每个元素为一个小切片对象，加权随机选择高变基因
    
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
    ad = scale_spatial_coordinates(ad,target_range=(0, 100))
    # 1. 初始基因过滤（保留vocab中存在的基因）
    valid_genes = [gene for gene in ad.var_names if gene in vocab]
    ad = ad[:, valid_genes].copy()
    
    # 2. 基因表达值log1p变换
    #sc.pp.log1p(ad)
    
    # 3. 直接使用原始空间坐标（取消缩放）
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
            
            # 7. 基因选择与填充
            X = slice_ad.X.toarray() if scipy.sparse.issparse(slice_ad.X) else slice_ad.X
            
            # 计算基因方差权重
            gene_vars = np.var(X, axis=0)
            weights = (gene_vars + 1e-5) / np.sum(gene_vars + 1e-5)  # 标准化加平滑
            
            # 加权随机选择基因
            n_genes = min(config['depth'], len(slice_ad.var))
            selected_idx = np.random.choice(
                len(slice_ad.var), 
                size=n_genes, 
                replace=False, 
                p=weights
            )
            
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

def preprocess(ad, vocab, config):
    """
    预处理adata对象：基因过滤、log1p处理并分割成小切片
    返回adata列表，每个元素为一个小切片对象，选择平均表达量最高的基因
    
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
    
    # 2. 直接使用原始空间坐标
    spatial_coords = ad.obsm['spatial']
    
    # 3. 计算空间网格划分
    min_x, min_y = np.min(spatial_coords, axis=0)
    max_x, max_y = np.max(spatial_coords, axis=0)
    grid_x = np.arange(min_x, max_x, config['w'])
    grid_y = np.arange(min_y, max_y, config['h'])
    
    adata_list = []
    pad_id = vocab.get(config.get('pad_token', '<pad>'), 0)
    
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
            
            # 5. 添加坐标元数据
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
            
            # 6. 平均表达量最高的基因选择与填充（核心修改）
            X = slice_ad.X.toarray() if scipy.sparse.issparse(slice_ad.X) else slice_ad.X
            
            # 计算基因平均表达量并排序（替换原来的方差计算）
            gene_means = np.mean(X, axis=0)  # 按列计算平均值[1](@ref)
            sorted_idx = np.argsort(gene_means)[::-1]  # 平均表达量降序排序[1](@ref)
            
            # 选择top平均表达量最高的基因
            n_genes = min(config['depth'], len(slice_ad.var))
            selected_idx = sorted_idx[:n_genes]  # 取平均表达量最高的前n_genes个基因
            
            # 创建新表达矩阵
            new_X = np.zeros((len(slice_ad), config['depth']))
            new_X[:, :n_genes] = X[:, selected_idx]
            
            # 创建基因ID映射
            gene_ids = [vocab[slice_ad.var_names[i]] for i in selected_idx]
            if len(gene_ids) < config['depth']:
                gene_ids += [pad_id] * (config['depth'] - len(gene_ids))
            
            # 7. 构建新AnnData对象
            new_ad = sc.AnnData(
                X=new_X,
                obs=slice_ad.obs,
                var=pd.DataFrame({'gene_ids': gene_ids})
            )
            adata_list.append(new_ad)
    
    return adata_list
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
    ad = scale_spatial_coordinates(ad,target_range=(0, 100))
    valid_genes = [gene for gene in ad.var_names if gene in vocab]
    ad = ad[:, valid_genes].copy()
    
    # 2. 基因表达值log1p变换
    #sc.pp.log1p(ad)
    
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
def gene_imputation(
    model,
    adata: ad.AnnData,
    adata_list: List[ad.AnnData],
    vocab: Dict,
    config: Dict,
    device: Union[str, torch.device] = None,
    mask_ratio: float = 0.8,
    random_seed: int = 42,
) -> ad.AnnData:
    """
    Args:
        model: 深度学习模型
        adata: 主AnnData对象
        adata_list: 空间转录组样本的AnnData列表
        vocab: 基因ID到索引的映射字典
        config: 配置参数，包含h, w, en_dim等
        device: 指定计算设备（'cpu'/'cuda'或torch.device对象）
        mask_ratio: 缺失数据的比例
        random_seed: 随机种子，保证可复现；设为 None 则使用默认随机状态
    Returns:
        更新后的主AnnData对象
        基因插值的mse
    """
    if config.get('is_bin', False):
        for i in adata_list:
            i.X = bin_matrix(i.X, config['bins'])
    # ------------------ 1. 随机种子设置 ------------------
    if random_seed is not None:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
            # 确保卷积等 CUDA 操作也确定
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # ------------------ 2. 设备自动检测与设置 ------------------
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    model = model.to(device)
    model.eval()

    # ------------------ 3. 遍历每个空间样本 ------------------
    recon_loss = 0.0
    total_mask_points = 0

    for sample_adata in adata_list:
        h, w = config['h'], config['w']
        c = sample_adata.n_vars
        expr_matrix = np.zeros((c, h, w), dtype=np.float32)

        x_coords = sample_adata.obs['x_rel'].astype(int).values
        y_coords = sample_adata.obs['y_rel'].astype(int).values
        valid_idx = (x_coords >= 0) & (x_coords < w) & (y_coords >= 0) & (y_coords < h)
        x_valid = x_coords[valid_idx]
        y_valid = y_coords[valid_idx]

        original_expr_matrix = np.zeros((c, h, w), dtype=np.float32)
        masked_expr_matrix = np.zeros((c, h, w), dtype=np.float32)
        spot_mask = np.zeros((h, w), dtype=bool)

        num_valid = len(x_valid)
        num_mask = int(mask_ratio * num_valid)
        mask_indices = np.random.choice(num_valid, num_mask, replace=False)

        for i, (x, y) in enumerate(zip(x_valid, y_valid)):
            if 0 <= x < w and 0 <= y < h:
                expr = sample_adata.X[i, :].flatten()
                original_expr_matrix[:, y, x] = expr
                masked_expr_matrix[:, y, x] = expr
                if i in mask_indices:
                    spot_mask[y, x] = True
                    masked_expr_matrix[:, y, x] = 0

        gene_ids = sample_adata.var['gene_ids'].map(
            lambda x: vocab.get(x, 0)
        ).values

        expr_tensor = torch.tensor(masked_expr_matrix, device=device).unsqueeze(0)
        original_tensor = torch.tensor(original_expr_matrix, device=device).unsqueeze(0)
        gene_ids_tensor = torch.tensor(gene_ids, device=device).unsqueeze(0)

        with torch.no_grad():
            recon, _, _, _ = model(expression=expr_tensor, gene_ids=gene_ids_tensor)

            mask_tensor = torch.zeros_like(recon, dtype=torch.bool)
            for y in range(h):
                for x in range(w):
                    if spot_mask[y, x]:
                        mask_tensor[0, :, y, x] = True

            squared_error = (recon - original_tensor) ** 2
            masked_error = squared_error[mask_tensor]

            if masked_error.numel() > 0:
                sample_loss = masked_error.mean().item()
                recon_loss += sample_loss * masked_error.numel()
                total_mask_points += masked_error.numel()

    mse = recon_loss / total_mask_points if total_mask_points > 0 else 0.0
    return adata, mse
'''

def gene_imputation(
    model,
    adata: ad.AnnData,
    adata_list: List[ad.AnnData],
    vocab: Dict,
    config: Dict,
    device: Union[str, torch.device] = None,
    spot_mask_ratio: float = 0.8,  # 新增参数：spot掩码比例
    gene_mask_ratio: float = 0.8,   # 新增参数：基因掩码比例
    random_seed: int = 42,
) -> ad.AnnData:
    """
    Args:
        model: 深度学习模型
        adata: 主AnnData对象
        adata_list: 空间转录组样本的AnnData列表
        vocab: 基因ID到索引的映射字典
        config: 配置参数，包含h, w, en_dim等
        device: 指定计算设备（'cpu'/'cuda'或torch.device对象）
        spot_mask_ratio: 需要掩码的spot比例（新增）
        gene_mask_ratio: 在掩码spot中需要掩码的基因比例（新增）
        random_seed: 随机种子，保证可复现；设为 None 则使用默认随机状态
    Returns:
        更新后的主AnnData对象
        基因插值的mse（只基于被掩码基因）
    """
    # ------------------ 1. 随机种子设置 ------------------
    if random_seed is not None:
        np.random.seed(random_seed)
        rng = np.random.default_rng(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # ------------------ 2. 设备自动检测与设置 ------------------
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    model = model.to(device)
    model.eval()

    # ------------------ 3. 检查是否需要bin处理 ------------------
    is_bin = config.get('is_bin', False)
    n_bins = config.get('bins', 10) if is_bin else None

    # ------------------ 4. 生成全局固定的基因掩码 ------------------
    # 获取第一个样本的基因数量（假设所有样本基因数量相同）
    c = adata_list[0].n_vars
    
    # 生成全局固定的基因掩码向量（长度为c），所有样本共享相同的基因掩码模式
    global_gene_mask = np.zeros(c, dtype=bool)
    num_mask_genes = max(1, int(round(gene_mask_ratio * c)))
    
    # 随机选择要掩码的基因索引（全局固定）
    mask_gene_indices = rng.choice(c, num_mask_genes, replace=False)
    global_gene_mask[mask_gene_indices] = True

    # ------------------ 5. 遍历每个空间样本 ------------------
    recon_loss = 0.0
    total_mask_points = 0

    for sample_adata in adata_list:
        h, w = config['h'], config['w']
        c = sample_adata.n_vars
        expr_matrix = np.zeros((c, h, w), dtype=np.float32)

        x_coords = sample_adata.obs['x_rel'].astype(int).values
        y_coords = sample_adata.obs['y_rel'].astype(int).values
        valid_idx = (x_coords >= 0) & (x_coords <= w) & (y_coords >= 0) & (y_coords <= h)
        x_valid = x_coords[valid_idx]
        y_valid = y_coords[valid_idx]

        original_expr_matrix = np.zeros((c, h, w), dtype=np.float32)
        masked_expr_matrix = np.zeros((c, h, w), dtype=np.float32)
        
        num_valid = len(x_valid)
        num_mask_spots = int(spot_mask_ratio * num_valid)  # 计算需要掩码的spot数量
        mask_spot_indices = np.random.choice(num_valid, num_mask_spots, replace=False)  # 随机选择掩码spot的索引

        # 创建基因级别的掩码矩阵（形状: (c, h, w)），初始为False
        genemask_matrix = np.zeros((c, h, w), dtype=bool)
        
        # 存储每个细胞的原始表达值，用于后续处理
        cell_expression_data = []
        
        # 填充原始表达矩阵和初始化masked矩阵，同时生成基因掩码
        for i, (x, y) in enumerate(zip(x_valid, y_valid)):
            if 0 <= x < w and 0 <= y < h:
                expr = sample_adata.X[i, :].flatten()
                original_expr_matrix[:, y, x] = expr
                masked_expr_matrix[:, y, x] = expr
                cell_expression_data.append((i, x, y, expr))
                
                # 如果当前spot被选为掩码spot，则使用全局固定的基因掩码
                if i in mask_spot_indices:
                    # 使用全局固定的基因掩码向量
                    genemask_matrix[:, y, x] = global_gene_mask

        # 如果需要bin处理，对每个细胞独立进行分位数bin处理
        if is_bin and n_bins > 1:
            for i, x, y, expr in cell_expression_data:
                if len(expr) > 0:
                    quantiles = np.quantile(expr, np.linspace(0, 1, n_bins + 1))
                    binned_expr = np.digitize(expr, quantiles[1:-1], right=True)
                    binned_expr_normalized = binned_expr / float(n_bins)
                    masked_expr_matrix[:, y, x] = binned_expr_normalized

        # 应用基因级别掩码：只将被掩码的基因位置设置为0
        masked_expr_matrix[genemask_matrix] = 0  # 向量化操作，高效应用掩码

        gene_ids = sample_adata.var['gene_ids']

        expr_tensor = torch.tensor(masked_expr_matrix, device=device).unsqueeze(0)
        original_tensor = torch.tensor(original_expr_matrix, device=device).unsqueeze(0)
        gene_ids_tensor = torch.tensor(gene_ids, device=device).unsqueeze(0)

        with torch.no_grad():
            recon, _, _, _ = model(expression=expr_tensor, gene_ids=gene_ids_tensor)
            # 创建基因级别的掩码张量（形状: (1, c, h, w)），用于提取被掩码基因的误差
            genemask_tensor = torch.tensor(genemask_matrix, device=device).unsqueeze(0)
            squared_error = (recon - original_tensor) ** 2
            masked_error = squared_error[genemask_tensor]  # 只提取被掩码基因的误差

            if masked_error.numel() > 0:
                sum_error = masked_error.sum().item()  # 总误差
                num_masked = masked_error.numel()      # 被掩码基因数量
                recon_loss += sum_error
                total_mask_points += num_masked

    mse = recon_loss / total_mask_points if total_mask_points > 0 else 0.0
    return adata, mse
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
        ad.X = np.expm1(adata.X) 
        sc.pp.normalize_total(adata, target_sum=config['normalize'])
    
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


if __name__ == '__main__':
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
        'h': 16,            # 高度
        'w': 16,            # 宽度
        'patch_size': 2,     # 块大小
        'emb_dim': 256,      # 嵌入维度
        'en_dim': 256,       # 编码器维度
        'de_dim': 256,       # 解码器维度
        'mlp1_depth': 2,     # MLP1深度
        'mlp2_depth': 4,     # MLP2深度
        'mask_ratio': 0.0,   # 掩码比例
        'lr': 2e-5,           # 学习率
        'weight_decay': 0.05, # 权重衰减
        'batch_size': 32,     # 批次大小
        'num_workers': 4,     # 数据加载工作进程数
        'epochs': 100,        # 训练轮数
        'mask_ratio_list': [0.0], # 掩码比例列表
        'data_dir': "project1/spatial_data/samples", # 数据目录
        'pad_id': vocab["<pad>"] if vocab else 0,  # 填充ID
        'num_genes': max(vocab.values()) + 1 if vocab else 1000, # 基因数量 (包括pad)
        'model_output_dir': 'project1/model_outputs', # 模型输出目录
        'model_type': 'transformer', # 模型类型
        'model_path': 'project1/model_outputs/maskdual_valid512(16)/checkpoint_epoch_40.pth',
    }
    h5ad_path = 'project1/spatial_data/down_stream_data/raw_data_DLPFC/DLPFC/151507.h5ad'
    adata = anndata.read_h5ad(h5ad_path)
    log1p_with_backup(adata)
    preprocessed_adata_list = preprocess(adata, vocab,config)
    if torch.cuda.is_available():
        device = get_least_used_gpu()
        print(f"使用GPU: {torch.cuda.get_device_name(device.index)}")
    else:
        device = torch.device("cpu")
        print("使用CPU进行训练")

    # 初始化模型
    model = MAEModel(config).to(device)
    model.load_state_dict(torch.load(config['model_path'])['model_state_dict'])

    adata,mse = gene_imputation(model,adata,preprocessed_adata_list,vocab,config,device,spot_mask_ratio=1.0,gene_mask_ratio=0.2)
    print(mse)
    #adata.write(h5ad_path)
