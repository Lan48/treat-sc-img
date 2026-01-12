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
import glob
import random
from tqdm import tqdm 
from pathlib import Path
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

def preprocess(ad, vocab, config):
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
    
    返回：
    adata_list - 分割后的小切片AnnData对象列表
    """
    ad = scale_spatial_coordinates(ad,target_range=(0, 100))
    normalize = config.get('normalize', 0)
    if normalize > 0:
        if _is_log_transformed(ad.X):
            ad.X = np.expm1(ad.X) 
        sc.pp.normalize_total(ad, target_sum=normalize)
        log1p_with_backup(ad)
    # 1. 初始基因过滤（保留vocab中存在的基因）
    valid_genes = [gene for gene in ad.var_names if gene in vocab]
    ad = ad[:, valid_genes].copy()
       
    # 2. 直接使用原始空间坐标
    spatial_coords = ad.obsm['spatial']
    
    # 3. 计算空间范围
    min_x, min_y = np.min(spatial_coords, axis=0)
    max_x, max_y = np.max(spatial_coords, axis=0)
    
    # 4. 计算样本面积和采样面积，确定采样次数
    sample_area = (max_x - min_x) * (max_y - min_y)  # 样本总面积
    patch_area = config['w'] * config['h']  # 每个采样区域的面积
    
    # 计算采样次数：样本面积/采样面积 * 2
    n_samples = int((sample_area / patch_area) * 2)
    
    adata_list = []
    pad_id = vocab.get(config.get('pad_token', '<pad>'), 0)
    
    # 5. 随机采样循环
    for sample_idx in range(n_samples):
        # 随机生成起始点，确保采样区域在样本范围内
        x_start = np.random.uniform(min_x, max_x - config['w'])
        y_start = np.random.uniform(min_y, max_y - config['h'])
        
        # 计算当前采样区域边界
        x_end = x_start + config['w']
        y_end = y_start + config['h']
        
        # 选择当前采样区域内的细胞
        in_slice = (
            (spatial_coords[:, 0] >= x_start) & 
            (spatial_coords[:, 0] < x_end) &
            (spatial_coords[:, 1] >= y_start) & 
            (spatial_coords[:, 1] < y_end)
        )
        
        if not np.any(in_slice):
            continue  # 跳过无细胞的采样区域
            
        slice_ad = ad[in_slice, :].copy()
        
        # 6. 添加坐标元数据
        abs_coords = spatial_coords[in_slice]
        
        # 计算网格化相对坐标（映射到h*w网格）
        rel_x = abs_coords[:, 0] - x_start
        rel_y = abs_coords[:, 1] - y_start
        
        # 映射到离散网格（整数索引）
        grid_x_idx = np.floor(rel_x).astype(int)
        grid_y_idx = np.floor(rel_y).astype(int)
        
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
    new_obs_attributes = ['x_rel', 'y_rel']  # 可根据实际扩展
    
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

def train_model(
    model, 
    adata_list: List[ad.AnnData], 
    vocab: Dict, 
    config: Dict,
    device: Union[str, torch.device] = None
):
    """
    训练空间转录组模型
    
    Args:
        model: 深度学习模型
        adata_list: 空间转录组样本的AnnData列表
        vocab: 基因ID到索引的映射字典
        config: 配置参数，包含训练超参数
        device: 指定计算设备
    
    Returns:
        model: 训练好的模型
        losses: 训练损失记录
    """
    # 1. 设备设置和模型初始化
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    
    model = model.to(device)
    model.train()  # 设置为训练模式
    
    # 2. 优化器和超参数设置
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.get('lr', 0.001),
        weight_decay=config.get('weight_decay', 1e-5)
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    ) if config.get('use_scheduler', True) else None
    
    num_epochs = config.get('epochs', 100)
    batch_size = config.get('batch_size', 4)
    
    losses = []  # 记录损失
    
    # 3. 训练循环 - 添加进度条[1,2,5](@ref)
    with tqdm(total=num_epochs, desc="Training Progress", unit="epoch") as pbar:
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # 随机打乱数据顺序
            random.shuffle(adata_list)
            
            # 批量处理 - 添加批次进度条[2,8](@ref)
            batch_range = range(0, len(adata_list), batch_size)
            for i in tqdm(batch_range, desc=f"Epoch {epoch+1} Batches", leave=False):
                batch_adatas = adata_list[i:i+batch_size]
                batch_loss = 0.0
                
                # 清零梯度
                optimizer.zero_grad()
                
                # 处理批次中的每个样本
                for sample_adata in batch_adatas:
                    # 3.1 数据准备（与原始代码相同）
                    h, w = config['h'], config['w']
                    c = sample_adata.n_vars
                    expr_matrix = np.zeros((c, h, w), dtype=np.float32)
                    
                    # 坐标处理
                    x_coords = sample_adata.obs['x_rel'].astype(int).values
                    y_coords = sample_adata.obs['y_rel'].astype(int).values
                    valid_idx = (x_coords >= 0) & (x_coords < w) & (y_coords >= 0) & (y_coords < h)
                    x_valid = x_coords[valid_idx]
                    y_valid = y_coords[valid_idx]
                    
                    # 表达矩阵填充
                    for j, (x, y) in enumerate(zip(x_coords, y_coords)):
                        if 0 <= x < w and 0 <= y < h:
                            expr = sample_adata.X[j, :].flatten()  
                            expr_matrix[:, y, x] = expr
                    
                    # 基因ID索引化
                    gene_ids = sample_adata.var['gene_ids'].values
                    # 转换为张量
                    binned_expression = torch.tensor(expr_matrix, device=device).unsqueeze(0)  # [1, c, h, w]
                    gene_ids_tensor = torch.tensor(gene_ids, device=device).unsqueeze(0)  # [1, c]
                    expression = binned_expression.clone()  # 真实表达值用于损失计算
                    
                    # 3.2 前向传播
                    recon, cls_pred, mask, _ = model(binned_expression, gene_ids_tensor)
                    
                    # 3.3 损失计算
                    recon_loss = model.loss_function(expression, recon, mask)
                    batch_loss += recon_loss
                
                # 3.4 反向传播和优化
                batch_loss /= len(batch_adatas)  # 平均损失
                batch_loss.backward()
                
                # 梯度裁剪（防止梯度爆炸）
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    max_norm=config.get('max_grad_norm', 1.0)
                )
                
                optimizer.step()
                
                epoch_loss += batch_loss.item()
                num_batches += 1
            
            # 计算epoch平均损失
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            losses.append(avg_epoch_loss)
            
            # 学习率调整
            if scheduler is not None:
                scheduler.step(avg_epoch_loss)
            
            # 3.5 更新进度条信息[4,5](@ref)
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'Loss': f'{avg_epoch_loss:.4f}',
                'LR': f'{current_lr:.6f}',
                'Batches': num_batches
            })
            pbar.update(1)
            
            # 3.6 训练进度输出（可选，与进度条互补）
            if (epoch + 1) % config.get('print_every', 10) == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}, LR: {current_lr:.6f}')
            
            # 早期停止检查
            if config.get('early_stopping', False) and epoch > 10:
                if len(losses) > 1 and abs(losses[-1] - losses[-2]) < config.get('stop_threshold', 1e-6):
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    
    # 4. 训练完成
    print("Training completed!")
    return model, losses
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

def get_h5ad_files(path: Union[str, Path]) -> List[Path]:
    """
    自动识别路径类型并返回所有需要处理的h5ad文件列表
    
    参数:
    path: 文件路径、目录路径或包含通配符的路径模式
    
    返回:
    h5ad文件路径列表
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"路径不存在: {path}")
    
    # 如果是文件且是.h5ad文件，直接返回
    if path.is_file() and path.suffix == '.h5ad':
        return [path]
    
    # 如果是目录，查找所有.h5ad文件
    elif path.is_dir():
        h5ad_files = list(path.glob("*.h5ad"))
        if not h5ad_files:
            raise ValueError(f"目录 {path} 中未找到.h5ad文件")
        return h5ad_files
    
    # 如果包含通配符，使用glob模式匹配
    elif '*' in str(path) or '?' in str(path):
        h5ad_files = [Path(f) for f in glob.glob(str(path)) if Path(f).suffix == '.h5ad']
        if not h5ad_files:
            raise ValueError(f"通配符模式 {path} 未匹配到任何.h5ad文件")
        return h5ad_files
    
    else:
        raise ValueError(f"无法识别的路径类型: {path}")

def process_h5ad_data(h5ad_path: Union[str, Path], vocab, config, parallel: bool = False) -> List:
    """
    统一处理h5ad路径的主函数，自动判断路径类型
    
    参数:
    h5ad_path: 文件路径、目录路径或通配符模式
    vocab: 词汇表参数
    config: 配置参数
    parallel: 是否启用并行处理（多个文件时有效）
    """
    # 获取所有需要处理的h5ad文件
    h5ad_files = get_h5ad_files(h5ad_path)
    
    print(f"找到 {len(h5ad_files)} 个h5ad文件进行处理:")
    for i, file_path in enumerate(h5ad_files, 1):
        print(f"  {i}. {file_path}")
    
    # 处理单个文件的辅助函数
    def process_single_file(file_path: Path):
        try:
            print(f"正在处理: {file_path.name}")
            adata = ad.read_h5ad(file_path)
            #log1p_with_backup(adata)
            return preprocess(adata, vocab, config)
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            return None
    
    # 根据文件数量选择处理方式
    if len(h5ad_files) == 1:
        # 单个文件，直接处理
        result = process_single_file(h5ad_files[0])
        return [result] if result is not None else []
    
    else:
        # 多个文件，可选择并行或顺序处理
        preprocessed_adata_list = []
        
        if parallel:
            # 并行处理（需要导入multiprocessing）
            from multiprocessing import Pool
            with Pool(processes=min(len(h5ad_files), os.cpu_count())) as pool:
                results = pool.map(process_single_file, h5ad_files)
        else:
            # 顺序处理
            results = [process_single_file(file_path) for file_path in h5ad_files]
        
        # 过滤掉处理失败的结果
        for result in results:
            if result is not None:
                if isinstance(result, list):
                    preprocessed_adata_list.extend(result)
                else:
                    preprocessed_adata_list.append(result)
        
        print(f"处理完成，成功处理 {len(preprocessed_adata_list)} 个数据对象")
        return preprocessed_adata_list

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
        'h': 14,            # 高度
        'w': 14,            # 宽度
        'patch_size': 1,     # 块大小
        'emb_dim': 256,      # 嵌入维度
        'en_dim': 256,       # 编码器维度
        'de_dim': 256,       # 解码器维度
        'mlp1_depth': 2,     # MLP1深度
        'mlp2_depth': 4,     # MLP2深度
        'mask_ratio': 0.0,   # 掩码比例
        'lr': 2e-5,           # 学习率
        'weight_decay': 0.05, # 权重衰减
        'batch_size': 2,     # 批次大小
        'num_workers': 4,     # 数据加载工作进程数
        'epochs': 30,        # 训练轮数
        'data_dir': "project1/spatial_data/samples", # 数据目录
        'pad_id': vocab["<pad>"] if vocab else 0,  # 填充ID
        'num_genes': max(vocab.values()) + 1 if vocab else 1000, # 基因数量 (包括pad)
        'model_output_dir': 'project1/model_outputs/maskdual_valid512(3)', # 模型输出目录
        'model_type': 'transformer', # 模型类型
        'model_path': 'project1/model_outputs/maskdual_valid512(3)/checkpoint_epoch_40.pth',
    }
    h5ad_path = 'project1/spatial_data/raw_data_DLPFC/DLPFC/'

    preprocessed_adata_list = process_h5ad_data(h5ad_path, vocab,config)
    if torch.cuda.is_available():
        device = get_least_used_gpu()
        print(f"使用GPU: {torch.cuda.get_device_name(device.index)}")
    else:
        device = torch.device("cpu")
        print("使用CPU进行训练")

    # 初始化模型
    model = MAEModel(config).to(device)
    model.load_state_dict(torch.load(config['model_path'])['model_state_dict'])

    
    # 冻结编码器参数
    for param in model.encoder.parameters():
        param.requires_grad = False
    

    model,_ = train_model(model, preprocessed_adata_list, vocab,config, device)
    state = {
        'model_state_dict': model.state_dict(),
    }
    # 创建输出目录
    os.makedirs(config['model_output_dir'], exist_ok=True)
    
    # 保存最新检查点
    checkpoint_path = os.path.join(config['model_output_dir'], "model40+30FreezeEncoder.pth")
    torch.save(state, checkpoint_path)
