import os
import scanpy as sc
import anndata as ad
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
from scipy.spatial import distance_matrix
from sklearn.preprocessing import normalize
from tqdm import tqdm
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

import os
import scanpy as sc
import anndata as ad
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
from scipy.spatial import distance_matrix
from sklearn.preprocessing import normalize
from tqdm import tqdm
import random
from sklearn.preprocessing import StandardScaler
import pandas as pd
import scipy
import warnings
from anndata._warnings import ImplicitModificationWarning
from scipy.sparse import issparse, csr_matrix
import re

def _is_log_transformed(X, threshold=10):
    """判断数据是否已经过对数化处理"""
    if scipy.sparse.issparse(X):
        X_data = X.data
    else:
        X_data = X
        
    max_val = np.max(X_data) if len(X_data) > 0 else 0
    
    if max_val > threshold:
        return False
    
    if scipy.sparse.issparse(X):
        sample_values = X_data[:min(1000, len(X_data))]
    else:
        sample_values = X.flatten()[::max(1, X.size // 1000)]
    
    has_fraction = np.any(np.modf(sample_values)[0] != 0)
    return has_fraction or max_val < 5

def log1p_with_backup(adata, layer_raw='counts', layer_log='log1p'):
    """对 adata.X 做 log1p（如果尚未对数化），同时完整保留原始计数"""
    if _is_log_transformed(adata.X):
        print("✅ 数据已经过对数化处理，跳过 log1p 转换")
        if adata.raw is None:
            adata.raw = adata.copy()
        if layer_raw not in adata.layers:
            adata.layers[layer_raw] = adata.raw.X.copy()
        if layer_log not in adata.layers:
            adata.layers[layer_log] = adata.X.copy()
        return
    
    adata.raw = adata.copy()
    adata.layers[layer_raw] = adata.X.copy()
    sc.pp.log1p(adata)
    adata.layers[layer_log] = adata.X.copy()
    #print("✅ log1p 完成")

def scale_spatial_coordinates(adata, target_range=(0, 1)):
    """将空间坐标缩放到目标范围"""
    if 'ori_spatial' in adata.obsm:
        #print("检测到 'ori_spatial'，空间坐标已缩放过，跳过本次操作")
        return adata

    spatial = adata.obsm['spatial'].copy()
    min_vals = spatial.min(axis=0)
    max_vals = spatial.max(axis=0)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1

    target_min, target_max = target_range
    scaled_spatial = (spatial - min_vals) / ranges * (target_max - target_min) + target_min

    adata.obsm['ori_spatial'] = spatial
    adata.obsm['spatial'] = scaled_spatial
    print(f"空间坐标已从范围 {min_vals} - {max_vals} 缩放到 {target_min} - {target_max}")
    return adata

class SpatialTranscriptomicsDataset(Dataset):
    def __init__(self, directory_path, config, gene_selection_method='hvg', 
                 target_size=(16, 16), mask_size=4, vocab=None, transform=None):
        """
        空间转录组数据集类
        
        Args:
            directory_path: h5ad文件目录路径
            config: 配置参数
            gene_selection_method: 基因选择方法
            target_size: 目标网格大小 (h, w)
            mask_size: 掩码区域大小
            vocab: 基因词汇表字典
            transform: 数据变换
        """
        self.directory_path = directory_path
        self.config = config
        self.gene_selection_method = gene_selection_method
        self.target_h, self.target_w = target_size
        self.mask_size = mask_size
        self.transform = transform
        self.pad_id = config['pad_id']
        self.vocab = vocab or {}
        
        # 获取所有h5ad文件
        self.h5ad_files = [f for f in os.listdir(directory_path) if f.endswith('.h5ad')]
        if not self.h5ad_files:
            raise ValueError(f"在目录 {directory_path} 中未找到h5ad文件")
        
        print(f"找到 {len(self.h5ad_files)} 个h5ad文件")
        if self.vocab:
            print(f"词汇表大小: {len(self.vocab)}")
        
    def __len__(self):
        return len(self.h5ad_files)
    
    def _select_top_genes(self, adata, method='hvg', num_genes=512):
        """选择top基因，先过滤不在词汇表中的基因，再选择高变基因"""
        if self.vocab is None or len(self.vocab) == 0:
            raise ValueError("词汇表为空或未提供，无法进行基因过滤")
        
        #print(f"原始基因数量: {adata.n_vars}")
        
        # 第一步：根据词汇表过滤基因
        all_gene_symbols = adata.var_names.tolist()
        genes_in_vocab = [symbol for symbol in all_gene_symbols if symbol in self.vocab]
        
        if len(genes_in_vocab) == 0:
            raise ValueError("没有基因在词汇表中，请检查基因命名规范")
        
        #print(f"在词汇表中的基因数量: {len(genes_in_vocab)}")
        adata_filtered_by_vocab = adata[:, genes_in_vocab].copy()
        
        # 第二步：选择高变基因
        actual_num_genes = min(num_genes, adata_filtered_by_vocab.n_vars)
        
        if method == 'hvg':
            if actual_num_genes <= num_genes:
                #print(f"注意: 只有{adata_filtered_by_vocab.n_vars}个基因在词汇表中，将选择{actual_num_genes}个高变基因")
            
                sc.pp.highly_variable_genes(adata_filtered_by_vocab, 
                                            n_top_genes=actual_num_genes, 
                                            flavor='seurat')
                adata_filtered = adata_filtered_by_vocab[:, adata_filtered_by_vocab.var.highly_variable].copy()
    

        else:
            # 选择表达量最高的基因
            gene_means = adata_filtered_by_vocab.X.mean(axis=0)
            if hasattr(gene_means, 'A1'):
                gene_means = gene_means.A1
            
            top_gene_indices = np.argsort(gene_means)[-actual_num_genes:]
            adata_filtered = adata_filtered_by_vocab[:, top_gene_indices].copy()
        
        #print(f"最终选择基因数量: {adata_filtered.n_vars}")
        
        # 第三步：映射gene_ids
        selected_genes = adata_filtered.var_names.tolist()
        gene_ids = [self.vocab[gene] for gene in selected_genes]
        
        # 将gene_ids存储在adata中
        adata_filtered.var['gene_id'] = gene_ids
        
        return adata_filtered

    def _create_grid_mapping(self, spatial_coords, grid_h, grid_w, bounds):
        """将空间坐标映射到网格"""
        min_x, max_x, min_y, max_y = bounds
        grid_indices = np.zeros((len(spatial_coords), 2), dtype=int)
        grid_occupancy = np.zeros((grid_h, grid_w), dtype=bool)  # 记录网格位置是否有spot
        
        for i, (x, y) in enumerate(spatial_coords):
            # 计算相对位置
            rel_x = (x - min_x) / (max_x - min_x)
            rel_y = (y - min_y) / (max_y - min_y)
            
            # 映射到网格
            grid_x = int(rel_x * grid_w)
            grid_y = int(rel_y * grid_h)
            
            # 确保在网格范围内
            grid_x = max(0, min(grid_x, grid_w - 1))
            grid_y = max(0, min(grid_y, grid_h - 1))
            
            grid_indices[i] = [grid_y, grid_x]
            grid_occupancy[grid_y, grid_x] = True
            
        return grid_indices, grid_occupancy
    
    def _create_random_mask(self, grid_h, grid_w, mask_size):
        """创建随机位置和大小的掩码区域"""
        # 随机选择掩码区域的左上角坐标
        mask_top = np.random.randint(0, grid_h - mask_size + 1) if grid_h > mask_size else 0
        mask_left = np.random.randint(0, grid_w - mask_size + 1) if grid_w > mask_size else 0
        
        # 确保掩码区域在网格范围内
        mask_bottom = min(mask_top + mask_size, grid_h)
        mask_right = min(mask_left + mask_size, grid_w)
        
        # 创建掩码
        mask = np.zeros((grid_h, grid_w), dtype=bool)
        mask[mask_top:mask_bottom, mask_left:mask_right] = True
        
        return mask, mask_top, mask_bottom, mask_left, mask_right
    def __getitem__(self, idx):
        """获取单个样本 - 修复gene_ids映射和有效spot计算，并随机选择物理区域构建网格"""
        h5ad_file = self.h5ad_files[idx]
        file_path = os.path.join(self.directory_path, h5ad_file)
        
        # 读取h5ad文件
        adata = sc.read_h5ad(file_path)
        
        # 数据预处理
        if self.config.get('normalize', 10000) > 0:
            if _is_log_transformed(adata.X):
                adata.X = np.expm1(adata.X) 
            sc.pp.normalize_total(adata, target_sum=self.config['normalize'])
            log1p_with_backup(adata)

        adata = scale_spatial_coordinates(adata, target_range=(0, 100))
        
        # 选择top基因
        c = self.config['c']
        if adata.n_vars < c:
            raise ValueError(f"文件 {h5ad_file} 中基因数量({adata.n_vars})少于要求的{c}个")
        
        adata_filtered = self._select_top_genes(adata, self.gene_selection_method, c)
        
        # 获取gene_ids（从vocab映射得到）
        gene_ids_list = adata_filtered.var['gene_id'].tolist()
        # 确保gene_ids长度与c一致，不足时用pad_id填充
        if len(gene_ids_list) < c:
            gene_ids_list.extend([self.pad_id] * (c - len(gene_ids_list)))
        elif len(gene_ids_list) > c:
            gene_ids_list = gene_ids_list[:c]
        
        gene_ids_tensor = torch.tensor(gene_ids_list, dtype=torch.long)
        
        # 获取空间坐标
        if 'spatial' not in adata.obsm:
            raise KeyError(f"文件 {h5ad_file} 的obsm中未找到'spatial'坐标")
        
        spatial_coords = adata.obsm['spatial']
        if spatial_coords.shape[1] != 2:
            raise ValueError(f"空间坐标应为2维，实际为{spatial_coords.shape[1]}维")
        
        # 获取空间坐标范围
        coords_min = spatial_coords.min(axis=0)
        coords_max = spatial_coords.max(axis=0)
        coords_range = coords_max - coords_min
        coords_range[coords_range == 0] = 1

        # 随机选择物理区域中心点（在整个空间坐标范围内）
        # 确保选择的区域不会超出坐标范围
        h_physical = self.config['h']  # 物理范围高度
        w_physical = self.config['w']  # 物理范围宽度
        
        # 计算中心点的可选范围，确保提取区域不会超出坐标边界
        center_x_min = coords_min[0] + w_physical / 2
        center_x_max = coords_max[0] - w_physical / 2
        center_y_min = coords_min[1] + h_physical / 2
        center_y_max = coords_max[1] - h_physical / 2
        
        # 如果坐标范围太小，无法容纳指定大小的区域，则使用整个范围
        if center_x_min > center_x_max:
            center_x = (coords_min[0] + coords_max[0]) / 2
            w_physical = min(w_physical, coords_range[0])  # 调整宽度不超过实际范围
        else:
            center_x = np.random.uniform(center_x_min, center_x_max)
        
        if center_y_min > center_y_max:
            center_y = (coords_min[1] + coords_max[1]) / 2
            h_physical = min(h_physical, coords_range[1])  # 调整高度不超过实际范围
        else:
            center_y = np.random.uniform(center_y_min, center_y_max)
        
        # 计算提取区域边界（以随机选择的中心点为中心）
        extract_min_x = center_x - w_physical / 2
        extract_max_x = center_x + w_physical / 2
        extract_min_y = center_y - h_physical / 2
        extract_max_y = center_y + h_physical / 2
        
        # 选择在提取区域内的spot
        in_extract_region = (
            (spatial_coords[:, 0] >= extract_min_x) & 
            (spatial_coords[:, 0] <= extract_max_x) & 
            (spatial_coords[:, 1] >= extract_min_y) & 
            (spatial_coords[:, 1] <= extract_max_y)
        )
        
        extract_coords = spatial_coords[in_extract_region]
        extract_data = adata_filtered[in_extract_region]
        
        if len(extract_coords) == 0:
            # 如果没有spot在区域内，创建空网格
            grid_data = np.zeros((c, self.target_h, self.target_w), dtype=np.float32)
            mask = np.zeros((self.target_h, self.target_w), dtype=bool)
            original_data = np.zeros((c, self.mask_size, self.mask_size), dtype=np.float32)
            valid_spot_mask = np.zeros((self.mask_size, self.mask_size), dtype=bool)
            return grid_data, mask, original_data, valid_spot_mask, gene_ids_tensor, h5ad_file
        
        # 创建网格映射 - 使用固定的网格大小 (target_h × target_w)
        bounds = [extract_min_x, extract_max_x, extract_min_y, extract_max_y]
        grid_indices, grid_occupancy = self._create_grid_mapping(
            extract_coords, self.target_h, self.target_w, bounds
        )
        
        # 构建网格数据
        grid_data = np.zeros((c, self.target_h, self.target_w), dtype=np.float32)
        if hasattr(extract_data.X, 'toarray'):
            expression_data = extract_data.X.toarray().T
        else:
            expression_data = extract_data.X.T
        
        # 将spot数据填充到网格
        spot_to_grid_map = {}
        for i, (grid_y, grid_x) in enumerate(grid_indices):
            if (grid_y, grid_x) not in spot_to_grid_map:
                spot_to_grid_map[(grid_y, grid_x)] = []
            spot_to_grid_map[(grid_y, grid_x)].append(i)
        
        # 对每个网格位置，如果有多个spot，取平均值
        for (grid_y, grid_x), spot_indices in spot_to_grid_map.items():
            if len(spot_indices) > 0:
                mean_expression = expression_data[:, spot_indices].mean(axis=1)
                min_dim = min(mean_expression.shape[0], c)
                grid_data[:min_dim, grid_y, grid_x] = mean_expression[:min_dim]
        
        # 创建随机掩码
        mask, mask_top, mask_bottom, mask_left, mask_right = self._create_random_mask(
            self.target_h, self.target_w, self.mask_size
        )
        
        # 保存原始掩码区域数据
        actual_mask_height = mask_bottom - mask_top
        actual_mask_width = mask_right - mask_left
        
        # 创建有效spot掩码 - 记录掩码区域内哪些位置原本有spot
        valid_spot_mask = np.zeros((self.mask_size, self.mask_size), dtype=bool)
        original_masked_data = np.zeros((c, self.mask_size, self.mask_size), dtype=np.float32)
        
        if actual_mask_height > 0 and actual_mask_width > 0:
            # 提取掩码区域的原始数据
            original_masked_data_slice = grid_data[:, mask_top:mask_bottom, mask_left:mask_right].copy()
            
            # 提取掩码区域的有效spot信息
            valid_spot_slice = grid_occupancy[mask_top:mask_bottom, mask_left:mask_right]
            
            # 如果实际掩码区域小于预期，进行填充
            if actual_mask_height < self.mask_size or actual_mask_width < self.mask_size:
                temp_data = np.zeros((c, self.mask_size, self.mask_size), dtype=np.float32)
                temp_valid = np.zeros((self.mask_size, self.mask_size), dtype=bool)
                
                height = min(actual_mask_height, self.mask_size)
                width = min(actual_mask_width, self.mask_size)
                
                temp_data[:, :height, :width] = original_masked_data_slice[:, :height, :width]
                temp_valid[:height, :width] = valid_spot_slice[:height, :width]
                
                original_masked_data = temp_data
                valid_spot_mask = temp_valid
            else:
                original_masked_data = original_masked_data_slice
                valid_spot_mask = valid_spot_slice
        
        # 注意：这里不预先应用掩码，保持网格数据完整
        return grid_data, mask, original_masked_data, valid_spot_mask, gene_ids_tensor, h5ad_file
    
def evaluate_mae_model(model, directory_path, config, gene_selection_method='hvg', 
                      num_experiments=5, batch_size=8, vocab=None):
    """
    评估MAE模型在空间转录组数据上的性能
    修改：使用vocab映射的gene_ids，并计算有效spot位置的MSE和MAE
    """
    device = next(model.parameters()).device
    model.eval()
    
    all_mse_results = []
    all_mae_results = []  # 新增：存储MAE结果
    all_null_mse_results = []  # 存储null MSE结果
    all_null_mae_results = []  # 新增：存储null MAE结果
    
    file_results = {}
    file_mae_results = {}  # 新增：存储文件级别的MAE结果
    file_null_results = {}  # 存储文件级别的null MSE
    file_null_mae_results = {}  # 新增：存储文件级别的null MAE
    
    mask_size = config.get('mask_size', 4)
    
    for exp_idx in range(num_experiments):
        print(f"\n进行第 {exp_idx + 1}/{num_experiments} 次实验...")
        
        # 创建数据集
        dataset = SpatialTranscriptomicsDataset(
            directory_path=directory_path,
            config=config,
            gene_selection_method=gene_selection_method,
            target_size=(config['h'], config['w']),
            mask_size=mask_size,
            vocab=vocab
        )
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=config.get('num_workers', 4)
        )
        
        experiment_mse = []
        experiment_mae = []  # 新增：实验级别的MAE
        experiment_null_mse = []  # 实验级别的null MSE
        experiment_null_mae = []  # 新增：实验级别的null MAE
        
        with torch.no_grad():
            for batch_idx, (grid_data, masks, original_data, valid_spot_masks, gene_ids, filenames) in enumerate(
                tqdm(dataloader, desc=f"实验 {exp_idx + 1}")
            ):
                # 移动到设备
                grid_data = grid_data.to(device).float()
                masks = masks.to(device).bool()
                original_data = original_data.to(device).float()
                valid_spot_masks = valid_spot_masks.to(device).bool()
                gene_ids = gene_ids.to(device).long()
                
                batch_size_current = grid_data.size(0)
                c, h, w = config['c'], config['h'], config['w']
                
                # 在输入模型前应用动态掩码
                masked_grid_data = grid_data.clone()
                for i in range(batch_size_current):
                    masked_grid_data[i][:, masks[i]] = 0
                
                # 使用从vocab映射的gene_ids输入模型
                reconstructed_image, cls_output, combined_mask, enc_output_spatial = model(
                    masked_grid_data, gene_ids
                )

                # 计算掩码区域的MSE和MAE - 只计算有效spot位置
                for i in range(batch_size_current):
                    # 提取掩码区域
                    mask_positions = masks[i]
                    if mask_positions.sum() == 0:
                        continue
                    
                    # 获取预测和真实值
                    pred_masked = reconstructed_image[i][:, mask_positions]  # [c, num_masked_pixels]
                    true_masked = original_data[i]  # [c, mask_h, mask_w]
                    
                    # 将true_masked重塑为二维
                    true_masked_flat = true_masked.reshape(true_masked.size(0), -1)  # [c, mask_h * mask_w]
                    
                    # 获取有效spot掩码并展平
                    valid_spot_mask_flat = valid_spot_masks[i].reshape(-1)  # [mask_h * mask_w]
                    
                    # 确保维度匹配
                    if pred_masked.size(1) != true_masked_flat.size(1):
                        min_dim = min(pred_masked.size(1), true_masked_flat.size(1))
                        pred_masked = pred_masked[:, :min_dim]
                        true_masked_flat = true_masked_flat[:, :min_dim]
                        valid_spot_mask_flat = valid_spot_mask_flat[:min_dim]
                    
                    # 只选择有效spot位置进行计算
                    if valid_spot_mask_flat.sum() > 0:
                        pred_valid = pred_masked[:, valid_spot_mask_flat]
                        true_valid = true_masked_flat[:, valid_spot_mask_flat]
                        
                        # 计算模型预测的MSE
                        mse = F.mse_loss(pred_valid, true_valid).item()
                        experiment_mse.append(mse)
                        
                        # 新增：计算模型预测的MAE
                        mae = F.l1_loss(pred_valid, true_valid).item()
                        experiment_mae.append(mae)
                        
                        # 计算原始值与0值的MSE（null MSE）
                        null_mse = F.mse_loss(torch.zeros_like(true_valid), true_valid).item()
                        experiment_null_mse.append(null_mse)
                        
                        # 新增：计算原始值与0值的MAE（null MAE）
                        null_mae = F.l1_loss(torch.zeros_like(true_valid), true_valid).item()
                        experiment_null_mae.append(null_mae)
                        
                        # 记录文件级别结果
                        filename = filenames[i]
                        if filename not in file_results:
                            file_results[filename] = []
                            file_mae_results[filename] = []  # 新增：初始化MAE结果存储
                            file_null_results[filename] = []
                            file_null_mae_results[filename] = []  # 新增：初始化null MAE结果存储
                        file_results[filename].append(mse)
                        file_mae_results[filename].append(mae)  # 新增：记录MAE
                        file_null_results[filename].append(null_mse)
                        file_null_mae_results[filename].append(null_mae)  # 新增：记录null MAE
        
        if experiment_mse:
            exp_avg_mse = np.mean(experiment_mse)
            exp_avg_mae = np.mean(experiment_mae)  # 新增：计算实验平均MAE
            exp_avg_null_mse = np.mean(experiment_null_mse)
            exp_avg_null_mae = np.mean(experiment_null_mae)  # 新增：计算实验平均null MAE
            
            all_mse_results.extend(experiment_mse)
            all_mae_results.extend(experiment_mae)  # 新增：收集所有MAE
            all_null_mse_results.extend(experiment_null_mse)
            all_null_mae_results.extend(experiment_null_mae)  # 新增：收集所有null MAE
            
            print(f"实验 {exp_idx + 1} 平均MSE: {exp_avg_mse:.6f} (基于{len(experiment_mse)}个有效spot)")
            print(f"实验 {exp_idx + 1} 平均MAE: {exp_avg_mae:.6f}")  # 新增：输出MAE
            print(f"实验 {exp_idx + 1} 原始值vs0值的平均MSE: {exp_avg_null_mse:.6f}")
            print(f"实验 {exp_idx + 1} 原始值vs0值的平均MAE: {exp_avg_null_mae:.6f}")  # 新增：输出null MAE
        else:
            print(f"实验 {exp_idx + 1} 无有效数据点")
    
    # 计算总体统计
    if all_mse_results:
        overall_avg_mse = np.mean(all_mse_results)
        overall_std_mse = np.std(all_mse_results)
        overall_avg_mae = np.mean(all_mae_results)  # 新增：总体MAE
        overall_std_mae = np.std(all_mae_results)   # 新增：MAE标准差
        overall_avg_null_mse = np.mean(all_null_mse_results)
        overall_std_null_mse = np.std(all_null_mse_results)
        overall_avg_null_mae = np.mean(all_null_mae_results)  # 新增：总体null MAE
        overall_std_null_mae = np.std(all_null_mae_results)   # 新增：null MAE标准差
        
        print(f"\n{'='*50}")
        print(f"评估结果汇总:")
        print(f"{'='*50}")
        print(f"总实验次数: {num_experiments}")
        print(f"模型预测平均MSE: {overall_avg_mse:.6f} ± {overall_std_mse:.6f}")
        print(f"模型预测平均MAE: {overall_avg_mae:.6f} ± {overall_std_mae:.6f}")  # 新增：输出MAE
        print(f"原始值vs0值平均MSE: {overall_avg_null_mse:.6f} ± {overall_std_null_mse:.6f}")
        print(f"原始值vs0值平均MAE: {overall_avg_null_mae:.6f} ± {overall_std_null_mae:.6f}")  # 新增：输出null MAE
        print(f"模型相对性能(MSE): {overall_avg_mse/overall_avg_null_mse*100:.2f}%")
        print(f"模型相对性能(MAE): {overall_avg_mae/overall_avg_null_mae*100:.2f}%")  # 新增：MAE相对性能
        print(f"最小MSE: {np.min(all_mse_results):.6f}")
        print(f"最小MAE: {np.min(all_mae_results):.6f}")  # 新增：最小MAE
        print(f"最大MSE: {np.max(all_mse_results):.6f}")
        print(f"最大MAE: {np.max(all_mae_results):.6f}")  # 新增：最大MAE
        
        # 输出每个文件的平均MSE和MAE
        print(f"\n各文件详细结果:")
        for filename, mse_list in file_results.items():
            file_avg_mse = np.mean(mse_list)
            file_avg_mae = np.mean(file_mae_results[filename])  # 新增：文件级别MAE
            file_null_avg_mse = np.mean(file_null_results[filename])
            file_null_avg_mae = np.mean(file_null_mae_results[filename])  # 新增：文件级别null MAE
            
            print(f"  {filename}:")
            print(f"    模型预测MSE: {file_avg_mse:.6f} (基于{len(mse_list)}个有效spot)")
            print(f"    模型预测MAE: {file_avg_mae:.6f}")  # 新增：输出文件MAE
            print(f"    原始值vs0值MSE: {file_null_avg_mse:.6f}")
            print(f"    原始值vs0值MAE: {file_null_avg_mae:.6f}")  # 新增：输出文件null MAE
            print(f"    相对性能(MSE): {file_avg_mse/file_null_avg_mse*100:.2f}%")
            print(f"    相对性能(MAE): {file_avg_mae/file_null_avg_mae*100:.2f}%")  # 新增：文件MAE相对性能
        
    else:
        overall_avg_mse = float('inf')
        overall_avg_mae = float('inf')  # 新增
        overall_avg_null_mse = float('inf')
        overall_avg_null_mae = float('inf')  # 新增
        print("警告: 所有实验均未产生有效MSE/MAE结果")
    
    results = {
        'overall_avg_mse': overall_avg_mse,
        'overall_avg_mae': overall_avg_mae,  # 新增
        'overall_avg_null_mse': overall_avg_null_mse,
        'overall_avg_null_mae': overall_avg_null_mae,  # 新增
        'all_mse_results': all_mse_results,
        'all_mae_results': all_mae_results,  # 新增
        'all_null_mse_results': all_null_mse_results,
        'all_null_mae_results': all_null_mae_results,  # 新增
        'file_results': file_results,
        'file_mae_results': file_mae_results,  # 新增
        'file_null_results': file_null_results,
        'file_null_mae_results': file_null_mae_results,  # 新增
        'num_experiments': num_experiments,
        'total_samples': len(all_mse_results)
    }
    
    return overall_avg_mse, overall_avg_mae, results  # 修改：返回MAE平均值

# 使用示例
if __name__ == "__main__":
    seed = 20  
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    vocab_path = "project1/spatial_data/spatial_data/new_vocab.json"
    if os.path.exists(vocab_path):
        print(f"从 {vocab_path} 加载基因索引...")
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
    else:
        vocab = None
        print(f"警告: 未找到词汇表文件 {vocab_path}")
    
    # 将model_size改为列表，支持多个尺寸测试
    model_sizes = [30,26,24,22,20]  # 可以修改为您需要测试的尺寸列表
    
    # 存储所有尺寸的结果
    all_results = {}
    
    for model_size in model_sizes:
        print(f"\n{'='*60}")
        print(f"开始测试模型尺寸: {model_size} × {model_size}")
        print(f"{'='*60}")
        
        # 配置参数 - 为每个尺寸动态生成
        config = {
            'normalize': 10000,
            "encoder_layers": 6,
            "decoder_layers": 2,
            'is_bin': False,
            'bins': 50,
            'is_mask': False,
            'c': 512,
            'depth': 512,
            'h': model_size,
            'w': model_size,
            'patch_size': 1,
            'emb_dim': 256,
            'en_dim': 256,
            'de_dim': 256,
            'mlp1_depth': 2,
            'mlp2_depth': 4,
            'mask_ratio': 0.0,
            'mask_size': min(14, model_size),  # 确保mask_size不超过模型尺寸
            'mask_ratio_list': [0.0],
            'lr': 2e-5,
            'weight_decay': 0.05,
            'batch_size': 4,
            'num_workers': 4,
            'epochs': 100,
            'data_dir': "project1/spatial_data/samples",
            'pad_id': vocab["<pad>"] if vocab else 0,
            'num_genes': max(vocab.values()) + 1 if vocab else 1000,
            'model_output_dir': 'project1/model_outputs',
            'model_type': 'transformer',
            'model_path': f'project1/model_outputs/maskdual_valid512({model_size})/checkpoint_epoch_40.pth',
            'emb_name': f'X_emb512_model40_{model_size}_hvg',
            'mode': 'spatial',
            'pad_size': 1.0
        }
        
        gene_selection_method = 'hvg'
        directory_path = 'project1/spatial_data/down_stream_data/Colorectal cancer histopathologyspatial transcriptomics data from Valdeolivas et al'
        
        if torch.cuda.is_available():
            device = get_least_used_gpu()
            print(f"使用GPU: {torch.cuda.get_device_name(device)}")
        else:
            device = torch.device("cpu")
            print("使用CPU进行训练")

        try:
            # 初始化模型
            model = MAEModel(config).to(device)
            # 检查模型文件是否存在
            if os.path.exists(config['model_path']):
                model.load_state_dict(torch.load(config['model_path'])['model_state_dict'])
                print(f"✅ 成功加载模型: {config['model_path']}")
            else:
                print(f"⚠️  模型文件不存在: {config['model_path']}，跳过该尺寸")
                continue
                
            eval_config = {
                'num_experiments': 10,
                'batch_size': 64,
                'gene_selection_method': 'hvg'
            }
            
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"模型参数数量: {total_params:,}")

            # 执行评估
            avg_mse, avg_mae, detailed_results = evaluate_mae_model(
                model=model,
                directory_path=directory_path,
                config=config,
                gene_selection_method=gene_selection_method,
                num_experiments=eval_config['num_experiments'],
                batch_size=eval_config['batch_size'],
                vocab=vocab
            )
            
            # 计算标准差
            overall_std_mse = np.std(detailed_results["all_mse_results"]) if len(detailed_results["all_mse_results"]) > 0 else 0
            overall_std_mae = np.std(detailed_results["all_mae_results"]) if len(detailed_results["all_mae_results"]) > 0 else 0
            overall_std_null_mse = np.std(detailed_results["all_null_mse_results"]) if len(detailed_results["all_null_mse_results"]) > 0 else 0
            overall_std_null_mae = np.std(detailed_results["all_null_mae_results"]) if len(detailed_results["all_null_mae_results"]) > 0 else 0

            # 存储当前尺寸的结果
            all_results[model_size] = {
                'avg_mse': avg_mse,
                'avg_mae': avg_mae,
                'std_mse': overall_std_mse,
                'std_mae': overall_std_mae,
                'null_mse': detailed_results["overall_avg_null_mse"],
                'null_mae': detailed_results["overall_avg_null_mae"],
                'std_null_mse': overall_std_null_mse,
                'std_null_mae': overall_std_null_mae,
                'detailed': detailed_results
            }
            
            print(f"\n模型尺寸 {model_size} × {model_size} 评估完成！")
            print(f"模型预测平均MSE:  {avg_mse:.6f} ± {overall_std_mse:.6f}")
            print(f"模型预测平均MAE:  {avg_mae:.6f} ± {overall_std_mae:.6f}")
            print(f'平均相对 0 MSE:   {detailed_results["overall_avg_null_mse"]:.6f} ± {overall_std_null_mse:.6f}')
            print(f'平均相对 0 MAE:   {detailed_results["overall_avg_null_mae"]:.6f} ± {overall_std_null_mae:.6f}')
            
        
            
        except Exception as e:
            print(f"尺寸 {model_size} 评估过程中出现错误: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # 输出所有尺寸的比较结果
    if all_results:
        print(f"\n{'='*80}")
        print(f"所有模型尺寸测试结果汇总")
        print(f"{'='*80}")
        
        # 打印表格头
        print(f"{'尺寸':<10} {'平均MSE':<15} {'MSE标准差':<15} {'平均MAE':<15} {'MAE标准差':<15} {'相对性能(MSE)':<15} {'相对性能(MAE)':<15}")
        print(f"{'-'*100}")
        
        # 按尺寸排序输出
        for size in sorted(all_results.keys()):
            result = all_results[size]
            mse_relative = (result['avg_mse'] / result['null_mse'] * 100) if result['null_mse'] > 0 else float('inf')
            mae_relative = (result['avg_mae'] / result['null_mae'] * 100) if result['null_mae'] > 0 else float('inf')
            
            print(f"{size}×{size}:{result['avg_mse']:>14.6f} ± {result['std_mse']:>13.6f} {result['avg_mae']:>14.6f} ± {result['std_mae']:>13.6f} {mse_relative:>14.2f}% {mae_relative:>14.2f}%")
        
        # 找出最佳性能的尺寸
        best_mse_size = min(all_results.keys(), key=lambda x: all_results[x]['avg_mse'])
        best_mae_size = min(all_results.keys(), key=lambda x: all_results[x]['avg_mae'])
        
        print(f"\n最佳MSE性能: 尺寸 {best_mse_size} × {best_mse_size} (MSE: {all_results[best_mse_size]['avg_mse']:.6f})")
        print(f"最佳MAE性能: 尺寸 {best_mae_size} × {best_mae_size} (MAE: {all_results[best_mae_size]['avg_mae']:.6f})")
        
    else:
        print("没有成功完成任何尺寸的测试")